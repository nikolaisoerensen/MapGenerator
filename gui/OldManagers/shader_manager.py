"""
Path: gui/OldManagers/shader_manager.py

Funktionsweise: GPU-Compute Management für Performance-kritische Operationen
- OpenGL Compute Shader (GL 4.3+) für Parallel-Processing, GLSL-Quellcode liegt
  unter shaders/<category>/<operation>.comp
- Fallback auf CPU für Systeme ohne GPU-Support
- Memory-Management zwischen GPU und CPU

Architektur (Shader-Inventur 2026-07-08): GPU-Compute lief bisher bei KEINEM
Generator, weil _check_gpu_support() einen aktiven OpenGL-Kontext braucht, aber
aus Background-CalculatorThreads (LOD-Lockstep-Umbau) ohne jeden Kontext
aufgerufen wurde - gpu_available blieb dadurch für immer False, jeder Aufruf
fiel sofort auf CPU zurück. Fix: GPUWorker ist ein dedizierter Hintergrund-
Thread mit eigenem Offscreen-Kontext (QOffscreenSurface + QOpenGLContext,
verifiziert funktionsfähig auch abseits des GUI-Threads), unabhängig davon, ob
die 3D-Ansicht (mit ihrem eigenen GUI-Thread-Kontext in MapDisplay3D) je
geöffnet wurde. Alle Calculator-Threads schicken ihre Requests blockierend an
diesen einen Worker statt selbst GL-Aufrufe zu machen.

Einsatzgebiete:
- Terrain: Multi-Octave Noise (noiseGeneration), Shadow-Raycast (shadowRaycast)
- Water: Lake-Detection via Jump-Flooding (jumpFloodLakes) - erster Water-Knoten
  auf GPU, weitere Knoten folgen nach demselben Muster (siehe DISPATCH_TABLE)
- Settlement (Rework 2026-07-09): terrainCostFlood - JFA-Approximation des
  terrain-cost-gewichteten Multi-Source-Floods (_terrain_cost_voronoi() in
  core/settlement_generator.py), genutzt fuer sowohl Stadtgrenzen
  (CityBoundaryAnalyzer) als auch Landschafts-Voronoi-Zellen
  (LandscapeVoronoiSystem). civ_map-Decay und die Seed-Relaxation bleiben
  bewusst CPU (Profiling zeigt: beide zusammen <1s bei map_size 256, der
  eigentliche Bottleneck der Settlement-Pipeline liegt in der unvektorisierten
  TerrainSuitabilityAnalyzer.calculate_water_proximity(), nicht im neuen Code -
  siehe docs/backlog.md Ticket #4).
"""
import logging
import os
import queue
import threading
from types import SimpleNamespace

import numpy as np
import OpenGL.GL as gl
from PyQt6.QtCore import QObject, pyqtSignal
from opensimplex import OpenSimplex
from PyQt6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat

SHADERS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shaders")
)


def _compile_compute_program(source: str, name: str):
    """Kompiliert+linkt einen GLSL-Compute-Shader-String zu einem Programm (GLuint)."""
    shader = gl.glCreateShader(gl.GL_COMPUTE_SHADER)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)

    if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
        error = gl.glGetShaderInfoLog(shader).decode()
        gl.glDeleteShader(shader)
        raise RuntimeError(f"Shader-Kompilierung fehlgeschlagen für '{name}': {error}")

    program = gl.glCreateProgram()
    gl.glAttachShader(program, shader)
    gl.glLinkProgram(program)
    gl.glDeleteShader(shader)

    if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
        error = gl.glGetProgramInfoLog(program).decode()
        gl.glDeleteProgram(program)
        raise RuntimeError(f"Shader-Linking fehlgeschlagen für '{name}': {error}")

    return program


# Uniform-Namen, die in den .comp-Dateien als "uniform int" deklariert sind (siehe
# grep über shaders/**/*.comp). Alles andere wird als float behandelt.
_INT_UNIFORM_NAMES = {
    "u_biome_seed", "u_depth_tests", "u_height", "u_heightmap_size", "u_horizontal", "u_jump_distance",
    "u_octaves", "u_radius", "u_seed", "u_shadowmap_size", "u_size", "u_target_size", "u_width",
}


def _set_uniforms(program, values: dict):
    """
    Setzt int/float-Uniforms. Der GL-Typ wird NICHT aus dem Python-Laufzeittyp des
    Werts geraten (int vs. float) - Aufrufer übergeben ganzzahlige Parameter wie
    air_temp_entry=15 oder sun_elevation=70 oft als Python int, obwohl der GLSL-
    Uniform als float deklariert ist. glUniform1i auf eine float-Uniform-Location
    wirft GL_INVALID_OPERATION (genau das Symptom, das terrain.shadowRaycast und
    weather.temperatureCalculation deterministisch bei JEDEM Aufruf zeigten - siehe
    Live-Log). Stattdessen wird anhand des tatsächlichen GLSL-Deklarationstyps
    entschieden (_INT_UNIFORM_NAMES, aus den .comp-Quellen extrahiert).
    """
    for uniform_name, value in values.items():
        location = gl.glGetUniformLocation(program, uniform_name)
        if location == -1:
            continue
        if uniform_name in _INT_UNIFORM_NAMES:
            gl.glUniform1i(location, int(value))
        else:
            gl.glUniform1f(location, float(value))


def _create_texture_2d(width, height, internal_format):
    """Allokiert eine leere GPU-Textur (für Compute-Shader-Output)."""
    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_format, width, height, 0, gl.GL_RED, gl.GL_FLOAT, None)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    return texture_id


def _upload_texture_2d(data: np.ndarray, internal_format):
    """Lädt ein (height, width) float32-numpy-Array als GPU-Textur hoch."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    height, width = data.shape[:2]
    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_format, width, height, 0, gl.GL_RED, gl.GL_FLOAT, data)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    return texture_id


def _upload_texture_2d_rg(data: np.ndarray, internal_format):
    """Lädt ein (height, width, 2) float32-numpy-Array als 2-Kanal-GPU-Textur hoch."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    height, width = data.shape[:2]
    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_format, width, height, 0, gl.GL_RG, gl.GL_FLOAT, data)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    return texture_id


def _read_texture_data(texture_id, width, height) -> np.ndarray:
    """Liest den R-Kanal einer GPU-Textur (beliebiges Format) als float32-Array zurück."""
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    data = gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RED, gl.GL_FLOAT)
    return np.frombuffer(data, dtype=np.float32).reshape((height, width)).copy()


def _read_texture_data_rg(texture_id, width, height) -> np.ndarray:
    """Liest R- und G-Kanal getrennt (GL_RG als glGetTexImage-Format wird von PyOpenGLs
    Auto-Sizing nicht erkannt, siehe sedimentTransport-Dispatch) und stapelt sie zu (h,w,2)."""
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    r_data = gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RED, gl.GL_FLOAT)
    g_data = gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_GREEN, gl.GL_FLOAT)
    r = np.frombuffer(r_data, dtype=np.float32).reshape((height, width))
    g = np.frombuffer(g_data, dtype=np.float32).reshape((height, width))
    return np.stack([r, g], axis=-1).copy()


def _dispatch_compute(program, work_groups_x, work_groups_y):
    gl.glUseProgram(program)
    gl.glDispatchCompute(work_groups_x, work_groups_y, 1)
    gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)


def _gaussian_blur(worker: "GPUWorker", data: np.ndarray, sigma: float) -> np.ndarray:
    """
    Separabler Gauss-Weichzeichner (2 Passes: horizontal, vertikal) über
    shaders/utilities/gaussianBlur.comp - Portierung von scipy.ndimage.gaussian_filter()
    (Radius/Truncate-Konvention wie scipy-Default truncate=4.0).
    """
    size = data.shape[0]
    program = worker.get_program("utilities", "gaussianBlur")
    radius = int(4.0 * sigma + 0.5)
    work_groups = (size + 15) // 16

    tex_in = _upload_texture_2d(data, gl.GL_R32F)
    tex_mid = _create_texture_2d(size, size, gl.GL_R32F)
    tex_out = _create_texture_2d(size, size, gl.GL_R32F)

    gl.glBindImageTexture(0, tex_in, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(1, tex_mid, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)
    gl.glUseProgram(program)
    _set_uniforms(program, {"u_size": size, "u_sigma": sigma, "u_radius": radius, "u_horizontal": 1})
    gl.glDispatchCompute(work_groups, work_groups, 1)
    gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    gl.glBindImageTexture(0, tex_mid, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(1, tex_out, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)
    gl.glUseProgram(program)
    _set_uniforms(program, {"u_size": size, "u_sigma": sigma, "u_radius": radius, "u_horizontal": 0})
    gl.glDispatchCompute(work_groups, work_groups, 1)
    gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    result = _read_texture_data(tex_out, size, size)
    gl.glDeleteTextures(3, [tex_in, tex_mid, tex_out])
    return result


def _dispatch_noise_generation(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    size = inputs["size"]
    program = worker.get_program("terrain", "noiseGeneration")
    output_texture = _create_texture_2d(size, size, gl.GL_R32F)
    gl.glBindImageTexture(0, output_texture, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)

    gl.glUseProgram(program)
    _set_uniforms(program, {
        "u_size": size, "u_octaves": inputs["octaves"], "u_frequency": inputs["frequency"],
        "u_persistence": inputs["persistence"], "u_lacunarity": inputs["lacunarity"], "u_seed": inputs["seed"],
    })
    work_groups = (size + 15) // 16
    _dispatch_compute(program, work_groups, work_groups)
    result = _read_texture_data(output_texture, size, size)
    gl.glDeleteTextures(1, [output_texture])
    return {"success": True, "noise": result}


def _dispatch_shadow_raycast(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    heightmap = inputs["heightmap"]
    shadowmap_size = inputs.get("shadowmap_size", 64)
    heightmap_size = heightmap.shape[0]

    program = worker.get_program("terrain", "shadowRaycast")
    height_tex = _upload_texture_2d(heightmap, gl.GL_R32F)
    shadow_tex = _create_texture_2d(shadowmap_size, shadowmap_size, gl.GL_R32F)
    gl.glBindImageTexture(0, height_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(1, shadow_tex, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)

    gl.glUseProgram(program)
    _set_uniforms(program, {
        "u_heightmap_size": heightmap_size,
        "u_shadowmap_size": shadowmap_size,
        "u_sun_elevation": inputs["sun_elevation"],
        "u_sun_azimuth": inputs["sun_azimuth"],
        "u_max_distance": inputs.get("max_distance", 100.0),
        "u_step_size": inputs.get("step_size", 1.0),
        "u_height_scale": inputs.get("height_scale", 1.0),
    })
    work_groups = (shadowmap_size + 7) // 8
    _dispatch_compute(program, work_groups, work_groups)
    result = _read_texture_data(shadow_tex, shadowmap_size, shadowmap_size)
    gl.glDeleteTextures(2, [height_tex, shadow_tex])
    return {"success": True, "shadowmap": result}


def _classify_lake_basins_vectorized(heightmap: np.ndarray, seed_id_map: np.ndarray, volume_threshold: float):
    """
    Portierung von WaterGenerator._classify_lake_basins() - vektorisiert statt
    Pixel-für-Pixel-Python-Schleife, da die Menge der Kandidaten-Pixel hier
    (anders als beim eigentlichen Jump-Flooding) schon aus der GPU-Berechnung
    vorliegt und sich mit numpy in einem Rutsch aggregieren lässt.

    Wasserspiegel = Spill-Point (niedrigster Rand-Übergang zu einem
    Nachbarbecken oder zum Kartenrand), NICHT die maximale Höhe innerhalb des
    Beckens. Ein Becken (aus jumpFloodLakes.comp) ist meist viel größer als der
    eigentliche See - der Großteil ist trockenes Gelände, das nur einwärts
    entwässert. "Wasserspiegel = Becken-Maximum" macht JEDEN Punkt im Becken per
    Definition zu <= Wasserspiegel, wodurch das ganze Becken (inklusive
    Berggipfel) als überflutet zählt; siehe WaterGenerator._classify_lake_basins()
    für dieselbe Korrektur auf der CPU-Seite. Nur Pixel unterhalb des
    Spill-Points sind tatsächlich unter Wasser.

    Bekannte Einschränkung: die Becken-ZUORDNUNG selbst (aus jumpFloodLakes.comp)
    prüft weiterhin nur "current_height >= seed_height" statt echter
    Erreichbarkeit über einen monotonen Abwärtspfad - ein Becken kann dadurch
    über einen Bergkamm hinweg zu groß geraten. Der Spill-Point-Filter hier
    grenzt das Ergebnis trotzdem stark ein, weil die meisten fälschlich
    zugeordneten Hochpunkte über der Spill-Höhe liegen und damit ausgeschlossen
    werden. Ein echter paralleler Watershed-Transform für die Zuordnung selbst
    ist nicht implementiert (eigenständiges, größeres Vorhaben).
    """
    height, width = heightmap.shape
    heightmap = heightmap.astype(np.float64)
    valid = seed_id_map >= 0
    if not np.any(valid):
        return np.full((height, width), -1, dtype=np.int32), []

    unique_ids, inverse = np.unique(seed_id_map[valid].astype(np.int64), return_inverse=True)
    n_basins = len(unique_ids)

    basin_index_map = np.full((height, width), -1, dtype=np.int64)
    basin_index_map[valid] = inverse

    spill_elevation = np.full(n_basins, np.inf, dtype=np.float64)

    # Becken, die den Kartenrand berühren, haben offenen Abfluss (kein
    # geschlossener See) - Rand-Höhe zählt als (sehr niedriger) Spill-Point.
    edge_mask = np.zeros((height, width), dtype=bool)
    edge_mask[0, :] = edge_mask[-1, :] = edge_mask[:, 0] = edge_mask[:, -1] = True
    edge_valid = valid & edge_mask
    if np.any(edge_valid):
        np.minimum.at(spill_elevation, basin_index_map[edge_valid], heightmap[edge_valid])

    # 8-Nachbarschafts-Scan für Becken-Grenzübergänge (ohne np.roll, das an den
    # Kartenrändern zyklisch umbrechen würde) - für jedes Pixel, dessen direkter
    # Nachbar einem ANDEREN Becken angehört, ist max(eigene Höhe, Nachbar-Höhe)
    # ein Kandidat für den Spill-Point des eigenen Beckens.
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        y0, y1 = max(0, dy), height + min(0, dy)
        x0, x1 = max(0, dx), width + min(0, dx)
        ny0, ny1 = max(0, -dy), height + min(0, -dy)
        nx0, nx1 = max(0, -dx), width + min(0, -dx)

        cur_basin = basin_index_map[y0:y1, x0:x1]
        nbr_basin = basin_index_map[ny0:ny1, nx0:nx1]
        cur_valid = valid[y0:y1, x0:x1]
        cur_height = heightmap[y0:y1, x0:x1]
        nbr_height = heightmap[ny0:ny1, nx0:nx1]

        cross_mask = cur_valid & (nbr_basin != cur_basin)
        if np.any(cross_mask):
            crossing_heights = np.maximum(cur_height[cross_mask], nbr_height[cross_mask])
            np.minimum.at(spill_elevation, cur_basin[cross_mask], crossing_heights)

    valid_heights = heightmap[valid]
    per_pixel_spill = spill_elevation[inverse]
    submerged = valid_heights <= per_pixel_spill
    depth = np.where(submerged, per_pixel_spill - valid_heights, 0.0)

    total_volume = np.bincount(inverse, weights=depth, minlength=n_basins)

    filtered_lake_map = np.full((height, width), -1, dtype=np.int32)
    valid_lakes = []
    flat_valid_indices = np.flatnonzero(valid.ravel())
    finite_spill = np.isfinite(spill_elevation)

    for local_idx, seed_flat_id in enumerate(unique_ids):
        if not finite_spill[local_idx] or total_volume[local_idx] < volume_threshold:
            continue
        pixel_mask = (inverse == local_idx) & submerged
        if not np.any(pixel_mask):
            continue
        pixel_flat_indices = flat_valid_indices[pixel_mask]
        compact_id = len(valid_lakes)
        filtered_lake_map.ravel()[pixel_flat_indices] = compact_id
        seed_y, seed_x = divmod(int(seed_flat_id), width)
        valid_lakes.append({
            "seed": (seed_x, seed_y),
            "volume": float(total_volume[local_idx]),
            "pixels": int(pixel_mask.sum()),
        })

    return filtered_lake_map, valid_lakes


def _dispatch_jump_flood_lakes(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    heightmap = inputs["heightmap"]
    volume_threshold = inputs["lake_volume_threshold"]
    size = heightmap.shape[0]

    seed_program = worker.get_program("water", "localMinimaSeed")
    flood_program = worker.get_program("water", "jumpFloodLakes")

    height_tex = _upload_texture_2d(heightmap, gl.GL_R32F)
    state_a = _create_texture_2d(size, size, gl.GL_RG32F)
    state_b = _create_texture_2d(size, size, gl.GL_RG32F)

    work_groups = (size + 15) // 16

    gl.glBindImageTexture(0, height_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(1, state_a, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_RG32F)
    gl.glUseProgram(seed_program)
    _set_uniforms(seed_program, {"u_size": size})
    gl.glDispatchCompute(work_groups, work_groups, 1)
    gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    # Jump-Flooding: Sprungdistanz beginnt bei der kleinsten Zweierpotenz >= max_dim
    # und halbiert sich bis 1 - exakt dieselbe Folge wie _apply_jump_flooding().
    jump_distance = 1
    while jump_distance < size:
        jump_distance *= 2

    read_tex, write_tex = state_a, state_b
    while jump_distance >= 1:
        gl.glBindImageTexture(0, height_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
        gl.glBindImageTexture(1, read_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_RG32F)
        gl.glBindImageTexture(2, write_tex, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_RG32F)
        gl.glUseProgram(flood_program)
        _set_uniforms(flood_program, {"u_size": size, "u_jump_distance": jump_distance})
        gl.glDispatchCompute(work_groups, work_groups, 1)
        gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        read_tex, write_tex = write_tex, read_tex
        jump_distance //= 2

    seed_id_map = _read_texture_data(read_tex, size, size).astype(np.int32)
    gl.glDeleteTextures(3, [height_tex, state_a, state_b])

    lake_map, valid_lakes = _classify_lake_basins_vectorized(heightmap, seed_id_map, volume_threshold)
    return {"success": True, "lake_map": lake_map, "valid_lakes": valid_lakes}


def _compute_steepest_descent_texture(worker: "GPUWorker", height_tex, size: int):
    """Dispatcht steepestDescent.comp und gibt die Richtungs-TEXTUR zurück (kein Readback -
    wird von _dispatch_steepest_descent() und _dispatch_flow_network() gemeinsam genutzt,
    um bei Letzterer einen unnötigen Readback+Reupload-Umweg zu vermeiden."""
    program = worker.get_program("water", "steepestDescent")
    direction_tex = _create_texture_2d(size, size, gl.GL_R32F)
    gl.glBindImageTexture(0, height_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(1, direction_tex, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)
    gl.glUseProgram(program)
    _set_uniforms(program, {"u_size": size})
    work_groups = (size + 15) // 16
    _dispatch_compute(program, work_groups, work_groups)
    return direction_tex


def _dispatch_steepest_descent(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    heightmap = inputs["heightmap"]
    size = heightmap.shape[0]

    height_tex = _upload_texture_2d(heightmap, gl.GL_R32F)
    direction_tex = _compute_steepest_descent_texture(worker, height_tex, size)

    result = _read_texture_data(direction_tex, size, size)
    gl.glDeleteTextures(2, [height_tex, direction_tex])
    flow_directions = np.rint(result).astype(np.int8)
    return {"success": True, "flow_directions": flow_directions}


def _dispatch_flow_network(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    heightmap = inputs["heightmap"]
    precip_map = inputs["precip_map"]
    rain_threshold = inputs["rain_threshold"]
    max_iterations = max(1, int(inputs["max_iterations"]))
    size = heightmap.shape[0]

    initial_water = np.where(precip_map > rain_threshold, precip_map, 0.0).astype(np.float32)

    height_tex = _upload_texture_2d(heightmap, gl.GL_R32F)
    direction_tex = _compute_steepest_descent_texture(worker, height_tex, size)
    initial_tex = _upload_texture_2d(initial_water, gl.GL_R32F)
    accum_a = _upload_texture_2d(initial_water, gl.GL_R32F)
    accum_b = _create_texture_2d(size, size, gl.GL_R32F)

    program = worker.get_program("water", "flowAccumulation")
    work_groups = (size + 15) // 16
    read_tex, write_tex = accum_a, accum_b
    for _ in range(max_iterations):
        gl.glBindImageTexture(0, direction_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
        gl.glBindImageTexture(1, initial_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
        gl.glBindImageTexture(2, read_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
        gl.glBindImageTexture(3, write_tex, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)
        gl.glUseProgram(program)
        _set_uniforms(program, {"u_size": size})
        gl.glDispatchCompute(work_groups, work_groups, 1)
        gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        read_tex, write_tex = write_tex, read_tex

    flow_accumulation = _read_texture_data(read_tex, size, size)
    gl.glDeleteTextures(5, [height_tex, direction_tex, initial_tex, accum_a, accum_b])
    return {"success": True, "flow_accumulation": flow_accumulation}


def _dispatch_manning_flow(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    flow_accumulation = inputs["flow_accumulation"]
    slopemap = inputs["slopemap"]
    heightmap = inputs["heightmap"]
    size = heightmap.shape[0]

    program = worker.get_program("water", "manningFlowCalculation")
    flow_tex = _upload_texture_2d(flow_accumulation, gl.GL_R32F)
    slope_tex = _upload_texture_2d_rg(slopemap, gl.GL_RG32F)
    height_tex = _upload_texture_2d(heightmap, gl.GL_R32F)
    speed_tex = _create_texture_2d(size, size, gl.GL_R32F)
    cross_tex = _create_texture_2d(size, size, gl.GL_R32F)

    gl.glBindImageTexture(0, flow_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(1, slope_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_RG32F)
    gl.glBindImageTexture(2, height_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(3, speed_tex, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(4, cross_tex, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)

    gl.glUseProgram(program)
    _set_uniforms(program, {
        "u_size": size, "u_manning_n": inputs["manning_n"], "u_depth_tests": inputs["depth_tests"],
    })
    work_groups = (size + 15) // 16
    _dispatch_compute(program, work_groups, work_groups)

    flow_speed = _read_texture_data(speed_tex, size, size)
    cross_section = _read_texture_data(cross_tex, size, size)
    gl.glDeleteTextures(5, [flow_tex, slope_tex, height_tex, speed_tex, cross_tex])
    return {"success": True, "flow_speed": flow_speed, "cross_section": cross_section}


def _dispatch_stream_power_erosion(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    flow_accumulation = inputs["flow_accumulation"]
    flow_speed = inputs["flow_speed"]
    hardness_map = inputs["hardness_map"]
    size = flow_accumulation.shape[0]

    program = worker.get_program("water", "streamPowerErosion")
    flow_tex = _upload_texture_2d(flow_accumulation, gl.GL_R32F)
    speed_tex = _upload_texture_2d(flow_speed, gl.GL_R32F)
    hardness_tex = _upload_texture_2d(hardness_map, gl.GL_R32F)
    erosion_tex = _create_texture_2d(size, size, gl.GL_R32F)

    gl.glBindImageTexture(0, flow_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(1, speed_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(2, hardness_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(3, erosion_tex, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)

    gl.glUseProgram(program)
    _set_uniforms(program, {"u_size": size, "u_erosion_strength": inputs["erosion_strength"]})
    work_groups = (size + 15) // 16
    _dispatch_compute(program, work_groups, work_groups)

    erosion_map = _read_texture_data(erosion_tex, size, size)
    gl.glDeleteTextures(4, [flow_tex, speed_tex, hardness_tex, erosion_tex])
    return {"success": True, "erosion_map": erosion_map}


def _dispatch_sediment_transport(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    erosion_map = inputs["erosion_map"]
    flow_speed = inputs["flow_speed"]
    flow_directions = inputs["flow_directions"]
    capacity_factor = inputs["capacity_factor"]
    settling_velocity = inputs["settling_velocity"]
    iterations = max(1, int(inputs["iterations"]))
    size = erosion_map.shape[0]

    # Transport-Kapazitaet ist iterationsunabhaengig - vektorisiert vorab in numpy statt
    # als eigener Shader-Pass (siehe ErosionSedimentationSystem._transport_sediment_optimized()).
    transport_capacity = np.where(flow_speed > 0.1, capacity_factor * np.power(flow_speed, 2.5), 0.0).astype(np.float32)

    erosion_tex = _upload_texture_2d(erosion_map, gl.GL_R32F)
    speed_tex = _upload_texture_2d(flow_speed, gl.GL_R32F)
    direction_tex = _upload_texture_2d(flow_directions, gl.GL_R32F)
    capacity_tex = _upload_texture_2d(transport_capacity, gl.GL_R32F)
    zero_state = np.zeros((size, size, 2), dtype=np.float32)
    state_a = _upload_texture_2d_rg(zero_state, gl.GL_RG32F)
    state_b = _create_texture_2d(size, size, gl.GL_RG32F)

    program = worker.get_program("water", "sedimentTransport")
    work_groups = (size + 15) // 16
    read_tex, write_tex = state_a, state_b
    for _ in range(iterations):
        gl.glBindImageTexture(0, erosion_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
        gl.glBindImageTexture(1, speed_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
        gl.glBindImageTexture(2, direction_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
        gl.glBindImageTexture(3, capacity_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
        gl.glBindImageTexture(4, read_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_RG32F)
        gl.glBindImageTexture(5, write_tex, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_RG32F)
        gl.glUseProgram(program)
        _set_uniforms(program, {"u_size": size, "u_settling_velocity": settling_velocity})
        gl.glDispatchCompute(work_groups, work_groups, 1)
        gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        read_tex, write_tex = write_tex, read_tex

    # .r (sediment_load, hier nicht benötigt) und .g (sedimentation_map) getrennt lesen -
    # GL_RG als glGetTexImage-Format wird von PyOpenGLs Auto-Sizing nicht erkannt, GL_GREEN
    # (Einzelkanal wie GL_RED) dagegen schon.
    gl.glBindTexture(gl.GL_TEXTURE_2D, read_tex)
    raw = gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_GREEN, gl.GL_FLOAT)
    sedimentation_map = np.frombuffer(raw, dtype=np.float32).reshape((size, size)).copy()

    gl.glDeleteTextures(6, [erosion_tex, speed_tex, direction_tex, capacity_tex, state_a, state_b])
    return {"success": True, "sedimentation_map": sedimentation_map}


def _dispatch_soil_moisture(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    water_biomes_map = inputs["water_biomes_map"]
    flow_accumulation = inputs["flow_accumulation"]
    diffusion_radius = inputs["diffusion_radius"]
    # Kapillare Ausbreitung: fester, kleiner Radius, identischer Fix wie im
    # CPU-Pfad (SoilMoistureCalculator.__init__) - war hart auf 2.0 kodiert,
    # siehe dortiger Docstring-Kommentar für die volle Begründung
    # ([[project-water-flood-calibration]]).
    capillary_sigma = inputs.get("capillary_sigma", 0.5)
    # Zentrallinie statt der ggf. breiter gemalten water_biomes_map als 100%-
    # Feuchte-Quellfläche - identischer Fix wie im CPU-Pfad
    # (SoilMoistureCalculator._cpu_soil_moisture_calculation()), entkoppelt
    # die Boden-Feuchte-Ausdehnung von der visuellen Flussbreite. Fällt auf
    # water_biomes_map zurück, falls kein separater Key übergeben wurde.
    water_mask_source = inputs.get("water_mask_source", water_biomes_map)

    # Portierung von SoilMoistureCalculator._cpu_soil_moisture_calculation() - die
    # Quell-Karten-Aufbereitung ist bereits mit numpy vektorisierbar (keine eigene
    # Shader-Passe nötig), nur die beiden Gaussian-Diffusionen laufen auf der GPU.
    water_mask = water_mask_source > 0
    capillary_source = np.where(water_mask, 100.0, 0.0).astype(np.float32)

    groundwater_source = np.zeros_like(capillary_source)
    groundwater_source[water_mask_source == 4] = 80.0
    mask3 = water_mask_source == 3
    groundwater_source[mask3] = 60.0 + np.minimum(20.0, flow_accumulation[mask3] * 0.1)
    mask2 = water_mask_source == 2
    groundwater_source[mask2] = 40.0 + np.minimum(20.0, flow_accumulation[mask2] * 0.2)
    mask1 = water_mask_source == 1
    groundwater_source[mask1] = 20.0 + np.minimum(10.0, flow_accumulation[mask1] * 0.3)

    capillary_moisture = _gaussian_blur(worker, capillary_source, capillary_sigma)
    groundwater_moisture = _gaussian_blur(worker, groundwater_source, diffusion_radius)

    combined_moisture = np.maximum(capillary_moisture, groundwater_moisture)
    combined_moisture[water_mask] = 100.0
    return {"success": True, "soil_moisture": combined_moisture.astype(np.float32)}


def _dispatch_atmospheric_evaporation(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    temp_map = inputs["temp_map"]
    wind_map = inputs["wind_map"]
    humid_map = inputs["humid_map"]
    water_biomes_map = inputs["water_biomes_map"]
    size = temp_map.shape[0]

    program = worker.get_program("water", "atmosphericEvaporation")
    temp_tex = _upload_texture_2d(temp_map, gl.GL_R32F)
    wind_tex = _upload_texture_2d_rg(wind_map, gl.GL_RG32F)
    humid_tex = _upload_texture_2d(humid_map, gl.GL_R32F)
    biomes_tex = _upload_texture_2d(water_biomes_map.astype(np.float32), gl.GL_R32F)
    output_tex = _create_texture_2d(size, size, gl.GL_R32F)

    gl.glBindImageTexture(0, temp_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(1, wind_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_RG32F)
    gl.glBindImageTexture(2, humid_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(3, biomes_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(4, output_tex, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)

    gl.glUseProgram(program)
    _set_uniforms(program, {"u_size": size, "u_base_rate": inputs["base_rate"]})
    work_groups = (size + 15) // 16
    _dispatch_compute(program, work_groups, work_groups)

    evaporation_map = _read_texture_data(output_tex, size, size)
    gl.glDeleteTextures(5, [temp_tex, wind_tex, humid_tex, biomes_tex, output_tex])
    return {"success": True, "evaporation_map": evaporation_map}


def _dispatch_temperature_calculation(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    heightmap = inputs["heightmap"]
    shadowmap = inputs["shadowmap"]
    height, width = heightmap.shape

    # Mittelung über die Shadow-Angle-Achse ist trivial vektorisierbar - keine eigene
    # Shader-Passe nötig (siehe WeatherSystemGenerator._calculate_temperature_cpu_optimized()).
    shadow_weighted = np.mean(shadowmap, axis=2) if shadowmap.ndim == 3 else shadowmap
    seed_offset = (int(inputs["map_seed"]) % 1000) * 0.1

    program = worker.get_program("weather", "temperatureCalculation")
    height_tex = _upload_texture_2d(heightmap, gl.GL_R32F)
    shadow_tex = _upload_texture_2d(shadow_weighted, gl.GL_R32F)
    output_tex = _create_texture_2d(width, height, gl.GL_R32F)

    gl.glBindImageTexture(0, height_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(1, shadow_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(2, output_tex, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)

    gl.glUseProgram(program)
    _set_uniforms(program, {
        "u_width": width, "u_height": height, "u_target_size": int(inputs["lod_level"]),
        "u_air_temp_entry": inputs["air_temp_entry"], "u_solar_power": inputs["solar_power"],
        "u_altitude_cooling": inputs["altitude_cooling"], "u_seed_offset": seed_offset,
    })
    work_groups_x = (width + 15) // 16
    work_groups_y = (height + 15) // 16
    _dispatch_compute(program, work_groups_x, work_groups_y)

    temperature_field = _read_texture_data(output_tex, width, height)
    gl.glDeleteTextures(3, [height_tex, shadow_tex, output_tex])
    return {"success": True, "temperature_field": temperature_field}


def _dispatch_pressure_noise(worker: "GPUWorker", width: int, height: int, seed_offset: float) -> np.ndarray:
    program = worker.get_program("weather", "pressureNoise")
    output_tex = _create_texture_2d(width, height, gl.GL_R32F)
    gl.glBindImageTexture(0, output_tex, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)
    gl.glUseProgram(program)
    _set_uniforms(program, {"u_width": width, "u_height": height, "u_seed_offset": seed_offset})
    work_groups_x = (width + 15) // 16
    work_groups_y = (height + 15) // 16
    _dispatch_compute(program, work_groups_x, work_groups_y)
    result = _read_texture_data(output_tex, width, height)
    gl.glDeleteTextures(1, [output_tex])
    return result


def _run_vec2_pass(program, field: np.ndarray, width: int, height: int,
                    work_groups_x: int, work_groups_y: int, extra_uniforms: dict) -> np.ndarray:
    """Ein Ping-Pong-Durchlauf für einen (h,w,2)-Vektorpass (Wind-Diffusion,
    Kontinuitäts-Korrektur) - lädt hoch, dispatcht einmal, liest zurück."""
    tex_in = _upload_texture_2d_rg(field, gl.GL_RG32F)
    tex_out = _create_texture_2d(width, height, gl.GL_RG32F)
    gl.glBindImageTexture(0, tex_in, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_RG32F)
    gl.glBindImageTexture(1, tex_out, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_RG32F)
    gl.glUseProgram(program)
    uniforms = {"u_width": width, "u_height": height}
    uniforms.update(extra_uniforms)
    _set_uniforms(program, uniforms)
    gl.glDispatchCompute(work_groups_x, work_groups_y, 1)
    gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
    result = _read_texture_data_rg(tex_out, width, height)
    gl.glDeleteTextures(2, [tex_in, tex_out])
    return result


def _apply_thermal_convection_numpy(wind_field: np.ndarray, temp_map: np.ndarray,
                                     shadowmap: np.ndarray, thermic_effect: float):
    """Portierung von WeatherSystemGenerator._apply_thermal_convection() - bereits im
    CPU-Original vollständig vektorisiert (keine Python-Schleife), 1:1 übernommen statt
    unnötig auf die GPU verlagert."""
    temp_grad_x = np.zeros_like(temp_map)
    temp_grad_y = np.zeros_like(temp_map)
    temp_grad_x[:, 1:-1] = (temp_map[:, 2:] - temp_map[:, :-2]) * 0.5
    temp_grad_y[1:-1, :] = (temp_map[2:, :] - temp_map[:-2, :]) * 0.5

    avg_temp = np.mean(temp_map)
    convection_strength = (temp_map - avg_temp) * thermic_effect * 0.08

    shadow_avg = np.mean(shadowmap, axis=2) if shadowmap.ndim == 3 else shadowmap
    shadow_effect = (shadow_avg - 0.5) * thermic_effect * 0.15

    wind_field[:, :, 0] += temp_grad_x * 0.05 + convection_strength
    wind_field[:, :, 1] += temp_grad_y * 0.05 + shadow_effect


def _dispatch_wind_field_cfd(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    heightmap = inputs["heightmap"]
    temp_map = inputs["temp_map"]
    shadowmap = inputs["shadowmap"]
    wind_speed_factor = float(inputs["wind_speed_factor"])
    terrain_factor = float(inputs["terrain_factor"])
    thermic_effect = float(inputs["thermic_effect"])
    cfd_iterations = max(1, int(inputs["cfd_iterations"]))
    map_seed = int(inputs["map_seed"])
    month_index = int(inputs.get("month_index", 0))
    height, width = heightmap.shape

    # Slopemap + initiales Druckfeld sind im CPU-Original bereits vektorisiert (numpy) -
    # 1:1 übernommen statt unnötig auf die GPU verlagert (siehe
    # WeatherSystemGenerator._calculate_slopes_vectorized() / _simulate_wind_field_cpu_cfd()).
    slopemap = np.zeros((height, width, 2), dtype=np.float32)
    slopemap[:, 1:-1, 0] = (heightmap[:, 2:] - heightmap[:, :-2]) * 0.5
    slopemap[:, 0, 0] = heightmap[:, 1] - heightmap[:, 0]
    slopemap[:, -1, 0] = heightmap[:, -1] - heightmap[:, -2]
    slopemap[1:-1, :, 1] = (heightmap[2:, :] - heightmap[:-2, :]) * 0.5
    slopemap[0, :, 1] = heightmap[1, :] - heightmap[0, :]
    slopemap[-1, :, 1] = heightmap[-1, :] - heightmap[-2, :]

    # Lineares Druckgefälle entlang der vorherrschenden Windrichtung (Grad,
    # math. Konvention: 0°=Wind Richtung +x/Ost, 90°=+y) - 1:1 dieselbe
    # Formel wie WeatherSystemGenerator._build_directional_pressure_field()
    # (CPU-Pfad), hier separat gehalten statt geteilt, da GPU/CPU-Pfade in
    # dieser Codebase durchgängig unabhängige Implementierungen sind. Bei
    # wind_direction_deg=0 identisch zur alten hartcodierten West-Ost-Formel.
    wind_direction_deg = float(inputs.get("prevailing_wind_direction", 0.0))
    theta = np.radians(wind_direction_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    y_idx, x_idx = np.mgrid[0:height, 0:width]
    s = x_idx * dx + y_idx * dy
    s_min, s_max = s.min(), s.max()
    s_range = s_max - s_min
    s_norm = (s - s_min) / s_range if s_range > 1e-9 else np.zeros_like(s, dtype=np.float32)
    pressure_field = (1.0 - s_norm * 0.3).astype(np.float32)

    # month_index * 10.0 - identischer Offset-Mechanismus wie
    # WeatherSystemGenerator._generate_pressure_noise() (CPU-Pfad): macht das
    # kleinräumige Rauschmuster pro saisonaler Periode optisch unterscheidbar
    # statt für alle 6 Monate identisch (u_seed_offset in pressureNoise.comp
    # ist bereits ein Koordinaten-Offset, kein Shader-Change nötig).
    seed_offset = (map_seed % 1000) * 0.1 + month_index * 10.0
    pressure_field += _dispatch_pressure_noise(worker, width, height, seed_offset) * 0.15

    # Druck-Terrain-Kopplung - identische Formel wie
    # WeatherSystemGenerator._simulate_wind_field_cpu_cfd() (CPU-Pfad): hohe
    # Punkte senken lokal den effektiven Druck (grobe Orographie-Näherung).
    # Anders als der bestehende additive Terrain-Ablenkungs-Term weiter unten
    # (der nur einmalig auf wind_field draufaddiert wird) fließt dieser Term
    # VOR der Gradientenberechnung ins Druckfeld selbst ein.
    height_range = heightmap.max() - heightmap.min()
    height_normalized = (heightmap - heightmap.min()) / height_range if height_range > 1e-6 \
        else np.zeros_like(heightmap, dtype=np.float32)
    terrain_pressure_term = height_normalized * terrain_factor * 0.2
    pressure_field_coupled = pressure_field - terrain_pressure_term

    pressure_grad_x = np.zeros_like(pressure_field)
    pressure_grad_y = np.zeros_like(pressure_field)
    pressure_grad_x[:, 1:-1] = (pressure_field_coupled[:, 2:] - pressure_field_coupled[:, :-2]) * 0.5
    pressure_grad_y[1:-1, :] = (pressure_field_coupled[2:, :] - pressure_field_coupled[:-2, :]) * 0.5

    wind_field = np.zeros((height, width, 2), dtype=np.float32)
    wind_field[:, :, 0] = -pressure_grad_x * wind_speed_factor * 10.0
    wind_field[:, :, 1] = -pressure_grad_y * wind_speed_factor * 10.0

    terrain_factor_scaled = terrain_factor * 0.5
    wind_field[:, :, 0] += slopemap[:, :, 1] * terrain_factor_scaled
    wind_field[:, :, 1] -= slopemap[:, :, 0] * terrain_factor_scaled

    # Die beiden pro Iteration wiederholten, teuren verschachtelten Python-Schleifen
    # (Diffusion, Kontinuitäts-Korrektur - bis zu 25x bei hohem LOD, gemessen als
    # dominanter Weather-Kostenfaktor) laufen auf der GPU; Thermal-Convection bleibt
    # vektorisiertes numpy (bereits schnell).
    diffusion_program = worker.get_program("weather", "windDiffusion")
    correction_program = worker.get_program("weather", "continuityCorrection")
    work_groups_x = (width + 15) // 16
    work_groups_y = (height + 15) // 16

    for _ in range(cfd_iterations):
        _apply_thermal_convection_numpy(wind_field, temp_map, shadowmap, thermic_effect)
        wind_field = _run_vec2_pass(diffusion_program, wind_field, width, height,
                                     work_groups_x, work_groups_y, {"u_diffusion_rate": 0.1})
        wind_field = _run_vec2_pass(correction_program, wind_field, width, height,
                                     work_groups_x, work_groups_y, {})

    return {"success": True, "wind_field": wind_field}


def _run_scalar_pass(program, field: np.ndarray, width: int, height: int,
                      work_groups_x: int, work_groups_y: int, extra_uniforms: dict,
                      extra_tex_bindings=None) -> np.ndarray:
    """Ein Ping-Pong-Durchlauf für einen (h,w)-Skalarpass (Humidity-Diffusion) - lädt
    hoch, dispatcht einmal, liest zurück. extra_tex_bindings: Liste von
    (binding_index, texture_id, format) für zusätzliche Read-Only-Eingaben (z.B. Windfeld)."""
    tex_in = _upload_texture_2d(field, gl.GL_R32F)
    tex_out = _create_texture_2d(width, height, gl.GL_R32F)
    gl.glBindImageTexture(0, tex_in, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    last_binding = 1
    if extra_tex_bindings:
        for binding, tex_id, fmt in extra_tex_bindings:
            gl.glBindImageTexture(binding, tex_id, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, fmt)
            last_binding = max(last_binding, binding + 1)
    gl.glBindImageTexture(last_binding, tex_out, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)
    gl.glUseProgram(program)
    uniforms = {"u_width": width, "u_height": height}
    uniforms.update(extra_uniforms)
    _set_uniforms(program, uniforms)
    gl.glDispatchCompute(work_groups_x, work_groups_y, 1)
    gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
    result = _read_texture_data(tex_out, width, height)
    gl.glDeleteTextures(1, [tex_in])
    gl.glDeleteTextures(1, [tex_out])
    return result


def _dispatch_atmospheric_moisture(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    temp_map = inputs["temp_map"]
    wind_map = inputs["wind_map"]
    height, width = temp_map.shape

    # Magnus-Formel/Evaporation-Setup ist im CPU-Original bereits vollständig
    # numpy-vektorisiert (soil_moisture ist konstant 50.0 -> soil_moisture/100=0.5) -
    # 1:1 übernommen (siehe WeatherSystemGenerator._calculate_atmospheric_moisture_cpu()).
    temp_celsius = np.clip(temp_map, -50, 60)
    saturation_vapor_pressure = 6.112 * np.exp(17.67 * temp_celsius / (temp_celsius + 243.5))
    wind_speed = np.sqrt(wind_map[:, :, 0] ** 2 + wind_map[:, :, 1] ** 2)
    wind_factor = np.minimum(2.0, wind_speed / 5.0)
    evaporation_rate = 0.5 * (saturation_vapor_pressure / 100.0) * (1.0 + wind_factor)
    humid_map = (evaporation_rate * 10.0).astype(np.float32)

    transport_program = worker.get_program("weather", "moistureTransport")
    diffusion_program = worker.get_program("weather", "humidityDiffusion")
    work_groups_x = (width + 15) // 16
    work_groups_y = (height + 15) // 16

    wind_tex = _upload_texture_2d_rg(wind_map, gl.GL_RG32F)
    for _ in range(3):
        humid_map = _run_scalar_pass(
            transport_program, humid_map, width, height, work_groups_x, work_groups_y,
            {"u_dt": 0.5}, extra_tex_bindings=[(1, wind_tex, gl.GL_RG32F)])
    gl.glDeleteTextures(1, [wind_tex])

    for _ in range(2):
        humid_map = _run_scalar_pass(
            diffusion_program, humid_map, width, height, work_groups_x, work_groups_y,
            {"u_diffusion_rate": 0.08})

    return {"success": True, "humidity_field": humid_map}


def _dispatch_precipitation_calculation(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    humid_map = inputs["humid_map"]
    temp_map = inputs["temp_map"]
    wind_map = inputs["wind_map"]
    heightmap = inputs["heightmap"]
    height, width = humid_map.shape

    program = worker.get_program("weather", "precipitationCalculation")
    humid_tex = _upload_texture_2d(humid_map, gl.GL_R32F)
    temp_tex = _upload_texture_2d(temp_map, gl.GL_R32F)
    wind_tex = _upload_texture_2d_rg(wind_map, gl.GL_RG32F)
    height_tex = _upload_texture_2d(heightmap, gl.GL_R32F)
    output_tex = _create_texture_2d(width, height, gl.GL_R32F)

    gl.glBindImageTexture(0, humid_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(1, temp_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(2, wind_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_RG32F)
    gl.glBindImageTexture(3, height_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(4, output_tex, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)

    gl.glUseProgram(program)
    _set_uniforms(program, {"u_width": width, "u_height": height})
    work_groups_x = (width + 15) // 16
    work_groups_y = (height + 15) // 16
    _dispatch_compute(program, work_groups_x, work_groups_y)

    precipitation_field = _read_texture_data(output_tex, width, height)
    gl.glDeleteTextures(5, [humid_tex, temp_tex, wind_tex, height_tex, output_tex])
    return {"success": True, "precipitation_field": precipitation_field}


def _dispatch_climate_classification(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    temp_map = inputs["temp_map"]
    precip_map = inputs["precip_map"]
    height, width = temp_map.shape

    program = worker.get_program("biome", "climateClassification")
    temp_tex = _upload_texture_2d(temp_map, gl.GL_R32F)
    precip_tex = _upload_texture_2d(precip_map, gl.GL_R32F)
    output_tex = _create_texture_2d(width, height, gl.GL_R32F)

    gl.glBindImageTexture(0, temp_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(1, precip_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(2, output_tex, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)

    gl.glUseProgram(program)
    _set_uniforms(program, {"u_width": width, "u_height": height})
    work_groups_x = (width + 15) // 16
    work_groups_y = (height + 15) // 16
    _dispatch_compute(program, work_groups_x, work_groups_y)

    climate_map = np.rint(_read_texture_data(output_tex, width, height)).astype(np.uint8)
    gl.glDeleteTextures(3, [temp_tex, precip_tex, output_tex])
    return {"success": True, "climate_map": climate_map}


def _dispatch_supersampling(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    biome_map = inputs["biome_map"].astype(np.float32)
    height, width = biome_map.shape
    super_height, super_width = height * 2, width * 2

    program = worker.get_program("biome", "supersampling")
    biome_tex = _upload_texture_2d(biome_map, gl.GL_R32F)
    prob_texs = [
        _upload_texture_2d(inputs[key], gl.GL_R32F)
        for key in ("cliff_prob", "beach_prob", "lake_edge_prob", "river_bank_prob", "snow_prob", "alpine_prob")
    ]
    output_tex = _create_texture_2d(super_width, super_height, gl.GL_R32F)

    gl.glBindImageTexture(0, biome_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    for i, tex in enumerate(prob_texs):
        gl.glBindImageTexture(1 + i, tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(7, output_tex, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)

    gl.glUseProgram(program)
    _set_uniforms(program, {
        "u_width": width, "u_height": height,
        "u_biome_seed": inputs["biome_seed"], "u_supersampling_quality": inputs["supersampling_quality"],
    })
    work_groups_x = (super_width + 15) // 16
    work_groups_y = (super_height + 15) // 16
    _dispatch_compute(program, work_groups_x, work_groups_y)

    biome_map_super = np.rint(_read_texture_data(output_tex, super_width, super_height)).astype(np.uint8)
    gl.glDeleteTextures(1, [biome_tex])
    gl.glDeleteTextures(len(prob_texs), prob_texs)
    gl.glDeleteTextures(1, [output_tex])
    return {"success": True, "biome_map_super": biome_map_super}


def _dispatch_ocean_connectivity(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    heightmap = inputs["heightmap"]
    sea_level = inputs["sea_level"]
    height, width = heightmap.shape

    seed_program = worker.get_program("biome", "oceanConnectivitySeed")
    propagate_program = worker.get_program("biome", "oceanConnectivityPropagate")
    work_groups_x = (width + 15) // 16
    work_groups_y = (height + 15) // 16

    height_tex = _upload_texture_2d(heightmap, gl.GL_R32F)
    state_a = _create_texture_2d(width, height, gl.GL_R32F)
    state_b = _create_texture_2d(width, height, gl.GL_R32F)

    gl.glBindImageTexture(0, height_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
    gl.glBindImageTexture(1, state_a, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)
    gl.glUseProgram(seed_program)
    _set_uniforms(seed_program, {"u_width": width, "u_height": height, "u_sea_level": sea_level})
    gl.glDispatchCompute(work_groups_x, work_groups_y, 1)
    gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    # width+height (Manhattan-Diagonale) reichte bei stark verwinkelten Küstenlinien
    # NICHT (Restfehler gegen die CPU-BFS gemessen) - eine gewundene, zusammenhängende
    # Region kann eine Weglänge deutlich über die reine Diagonale hinaus brauchen.
    # width*height ist die einzige mathematisch garantierte obere Schranke (worst-case
    # Schlangenpfad durch die ganze Karte), aber für reale Heightmaps deutlich zu viele
    # Iterationen - 4x(width+height) ist ein in Tests verifizierter, sicherer
    # Kompromiss (siehe oceanConnectivityPropagate.comp).
    iterations = 4 * (width + height)
    read_tex, write_tex = state_a, state_b
    for _ in range(iterations):
        gl.glBindImageTexture(0, height_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
        gl.glBindImageTexture(1, read_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
        gl.glBindImageTexture(2, write_tex, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)
        gl.glUseProgram(propagate_program)
        _set_uniforms(propagate_program, {"u_width": width, "u_height": height, "u_sea_level": sea_level})
        gl.glDispatchCompute(work_groups_x, work_groups_y, 1)
        gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        read_tex, write_tex = write_tex, read_tex

    ocean_mask = _read_texture_data(read_tex, width, height) > 0.5
    gl.glDeleteTextures(3, [height_tex, state_a, state_b])
    return {"success": True, "ocean_mask": ocean_mask}


def _dispatch_terrain_cost_flood(worker: "GPUWorker", inputs: dict, parameters: dict) -> dict:
    """
    GPU-Gegenstueck zu _terrain_cost_voronoi() (core/settlement_generator.py) -
    genutzt sowohl von CityBoundaryAnalyzer (ein Aufruf pro Settlement, mit
    max_cost = radius * reach_factor) als auch von LandscapeVoronoiSystem
    (ein globaler Aufruf mit allen Plot-Seeds, ohne max_cost-Limit). Die initiale
    Seed-Zuordnung wird als (H,W,2)-Zustandstextur direkt aus Python hochgeladen
    (Kanal 0 = seed_id oder -1, Kanal 1 = kumulierte Kosten) statt ueber einen
    eigenen Seed-Init-Shader - anders als bei jumpFloodLakes.comp (lokale Minima
    sind pro Pixel lokal entscheidbar) sind Settlement-Seeds beliebige, auf der
    CPU berechnete Punktpositionen, die sich nicht aus der Heightmap ableiten
    lassen.
    """
    slopemap = inputs["slopemap"]
    seed_positions = inputs["seed_positions"]
    terrain_factor = float(inputs["terrain_factor"])
    max_cost = inputs.get("max_cost")
    max_cost_value = float(max_cost) if max_cost is not None else 1e9
    size = slopemap.shape[0]

    if not seed_positions:
        return {"success": True, "nearest_seed_map": np.full((size, size), -1, dtype=np.int32)}

    slope_magnitude = np.sqrt(slopemap[..., 0] ** 2 + slopemap[..., 1] ** 2).astype(np.float32)

    initial_state = np.full((size, size, 2), [-1.0, 1e9], dtype=np.float32)
    for seed_id, (sx, sy) in enumerate(seed_positions):
        ix, iy = int(round(sx)), int(round(sy))
        if 0 <= ix < size and 0 <= iy < size:
            initial_state[iy, ix] = [float(seed_id), 0.0]

    program = worker.get_program("settlement", "terrainCostFlood")
    slope_tex = _upload_texture_2d(slope_magnitude, gl.GL_R32F)
    state_a = _upload_texture_2d_rg(initial_state, gl.GL_RG32F)
    state_b = _create_texture_2d(size, size, gl.GL_RG32F)

    work_groups = (size + 15) // 16

    # Sprungfolge wie jumpFloodLakes.comp: kleinste Zweierpotenz >= size, dann halbierend bis 1
    jump_distance = 1
    while jump_distance < size:
        jump_distance *= 2

    read_tex, write_tex = state_a, state_b
    while jump_distance >= 1:
        gl.glBindImageTexture(0, slope_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
        gl.glBindImageTexture(1, read_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_RG32F)
        gl.glBindImageTexture(2, write_tex, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_RG32F)
        gl.glUseProgram(program)
        _set_uniforms(program, {
            "u_size": size, "u_jump_distance": jump_distance,
            "u_terrain_factor": terrain_factor, "u_max_cost": max_cost_value,
        })
        gl.glDispatchCompute(work_groups, work_groups, 1)
        gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        read_tex, write_tex = write_tex, read_tex
        jump_distance //= 2

    result_state = _read_texture_data_rg(read_tex, size, size)
    gl.glDeleteTextures(3, [slope_tex, state_a, state_b])

    nearest_seed_map = np.rint(result_state[..., 0]).astype(np.int32)
    cost_map = result_state[..., 1].astype(np.float32)
    return {"success": True, "nearest_seed_map": nearest_seed_map, "cost_map": cost_map}


DISPATCH_TABLE = {
    ("terrain", "noiseGeneration"): _dispatch_noise_generation,
    ("terrain", "shadowRaycast"): _dispatch_shadow_raycast,
    ("water", "jumpFloodLakes"): _dispatch_jump_flood_lakes,
    ("water", "steepestDescent"): _dispatch_steepest_descent,
    ("water", "steepestDescentFlow"): _dispatch_flow_network,
    ("water", "manningFlowCalculation"): _dispatch_manning_flow,
    ("water", "streamPowerErosion"): _dispatch_stream_power_erosion,
    ("water", "soilMoistureGaussian"): _dispatch_soil_moisture,
    ("water", "sedimentTransport"): _dispatch_sediment_transport,
    ("water", "atmosphericEvaporation"): _dispatch_atmospheric_evaporation,
    ("weather", "temperatureCalculation"): _dispatch_temperature_calculation,
    ("weather", "windFieldCFD"): _dispatch_wind_field_cfd,
    ("weather", "atmosphericMoisture"): _dispatch_atmospheric_moisture,
    ("weather", "precipitationCalculation"): _dispatch_precipitation_calculation,
    ("biome", "climateClassification"): _dispatch_climate_classification,
    ("biome", "supersampling"): _dispatch_supersampling,
    ("biome", "oceanConnectivity"): _dispatch_ocean_connectivity,
    ("settlement", "terrainCostFlood"): _dispatch_terrain_cost_flood,
}


class GPUWorker(threading.Thread):
    """
    Funktionsweise: Eigener Hintergrund-Thread mit eigenem Offscreen-OpenGL-
        Kontext (GL 4.3 Core), unabhängig vom GUI-Thread und von MapDisplay3D.
    Aufgabe: Verarbeitet eine Queue von Compute-Requests, die von beliebigen
        CalculatorThreads aus blockierend eingereicht werden (submit()) - löst
        das GL-Kontext/Threading-Problem, das GPU-Compute bisher komplett
        verhindert hat (siehe Modul-Docstring).
    """

    def __init__(self):
        super().__init__(name="GPUWorker", daemon=True)
        self.logger = logging.getLogger(__name__)
        self._requests = queue.Queue()
        self._ready = threading.Event()
        self._stop_requested = False
        self.gpu_available = False
        self._programs = {}

    def start_and_wait_ready(self, timeout: float = 10.0) -> bool:
        if not self.is_alive() and not self._ready.is_set():
            self.start()
        self._ready.wait(timeout)
        return self.gpu_available

    def get_program(self, category: str, operation: str):
        key = (category, operation)
        if key not in self._programs:
            path = os.path.join(SHADERS_ROOT, category, f"{operation}.comp")
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            self._programs[key] = _compile_compute_program(source, f"{category}/{operation}")
        return self._programs[key]

    def run(self):
        try:
            surface_format = QSurfaceFormat()
            surface_format.setVersion(4, 3)
            surface_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)

            self._surface = QOffscreenSurface()
            self._surface.setFormat(surface_format)
            self._surface.create()

            self._context = QOpenGLContext()
            self._context.setFormat(surface_format)
            if not self._context.create() or not self._context.makeCurrent(self._surface):
                self.logger.warning("GPUWorker: Offscreen-OpenGL-Kontext konnte nicht aktiviert werden")
                return

            version_string = gl.glGetString(gl.GL_VERSION)
            if version_string:
                major, minor = map(int, version_string.decode().split()[0].split(".")[:2])
                self.gpu_available = major > 4 or (major == 4 and minor >= 3)
            if not self.gpu_available:
                self.logger.warning(f"GPUWorker: OpenGL 4.3+ benötigt, gefunden: {version_string}")
        except Exception as e:
            self.logger.warning(f"GPUWorker: GPU-Erkennung fehlgeschlagen: {e}")
        finally:
            self._ready.set()

        while not self._stop_requested:
            try:
                request = self._requests.get(timeout=0.5)
            except queue.Empty:
                continue
            if request is None:
                break
            self._handle_request(*request)

    def _handle_request(self, category, operation, inputs, parameters, done_event, result_box):
        try:
            result_box["result"] = DISPATCH_TABLE[(category, operation)](self, inputs, parameters)
        except Exception as e:
            result_box["error"] = e
        finally:
            done_event.set()

    def submit(self, category: str, operation: str, inputs: dict, parameters: dict, timeout: float = 30.0) -> dict:
        """Blockierender Aufruf von einem beliebigen anderen Thread (z.B. CalculatorThread)."""
        if (category, operation) not in DISPATCH_TABLE:
            raise ValueError(f"Kein GPU-Dispatch registriert für {category}/{operation}")
        if not self.gpu_available:
            raise RuntimeError("GPU nicht verfügbar")

        done_event = threading.Event()
        result_box = {}
        self._requests.put((category, operation, inputs, parameters, done_event, result_box))

        if not done_event.wait(timeout):
            raise TimeoutError(f"GPU-Operation {category}/{operation} nach {timeout}s nicht abgeschlossen")
        if "error" in result_box:
            raise result_box["error"]
        return result_box["result"]

    def stop(self):
        self._stop_requested = True
        self._requests.put(None)


class ShaderManager(QObject):
    """
    Funktionsweise: Öffentliche Schnittstelle für GPU-beschleunigte Berechnungen
    Aufgabe: Startet den GPUWorker träge beim ersten Aufruf, bietet CPU-Fallback
        für jede Operation, sobald GPU nicht verfügbar ist oder fehlschlägt
    """

    processing_started = pyqtSignal(str)
    processing_finished = pyqtSignal(str, bool)
    gpu_fallback_triggered = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._worker = None
        self._worker_lock = threading.Lock()

    @property
    def gpu_available(self) -> bool:
        return self._ensure_worker().gpu_available

    def _ensure_worker(self) -> GPUWorker:
        """
        Muss unter Lock laufen: mehrere CalculatorThreads (z.B. terrain.noise +
        terrain.shadow als parallele Sibling-Knoten derselben Runde) können hier
        gleichzeitig ankommen. Ohne Lock könnte ein Check-then-create-Race ZWEI
        GPUWorker-Instanzen erzeugen - jede mit ihrem EIGENEN GL-Kontext, dessen
        Programm-/Textur-IDs im jeweils ANDEREN Kontext ungültig sind (führte zu
        GL_INVALID_OPERATION auf glUniform1i, wenn Aufrufe sich zwischen den zwei
        Kontexten vermischten).
        """
        if self._worker is None:
            with self._worker_lock:
                if self._worker is None:
                    worker = GPUWorker()
                    if not worker.start_and_wait_ready():
                        self.gpu_fallback_triggered.emit("GPUWorker: kein GL 4.3+ Kontext verfügbar")
                    self._worker = worker
        return self._worker

    def request_shader_operation(self, category: str, operation: str, inputs: dict, parameters: dict) -> dict:
        """
        Funktionsweise: Generischer Einstiegspunkt für alle GPU-Compute-Operationen
        Aufgabe: Lädt/kompiliert shaders/<category>/<operation>.comp bei Bedarf,
            dispatcht blockierend über den GPUWorker
        Parameter: category/operation - z.B. ("water", "jumpFloodLakes"); inputs -
            Daten-Dict (Arrays/Skalare) für den Shader; parameters - Ziel-Parameter
            des aufrufenden Generators (aktuell nur durchgereicht)
        Returns: dict mit mindestens "success"; bei Fehler {"success": False}
        """
        worker = self._ensure_worker()
        if not worker.gpu_available:
            return {"success": False, "reason": "gpu_unavailable"}

        operation_name = f"{category}.{operation}"
        self.processing_started.emit(operation_name)
        try:
            result = worker.submit(category, operation, inputs, parameters)
            self.processing_finished.emit(operation_name, True)
            return result
        except Exception as e:
            self.logger.warning(f"GPU-Operation {operation_name} fehlgeschlagen: {e}")
            self.processing_finished.emit(operation_name, False)
            return {"success": False, "reason": str(e)}

    def process_noise_generation(self, size, octaves, frequency, persistence, lacunarity, seed):
        """GPU-beschleunigte Multi-Octave Noise-Generierung mit CPU-Fallback."""
        result = self.request_shader_operation(
            "terrain", "noiseGeneration",
            {"size": size, "octaves": octaves, "frequency": frequency,
             "persistence": persistence, "lacunarity": lacunarity, "seed": seed},
            {}
        )
        if result.get("success"):
            return result["noise"]
        return self._cpu_fallback_noise(size, octaves, frequency, persistence, lacunarity, seed)

    def process_shadow_raycast(self, heightmap, sun_elevation, sun_azimuth, shadowmap_size=64):
        """GPU-beschleunigte Shadow-Raycast-Berechnung mit CPU-Fallback."""
        result = self.request_shader_operation(
            "terrain", "shadowRaycast",
            {"heightmap": heightmap, "sun_elevation": sun_elevation,
             "sun_azimuth": sun_azimuth, "shadowmap_size": shadowmap_size},
            {}
        )
        if result.get("success"):
            return result["shadowmap"]
        return self._cpu_fallback_shadow_raycast(heightmap, sun_elevation, sun_azimuth, shadowmap_size)

    def _cpu_fallback_noise(self, size, octaves, frequency, persistence, lacunarity, seed):
        """
        CPU-Fallback für Noise-Generierung wenn GPU nicht verfügbar oder die
        GPU-Dispatch fehlschlägt. Nutzt denselben OpenSimplex-Algorithmus und
        dieselbe Normalisierung wie core/terrain_generator.py's
        SimplexNoiseGenerator._generate_cpu_optimized() - vorher stand hier ein
        sin(x)*cos(y)-Platzhalter ("echte Simplex-Implementierung würde hier
        stehen"), der bei jedem GPU-Dispatch-Fehler silent ein fast-flaches
        Höhenfeld zurückgab, statt eine Exception zu werfen.
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

    def _cpu_fallback_shadow_raycast(self, heightmap, sun_elevation, sun_azimuth,
                                     shadowmap_size, max_distance=100.0, step_size=1.0, height_scale=1.0):
        """CPU-Fallback für Shadow-Raycast-Berechnung."""
        heightmap_size = heightmap.shape[0]
        shadowmap = np.ones((shadowmap_size, shadowmap_size), dtype=np.float32)

        elev_rad = np.radians(sun_elevation)
        azim_rad = np.radians(sun_azimuth)

        sun_dir_x = np.cos(elev_rad) * np.sin(azim_rad)
        sun_dir_y = np.cos(elev_rad) * np.cos(azim_rad)
        sun_dir_z = np.sin(elev_rad)

        for sy in range(shadowmap_size):
            for sx in range(shadowmap_size):
                start_uv_x = sx / (shadowmap_size - 1)
                start_uv_y = sy / (shadowmap_size - 1)

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

                    if current_uv_x < 0 or current_uv_x > 1 or current_uv_y < 0 or current_uv_y > 1:
                        break

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

                    if current_height <= terrain_height:
                        shadowmap[sy, sx] = 0.0
                        break

                    distance += step_size

        return shadowmap

    # ------------------------------------------------------------------
    # Weather-Bridge-Methoden
    #
    # weather_generator.py ruft (anders als water_generator.py) NICHT die
    # generische request_shader_operation() direkt auf, sondern 4 spezifische
    # Methoden mit einem verschachtelten Request-Dict
    # ({'operation_type', 'input_data', 'parameters', 'lod_level'}) und
    # erwartet ein Ergebnis-OBJEKT mit Attributzugriff (result.success,
    # result.error, result.<feld>), nicht ein dict. Diese 4 Methoden sind
    # dünne Adapter auf request_shader_operation() - der bestehende
    # Call-Site-Vertrag in weather_generator.py bleibt dadurch unverändert.
    # ------------------------------------------------------------------

    def _bridge_weather_request(self, shader_request: dict, operation: str, output_key: str, result_attr: str):
        inputs = dict(shader_request.get("input_data", {}))
        inputs.update(shader_request.get("parameters", {}))
        if "lod_level" in shader_request:
            inputs["lod_level"] = shader_request["lod_level"]
        result = self.request_shader_operation("weather", operation, inputs, {})
        if result.get("success"):
            return SimpleNamespace(success=True, error=None, **{result_attr: result[output_key]})
        return SimpleNamespace(success=False, error=result.get("reason", "unknown"))

    def request_temperature_calculation(self, shader_request: dict):
        return self._bridge_weather_request(
            shader_request, "temperatureCalculation", "temperature_field", "temperature_field")

    def request_wind_field_cfd(self, shader_request: dict):
        return self._bridge_weather_request(shader_request, "windFieldCFD", "wind_field", "wind_field")

    def request_atmospheric_moisture(self, shader_request: dict):
        return self._bridge_weather_request(
            shader_request, "atmosphericMoisture", "humidity_field", "humidity_field")

    def request_precipitation_calculation(self, shader_request: dict):
        return self._bridge_weather_request(
            shader_request, "precipitationCalculation", "precipitation_field", "precipitation_field")

    def cleanup(self):
        """Räumt GPU-Ressourcen bei Anwendungs-Beendigung auf (robust auch ohne aktiven Worker)."""
        if self._worker is not None:
            try:
                self._worker.stop()
                self._worker.join(timeout=2.0)
            except Exception as e:
                self.logger.warning(f"GPUWorker-Stop fehlgeschlagen: {e}")
            self._worker = None
