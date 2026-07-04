"""
Path: gui/managers/export_manager.py

Funktionsweise: Memory-effiziente Export-Funktionalität für große Dateien mit Streaming und Chunk-Processing
Aufgabe: Streaming-Exports, Multi-Format-Support, Memory-optimierte Verarbeitung großer Arrays
Features: Chunk-based Processing, Multiple Formats, Progress-Tracking, Memory-Management
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QDateTime, QThread, QMutex


@dataclass
class ExportJob:
    """Export-Job für Queue-Management"""
    job_id: str
    export_type: str
    data: Any
    output_path: str
    format: str
    options: Dict[str, Any]
    priority: int = 5
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class StreamingExportManager(QObject):
    """
    Funktionsweise: Memory-effiziente Export-Funktionalität mit Streaming
    Aufgabe: Große Arrays chunk-wise exportieren ohne Memory-Explosion
    Features: Streaming-IO, Chunk-Processing, Progress-Tracking, Format-Support
    """

    # Signals für Export-Progress
    export_started = pyqtSignal(str, str)        # (job_id, format)
    export_progress = pyqtSignal(str, int)       # (job_id, percent)
    export_completed = pyqtSignal(str, bool)     # (job_id, success)
    export_error = pyqtSignal(str, str)          # (job_id, error_message)

    def __init__(self, chunk_size_mb: int = 10):
        super().__init__()

        self.chunk_size = chunk_size_mb * 1024 * 1024  # Convert to bytes
        self.export_formats = ["json", "png", "obj", "csv", "numpy", "tiff"]
        self.active_exports = {}    # {job_id: export_info}
        self.export_queue = []      # List[ExportJob]

        self.logger = logging.getLogger(__name__)

    def export_large_array_json(self, array: np.ndarray, output_file: str,
                               metadata: Optional[Dict[str, Any]] = None,
                               job_id: Optional[str] = None) -> str:
        """
        Funktionsweise: Exportiert große Arrays chunk-wise als JSON
        Parameter: array, output_file, metadata, job_id
        Return: job_id
        """
        if job_id is None:
            job_id = f"json_export_{int(time.time())}"

        self.export_started.emit(job_id, "json")

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('{\n')

                # Metadata
                if metadata:
                    f.write(f'"metadata": {json.dumps(metadata, indent=2)},\n')

                # Array-Info
                f.write(f'"shape": {list(array.shape)},\n')
                f.write(f'"dtype": "{array.dtype}",\n')
                f.write('"data": [')

                # Chunk-wise Export mit Progress-Tracking
                flat_array = array.flat
                total_elements = array.size
                elements_per_chunk = self.chunk_size // array.dtype.itemsize

                for i in range(0, total_elements, elements_per_chunk):
                    chunk_end = min(i + elements_per_chunk, total_elements)
                    chunk = [float(x) for x in flat_array[i:chunk_end]]

                    json.dump(chunk, f)
                    if chunk_end < total_elements:
                        f.write(',\n')

                    # Progress Update
                    progress = int((chunk_end / total_elements) * 100)
                    self.export_progress.emit(job_id, progress)

                f.write(']\n}')

            self.export_completed.emit(job_id, True)
            self.logger.info(f"JSON export completed: {output_file}")
            return job_id

        except Exception as e:
            error_msg = f"JSON export failed: {e}"
            self.logger.error(error_msg)
            self.export_error.emit(job_id, error_msg)
            self.export_completed.emit(job_id, False)
            return job_id

    def export_array_as_png(self, array: np.ndarray, output_file: str,
                           colormap: str = 'viridis', job_id: Optional[str] = None) -> str:
        """
        Funktionsweise: Exportiert Array als PNG mit Memory-Optimierung
        Parameter: array, output_file, colormap, job_id
        Return: job_id
        """
        if job_id is None:
            job_id = f"png_export_{int(time.time())}"

        self.export_started.emit(job_id, "png")

        try:
            import matplotlib.pyplot as plt

            # Memory-optimierte Darstellung
            fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

            if len(array.shape) == 3:  # RGB
                ax.imshow(array)
            else:  # 2D Grayscale/Colormap
                im = ax.imshow(array, cmap=colormap)
                plt.colorbar(im, ax=ax, shrink=0.8)

            ax.set_title(os.path.basename(output_file).replace('.png', ''))
            ax.axis('off')

            self.export_progress.emit(job_id, 50)

            plt.savefig(output_file, bbox_inches='tight', dpi=150,
                       facecolor='white', edgecolor='none')
            plt.close(fig)  # Memory cleanup

            self.export_progress.emit(job_id, 100)
            self.export_completed.emit(job_id, True)
            self.logger.info(f"PNG export completed: {output_file}")
            return job_id

        except Exception as e:
            error_msg = f"PNG export failed: {e}"
            self.logger.error(error_msg)
            self.export_error.emit(job_id, error_msg)
            self.export_completed.emit(job_id, False)
            return job_id

    def export_array_as_tiff(self, array: np.ndarray, output_file: str,
                            job_id: Optional[str] = None) -> str:
        """
        Funktionsweise: Exportiert Array als TIFF (verlustfrei)
        Parameter: array, output_file, job_id
        Return: job_id
        """
        if job_id is None:
            job_id = f"tiff_export_{int(time.time())}"

        self.export_started.emit(job_id, "tiff")

        try:
            from PIL import Image

            # Array für TIFF vorbereiten
            if array.dtype != np.uint8:
                # Normalisieren auf 0-255 range
                normalized = ((array - np.min(array)) / (np.max(array) - np.min(array)) * 255).astype(np.uint8)
            else:
                normalized = array

            self.export_progress.emit(job_id, 30)

            if len(normalized.shape) == 3:
                # RGB/RGBA
                if normalized.shape[2] == 3:
                    mode = 'RGB'
                elif normalized.shape[2] == 4:
                    mode = 'RGBA'
                else:
                    mode = 'L'
                    normalized = normalized[:, :, 0]
            else:
                # Grayscale
                mode = 'L'

            self.export_progress.emit(job_id, 60)

            image = Image.fromarray(normalized, mode=mode)
            image.save(output_file, format='TIFF', compression='lzw')

            self.export_progress.emit(job_id, 100)
            self.export_completed.emit(job_id, True)
            self.logger.info(f"TIFF export completed: {output_file}")
            return job_id

        except Exception as e:
            error_msg = f"TIFF export failed: {e}"
            self.logger.error(error_msg)
            self.export_error.emit(job_id, error_msg)
            self.export_completed.emit(job_id, False)
            return job_id

    def export_multi_array_collection(self, array_dict: Dict[str, np.ndarray],
                                     output_dir: str, format: str = "png",
                                     job_id: Optional[str] = None) -> str:
        """
        Funktionsweise: Exportiert Collection von Arrays in verschiedenen Formaten
        Parameter: array_dict, output_dir, format, job_id
        Return: job_id
        """
        if job_id is None:
            job_id = f"collection_export_{int(time.time())}"

        self.export_started.emit(job_id, f"collection_{format}")

        try:
            os.makedirs(output_dir, exist_ok=True)

            total_arrays = len(array_dict)
            completed_arrays = 0

            for name, array in array_dict.items():
                output_file = os.path.join(output_dir, f"{name}.{format}")

                if format == "png":
                    self.export_array_as_png(array, output_file, job_id=f"{job_id}_{name}")
                elif format == "tiff":
                    self.export_array_as_tiff(array, output_file, job_id=f"{job_id}_{name}")
                elif format == "json":
                    self.export_large_array_json(array, output_file, job_id=f"{job_id}_{name}")
                elif format == "csv" and len(array.shape) == 2:
                    np.savetxt(output_file, array, delimiter=',')
                elif format == "numpy":
                    np.save(output_file.replace(f".{format}", ".npy"), array)

                completed_arrays += 1
                progress = int((completed_arrays / total_arrays) * 100)
                self.export_progress.emit(job_id, progress)

            self.export_completed.emit(job_id, True)
            self.logger.info(f"Collection export completed: {output_dir}")
            return job_id

        except Exception as e:
            error_msg = f"Collection export failed: {e}"
            self.logger.error(error_msg)
            self.export_error.emit(job_id, error_msg)
            self.export_completed.emit(job_id, False)
            return job_id

    def export_3d_obj_mesh(self, heightmap: np.ndarray, output_file: str,
                          scale: float = 1.0, job_id: Optional[str] = None) -> str:
        """
        Funktionsweise: Exportiert Heightmap als 3D OBJ-Mesh
        Parameter: heightmap, output_file, scale, job_id
        Return: job_id
        """
        if job_id is None:
            job_id = f"obj_export_{int(time.time())}"

        self.export_started.emit(job_id, "obj")

        try:
            height, width = heightmap.shape

            with open(output_file, 'w') as f:
                f.write(f"# OBJ file generated from heightmap\n")
                f.write(f"# Size: {width}x{height}\n\n")

                # Vertices schreiben (chunk-wise für große Meshes)
                vertex_count = 0
                chunk_size = 1000  # Vertices per chunk

                for y in range(height):
                    for x in range(width):
                        z = heightmap[y, x] * scale
                        f.write(f"v {x} {z} {y}\n")
                        vertex_count += 1

                        if vertex_count % chunk_size == 0:
                            progress = int((vertex_count / (height * width)) * 50)  # 50% für Vertices
                            self.export_progress.emit(job_id, progress)

                # Faces schreiben (Triangles)
                face_count = 0
                total_faces = (height - 1) * (width - 1) * 2

                for y in range(height - 1):
                    for x in range(width - 1):
                        # Vertex-Indices (OBJ ist 1-indexed)
                        v1 = y * width + x + 1
                        v2 = y * width + (x + 1) + 1
                        v3 = (y + 1) * width + x + 1
                        v4 = (y + 1) * width + (x + 1) + 1

                        # Zwei Triangles per Quad
                        f.write(f"f {v1} {v2} {v3}\n")
                        f.write(f"f {v2} {v4} {v3}\n")

                        face_count += 2

                        if face_count % chunk_size == 0:
                            progress = 50 + int((face_count / total_faces) * 50)  # 50-100% für Faces
                            self.export_progress.emit(job_id, progress)

            self.export_progress.emit(job_id, 100)
            self.export_completed.emit(job_id, True)
            self.logger.info(f"OBJ export completed: {output_file}")
            return job_id

        except Exception as e:
            error_msg = f"OBJ export failed: {e}"
            self.logger.error(error_msg)
            self.export_error.emit(job_id, error_msg)
            self.export_completed.emit(job_id, False)
            return job_id

    def get_export_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Funktionsweise: Gibt Status eines Export-Jobs zurück
        Parameter: job_id
        Return: Status dict oder None
        """
        return self.active_exports.get(job_id)

    def cancel_export(self, job_id: str) -> bool:
        """
        Funktionsweise: Bricht Export-Job ab (falls möglich)
        Parameter: job_id
        Return: bool - Cancellation erfolgreich
        """
        if job_id in self.active_exports:
            # Set cancellation flag
            self.active_exports[job_id]["cancelled"] = True
            self.logger.info(f"Export cancelled: {job_id}")
            return True
        return False


class CompleteWorldExporter(QObject):
    """
    Funktionsweise: Exportiert komplette Welt mit allen Generator-Daten
    Aufgabe: Koordiniert Export aller Map-Daten, Parameter und Metadata
    Features: Multi-Format-Export, Batch-Processing, World-Archive-Creation
    """

class CompleteWorldExporter(QObject):
    """
    Funktionsweise: Exportiert komplette Welt mit allen Generator-Daten
    Aufgabe: Koordiniert Export aller Map-Daten, Parameter und Metadata
    Features: Multi-Format-Export, Batch-Processing, World-Archive-Creation
    """

    # Signals für World-Export
    world_export_started = pyqtSignal(str)        # (export_id)
    world_export_progress = pyqtSignal(str, int)  # (export_id, percent)
    world_export_completed = pyqtSignal(str, bool, str)  # (export_id, success, output_path)
    component_exported = pyqtSignal(str, str)     # (export_id, component_name)

    def __init__(self, streaming_manager: StreamingExportManager):
        super().__init__()

        self.streaming_manager = streaming_manager
        self.active_exports = {}  # {export_id: export_info}

        self.logger = logging.getLogger(__name__)

    def export_complete_world(self, world_data: Dict[str, Dict[str, np.ndarray]],
                             all_parameters: Dict[str, Dict[str, Any]],
                             export_options: Dict[str, Any],
                             export_id: Optional[str] = None) -> str:
        """
        Funktionsweise: Exportiert komplette Welt mit allen Daten
        Parameter: world_data, all_parameters, export_options, export_id
        Return: export_id
        """
        if export_id is None:
            export_id = f"world_export_{int(time.time())}"

        self.world_export_started.emit(export_id)

        try:
            export_dir = export_options.get("export_directory", "world_export")
            formats = export_options.get("formats", ["png", "json"])
            include_3d = export_options.get("include_3d", False)
            create_archive = export_options.get("create_archive", False)

            # Hauptverzeichnis erstellen
            os.makedirs(export_dir, exist_ok=True)

            total_components = len(world_data) + 2  # +2 für Parameter und Metadata
            completed_components = 0

            # Generator-spezifische Exports
            for generator, maps in world_data.items():
                if not maps:
                    continue

                generator_dir = os.path.join(export_dir, generator)

                # PNG Exports
                if "png" in formats:
                    png_dir = os.path.join(generator_dir, "png")
                    self.streaming_manager.export_multi_array_collection(
                        maps, png_dir, "png", f"{export_id}_{generator}_png"
                    )

                # JSON Exports
                if "json" in formats:
                    json_dir = os.path.join(generator_dir, "json")
                    self.streaming_manager.export_multi_array_collection(
                        maps, json_dir, "json", f"{export_id}_{generator}_json"
                    )

                # TIFF Exports (verlustfrei)
                if "tiff" in formats:
                    tiff_dir = os.path.join(generator_dir, "tiff")
                    self.streaming_manager.export_multi_array_collection(
                        maps, tiff_dir, "tiff", f"{export_id}_{generator}_tiff"
                    )

                # 3D OBJ Export (nur für Heightmaps)
                if include_3d and "heightmap" in maps:
                    obj_file = os.path.join(generator_dir, f"{generator}_terrain.obj")
                    self.streaming_manager.export_3d_obj_mesh(
                        maps["heightmap"], obj_file, job_id=f"{export_id}_{generator}_obj"
                    )

                completed_components += 1
                progress = int((completed_components / total_components) * 90)  # 90% für Daten
                self.world_export_progress.emit(export_id, progress)
                self.component_exported.emit(export_id, generator)

            # Parameter-Export
            param_file = os.path.join(export_dir, "complete_parameters.json")
            self._export_parameters(all_parameters, param_file, export_options)

            completed_components += 1
            progress = int((completed_components / total_components) * 95)
            self.world_export_progress.emit(export_id, progress)
            self.component_exported.emit(export_id, "parameters")

            # Metadata-Export
            metadata_file = os.path.join(export_dir, "world_metadata.json")
            self._export_world_metadata(world_data, all_parameters, export_options, metadata_file)

            completed_components += 1
            progress = int((completed_components / total_components) * 98)
            self.world_export_progress.emit(export_id, progress)
            self.component_exported.emit(export_id, "metadata")

            # Archive erstellen (optional)
            final_output = export_dir
            if create_archive:
                final_output = self._create_world_archive(export_dir, export_id)

            self.world_export_progress.emit(export_id, 100)
            self.world_export_completed.emit(export_id, True, final_output)

            self.logger.info(f"World export completed: {final_output}")
            return export_id

        except Exception as e:
            error_msg = f"World export failed: {e}"
            self.logger.error(error_msg)
            self.world_export_completed.emit(export_id, False, error_msg)
            return export_id

    def export_3d_world_with_materials(self, world_data: Dict[str, Dict[str, np.ndarray]],
                                      output_file: str, export_id: Optional[str] = None) -> str:
        """
        Funktionsweise: Exportiert 3D-Welt mit Biome-Materials und Settlement-Objects
        Parameter: world_data, output_file, export_id
        Return: export_id
        """
        if export_id is None:
            export_id = f"3d_world_export_{int(time.time())}"

        self.world_export_started.emit(export_id)

        try:
            base_name = output_file.replace('.obj', '')
            obj_file = f"{base_name}.obj"
            mtl_file = f"{base_name}.mtl"

            # Terrain-Mesh mit Heightmap
            heightmap = world_data.get("terrain", {}).get("heightmap")
            if heightmap is None:
                raise ValueError("No heightmap found for 3D export")

            self.world_export_progress.emit(export_id, 10)

            with open(obj_file, 'w') as obj_f, open(mtl_file, 'w') as mtl_f:
                obj_f.write(f"mtllib {os.path.basename(mtl_file)}\n")

                # Material-Definition für Biomes
                biome_map = world_data.get("biome", {}).get("biome_map")
                if biome_map is not None:
                    self._write_biome_materials(mtl_f, biome_map)

                self.world_export_progress.emit(export_id, 20)

                # Terrain-Vertices schreiben
                height, width = heightmap.shape
                vertex_count = 0

                for y in range(height):
                    for x in range(width):
                        z = heightmap[y, x]
                        obj_f.write(f"v {x} {z} {y}\n")
                        vertex_count += 1

                    if y % 50 == 0:  # Progress update
                        progress = 20 + int((y / height) * 40)
                        self.world_export_progress.emit(export_id, progress)

                # Biome-basierte Material-Gruppen und Faces
                if biome_map is not None:
                    self._write_biome_faces(obj_f, biome_map, width, height)
                else:
                    self._write_standard_faces(obj_f, width, height)

                self.world_export_progress.emit(export_id, 80)

                # Settlement-Objects als separate Meshes
                settlement_data = world_data.get("settlement", {})
                if "settlement_list" in settlement_data:
                    self._write_settlement_objects(obj_f, settlement_data["settlement_list"])

                # Water-Features als transparente Planes
                water_data = world_data.get("water", {})
                if "water_map" in water_data:
                    self._write_water_features(obj_f, mtl_f, water_data["water_map"])

            self.world_export_progress.emit(export_id, 100)
            self.world_export_completed.emit(export_id, True, obj_file)

            self.logger.info(f"3D world export completed: {obj_file}")
            return export_id

        except Exception as e:
            error_msg = f"3D world export failed: {e}"
            self.logger.error(error_msg)
            self.world_export_completed.emit(export_id, False, error_msg)
            return export_id

    def _export_parameters(self, all_parameters: Dict[str, Dict[str, Any]],
                          output_file: str, export_options: Dict[str, Any]):
        """Exportiert Parameter mit Metadata"""
        export_data = {
            "parameters": all_parameters,
            "metadata": {
                "export_timestamp": QDateTime.currentDateTime().toString(),
                "export_options": export_options,
                "parameter_count": sum(len(params) for params in all_parameters.values()),
                "format_version": "1.0"
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

    def _export_world_metadata(self, world_data: Dict[str, Dict[str, np.ndarray]],
                              all_parameters: Dict[str, Dict[str, Any]],
                              export_options: Dict[str, Any], output_file: str):
        """Erstellt World-Metadata-File"""
        metadata = {
            "world_info": {
                "creation_date": QDateTime.currentDateTime().toString(),
                "world_size": self._calculate_world_size(world_data),
                "generator_count": len(world_data),
                "total_maps": sum(len(maps) for maps in world_data.values())
            },
            "generators": {
                gen_name: {
                    "map_count": len(maps),
                    "map_names": list(maps.keys()),
                    "total_size_mb": sum(arr.nbytes for arr in maps.values()) / (1024*1024)
                }
                for gen_name, maps in world_data.items()
            },
            "export_info": export_options,
            "parameter_summary": {
                gen_name: len(params)
                for gen_name, params in all_parameters.items()
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _calculate_world_size(self, world_data: Dict[str, Dict[str, np.ndarray]]) -> str:
        """Berechnet Gesamt-Weltgröße"""
        total_size = 0
        for generator, maps in world_data.items():
            for name, array in maps.items():
                if isinstance(array, np.ndarray):
                    total_size += array.nbytes
        return f"{total_size / (1024*1024):.1f} MB"

    def _create_world_archive(self, export_dir: str, export_id: str) -> str:
        """Erstellt ZIP-Archive der World-Daten"""
        import zipfile

        archive_path = f"{export_dir}.zip"

        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, export_dir)
                    zipf.write(file_path, arc_name)

        return archive_path

    def _write_biome_materials(self, mtl_f, biome_map: np.ndarray):
        """Schreibt Material-Definitionen für Biomes"""
        unique_biomes = np.unique(biome_map)

        # Standard-Biome-Colors (können angepasst werden)
        biome_colors = {
            0: (0.8, 0.8, 0.8),   # Unknown - Gray
            1: (0.2, 0.8, 0.2),   # Forest - Green
            2: (0.9, 0.9, 0.3),   # Desert - Yellow
            3: (0.3, 0.3, 0.8),   # Water - Blue
            4: (0.6, 0.9, 0.6),   # Grassland - Light Green
            5: (0.4, 0.2, 0.1),   # Mountain - Brown
        }

        for biome_id in unique_biomes:
            color = biome_colors.get(int(biome_id), (0.5, 0.5, 0.5))
            mtl_f.write(f"newmtl biome_{biome_id}\n")
            mtl_f.write(f"Kd {color[0]} {color[1]} {color[2]}\n")
            mtl_f.write("Ka 0.2 0.2 0.2\n")
            mtl_f.write("Ks 0.1 0.1 0.1\n")
            mtl_f.write("Ns 10\n\n")

    def _write_biome_faces(self, obj_f, biome_map: np.ndarray, width: int, height: int):
        """Schreibt Faces gruppiert nach Biomes"""
        for biome_id in np.unique(biome_map):
            obj_f.write(f"usemtl biome_{biome_id}\n")

            for y in range(height - 1):
                for x in range(width - 1):
                    if biome_map[y, x] == biome_id:
                        # Vertex-Indices (OBJ ist 1-indexed)
                        v1 = y * width + x + 1
                        v2 = y * width + (x + 1) + 1
                        v3 = (y + 1) * width + x + 1
                        v4 = (y + 1) * width + (x + 1) + 1

                        # Zwei Triangles per Quad
                        obj_f.write(f"f {v1} {v2} {v3}\n")
                        obj_f.write(f"f {v2} {v4} {v3}\n")

    def _write_standard_faces(self, obj_f, width: int, height: int):
        """Schreibt Standard-Faces ohne Biome-Gruppierung"""
        for y in range(height - 1):
            for x in range(width - 1):
                v1 = y * width + x + 1
                v2 = y * width + (x + 1) + 1
                v3 = (y + 1) * width + x + 1
                v4 = (y + 1) * width + (x + 1) + 1

                obj_f.write(f"f {v1} {v2} {v3}\n")
                obj_f.write(f"f {v2} {v4} {v3}\n")

    def _write_settlement_objects(self, obj_f, settlement_list: List[Dict[str, Any]]):
        """Schreibt Settlement-Objects als einfache Cubes"""
        obj_f.write("# Settlement Objects\n")

        for i, settlement in enumerate(settlement_list):
            x = settlement.get("x", 0)
            y = settlement.get("y", 0)  # height
            z = settlement.get("z", 0)
            size = settlement.get("size", 1)

            # Simple Cube für Settlement
            vertices = [
                (x, y, z), (x+size, y, z), (x+size, y+size, z), (x, y+size, z),
                (x, y, z+size), (x+size, y, z+size), (x+size, y+size, z+size), (x, y+size, z+size)
            ]

            obj_f.write(f"# Settlement {i}\n")
            for vx, vy, vz in vertices:
                obj_f.write(f"v {vx} {vy} {vz}\n")

    def _write_water_features(self, obj_f, mtl_f, water_map: np.ndarray):
        """Schreibt Water-Features als transparente Planes"""
        # Water-Material
        mtl_f.write("newmtl water\n")
        mtl_f.write("Kd 0.3 0.3 0.8\n")
        mtl_f.write("Ka 0.1 0.1 0.3\n")
        mtl_f.write("Ks 0.8 0.8 0.8\n")
        mtl_f.write("Ns 100\n")
        mtl_f.write("d 0.7\n")  # Transparency
        mtl_f.write("Tr 0.3\n\n")

        obj_f.write("usemtl water\n")
        obj_f.write("# Water Features\n")

        # Vereinfachtes Water-Mesh (nur Bereiche mit Wasser > Threshold)
        water_threshold = 0.1
        height, width = water_map.shape

        for y in range(height - 1):
            for x in range(width - 1):
                if water_map[y, x] > water_threshold:
                    # Water-Plane auf fixer Höhe
                    water_level = 0.5

                    v1 = f"v {x} {water_level} {y}"
                    v2 = f"v {x+1} {water_level} {y}"
                    v3 = f"v {x} {water_level} {y+1}"
                    v4 = f"v {x+1} {water_level} {y+1}"

                    obj_f.write(f"{v1}\n{v2}\n{v3}\n{v4}\n")
                    obj_f.write("f -4 -3 -2\nf -3 -1 -2\n")  # Relative indices


# Utility Functions für Export-Manager

def create_complete_export_system(chunk_size_mb: int = 10) -> tuple[StreamingExportManager, CompleteWorldExporter]:
    """
    Funktionsweise: Factory für komplettes Export-System
    Parameter: chunk_size_mb
    Return: (StreamingExportManager, CompleteWorldExporter)
    """
    streaming_manager = StreamingExportManager(chunk_size_mb)
    world_exporter = CompleteWorldExporter(streaming_manager)

    return streaming_manager, world_exporter


def export_world_quick_preset(world_data: Dict[str, Dict[str, np.ndarray]],
                             all_parameters: Dict[str, Dict[str, Any]],
                             output_directory: str) -> str:
    """
    Funktionsweise: Quick-Export mit Standard-Einstellungen
    Parameter: world_data, all_parameters, output_directory
    Return: export_id
    """
    streaming_manager, world_exporter = create_complete_export_system()

    export_options = {
        "export_directory": output_directory,
        "formats": ["png", "json"],
        "include_3d": True,
        "create_archive": True
    }

    return world_exporter.export_complete_world(world_data, all_parameters, export_options)