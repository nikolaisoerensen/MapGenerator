"""
Path: gui/utils/map_export.py

Funktionsweise: Exportiert alle radio-button-Layer der Generator-Tabs
(Terrain/Geology/Weather/Water/Biome/Settlement) als Bilddateien in einen
benannten Ordner - Skalarfelder als 16-bit Graustufen-PNG (z.B. für Godots
Terrain3D-Plugin als Heightmap-Import geeignet, 8-bit/BMP wäre zu grob und
würde sichtbares Terracing verursachen), kategorische Maps (Biome-Klassen-
IDs etc.) als 8-bit PNG, bereits-RGB-Layer (rock_map) unverändert. Ein
manifest.json hält Map-Seed, Map-Größe und pro Layer den echten Wertebereich
fest, der bei der 16-bit-Normalisierung verwendet wurde - ohne das wären die
normalisierten Pixelwerte nicht auf reale Einheiten (Meter, °C, ...)
zurückrechenbar.
Aufgabe: Von OverviewTab's Export-Button (LayerExportWidget) aufgerufen.
"""
import json
import logging
import os
import time

import numpy as np
from PIL import Image

from gui.config.gui_default import CanvasSettings

logger = logging.getLogger(__name__)

# Projekt-Root, unabhängig vom aktuellen Arbeitsverzeichnis (3 Ebenen über
# dieser Datei: gui/utils/map_export.py -> gui/utils -> gui -> Projekt-Root).
DEFAULT_EXPORT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "exports")

# (label/dateiname, export_kind, getter) - getter bekommt data_lod_manager und
# liefert die rohen Daten (oder None, wenn dieser Layer noch nicht generiert
# wurde). export_kind bestimmt die Bildkonvertierung:
#   "scalar"           - (H,W) float, 16-bit PNG, linear normalisiert
#   "slope_magnitude"  - (H,W,2) dx/dy-Gradient -> Grad (identisch zur
#                         TerrainTab-Anzeige), dann wie "scalar"
#   "vector_magnitude" - (H,W,2) Vektorfeld -> sqrt(a²+b²), dann wie "scalar"
#   "rgb"               - bereits (H,W,3) uint8, unverändert gespeichert
#   "categorical"       - (H,W) int/bool Klassen-IDs, 8-bit PNG ohne Normalisierung
_LAYER_SPECS = [
    ("heightmap", "scalar", lambda dlm: dlm.get_terrain_data_combined("heightmap")),
    ("slopemap", "slope_magnitude", lambda dlm: dlm.get_terrain_data("slopemap")),
    ("rock_map", "rgb", lambda dlm: dlm.get_geology_data("rock_map")),
    ("hardness_map", "scalar", lambda dlm: dlm.get_geology_data("hardness_map")),
    ("temp_map", "scalar", lambda dlm: dlm.get_weather_data("temp_map")),
    ("precip_map", "scalar", lambda dlm: dlm.get_weather_data("precip_map")),
    ("humid_map", "scalar", lambda dlm: dlm.get_weather_data("humid_map")),
    ("wind_map", "vector_magnitude", lambda dlm: dlm.get_weather_data("wind_map")),
    ("water_map", "scalar", lambda dlm: dlm.get_water_data("water_map")),
    ("flow_map", "scalar", lambda dlm: dlm.get_water_data("flow_map")),
    ("erosion_map", "scalar", lambda dlm: dlm.get_water_data("erosion_map")),
    ("sedimentation_map", "scalar", lambda dlm: dlm.get_water_data("sedimentation_map")),
    ("soil_moist_map", "scalar", lambda dlm: dlm.get_water_data("soil_moist_map")),
    ("biome_map", "categorical", lambda dlm: dlm.get_biome_data("biome_map")),
    ("biome_map_super", "categorical", lambda dlm: dlm.get_biome_data("biome_map_super")),
    ("super_biome_mask", "categorical", lambda dlm: dlm.get_biome_data("super_biome_mask")),
    ("suitability_map", "scalar", lambda dlm: dlm.get_settlement_data("combined_suitability_map")),
    ("civ_map", "scalar", lambda dlm: dlm.get_settlement_data("civ_map")),
]


def _fixed_range(layer_key):
    """Feste (vmin, vmax) aus CanvasSettings.CANVAS_2D['layer_ranges'], falls
    vorhanden - sonst None (Aufrufer normalisiert dann auf das tatsächliche
    Datenmin/-max dieser Karte). Die optionale 4. Tupel-Stelle ("log") ist ein
    reines Anzeige-/Farbskalen-Detail und wird hier bewusst ignoriert - der
    Export normalisiert immer LINEAR, damit die gespeicherten Werte die
    physikalische Größe proportional abbilden (wichtig für z.B. Terrain3D-
    Import)."""
    entry = CanvasSettings.CANVAS_2D.get("layer_ranges", {}).get(layer_key)
    if entry is None:
        return None
    return float(entry[1]), float(entry[2])


def _normalize_to_16bit(values, vmin, vmax):
    values = np.asarray(values, dtype=np.float64)
    span = vmax - vmin
    if span <= 0:
        normalized = np.zeros_like(values)
    else:
        normalized = np.clip((values - vmin) / span, 0.0, 1.0)
    return (normalized * 65535.0).round().astype(np.uint16)


def export_all_layers(data_lod_manager, parameter_manager, output_root, filename_prefix):
    """
    Funktionsweise: Exportiert jeden in _LAYER_SPECS gelisteten Layer, der
    aktuell tatsächlich generiert vorliegt, als PNG in output_root/
    filename_prefix/ - Layer, die (noch) nicht existieren oder beim Export
    einen Fehler werfen, werden übersprungen (nicht die gesamte Operation
    abgebrochen), damit auch ein teilweise generierter Kartenstand exportiert
    werden kann.
    Parameter: data_lod_manager (DataLODManager), parameter_manager
    (ParameterManager, für map_seed) - beide dürfen None sein.
    output_root (str) - übergeordnetes Export-Verzeichnis, wird bei Bedarf
    angelegt. filename_prefix (str) - Name des neu anzulegenden Unterordners.
    Return: (success: bool, message: str, output_dir: str oder None)
    """
    output_dir = os.path.join(output_root, filename_prefix)
    os.makedirs(output_dir, exist_ok=True)

    manifest = {
        "filename_prefix": filename_prefix,
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "map_seed": None,
        "layers": {},
        "skipped": [],
    }

    if parameter_manager is not None:
        try:
            manifest["map_seed"] = parameter_manager.get_tab_parameters("terrain").get("map_seed")
        except Exception as e:
            logger.debug(f"map_export: map_seed nicht lesbar: {e}")

    exported_count = 0

    for label, kind, getter in _LAYER_SPECS:
        try:
            data = getter(data_lod_manager) if data_lod_manager is not None else None
        except Exception as e:
            manifest["skipped"].append(f"{label} (getter error: {e})")
            continue

        if data is None:
            manifest["skipped"].append(f"{label} (not generated yet)")
            continue

        try:
            data = np.asarray(data)
            file_name = f"{label}.png"
            file_path = os.path.join(output_dir, file_name)

            if kind == "rgb":
                arr = np.clip(data, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(file_path)
                manifest["layers"][label] = {"file": file_name, "kind": "rgb"}

            elif kind == "categorical":
                arr = (data.astype(np.uint8) * 255) if data.dtype == bool \
                    else np.clip(data, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(file_path)
                manifest["layers"][label] = {"file": file_name, "kind": "categorical_8bit"}

            else:
                if kind == "slope_magnitude" and data.ndim == 3:
                    magnitude = np.sqrt(data[:, :, 0] ** 2 + data[:, :, 1] ** 2)
                    values = np.degrees(np.arctan(magnitude))
                elif kind == "vector_magnitude" and data.ndim == 3:
                    values = np.sqrt(data[:, :, 0] ** 2 + data[:, :, 1] ** 2)
                else:
                    values = data

                fixed = _fixed_range(label)
                if fixed is not None:
                    vmin, vmax = fixed
                else:
                    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))

                arr16 = _normalize_to_16bit(values, vmin, vmax)
                Image.fromarray(arr16).save(file_path)
                manifest["layers"][label] = {
                    "file": file_name, "kind": "scalar_16bit",
                    "value_min": vmin, "value_max": vmax,
                }

            exported_count += 1
        except Exception as e:
            logger.warning(f"map_export: Layer '{label}' konnte nicht exportiert werden: {e}")
            manifest["skipped"].append(f"{label} (export error: {e})")

    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if exported_count == 0:
        return False, "Keine Layer verfügbar - erst eine Karte generieren.", None

    message = f"{exported_count} Layer exportiert nach {output_dir}"
    if manifest["skipped"]:
        message += f" ({len(manifest['skipped'])} übersprungen)"
    return True, message, output_dir
