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

# EXPORTGROESSE FUER DIE GANZE WELT (docs/OFFENE_PUNKTE.md 13.1,
# Nutzerentscheidung).
#
# 2048 px fuer die GESAMTE Karte, nicht je Region. Bei 21.3 km Weltbreite sind
# das 10.4 m/px - halb so grob wie die 20.8 m/px bei 1024. Die Alternative
# "je Region 4096 px = 1 m/px" wurde bewusst verworfen: 30-60 Minuten Backzeit
# je Region.
#
# EHRLICH BENANNTE FOLGE: eine Spielfigur steht damit auf 10-m-Dreiecken. Das
# Feindetail muss aus dem Engine-Rauschen kommen; daraus entstehen aber nur
# Rauheit, keine echten Gelaendeformen (Rinnen, Felsabbrueche,
# Geologie-Aufschluesse). Wo das Rauschen NICHT wirken darf - unter Wegen,
# Bauflaechen, an Ufern -, regelt die Daempfungsmaske aus 13.2.
EXPORT_KANTENLAENGE_PX = 2048

# Wie ein Layer auf die Exportgroesse gebracht wird. Kategorien und Farbbilder
# duerfen NICHT interpoliert werden - zwischen Biom 3 und Biom 7 liegt kein
# Biom 5, und ein gemitteltes Kuestenpixel waere ein Biom, das es nicht gibt.
_GLAETTEND = {"scalar", "slope_magnitude", "vector_magnitude"}


def _auf_exportgroesse(werte, kind, kante=EXPORT_KANTENLAENGE_PX):
    """
    Layer auf `kante` x `kante` bringen.

    Rueckgabe (array, vermerk) - `vermerk` beschreibt, was passiert ist, und
    landet im Manifest. Ohne den waere im fertigen Export nicht mehr zu sehen,
    ob ein Layer hochgerechnet wurde oder nativ in dieser Groesse vorlag - und
    hochgerechnete Daten sehen genauso scharf aus wie echte.
    """
    if werte.ndim < 2 or werte.shape[0] == werte.shape[1] == kante:
        return werte, "nativ"

    hoehe, breite = werte.shape[:2]
    # Ganzzahlige Stuetzstellen des Zielrasters auf das Quellraster abbilden
    ys = np.linspace(0, hoehe - 1, kante)
    xs = np.linspace(0, breite - 1, kante)

    if kind in _GLAETTEND and np.issubdtype(werte.dtype, np.floating):
        # Bilinear, von Hand - scipy.ndimage.zoom waere hier eine Abhaengigkeit
        # mehr im Exportpfad, und die Rechnung ist drei Zeilen.
        y0 = np.clip(np.floor(ys).astype(int), 0, hoehe - 1)
        y1 = np.clip(y0 + 1, 0, hoehe - 1)
        x0 = np.clip(np.floor(xs).astype(int), 0, breite - 1)
        x1 = np.clip(x0 + 1, 0, breite - 1)
        fy = (ys - y0)[:, None]
        fx = (xs - x0)[None, :]
        oben = werte[np.ix_(y0, x0)] * (1 - fx) + werte[np.ix_(y0, x1)] * fx
        unten = werte[np.ix_(y1, x0)] * (1 - fx) + werte[np.ix_(y1, x1)] * fx
        neu = oben * (1 - fy) + unten * fy
        return neu, f"bilinear {breite}->{kante}"

    yi = np.clip(np.round(ys).astype(int), 0, hoehe - 1)
    xi = np.clip(np.round(xs).astype(int), 0, breite - 1)
    return werte[np.ix_(yi, xi)], f"naechster Nachbar {breite}->{kante}"


# --- Daempfungsmaske fuers Engine-Rauschen (13.2) --------------------------
#
# Addiert Godot Hoehenrauschen, verschiebt sich der Boden UNTER Wegen,
# Grundstuecken und Siedlungen: ein bei uns ebener Weg bekommt Wellen, ein
# Stadtgrundriss steht schief. Diese Maske sagt der Engine, wo sie das
# Rauschen herunterregeln soll - 0 = gar nicht rauschen, 1 = volle Wildnis.
#
# Die Radien sind in METERN angegeben, nicht in Pixeln. Sonst haengt die
# Breite des geschuetzten Streifens an der Exportaufloesung, und dieselbe
# Welt bekaeme bei 1024 und 2048 px verschieden breite Wegkorridore.
WEG_SCHUTZ_M = 45.0        # halbe Korridorbreite um einen Weg
UFER_SCHUTZ_M = 35.0       # Streifen beidseits der Wasserlinie
BAU_SCHUTZ_M = 25.0        # Rand um Bauflaechen
UEBERGANG_M = 60.0         # Breite des weichen Auslaufs nach aussen


def _maske_weichzeichnen(hart, meter_pro_pixel, uebergang_m=UEBERGANG_M):
    """
    Aus einer harten 0/1-Maske eine weiche machen: innen 0, nach aussen
    linear auf 1.

    Ohne den Uebergang saehe man im Spiel eine scharfe Kante, an der das
    Rauschen abrupt einsetzt - genau die Art Artefakt, die eine Maske
    verhindern soll.
    """
    from scipy import ndimage
    if not hart.any():
        return np.ones(hart.shape, dtype=np.float32)
    abstand_px = ndimage.distance_transform_edt(~hart)
    abstand_m = abstand_px * float(meter_pro_pixel)
    return np.clip(abstand_m / max(uebergang_m, 1e-6), 0.0, 1.0).astype(np.float32)


def daempfungsmaske(data_lod_manager, meter_pro_pixel):
    """
    Graustufenmaske fuer die Rauschamplitude in der Engine (13.2).

    Rueckgabe (H,W) float32 in [0,1] oder None, wenn nicht einmal eine
    Heightmap vorliegt. Was fehlt, wird ausgelassen - eine Karte ohne Wege
    bekommt eben keinen Wegschutz, das ist kein Fehler.
    """
    dlm = data_lod_manager
    if dlm is None:
        return None
    try:
        hoehe = dlm.get_terrain_data_combined("heightmap")
    except Exception:                                   # noqa: BLE001
        hoehe = None
    if hoehe is None:
        return None

    hoehe = np.asarray(hoehe)
    form = hoehe.shape[:2]
    hart = np.zeros(form, dtype=bool)

    def _hole(bereich, schluessel):
        try:
            return getattr(dlm, f"get_{bereich}_data")(schluessel)
        except Exception:                               # noqa: BLE001
            return None

    # Bauflaechen und Strassenraster der Staedte
    for schluessel in ("city_mask", "street_mask", "house_parcel_map"):
        feld = _hole("settlement", schluessel)
        if feld is not None:
            feld = np.asarray(feld)
            if feld.shape[:2] == form:
                hart |= feld.astype(bool)

    # Ufer: der Streifen um die Wasserlinie
    ufer_px = max(int(round(UFER_SCHUTZ_M / max(meter_pro_pixel, 1e-6))), 1)
    if ufer_px > 0:
        from scipy import ndimage
        land = hoehe > 0
        rand = land ^ ndimage.binary_erosion(land, iterations=ufer_px)
        rand |= (~land) ^ ndimage.binary_erosion(~land, iterations=ufer_px)
        hart |= rand

    # Wege als Linienzuege rastern
    weg_px = max(int(round(WEG_SCHUTZ_M / max(meter_pro_pixel, 1e-6))), 1)
    linien = []
    for schluessel in ("roads", "sea_roads"):
        wege = _hole("settlement", schluessel)
        if wege:
            linien.extend(wege)
    if linien:
        spur = np.zeros(form, dtype=bool)
        for weg in linien:
            for punkt in weg:
                x = int(np.clip(round(float(punkt[0])), 0, form[1] - 1))
                y = int(np.clip(round(float(punkt[1])), 0, form[0] - 1))
                spur[y, x] = True
        if weg_px > 1:
            from scipy import ndimage
            spur = ndimage.binary_dilation(spur, iterations=weg_px)
        hart |= spur

    return _maske_weichzeichnen(hart, meter_pro_pixel)


# --- Vektordaten fuer die Engine (docs/OFFENE_PUNKTE.md 13.5) --------------
#
# Wege, Grundstuecksgrenzen und Ortslagen liegen intern bereits als
# Linienzuege bzw. Punkte vor. Als Raster exportiert waeren sie in Godot
# wertlos - dort sollen daraus Splines werden (Wege als Decal oder Mesh,
# Grenzen als Zaun, Orte als Platzierungspunkt). Deshalb ein eigenes JSON.
#
# KOORDINATEN IN METERN, nicht in Pixeln. Der Pixelwert haengt an der
# Kartengroesse; Meter haengen an der Welt. Sonst muesste die Engine den
# Massstab kennen und selbst umrechnen - und genau da entstehen die Fehler,
# die niemand bemerkt, weil die Wege trotzdem irgendwo liegen.
VEKTOR_DATEI = "vektor.json"


def _pfad_in_meter(pfad, meter_pro_pixel, nachkomma=2):
    """Linienzug aus Pixelkoordinaten in Meter, gerundet."""
    punkte = []
    for punkt in pfad:
        try:
            x, y = float(punkt[0]), float(punkt[1])
        except (TypeError, IndexError, ValueError):
            continue
        punkte.append([round(x * meter_pro_pixel, nachkomma),
                       round(y * meter_pro_pixel, nachkomma)])
    return punkte


def vektordaten(data_lod_manager, meter_pro_pixel):
    """
    Wege, Seewege, Grundstuecksgrenzen und Ortslagen als JSON-taugliches dict.

    Was fehlt, wird ausgelassen und unter "fehlt" begruendet - eine Karte
    ohne Siedlungen bekommt eben keine Ortsliste. Ein stilles Weglassen waere
    hier besonders teuer, weil im fertigen Export niemand sieht, ob eine
    leere Liste "gibt es nicht" oder "ging schief" bedeutet.
    """
    dlm = data_lod_manager
    daten = {"einheit": "meter", "meter_pro_pixel": round(meter_pro_pixel, 4),
             "wege": [], "seewege": [], "grundstuecksgrenzen": [],
             "orte": [], "fehlt": []}
    if dlm is None:
        daten["fehlt"].append("kein DataLODManager uebergeben")
        return daten

    def _hole(bereich, schluessel):
        try:
            return getattr(dlm, f"get_{bereich}_data")(schluessel)
        except Exception as e:                          # noqa: BLE001
            daten["fehlt"].append(f"{schluessel} ({e})")
            return None

    for ziel, schluessel in (("wege", "roads"), ("seewege", "sea_roads"),
                             ("grundstuecksgrenzen", "plot_edges")):
        linien = _hole("settlement", schluessel)
        if not linien:
            daten["fehlt"].append(f"{schluessel} (nicht generiert)")
            continue
        for linie in linien:
            punkte = _pfad_in_meter(linie, meter_pro_pixel)
            if len(punkte) >= 2:
                daten[ziel].append(punkte)

    for schluessel, art in (("settlement_list", "siedlung"),
                            ("landmark_list", "landmarke"),
                            ("roadsite_list", "wegstation")):
        orte = _hole("settlement", schluessel)
        if not orte:
            daten["fehlt"].append(f"{schluessel} (nicht generiert)")
            continue
        for ort in orte:
            x = getattr(ort, "x", None)
            y = getattr(ort, "y", None)
            if x is None or y is None:
                continue
            daten["orte"].append({
                "art": art,
                "x": round(float(x) * meter_pro_pixel, 2),
                "y": round(float(y) * meter_pro_pixel, 2),
                "typ": getattr(ort, "settlement_type", "") or "",
                "rang": getattr(ort, "rank", "") or "",
                "kultur": getattr(ort, "culture", "") or "",
                "haeuser": int(getattr(ort, "house_count", 0) or 0),
                "radius_m": round(float(getattr(ort, "radius", 0.0) or 0.0)
                                  * meter_pro_pixel, 2),
            })

    # FLUESSE FEHLEN, UND ZWAR AUS EINEM BESTIMMTEN GRUND.
    #
    # `core/terrain_weltfluesse.flussnetz()` baut sehr wohl einen Knotengraphen
    # (`punkte`, `eltern`, `reihenfolge`, Strahler-Ordnung) - aber
    # `core/terrain_generator.py` behaelt daraus nur die RASTER `river_mask`
    # und `river_order` (dort Zeile 1707) und wirft den Graphen weg. Aus einem
    # Raster wieder Linienzuege zu machen (Skelettieren, Graph verfolgen,
    # Ausduennen) waere Arbeit mit eigenen Fehlerquellen fuer etwas, das
    # vorher schon vorlag.
    #
    # Der richtige Weg ist, den Graphen aufzuheben. Das ist ein Eingriff in
    # die Ausgaben des Terrain-Generators und gehoert nicht in den Exporteur.
    daten["fehlt"].append(
        "fluesse (Knotengraph wird in core/terrain_generator.py:1707 "
        "verworfen, es bleiben nur river_mask/river_order als Raster)")
    return daten


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
        # Der Massstab gehoert MIT in den Export. Ohne ihn ist ein 2048er PNG
        # nur eine Zahl - die Engine kann daraus nicht ableiten, wie gross ein
        # Pixel in der Welt ist (13.1).
        "export_kantenlaenge_px": EXPORT_KANTENLAENGE_PX,
        "welt_km": None,
        "meter_pro_pixel": None,
        "layers": {},
        "skipped": [],
    }

    if parameter_manager is not None:
        try:
            _terrain = parameter_manager.get_tab_parameters("terrain")
            manifest["map_seed"] = _terrain.get("map_seed")
            _km = _terrain.get("map_distance_km")
            if _km:
                manifest["welt_km"] = float(_km)
                manifest["meter_pro_pixel"] = round(
                    float(_km) * 1000.0 / EXPORT_KANTENLAENGE_PX, 4)
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
            data, groessen_vermerk = _auf_exportgroesse(data, kind)
            file_name = f"{label}.png"
            file_path = os.path.join(output_dir, file_name)

            if kind == "rgb":
                arr = np.clip(data, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(file_path)
                manifest["layers"][label] = {"file": file_name, "kind": "rgb",
                                             "groesse": groessen_vermerk}

            elif kind == "categorical":
                arr = (data.astype(np.uint8) * 255) if data.dtype == bool \
                    else np.clip(data, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(file_path)
                manifest["layers"][label] = {"file": file_name, "kind": "categorical_8bit",
                                             "groesse": groessen_vermerk}

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
                    "groesse": groessen_vermerk,
                }

            exported_count += 1
        except Exception as e:
            logger.warning(f"map_export: Layer '{label}' konnte nicht exportiert werden: {e}")
            manifest["skipped"].append(f"{label} (export error: {e})")

    # DAEMPFUNGSMASKE (docs/OFFENE_PUNKTE.md 13.2) - eigener Zweig, weil sie
    # nicht aus einem einzelnen Layer kommt, sondern aus mehreren
    # zusammengesetzt wird (Bauflaechen, Strassenraster, Wege, Ufer).
    try:
        mpp = manifest.get("meter_pro_pixel")
        if mpp is None:
            # Ohne map_distance_km im Parametersatz aus der Vorgabe rechnen,
            # damit die Maske nicht ganz ausfaellt - und das im Manifest
            # vermerken, sonst sieht ein geschaetzter Massstab aus wie ein
            # gemessener.
            mpp = 21300.0 / EXPORT_KANTENLAENGE_PX
            manifest["meter_pro_pixel_geschaetzt"] = round(mpp, 4)
        maske = daempfungsmaske(data_lod_manager, mpp)
        if maske is not None:
            maske, groessen_vermerk = _auf_exportgroesse(maske, "scalar")
            datei = "noise_damping_mask.png"
            Image.fromarray(_normalize_to_16bit(maske, 0.0, 1.0)).save(
                os.path.join(output_dir, datei))
            manifest["layers"]["noise_damping_mask"] = {
                "file": datei, "kind": "scalar_16bit",
                "value_min": 0.0, "value_max": 1.0,
                "groesse": groessen_vermerk,
                "bedeutung": "Faktor fuer die Rauschamplitude der Engine: "
                             "0 = nicht rauschen (Wege, Bauflaechen, Ufer), "
                             "1 = volle Wildnis.",
            }
            exported_count += 1
        else:
            manifest["skipped"].append("noise_damping_mask (keine Heightmap)")
    except Exception as e:                              # noqa: BLE001
        logger.warning(f"map_export: Daempfungsmaske fehlgeschlagen: {e}")
        manifest["skipped"].append(f"noise_damping_mask (error: {e})")

    # VEKTORDATEN (13.5) - eigene Datei, kein PNG.
    try:
        mpp_v = manifest.get("meter_pro_pixel") or manifest.get(
            "meter_pro_pixel_geschaetzt") or (21300.0 / EXPORT_KANTENLAENGE_PX)
        vektor = vektordaten(data_lod_manager, mpp_v)
        with open(os.path.join(output_dir, VEKTOR_DATEI), "w",
                  encoding="utf-8") as vf:
            json.dump(vektor, vf, indent=2, ensure_ascii=False)
        manifest["vektor"] = {
            "file": VEKTOR_DATEI,
            "wege": len(vektor["wege"]),
            "seewege": len(vektor["seewege"]),
            "grundstuecksgrenzen": len(vektor["grundstuecksgrenzen"]),
            "orte": len(vektor["orte"]),
            "fehlt": vektor["fehlt"],
        }
    except Exception as e:                              # noqa: BLE001
        logger.warning(f"map_export: Vektordaten fehlgeschlagen: {e}")
        manifest["skipped"].append(f"{VEKTOR_DATEI} (error: {e})")

    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if exported_count == 0:
        return False, "Keine Layer verfügbar - erst eine Karte generieren.", None

    message = f"{exported_count} Layer exportiert nach {output_dir}"
    if manifest["skipped"]:
        message += f" ({len(manifest['skipped'])} übersprungen)"
    return True, message, output_dir
