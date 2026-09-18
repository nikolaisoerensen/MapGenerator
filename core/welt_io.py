"""
Path: core/welt_io.py

DIE NAHT (Ticket #38): welt_backen() / welt_laden()
====================================================

Bis zu diesem Ticket konnte der Map Editor eine Welt generieren, aber nicht
speichern - eine generierte Welt war beim Schliessen des Programms verloren.
Dieses Modul ist die EINZIGE Stelle, die eine Welt auf Platte schreibt oder
von dort zurückliest. Kein anderer Code darf `pickle`/`open(...)` direkt auf
einen Welt-Ordner anwenden - jede neue Stelle, die "auch noch schnell etwas
sichert", ist genau die Art von zweitem Pfad, die früher zu lautlos
verschwundenen Daten geführt hat (siehe Ticket #54 und die Beispiele dazu in
CLAUDE.md, Abschnitt "Gruene Tests koennen eine tote Funktion verdecken").

WAS "EINE WELT" IST (das vollständige Feld, das gesichert/geladen wird)
------------------------------------------------------------------------

1. Die sieben Generator-Kategorien, wie sie DataLODManager selbst führt
   (siehe dort `_KATEGORIE_SPEICHER` sowie `get_all_data()`/`set_all_data()`,
   die dieses Ticket dort ergänzt hat):

   - terrain    : heightmap, slopemap, shadowmap, optional (Weltkarten-Modus)
                  river_mask, river_order, river_generation, river_water,
                  hinterland_height, voronoi_map, region_map, klima_map,
                  seegrad, ... (siehe set_terrain_data_complete_lod());
                  ausserdem "terrain_data_object" (das komplette TerrainData-
                  Objekt, redundant zu den Einzel-Arrays, aber Teil des
                  bestehenden Speichers und daher Teil des Snapshots)
   - geology    : rock_map, hardness_map, height_delta, layer_id_map,
                  fault_distance_map, intrusion_distance_map,
                  metamorphic_grade_map, layer_boundaries (3D-Array),
                  delta_components (dict von Arrays)
   - weather    : wind_map, temp_map, precip_map, humid_map, sowie die
                  Layer-Varianten (*_map_layers) und die sechs monatlichen
                  Listen (*_map_monthly, *_map_layers_monthly)
   - erosion    : siehe ErosionSystemGenerator.EROSION_DATA_KEYS
                  (erosion_map, sedimentation_map, thermal_erosion_map,
                  thermal_deposition_map, sediment_load_map, water_depth_map,
                  flow_velocity_map) plus EROSION_SCALAR_KEYS (steps_taken,
                  converged, mass_balance, simulation_resolution)
   - water      : siehe HydrologySystemGenerator.WATER_DATA_KEYS (water_map,
                  flow_map, flow_speed, cross_section, soil_moist_map,
                  evaporation_map, ocean_outflow, water_biomes_map)
   - biome      : biome_map, biome_map_super, super_biome_mask,
                  climate_classification, biome_statistics (dict)
   - settlement : plot_map, civ_map, combined_suitability_map, city_mask,
                  voronoi_cell_map, street_mask, house_parcel_map,
                  potential_field, sowie die Nicht-Array-Listen
                  settlement_list, landmark_list, roadsite_list, roads,
                  sea_roads, plots, plot_nodes, landmark_roads, plot_edges,
                  plot_cores, wilderness_polygons, plot_node_positions

   Diese Liste hier ist NUR zur menschlichen Orientierung. Der tatsächliche
   Mechanismus fragt NICHTS davon namentlich ab - er nimmt, was
   `DataLODManager.get_all_data(kategorie)` beim jeweils aktuellen LOD
   tatsächlich liefert. Kommt ein Generator morgen um ein neues Feld reicher,
   landet es automatisch im nächsten `welt_backen()`, ohne dass dieses Modul
   angefasst werden muss - das war der Sinn der Ergänzung von `get_all_data()`
   in `managers/data_lod_manager.py`.

2. Drei globale Werte, die zu keiner Kategorie gehören:
   `map_seed`, `map_distance_km`, `map_latitude` (siehe
   `DataLODManager.get_map_seed()` etc.) - ohne sie ist eine geladene Welt
   nicht deterministisch reproduzierbar.

3. OPTIONAL, nur wenn ein `parameter_manager` übergeben wird: die aktuellen
   GUI-Regler-Werte aller registrierten Tabs (`ParameterManager.
   get_all_parameters()`). Das ist bewusst NICHT Teil der lauten
   Fehlerprüfung unten - ein Tab kann beim Laden noch gar nicht geöffnet
   bzw. registriert sein (z.B. Laden vor dem ersten Tab-Wechsel), das ist ein
   normaler Zustand und kein Datenverlust. Fehlt ein Tab bei
   `set_tab_parameters()`, wird das geloggt, aber welt_laden() bricht deshalb
   NICHT ab. Fehlt dagegen die komplette `parameter.json`, obwohl das
   Manifest sie verspricht, ist DAS ein lauter Fehler (Datei fehlt, kein
   "als ob nichts gesichert wäre").

DATEIFORMAT - Entscheidung und OFFENE FRAGE für die Nutzer-Freigabe morgen
---------------------------------------------------------------------------

Ein Welt-Ordner (Parameter `pfad`) enthält nach `welt_backen()` DREI
getrennte Unterordner mit unterschiedlichem Zweck - das ist die zentrale
Entwurfsentscheidung dieses Moduls:

    <pfad>/
        welt_manifest.json      <- Inhaltsverzeichnis, siehe unten
        zustand/                <- verlustfrei, NUR für welt_laden() gedacht
            terrain.pkl ...
            globals.json
            parameter.json       (optional)
        godot/                  <- verlustbehaftet, für die Godot/Terrain3D-
                                    Engine gedacht, wird von welt_laden()
                                    NICHT zurückgelesen (Einbahnstrasse)
            godot/manifest.json, *.png, vektor.json, ...
        vorschau/               <- schnelle PNG-Vorschau + Textstatistik für
                                    Menschen, ebenfalls nicht zurückgelesen

Warum drei getrennte Ordner statt eines gemeinsamen Formats:

Die Ticket-Vorgabe verlangt ZWEI Dinge, die sich gegenseitig ausschliessen,
wenn man sie in einem Format lösen will:
  (a) "bitgenauer" Rundlauf (bake -> load -> exakt dasselbe Ergebnis)
  (b) das Format soll in erster Linie das sein, was Godot/Terrain3D braucht

Der bestehende, bereits getestete Godot-Export (`gui/utils/map_export.py`,
`export_all_layers()`, siehe `tests/smoke_test_export_2048.py`) ist mit
Absicht verlustbehaftet: er skaliert jede Karte auf eine feste Kante von
2048 px (Interpolation bei Skalar-Layern) und quantisiert auf 16-Bit-PNG
(`_normalize_to_16bit`). Das ist für eine Spiel-Engine richtig - sie braucht
eine feste Texturgrösse, kein Float64. Für den geforderten bitgenauen
Rundlauf ist es aber UNGEEIGNET, und ich wollte diesen bereits getesteten
Pfad nicht anfassen, nur um ihn verlustfrei zu machen, ohne den Godot-
Entwickler dazu befragen zu können.

Deshalb: `zustand/` ist ein bewusst einfaches, verlustfreies internes
Backup (ein Pickle je Kategorie, mit `numpy`-Arrays in ihrem echten dtype -
das prüft `tests/smoke_test_welt_io_roundtrip.py` empirisch nach), das
NUR von `welt_laden()` gelesen wird. `godot/` ist der unveränderte,
bestehende Export - eine Kopie/ein Aufruf des bereits vorhandenen Codes,
keine Neuentwicklung. `welt_laden()` liest `godot/` nicht zurück; es ist
ein einseitiger Ausgabe-Pfad für die Engine, kein Teil des Rundlaufs.

OFFENE FRAGE FÜR MORGEN (Godot-Entwickler/Nutzer-Freigabe):

Dieses Pickle-Format für `zustand/` ist eine PROVISORISCHE, reine
Verlegenheitslösung für die Nacht - kein für die Ewigkeit gedachtes
Speicherformat. Zwei konkret geprüfte Alternativen für später:

  - Ich habe empirisch verifiziert (Scratch-Skript, PIL 12.3.0), dass
    TIFF im Modus 'F' (32-Bit-Float-Graustufen) `float32`-Arrays bitgenau
    rundlauffähig serialisiert - falls Godot/Terrain3D lieber TIFF als
    PNG einliest, wäre DAS vermutlich das bessere gemeinsame Format für
    `zustand/` UND für einen künftig verlustfreien Godot-Heightmap-Export,
    statt der aktuellen 16-Bit-PNG-Quantisierung.
  - Pickle selbst ist nicht menschenlesbar, nicht sprachübergreifend
    (Godot/GDScript kann es nicht einlesen) und nicht versionsstabil über
    grosse Python-Versionssprünge hinweg. Für ein reines Werkzeug-internes
    Backup (Speichern/Laden IM SELBEN Programm) ist das unproblematisch;
    als Austauschformat zu einer anderen Engine wäre es das nicht.

Ich habe diese Entscheidung NICHT weiter in Richtung eines Godot-
spezifischen Formats optimiert, weil das laut Ticket-Vorgabe explizit erst
nach Rücksprache mit dem Godot-Entwickler passieren soll. Der wichtige Punkt
heute Nacht ist, dass die Naht existiert, vollständig ist und der Rundlauf
bitgenau funktioniert - nicht, dass das Format schon das externe Endformat
ist.

DAS MANIFEST (`welt_manifest.json`)
------------------------------------

Ist das Inhaltsverzeichnis, das `welt_laden()` gegen die tatsächlich
vorhandenen Dateien prüft, BEVOR und NACHDEM es etwas zurückschreibt. Enthält
je Kategorie, ob sie vorhanden war und welche Keys sie enthielt (zur
Prüfung "kam alles wieder an, was gesichert wurde" - kein stiller
Ersatzpfad, siehe Ticket #54: sieben frühere Vorfälle in genau diesem
Projekt, bei denen ein fehlendes Feld lautlos durch einen Platzhalter
ersetzt wurde, statt einen Fehler zu werfen).

WIEDERVERWENDUNG VON gui/tabs/overview_tab.py (Ticket-Vorgabe: wiederverwenden,
nicht duplizieren)
------------------------------------------------------------------------------

Die drei im Ticket namentlich genannten Funktionen (`export_single_map_png`,
`export_material_file`, `export_world_statistics_txt`) waren zum
Bearbeitungszeitpunkt noch unveränderte `OverviewTab`-Methoden, die intern
`self` nie benutzten. Sie wurden nach `gui/utils/map_export.py` verschoben
(dort bereits Qt-frei, bereits der Ort von `export_all_layers`), und
`OverviewTab` behält dünne Wrapper-Methoden, die die verschobenen Funktionen
aufrufen - bestehende Aufrufer innerhalb von `OverviewTab` funktionieren
unverändert weiter. Dieses Modul hier ruft dieselben, jetzt modul-
eigenständigen Funktionen für den `vorschau/`-Ordner auf.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Dieselben sieben Kategorien wie DataLODManager._KATEGORIE_SPEICHER - hier
# nochmals genannt, damit welt_backen()/welt_laden() nicht auf ein privates
# Attribut eines anderen Moduls zugreifen müssen; get_all_data()/
# set_all_data() sind der öffentliche Zugriffsweg.
KATEGORIEN = ("terrain", "geology", "settlement", "weather", "erosion", "water", "biome")

_GLOBAL_FELDER = ("map_seed", "map_distance_km", "map_latitude")

_MANIFEST_DATEI = "welt_manifest.json"
_ZUSTAND_ORDNER = "zustand"
_GODOT_ORDNER = "godot"
_VORSCHAU_ORDNER = "vorschau"

_FORMAT_VERSION = 1


class WeltBackenFehler(Exception):
    """Wird geworfen, wenn welt_backen() eine Welt nicht vollständig und
    nachweisbar korrekt schreiben konnte. Kein Teilzustand wird als Erfolg
    gemeldet."""


class WeltLadenFehler(Exception):
    """Wird geworfen, wenn welt_laden() ein für die geladene Welt laut
    Manifest erforderliches Feld nicht wiederherstellen konnte. Siehe
    Moduldocstring: es gibt hier bewusst KEINEN stillen Ersatzpfad (Ticket
    #54) - fehlt etwas, bricht das Laden ab, statt mit einem unvollständigen
    Weltzustand weiterzumachen."""


def welt_backen(pfad: str, data_lod_manager, parameter_manager=None) -> Dict[str, Any]:
    """
    Funktionsweise: Schreibt eine vollständige Welt nach `pfad` (wird bei
    Bedarf angelegt). Siehe Moduldocstring für das Drei-Ordner-Format
    (`zustand/`, `godot/`, `vorschau/`) und die Begründung dafür.
    Aufgabe: Die eine Stelle, über die JEDES zu einer Welt gehörende
    Datenprodukt auf Platte geschrieben wird (Ticket #38, "die Naht").
    Parameter:
        pfad             - Zielordner für die Welt
        data_lod_manager - DataLODManager-Instanz (oder ein Objekt mit
                            identischer get_all_data()/get_map_seed()/
                            get_map_distance_km()/get_map_latitude()-
                            Schnittstelle, z.B. ein Test-Double)
        parameter_manager - optional; wenn gegeben, werden zusätzlich die
                            aktuellen GUI-Parameter aller registrierten Tabs
                            gesichert (siehe Moduldocstring, Punkt 3)
    Return: das geschriebene Manifest-dict (dieselbe Struktur wie
        welt_manifest.json)
    Wirft: WeltBackenFehler, wenn das Manifest oder die globalen Werte nicht
        geschrieben werden konnten. Einzelne fehlende Kategorien sind KEIN
        Fehler (eine Welt, bei der z.B. noch keine Siedlung generiert wurde,
        ist gültig) - das Manifest vermerkt lediglich "nicht vorhanden".
    """
    os.makedirs(pfad, exist_ok=True)
    zustand_dir = os.path.join(pfad, _ZUSTAND_ORDNER)
    os.makedirs(zustand_dir, exist_ok=True)

    manifest: Dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "gebacken_am": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kategorien": {},
        "globals": {},
        "parameter_gesichert": False,
        "godot": None,
        "vorschau": None,
    }

    for kategorie in KATEGORIEN:
        try:
            data = data_lod_manager.get_all_data(kategorie)
        except Exception as exc:
            raise WeltBackenFehler(
                f"Kategorie '{kategorie}' konnte nicht gelesen werden: {exc}") from exc

        if not data:
            manifest["kategorien"][kategorie] = {"vorhanden": False, "keys": []}
            continue

        datei = os.path.join(zustand_dir, f"{kategorie}.pkl")
        try:
            with open(datei, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise WeltBackenFehler(
                f"Kategorie '{kategorie}' konnte nicht nach {datei} geschrieben werden: {exc}"
            ) from exc
        manifest["kategorien"][kategorie] = {"vorhanden": True, "keys": sorted(data.keys())}

    # Globale Werte (Seed/Massstab/Breitengrad) - ohne sie ist eine geladene
    # Welt nicht deterministisch reproduzierbar, siehe Moduldocstring Punkt 2.
    try:
        globals_werte = {
            "map_seed": data_lod_manager.get_map_seed(),
            "map_distance_km": data_lod_manager.get_map_distance_km(),
            "map_latitude": data_lod_manager.get_map_latitude(),
        }
        with open(os.path.join(zustand_dir, "globals.json"), "w", encoding="utf-8") as f:
            json.dump(globals_werte, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        raise WeltBackenFehler(f"Globale Werte konnten nicht geschrieben werden: {exc}") from exc
    manifest["globals"] = globals_werte

    # Optionale Parameter-Sicherung (siehe Moduldocstring Punkt 3 - bewusst
    # nicht Teil der lauten Kern-Fehlerprüfung beim Laden).
    if parameter_manager is not None:
        try:
            parameter = parameter_manager.get_all_parameters()
            with open(os.path.join(zustand_dir, "parameter.json"), "w", encoding="utf-8") as f:
                json.dump(parameter, f, ensure_ascii=False, indent=2, default=str)
            manifest["parameter_gesichert"] = True
        except Exception as exc:
            raise WeltBackenFehler(f"Parameter konnten nicht gesichert werden: {exc}") from exc

    # Godot-Export: bestehende, bereits getestete Pipeline unverändert
    # wiederverwendet (siehe Moduldocstring). Einbahnstrasse - wird von
    # welt_laden() nicht zurückgelesen, daher kein harter Fehler, wenn er
    # scheitert (z.B. weil noch gar kein Terrain generiert wurde);
    # das Ergebnis wird trotzdem sichtbar im Manifest vermerkt.
    try:
        from gui.utils.map_export import export_all_layers
        erfolg, meldung, _ziel = export_all_layers(
            data_lod_manager, parameter_manager, pfad, _GODOT_ORDNER)
        manifest["godot"] = {"erfolgreich": bool(erfolg), "meldung": meldung}
    except Exception as exc:
        logger.warning("Godot-Export in welt_backen() fehlgeschlagen: %s", exc)
        manifest["godot"] = {"erfolgreich": False, "meldung": str(exc)}

    # Vorschau: PNG-Previews + Text-Statistik ueber dieselben Funktionen, die
    # OverviewTab benutzt (Ticket-Vorgabe: wiederverwenden, nicht
    # duplizieren - siehe Moduldocstring). Ebenfalls kein harter Fehler.
    try:
        vorschau_dir = os.path.join(pfad, _VORSCHAU_ORDNER)
        os.makedirs(vorschau_dir, exist_ok=True)
        from gui.utils.map_export import (export_single_map_png,
                                          export_world_statistics_txt)

        vorschau_bilder = []
        terrain = data_lod_manager.get_all_data("terrain") or {}
        biome = data_lod_manager.get_all_data("biome") or {}
        for quelle, key in ((terrain, "heightmap"), (terrain, "slopemap"),
                           (biome, "biome_map")):
            arr = quelle.get(key)
            if isinstance(arr, np.ndarray) and arr.size > 0:
                export_single_map_png(arr, key, vorschau_dir, dpi=100)
                vorschau_bilder.append(key)

        statistik = {
            kategorie: {"anzahl_felder": len(manifest["kategorien"][kategorie]["keys"])}
            for kategorie in KATEGORIEN
        }
        export_world_statistics_txt(statistik, os.path.join(vorschau_dir, "statistik.txt"))
        manifest["vorschau"] = {"bilder": vorschau_bilder}
    except Exception as exc:
        logger.warning("Vorschau-Export in welt_backen() fehlgeschlagen: %s", exc)
        manifest["vorschau"] = {"bilder": [], "fehler": str(exc)}

    manifest_pfad = os.path.join(pfad, _MANIFEST_DATEI)
    try:
        with open(manifest_pfad, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        raise WeltBackenFehler(f"Manifest konnte nicht geschrieben werden: {exc}") from exc

    return manifest


def welt_laden(pfad: str, data_lod_manager, parameter_manager=None) -> Dict[str, Any]:
    """
    Funktionsweise: Liest eine mit welt_backen() geschriebene Welt aus `pfad`
    zurück und stellt sie im übergebenen data_lod_manager wieder her (über
    dessen set_all_data(), das Ticket #38 dort ergänzt hat).
    Aufgabe: Gegenstück zu welt_backen() - die einzige Stelle, die eine Welt
    von Platte liest. Liest NUR aus `zustand/`; `godot/` und `vorschau/` sind
    Einbahnstrassen-Ausgaben und werden hier nicht angefasst (siehe
    Moduldocstring).
    Parameter:
        pfad              - Ordner einer zuvor mit welt_backen() erzeugten
                            Welt
        data_lod_manager  - Ziel-DataLODManager (oder Test-Double mit
                            identischer get_all_data()/set_all_data()/
                            set_map_seed()/set_map_distance_km()/
                            set_map_latitude()-Schnittstelle)
        parameter_manager - optional; wenn gegeben UND die Welt Parameter
                            gesichert hat, werden sie auf registrierte Tabs
                            zurückgeschrieben (siehe Moduldocstring Punkt 3 -
                            ein nicht registrierter Tab ist dabei kein
                            Fehler)
    Return: das gelesene Manifest-dict
    Wirft: WeltLadenFehler, sobald ein laut Manifest vorhandenes Feld beim
        Zurücklesen ODER beim Zurückschreiben in data_lod_manager fehlt.
        Das ist die zentrale Zusicherung dieses Tickets: kein Feld wird
        lautlos durch nichts ersetzt (Ticket #54).
    """
    manifest_pfad = os.path.join(pfad, _MANIFEST_DATEI)
    if not os.path.isfile(manifest_pfad):
        raise WeltLadenFehler(
            f"Kein {_MANIFEST_DATEI} unter '{pfad}' - das ist keine mit "
            f"welt_backen() erzeugte Welt (oder der Pfad ist falsch).")

    with open(manifest_pfad, encoding="utf-8") as f:
        manifest = json.load(f)

    zustand_dir = os.path.join(pfad, _ZUSTAND_ORDNER)

    for kategorie, info in manifest.get("kategorien", {}).items():
        if not info.get("vorhanden"):
            continue

        datei = os.path.join(zustand_dir, f"{kategorie}.pkl")
        if not os.path.isfile(datei):
            raise WeltLadenFehler(
                f"Manifest nennt Kategorie '{kategorie}' als vorhanden, aber "
                f"'{datei}' fehlt.")

        try:
            with open(datei, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            raise WeltLadenFehler(
                f"Kategorie '{kategorie}' konnte nicht aus '{datei}' gelesen werden: {exc}"
            ) from exc

        erwartete_keys = set(info.get("keys", []))
        fehlend_in_datei = erwartete_keys - set(data.keys())
        if fehlend_in_datei:
            raise WeltLadenFehler(
                f"Kategorie '{kategorie}': im Manifest erwartete Keys fehlen in "
                f"'{datei}': {sorted(fehlend_in_datei)}")

        data_lod_manager.set_all_data(kategorie, data, lod_level=1)

        # Laute Nachpruefung: set_all_data()/_set_data_lod() lehnen ungueltige
        # Einzel-Keys OHNE Exception ab (bestehender Vertrag, siehe
        # data_lod_manager.set_all_data()-Docstring). Ein akzeptierter
        # Rundlauf darf sich davon nicht taeuschen lassen - deshalb hier
        # direkt zurücklesen und exakt die Key-Menge vergleichen.
        wiederhergestellt = data_lod_manager.get_all_data(kategorie)
        fehlend_nach_schreiben = erwartete_keys - set(wiederhergestellt.keys())
        if fehlend_nach_schreiben:
            raise WeltLadenFehler(
                f"Kategorie '{kategorie}': folgende Keys wurden beim "
                f"Zurückschreiben in data_lod_manager abgelehnt: "
                f"{sorted(fehlend_nach_schreiben)}")

    # Globale Werte
    globals_pfad = os.path.join(zustand_dir, "globals.json")
    if not os.path.isfile(globals_pfad):
        raise WeltLadenFehler(
            f"'{globals_pfad}' fehlt - Seed/Massstab/Breitengrad der Welt "
            f"sind nicht rekonstruierbar.")
    with open(globals_pfad, encoding="utf-8") as f:
        globals_werte = json.load(f)
    for feld in _GLOBAL_FELDER:
        if feld not in globals_werte or globals_werte[feld] is None:
            raise WeltLadenFehler(f"'{globals_pfad}' enthält kein gültiges '{feld}'.")
    data_lod_manager.set_map_seed(globals_werte["map_seed"])
    data_lod_manager.set_map_distance_km(globals_werte["map_distance_km"])
    data_lod_manager.set_map_latitude(globals_werte["map_latitude"])

    # Optionale Parameter (siehe Moduldocstring Punkt 3 - bewusst NICHT Teil
    # der lauten Kernprüfung: ein zum Ladezeitpunkt noch nicht registrierter
    # Tab ist normal, kein Datenverlust an einem Pflichtfeld).
    if manifest.get("parameter_gesichert"):
        parameter_pfad = os.path.join(zustand_dir, "parameter.json")
        if not os.path.isfile(parameter_pfad):
            raise WeltLadenFehler(
                f"Manifest verspricht gesicherte Parameter, aber "
                f"'{parameter_pfad}' fehlt.")
        with open(parameter_pfad, encoding="utf-8") as f:
            parameter = json.load(f)
        if parameter_manager is not None:
            for tab_name, tab_parameter in parameter.items():
                erfolg = parameter_manager.set_tab_parameters(
                    tab_name, tab_parameter, validate=False, notify_listeners=False)
                if erfolg is False:
                    logger.warning(
                        "welt_laden(): Parameter fuer Tab '%s' nicht uebernommen "
                        "(Tab vermutlich noch nicht registriert) - kein Abbruch, "
                        "da Parameter optional sind (siehe Moduldocstring Punkt 3).",
                        tab_name)

    return manifest
