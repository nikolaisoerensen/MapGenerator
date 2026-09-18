"""
Path: gui/utils/welt_format.py

Ticket #38, "welt_backen und welt_laden als Naht".

Naht: die EINE Stelle, an der eine ganze generierte Welt weggeschrieben und
wieder eingelesen wird. Vorher gab es nur "Export" (sechs Methoden verstreut
in gui/tabs/overview_tab.py) und gar kein "Import" - der Editor konnte eine
Welt erzeugen, aber nicht wieder laden.

welt_backen(pfad, data_lod_manager, parameter_manager) schreibt.
welt_laden(pfad, data_lod_manager) liest zurück und stellt den Zustand in
einem DataLODManager wieder her.

Baustein statt Duplikat: welt_backen() ruft export_complete_json() aus
gui/tabs/overview_tab.py auf - das JSON-Format (Metadaten, Parameter,
world_data mit shape/dtype je Array, Statistik) entsteht dort bereits
korrekt, siehe dessen Docstring zu den zwei Fehlern, die Ticket #38 nebenbei
mitbehoben hat (Location-Dataclasses ohne JSON-Handler, verlustbehaftete
`str(map_data)`-Ersatzschreibung). Diese Datei baut das Format nicht ein
zweites Mal, sondern liest exakt das wieder ein, was dort geschrieben wird.

Feldliste: docs/WELT_FORMAT.md. Godot/Terrain3D-Bedarf: siehe dort, Abschnitt
"Was Terrain3D braucht" (verweist auf docs/OFFENE_PUNKTE.md Punkt 13.3, wo
das schon einmal benannt wurde, aber noch offen ist - dieses Ticket deckt nur
die Naht ab, nicht den Terrain3D-Export selbst).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from gui.tabs.overview_tab import (
    WELT_DATEN_SCHLUESSEL,
    REQUIRED_WORLD_DATA,
    collect_all_available_data,
    get_all_parameters,
    export_complete_json,
)

logger = logging.getLogger(__name__)

# Nur informativ (Metadaten-Feld) - keine Kompatibilitätsprüfung eingebaut,
# weil es bislang nur genau eine Version gibt. Wird das Format irgendwann
# geändert, ist das der Anknüpfungspunkt für eine Migrationsprüfung in
# welt_laden().
WELT_FORMAT_VERSION = "1.0"

PfadTyp = Union[str, Path]


class WeltFormatFehler(Exception):
    """
    Wird geworfen, wenn eine Welt-Datei nicht zu welt_laden() passt: Datei
    fehlt, ist kein gültiges Welt-JSON, oder ein Pflichtfeld aus
    REQUIRED_WORLD_DATA fehlt.

    Bewusst eine laute Exception statt eines stillen Ersatzwerts - siehe
    CLAUDE.md, Abschnitt "Gruene Tests koennen eine tote Funktion
    verdecken": ein `.get(key, default)` an dieser Stelle wäre von einer
    echten, vollständigen Welt nicht mehr zu unterscheiden.
    """


def welt_backen(pfad: PfadTyp, data_lod_manager, parameter_manager=None) -> None:
    """
    Funktionsweise: Schreibt den kompletten Weltzustand (alle
    Generator-Daten aus WELT_DATEN_SCHLUESSEL + alle Tab-Parameter) nach
    `pfad`. Baustein: export_complete_json() aus gui/tabs/overview_tab.py -
    dessen JSON-Struktur wird hier verwendet, nicht dupliziert.
    Parameter:
        pfad - Zielpfad (str oder Path), z.B. "meine_welt.json"
        data_lod_manager - DataLODManager der laufenden Sitzung
        parameter_manager - ParameterManager der laufenden Sitzung (darf
            None sein - dann werden leere Parameter-Dicts geschrieben, siehe
            get_all_parameters())
    Raises: WeltFormatFehler, wenn export_complete_json() fehlschlägt (Grund
        steht im Log - export_complete_json() fängt die Exception ab und
        gibt nur False zurück).
    """
    pfad = str(pfad)
    verfuegbare_daten = collect_all_available_data(data_lod_manager)
    alle_parameter = get_all_parameters(parameter_manager)

    speicherverbrauch_mb = 0.0
    try:
        speicherverbrauch_mb = sum(data_lod_manager.get_memory_usage().values())
    except Exception as e:
        logger.debug(f"Speicherverbrauch nicht ermittelbar: {e}")

    optionen = {"export_file": pfad}
    erfolg = export_complete_json(verfuegbare_daten, alle_parameter, optionen, speicherverbrauch_mb)
    if not erfolg:
        raise WeltFormatFehler(
            f"welt_backen() nach '{pfad}' fehlgeschlagen - export_complete_json() hat "
            f"False zurückgegeben, die genaue Ursache steht im Log (logger "
            f"'gui.tabs.overview_tab')."
        )

    logger.info(f"Welt gebacken nach '{pfad}'")


def _ist_array_eintrag(wert: Any) -> bool:
    """Erkennt die Array-Markierung, die export_complete_json() für jedes
    np.ndarray schreibt: {"data": ..., "shape": [...], "dtype": "..."}."""
    return isinstance(wert, dict) and {"data", "shape", "dtype"} <= wert.keys()


def _array_aus_eintrag(eintrag: Dict[str, Any]) -> np.ndarray:
    """Rekonstruiert ein np.ndarray verlustfrei aus data/shape/dtype - das
    Gegenstück zu `map_data.tolist()` + `str(map_data.dtype)` in
    export_complete_json()."""
    flach = np.array(eintrag["data"], dtype=eintrag["dtype"])
    return flach.reshape(eintrag["shape"])


def _pruefe_pflichtfelder(pfad: str, world_data: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """
    Prüft NICHT REQUIRED_WORLD_DATA als feste Liste gegen die Datei - siehe
    der Kommentar direkt über REQUIRED_WORLD_DATA in overview_tab.py: das
    wäre die GUI-Ampel (WorldCompletenessWidget), keine Dateiformat-Prüfung,
    und würde jede echte, nur teilweise generierte Welt (z.B. ohne Erosion
    gerechnet) als "kaputt" melden.

    Stattdessen wird gegen das geprüft, was welt_backen() beim Schreiben
    SELBST als vollständig deklariert hat: metadata["data_completeness"]
    ["generator_status"], von analyze_data_completeness() zum Zeitpunkt des
    Backens berechnet (True je Kategorie, wenn zu dem Zeitpunkt alle
    REQUIRED_WORLD_DATA-Felder dieser Kategorie vorhanden waren). Nur für
    eine Kategorie, die bei jenem Snapshot als vollständig galt, ist ein
    fehlendes Pflichtfeld JETZT ein Widerspruch - also ein Zeichen für eine
    zwischen Backen und Laden beschädigte/verstümmelte Datei, kein normaler
    Zwischenstand einer unfertigen Welt.
    """
    vollstaendigkeit = metadata.get("data_completeness")
    if not isinstance(vollstaendigkeit, dict) or "generator_status" not in vollstaendigkeit:
        raise WeltFormatFehler(
            f"welt_laden(): '{pfad}' hat kein 'metadata.data_completeness.generator_status' - "
            f"das schreibt export_complete_json() bei JEDEM welt_backen()-Lauf. Die Datei "
            f"stammt nicht aus welt_backen() oder ist beschädigt."
        )
    generator_status = vollstaendigkeit["generator_status"]

    fehlend = []
    for kategorie, als_vollstaendig_deklariert in generator_status.items():
        if not als_vollstaendig_deklariert:
            continue  # diese Kategorie war beim Backen selbst schon unvollstaendig - kein Fehler
        pflicht_schluessel = REQUIRED_WORLD_DATA.get(kategorie, [])
        kategorie_daten = world_data.get(kategorie, {}) or {}
        for schluessel in pflicht_schluessel:
            wert = kategorie_daten.get(schluessel)
            if wert is None:
                fehlend.append(f"{kategorie}.{schluessel}")

    if fehlend:
        raise WeltFormatFehler(
            f"welt_laden(): '{pfad}' war beim Backen als vollständig deklariert, jetzt fehlen "
            f"aber: {', '.join(fehlend)}. Kein stiller Ersatzwert - die Datei ist zwischen "
            f"Backen und Laden beschädigt oder verstümmelt worden, das Laden bricht ab."
        )


def welt_laden(pfad: PfadTyp, data_lod_manager) -> Dict[str, Any]:
    """
    Funktionsweise: Liest eine mit welt_backen() geschriebene Datei zurück
    und stellt den Zustand in `data_lod_manager` wieder her - für jeden
    Schlüssel aus WELT_DATEN_SCHLUESSEL unter einem neuen, kategorieeigenen
    LOD-Level (aktuelles LOD + 1). Damit liefert
    data_lod_manager.get_<kategorie>_data(schluessel) (ohne explizites
    LOD-Level, "höchstes verfügbares") danach genau den geladenen Wert -
    siehe DataLODManager._get_data_lod().
    Parameter:
        pfad - Quelldatei (wie an welt_backen() übergeben)
        data_lod_manager - DataLODManager, der befüllt werden soll
    Return: die in der Datei gespeicherten Parameter (all_parameters aus
        welt_backen(), also dict Generator -> Parameter-dict)
    Raises: WeltFormatFehler, wenn die Datei fehlt, kein gültiges Welt-JSON
        ist, oder ein Pflichtfeld aus REQUIRED_WORLD_DATA fehlt.
    """
    pfad = str(pfad)
    weg = Path(pfad)
    if not weg.exists():
        raise WeltFormatFehler(f"welt_laden(): Datei nicht gefunden: '{pfad}'")

    with open(weg, "r", encoding="utf-8") as f:
        try:
            rohdaten = json.load(f)
        except json.JSONDecodeError as e:
            raise WeltFormatFehler(f"welt_laden(): '{pfad}' ist kein gültiges JSON: {e}") from e

    if not isinstance(rohdaten, dict) or "world_data" not in rohdaten or "metadata" not in rohdaten:
        raise WeltFormatFehler(
            f"welt_laden(): '{pfad}' ist kein gültiges Welt-Format - "
            f"'world_data'/'metadata' fehlen auf oberster Ebene."
        )

    world_data = rohdaten["world_data"]
    metadata = rohdaten["metadata"]
    _pruefe_pflichtfelder(pfad, world_data, metadata)

    for kategorie, schluessel_liste in WELT_DATEN_SCHLUESSEL.items():
        kategorie_daten = world_data.get(kategorie, {}) or {}
        if not kategorie_daten:
            continue

        store = getattr(data_lod_manager, f"_{kategorie}_data", None)
        if store is None:
            raise WeltFormatFehler(
                f"welt_laden(): DataLODManager hat keinen Speicher für Kategorie "
                f"'{kategorie}' (erwartet Attribut '_{kategorie}_data') - Feldliste "
                f"und Manager sind auseinandergelaufen, siehe docs/WELT_FORMAT.md."
            )

        ziel_lod = data_lod_manager._current_lods.get(kategorie, 0) + 1
        geschriebene_schluessel = []

        for schluessel in schluessel_liste:
            if schluessel not in kategorie_daten:
                continue
            wert = kategorie_daten[schluessel]
            if wert is None:
                continue

            if _ist_array_eintrag(wert):
                wert = _array_aus_eintrag(wert)
                data_lod_manager._set_data_lod(kategorie, store, schluessel, wert, ziel_lod, {})
            else:
                data_lod_manager._set_data_lod(
                    kategorie, store, schluessel, wert, ziel_lod, {}, require_array=False
                )
            geschriebene_schluessel.append(schluessel)

        if geschriebene_schluessel:
            # Dieselben zwei Signale, die auch set_<kategorie>_data_complete_lod()
            # am Ende feuert - ohne sie bliebe die GUI (Reiter, die auf
            # data_updated hören) nach dem Laden auf dem alten Anzeigestand.
            data_lod_manager.lod_data_stored.emit(kategorie, ziel_lod, geschriebene_schluessel)
            data_lod_manager.data_updated.emit(kategorie, "complete")

    logger.info(f"Welt geladen aus '{pfad}'")
    return rohdaten.get("parameters", {})
