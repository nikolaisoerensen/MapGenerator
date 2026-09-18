"""
Path: tests/smoke_test_welt_backen_laden.py

Rundreise-Test fuer Ticket #38 ("welt_backen und welt_laden als Naht").

Baut eine synthetische Welt in einem DataLODManager (ein Wert je Feld aus
WELT_DATEN_SCHLUESSEL, mit denselben dtypes wie im echten Betrieb -
float32-Karten, uint8-rock_map), baeckt sie mit welt_backen() in eine
temporaere Datei und laedt sie in einem ZWEITEN, frischen DataLODManager mit
welt_laden() zurueck. Verglichen wird Bildpunkt fuer Bildpunkt
(np.array_equal + gleicher dtype) mit dem Original - das ist das
Abnahmekriterium "Ergebnis bildpunktgenau identisch wie vorher".

Zusaetzlich zwei Fehlerfaelle, weil das Abnahmekriterium "Fehlt beim Laden
ein Feld: laute Fehlermeldung, kein stilles Ersatzfeld" sich nicht von
selbst prueft:
    - welt_laden() auf eine nicht existierende Datei
    - welt_laden() auf eine Datei ohne ein Pflichtfeld aus REQUIRED_WORLD_DATA
Beide muessen WeltFormatFehler werfen, nicht None/leeres dict liefern.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_welt_backen_laden.py
"""

import json
import os
import sys
import tempfile
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Die Qt-Anwendung MUSS modulweit gehalten werden - als lokale Variable
# raeumt Python sie waehrend des Laufs ab (siehe
# tests/smoke_test_pipeline_outputs.py, derselbe Kommentar dort).
_QT_APP = None


def _qt():
    global _QT_APP
    from PyQt6.QtGui import QGuiApplication
    if _QT_APP is None:
        _QT_APP = QGuiApplication.instance() or QGuiApplication([])
    return _QT_APP


SIZE = 32  # klein und schnell - reine Rundreise-Pruefung, keine Geometrieaussage


def _synthetische_welt(manager):
    """
    Befuellt einen frischen DataLODManager direkt ueber dessen eigenen,
    generischen Ablage-Pfad (_set_data_lod) mit einem Wert je Feld aus
    WELT_DATEN_SCHLUESSEL - realistischen dtypes (float32 fuer Karten,
    uint8 fuer rock_map, int32 fuer Klassifikationskarten), damit der
    Rundreise-Test echte Rundungsfallen abdeckt, nicht nur float64-Listen.
    Return: dasselbe verschachtelte dict, als Referenz fuer den Vergleich
        nach dem Laden.
    """
    rng = np.random.default_rng(20260918)

    werte = {
        "terrain": {
            "heightmap": rng.uniform(0, 1, (SIZE, SIZE)).astype(np.float32),
            "slopemap": rng.uniform(0, 1, (SIZE, SIZE)).astype(np.float32),
            "shadowmap": rng.uniform(0, 1, (SIZE, SIZE)).astype(np.float32),
        },
        "geology": {
            "rock_map": rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8),
            "hardness_map": rng.uniform(0, 1, (SIZE, SIZE)).astype(np.float32),
        },
        "erosion": {
            "erosion_map": rng.uniform(0, 1, (SIZE, SIZE)).astype(np.float32),
            "sedimentation_map": rng.uniform(0, 1, (SIZE, SIZE)).astype(np.float32),
            "sediment_load_map": rng.uniform(0, 1, (SIZE, SIZE)).astype(np.float32),
        },
        "weather": {
            "temp_map": rng.uniform(-20, 40, (SIZE, SIZE)).astype(np.float32),
            "precip_map": rng.uniform(0, 3000, (SIZE, SIZE)).astype(np.float32),
        },
        "water": {
            "water_map": rng.uniform(0, 1, (SIZE, SIZE)).astype(np.float32),
            "soil_moist_map": rng.uniform(0, 1, (SIZE, SIZE)).astype(np.float32),
            "water_biomes_map": rng.integers(0, 5, (SIZE, SIZE)).astype(np.int32),
            "ocean_outflow": 1234.5,  # Skalar statt Array - anderer Codepfad
        },
        "biome": {
            "biome_map": rng.integers(0, 27, (SIZE, SIZE)).astype(np.int32),
        },
        "settlement": {
            "civ_map": rng.uniform(0, 1, (SIZE, SIZE)).astype(np.float32),
            "settlement_list": [{"name": "Testburg", "x": 3, "y": 4}],  # Liste statt Array
        },
    }

    for kategorie, felder in werte.items():
        store = getattr(manager, f"_{kategorie}_data")
        for schluessel, wert in felder.items():
            if isinstance(wert, np.ndarray):
                manager._set_data_lod(kategorie, store, schluessel, wert, 1, {})
            else:
                manager._set_data_lod(kategorie, store, schluessel, wert, 1, {}, require_array=False)

    return werte


def _vergleiche(original_werte, manager_geladen, protokoll):
    ok = True
    for kategorie, felder in original_werte.items():
        for schluessel, original in felder.items():
            geladen = getattr(manager_geladen, f"get_{kategorie}_data")(schluessel)
            if isinstance(original, np.ndarray):
                if geladen is None:
                    protokoll.append(f"FEHLT NACH LADEN: {kategorie}.{schluessel}")
                    ok = False
                    continue
                if not isinstance(geladen, np.ndarray) or geladen.dtype != original.dtype:
                    protokoll.append(
                        f"DTYPE ABWEICHT: {kategorie}.{schluessel} "
                        f"vorher={original.dtype} nachher={getattr(geladen, 'dtype', type(geladen))}")
                    ok = False
                    continue
                if not np.array_equal(original, geladen):
                    protokoll.append(f"BILDPUNKTE ABWEICHEND: {kategorie}.{schluessel}")
                    ok = False
            else:
                if geladen != original:
                    protokoll.append(
                        f"WERT ABWEICHT: {kategorie}.{schluessel} vorher={original!r} nachher={geladen!r}")
                    ok = False
    return ok


def haupttest():
    _qt()
    from managers.data_lod_manager import DataLODManager
    from gui.tabs.overview_tab import WELT_DATEN_SCHLUESSEL
    from gui.utils.welt_format import welt_backen, welt_laden, WeltFormatFehler

    protokoll = []
    rundreise_ok = True

    manager_original = DataLODManager()
    original_werte = _synthetische_welt(manager_original)

    with tempfile.TemporaryDirectory() as tmp:
        pfad = os.path.join(tmp, "testwelt.json")
        welt_backen(pfad, manager_original, parameter_manager=None)

        if not os.path.exists(pfad):
            protokoll.append("welt_backen() hat keine Datei geschrieben")
            rundreise_ok = False

        # --- 1) Backen + Laden, Bildpunkt-Rundreise ---
        manager_geladen = DataLODManager()
        welt_laden(pfad, manager_geladen)
        if not _vergleiche(original_werte, manager_geladen, protokoll):
            rundreise_ok = False

        # --- 2) Feldliste: jede Kategorie aus WELT_DATEN_SCHLUESSEL muss im
        #     JSON auftauchen (auch wenn hier nicht jedes optionale Feld
        #     befuellt wurde). ---
        with open(pfad, "r", encoding="utf-8") as f:
            rohdaten = json.load(f)
        for kategorie in WELT_DATEN_SCHLUESSEL:
            if kategorie not in rohdaten.get("world_data", {}):
                protokoll.append(f"KATEGORIE FEHLT IM JSON: {kategorie}")
                rundreise_ok = False

        # --- 3) Fehlende Datei -> laute Exception, kein stiller Ersatzwert ---
        fehlt_pfad = os.path.join(tmp, "gibt_es_nicht.json")
        try:
            welt_laden(fehlt_pfad, DataLODManager())
            protokoll.append("welt_laden() auf fehlende Datei hat NICHT geworfen")
            rundreise_ok = False
        except WeltFormatFehler:
            pass

        # --- 4) Fehlendes Pflichtfeld -> laute Exception ---
        kaputter_pfad = os.path.join(tmp, "kaputte_welt.json")
        rohdaten_kaputt = json.loads(json.dumps(rohdaten))  # tiefe Kopie
        del rohdaten_kaputt["world_data"]["terrain"]["heightmap"]
        with open(kaputter_pfad, "w", encoding="utf-8") as f:
            json.dump(rohdaten_kaputt, f)
        try:
            welt_laden(kaputter_pfad, DataLODManager())
            protokoll.append("welt_laden() auf fehlendes Pflichtfeld hat NICHT geworfen")
            rundreise_ok = False
        except WeltFormatFehler:
            pass

    print("=" * 70)
    print("RUNDREISE-TEST welt_backen() / welt_laden() (Ticket #38)")
    print("=" * 70)
    if not protokoll:
        print("Alle Bildpunkte, Skalare und Listen nach dem Laden identisch mit vorher.")
        print("Fehlende Datei und fehlendes Pflichtfeld brechen beide laut ab (WeltFormatFehler).")
    else:
        for zeile in protokoll:
            print(" - " + zeile)

    return rundreise_ok


if __name__ == "__main__":
    try:
        erfolg = haupttest()
    except Exception:
        traceback.print_exc()
        erfolg = False
    sys.exit(0 if erfolg else 1)
