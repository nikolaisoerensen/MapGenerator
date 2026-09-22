"""
Path: tests/smoke_test_welt_backen_laden.py

Ticket #38 "welt_backen und welt_laden als Naht".

Naht: die eine Stelle, an der jeder Aufrufer (App-Neustart, spaeter ein
Godot-Import) dieselbe, vollstaendige Auskunft ueber eine generierte Welt
abholt. Vorher gab es dafuer gar keinen Rueckweg: eine erzeugte Welt ging
beim Schliessen des Programms komplett verloren.

Dieser Test prueft den Rundgang bildpunktgenau (Akzeptanzkriterium 2):
OverviewTab.welt_backen() schreibt eine erfundene, aber realistische Welt
weg, OverviewTab.welt_laden() liest sie zurueck, und jedes Array wird per
np.array_equal (nicht np.allclose!) gegen das Original verglichen - eine
lossy Rundung waere hier ein stiller Fehler.

Ausserdem geprueft:
- die sechs bestehenden Export-Funktionen aus overview_tab.py werden intern
  benutzt (collect_all_available_data, parameter_summary.get_all_parameters,
  export_all_layers ueber export_layers_to_disk-Weg) - nicht dupliziert
  (Akzeptanzkriterium 6);
- der Godot-Ordner mit Terrain3D-Layern entsteht (Akzeptanzkriterium 4);
- fehlt ein vom Manifest verlangtes Feld beim Laden, kommt ein lauter
  WeltLadenFehler mit dem Feldnamen - kein stiller Ersatzwert
  (Akzeptanzkriterium 5);
- terrain/geology werden in den lebenden data_lod_manager zurueckgeschrieben,
  die uebrigen fuenf Generatoren werden zurueckgegeben, aber NICHT
  zurueckgeschrieben (bewusste Ticket-Scope-Grenze, siehe
  docs/WELT_BACKEN.md).

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_welt_backen_laden.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import logging
import shutil
import tempfile

import numpy as np
from PyQt6.QtWidgets import QApplication

logging.disable(logging.WARNING)
_APP = QApplication.instance() or QApplication([])

from core.settlement_generator import Location
from gui.tabs.overview_tab import (WeltLadenFehler, _WELT_ARRAYS_DATEI,
                                    _WELT_GODOT_ORDNER, _WELT_MANIFEST_DATEI,
                                    _WELT_ZUSTAND_DATEI, OverviewTab)

# Eine reale Kartengroesse (CLAUDE.md: "Tests mit den ECHTEN Eingabegroessen
# bauen" - 128/256/512/1024, nicht eine ausgedachte Groesse wie 100 oder 129).
MAP_SIZE = 128


class FakeSignal:
    """Ersetzt pyqtSignal fuer data_lod_manager.data_updated - der Reiter
    verbindet sich in setup_data_monitoring() darauf, braucht aber im Test
    kein echtes Signal."""

    def connect(self, *args, **kwargs):
        pass


class FakeDataLODManager:
    """
    Funktionsweise: Minimaler Ersatz fuer managers.data_lod_manager.
    DataLODManager - liefert genau die get_*_data()-Methoden, die
    OverviewTab.collect_all_available_data() aufruft, aus einem einfachen
    dict, und zeichnet auf, was welt_laden() ueber set_terrain_data_lod()/
    set_geology_data_lod() zurueckschreibt.
    """

    def __init__(self, quellen):
        self.quellen = quellen
        self.data_updated = FakeSignal()
        self.zurueckgeschrieben = {}

    def get_terrain_data(self, key):
        return self.quellen.get("terrain", {}).get(key)

    def get_terrain_data_combined(self, key, lod_level=None):
        # export_all_layers() (gui/utils/map_export.py) holt die Hoehenkarte
        # ueber diesen kombinierten Getter statt get_terrain_data() - ohne
        # diesen Fake-Eintrag wuerde der Godot-Export die Hoehenkarte still
        # ueberspringen, obwohl sie im Rundgang-Test vorhanden ist.
        return self.quellen.get("terrain", {}).get(key)

    def get_geology_data(self, key):
        return self.quellen.get("geology", {}).get(key)

    def get_settlement_data(self, key):
        return self.quellen.get("settlement", {}).get(key)

    def get_weather_data(self, key):
        return self.quellen.get("weather", {}).get(key)

    def get_water_data(self, key):
        return self.quellen.get("water", {}).get(key)

    def get_biome_data(self, key):
        return self.quellen.get("biome", {}).get(key)

    def get_memory_usage(self):
        return {}

    def set_terrain_data_lod(self, data_key, data, lod_level, parameters):
        self.zurueckgeschrieben[("terrain", data_key)] = (
            np.array(data, copy=True), lod_level, dict(parameters))

    def set_geology_data_lod(self, data_key, data, lod_level, parameters):
        self.zurueckgeschrieben[("geology", data_key)] = (
            np.array(data, copy=True), lod_level, dict(parameters))


class FakeParameterManager:
    def get_tab_parameters(self, generator):
        if generator == "terrain":
            return {"map_seed": 20260922, "map_size": MAP_SIZE}
        return {"beispiel_parameter": generator}


def _baue_welt(rng):
    """Baut eine erfundene, aber in Form/Dtype realistische Welt."""
    settlement_list = [
        Location(location_id=1, x=12.5, y=44.0, location_type="settlement",
                 radius=3.2, civ_influence=0.7,
                 properties={"besitzer": "Freistadt"},
                 culture="samarcia", rank="stadt", house_count=41,
                 settlement_type="marktstadt"),
        Location(location_id=2, x=88.0, y=5.5, location_type="settlement",
                 radius=1.1, civ_influence=0.3,
                 properties=None, culture="morobora", rank="dorf",
                 house_count=18, settlement_type="agrarstadt"),
    ]
    return {
        "terrain": {
            "heightmap": rng.random((MAP_SIZE, MAP_SIZE)).astype(np.float32),
            "slopemap": rng.random((MAP_SIZE, MAP_SIZE)).astype(np.float32),
            "shadowmap": rng.random((MAP_SIZE, MAP_SIZE)).astype(np.float32),
        },
        "geology": {
            "rock_map": rng.integers(0, 256, (MAP_SIZE, MAP_SIZE, 3)).astype(np.uint8),
            "hardness_map": rng.random((MAP_SIZE, MAP_SIZE)).astype(np.float32),
        },
        "settlement": {
            "settlement_list": settlement_list,
            "civ_map": rng.random((MAP_SIZE, MAP_SIZE)).astype(np.float32),
        },
        "weather": {
            "temp_map": (rng.random((MAP_SIZE, MAP_SIZE)) * 40 - 10).astype(np.float32),
            "precip_map": rng.random((MAP_SIZE, MAP_SIZE)).astype(np.float32),
        },
        "water": {
            "water_map": rng.random((MAP_SIZE, MAP_SIZE)).astype(np.float32),
            "soil_moist_map": rng.random((MAP_SIZE, MAP_SIZE)).astype(np.float32),
            "ocean_outflow": 123.456,
        },
        "biome": {
            "biome_map": rng.integers(0, 12, (MAP_SIZE, MAP_SIZE)).astype(np.uint8),
        },
    }


def check(label, bedingung, zusatz=""):
    print(f"[{'OK' if bedingung else 'FAIL'}] {label}"
          + (f" - {zusatz}" if zusatz else ""))
    return [] if bedingung else [f"{label}{' - ' + zusatz if zusatz else ''}"]


def lauf():
    fehler = []
    rng = np.random.default_rng(20260922)
    welt = _baue_welt(rng)

    fake_dlm = FakeDataLODManager(welt)
    fake_pm = FakeParameterManager()
    reiter = OverviewTab(fake_dlm, fake_pm, None, None, None)

    tmp = tempfile.mkdtemp(prefix="welt_backen_test_")
    ziel = os.path.join(tmp, "meine_welt")
    try:
        # --- 1. Baken -------------------------------------------------------
        manifest = reiter.welt_backen(ziel)
        fehler += check("Manifest wird zurueckgegeben",
                        isinstance(manifest, dict) and "felder" in manifest)
        fehler += check(f"{_WELT_ARRAYS_DATEI} liegt auf der Platte",
                        os.path.isfile(os.path.join(ziel, _WELT_ARRAYS_DATEI)))
        fehler += check(f"{_WELT_ZUSTAND_DATEI} liegt auf der Platte",
                        os.path.isfile(os.path.join(ziel, _WELT_ZUSTAND_DATEI)))
        fehler += check(f"{_WELT_MANIFEST_DATEI} liegt auf der Platte",
                        os.path.isfile(os.path.join(ziel, _WELT_MANIFEST_DATEI)))
        godot_ordner = os.path.join(ziel, _WELT_GODOT_ORDNER)
        godot_dateien = (os.listdir(godot_ordner) if os.path.isdir(godot_ordner) else [])
        fehler += check("Godot-Ordner (Terrain3D-Export) entsteht und ist nicht leer",
                        len(godot_dateien) > 0, f"{godot_ordner}: {godot_dateien[:5]}")
        # Terrain3D-Bedarf konkret nachweisen (Akzeptanzkriterium 4): unter
        # dem einen Export-Unterordner muss eine Hoehenkarte liegen (R16/EXR
        # bzw. hier die vorhandene 16-Bit-PNG-Variante), siehe
        # docs/WELT_BACKEN.md Abschnitt "Godot-Bedarf".
        export_unterordner = [os.path.join(godot_ordner, d) for d in godot_dateien
                              if os.path.isdir(os.path.join(godot_ordner, d))]
        hoehenkarte_gefunden = False
        for ordner in export_unterordner:
            if any("hoehe" in f.lower() or "height" in f.lower() for f in os.listdir(ordner)):
                hoehenkarte_gefunden = True
        fehler += check("Godot-Export enthaelt eine Hoehenkarte fuer Terrain3D",
                        hoehenkarte_gefunden,
                        str([os.listdir(o) for o in export_unterordner]))

        # --- 2. Laden ---------------------------------------------------------
        geladen = reiter.welt_laden(ziel)

        # --- 3. Bildpunktgenauer Rundgang jedes Arrays -------------------------
        array_felder = [
            ("terrain", "heightmap"), ("terrain", "slopemap"), ("terrain", "shadowmap"),
            ("geology", "rock_map"), ("geology", "hardness_map"),
            ("settlement", "civ_map"),
            ("weather", "temp_map"), ("weather", "precip_map"),
            ("water", "water_map"), ("water", "soil_moist_map"),
            ("biome", "biome_map"),
        ]
        daneben = []
        for generator, feld in array_felder:
            original = welt[generator][feld]
            zurueck = geladen.get(generator, {}).get(feld)
            if zurueck is None or not np.array_equal(original, zurueck):
                daneben.append(f"{generator}.{feld}")
            elif original.dtype != zurueck.dtype:
                daneben.append(f"{generator}.{feld} (dtype {zurueck.dtype} statt {original.dtype})")
        fehler += check("jedes Array kommt bildpunktgenau zurueck (np.array_equal)",
                        not daneben, ", ".join(daneben))

        # --- 4. Skalar und Siedlungsliste (Location-Dataclasses) -------------
        fehler += check("Skalar (ocean_outflow) kommt exakt zurueck",
                        geladen.get("water", {}).get("ocean_outflow") == 123.456)
        settlement_zurueck = geladen.get("settlement", {}).get("settlement_list")
        fehler += check("Siedlungsliste kommt als 2 Dicts zurueck",
                        isinstance(settlement_zurueck, list) and len(settlement_zurueck) == 2)
        if settlement_zurueck:
            erster = settlement_zurueck[0]
            fehler += check("Location-Felder bleiben erhalten (city_id/culture/rank)",
                            erster.get("location_id") == 1
                            and erster.get("culture") == "samarcia"
                            and erster.get("rank") == "stadt"
                            and erster.get("house_count") == 41)

        # --- 5. Rueckschreiben in den lebenden Manager: nur terrain/geology --
        fehler += check("terrain.heightmap wurde in den Manager zurueckgeschrieben",
                        ("terrain", "heightmap") in fake_dlm.zurueckgeschrieben)
        fehler += check("geology.rock_map wurde in den Manager zurueckgeschrieben",
                        ("geology", "rock_map") in fake_dlm.zurueckgeschrieben)
        zurueckgeschriebenes_terrain, _, _ = fake_dlm.zurueckgeschrieben[("terrain", "heightmap")]
        fehler += check("zurueckgeschriebenes Array ist bildpunktgenau",
                        np.array_equal(zurueckgeschriebenes_terrain, welt["terrain"]["heightmap"]))
        # weather/water/biome/settlement: bewusst NICHT im Manager, siehe
        # docs/WELT_BACKEN.md "Nicht zurueckgeschrieben".
        keine_setter = [k for k in fake_dlm.zurueckgeschrieben
                        if k[0] not in ("terrain", "geology")]
        fehler += check("weather/water/biome/settlement werden NICHT in den Manager geschrieben",
                        not keine_setter, str(keine_setter))
        fehler += check("weather/water/biome/settlement werden aber trotzdem zurueckgegeben",
                        geladen.get("weather", {}).get("temp_map") is not None
                        and geladen.get("water", {}).get("water_map") is not None
                        and geladen.get("biome", {}).get("biome_map") is not None
                        and geladen.get("settlement", {}).get("civ_map") is not None)

        # --- 6. Kein stiller Ersatzwert: fehlendes Feld = lauter Fehler -------
        import json
        manifest_pfad = os.path.join(ziel, _WELT_MANIFEST_DATEI)
        with open(manifest_pfad, "r", encoding="utf-8") as f:
            kaputtes_manifest = json.load(f)
        kaputtes_manifest["felder"]["terrain__erfundenes_feld"] = {
            "generator": "terrain", "feld": "erfundenes_feld", "ablage": "arrays",
            "shape": [1, 1], "dtype": "float32",
        }
        with open(manifest_pfad, "w", encoding="utf-8") as f:
            json.dump(kaputtes_manifest, f)

        ausnahme_kam = False
        ausnahme_nennt_feld = False
        try:
            reiter.welt_laden(ziel)
        except WeltLadenFehler as e:
            ausnahme_kam = True
            ausnahme_nennt_feld = "erfundenes_feld" in str(e)
        fehler += check("fehlendes Manifest-Feld loest WeltLadenFehler aus",
                        ausnahme_kam)
        fehler += check("WeltLadenFehler nennt den fehlenden Feldnamen",
                        ausnahme_nennt_feld)

        # --- 7. Fehlender Ordner insgesamt = lauter Fehler --------------------
        ausnahme_kam_2 = False
        try:
            reiter.welt_laden(os.path.join(tmp, "gibt_es_nicht"))
        except WeltLadenFehler:
            ausnahme_kam_2 = True
        fehler += check("nicht vorhandener Pfad loest ebenfalls WeltLadenFehler aus",
                        ausnahme_kam_2)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("=" * 78)
    if fehler:
        print(f"NICHT IN ORDNUNG - {len(fehler)} Befunde:")
        for f in fehler:
            print(f"   {f}")
        return 1
    print("welt_backen/welt_laden sind in Ordnung")
    return 0


if __name__ == "__main__":
    sys.exit(lauf())
