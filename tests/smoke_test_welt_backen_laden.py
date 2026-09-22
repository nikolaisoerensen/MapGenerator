"""
Path: tests/smoke_test_welt_backen_laden.py

Ticket #38 "welt_backen und welt_laden als Naht" - die GUI-Seite der Naht.

Naht: die eine Stelle, an der jeder Aufrufer (App-Neustart, spaeter ein
Godot-Import) dieselbe, vollstaendige Auskunft ueber eine generierte Welt
abholt. Vorher gab es dafuer gar keinen Rueckweg: eine erzeugte Welt ging
beim Schliessen des Programms komplett verloren.

ABGRENZUNG ZU tests/smoke_test_welt_io_roundtrip.py

Jener Test prueft core/welt_io.py fuer sich, gegen ein Test-Double des
Managers - schnell, ohne Qt. DIESER Test prueft die beiden Dinge, die dort
prinzipiell nicht vorkommen koennen:

1. den ECHTEN managers.data_lod_manager.DataLODManager, also die Frage, ob
   eine geladene Welt im laufenden Programm auch WIRKLICH ankommt (alle
   sieben Kategorien, das richtige LOD, und das zusammengesetzte
   terrain_data_object, an dem der GenerationOrchestrator haengt);
2. gui.tabs.overview_tab.OverviewTab.welt_backen()/welt_laden(), die seit
   der Zusammenfuehrung nur noch nach core/welt_io.py weiterleiten - dass
   diese Weiterleitung steht und die alte Antwortform liefert.

Geprueft wird bildpunktgenau (Akzeptanzkriterium 2): jedes Array per
np.array_equal (nicht np.allclose!) gegen das Original - eine lossy Rundung
waere hier ein stiller Fehler.

Ausserdem geprueft:
- der Godot-Ordner mit Terrain3D-Layern entsteht (Akzeptanzkriterium 4);
- fehlt eine vom Manifest verlangte Kategorie beim Laden, kommt ein lauter
  WeltLadenFehler mit dem Namen - kein stiller Ersatzwert
  (Akzeptanzkriterium 5);
- ALLE SIEBEN Generatoren werden in den lebenden data_lod_manager
  zurueckgeschrieben. Das ist die Aenderung gegenueber der frueheren Fassung
  dieses Tests, die nur terrain/geology zurueckschrieb und die uebrigen fuenf
  bloss zurueckgab - eine geladene Welt war damit halb unsichtbar.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_welt_backen_laden.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import logging
import shutil
import tempfile

import numpy as np
from PyQt6.QtWidgets import QApplication

logging.disable(logging.WARNING)
_APP = QApplication.instance() or QApplication([])

from core.settlement_generator import Location
from core.welt_io import KATEGORIEN
from gui.tabs.overview_tab import (WeltLadenFehler, _WELT_GODOT_ORDNER,
                                   _WELT_MANIFEST_DATEI, OverviewTab)
from managers.data_lod_manager import DataLODManager

# Eine reale Kartengroesse (CLAUDE.md: "Tests mit den ECHTEN Eingabegroessen
# bauen" - 128/256/512/1024, nicht eine ausgedachte Groesse wie 100 oder 129).
MAP_SIZE = 128

# Das LOD, auf dem die Testwelt liegt. Bewusst NICHT 1: nur so faellt auf, wenn
# welt_laden() pauschal auf LOD 1 zurueckschriebe - die Welt laege dann
# unterhalb des aktuellen Standes und waere fuer jeden Leser unsichtbar.
TEST_LOD = 3


class FakeParameterManager:
    """Nur die drei Methoden, die auf dem Weg durch welt_backen()/welt_laden()
    wirklich aufgerufen werden: get_all_parameters() (Parameter-Sicherung in
    core/welt_io.py), get_tab_parameters() (Meereshoehe/Seed im Godot-Export,
    gui/utils/map_export.py) und set_tab_parameters() (Parameter-
    Wiederherstellung beim Laden)."""

    def __init__(self):
        self.zurueckgeschriebene_tabs = []

    def set_tab_parameters(self, tab_name, parameter, validate=True,
                           notify_listeners=True):
        self.zurueckgeschriebene_tabs.append(tab_name)
        return True

    def get_tab_parameters(self, generator):
        if generator == "terrain":
            return {"map_seed": 20260922, "map_size": MAP_SIZE}
        if generator == "biome":
            return {"sea_level": 0.35}
        return {"beispiel_parameter": generator}

    def get_all_parameters(self):
        return {generator: self.get_tab_parameters(generator)
                for generator in KATEGORIEN}


def _baue_welt(rng):
    """Baut eine erfundene, aber in Form/Dtype realistische Welt - je
    Kategorie mindestens ein Feld, damit die Zusicherung "alle sieben kommen
    zurueck" ueberhaupt pruefbar ist."""
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
        # erosion war in der frueheren Fassung dieses Tests gar nicht
        # vertreten, weil collect_all_available_data() diese Kategorie nie
        # einsammelte. get_all_data("erosion") kann es - deshalb steht sie
        # hier jetzt mit drin.
        "erosion": {
            "erosion_map": rng.random((MAP_SIZE, MAP_SIZE)).astype(np.float32),
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


def _fuelle_manager(welt):
    """Legt die Testwelt ueber den echten Schreibweg im echten Manager ab."""
    dlm = DataLODManager()
    dlm.set_map_seed(20260922)
    dlm.set_map_distance_km(64.0)
    dlm.set_map_latitude(51.25)
    for kategorie, felder in welt.items():
        dlm.set_all_data(kategorie, felder, lod_level=TEST_LOD)
    return dlm


def check(label, bedingung, zusatz=""):
    print(f"[{'OK' if bedingung else 'FAIL'}] {label}"
          + (f" - {zusatz}" if zusatz else ""))
    return [] if bedingung else [f"{label}{' - ' + zusatz if zusatz else ''}"]


def lauf():
    fehler = []
    rng = np.random.default_rng(20260922)
    welt = _baue_welt(rng)

    quelle_dlm = _fuelle_manager(welt)
    fake_pm = FakeParameterManager()
    reiter = OverviewTab(quelle_dlm, fake_pm, None, None, None)

    tmp = tempfile.mkdtemp(prefix="welt_backen_test_")
    ziel = os.path.join(tmp, "meine_welt")
    try:
        # --- 1. Baken ---------------------------------------------------------
        manifest = reiter.welt_backen(ziel)
        fehler += check("Manifest wird zurueckgegeben",
                        isinstance(manifest, dict) and "kategorien" in manifest)
        fehler += check(f"{_WELT_MANIFEST_DATEI} liegt auf der Platte",
                        os.path.isfile(os.path.join(ziel, _WELT_MANIFEST_DATEI)))
        zustand_ordner = os.path.join(ziel, "zustand")
        zustand_dateien = sorted(os.listdir(zustand_ordner)) if os.path.isdir(zustand_ordner) else []
        fehler += check("zustand/ enthaelt je Kategorie eine Datei plus globals.json",
                        all(f"{k}.pkl" in zustand_dateien for k in KATEGORIEN)
                        and "globals.json" in zustand_dateien,
                        str(zustand_dateien))
        fehler += check("Manifest haelt fuer jede Kategorie das LOD fest",
                        all(manifest["kategorien"][k]["lod"] == TEST_LOD for k in KATEGORIEN),
                        str({k: manifest["kategorien"][k]["lod"] for k in KATEGORIEN}))
        fehler += check("lesbare Zustandsfassung (vorschau/zustand_lesbar.json) entsteht",
                        manifest.get("zustand_lesbar") is True
                        and os.path.isfile(os.path.join(ziel, "vorschau", "zustand_lesbar.json")),
                        str(manifest.get("zustand_lesbar")))

        godot_ordner = os.path.join(ziel, _WELT_GODOT_ORDNER)
        godot_dateien = (os.listdir(godot_ordner) if os.path.isdir(godot_ordner) else [])
        fehler += check("Godot-Ordner (Terrain3D-Export) entsteht und ist nicht leer",
                        len(godot_dateien) > 0, f"{godot_ordner}: {godot_dateien[:5]}")
        # Terrain3D-Bedarf konkret nachweisen (Akzeptanzkriterium 4): irgendwo
        # unter godot/ muss eine Hoehenkarte liegen (R16/EXR bzw. hier die
        # vorhandene 16-Bit-PNG-Variante), siehe docs/WELT_BACKEN.md
        # Abschnitt "Godot-Bedarf".
        hoehenkarte_gefunden = False
        for wurzel, _dirs, dateien in os.walk(godot_ordner):
            if any("hoehe" in d.lower() or "height" in d.lower() for d in dateien):
                hoehenkarte_gefunden = True
        fehler += check("Godot-Export enthaelt eine Hoehenkarte fuer Terrain3D",
                        hoehenkarte_gefunden, str(godot_dateien[:10]))

        # --- 2. Laden in einen FRISCHEN, echten Manager -----------------------
        # Frisch, weil eine geladene Welt eine neue Welt ist. In einen Manager
        # zu laden, der noch die alte haelt, muss LAUT scheitern - das ist
        # Pruefung 7 weiter unten.
        ziel_dlm = DataLODManager()
        reiter.data_lod_manager = ziel_dlm
        geladen = reiter.welt_laden(ziel)

        # --- 3. Bildpunktgenauer Rundgang jedes Arrays ------------------------
        array_felder = [
            ("terrain", "heightmap"), ("terrain", "slopemap"), ("terrain", "shadowmap"),
            ("geology", "rock_map"), ("geology", "hardness_map"),
            ("settlement", "civ_map"),
            ("weather", "temp_map"), ("weather", "precip_map"),
            ("erosion", "erosion_map"),
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

        # --- 4. Skalar und Siedlungsliste (Location-Dataclasses) --------------
        fehler += check("Skalar (ocean_outflow) kommt exakt zurueck",
                        geladen.get("water", {}).get("ocean_outflow") == 123.456)
        settlement_zurueck = geladen.get("settlement", {}).get("settlement_list")
        fehler += check("Siedlungsliste kommt mit 2 Eintraegen zurueck",
                        isinstance(settlement_zurueck, list) and len(settlement_zurueck) == 2)
        if settlement_zurueck:
            erster = settlement_zurueck[0]
            # Neu gegenueber der frueheren Fassung: es sind wieder ECHTE
            # Location-Objekte, keine dicts. Die alte Fassung schrieb sie ueber
            # dataclasses.asdict() nach JSON und konnte sie nicht zurueckbauen -
            # das Programm bekam Woerterbuecher, wo es Objekte erwartete.
            fehler += check("Siedlungen kommen als Location-Objekte zurueck, nicht als dict",
                            isinstance(erster, Location), type(erster).__name__)
            fehler += check("Location-Felder bleiben erhalten (location_id/culture/rank/house_count)",
                            getattr(erster, "location_id", None) == 1
                            and getattr(erster, "culture", None) == "samarcia"
                            and getattr(erster, "rank", None) == "stadt"
                            and getattr(erster, "house_count", None) == 41)

        # --- 5. Rueckschreiben in den lebenden Manager: ALLE SIEBEN -----------
        nicht_angekommen = [k for k in KATEGORIEN if not ziel_dlm.get_all_data(k)]
        fehler += check("alle sieben Kategorien sind im lebenden Manager angekommen",
                        not nicht_angekommen, str(nicht_angekommen))
        fehler += check("terrain.heightmap im Manager ist bildpunktgenau",
                        np.array_equal(ziel_dlm.get_terrain_data("heightmap"),
                                       welt["terrain"]["heightmap"]))
        fehler += check("weather.temp_map im Manager ist bildpunktgenau (frueher NICHT geschrieben)",
                        np.array_equal(ziel_dlm.get_weather_data("temp_map"),
                                       welt["weather"]["temp_map"]))
        fehler += check("water.water_map im Manager ist bildpunktgenau (frueher NICHT geschrieben)",
                        np.array_equal(ziel_dlm.get_water_data("water_map"),
                                       welt["water"]["water_map"]))
        fehler += check("biome.biome_map im Manager ist bildpunktgenau (frueher NICHT geschrieben)",
                        np.array_equal(ziel_dlm.get_biome_data("biome_map"),
                                       welt["biome"]["biome_map"]))
        fehler += check("settlement.settlement_list im Manager vorhanden (frueher NICHT geschrieben)",
                        len(ziel_dlm.get_settlement_data("settlement_list") or []) == 2)
        falsche_lods = {k: ziel_dlm.get_current_lod_level(k) for k in KATEGORIEN
                        if ziel_dlm.get_current_lod_level(k) != TEST_LOD}
        fehler += check("jede Kategorie liegt wieder auf ihrem urspruenglichen LOD",
                        not falsche_lods, str(falsche_lods))

        # --- 6. Das zusammengesetzte Terrain-Objekt ---------------------------
        # get_terrain_data("complete") ist der einzige der sieben
        # *_data_object-Schluessel, der irgendwo wieder gelesen wird (der
        # GenerationOrchestrator haengt daran). Ohne ihn haette eine geladene
        # Welt zwar heightmap und slopemap, aber "complete" waere None - genau
        # der halbe Zustand, den dieses Ticket verhindern soll.
        complete = ziel_dlm.get_terrain_data("complete")
        fehler += check("get_terrain_data('complete') ist nach dem Laden nicht None",
                        complete is not None)
        if complete is not None:
            fehler += check("das zusammengesetzte Terrain-Objekt traegt die geladene Hoehenkarte",
                            np.array_equal(getattr(complete, "heightmap", None),
                                           welt["terrain"]["heightmap"]))
            fehler += check("das zusammengesetzte Terrain-Objekt kennt sein LOD und seine Groesse",
                            getattr(complete, "lod_level", None) == TEST_LOD
                            and getattr(complete, "actual_size", None) == MAP_SIZE,
                            f"lod={getattr(complete, 'lod_level', None)}, "
                            f"size={getattr(complete, 'actual_size', None)}")

        # --- 7. Laden ueber einen belegten Manager muss LAUT scheitern --------
        # Der Manager haelt hier bereits eine andere Welt auf einem HOEHEREN
        # LOD. Ein pauschales Zurueckschreiben wuerde unterhalb davon landen:
        # gleiche Key-Namen, alte Daten, kein Fehler - die geladene Welt bliebe
        # unsichtbar. welt_laden() muss das bemerken.
        # Dieselben Keys wie die gespeicherte Welt, andere Werte, hoeheres LOD -
        # damit die Key-Mengen-Pruefung gerade NICHT anschlaegt und die
        # schaerfere Identitaetspruefung ("ist das wirklich das, was ich eben
        # geschrieben habe?") geprueft wird.
        belegt_dlm = DataLODManager()
        belegt_dlm.set_all_data("terrain", {
            feld: np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
            for feld in welt["terrain"]},
            lod_level=TEST_LOD + 2)
        reiter.data_lod_manager = belegt_dlm
        verschattung_laut = False
        verschattung_meldung = ""
        try:
            reiter.welt_laden(ziel)
        except WeltLadenFehler as e:
            verschattung_laut = True
            verschattung_meldung = str(e)
        fehler += check("Laden in einen bereits belegten Manager loest WeltLadenFehler aus",
                        verschattung_laut)
        fehler += check("die Meldung nennt die verschatteten Felder namentlich",
                        "heightmap" in verschattung_meldung
                        and "slopemap" in verschattung_meldung,
                        verschattung_meldung[:160])

        # --- 8. Kein stiller Ersatzwert: fehlende Kategorie = lauter Fehler ---
        reiter.data_lod_manager = DataLODManager()
        manifest_pfad = os.path.join(ziel, _WELT_MANIFEST_DATEI)
        with open(manifest_pfad, "r", encoding="utf-8") as f:
            kaputtes_manifest = json.load(f)
        kaputtes_manifest["kategorien"]["geology"]["keys"].append("erfundenes_feld")
        with open(manifest_pfad, "w", encoding="utf-8") as f:
            json.dump(kaputtes_manifest, f)

        ausnahme_kam = False
        ausnahme_nennt_feld = False
        try:
            reiter.welt_laden(ziel)
        except WeltLadenFehler as e:
            ausnahme_kam = True
            ausnahme_nennt_feld = "erfundenes_feld" in str(e)
        fehler += check("fehlendes Manifest-Feld loest WeltLadenFehler aus", ausnahme_kam)
        fehler += check("WeltLadenFehler nennt den fehlenden Feldnamen", ausnahme_nennt_feld)

        # --- 9. Fehlender Ordner insgesamt = lauter Fehler --------------------
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
