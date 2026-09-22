"""
Path: tests/smoke_test_welt_io_roundtrip.py

Prueft die Naht aus Ticket #38 (core/welt_io.py: welt_backen()/welt_laden()).

DIE ZUSICHERUNG, AUF DIE ES ANKOMMT

welt_backen() -> welt_laden() muss BITGENAU sein: jedes Array kommt exakt so
zurueck, wie es hineinging (gleicher dtype, gleiche Werte, kein
Rundungsfehler). Ausserdem darf ein fehlendes Pflichtfeld beim Laden NIEMALS
lautlos durch nichts ersetzt werden (Ticket #54) - ein absichtlich
beschaedigter Weltordner muss WeltLadenFehler werfen, nicht ein
unvollstaendiges Ergebnis liefern.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_welt_io_roundtrip.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, ".")

from core.welt_io import KATEGORIEN, WeltLadenFehler, welt_backen, welt_laden


def check(label, bedingung, zusatz=""):
    status = "OK" if bedingung else "FAIL"
    print(f"[{status}] {label}{(' - ' + zusatz) if zusatz else ''}")
    return bool(bedingung)


class FakeDLM:
    """Minimaler Nachbau der DataLODManager-Schnittstelle, die welt_io.py
    tatsaechlich benutzt: get_all_data()/set_all_data()/get_current_lod_level()
    je Kategorie plus die drei globalen Getter/Setter. Kein Qt, keine
    Generator-Objekte - genau so viel, wie zum Pruefen des Rundlaufs noetig ist
    (Projektkonvention, siehe tests/smoke_test_export_2048.py).

    get_current_lod_level() bildet den echten Manager darin nach, dass ein
    Schreibvorgang das aktuelle LOD auf das hoechste je geschriebene hebt -
    genau darauf stuetzt sich welt_backen(), wenn es das LOD ins Manifest
    schreibt, und welt_laden(), wenn es dieselbe Stufe wiederherstellt."""

    def __init__(self):
        self._stores = {k: {} for k in KATEGORIEN}
        self._lods = {k: 0 for k in KATEGORIEN}
        self._seed = 20260918
        self._distance_km = 128.0
        self._latitude = 47.5

    def get_all_data(self, category):
        return dict(self._stores[category])

    def set_all_data(self, category, data, lod_level=1, parameters=None):
        self._stores[category].update(data)
        self._lods[category] = max(self._lods[category], int(lod_level))

    def get_current_lod_level(self, category):
        return self._lods[category]

    def get_map_seed(self):
        return self._seed

    def set_map_seed(self, value):
        self._seed = value

    def get_map_distance_km(self):
        return self._distance_km

    def set_map_distance_km(self, value):
        self._distance_km = value

    def get_map_latitude(self):
        return self._latitude

    def set_map_latitude(self, value):
        self._latitude = value


# Eine ECHTE Kartengroesse (CLAUDE.md: "Tests mit den ECHTEN Eingabegroessen
# bauen" - 128/256/512/1024). Eine ausgedachte Groesse hat in diesem Projekt
# schon einmal wochenlang eine tote Funktion gruen aussehen lassen.
MAP_SIZE = 128

# Das LOD, auf dem die Testwelt liegt. Bewusst NICHT 1: nur so faellt auf, wenn
# welt_laden() pauschal auf LOD 1 zurueckschriebe - die Welt laege dann
# unterhalb des aktuellen Standes und waere fuer jeden Leser unsichtbar.
TEST_LOD = 3


def _baue_gefuellten_dlm():
    dlm = FakeDLM()
    rng = np.random.default_rng(42)
    # float32-Heightmap mit Werten, die bei verlustbehafteter Rundung (z.B.
    # 16-Bit-Quantisierung) garantiert nicht bitgenau blieben - das macht den
    # Test scharf gegen eine versehentliche Wiederverwendung des
    # verlustbehafteten Godot-Exportpfads fuer den Rundlauf selbst.
    dlm.set_all_data("terrain", {
        "heightmap": (rng.random((MAP_SIZE, MAP_SIZE), dtype=np.float64)
                      .astype(np.float32) * 1000.0 - 137.123456),
        "slopemap": rng.random((MAP_SIZE, MAP_SIZE)).astype(np.float32),
        # Nicht-quadratisch und damit empfindlich gegen ein versehentliches
        # Transponieren, das bei quadratischen Karten unbemerkt bliebe.
        "calculated_sun_angles": rng.random((MAP_SIZE, 3)).astype(np.float64),
    }, lod_level=TEST_LOD)
    dlm.set_all_data("biome", {
        "biome_map": rng.integers(0, 9, size=(MAP_SIZE, MAP_SIZE)).astype(np.int32),
    }, lod_level=TEST_LOD)
    dlm.set_all_data("settlement", {
        "settlement_list": [{"name": "Stadt A", "pos": (3, 4)},
                            {"name": "Stadt B", "pos": (10, 2)}],
    }, lod_level=TEST_LOD)
    dlm.set_all_data("water", {"ocean_outflow": 12345.6789}, lod_level=TEST_LOD)
    return dlm


def run_rundlauf_bitgenau():
    quelle = _baue_gefuellten_dlm()
    ordner = tempfile.mkdtemp(prefix="welt_io_test_")
    try:
        manifest = welt_backen(ordner, quelle, parameter_manager=None)
        ok = check("welt_backen liefert ein Manifest", isinstance(manifest, dict))
        ok &= check("Manifest listet 'terrain' als vorhanden",
                    manifest["kategorien"]["terrain"]["vorhanden"])
        ok &= check("zustand/ wurde angelegt",
                    os.path.isdir(os.path.join(ordner, "zustand")))

        ziel = FakeDLM()
        geladenes_manifest = welt_laden(ordner, ziel, parameter_manager=None)
        ok &= check("welt_laden liefert dasselbe Manifest zurueck",
                    geladenes_manifest["gebacken_am"] == manifest["gebacken_am"])

        h_quelle = quelle._stores["terrain"]["heightmap"]
        h_ziel = ziel._stores["terrain"]["heightmap"]
        ok &= check("heightmap ist bitgenau identisch (Werte)",
                    np.array_equal(h_quelle, h_ziel))
        ok &= check("heightmap behaelt exakten dtype",
                    h_ziel.dtype == h_quelle.dtype, f"{h_ziel.dtype}")

        ok &= check("Manifest haelt das LOD der Kategorie fest",
                    manifest["kategorien"]["terrain"]["lod"] == TEST_LOD,
                    f"lod={manifest['kategorien']['terrain']['lod']}")
        ok &= check("welt_laden stellt dasselbe LOD wieder her (kein Rueckfall auf 1)",
                    ziel.get_current_lod_level("terrain") == TEST_LOD,
                    f"lod={ziel.get_current_lod_level('terrain')}")

        sun_quelle = quelle._stores["terrain"]["calculated_sun_angles"]
        sun_ziel = ziel._stores["terrain"]["calculated_sun_angles"]
        ok &= check("nicht-quadratisches Array behaelt Form und Werte",
                    sun_ziel.shape == sun_quelle.shape
                    and np.array_equal(sun_quelle, sun_ziel),
                    f"{sun_ziel.shape}")

        s_quelle = quelle._stores["terrain"]["slopemap"]
        s_ziel = ziel._stores["terrain"]["slopemap"]
        ok &= check("slopemap ist bitgenau identisch", np.array_equal(s_quelle, s_ziel))

        b_quelle = quelle._stores["biome"]["biome_map"]
        b_ziel = ziel._stores["biome"]["biome_map"]
        ok &= check("biome_map (kategorisch, int32) ist bitgenau identisch",
                    np.array_equal(b_quelle, b_ziel) and b_ziel.dtype == b_quelle.dtype)

        ok &= check("settlement_list (Liste von dicts, kein Array) kommt identisch zurueck",
                    ziel._stores["settlement"]["settlement_list"]
                    == quelle._stores["settlement"]["settlement_list"])

        ok &= check("ocean_outflow (Skalar) kommt bitgenau zurueck",
                    ziel._stores["water"]["ocean_outflow"]
                    == quelle._stores["water"]["ocean_outflow"])

        ok &= check("map_seed rundlauffest", ziel.get_map_seed() == quelle.get_map_seed())
        ok &= check("map_distance_km rundlauffest",
                    ziel.get_map_distance_km() == quelle.get_map_distance_km())
        ok &= check("map_latitude rundlauffest",
                    ziel.get_map_latitude() == quelle.get_map_latitude())

        ok &= check("godot/-Export wurde versucht (Manifest-Eintrag vorhanden)",
                    manifest.get("godot") is not None)
        ok &= check("vorschau/-Export wurde versucht (Manifest-Eintrag vorhanden)",
                    manifest.get("vorschau") is not None)
        return ok
    finally:
        shutil.rmtree(ordner, ignore_errors=True)


def run_fehlende_datei_ist_laut():
    """Ticket #54: ein Manifest, das eine Kategorie verspricht, deren Datei
    aber fehlt, muss WeltLadenFehler werfen - nicht mit leeren/None-Werten
    weiterlaufen."""
    quelle = _baue_gefuellten_dlm()
    ordner = tempfile.mkdtemp(prefix="welt_io_test_kaputt_")
    try:
        welt_backen(ordner, quelle, parameter_manager=None)
        kaputte_datei = os.path.join(ordner, "zustand", "terrain.pkl")
        ok = check("Terrain-Zustandsdatei existiert vor dem Kaputtmachen",
                    os.path.isfile(kaputte_datei))
        os.remove(kaputte_datei)

        try:
            welt_laden(ordner, FakeDLM(), parameter_manager=None)
            ok &= check("welt_laden wirft WeltLadenFehler bei fehlender Datei", False,
                        "keine Exception geworfen")
        except WeltLadenFehler as exc:
            ok &= check("welt_laden wirft WeltLadenFehler bei fehlender Datei", True, str(exc))
        return ok
    finally:
        shutil.rmtree(ordner, ignore_errors=True)


def run_fehlendes_manifest_ist_laut():
    ordner = tempfile.mkdtemp(prefix="welt_io_test_leer_")
    try:
        try:
            welt_laden(ordner, FakeDLM(), parameter_manager=None)
            return check("welt_laden wirft bei fehlendem Manifest", False,
                        "keine Exception geworfen")
        except WeltLadenFehler as exc:
            return check("welt_laden wirft bei fehlendem Manifest", True, str(exc))
    finally:
        shutil.rmtree(ordner, ignore_errors=True)


def run_leere_welt_ist_kein_fehler():
    """Eine frisch begonnene Welt, bei der noch nichts generiert wurde, ist
    ein gueltiger, wenn auch leerer Zustand - kein Fehler."""
    dlm = FakeDLM()
    ordner = tempfile.mkdtemp(prefix="welt_io_test_leerwelt_")
    try:
        manifest = welt_backen(ordner, dlm, parameter_manager=None)
        ok = check("leere Welt wird ohne Fehler gebacken", isinstance(manifest, dict))
        ok &= check("keine Kategorie ist als vorhanden markiert",
                    all(not v["vorhanden"] for v in manifest["kategorien"].values()))
        geladen = welt_laden(ordner, FakeDLM(), parameter_manager=None)
        ok &= check("leere Welt kann ohne Fehler geladen werden", isinstance(geladen, dict))
        return ok
    finally:
        shutil.rmtree(ordner, ignore_errors=True)


def main():
    print("=" * 70)
    print("welt_backen()/welt_laden() - Rundlauf und laute Fehler (Ticket #38)")
    print("=" * 70)
    ergebnisse = []
    for name, funktion in [("Bitgenauer Rundlauf", run_rundlauf_bitgenau),
                           ("Fehlende Zustandsdatei ist laut", run_fehlende_datei_ist_laut),
                           ("Fehlendes Manifest ist laut", run_fehlendes_manifest_ist_laut),
                           ("Leere Welt ist kein Fehler", run_leere_welt_ist_kein_fehler)]:
        print(f"\n--- {name} ---")
        ergebnisse.append(funktion())
    print("\n" + "=" * 70)
    fehlend = ergebnisse.count(False)
    print(f"{len(ergebnisse) - fehlend}/{len(ergebnisse)} Gruppen gruen")
    return 0 if fehlend == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
