"""
Path: tests/smoke_test_dateimenue_welt_io.py

Prueft die Verdrahtung aus Ticket #40 (gui/map_editor.py: "Welt oeffnen",
"Welt speichern", "Welt exportieren" im Datei-Menue rufen jetzt tatsaechlich
core/welt_io.py's welt_backen()/welt_laden() bzw. den bestehenden Godot-
Export auf, statt einer "wird in einer zukuenftigen Version umgesetzt"-
Attrappe).

KEIN LIVE-GUI-TEST: In dieser Umgebung gibt es kein Display. Dieses Skript
ruft die drei Handler-Methoden (_open_world, _save_world, _export_world)
UNGEBUNDEN auf einem minimalen Fake-"self" auf (Duck-Typing statt echtem
MapEditorWindow, dessen Konstruktor ein volles Qt-Fenster mit 8 Tabs
aufbaut). QFileDialog/QMessageBox/QInputDialog werden im Modul-Namensraum
von gui.map_editor durch Fakes ersetzt, damit kein echter Dialog (und keine
QApplication) noetig ist.

Die visuelle Live-Pruefung (speichern -> Programm schliessen -> oeffnen ->
dieselbe Welt ist da, am echten Fenster) MUSS zusaetzlich ein Mensch machen -
das steht auch am Ende der Ausgabe dieses Skripts.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_dateimenue_welt_io.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, ".")

import gui.map_editor as map_editor_module
from gui.map_editor import MapEditorWindow
from core.welt_io import KATEGORIEN, welt_backen, WeltBackenFehler, WeltLadenFehler


def check(label, bedingung, zusatz=""):
    status = "OK" if bedingung else "FAIL"
    print(f"[{status}] {label}{(' - ' + zusatz) if zusatz else ''}")
    return bool(bedingung)


# --------------------------------------------------------------------------
# Fakes: dieselbe minimale-Nachbau-Konvention wie
# tests/smoke_test_welt_io_roundtrip.py (FakeDLM dort), hier um
# clear_all_data() erweitert, weil _open_world() darueber den alten Zustand
# vor dem Laden wegwirft.
# --------------------------------------------------------------------------

class FakeDLM:
    def __init__(self):
        self._stores = {k: {} for k in KATEGORIEN}
        self._seed = 20260918
        self._distance_km = 128.0
        self._latitude = 47.5
        self.clear_all_data_calls = 0

    def get_all_data(self, category):
        return dict(self._stores[category])

    def set_all_data(self, category, data, lod_level=1, parameters=None):
        self._stores[category].update(data)

    def clear_all_data(self):
        self.clear_all_data_calls += 1
        for k in self._stores:
            self._stores[k].clear()

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


class FakeParameterManager:
    def __init__(self):
        self._params = {"terrain": {"map_seed": 20260918, "map_distance_km": 128.0}}

    def get_all_parameters(self):
        return dict(self._params)

    def get_tab_parameters(self, tab_name):
        return dict(self._params.get(tab_name, {}))

    def set_tab_parameters(self, tab_name, parameters, validate=False, notify_listeners=False):
        self._params[tab_name] = parameters
        return True


class FakeTab:
    def __init__(self, raises=False):
        self.update_calls = 0
        self.raises = raises

    def update_display_mode(self):
        self.update_calls += 1
        if self.raises:
            raise RuntimeError("Absichtlich kaputter Tab fuer den Fehlerpfad-Test")


class FakeLogger:
    def __init__(self):
        self.errors = []
        self.infos = []
        self.debugs = []
        self.warnings = []

    def error(self, msg):
        self.errors.append(msg)

    def info(self, msg):
        self.infos.append(msg)

    def debug(self, msg):
        self.debugs.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)


class FakeStatusIndicator:
    def __init__(self):
        self.last_success = None

    def set_success(self, msg):
        self.last_success = msg


class FakeSelf:
    """Duck-typed Ersatz fuer MapEditorWindow - nur die Attribute, die
    _open_world()/_save_world()/_export_world()/_clear_and_reset_all_generators()
    tatsaechlich benutzen."""

    def __init__(self, dlm=None, pm=None, tabs=None):
        self.data_lod_manager = dlm if dlm is not None else FakeDLM()
        self.parameter_manager = pm if pm is not None else FakeParameterManager()
        self.generation_orchestrator = None  # ueberspringt reset_lod_status()-Zweig
        self.tabs = tabs if tabs is not None else {}
        self.logger = FakeLogger()
        self.status_indicator = FakeStatusIndicator()
        self.active_generations = set()
        self.tab_generation_status = {}

    # echte Methode von MapEditorWindow, ungebunden aufgerufen
    _clear_and_reset_all_generators = MapEditorWindow._clear_and_reset_all_generators


# --------------------------------------------------------------------------
# Fakes fuer QFileDialog/QMessageBox/QInputDialog - ersetzen die Namen im
# Modul-Namensraum von gui.map_editor, damit kein echtes Qt-Fenster/keine
# QApplication noetig ist.
# --------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.criticals = []
        self.informations = []
        self.warnings = []


_rec = _Recorder()


class FakeQFileDialogOption:
    ShowDirsOnly = 1


class FakeQFileDialog:
    Option = FakeQFileDialogOption
    _next_dir = ""

    @staticmethod
    def getExistingDirectory(*args, **kwargs):
        return FakeQFileDialog._next_dir


class FakeQMessageBox:
    @staticmethod
    def critical(parent, title, text, *args, **kwargs):
        _rec.criticals.append((title, text))

    @staticmethod
    def information(parent, title, text, *args, **kwargs):
        _rec.informations.append((title, text))

    @staticmethod
    def warning(parent, title, text, *args, **kwargs):
        _rec.warnings.append((title, text))


class FakeQInputDialog:
    _next_result = ("Mapseed_test", True)

    @staticmethod
    def getText(*args, **kwargs):
        return FakeQInputDialog._next_result


def _patch_qt_fakes():
    map_editor_module.QFileDialog = FakeQFileDialog
    map_editor_module.QMessageBox = FakeQMessageBox
    map_editor_module.QInputDialog = FakeQInputDialog


def _reset_recorder():
    _rec.criticals.clear()
    _rec.informations.clear()
    _rec.warnings.clear()


def _baue_gefuellten_dlm():
    dlm = FakeDLM()
    rng = np.random.default_rng(7)
    dlm._stores["terrain"]["heightmap"] = rng.random((17, 19)).astype(np.float32)
    dlm._stores["biome"]["biome_map"] = rng.integers(0, 5, size=(17, 19)).astype(np.int32)
    return dlm


# --------------------------------------------------------------------------
# Testfaelle
# --------------------------------------------------------------------------

def run_save_world_ruft_welt_backen_auf():
    _reset_recorder()
    ordner = tempfile.mkdtemp(prefix="dateimenue_save_")
    try:
        dlm = _baue_gefuellten_dlm()
        fake_self = FakeSelf(dlm=dlm)
        FakeQFileDialog._next_dir = ordner

        MapEditorWindow._save_world(fake_self)

        ok = check("welt_manifest.json wurde tatsaechlich geschrieben (welt_backen lief)",
                    os.path.isfile(os.path.join(ordner, "welt_manifest.json")))
        ok &= check("Erfolg wurde als QMessageBox.information gemeldet",
                    len(_rec.informations) == 1, str(_rec.informations))
        ok &= check("keine QMessageBox.critical im Erfolgsfall",
                    len(_rec.criticals) == 0)
        ok &= check("status_indicator zeigt Erfolg", fake_self.status_indicator.last_success is not None)
        return ok
    finally:
        shutil.rmtree(ordner, ignore_errors=True)


def run_save_world_fehler_wird_als_dialog_gezeigt():
    """welt_backen() wirft WeltBackenFehler -> muss als QMessageBox.critical
    erscheinen, nicht als unbehandelte Exception (Abnahmekriterium: Fehler im
    Fenster, nicht nur im Log)."""
    _reset_recorder()

    def kaputtes_welt_backen(*args, **kwargs):
        raise WeltBackenFehler("Absichtlich ausgeloester Testfehler")

    original = map_editor_module.welt_backen
    map_editor_module.welt_backen = kaputtes_welt_backen
    try:
        fake_self = FakeSelf()
        FakeQFileDialog._next_dir = tempfile.gettempdir()

        try:
            MapEditorWindow._save_world(fake_self)
        except Exception as exc:
            return check("kein Absturz bei welt_backen()-Fehler", False, f"Exception entkam: {exc}")

        ok = check("Fehler erscheint als QMessageBox.critical", len(_rec.criticals) == 1, str(_rec.criticals))
        ok &= check("Fehlertext enthaelt die WeltBackenFehler-Meldung",
                    _rec.criticals and "Absichtlich ausgeloester Testfehler" in _rec.criticals[0][1])
        ok &= check("kein Erfolgsdialog im Fehlerfall", len(_rec.informations) == 0)
        return ok
    finally:
        map_editor_module.welt_backen = original


def run_save_world_abbruch_ohne_ordnerwahl_tut_nichts():
    _reset_recorder()
    fake_self = FakeSelf()
    FakeQFileDialog._next_dir = ""  # Nutzer bricht den Dialog ab

    MapEditorWindow._save_world(fake_self)

    return check("kein Dialog und kein welt_backen()-Aufruf bei abgebrochener Ordnerwahl",
                len(_rec.criticals) == 0 and len(_rec.informations) == 0)


def run_open_world_ruft_welt_laden_auf_und_aktualisiert_tabs():
    _reset_recorder()
    ordner = tempfile.mkdtemp(prefix="dateimenue_open_")
    try:
        quelle = _baue_gefuellten_dlm()
        welt_backen(ordner, quelle, parameter_manager=None)

        ziel_dlm = FakeDLM()
        tab_a, tab_b = FakeTab(), FakeTab()
        fake_self = FakeSelf(dlm=ziel_dlm, tabs={"terrain": tab_a, "biome": tab_b})
        FakeQFileDialog._next_dir = ordner

        MapEditorWindow._open_world(fake_self)

        ok = check("data_lod_manager wurde vor dem Laden geleert (_clear_and_reset_all_generators)",
                    ziel_dlm.clear_all_data_calls == 1)
        ok &= check("heightmap kam nach dem Laden tatsaechlich im Ziel-Manager an",
                    np.array_equal(ziel_dlm._stores["terrain"]["heightmap"],
                                    quelle._stores["terrain"]["heightmap"]))
        ok &= check("Tab 'terrain' wurde nach dem Laden zum Redraw aufgefordert (2D+3D via update_display_mode)",
                    tab_a.update_calls == 1)
        ok &= check("Tab 'biome' wurde ebenfalls aufgefordert",
                    tab_b.update_calls == 1)
        ok &= check("Erfolg wurde als QMessageBox.information gemeldet",
                    len(_rec.informations) == 1, str(_rec.informations))
        ok &= check("keine QMessageBox.critical im Erfolgsfall", len(_rec.criticals) == 0)
        return ok
    finally:
        shutil.rmtree(ordner, ignore_errors=True)


def run_open_world_fehlender_ordner_zeigt_dialog_ohne_daten_zu_loeschen():
    """Kein welt_manifest.json im gewaehlten Ordner -> muss als Dialog
    erscheinen, UND darf den aktuellen Arbeitsstand NICHT vorher wegwerfen
    (siehe Docstring von _open_world: die Vorab-Pruefung existiert genau
    dafuer)."""
    _reset_recorder()
    leerer_ordner = tempfile.mkdtemp(prefix="dateimenue_open_leer_")
    try:
        dlm = FakeDLM()
        dlm._stores["terrain"]["heightmap"] = np.ones((3, 3), dtype=np.float32)
        fake_self = FakeSelf(dlm=dlm)
        FakeQFileDialog._next_dir = leerer_ordner

        MapEditorWindow._open_world(fake_self)

        ok = check("fehlendes Manifest erscheint als QMessageBox.critical",
                    len(_rec.criticals) == 1, str(_rec.criticals))
        ok &= check("data_lod_manager wurde NICHT geleert (Vorab-Pruefung griff vor dem Loeschen)",
                    dlm.clear_all_data_calls == 0)
        ok &= check("alte Daten sind noch da",
                    "heightmap" in dlm._stores["terrain"])
        return ok
    finally:
        shutil.rmtree(leerer_ordner, ignore_errors=True)


def run_open_world_welt_laden_fehler_wird_als_dialog_gezeigt():
    """welt_laden() wirft WeltLadenFehler (z.B. beschaedigter Weltordner) ->
    muss im Fenster erscheinen, kein Absturz."""
    _reset_recorder()
    ordner = tempfile.mkdtemp(prefix="dateimenue_open_kaputt_")
    try:
        quelle = _baue_gefuellten_dlm()
        welt_backen(ordner, quelle, parameter_manager=None)
        # Manifest verspricht "terrain", die Datei fehlt danach -> WeltLadenFehler
        os.remove(os.path.join(ordner, "zustand", "terrain.pkl"))

        fake_self = FakeSelf()
        FakeQFileDialog._next_dir = ordner

        try:
            MapEditorWindow._open_world(fake_self)
        except Exception as exc:
            return check("kein Absturz bei welt_laden()-Fehler", False, f"Exception entkam: {exc}")

        ok = check("WeltLadenFehler erscheint als QMessageBox.critical",
                    len(_rec.criticals) == 1, str(_rec.criticals))
        ok &= check("kein Erfolgsdialog im Fehlerfall", len(_rec.informations) == 0)
        return ok
    finally:
        shutil.rmtree(ordner, ignore_errors=True)


def run_export_world_ruft_export_all_layers_auf():
    _reset_recorder()
    ordner = tempfile.mkdtemp(prefix="dateimenue_export_")
    aufrufe = []

    def fake_export_all_layers(dlm, pm, output_root, filename_prefix):
        aufrufe.append((output_root, filename_prefix))
        ziel = os.path.join(output_root, filename_prefix)
        os.makedirs(ziel, exist_ok=True)
        return True, "2 Layer exportiert", ziel

    import gui.utils.map_export as map_export_module
    original = map_export_module.export_all_layers
    map_export_module.export_all_layers = fake_export_all_layers
    try:
        fake_self = FakeSelf()
        FakeQFileDialog._next_dir = ordner
        FakeQInputDialog._next_result = ("Mapseed_test", True)

        MapEditorWindow._export_world(fake_self)

        ok = check("export_all_layers wurde mit dem gewaehlten Ordner+Praefix aufgerufen",
                    aufrufe == [(ordner, "Mapseed_test")], str(aufrufe))
        ok &= check("Erfolg wurde als QMessageBox.information gemeldet",
                    len(_rec.informations) == 1, str(_rec.informations))
        return ok
    finally:
        map_export_module.export_all_layers = original
        shutil.rmtree(ordner, ignore_errors=True)


def run_export_world_misserfolg_zeigt_warnung_kein_absturz():
    _reset_recorder()

    def fake_export_all_layers(dlm, pm, output_root, filename_prefix):
        return False, "Kein Layer verfuegbar - noch nichts generiert", None

    import gui.utils.map_export as map_export_module
    original = map_export_module.export_all_layers
    map_export_module.export_all_layers = fake_export_all_layers
    try:
        fake_self = FakeSelf()
        FakeQFileDialog._next_dir = tempfile.gettempdir()
        FakeQInputDialog._next_result = ("Mapseed_test", True)

        MapEditorWindow._export_world(fake_self)

        return check("Misserfolg erscheint als QMessageBox.warning (kein Absturz, kein Erfolgsdialog)",
                    len(_rec.warnings) == 1 and len(_rec.informations) == 0, str(_rec.warnings))
    finally:
        map_export_module.export_all_layers = original


def run_export_world_abbruch_bei_inputdialog_tut_nichts():
    _reset_recorder()
    fake_self = FakeSelf()
    FakeQFileDialog._next_dir = tempfile.gettempdir()
    FakeQInputDialog._next_result = ("", False)  # Nutzer bricht ab

    MapEditorWindow._export_world(fake_self)

    return check("kein Dialog bei abgebrochener Praefix-Eingabe",
                len(_rec.criticals) == 0 and len(_rec.informations) == 0 and len(_rec.warnings) == 0)


def main():
    _patch_qt_fakes()

    print("=" * 78)
    print("Datei-Menue <-> welt_backen()/welt_laden() Verdrahtung (Ticket #40)")
    print("=" * 78)

    faelle = [
        ("Save World ruft welt_backen() auf", run_save_world_ruft_welt_backen_auf),
        ("Save World: Fehler als Dialog", run_save_world_fehler_wird_als_dialog_gezeigt),
        ("Save World: Abbruch tut nichts", run_save_world_abbruch_ohne_ordnerwahl_tut_nichts),
        ("Open World ruft welt_laden() auf + aktualisiert Tabs",
         run_open_world_ruft_welt_laden_auf_und_aktualisiert_tabs),
        ("Open World: fehlender Ordner zeigt Dialog, loescht nichts",
         run_open_world_fehlender_ordner_zeigt_dialog_ohne_daten_zu_loeschen),
        ("Open World: welt_laden()-Fehler als Dialog",
         run_open_world_welt_laden_fehler_wird_als_dialog_gezeigt),
        ("Export World ruft export_all_layers() auf", run_export_world_ruft_export_all_layers_auf),
        ("Export World: Misserfolg als Warnung", run_export_world_misserfolg_zeigt_warnung_kein_absturz),
        ("Export World: Abbruch tut nichts", run_export_world_abbruch_bei_inputdialog_tut_nichts),
    ]

    ergebnisse = []
    for name, funktion in faelle:
        print(f"\n--- {name} ---")
        ergebnisse.append(funktion())

    print("\n" + "=" * 78)
    fehlend = ergebnisse.count(False)
    print(f"{len(ergebnisse) - fehlend}/{len(ergebnisse)} Gruppen gruen")
    print("=" * 78)
    print(
        "WICHTIG: Dies ist ein HEADLESS-Test der Handler-Logik (kein echtes "
        "Fenster, kein Display in dieser Umgebung). Die visuelle Live-"
        "Pruefung - speichern, Programm schliessen, oeffnen, dieselbe Welt "
        "ist da - hat ein Mensch NOCH NICHT gemacht und muss sie am "
        "laufenden Programm nachholen."
    )
    return 0 if fehlend == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
