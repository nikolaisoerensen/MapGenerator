"""
Path: tests/smoke_test_regionsdaten_vollstaendig.py

Ticket #29: die neun Regionsparametersaetze leben jetzt in
core/daten/regionen.toml statt als Python-Literal in
core/terrain_weltkarte.py. Dieser Test prueft die Vollstaendigkeit dieser
Datei von AUSSEN, unabhaengig von der Ladefunktion selbst:

  1. Alle neun Regionen sind da, auf allen neun Gitterpositionen genau
     einmal (kein Loch, keine Dopplung).
  2. Jede Region hat jedes Pflichtfeld.
  3. Jede Region hat genau drei Kuestenarchetypen mit allen Pflichtfeldern.
  4. NIEDERSCHLAG_ZIEL und KLIMA_ZIEL (aus REGIONEN abgeleitet, siehe
     core/terrain_weltkarte.py) decken dieselben neun Namen ab.
  5. KEIN STILLER RUECKFALL: fehlt in core/daten/regionen.toml ein
     Pflichtfeld, muss der Ladevorgang selbst (_lade_regionsdaten()) mit
     einem lauten RuntimeError abbrechen statt mit einem Vorgabewert
     weiterzurechnen - negativ getestet mit einer absichtlich
     unvollstaendigen Kopie der Datei in einem Temp-Verzeichnis.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_regionsdaten_vollstaendig.py
"""

import sys
import tempfile
from pathlib import Path

_WURZEL = r"C:\Lokale Dateien\Projects\Python\MapGenerator\.claude\worktrees\agent-af1fe1d683e85a11c"
sys.path.insert(0, _WURZEL)

from core import terrain_weltkarte as rw

ERWARTETE_REGIONSNAMEN = {
    "Clonagh", "Skerrheim", "Morobora",
    "Estrande", "Nevadin", "Nebelrode",
    "Samarcia", "Macchia", "Thalassia",
}


def pruefe_neun_regionen_vollstaendig():
    gefunden = set()
    for zi, zeile in enumerate(rw.REGIONEN):
        assert len(zeile) == 3, (
            "Zeile %d hat %d Eintraege statt drei." % (zi, len(zeile)))
        for si, region in enumerate(zeile):
            assert region is not None, (
                "Gitterposition (%d, %d) ist leer." % (zi, si))
            for feld in rw._REGION_PFLICHTFELDER:
                assert feld in region, (
                    "Region %r fehlt Pflichtfeld %r." % (region.get("name"), feld))
            gefunden.add(region["name"])
    assert gefunden == ERWARTETE_REGIONSNAMEN, (
        "Regionsnamen weichen ab. Erwartet: %s, gefunden: %s"
        % (ERWARTETE_REGIONSNAMEN, gefunden))
    print("1. Neun Regionen vollstaendig, alle Pflichtfelder da: OK")


def pruefe_kuesten_archetypen_vollstaendig():
    assert set(rw.KUESTEN_ARCHETYPEN.keys()) == ERWARTETE_REGIONSNAMEN, (
        "KUESTEN_ARCHETYPEN deckt nicht alle neun Regionen ab: %s"
        % sorted(rw.KUESTEN_ARCHETYPEN.keys()))
    for name, archetypen in rw.KUESTEN_ARCHETYPEN.items():
        assert len(archetypen) == 3, (
            "Region %r hat %d Kuestenarchetypen statt drei." % (name, len(archetypen)))
        for archetyp in archetypen:
            for feld in rw._ARCHETYP_PFLICHTFELDER:
                assert feld in archetyp, (
                    "Ein Kuestenarchetyp von %r fehlt Feld %r." % (name, feld))
    print("2. 27 Kuestenarchetypen (9x3) vollstaendig: OK")


def pruefe_abgeleitete_ziele():
    assert set(rw.NIEDERSCHLAG_ZIEL.keys()) == ERWARTETE_REGIONSNAMEN, (
        "NIEDERSCHLAG_ZIEL deckt nicht alle neun Regionen ab.")
    assert set(rw.KLIMA_ZIEL.keys()) == ERWARTETE_REGIONSNAMEN, (
        "KLIMA_ZIEL deckt nicht alle neun Regionen ab.")
    print("3. NIEDERSCHLAG_ZIEL/KLIMA_ZIEL decken alle neun Regionen ab: OK")


def pruefe_lauter_fehlschlag_bei_fehlendem_feld():
    """
    TDD-Gegenprobe: eine Kopie von regionen.toml, der ein Pflichtfeld
    (hoehe_m der ersten Region) fehlt, muss _lade_regionsdaten() mit einem
    RuntimeError abbrechen lassen - nicht mit einem Vorgabewert weiterlaufen.
    """
    original_text = rw._REGIONENDATEI.read_text(encoding="utf-8")
    kaputt = original_text.replace("hoehe_m = 165.3", "# hoehe_m entfernt", 1)
    assert kaputt != original_text, (
        "Die Ersetzung hat nicht gegriffen - Testvorbereitung fehlerhaft.")

    with tempfile.TemporaryDirectory() as tmp:
        pfad = Path(tmp) / "regionen_kaputt.toml"
        pfad.write_text(kaputt, encoding="utf-8")
        try:
            rw._lade_regionsdaten(pfad)
        except RuntimeError as fehler:
            assert "hoehe_m" in str(fehler) or "Feld" in str(fehler), (
                "RuntimeError kam, nennt aber nicht das fehlende Feld: %s" % fehler)
            print("4. Fehlendes Pflichtfeld bricht laut ab (RuntimeError): OK")
            print("   Meldung: %s" % fehler)
        else:
            raise AssertionError(
                "_lade_regionsdaten() ist bei fehlendem hoehe_m NICHT "
                "fehlgeschlagen - das waere ein stiller Rueckfall auf einen "
                "Vorgabewert, genau das, was Ticket #29 verhindern soll.")


def pruefe_lauter_fehlschlag_bei_fehlender_datei():
    with tempfile.TemporaryDirectory() as tmp:
        pfad = Path(tmp) / "gibt_es_nicht.toml"
        try:
            rw._lade_regionsdaten(pfad)
        except RuntimeError:
            print("5. Fehlende Datei bricht laut ab (RuntimeError): OK")
        else:
            raise AssertionError(
                "_lade_regionsdaten() haette bei fehlender Datei abbrechen "
                "muessen.")


if __name__ == "__main__":
    pruefe_neun_regionen_vollstaendig()
    pruefe_kuesten_archetypen_vollstaendig()
    pruefe_abgeleitete_ziele()
    pruefe_lauter_fehlschlag_bei_fehlendem_feld()
    pruefe_lauter_fehlschlag_bei_fehlender_datei()
    print("\nAlle Pruefungen gruen.")
