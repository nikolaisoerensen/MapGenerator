"""
Path: tests/smoke_test_dokumentbehauptungen.py

Prueft harte Behauptungen in docs/ gegen den echten Code (Ticket #53).

Anlass: der Testbericht behauptete "die Erosionskette ist abgeschaltet",
waehrend gui/config/value_default.py EROSION_AKTIV = True stehen hatte -
niemand hat es bemerkt, weil nichts es geprueft hat. Aehnlich beim
Knotenzaehler: mehrere Dokumente nennen 39 Rechenknoten im Calculator-Graph,
andere 38 - beides unmoeglich zugleich richtig, und wieder hat es nichts
geprueft.

Format ("PRUEFBAR"-Markup)
---------------------------
Eine harte Behauptung (eine Zahl, ein Dateiname, ein Schalterzustand, eine
Anzahl - keine Meinung, keine Einschaetzung) wird in einer Markdown-Datei
unter docs/ mit einer eigenen Zeile UNMITTELBAR VOR der Behauptung
ausgezeichnet, als HTML-Kommentar (unsichtbar beim Rendern):

    <!-- PRUEFBAR: pfad=gui/config/value_default.py:1088 attribut=EROSION_AKTIV erwartet=True -->
    ... EROSION_AKTIV = True ...

    <!-- PRUEFBAR: ausdruck=len(managers.calculator_graph.CALCULATOR_GRAPH) erwartet=38 -->
    ... 38 Knoten ...

Zwei Formen, unterscheidbar am Schluessel:

1. `pfad=<Datei relativ zur Projektwurzel>[:<Zeile>] attribut=<Name> erwartet=<Wert>`
   Die Datei wird als Python-Modul importiert (der Pfad muss darum ein
   importierbares .py-Modul sein) und `getattr(modul, attribut)` gegen
   `erwartet` verglichen. Die Zeilennummer nach dem Doppelpunkt ist rein
   informativ fuer Menschen, die die Stelle nachschlagen wollen - fuer die
   Pruefung selbst zaehlt nur der Attributname, damit sich verschobene
   Zeilennummern nicht als Fehlschlag tarnen und ein wirklich verschwundenes
   Attribut sich nicht hinter einer falschen Zeilennummer versteckt.

2. `ausdruck=<Ausdruck> erwartet=<Wert>`
   Fuer Behauptungen, die kein einzelnes Attribut sind (z.B. "Anzahl der
   Eintraege in einem Dict"). Aus Sicherheitsgruenden gibt es KEIN freies
   eval() - `ausdruck` muss woertlich in ALLOWLIST_AUSDRUECKE (unten in
   dieser Datei) stehen. Ein nicht gelisteter Ausdruck ist ein Fehlschlag,
   kein stilles Uebergehen - neue Ausdruecke werden hier eingetragen.

`erwartet` wird nach Moeglichkeit mit ast.literal_eval gelesen (True/False/
Zahlen/'Text in Anfuehrungszeichen'), sonst als reine Zeichenkette verglichen.

Was bei einem Fehlschlag passiert
----------------------------------
Jede gepruefte Behauptung, die nicht (mehr) stimmt, fuehrt zu einem
Fehlschlag mit SOLL und IST. Verschwindet die Codestelle selbst (Datei nicht
mehr da, Modul importiert nicht, Attribut nicht mehr vorhanden, Ausdruck
nicht mehr in der Allowlist), ist das ebenfalls ein Fehlschlag - kein
try/except, das den Fall verschluckt. Genau dieses stille Verschlucken ist
in CLAUDE.md als der wiederkehrende Fehler dieses Projekts beschrieben.

Laufzeit: liest nur Textdateien und importiert zwei kleine, PyQt-freie
Module (value_default.py, calculator_graph.py) - keine Pipeline, keine GPU.
Gehoert deshalb in die Waechter-/Schnellreihe.
"""

import ast
import importlib
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = PROJECT_ROOT / "docs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Ausdruecke, die "ausdruck=" verwenden duerfen. Bewusst eine Allowlist statt
# eval() auf beliebigen Text - siehe Modul-Docstring. Jeder Eintrag ist eine
# Funktion ohne Argumente, die den IST-Wert liefert oder eine Exception wirft
# (die Exception wird vom Aufrufer als Fehlschlag gewertet, nicht verschluckt).
def _anzahl_calculator_knoten():
    modul = importlib.import_module("managers.calculator_graph")
    return len(modul.CALCULATOR_GRAPH)


ALLOWLIST_AUSDRUECKE = {
    "len(managers.calculator_graph.CALCULATOR_GRAPH)": _anzahl_calculator_knoten,
}


PRUEFBAR_MUSTER = re.compile(r"<!--\s*PRUEFBAR:\s*(?P<rumpf>.*?)\s*-->")
SCHLUESSEL_WERT_MUSTER = re.compile(r"(\w+)=(\S+)")


class Behauptung:
    def __init__(self, datei, zeile, rumpf, felder):
        self.datei = datei
        self.zeile = zeile
        self.rumpf = rumpf
        self.felder = felder

    def ort(self):
        return "%s:%d" % (self.datei, self.zeile)


def _parse_erwartet(text):
    """Liest 'erwartet=...' moeglichst als echten Python-Wert, sonst als String."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _finde_behauptungen(pfad):
    text = pfad.read_text(encoding="utf-8")
    rel = pfad.relative_to(PROJECT_ROOT).as_posix()
    ergebnisse = []
    for treffer in PRUEFBAR_MUSTER.finditer(text):
        rumpf = treffer.group("rumpf")
        zeile = text.count("\n", 0, treffer.start()) + 1
        felder = dict(SCHLUESSEL_WERT_MUSTER.findall(rumpf))
        ergebnisse.append(Behauptung(rel, zeile, rumpf, felder))
    return ergebnisse


def _pruefe_attribut(behauptung):
    """Form 1: pfad=...[:zeile] attribut=... erwartet=... . Gibt (ok, meldung) zurueck."""
    felder = behauptung.felder
    roh_pfad = felder.get("pfad", "")
    attribut = felder.get("attribut")
    erwartet_roh = felder.get("erwartet")
    if not roh_pfad or not attribut or erwartet_roh is None:
        return False, ("unvollstaendige PRUEFBAR-Angabe (braucht pfad=, attribut=, "
                        "erwartet=): %r" % behauptung.rumpf)

    # Optionale ":zeile" abtrennen - rein informativ, siehe Modul-Docstring.
    zielpfad_teil = roh_pfad.rsplit(":", 1)
    if len(zielpfad_teil) == 2 and zielpfad_teil[1].isdigit():
        rel_datei = zielpfad_teil[0]
    else:
        rel_datei = roh_pfad

    ziel_datei = PROJECT_ROOT / rel_datei
    if not ziel_datei.is_file():
        return False, ("Datei nicht gefunden: %s (behauptet in %s)"
                        % (rel_datei, behauptung.ort()))
    if ziel_datei.suffix != ".py":
        return False, ("pfad= muss auf ein .py-Modul zeigen, ist aber %s (in %s)"
                        % (rel_datei, behauptung.ort()))

    modulpfad = ziel_datei.relative_to(PROJECT_ROOT).with_suffix("")
    modulname = ".".join(modulpfad.parts)
    try:
        modul = importlib.import_module(modulname)
    except Exception as fehler:  # Import-Fehlschlag ist ein echter Fehlschlag, kein Skip.
        return False, ("Modul %s (aus %s) laesst sich nicht importieren: %s"
                        % (modulname, behauptung.ort(), fehler))

    if not hasattr(modul, attribut):
        return False, ("Attribut '%s' existiert nicht mehr in %s (behauptet in %s)."
                        " Soll: vorhanden mit Wert %s. Ist: Attribut fehlt."
                        % (attribut, modulname, behauptung.ort(), erwartet_roh))

    ist_wert = getattr(modul, attribut)
    erwartet_wert = _parse_erwartet(erwartet_roh)
    if ist_wert == erwartet_wert or str(ist_wert) == str(erwartet_wert):
        return True, "%s.%s == %r (ok)" % (modulname, attribut, ist_wert)
    return False, ("%s.%s: Soll (laut %s) = %r, Ist = %r"
                    % (modulname, attribut, behauptung.ort(), erwartet_wert, ist_wert))


def _pruefe_ausdruck(behauptung):
    """Form 2: ausdruck=... erwartet=... . Gibt (ok, meldung) zurueck."""
    felder = behauptung.felder
    ausdruck = felder.get("ausdruck")
    erwartet_roh = felder.get("erwartet")
    if not ausdruck or erwartet_roh is None:
        return False, ("unvollstaendige PRUEFBAR-Angabe (braucht ausdruck=, "
                        "erwartet=): %r" % behauptung.rumpf)

    funktion = ALLOWLIST_AUSDRUECKE.get(ausdruck)
    if funktion is None:
        return False, ("Ausdruck nicht in ALLOWLIST_AUSDRUECKE: %r (behauptet in %s)."
                        " Neue Ausdruecke muessen in "
                        "tests/smoke_test_dokumentbehauptungen.py eingetragen werden -"
                        " kein freies eval() erlaubt." % (ausdruck, behauptung.ort()))

    try:
        ist_wert = funktion()
    except Exception as fehler:
        return False, ("Ausdruck '%s' (behauptet in %s) wirft beim Auswerten: %s"
                        % (ausdruck, behauptung.ort(), fehler))

    erwartet_wert = _parse_erwartet(erwartet_roh)
    if ist_wert == erwartet_wert:
        return True, "%s == %r (ok)" % (ausdruck, ist_wert)
    return False, ("%s: Soll (laut %s) = %r, Ist = %r"
                    % (ausdruck, behauptung.ort(), erwartet_wert, ist_wert))


def pruefe_behauptung(behauptung):
    if "attribut" in behauptung.felder:
        return _pruefe_attribut(behauptung)
    if "ausdruck" in behauptung.felder:
        return _pruefe_ausdruck(behauptung)
    return False, ("unbekannte PRUEFBAR-Form (weder attribut= noch ausdruck=) in %s: %r"
                    % (behauptung.ort(), behauptung.rumpf))


def run_behauptungen_stimmen_gegen_code():
    print("\n--- Harte Dokumentbehauptungen gegen den echten Code (Ticket #53) ---")
    if not DOCS_ROOT.is_dir():
        print("  FEHLER: docs/ nicht gefunden unter %s" % DOCS_ROOT)
        return False

    alle_behauptungen = []
    for pfad in sorted(DOCS_ROOT.rglob("*.md")):
        alle_behauptungen.extend(_finde_behauptungen(pfad))

    print("  Gefundene PRUEFBAR-Markierungen: %d" % len(alle_behauptungen))
    if not alle_behauptungen:
        print("  Keine einzige Markierung gefunden - das ist fuer sich genommen kein"
              " Fehlschlag, aber ohne mindestens Erosionsschalter und Knotenzahl"
              " waere Ticket #53 nicht erfuellt.")

    ok = True
    fehlschlaege = []
    for behauptung in alle_behauptungen:
        erfolg, meldung = pruefe_behauptung(behauptung)
        praefix = "  ok " if erfolg else "  FEHLSCHLAG "
        print("%s%-70s %s" % (praefix, behauptung.ort(), meldung))
        ok = ok and erfolg
        if not erfolg:
            fehlschlaege.append("%s: %s" % (behauptung.ort(), meldung))

    if fehlschlaege:
        print("\n  %d von %d Behauptungen stimmen NICHT mit dem Code ueberein:"
              % (len(fehlschlaege), len(alle_behauptungen)))
        for f in fehlschlaege:
            print("    - %s" % f)
    return ok


if __name__ == "__main__":
    ergebnisse = {
        "behauptungen_stimmen_gegen_code": run_behauptungen_stimmen_gegen_code(),
    }
    print("\n=== SUMMARY ===")
    for name, ergebnis in ergebnisse.items():
        print("%-36s %s" % (name, "PASS" if ergebnis else "FAIL"))
    sys.exit(0 if all(ergebnisse.values()) else 1)
