"""
Path: nachtbetrieb/seeds.py

Die Seedfuehrung fuer die beiden Test-Raenge: Waechter und Eichung.

WARUM ES DAS GIBT: die generierte Karte haengt an einem Zufalls-Seed, und
welcher Seed in welchem Test benutzt wird, war bisher nirgends geregelt -
jede Testdatei waehlte ihren eigenen. Das genuegt fuer den einzelnen Test,
aber nicht fuer das Verfahren als Ganzes: zwei Range brauchen zwei
GEGENSAETZLICHE Eigenschaften, und ohne eine gemeinsame Stelle, die das
festschreibt, verwischt der Unterschied mit der Zeit.

DIE ZWEI RAENGE, UND WARUM SIE GEGENSAETZLICHES BRAUCHEN:

  * WAECHTER (die schnellen Tests bei jeder Aenderung, tests/smoke_test_*.py)
    brauchen einen FESTEN Seed. Ein Waechter soll genau eine Frage
    beantworten: "hat die Aenderung etwas kaputtgemacht?" Zieht er dazu
    jedesmal eine andere Karte, kann ein Fehlschlag auch bedeuten "diese
    eine Karte war unguenstig" - und man muesste erst den Zufall
    ausschliessen, bevor man der eigentlichen Aenderung nachgeht. Bei
    festem Seed gibt es diese zweite Erklaerung nicht: schlaegt der Test
    fehl, liegt es an der Aenderung.

  * EICHUNG (der naechtliche Kalibrierlauf, siehe tools/test_raenge.py)
    braucht das Gegenteil: WECHSELNDE Seeds. Ihre Aufgabe ist nicht "hat
    sich etwas veraendert", sondern "haelt das Verfahren auch auf
    Landschaften, die wir noch nicht gesehen haben". Ein Fehler, der nur
    bei bestimmten Kuestenformen oder Flusslaeufen auftritt, bleibt bei
    einem einzigen festen Seed fuer immer unsichtbar - das ist exakt die
    Fehlerklasse aus dem Archetyp-Verteilungsbug vom 2026-08-24 (siehe
    CLAUDE.md, Abschnitt "Wenn alle Einzelpruefungen gruen sind").

DIE NAHT: eichungs_seeds(quelle) draussen, mit_seed(text, seed) fuer die
Fehlermeldung. Wer eine Eichungsmeldung baut, tut das ueber mit_seed() -
dann kann eine Meldung nicht mehr vergessen, den Seed zu nennen.

KEIN SEED AUS DER UHRZEIT: eichungs_seeds() nimmt bewusst KEINEN
Standardwert fuer `quelle` und ruft nirgends time.time() oder
datetime.now() auf. Ein Default wie datetime.date.today() saehe bequem
aus, waere aber genau der unkontrollierte Seed, den dieses Ticket
abschaffen soll: bei einem Rerun am naechsten Tag liefe er auf einen
ANDEREN Seed als beim Fehlschlag - der waere dann eben NICHT mehr
nachstellbar. `quelle` muss deshalb von aussen hereingegeben werden, aus
einem bereits kontrollierten, geloggten Wert - siehe
`quelle_aus_nachtbranch()` unten, die genau das aus dem einmal beim
Nachtstart angelegten Branchnamen liest (nachtbetrieb/branch.py,
`starte_nacht()`), statt die Uhr ein zweites Mal zu fragen.

NOCH NICHT VERDRAHTET: tools/test_raenge.py existiert in diesem Worktree
nicht (siehe Kommentar am Kopf des Ticket-Auftrags - #46 ist auf einem
anderen Branch gemergt, dieser Worktree wurde von main VOR dem Merge
abgezweigt). Diese Datei legt Konstante und Verfahren fest und beweist sie
unten in ihrer eigenen Guard-Funktion (`finde_unkontrollierte_seeds`) und
in tests/smoke_test_seedfuehrung.py. Das tatsaechliche Einhaengen in
tools/test_raenge.py - `WAECHTER_SEED` dort anstelle der Werte-Duplikate
zu benutzen und `eichungs_seeds()` fuer die drei Eichungslaeufe
aufzurufen - ist ein manueller Nacharbeitsschritt, den ein Mensch (oder
ein spaeterer Lauf mit Sicht auf beide Branches) erledigen muss.
"""

import hashlib
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# WAECHTER: ein fester Seed.
# ---------------------------------------------------------------------------

# 20260804 ist kein neuer Wert - er ist bereits der de-facto-Standard: ueber
# zwanzig bestehende Waechtertests (smoke_test_regionen_welt.py,
# smoke_test_kuestenprofiltreue.py, smoke_test_archetyp_verteilung.py,
# smoke_test_erosionsfilter_baender.py, smoke_test_regionsfeld.py, ... -
# `grep -rn "SEED = 20260804" tests/` zeigt den vollen Stand) tragen ihn
# schon als eigene Modulkonstante. Diese Datei macht daraus die EINE
# benannte, begruendete Stelle, auf die neue Waechtertests zeigen koennen,
# statt die Zahl ein weiteres Mal von Hand abzuschreiben.
#
# Der Wert selbst ist willkuerlich (ein Datum, an dem zufaellig die ersten
# Regions-Waechter gebaut wurden) - das ist Absicht. Ein Waechter-Seed muss
# nicht "gut" sein, nur STABIL: er darf sich nie mehr aendern, sonst
# verliert jeder Test, der ihn benutzt, seine Referenzwerte gleichzeitig.
WAECHTER_SEED = 20260804


# ---------------------------------------------------------------------------
# EICHUNG: drei wechselnde Seeds, deterministisch aus einer kontrollierten
# Quelle abgeleitet.
# ---------------------------------------------------------------------------

EICHUNGS_ANZAHL = 3


def eichungs_seeds(quelle):
    """Leitet EICHUNGS_ANZAHL verschiedene, deterministische Seeds ab.

    `quelle` ist ein bereits kontrollierter, geloggter Text - typischerweise
    das Datum des Nachtbranches (z.B. "2026-09-21" aus `nacht/2026-09-21`,
    siehe `quelle_aus_nachtbranch()`). Kein Default, absichtlich: ein Default
    wie das heutige Datum waere bequem, aber bei jedem Aufruf ohne
    ausdruecklich uebergebene Quelle wieder ein an die Uhr gebundener,
    unkontrollierter Seed durch die Hintertuer.

    Dieselbe `quelle` liefert IMMER dieselben drei Seeds (SHA-256 ueber die
    Quelle, in drei 32-Bit-Haeppchen zerlegt) - das ist die ganze
    Nachstellbarkeit: wer den Seed aus einer Fehlermeldung hat, muss nicht
    zusaetzlich wissen, WIE er entstand, um ihn direkt an den betroffenen
    Generator zu uebergeben (siehe `mit_seed()` unten, die genau diesen
    fertigen Seed in die Meldung setzt - nicht die Quelle).
    """
    if not quelle:
        raise ValueError(
            "eichungs_seeds() braucht eine nicht-leere, kontrollierte "
            "Quelle (z.B. das Nachtbranch-Datum). Ein leerer oder "
            "fehlender Wert waere kein Verfahren mehr, sondern Zufall.")
    digest = hashlib.sha256(str(quelle).encode("utf-8")).digest()
    return tuple(
        int.from_bytes(digest[i:i + 4], "big")
        for i in range(0, EICHUNGS_ANZAHL * 4, 4)
    )


_NACHTBRANCH_DATUM = re.compile(r"nacht/(\d{4}-\d{2}-\d{2})")


def quelle_aus_nachtbranch(branchname):
    """Zieht das Datum aus einem Nachtbranch-Namen, z.B.

    "nacht/2026-09-21"   -> "2026-09-21"
    "nacht/2026-09-21-b" -> "2026-09-21" (Kollisionsbuchstabe faellt weg,
                             siehe branch.naechster_branchname - dieselbe
                             Nacht soll dieselben Eichungs-Seeds bekommen,
                             auch wenn ihr Branch wegen eines Vortages mit
                             Buchstaben-Suffix laeuft)

    Das Datum steckt schon im Branchnamen, den `branch.starte_nacht()`
    EINMAL beim Nachtstart erzeugt und danach fuer die ganze Nacht
    unveraendert laesst - das ist der kontrollierte Wert. Ein zweiter
    `datetime.date.today()`-Aufruf hier waere unnoetig UND riskant: liefe
    der Nachtlauf ueber Mitternacht hinweg, koennte er ein anderes Datum
    liefern als der Branchname traegt.
    """
    treffer = _NACHTBRANCH_DATUM.search(str(branchname))
    if not treffer:
        raise ValueError(
            "%r sieht nicht wie ein Nachtbranch-Name (nacht/JJJJ-MM-TT) "
            "aus - daraus laesst sich keine Eichungs-Quelle ziehen."
            % (branchname,))
    return treffer.group(1)


def mit_seed(text, seed):
    """Haengt den benutzten Seed lesbar an eine Fehlermeldung an.

    Abnahmekriterium des Tickets: "Jede Fehlermeldung nennt den benutzten
    Seed." Das ist nur dann verlaesslich wahr, wenn es EINE Stelle gibt,
    die die Meldung baut, statt dass jeder Test sein eigenes
    f"...{seed}..." formuliert und irgendwann vergisst.
    """
    return "%s (Seed %d)" % (text.rstrip(), seed)


# ---------------------------------------------------------------------------
# Guard: kein Seed mehr aus der Uhrzeit oder sonst unkontrolliert.
# ---------------------------------------------------------------------------

# Diese Muster ziehen einen Seed (oder tun so, als taeten sie es) aus etwas,
# das sich bei jedem Lauf aendert - und sind damit fuer JEDEN Test verboten,
# nicht nur fuer die Eichung. Ein Waechter mit einem solchen Muster waere
# gar kein Waechter mehr (siehe Modul-Docstring), und eine Eichung damit
# waere nicht mehr die Art "wechselnd", die dieses Ticket meint: nicht
# nachstellbar aus dem genannten Seed, sondern einfach undokumentiert.
_UNKONTROLLIERT = (
    re.compile(r"default_rng\(\s*\)"),
    re.compile(r"np\.random\.seed\(\s*\)"),
    re.compile(r"random\.seed\(\s*\)"),
    re.compile(r"seed\s*=\s*[^,\)\n]*time\.time\(\)"),
    re.compile(r"seed\s*=\s*[^,\)\n]*datetime\.now\("),
    re.compile(r"seed\s*=\s*[^,\)\n]*os\.urandom"),
)


def finde_unkontrollierte_seeds(pfade):
    """Durchsucht die uebergebenen Dateien nach den Mustern oben.

    Gibt eine Liste (pfad, zeilennummer, zeile) zurueck - leer heisst
    sauber. Nimmt Pfade statt selbst zu suchen, aus demselben Grund wie
    `sperre.pruefe()`: die Naht ist die Pruefung, nicht die Dateisuche -
    wer die Liste der zu pruefenden Dateien hat (git diff, ein
    Testverzeichnis, eine Hand voll Pfade), kann sie hier durchreichen.
    """
    treffer = []
    for pfad in pfade:
        pfad = Path(pfad)
        try:
            zeilen = pfad.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for nr, zeile in enumerate(zeilen, start=1):
            if any(m.search(zeile) for m in _UNKONTROLLIERT):
                treffer.append((str(pfad), nr, zeile.strip()))
    return treffer
