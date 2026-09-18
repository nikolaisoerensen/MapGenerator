# -*- coding: utf-8 -*-
"""
Waechter fuer Ticket #49 ("Die zehn roten Tests einzeln aufschluesseln").

docs/TESTBERICHT.md, Abschnitt 3 ("Bekannt und unveraendert") fuehrte bis
zum 16.09.2026 eine Tabelle mit rohen Messwerten je bekanntem Fehlschlag
("Macchia Hang 18,6 statt 14,5", "Skerrheim 7,86 statt 8,60 K", ...).
Eine solche Tabelle ist nuetzlich, solange sie eine Liste von Schulden mit
Frist ist - und schaedlich, sobald sie zum Ablageort wird, in den jeder
neue Fund unverbindlich eingetragen wird, ohne dass je ein Ticket
entsteht.

Entschieden (Ticket #49): jede Zeile verweist auf genau eine
GitHub-Issue-Nummer statt einen Rohwert zu nennen. Ist-Wert, Soll-Wert,
erstes Auftreten und Frist stehen im Ticket, nicht mehr hier.

Dieser Test haelt das fest, indem er die Tabelle in Abschnitt 3 parst und
bei jeder Datenzeile prueft:

1. Die letzte Spalte ("Ticket") ist NICHTS als eine Ticket-Nummer der
   Form "#123" (mit optionalem fuehrenden/folgendem Leerraum).
2. Die Befund-Spalte enthaelt keine Dezimalzahl mit Komma (deutsches
   Zahlenformat, z.B. "18,6" oder "7,86") und kein Prozentzeichen - das
   sind die Muster, in denen sich ein roher Messwert einschleicht. Reine
   Kardinalzahlen ohne Komma (z.B. Aufloesungen wie "512" oder "1024 px")
   sind erlaubt, weil sie einen Parameter benennen, keinen gemessenen
   Wert.

Kommt ein neuer Fehlschlag dazu, MUSS also erst ein Ticket angelegt und
dessen Nummer eingetragen werden - ein direkt eingetragener Messwert laesst
diesen Test rot werden.
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTBERICHT = REPO_ROOT / "docs" / "TESTBERICHT.md"

ABSCHNITT_UEBERSCHRIFT = "## 3. Bekannt und unverändert"
TICKET_MUSTER = re.compile(r"^#\d+$")
# Deutsches Dezimalkomma, z.B. "18,6" oder "0,22" - Kennzeichen eines
# rohen Messwerts.
KOMMAZAHL_MUSTER = re.compile(r"\d+,\d+")


def _abschnitt_3_tabelle(text: str) -> list[str]:
    """Liefert die rohen Tabellenzeilen (Header + Trenner + Daten) von
    Abschnitt 3, bis zur naechsten '## '-Ueberschrift."""
    zeilen = text.splitlines()
    start = None
    for i, zeile in enumerate(zeilen):
        if zeile.strip() == ABSCHNITT_UEBERSCHRIFT:
            start = i
            break
    if start is None:
        raise AssertionError(
            f"Ueberschrift '{ABSCHNITT_UEBERSCHRIFT}' nicht in {TESTBERICHT} gefunden - "
            "wurde Abschnitt 3 umbenannt oder entfernt?")

    ende = len(zeilen)
    for i in range(start + 1, len(zeilen)):
        if zeilen[i].startswith("## "):
            ende = i
            break

    tabellenzeilen = [z for z in zeilen[start:ende] if z.strip().startswith("|")]
    if not tabellenzeilen:
        raise AssertionError(
            "Abschnitt 3 enthaelt keine Markdown-Tabelle mehr - "
            "wurde sie versehentlich geloescht statt umgeschrieben?")
    return tabellenzeilen


def _parse_zeile(zeile: str) -> list[str]:
    # Markdown-Tabellenzeile "| a | b | c |" -> ["a", "b", "c"]
    innen = zeile.strip().strip("|")
    return [spalte.strip() for spalte in innen.split("|")]


def run_tabelle_verweist_nur_auf_tickets() -> bool:
    text = TESTBERICHT.read_text(encoding="utf-8")
    zeilen = _abschnitt_3_tabelle(text)

    header = _parse_zeile(zeilen[0])
    erwartet_header = ["Test", "Befund", "Ticket"]
    if header != erwartet_header:
        print(f"[FAIL] Tabellenkopf ist {header}, erwartet {erwartet_header}")
        return False

    # zeilen[1] ist die Markdown-Trennzeile ("|---|---|---|"), Datenzeilen
    # beginnen bei Index 2.
    datenzeilen = zeilen[2:]
    if not datenzeilen:
        print("[FAIL] Abschnitt 3 hat eine Tabelle ohne Datenzeilen")
        return False

    fehler = []
    for zeile in datenzeilen:
        spalten = _parse_zeile(zeile)
        if len(spalten) != 3:
            fehler.append(f"Zeile hat {len(spalten)} statt 3 Spalten: {zeile!r}")
            continue
        test_name, befund, ticket = spalten

        if not TICKET_MUSTER.match(ticket):
            fehler.append(
                f"'{test_name}': Ticket-Spalte ist {ticket!r}, "
                "erwartet ein reiner Verweis der Form '#123' - "
                "steht dort ein Rohwert statt eines Tickets?")

        if KOMMAZAHL_MUSTER.search(befund):
            fehler.append(
                f"'{test_name}': Befund-Spalte {befund!r} enthaelt eine "
                "Dezimalzahl (Komma-Schreibweise) - das sieht nach einem rohen "
                "Messwert aus, der ins Ticket gehoert, nicht in diese Tabelle.")

        if "%" in befund:
            fehler.append(
                f"'{test_name}': Befund-Spalte {befund!r} enthaelt ein "
                "Prozentzeichen - das sieht nach einem rohen Messwert aus, "
                "der ins Ticket gehoert, nicht in diese Tabelle.")

    if fehler:
        for f in fehler:
            print(f"[FAIL] {f}")
        return False

    print(f"[OK] Abschnitt 3 hat {len(datenzeilen)} Zeilen, jede verweist auf ein Ticket, "
          "keine enthaelt einen rohen Messwert")
    return True


def main() -> int:
    ergebnisse = {
        "tabelle_verweist_nur_auf_tickets": run_tabelle_verweist_nur_auf_tickets(),
    }
    print("\n=== SUMMARY ===")
    alle_ok = True
    for name, ok in ergebnisse.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            alle_ok = False
        print(f"{name}: {status}")
    return 0 if alle_ok else 1


if __name__ == "__main__":
    sys.exit(main())
