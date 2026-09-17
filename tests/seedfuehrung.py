"""
Path: tests/seedfuehrung.py

Die EINE Stelle, an der festgelegt ist, welcher Seed in welchem Testrang
benutzt wird (Ticket #48). Die Unterscheidung Waechter/Eichung als offizielle
Testraenge fuehrt erst Ticket #46 ein (baut auf #45 auf) - diese Datei ist die
Infrastruktur, die #46 danach nur noch zuordnen muss: einen Namen fuer den
Waechter-Seed, eine Funktion fuer die drei Eichungs-Seeds, und einen Helfer,
der jede Fehlermeldung mit dem benutzten Seed versieht.

WARUM ES DIESE DATEI BRAUCHT. Vor #48 war die Seedwahl in rund vierzig
Testdateien einzeln als `SEED = 20260804` hingeschrieben - immer derselbe
Wert, aber an vierzig Stellen kopiert statt an einer benannt. Das ist kein
Fehler (alle vierzig meinen bewusst denselben festen Seed), aber es gibt
keine Stelle, an der die Wahl BEGRUENDET steht, und keine Stelle, an der eine
Eichung - die laut Ticket #48 gerade NICHT immer denselben Seed nehmen soll -
ansetzen koennte.

DIE ZWEI RAENGE, UND WARUM SIE UNTERSCHIEDLICH GEFUEHRT WERDEN.

  WAECHTER  Ein einziger, für immer fester Seed. Ticket-Text: "Der Test soll
            reproduzierbar sein; schlaegt er fehl, liegt es an der Aenderung
            und nicht am Wuerfel." Ein Waechter, der bei jedem Lauf eine
            andere Landschaft bekaeme, koennte nicht mehr zwischen "die
            Aenderung hat etwas kaputtgemacht" und "dieser Seed war zufaellig
            ungeeignet" unterscheiden.

  EICHUNG   Drei WECHSELNDE Seeds - "wechselnd" ist hier woertlich aus dem
            Ticket uebernommen, nicht "drei fuer immer feste Extra-Seeds".
            Ein fester Dreiersatz wuerde genau dieselbe Schwaeche wie ein
            einzelner fester Seed haben, nur dreifach: er versteckt jeden
            Fehler, der auf einer der uebrigen unendlich vielen Landschaften
            auftritt, fuer immer. Die drei Eichungs-Seeds haengen deshalb am
            KALENDERTAG des Laufs (nicht an der Uhrzeit - das verbietet
            Ticket #48 ausdruecklich) und aendern sich damit jede Nacht, ohne
            je unkontrolliert zu sein: die Formel ist hier fest hingeschrieben,
            und jede Fehlermeldung nennt den tatsaechlich gezogenen Seed
            (siehe `seed_meldung()`) - wer eine rote Eichung nachstellen will,
            braucht dafuer nur die genannte Zahl, nicht das Datum.

Naht: `eichung_seeds()` ist die eine Stelle, an der alle Aufrufer dieselbe
Auskunft abholen, welche drei Seeds heute Nacht gezogen werden - ob der
Aufrufer ein Testlauf, ein manueller Nachvollzug oder #46s Testraenge-Logik
ist, ist dieser Funktion gleich.
"""

import datetime

# ---------------------------------------------------------------------------
# WAECHTER-SEED
# ---------------------------------------------------------------------------
#
# 20260804, nicht neu erfunden: dieser Wert ist bereits der de-facto-Standard
# in ueber vierzig `tests/smoke_test_*.py`-Dateien (z. B.
# smoke_test_regionen_welt.py - laut CLAUDE.md "der empfindlichste Waechter
# fuer Gelaendeform" dieses Projekts) und steckt als "Seed 20260804" bereits
# in etlichen dokumentierten Messwerten (docs/OFFENE_PUNKTE.md, CLAUDE.md).
# Ticket #48 schreibt diesen Wert deshalb NUR FEST, statt ihn zu aendern -
# ein neuer Wert wuerde jede daran haengende Messung entwerten, ohne dass
# sich am Programm etwas geaendert haette.
WAECHTER_SEED = 20260804

# Woraus der Wert historisch stammt (Datum im Format JJJJMMTT, 04.08.2026) -
# reiner Kommentar, keine Berechnung. Diente urspruenglich nur als leicht zu
# merkende Konstante; die EICHUNGS-Formel unten macht sich dasselbe Format
# absichtlich zunutze (siehe `eichung_seeds()`).

# Abstand zwischen den drei Eichungs-Seeds. Kein neuer Einfall: derselbe
# Abstand (1013, eine Primzahl fern von jeder Kartengroesse/jedem Raster)
# erzeugt bereits in tests/smoke_test_archetyp_verteilung.py
# (`seed = SEED + 1013 * k`) mehrere sichtbar verschiedene Karten aus einem
# Basiswert. Wiederverwendet statt neu erfunden, damit "wie sie gewaehlt
# werden" an einer zweiten, unabhaengigen Stelle im Projekt bereits
# nachvollzogen werden kann.
_EICHUNG_ABSTAND = 1013
_EICHUNG_ANZAHL = 3


def _tagesbasis(datum):
    """JJJJMMTT dieses Kalendertages als Zahl - dasselbe Format, in dem
    WAECHTER_SEED selbst schon geschrieben ist (20260804 = 04.08.2026)."""
    return datum.year * 10000 + datum.month * 100 + datum.day


def eichung_seeds(datum=None):
    """
    Die drei Seeds fuer den heutigen Eichungslauf.

    `datum` ist ein `datetime.date` (Standard: der heutige Kalendertag - NIE
    die Uhrzeit, das ist die von Ticket #48 verbotene unkontrollierte Quelle).
    Wird ein Datum explizit uebergeben, ist der Rueckgabewert reine Funktion
    davon und beliebig oft nachvollziehbar - fuer den Nachtlauf selbst reicht
    aber der Standardfall, weil jede Fehlermeldung den GEZOGENEN Seed ohnehin
    als Zahl nennt (`seed_meldung()`); das Datum muss dafuer niemand mehr
    kennen.

    Formel: Tagesbasis (JJJJMMTT) plus 0/1013/2026 - drei Karten, die sich
    voneinander so stark unterscheiden wie zwei um 1013 versetzte Karten es
    in smoke_test_archetyp_verteilung.py bereits nachweislich tun, und die
    sich jede Nacht mit dem Kalendertag weiterbewegen.
    """
    if datum is None:
        datum = datetime.date.today()
    basis = _tagesbasis(datum)
    return tuple(basis + _EICHUNG_ABSTAND * k for k in range(_EICHUNG_ANZAHL))


def seed_meldung(text, seed):
    """
    Haengt den benutzten Seed lesbar an eine Fehlermeldung an.

    Abnahmekriterium: "Jede Fehlermeldung nennt den benutzten Seed." Diese
    Funktion ist der EINE Ort, der das Format festlegt (`"<Text> (seed=<N>)"`),
    damit #46 und kuenftige Eichungstests nicht jeder ihr eigenes Format
    erfinden - und damit ein Nachvollzug per Textsuche nach "seed=" ueber alle
    Eichungsmeldungen hinweg funktioniert.
    """
    return f"{text} (seed={seed})"


if __name__ == "__main__":
    # Kurzer Handnachvollzug, kein Testlauf: zeigt, was heute Nacht gezogen
    # wuerde, ohne irgendetwas zu berechnen.
    print(f"Waechter-Seed: {WAECHTER_SEED}")
    heute = eichung_seeds()
    print(f"Eichungs-Seeds heute ({datetime.date.today()}): {heute}")
    print(seed_meldung("Beispiel-Fehlschlag", heute[0]))
