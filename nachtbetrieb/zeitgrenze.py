"""
Path: nachtbetrieb/zeitgrenze.py

Die Zeitgrenze je Ticket und die Notiz, die beim Abbruch entsteht.

WARUM ES DAS GIBT: ein Agent, der nachts an einem Ticket haengenbleibt,
haengt bis zum Morgen. Er merkt es nicht - jeder einzelne Versuch sieht
vernuenftig aus. Am Morgen ist eine Nacht weg und die anderen Tickets sind
unberuehrt. Die Zeitgrenze ist die Reissleine.

WAS DIE GRENZE NICHT IST: eine runde Zahl. Eine feste halbe Stunde waere fuer
ein Ticket mit schnellen Tests verschwenderisch und fuer eines mit der
Eichungsreihe (rund zwei Minuten je Lauf) laecherlich - der Agent kaeme nicht
einmal durch drei Versuche. Deshalb wird sie GERECHNET, aus der Laufzeit
genau der Tests, die das Ticket braucht: grenze(test_dauer_s).

DIE NAHT: alles laeuft ueber grenze(test_dauer_s) und notiz(steckenbleib).
Die erste sagt, wieviel Zeit ein Ticket bekommt und warum; die zweite macht
aus einem Abbruch einen lesbaren Absatz. Beide rechnen nur - sie rufen kein
git auf, schreiben nichts fest und brechen nichts ab. Genau das ist der
"saubere Abbruch": der Branch bleibt, die Arbeit bleibt liegen, verworfen wird
nichts. Wer aufraeumen wollte, muesste es hier ausdruecklich tun, und es tut
es niemand.

MESSWARNUNG, die zur Rechnung gehoert: diese Maschine schwankt um Faktor 2-3.
Dieselbe feste Referenzlast mass zwischen 0.80 s und 1.91 s ohne erkennbare
Fremdlast (siehe tools/testlauf.py). Laufzeiten sind Groessenordnungen, keine
Messwerte - deshalb steht der Streufaktor in der Formel und nicht im Kopf des
Lesers.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

WURZEL = Path(__file__).resolve().parent.parent
LAUFORDNER = Path(__file__).resolve().parent / "laufberichte"

# Die drei Groessen der Rechnung. Jede einzeln begruendet, damit man sie
# einzeln bestreiten kann - eine Gesamtzahl koennte man nur glauben.

# Ein Ticket ist erst dann ernsthaft versucht, wenn der Agent dreimal
# durchgelaufen ist: einmal bauen, einmal reparieren, einmal bestaetigen.
# Weniger heisst, er hat aufgegeben, bevor er etwas wusste.
VERSUCHE = 3

# Was ein Versuch neben den Tests kostet: lesen, aendern, entscheiden. Das ist
# geschaetzt und nicht gemessen - es gibt keine Uhr, die das Nachdenken eines
# Agenten misst. Fuenf Minuten je Versuch ist die ehrlichste Zahl, die hier
# stehen kann, und sie steht ausdruecklich als Schaetzung da.
ARBEIT_JE_VERSUCH_S = 300

# Der Streufaktor aus der Messwarnung: eine mit 120 s gemessene Testreihe darf
# 360 s brauchen, ohne dass etwas kaputt ist. Wer ohne ihn rechnet, bricht
# Tickets ab, die nur Pech mit der Maschine hatten.
STREUFAKTOR = 3

# Untergrenze: auch ein Ticket ohne eigene Tests bekommt eine halbe Stunde.
# Darunter bricht die Grenze Arbeit ab, die einfach nur angefangen hat.
GRUNDGRENZE_S = 1800

# Obergrenze: drei Stunden. Eine Nacht hat etwa acht; ein Ticket, das laenger
# darf, kann allein eine halbe Nacht verbrauchen, und der Morgen sieht ein
# leeres Ergebnis statt drei fertiger Tickets. Wer mehr braucht, ist zu gross
# geschnitten - das ist ein Ticketfehler, kein Zeitproblem.
DECKEL_S = 10800


class NotizUnvollstaendig(Exception):
    """Die Steckenbleib-Notiz taugt nicht - es fehlt, was sie brauchbar macht."""


@dataclass(frozen=True)
class Grenze:
    """Die Grenze und ihre Begruendung in einem Stueck.

    Zusammen, nicht getrennt: eine Zahl ohne Begruendung ist im Morgenbericht
    wertlos, und eine Begruendung, die man extra anfordern muss, fordert
    niemand an.
    """
    sekunden: int
    test_dauer_s: float
    gedeckelt: bool
    begruendung: str

    @property
    def minuten(self):
        return self.sekunden / 60.0


def grenze(test_dauer_s=0.0):
    """Wieviel Zeit ein Ticket bekommt, gerechnet aus seiner Testlaufzeit.

    test_dauer_s ist die GEMESSENE Laufzeit genau der Tests, die dieses Ticket
    gruen bekommen muss - aus tools/testlauf.py --bericht, nicht geschaetzt.
    Ohne Angabe gilt die Untergrenze.
    """
    test_dauer_s = max(0.0, float(test_dauer_s))
    roh = VERSUCHE * (ARBEIT_JE_VERSUCH_S + STREUFAKTOR * test_dauer_s)
    sekunden = int(max(GRUNDGRENZE_S, min(DECKEL_S, roh)))
    gedeckelt = roh > DECKEL_S

    teile = [
        "%d Versuche a (%d s Arbeit + %d x %.0f s Tests) = %.0f s"
        % (VERSUCHE, ARBEIT_JE_VERSUCH_S, STREUFAKTOR, test_dauer_s, roh),
    ]
    if roh < GRUNDGRENZE_S:
        teile.append("angehoben auf die Untergrenze %d s (%.0f min)"
                     % (GRUNDGRENZE_S, GRUNDGRENZE_S / 60.0))
    if gedeckelt:
        teile.append(
            "GEDECKELT auf %d s (%.0f min). Ein Ticket, das nach dieser "
            "Rechnung laenger braucht, ist zu gross geschnitten - das gehoert "
            "in den Morgenbericht, nicht stillschweigend in die Zahl."
            % (DECKEL_S, DECKEL_S / 60.0))
    teile.append("Streufaktor %d, weil dieselbe Last auf dieser Maschine um "
                 "Faktor 2-3 schwankt." % STREUFAKTOR)
    return Grenze(sekunden=sekunden, test_dauer_s=test_dauer_s,
                  gedeckelt=gedeckelt, begruendung=" ".join(teile))


class Uhr:
    """Misst, wie lange ein Ticket schon laeuft.

    Die Zeitquelle ist ein Argument, kein eingebautes time.monotonic(). Damit
    kann der Test einen Abbruch vorfuehren, ohne wirklich eine Stunde zu
    warten - ein Test, der schlaeft, wird irgendwann abgeschaltet, und dann
    prueft niemand mehr den Abbruch.
    """

    def __init__(self, grenze_s, jetzt=None):
        self._jetzt = jetzt or time.monotonic
        self.grenze_s = float(grenze_s)
        self.beginn = self._jetzt()

    def verstrichen(self):
        return self._jetzt() - self.beginn

    def rest(self):
        return self.grenze_s - self.verstrichen()

    def abgelaufen(self):
        return self.verstrichen() >= self.grenze_s


@dataclass
class Steckenbleib:
    """Was ein abgebrochenes Ticket hinterlaesst.

    Die Felder sind genau das, was der Naechste braucht, um nicht bei null
    anzufangen: wo es steht, was rot ist und wie es sich meldet, was schon
    versucht wurde, und woran man als naechstes glaubt. Fehlt eines davon,
    laesst notiz() sich nicht schreiben - siehe dort.
    """
    nummer: int
    titel: str
    grenze_s: int
    verstrichen_s: float
    stand: str = ""
    roter_test: str = ""
    meldung: str = ""
    versuche: list = field(default_factory=list)
    vermutung: str = ""
    grenze_begruendung: str = ""


def notiz(steckenbleib):
    """Der Absatz, der ins Ticket kommt. Weigert sich, wenn er leer waere.

    "Zeitlimit erreicht" ist keine Notiz, das ist eine Uhr. Deshalb ist die
    Vollstaendigkeitspruefung hier hart und nicht als Hinweis formuliert: eine
    Notiz, die nur den Abbruch meldet, kostet den Naechsten genau die Stunde,
    die der Abgebrochene schon verloren hat.

    Der rote Test darf fehlen - ein Ticket kann steckenbleiben, bevor
    ueberhaupt ein Test lief. Dann muss der Stand das sagen. Was nie fehlen
    darf: Stand, mindestens ein Versuch, und eine naechste Vermutung.
    """
    fehlend = []
    if not steckenbleib.stand.strip():
        fehlend.append("stand (wo die Arbeit steht)")
    if not [v for v in steckenbleib.versuche if str(v).strip()]:
        fehlend.append("versuche (was schon probiert wurde)")
    if not steckenbleib.vermutung.strip():
        fehlend.append("vermutung (woran es als naechstes liegen koennte)")
    if steckenbleib.roter_test.strip() and not steckenbleib.meldung.strip():
        fehlend.append("meldung (die Fehlermeldung des roten Tests, nicht nur "
                       "sein Name)")
    if fehlend:
        raise NotizUnvollstaendig(
            "Diese Steckenbleib-Notiz waere wertlos, es fehlt:\n  - %s\n\n"
            "„Zeitlimit erreicht“ ist keine Notiz, das ist eine Uhr. "
            "Wer morgens hier weitermacht, faengt sonst bei null an und "
            "verliert dieselbe Zeit noch einmal."
            % "\n  - ".join(fehlend))

    zeilen = [
        "STECKENGEBLIEBEN an der Zeitgrenze - #%d %s"
        % (steckenbleib.nummer, steckenbleib.titel),
        "",
        "Gelaufen: %.0f min von %.0f min erlaubt."
        % (steckenbleib.verstrichen_s / 60.0, steckenbleib.grenze_s / 60.0),
    ]
    if steckenbleib.grenze_begruendung:
        zeilen.append("Die Grenze kam so zustande: %s"
                      % steckenbleib.grenze_begruendung)
    zeilen += ["", "Wo die Arbeit steht:", "  " + steckenbleib.stand.strip()]

    if steckenbleib.roter_test.strip():
        zeilen += ["", "Was rot ist:",
                   "  " + steckenbleib.roter_test.strip(),
                   "  Meldung: " + steckenbleib.meldung.strip()]
    else:
        zeilen += ["", "Was rot ist:",
                   "  kein Test gelaufen - siehe Stand."]

    zeilen += ["", "Was schon versucht wurde:"]
    for v in steckenbleib.versuche:
        if str(v).strip():
            zeilen.append("  - " + str(v).strip())

    zeilen += ["", "Naechste Vermutung:", "  " + steckenbleib.vermutung.strip(),
               "",
               "Das Ticket bleibt OFFEN und wird NICHT neu gestartet. Der "
               "Branch und die begonnene Arbeit liegen unveraendert da - "
               "abgebrochen heisst hier nicht verworfen."]
    return "\n".join(zeilen)


def festhalten(steckenbleib, ordner=None):
    """Schreibt die Notiz weg, damit der Morgenbericht sie findet.

    Als JSON und nicht als fertiger Text: der Morgenbericht will die Felder
    einzeln (er listet Nummer und Titel in einer Zeile), das Ticket will den
    Absatz. Aus Feldern laesst sich der Absatz bauen, aus dem Absatz nicht
    die Felder.
    """
    notiz(steckenbleib)  # erst pruefen, dann schreiben - keine leeren Notizen
    ordner = Path(ordner) if ordner else LAUFORDNER
    ordner.mkdir(parents=True, exist_ok=True)
    ziel = ordner / ("steckenbleib-%d.json" % steckenbleib.nummer)
    ziel.write_text(json.dumps(asdict(steckenbleib), ensure_ascii=False,
                               indent=2), encoding="utf-8")
    return ziel


def sammle(ordner=None):
    """Alle Steckenbleib-Notizen dieses Laufs, nach Ticketnummer sortiert."""
    ordner = Path(ordner) if ordner else LAUFORDNER
    if not ordner.exists():
        return []
    gefunden = []
    for pfad in sorted(ordner.glob("steckenbleib-*.json")):
        roh = json.loads(pfad.read_text(encoding="utf-8"))
        gefunden.append(Steckenbleib(**roh))
    return sorted(gefunden, key=lambda s: s.nummer)


def aufraeumen(ordner=None):
    """Loescht die Notizen des vorigen Laufs. Nur zu Beginn einer Nacht.

    Sonst zaehlte der Morgenbericht die Steckenbleiber von gestern mit, und
    zwar ohne dass es auffiele - eine Zahl, die zu hoch ist, sieht genauso
    aus wie eine, die stimmt.
    """
    ordner = Path(ordner) if ordner else LAUFORDNER
    if not ordner.exists():
        return 0
    anzahl = 0
    for pfad in ordner.glob("steckenbleib-*.json"):
        pfad.unlink()
        anzahl += 1
    return anzahl


if __name__ == "__main__":
    import sys
    dauer = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    g = grenze(dauer)
    print("Testlaufzeit %.0f s  ->  Zeitgrenze %.0f min" % (dauer, g.minuten))
    print(g.begruendung)
