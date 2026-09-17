"""
Path: nachtbetrieb/branch.py

Das Branch- und Commit-Verfahren einer Nacht.

WARUM ES DAS GIBT: nachts arbeitet ein Agent ohne Zuschauer. Morgens muss
zweierlei gelten - main ist unveraendert, und jedes einzelne Stueck Arbeit ist
fuer sich pruefbar und fuer sich ruecknehmbar. Beides kommt aus derselben
Regel: ein Branch je Nacht, ein Commit je Ticket.

Branch: eine eigene Entwicklungslinie. Was dort passiert, beruehrt main erst,
wenn jemand es zusammenfuehrt.

WO DER TESTSTAND STEHT: in der Commit-Nachricht, als Zeile "Tests: ...". Nicht
in einem Begleitprotokoll. Damit gilt dreierlei: stand() liest ihn direkt aus
git log zurueck, er ueberlebt ein Zuruecknehmen (der Commit bleibt in der
Historie), und er steht im Code-Review neben genau der Aenderung, fuer die er
behauptet wurde. Ein Begleitprotokoll haette all das nicht - und es waere eine
zweite Wahrheit, die still auseinanderlaufen kann.
"""

import datetime
import re
import subprocess
from pathlib import Path

from nachtbetrieb import sperre

WURZEL = Path(__file__).resolve().parent.parent
NACHT_PRAEFIX = "nacht/"
TICKET_URL = "https://github.com/nikolaisoerensen/MapGenerator/issues/%d"


class NachtlaufFehler(Exception):
    """Etwas am Verfahren stimmt nicht - kein Grund, trotzdem weiterzumachen."""


def _git(*args, repo=None, pflicht=True):
    repo = Path(repo) if repo else WURZEL
    ergebnis = subprocess.run(("git",) + args, cwd=str(repo), capture_output=True,
                              text=True, encoding="utf-8")
    if pflicht and ergebnis.returncode != 0:
        raise NachtlaufFehler("git %s schlug fehl:\n%s"
                              % (" ".join(args), ergebnis.stderr.strip()))
    return ergebnis.stdout.strip()


def aktueller_branch(repo=None):
    return _git("rev-parse", "--abbrev-ref", "HEAD", repo=repo)


def ist_nachtbranch(repo=None):
    return aktueller_branch(repo=repo).startswith(NACHT_PRAEFIX)


def _vorhandene_nachtbranches(repo=None):
    zeilen = _git("branch", "--list", NACHT_PRAEFIX + "*",
                  "--format=%(refname:short)", repo=repo).splitlines()
    return {z.strip() for z in zeilen if z.strip()}


def naechster_branchname(datum=None, repo=None):
    """nacht/2026-09-16, bei Kollision -b, -c, ...

    Die Kollision ist der Normalfall, nicht die Ausnahme: eine Nacht wird
    abgebrochen, am naechsten Abend soll trotzdem ein Lauf starten, und der
    alte Branch liegt noch da. Ihn zu ueberschreiben hiesse, die Arbeit der
    letzten Nacht zu loeschen - also bekommt der neue einen Buchstaben.
    """
    datum = datum or datetime.date.today()
    basis = NACHT_PRAEFIX + datum.isoformat()
    vorhanden = _vorhandene_nachtbranches(repo=repo)
    if basis not in vorhanden:
        return basis
    for buchstabe in "bcdefghijklmnopqrstuvwxyz":
        kandidat = "%s-%s" % (basis, buchstabe)
        if kandidat not in vorhanden:
            return kandidat
    raise NachtlaufFehler(
        "Fuer %s gibt es schon 26 Nachtbranches. Da ist etwas anderes kaputt."
        % basis)


def starte_nacht(datum=None, repo=None, basis="main"):
    """Legt den Nachtbranch aus main an und wechselt hinein."""
    offen = _git("status", "--porcelain", repo=repo)
    if offen:
        raise NachtlaufFehler(
            "Im Arbeitsverzeichnis liegen noch uncommittete Aenderungen. Ein "
            "Nachtlauf startet nur auf sauberem Stand, sonst wandert fremde "
            "Arbeit in seine Commits:\n" + offen)
    name = naechster_branchname(datum=datum, repo=repo)
    _git("checkout", "-b", name, basis, repo=repo)
    return name


def _commit_text(nummer, titel, beschreibung, tests):
    zeilen = ["nacht(#%d): %s" % (nummer, titel), ""]
    if beschreibung:
        zeilen += [beschreibung.strip(), ""]
    zeilen.append("Ticket: " + TICKET_URL % nummer)
    zeilen.append("Tests: " + (tests.strip() if tests else "nicht gelaufen"))
    zeilen.append("Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>")
    return "\n".join(zeilen) + "\n"


def ticket_abschliessen(nummer, titel, dateien, beschreibung="", tests="",
                        repo=None):
    """Ein Ticket, ein Commit - fuer genau die uebergebenen Dateien.

    `dateien` sind die Pfade, die zu diesem Ticket gehoeren (wie an
    "git add" uebergeben, relativ zum Repo-Root). Nur sie werden gestaged
    und committet.

    Vorher stand hier ein uneingeschraenktes "git add -A". Das zog JEDE
    uncommittete Datei im Arbeitsverzeichnis in den Ticket-Commit hinein -
    auch voellig unabhaengige Handarbeit im selben Checkout, die zufaellig
    gerade herumlag. Gefunden 2026-09-17, als ein Ticket-Commit fremde
    Scratch-Dateien mitgerissen haette, waeren sie nicht von Hand
    committet worden. Deshalb jetzt: die Dateien werden benannt, nicht
    erraten.

    Weigert sich auf main - "Kein Nachtlauf schreibt auf main" ist keine
    Absichtserklaerung, sondern wird hier durchgesetzt.
    """
    if not dateien:
        raise NachtlaufFehler(
            "ticket_abschliessen() braucht die Liste der Dateien dieses "
            "Tickets. Ohne sie waere es wieder ein 'git add -A' durch die "
            "Hintertuer.")
    if not ist_nachtbranch(repo=repo):
        raise NachtlaufFehler(
            "Aktueller Branch ist %s, kein Nachtbranch. Ein Nachtlauf schreibt "
            "nie auf main. Zuerst: tools/nachtlauf.py starten"
            % aktueller_branch(repo=repo))
    darf, notiz = sperre.pruefe_arbeitsstand(repo=repo)
    if not darf:
        raise NachtlaufFehler(notiz)
    if not _git("status", "--porcelain", "--", *dateien, repo=repo):
        raise NachtlaufFehler(
            "Nichts zu committen unter den angegebenen Dateien (%s). Ein "
            "Ticket ohne Aenderung wird nicht abgeschlossen, sonst "
            "behauptet die Historie Arbeit, die es nicht gibt."
            % ", ".join(dateien))
    _git("add", "--", *dateien, repo=repo)
    _git("commit", "-m", _commit_text(nummer, titel, beschreibung, tests),
         "--", *dateien, repo=repo)
    return _git("rev-parse", "--short", "HEAD", repo=repo), notiz


def commits_der_nacht(basis="main", repo=None):
    """Was auf diesem Branch seit main liegt - je Commit Nummer, Titel, Tests."""
    roh = _git("log", "%s..HEAD" % basis, "--format=%H%x1f%s%x1f%b%x1e", repo=repo)
    eintraege = []
    for block in roh.split("\x1e"):
        if not block.strip():
            continue
        teile = block.strip("\n").split("\x1f")
        if len(teile) < 3:
            continue
        hash_, betreff, koerper = teile[0].strip(), teile[1], teile[2]
        treffer = re.match(r"nacht\(#(\d+)\):\s*(.*)", betreff)
        tests = next((z.split(":", 1)[1].strip()
                      for z in koerper.splitlines() if z.startswith("Tests:")),
                     "nicht genannt")
        eintraege.append({
            "hash": hash_[:8],
            "nummer": int(treffer.group(1)) if treffer else None,
            "titel": treffer.group(2) if treffer else betreff,
            "tests": tests,
        })
    return list(reversed(eintraege))


def stand(basis="main", repo=None):
    """Der eine Befehl: welche Commits, welche Tickets, welche Tests gruen."""
    branch = aktueller_branch(repo=repo)
    eintraege = commits_der_nacht(basis=basis, repo=repo)
    zeilen = ["Branch: %s  (gegen %s)" % (branch, basis)]
    if not ist_nachtbranch(repo=repo):
        zeilen.append("ACHTUNG: das ist kein Nachtbranch.")
    if not eintraege:
        zeilen.append("Noch kein Ticket abgeschlossen.")
    else:
        zeilen.append("%d Ticket(s) abgeschlossen:" % len(eintraege))
        for e in eintraege:
            nummer = ("#%d" % e["nummer"]) if e["nummer"] else "(ohne Nummer)"
            zeilen.append("  %s  %-6s %s" % (e["hash"], nummer, e["titel"]))
            zeilen.append("            Tests: %s" % e["tests"])
    darf, notiz = sperre.pruefe_branch(basis=basis, repo=repo)
    zeilen.append("")
    zeilen.append("Sperrliste: " + ("sauber" if darf else "VERLETZT"))
    if notiz != "Keine gesperrte Datei beruehrt.":
        zeilen.append(notiz)
    return "\n".join(zeilen)


def ticket_zuruecknehmen(nummer, basis="main", repo=None):
    """Nimmt genau das Ticket zurueck, ohne die anderen anzufassen.

    Das ist der Grund fuer ein Commit je Ticket: git revert kann genau diesen
    einen Commit umkehren. Bei einem Sammelcommit ginge das nicht - man muesste
    von Hand auseinanderklauben, was zusammengehoert.
    """
    treffer = [e for e in commits_der_nacht(basis=basis, repo=repo)
               if e["nummer"] == nummer]
    if not treffer:
        raise NachtlaufFehler(
            "Auf diesem Branch gibt es keinen Commit fuer #%d." % nummer)
    if len(treffer) > 1:
        raise NachtlaufFehler(
            "Fuer #%d gibt es %d Commits. Ein Ticket, ein Commit - das ist "
            "verletzt worden, hier muss jemand von Hand hinsehen."
            % (nummer, len(treffer)))
    _git("revert", "--no-edit", treffer[0]["hash"], repo=repo)
    return _git("rev-parse", "--short", "HEAD", repo=repo)


if __name__ == "__main__":
    print(stand())
