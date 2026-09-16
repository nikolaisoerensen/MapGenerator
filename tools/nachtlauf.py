"""
Path: tools/nachtlauf.py

Die Bedienung des Nachtbetriebs. Ein Befehl je Handgriff.

    python tools/nachtlauf.py starten
    python tools/nachtlauf.py sperre-pruefen [main..HEAD]
    python tools/nachtlauf.py abschliessen 57 "Sperrliste anlegen" --tests "gruen - tests/smoke_test_nachtbetrieb.py"
    python tools/nachtlauf.py stand
    python tools/nachtlauf.py zuruecknehmen 57

Warum ein eigenes Werkzeug und nicht nackte git-Befehle: das Verfahren ist die
Sache, nicht die einzelnen Kommandos. Wer "abschliessen" tippt, bekommt die
Sperrpruefung, den Branchschutz und das feste Nachrichtenformat automatisch
mit. Wer die Handgriffe von Hand macht, vergisst nachts um drei genau einen
davon - und morgens sieht niemand, welchen.

Der Rueckgabewert ist ernst gemeint: 0 heisst in Ordnung, 1 heisst abgebrochen.
Ein Skript, das den Lauf steuert, kann sich daran halten.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nachtbetrieb import branch, sperre  # noqa: E402


def _starten(args):
    name = branch.starte_nacht(basis=args.basis)
    print("Nachtbranch angelegt und ausgecheckt: %s" % name)
    print("Alles Weitere passiert hier. main bleibt unberuehrt.")
    return 0


def _sperre_pruefen(args):
    if args.arbeitsstand:
        darf, notiz = sperre.pruefe_arbeitsstand()
        print("Geprueft: der uncommittete Arbeitsstand.")
    else:
        darf, notiz = sperre.pruefe_branch(basis=args.basis)
        print("Geprueft: %s..HEAD" % args.basis)
    print(notiz)
    return 0 if darf else 1


def _abschliessen(args):
    try:
        kurz, notiz = branch.ticket_abschliessen(
            args.nummer, args.titel,
            beschreibung=args.beschreibung or "",
            tests=args.tests or "")
    except branch.NachtlaufFehler as fehler:
        print(str(fehler))
        return 1
    print("Commit %s fuer #%d angelegt." % (kurz, args.nummer))
    if notiz != "Keine gesperrte Datei beruehrt.":
        print(notiz)
    return 0


def _stand(args):
    print(branch.stand(basis=args.basis))
    return 0


def _zuruecknehmen(args):
    try:
        kurz = branch.ticket_zuruecknehmen(args.nummer, basis=args.basis)
    except branch.NachtlaufFehler as fehler:
        print(str(fehler))
        return 1
    print("#%d zurueckgenommen, Gegen-Commit %s. Die anderen Tickets bleiben, "
          "wie sie waren." % (args.nummer, kurz))
    return 0


def baue_parser():
    p = argparse.ArgumentParser(
        prog="nachtlauf",
        description="Branch, Commits und Sperrpruefung eines Nachtlaufs.")
    p.add_argument("--basis", default="main",
                   help="Branch, gegen den gerechnet wird (Vorgabe: main)")
    unter = p.add_subparsers(dest="befehl", required=True)

    unter.add_parser("starten", help="Nachtbranch aus main anlegen"
                     ).set_defaults(funktion=_starten)

    s = unter.add_parser("sperre-pruefen",
                         help="prueft gegen nachtbetrieb/sperrliste.toml")
    s.add_argument("--arbeitsstand", action="store_true",
                   help="statt des Branches den uncommitteten Stand pruefen")
    s.set_defaults(funktion=_sperre_pruefen)

    a = unter.add_parser("abschliessen", help="ein Ticket, ein Commit")
    a.add_argument("nummer", type=int)
    a.add_argument("titel")
    a.add_argument("--beschreibung", default="")
    a.add_argument("--tests", default="",
                   help='z.B. "gruen - tests/smoke_test_nachtbetrieb.py"')
    a.set_defaults(funktion=_abschliessen)

    unter.add_parser("stand", help="Commits, Tickets, Testlage, Sperrlage"
                     ).set_defaults(funktion=_stand)

    z = unter.add_parser("zuruecknehmen", help="genau ein Ticket rueckgaengig")
    z.add_argument("nummer", type=int)
    z.set_defaults(funktion=_zuruecknehmen)
    return p


if __name__ == "__main__":
    argumente = baue_parser().parse_args()
    sys.exit(argumente.funktion(argumente))
