"""
Path: tools/nachtlauf.py

Die Bedienung des Nachtbetriebs. Ein Befehl je Handgriff.

    python tools/nachtlauf.py starten
    python tools/nachtlauf.py sperre-pruefen [main..HEAD]
    python tools/nachtlauf.py abschliessen 57 "Sperrliste anlegen" --dateien nachtbetrieb/sperrliste.toml tests/smoke_test_nachtbetrieb.py --tests "gruen - tests/smoke_test_nachtbetrieb.py"
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

from nachtbetrieb import branch, morgenbericht, sperre, zeitgrenze  # noqa: E402


def _starten(args):
    name = branch.starte_nacht(basis=args.basis)
    geloescht = zeitgrenze.aufraeumen()
    print("Nachtbranch angelegt und ausgecheckt: %s" % name)
    if geloescht:
        print("%d Steckenbleib-Notiz(en) der letzten Nacht entfernt - sonst "
              "zaehlte der Morgenbericht sie mit." % geloescht)
    print("Alles Weitere passiert hier. main bleibt unberuehrt.")
    return 0


def _grenze(args):
    g = zeitgrenze.grenze(args.tests_dauer)
    print("Testlaufzeit %.0f s  ->  Zeitgrenze %.0f min (%d s)"
          % (g.test_dauer_s, g.minuten, g.sekunden))
    print(g.begruendung)
    return 0


def _steckengeblieben(args):
    """Bricht ein Ticket sauber ab: Notiz schreiben, sonst nichts anfassen."""
    eintrag = zeitgrenze.Steckenbleib(
        nummer=args.nummer, titel=args.titel,
        grenze_s=zeitgrenze.grenze(args.tests_dauer).sekunden,
        verstrichen_s=args.gelaufen * 60.0,
        stand=args.stand, roter_test=args.roter_test or "",
        meldung=args.meldung or "", versuche=list(args.versuch or ()),
        vermutung=args.vermutung,
        grenze_begruendung=zeitgrenze.grenze(args.tests_dauer).begruendung)
    try:
        text = zeitgrenze.notiz(eintrag)
    except zeitgrenze.NotizUnvollstaendig as fehler:
        print(str(fehler))
        return 1
    pfad = zeitgrenze.festhalten(eintrag)
    print(text)
    print("\nAbgelegt: %s" % pfad)
    print("Der Branch und die begonnene Arbeit bleiben unveraendert liegen.")
    return 0


def _bericht(args):
    text = morgenbericht.sammle_und_baue(
        basis=args.basis, testlauf_json=args.testlauf,
        protokolle=args.protokoll or ())
    pfad = morgenbericht.schreibe(text, pfad=args.ziel)
    print(text)
    print("\nGeschrieben: %s" % pfad)
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
            args.nummer, args.titel, args.dateien,
            beschreibung=args.beschreibung or "",
            tests=args.tests or "")
    except branch.NachtlaufFehler as fehler:
        print(str(fehler))
        return 1
    print("Commit %s fuer #%d angelegt (%d Datei(en))."
          % (kurz, args.nummer, len(args.dateien)))
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
    a.add_argument("--dateien", nargs="+", required=True,
                   help="genau die Dateien dieses Tickets (git-add-Pfade), "
                        "z.B. gui/x.py tests/smoke_test_x.py - kein "
                        "'git add -A' mehr, siehe branch.py")
    a.add_argument("--beschreibung", default="")
    a.add_argument("--tests", default="",
                   help='z.B. "gruen - tests/smoke_test_nachtbetrieb.py"')
    a.set_defaults(funktion=_abschliessen)

    unter.add_parser("stand", help="Commits, Tickets, Testlage, Sperrlage"
                     ).set_defaults(funktion=_stand)

    z = unter.add_parser("zuruecknehmen", help="genau ein Ticket rueckgaengig")
    z.add_argument("nummer", type=int)
    z.set_defaults(funktion=_zuruecknehmen)

    g = unter.add_parser("grenze",
                         help="wieviel Zeit ein Ticket bekommt, und warum")
    g.add_argument("--tests-dauer", type=float, default=0.0,
                   help="GEMESSENE Laufzeit der Tests dieses Tickets in "
                        "Sekunden, aus tools/testlauf.py --bericht")
    g.set_defaults(funktion=_grenze)

    k = unter.add_parser(
        "steckengeblieben",
        help="Ticket sauber abbrechen: Notiz schreiben, sonst nichts anfassen")
    k.add_argument("nummer", type=int)
    k.add_argument("titel")
    k.add_argument("--gelaufen", type=float, required=True,
                   help="wie lange das Ticket schon lief, in Minuten")
    k.add_argument("--tests-dauer", type=float, default=0.0)
    k.add_argument("--stand", required=True,
                   help="wo die Arbeit steht - was fertig ist, was halb")
    k.add_argument("--roter-test", default="",
                   help="welcher Test rot ist (darf fehlen, wenn keiner lief)")
    k.add_argument("--meldung", default="",
                   help="seine Fehlermeldung, nicht nur sein Name")
    k.add_argument("--versuch", action="append", default=[],
                   help="was schon probiert wurde; mehrfach angebbar")
    k.add_argument("--vermutung", required=True,
                   help="woran es als naechstes liegen koennte")
    k.set_defaults(funktion=_steckengeblieben)

    b = unter.add_parser("bericht", help="der Morgenbericht auf einer Seite")
    b.add_argument("--testlauf", default=None,
                   help="JSON aus tools/testlauf.py --bericht")
    b.add_argument("--protokoll", action="append", default=[],
                   help="mitgeschriebene Ausgabe fuer Block 4; mehrfach")
    b.add_argument("--ziel", default=None,
                   help="Zieldatei (Vorgabe: nachtbetrieb/laufberichte/"
                        "morgenbericht.md)")
    b.set_defaults(funktion=_bericht)
    return p


if __name__ == "__main__":
    argumente = baue_parser().parse_args()
    sys.exit(argumente.funktion(argumente))
