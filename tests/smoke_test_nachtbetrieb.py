"""
Path: tests/smoke_test_nachtbetrieb.py

Prueft das Nachtverfahren: Sperrliste (#57) und Branchverfahren (#58).

Der Test baut sich fuer die Branch-Teile ein eigenes Wegwerf-Repository in
einem Temperaeverzeichnis und arbeitet dort. Er fasst das echte Repository
NICHT an - ein Test, der zum Vorfuehren des Ruecknehmens im Projekt selbst
committet, waere genau die Sorte Nebenwirkung, gegen die der Nachtbetrieb
gebaut ist.

Die Sperrliste dagegen wird die ECHTE geprueft, nicht eine ausgedachte. Sonst
pruefte der Test eine ausgedachte Situation - derselbe Fehler wie beim
adaptiven Netz, das monatelang nur mit Groessen getestet wurde, die im
Programm nicht vorkommen.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nachtbetrieb import branch, sperre  # noqa: E402


def _git(repo, *args):
    ergebnis = subprocess.run(("git",) + args, cwd=str(repo), capture_output=True,
                              text=True, encoding="utf-8")
    if ergebnis.returncode != 0:
        raise RuntimeError("git %s: %s" % (" ".join(args), ergebnis.stderr))
    return ergebnis.stdout.strip()


def _wegwerf_repo():
    """Ein frisches Repository mit einem Commit auf main."""
    ordner = Path(tempfile.mkdtemp(prefix="nachtbetrieb_"))
    _git(ordner, "init", "-q", "-b", "main")
    _git(ordner, "config", "user.email", "test@example.invalid")
    _git(ordner, "config", "user.name", "Nachtbetrieb-Test")
    (ordner / "start.txt").write_text("Anfang\n", encoding="utf-8")
    _git(ordner, "add", "-A")
    _git(ordner, "commit", "-q", "-m", "Anfang")
    return ordner


def run_sperrliste_laedt():
    print("\n--- Die echte Sperrliste laedt und ist vollstaendig ---")
    eintraege = sperre.lade_sperrliste()
    print("Eintraege: %d" % len(eintraege))
    ok = len(eintraege) >= 1
    for e in eintraege:
        vollstaendig = bool(e.name and e.muster and e.grund)
        stufe_ok = e.stufe in sperre.STUFEN
        print("  %-34s %-8s %d Muster  Grund: %s"
              % (e.name, e.stufe, len(e.muster), "ja" if e.grund else "FEHLT"))
        ok = ok and vollstaendig and stufe_ok
    return ok


def run_selbstschutz_greift():
    print("\n--- Eine Sperrliste ohne Selbstschutz wird abgelehnt ---")
    ordner = Path(tempfile.mkdtemp(prefix="sperrliste_"))
    try:
        entschaerft = ordner / "sperrliste.toml"
        entschaerft.write_text(
            '[[eintrag]]\nname = "Nur irgendwas"\nstufe = "sperre"\n'
            'muster = ["egal.txt"]\ngrund = "keiner"\n', encoding="utf-8")
        try:
            sperre.lade_sperrliste(entschaerft)
        except sperre.SperrlisteKaputt as fehler:
            print("  abgelehnt, wie es sein soll:")
            print("  " + str(fehler).replace("\n", "\n  "))
            return True
        print("  FEHLER: die entschaerfte Liste wurde angenommen.")
        return False
    finally:
        shutil.rmtree(ordner, ignore_errors=True)


def run_muster_treffen():
    print("\n--- Muster treffen die richtigen Dateien ---")
    erwartet = [
        ("requirements.txt", "sperre"),
        ("shaders/erosion/hydraulic.comp", "sperre"),
        ("tests/daten/referenz.json", "sperre"),
        ("bandgrenzen_wasser.toml", "sperre"),
        ("core/bandgrenzen_biome.json", "sperre"),
        ("nachtbetrieb/sperrliste.toml", "sperre"),
        ("nachtbetrieb/sperre.py", "sperre"),
        ("CLAUDE.md", "sperre"),
        (".github/workflows/ci.yml", "sperre"),
        ("core/terrain_weltkarte.py", "warnung"),
        ("docs/TESTBERICHT.md", "warnung"),
        ("gui/tabs/biome_tab.py", None),
        ("core/terrain_generator.py", None),
        ("docs/SITZUNGSLOG_ALT.md", None),
    ]
    liste = sperre.lade_sperrliste()
    ok = True
    for pfad, soll in erwartet:
        treffer = sperre.pruefe([pfad], liste)
        ist = treffer[0].stufe if treffer else None
        passt = ist == soll
        ok = ok and passt
        print("  %-38s soll %-8s ist %-8s %s"
              % (pfad, soll or "frei", ist or "frei", "ok" if passt else "FALSCH"))
    return ok


def run_durchsetzung_bricht_ab():
    print("\n--- Eine gesperrte Datei bricht den Lauf ab, mit lesbarer Notiz ---")
    treffer = sperre.pruefe(["requirements.txt", "core/terrain_generator.py"])
    if not any(t.bricht_ab for t in treffer):
        print("  FEHLER: requirements.txt wurde durchgelassen.")
        return False
    text = sperre.notiz(treffer)
    print("  " + text.replace("\n", "\n  "))
    noetig = ["requirements.txt", "Abhaengigkeiten", "numba"]
    fehlt = [w for w in noetig if w not in text]
    if fehlt:
        print("  FEHLER: in der Notiz fehlt: %s" % ", ".join(fehlt))
        return False
    print("  Die Notiz nennt Datei, Regel und Grund.")
    return True


def run_warnung_bricht_nicht_ab():
    print("\n--- Eine Warnung wird gemeldet, bricht aber nicht ab ---")
    treffer = sperre.pruefe(["docs/TESTBERICHT.md"])
    darf = not any(t.bricht_ab for t in treffer)
    print("  Treffer: %d, Lauf darf weiter: %s" % (len(treffer), darf))
    text = sperre.notiz(treffer)
    print("  " + text.replace("\n", "\n  "))
    return darf and len(treffer) == 1 and "Morgenbericht" in text


def run_branchname_kollidiert_nicht():
    print("\n--- Ein liegengebliebener Branch kollidiert nicht ---")
    repo = _wegwerf_repo()
    try:
        erster = branch.naechster_branchname(repo=repo)
        print("  erster Abend:  %s" % erster)
        _git(repo, "branch", erster)
        zweiter = branch.naechster_branchname(repo=repo)
        print("  zweiter Abend: %s" % zweiter)
        _git(repo, "branch", zweiter)
        dritter = branch.naechster_branchname(repo=repo)
        print("  dritter Abend: %s" % dritter)
        return (erster != zweiter != dritter
                and zweiter.endswith("-b") and dritter.endswith("-c"))
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def run_kein_commit_auf_main():
    print("\n--- Auf main schliesst kein Ticket ab ---")
    repo = _wegwerf_repo()
    try:
        (repo / "neu.txt").write_text("x\n", encoding="utf-8")
        try:
            branch.ticket_abschliessen(99, "Darf nicht", repo=repo)
        except branch.NachtlaufFehler as fehler:
            print("  verweigert, wie es sein soll: %s"
                  % str(fehler).splitlines()[0])
            return True
        print("  FEHLER: es wurde auf main committet.")
        return False
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def run_ein_commit_je_ticket_und_ruecknahme():
    print("\n--- Zwei Tickets, zwei Commits, eines davon zurueckgenommen ---")
    repo = _wegwerf_repo()
    try:
        name = branch.starte_nacht(repo=repo)
        print("  Nachtbranch: %s" % name)

        (repo / "ticket_a.txt").write_text("Arbeit A\n", encoding="utf-8")
        branch.ticket_abschliessen(101, "Erstes Ticket", repo=repo,
                                   tests="gruen - tests/smoke_test_a.py")
        (repo / "ticket_b.txt").write_text("Arbeit B\n", encoding="utf-8")
        branch.ticket_abschliessen(102, "Zweites Ticket", repo=repo,
                                   tests="gruen - tests/smoke_test_b.py")

        print("  --- stand() ---")
        print("  " + branch.stand(repo=repo).replace("\n", "\n  "))

        eintraege = branch.commits_der_nacht(repo=repo)
        if len(eintraege) != 2:
            print("  FEHLER: %d Commits statt 2." % len(eintraege))
            return False
        if [e["nummer"] for e in eintraege] != [101, 102]:
            print("  FEHLER: falsche Ticketnummern: %s"
                  % [e["nummer"] for e in eintraege])
            return False
        if eintraege[0]["tests"] != "gruen - tests/smoke_test_a.py":
            print("  FEHLER: der Teststand wurde nicht zurueckgelesen.")
            return False

        branch.ticket_zuruecknehmen(102, repo=repo)
        a_da = (repo / "ticket_a.txt").exists()
        b_weg = not (repo / "ticket_b.txt").exists()
        print("  nach der Ruecknahme von #102: ticket_a.txt vorhanden=%s, "
              "ticket_b.txt vorhanden=%s" % (a_da, not b_weg))
        if not (a_da and b_weg):
            print("  FEHLER: die Ruecknahme hat das falsche Ticket getroffen.")
            return False

        nachher = branch.commits_der_nacht(repo=repo)
        print("  Historie bleibt vollstaendig: %d Commits (2 Tickets + 1 "
              "Gegen-Commit)" % len(nachher))
        return len(nachher) == 3
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def run_sperre_verhindert_commit():
    print("\n--- Der Commit selbst faellt aus, wenn eine Sperre getroffen ist ---")
    repo = _wegwerf_repo()
    try:
        branch.starte_nacht(repo=repo)
        (repo / "requirements.txt").write_text("numpy==2.5.1\n", encoding="utf-8")
        try:
            branch.ticket_abschliessen(103, "numpy aktualisieren", repo=repo)
        except branch.NachtlaufFehler as fehler:
            print("  abgebrochen, wie es sein soll:")
            print("  " + str(fehler).replace("\n", "\n  "))
            offen = _git(repo, "status", "--porcelain")
            print("  Aenderung liegt weiterhin uncommittet da: %s"
                  % bool(offen))
            return bool(offen)
        print("  FEHLER: der Commit ging durch.")
        return False
    finally:
        shutil.rmtree(repo, ignore_errors=True)


if __name__ == "__main__":
    ergebnisse = {
        "sperrliste_laedt": run_sperrliste_laedt(),
        "selbstschutz_greift": run_selbstschutz_greift(),
        "muster_treffen": run_muster_treffen(),
        "durchsetzung_bricht_ab": run_durchsetzung_bricht_ab(),
        "warnung_bricht_nicht_ab": run_warnung_bricht_nicht_ab(),
        "branchname_kollidiert_nicht": run_branchname_kollidiert_nicht(),
        "kein_commit_auf_main": run_kein_commit_auf_main(),
        "ein_commit_je_ticket": run_ein_commit_je_ticket_und_ruecknahme(),
        "sperre_verhindert_commit": run_sperre_verhindert_commit(),
    }
    print("\n=== SUMMARY ===")
    for name, ergebnis in ergebnisse.items():
        print("%-32s %s" % (name, "PASS" if ergebnis else "FAIL"))
    sys.exit(0 if all(ergebnisse.values()) else 1)
