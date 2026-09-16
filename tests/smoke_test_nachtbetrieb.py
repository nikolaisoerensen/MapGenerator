"""
Path: tests/smoke_test_nachtbetrieb.py

Prueft das Nachtverfahren: Sperrliste (#57), Branchverfahren (#58),
Zeitgrenze samt Steckenbleib-Notiz (#59) und Morgenbericht (#60).

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

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nachtbetrieb import branch, morgenbericht, sperre, zeitgrenze  # noqa: E402


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


def run_grenze_waechst_mit_den_tests():
    print("\n--- Die Zeitgrenze wird gerechnet, nicht geraten ---")
    faelle = [
        (0.0, "ohne eigene Tests"),
        (2.0, "ein schneller Test"),
        (35.0, "der langsamste Schnelltest (erosion_quality, gemessen)"),
        (120.0, "die Eichungsreihe smoke_test_regionen_welt (rund 2 min)"),
        (161.0, "die ganze Schnellreihe, 28 Dateien (gemessen 2026-09-16)"),
        (1200.0, "Tests, die allein 20 Minuten brauchen"),
    ]
    for dauer, was in faelle:
        g = zeitgrenze.grenze(dauer)
        print("  %-58s %5.0f s Tests -> %3.0f min%s"
              % (was, dauer, g.minuten, "  (gedeckelt)" if g.gedeckelt else ""))
    # Die Forderung des Tickets woertlich: ein Ticket, dessen Tests allein
    # schon 20 Minuten brauchen, kann keine 30-Minuten-Grenze haben.
    lang = zeitgrenze.grenze(1200.0)
    kurz = zeitgrenze.grenze(0.0)
    ok = (kurz.sekunden == zeitgrenze.GRUNDGRENZE_S
          and lang.sekunden > kurz.sekunden
          and lang.minuten >= 120
          and lang.gedeckelt
          and "Streufaktor" in kurz.begruendung)
    print("  20-Minuten-Tests bekommen %.0f min, nicht 30 - wie gefordert."
          % lang.minuten)
    print("  Monoton: %s" % all(
        zeitgrenze.grenze(a).sekunden <= zeitgrenze.grenze(b).sekunden
        for a, b in zip([0, 2, 35, 120, 161, 1200], [2, 35, 120, 161, 1200, 3000])))
    return ok


def run_notiz_verweigert_die_uhr():
    print("\n--- 'Zeitlimit erreicht' wird als Notiz abgelehnt ---")
    nackt = zeitgrenze.Steckenbleib(nummer=77, titel="Irgendwas",
                                    grenze_s=1800, verstrichen_s=1800.0)
    try:
        zeitgrenze.notiz(nackt)
    except zeitgrenze.NotizUnvollstaendig as fehler:
        print("  abgelehnt, wie es sein soll:")
        print("  " + str(fehler).replace("\n", "\n  "))
    else:
        print("  FEHLER: die leere Notiz wurde angenommen.")
        return False

    # Roter Test ohne Meldung ist ebenfalls zu wenig - der Name eines Tests
    # sagt nicht, woran er scheitert.
    halb = zeitgrenze.Steckenbleib(
        nummer=77, titel="Irgendwas", grenze_s=1800, verstrichen_s=1800.0,
        stand="halb", versuche=["a"], vermutung="b",
        roter_test="tests/smoke_test_x.py")
    try:
        zeitgrenze.notiz(halb)
        print("  FEHLER: roter Test ohne Meldung ging durch.")
        return False
    except zeitgrenze.NotizUnvollstaendig:
        print("  Roter Test ohne Fehlermeldung: ebenfalls abgelehnt.")
    return True


def run_abbruch_laesst_alles_liegen():
    print("\n--- Kuenstlich verzoegertes Ticket: Abbruch ohne Verlust ---")
    repo = _wegwerf_repo()
    ordner = Path(tempfile.mkdtemp(prefix="laufbericht_"))
    try:
        branch.starte_nacht(repo=repo)
        vorher_branch = branch.aktueller_branch(repo=repo)
        (repo / "halbe_arbeit.py").write_text("# bis hierhin gekommen\n",
                                              encoding="utf-8")

        # Die Verzoegerung ist eine gestellte Uhr, kein echtes Warten. Ein
        # Test, der eine halbe Stunde schlaeft, wird abgeschaltet - und dann
        # prueft niemand mehr den Abbruch.
        g = zeitgrenze.grenze(120.0)
        gestellt = [0.0]
        uhr = zeitgrenze.Uhr(g.sekunden, jetzt=lambda: gestellt[0])
        print("  Grenze: %.0f min. Nach 10 min abgelaufen? %s"
              % (g.minuten, (gestellt.__setitem__(0, 600) or uhr.abgelaufen())))
        gestellt[0] = g.sekunden + 90
        print("  Nach %.0f min abgelaufen? %s  (Rest %.0f s)"
              % (gestellt[0] / 60.0, uhr.abgelaufen(), uhr.rest()))
        if not uhr.abgelaufen():
            print("  FEHLER: die Uhr laeuft nicht ab.")
            return False

        eintrag = zeitgrenze.Steckenbleib(
            nummer=104, titel="Erosionskanaele zusammenhaengend bekommen",
            grenze_s=g.sekunden, verstrichen_s=uhr.verstrichen(),
            grenze_begruendung=g.begruendung,
            stand="Schwellenlogik in core/erosion_generator.py umgestellt, "
                  "Kanalnetz-Test laeuft, Sedimentation noch unberuehrt.",
            roter_test="tests/smoke_test_erosion_quality.py",
            meldung="(c) zusammenhaengendes Kanalnetz "
                    "(groesste Komponente 45 px, Schwelle > 60)",
            versuche=["Schwelle von 0.6 auf 0.4 gesenkt - Komponente wuchs "
                      "auf 51 px, reicht nicht",
                      "Erosionsschritte verdoppelt - Laufzeit x2, Komponente "
                      "unveraendert"],
            vermutung="Nicht die Schwelle, sondern die Reihenfolge: die "
                      "Sedimentation fuellt die Rinne wieder auf, bevor der "
                      "naechste Schritt sie vertieft. Erst die Kopplung "
                      "messen, dann wieder an Zahlen drehen.")
        pfad = zeitgrenze.festhalten(eintrag, ordner)
        text = zeitgrenze.notiz(eintrag)
        print("  --- die Notiz ---")
        print("  " + text.replace("\n", "\n  "))

        # Was der saubere Abbruch NICHT getan haben darf.
        noch_da = (repo / "halbe_arbeit.py").exists()
        offen = bool(_git(repo, "status", "--porcelain"))
        gleicher_branch = branch.aktueller_branch(repo=repo) == vorher_branch
        keine_commits = len(branch.commits_der_nacht(repo=repo)) == 0
        print("\n  Arbeit liegt noch da:      %s" % noch_da)
        print("  uncommittet geblieben:     %s" % offen)
        print("  Branch unveraendert:       %s (%s)" % (gleicher_branch,
                                                        vorher_branch))
        print("  kein Commit angelegt:      %s" % keine_commits)
        print("  Notiz abgelegt:            %s" % pfad.name)

        gesammelt = zeitgrenze.sammle(ordner)
        gefunden = len(gesammelt) == 1 and gesammelt[0].nummer == 104
        print("  wieder einlesbar:          %s" % gefunden)
        return all([noch_da, offen, gleicher_branch, keine_commits, gefunden])
    finally:
        shutil.rmtree(repo, ignore_errors=True)
        shutil.rmtree(ordner, ignore_errors=True)


def run_rueckfallmarken_zeigen_ins_ziel():
    print("\n--- Jede Rueckfallmarke kommt im Quelltext wirklich vor ---")
    wurzel = Path(__file__).resolve().parent.parent
    quellen = []
    for ordner in ("core", "managers", "gui"):
        for pfad in (wurzel / ordner).rglob("*.py"):
            quellen.append(pfad.read_text(encoding="utf-8", errors="replace"))
    gesamt = "\n".join(quellen)
    ok = True
    for marke in morgenbericht.RUECKFAELLE:
        trifft = bool(re.search(marke.muster, gesamt))
        ok = ok and trifft
        print("  %-38s %s  %s" % (marke.name, "ja " if trifft else "NEIN",
                                  marke.ort))
    if not ok:
        print("  Eine Marke, die ins Leere zeigt, meldet fuer immer null "
              "Rueckfaelle - und null sieht aus wie 'alles in Ordnung'.")
    return ok


def run_bericht_hat_vier_bloecke():
    print("\n--- Der Morgenbericht: vier Bloecke plus Steckengebliebene ---")
    testlauf = {"gesamtdauer_s": 161.0, "ergebnisse": [
        {"datei": "smoke_test_a.py", "code": 0, "dauer": 1.0, "fails": [],
         "ausgabe": "alles gut"},
        {"datei": "smoke_test_erosion_quality.py", "code": 1, "dauer": 35.0,
         "fails": ["[FAIL] (c) zusammenhaengendes Kanalnetz (45 px)",
                   "[FAIL] (d) Sedimentation erzeugt Ebenen (7.2%)",
                   "[FAIL] Schwelle senkt Erosionsanteil",
                   "[FAIL] Schrittzahl bleibt vergleichbar"],
         "ausgabe": "WARNING core.erosion_generator GPU-Erosion "
                    "fehlgeschlagen (kein Kontext) - CPU-Pfad\n"
                    "WARNING GPU-Erosion nicht verfuegbar () - CPU-Pfad\n"
                    "DEBUG: Adaptives Mesh nicht anwendbar (Heightmap-"
                    "Groesse) - Gleichmaessig-Gitter"},
    ]}
    treffer, zeilen = morgenbericht.zaehle_rueckfaelle(
        [e["ausgabe"] for e in testlauf["ergebnisse"]])
    heute = morgenbericht.kennzahlen_aus_testlauf(testlauf)
    gestern = {"Testdateien gruen": 2, "Testdateien rot": 0,
               "Gesamtlaufzeit der Tests (s)": 140.0}
    steckenbleib = [zeitgrenze.Steckenbleib(
        nummer=104, titel="Erosionskanaele", grenze_s=1980,
        verstrichen_s=2070.0, stand="halb", versuche=["x"],
        vermutung="Reihenfolge von Erosion und Sedimentation")]

    text = morgenbericht.baue(
        commits=[{"nummer": 57, "titel": "Sperrliste anlegen",
                  "tests": "gruen - tests/smoke_test_nachtbetrieb.py"}],
        testlauf=testlauf, kennzahlen_heute=heute, kennzahlen_gestern=gestern,
        rueckfaelle=treffer, durchsuchte_zeilen=zeilen,
        steckengeblieben=steckenbleib, branch="nacht/2026-09-16",
        datum="2026-09-17 07:00")
    print("  " + text.replace("\n", "\n  "))

    noetig = {
        "Block 1": "#57",
        "Block 2 nennt die Meldung": "zusammenhaengendes Kanalnetz",
        "Block 2 kuerzt ehrlich": "und 1 weitere",
        "Block 3 Vortagswert": "140.0",
        "Block 3 Prozent": "%",
        "Block 4 Anzahl": "2x",
        "Block 4 Ort": "core/erosion_generator.py",
        "Steckengeblieben": "#104",
        "Quellen genannt": "Quellen:",
    }
    ok = True
    print("")
    for was, wort in noetig.items():
        da = wort in text
        ok = ok and da
        print("  %-30s %s" % (was, "ok" if da else "FEHLT: " + wort))
    zeilenzahl = len(text.splitlines())
    passt = zeilenzahl <= 60
    print("  %-30s %d Zeilen %s" % ("Eine Seite", zeilenzahl,
                                    "ok" if passt else "ZU LANG"))
    return ok and passt


def run_bericht_ereignislos_und_leer():
    print("\n--- Ruhige Nacht in einem Satz, leere Messung als Befund ---")
    ruhig = morgenbericht.baue(
        commits=[{"nummer": 57, "titel": "Sperrliste", "tests": "gruen"}],
        testlauf={"gesamtdauer_s": 1.0, "ergebnisse":
                  [{"datei": "a.py", "code": 0, "dauer": 1.0, "fails": [],
                    "ausgabe": ""}]},
        kennzahlen_heute={"Testdateien gruen": 1},
        durchsuchte_zeilen=12, datum="2026-09-17 07:00")
    print("  " + ruhig.replace("\n", "\n  "))
    hat_satz = "Ruhige Nacht" in ruhig

    leer = morgenbericht.baue(durchsuchte_zeilen=0, datum="2026-09-17 07:00")
    ehrlich = ("Null durchsuchte Zeilen heisst NICHT null Rueckfaelle" in leer
               and "wurde nicht gemessen" in leer)
    print("\n  Ruhige Nacht in einem Satz gesagt:            %s" % hat_satz)
    print("  Nichtmessung wird als Nichtmessung gemeldet:  %s" % ehrlich)
    if not ehrlich:
        print("  Ein leerer Block, der aussieht wie ein gruener, ist genau "
              "der Fehler, gegen den dieser Bericht gebaut ist.")
    return hat_satz and ehrlich


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
        "grenze_waechst_mit_den_tests": run_grenze_waechst_mit_den_tests(),
        "notiz_verweigert_die_uhr": run_notiz_verweigert_die_uhr(),
        "abbruch_laesst_alles_liegen": run_abbruch_laesst_alles_liegen(),
        "rueckfallmarken_zeigen_ins_ziel": run_rueckfallmarken_zeigen_ins_ziel(),
        "bericht_hat_vier_bloecke": run_bericht_hat_vier_bloecke(),
        "bericht_ereignislos_und_leer": run_bericht_ereignislos_und_leer(),
    }
    print("\n=== SUMMARY ===")
    for name, ergebnis in ergebnisse.items():
        print("%-36s %s" % (name, "PASS" if ergebnis else "FAIL"))
    sys.exit(0 if all(ergebnisse.values()) else 1)
