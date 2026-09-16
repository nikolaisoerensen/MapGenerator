"""
Path: nachtbetrieb/sperre.py

Setzt nachtbetrieb/sperrliste.toml durch.

WARUM ES DAS GIBT: nachts arbeitet ein Agent allein, und die teuersten Fehler
dieses Projekts waren nie Abstuerze, sondern Aenderungen, die plausibel
aussahen und still etwas kaputtmachten - ein numpy-Upgrade, das den JIT
abschaltet (Faktor 213), ein aufgeweichtes Testband, ein erneuertes
Referenzbild. Nichts davon wird rot. Die Sperrliste ist der Zaun darum.

DIE NAHT: alles laeuft ueber pruefe(dateien). Wer wissen will, ob eine
Aenderung durchdarf, uebergibt Pfade und bekommt Treffer zurueck - ob die
Pfade aus git diff, aus einem Test oder aus der Hand kommen, ist dieser
Funktion gleich. Genau deshalb kann tests/smoke_test_nachtbetrieb.py die
Durchsetzung vorfuehren, ohne einen Nachtlauf zu starten.
"""

import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

WURZEL = Path(__file__).resolve().parent.parent
SPERRLISTE = Path(__file__).resolve().parent / "sperrliste.toml"

# Der Selbstschutz. Diese Muster MUESSEN in der geladenen Liste stehen, sonst
# koennte ein Nachtlauf die Sperrliste entschaerfen und danach alles andere
# anfassen. Fehlen sie, laedt die Liste nicht - lieber gar kein Nachtlauf als
# einer ohne Zaun.
SELBSTSCHUTZ = ("nachtbetrieb/sperrliste.toml", "nachtbetrieb/sperre.py")

STUFEN = ("sperre", "warnung")


class SperrlisteKaputt(Exception):
    """Die Sperrliste ist unlesbar, unvollstaendig oder ohne Selbstschutz."""


@dataclass(frozen=True)
class Eintrag:
    name: str
    stufe: str
    muster: tuple
    ausnahme: tuple
    grund: str


@dataclass(frozen=True)
class Treffer:
    datei: str
    eintrag: str
    muster: str
    stufe: str
    grund: str

    @property
    def bricht_ab(self):
        return self.stufe == "sperre"


def _glob_zu_regex(muster):
    """Glob in Regex, mit dem ueblichen Unterschied zwischen * und **.

    Ein einzelner Stern trifft keinen Schraegstrich, ein doppelter schon.
    Zwei Sterne mit Schraegstrich duerfen auch NICHTS treffen, damit ein
    Muster wie **/bandgrenzen*.toml auch eine Datei direkt in der Wurzel
    findet - sonst waere das Muster nur fuer Unterordner gut, und niemand
    merkte es, bis die Datei dort landet.
    """
    teile = []
    i = 0
    while i < len(muster):
        if muster.startswith("**/", i):
            teile.append("(?:.*/)?")
            i += 3
        elif muster.startswith("**", i):
            teile.append(".*")
            i += 2
        elif muster[i] == "*":
            teile.append("[^/]*")
            i += 1
        elif muster[i] == "?":
            teile.append("[^/]")
            i += 1
        else:
            teile.append(re.escape(muster[i]))
            i += 1
    return re.compile("".join(teile) + r"\Z")


def _normiere(pfad):
    """Pfad auf die Schreibweise bringen, in der die Muster formuliert sind."""
    text = str(pfad).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def lade_sperrliste(pfad=None):
    """Liest die Sperrliste und prueft sie auf Vollstaendigkeit.

    Wirft SperrlisteKaputt, wenn ein Eintrag unvollstaendig ist, eine
    unbekannte Stufe traegt oder der Selbstschutz fehlt. Kein stiller
    Rueckfall auf eine leere Liste - eine leere Sperrliste sieht von aussen
    aus wie eine erfuellte Pruefung.
    """
    pfad = Path(pfad) if pfad else SPERRLISTE
    if not pfad.exists():
        raise SperrlisteKaputt("Sperrliste fehlt: %s" % pfad)
    try:
        roh = tomllib.loads(pfad.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as fehler:
        raise SperrlisteKaputt("Sperrliste ist kein gueltiges TOML: %s" % fehler)

    eintraege = []
    for nr, e in enumerate(roh.get("eintrag", []), start=1):
        fehlend = [f for f in ("name", "stufe", "muster", "grund") if not e.get(f)]
        if fehlend:
            raise SperrlisteKaputt(
                "Eintrag %d ist unvollstaendig, es fehlt: %s"
                % (nr, ", ".join(fehlend)))
        if e["stufe"] not in STUFEN:
            raise SperrlisteKaputt(
                "Eintrag %s hat die unbekannte Stufe %s. Erlaubt: %s"
                % (e["name"], e["stufe"], ", ".join(STUFEN)))
        eintraege.append(Eintrag(
            name=e["name"],
            stufe=e["stufe"],
            muster=tuple(e["muster"]),
            ausnahme=tuple(e.get("ausnahme", ())),
            grund=e["grund"].strip(),
        ))

    if not eintraege:
        raise SperrlisteKaputt("Die Sperrliste ist leer.")

    alle_muster = {m for e in eintraege if e.stufe == "sperre" for m in e.muster}
    fehlt = [m for m in SELBSTSCHUTZ if m not in alle_muster]
    if fehlt:
        raise SperrlisteKaputt(
            "Der Selbstschutz fehlt. Diese Muster muessen auf der Stufe sperre "
            "in der Liste stehen, sonst kann ein Nachtlauf die Sperrliste "
            "selbst aendern: " + ", ".join(fehlt))
    return eintraege


def pruefe(dateien, sperrliste=None):
    """Die Naht. Pfade rein, Treffer raus - sortiert, Sperren zuerst."""
    eintraege = sperrliste if sperrliste is not None else lade_sperrliste()
    treffer = []
    for datei in dateien:
        pfad = _normiere(datei)
        for e in eintraege:
            if any(_glob_zu_regex(a).match(pfad) for a in e.ausnahme):
                continue
            passendes = next(
                (m for m in e.muster if _glob_zu_regex(m).match(pfad)), None)
            if passendes:
                treffer.append(Treffer(pfad, e.name, passendes, e.stufe, e.grund))
    return sorted(treffer, key=lambda t: (t.stufe != "sperre", t.datei))


def geaenderte_dateien(basis="main", stand="HEAD", repo=None):
    """Welche Dateien zwischen basis und stand angefasst wurden.

    stand="ARBEITSSTAND" nimmt stattdessen alles, was gerade uncommittet
    herumliegt - das ist der Fall, den ein Agent vor seinem Commit braucht.
    Neu angelegte, noch nicht bekannte Dateien zaehlen mit; sonst koennte man
    eine gesperrte Datei anlegen, ohne dass die Pruefung sie sieht.
    """
    repo = Path(repo) if repo else WURZEL
    if stand == "ARBEITSSTAND":
        befehle = [["git", "diff", "--name-only", "HEAD"],
                   ["git", "ls-files", "--others", "--exclude-standard"]]
    else:
        befehle = [["git", "diff", "--name-only", "%s...%s" % (basis, stand)]]
    dateien = []
    for befehl in befehle:
        ergebnis = subprocess.run(befehl, cwd=str(repo), capture_output=True,
                                  text=True, encoding="utf-8")
        if ergebnis.returncode != 0:
            raise RuntimeError("%s schlug fehl: %s"
                               % (" ".join(befehl), ergebnis.stderr.strip()))
        dateien += [z for z in ergebnis.stdout.splitlines() if z.strip()]
    return sorted(set(dateien))


def notiz(treffer):
    """Die verstaendliche Notiz fuers Ticket - welche Datei, welche Regel, warum.

    Absichtlich als Text und nicht als Log: sie wird so, wie sie ist, in einen
    Ticketkommentar kopiert. "Sperrliste verletzt" waere keine Notiz, das waere
    eine Absage.
    """
    if not treffer:
        return "Keine gesperrte Datei beruehrt."
    sperren = [t for t in treffer if t.bricht_ab]
    warnungen = [t for t in treffer if not t.bricht_ab]
    zeilen = []
    if sperren:
        zeilen.append("ABGEBROCHEN: der Lauf hat gesperrte Dateien angefasst.")
        zeilen.append("")
        for t in sperren:
            zeilen.append("  " + t.datei)
            zeilen.append("    gesperrt durch: %s  (Muster %s)" % (t.eintrag, t.muster))
            for z in t.grund.splitlines():
                zeilen.append("    " + z if z.strip() else "")
            zeilen.append("")
        zeilen.append("Was jetzt zu tun ist: das Ticket stehenlassen und diese "
                      "Notiz hineinschreiben. Die Sperrliste wird NICHT "
                      "geaendert - Eintraege entfernt nur der Nutzer.")
    if warnungen:
        if sperren:
            zeilen.append("")
        zeilen.append("Zur Kenntnis (bricht nicht ab, gehoert in den Morgenbericht):")
        for t in warnungen:
            zeilen.append("  %s - %s (Muster %s)" % (t.datei, t.eintrag, t.muster))
    return "\n".join(zeilen).rstrip()


def pruefe_arbeitsstand(repo=None):
    """Was ein Agent vor seinem Commit aufruft. Gibt (darf_weiter, Notiz)."""
    treffer = pruefe(geaenderte_dateien(stand="ARBEITSSTAND", repo=repo))
    return (not any(t.bricht_ab for t in treffer)), notiz(treffer)


def pruefe_branch(basis="main", stand="HEAD", repo=None):
    """Was vor dem Merge aufgerufen wird. Gibt (darf_gemergt_werden, Notiz)."""
    treffer = pruefe(geaenderte_dateien(basis=basis, stand=stand, repo=repo))
    return (not any(t.bricht_ab for t in treffer)), notiz(treffer)


if __name__ == "__main__":
    ziel = sys.argv[1] if len(sys.argv) > 1 else "ARBEITSSTAND"
    if ziel == "ARBEITSSTAND":
        ok, text = pruefe_arbeitsstand()
    else:
        ok, text = pruefe_branch(stand=ziel)
    print(text)
    sys.exit(0 if ok else 1)
