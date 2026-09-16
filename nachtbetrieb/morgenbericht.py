"""
Path: nachtbetrieb/morgenbericht.py

Der Morgenbericht: eine Seite, vier Bloecke, plus die Steckengebliebenen.

WARUM ES DAS GIBT - und der Grund ist Block 4, nicht Block 1. Was nachts
geschlossen wurde, steht ohnehin im git log. Was rot ist, meldet der Testlauf.
Aber ein stiller Rueckfall meldet sich NIE. Er sieht von aussen aus wie
Erfolg: das Programm rechnet, die Tests sind gruen, nur eben auf dem
Ersatzpfad. Genau so lief das adaptive 3D-Netz monatelang gar nicht, und genau
so rechnete die halbe Pipeline nach einem Dateiumzug ohne GPU weiter. Block 4
ist der einzige Ort, an dem so etwas laut wird, ohne dass jemand danach sucht.

WORAUS ER GEBAUT IST - und woraus ausdruecklich nicht:

  Block 1 Geschlossen   aus git log ueber branch.commits_der_nacht()
  Block 2 Rot           aus tools/testlauf.py --bericht <json>
  Block 3 Kennzahlen    aus demselben Bericht, plus optional abgelegte Werte
  Block 4 Rueckfaelle   aus den mitgeschriebenen Ausgaben, gegen RUECKFAELLE
  Steckengeblieben      aus nachtbetrieb/zeitgrenze.sammle()

Es gibt in diesem Projekt KEIN dauerhaftes Logfile, in dem Warnungen landen:
main.py haengt einen StreamHandler an die Konsole, und nur der Error-Handler
schreibt eine Datei (map_generator_errors.log, nur Fehler). Ein Bericht, der
so tut, als lese er ein Warnungs-Log, wuerde immer null Rueckfaelle melden -
und null waere von "alles in Ordnung" nicht zu unterscheiden. Deshalb zaehlt
Block 4 ausschliesslich in Text, den der Nachtlauf SELBST mitgeschrieben hat,
und sagt ausdruecklich, wieviel Text er durchsucht hat.

EINE SEITE, NICHT ZWEI: der Bericht nennt Pfade und Zahlen, er enthaelt keine
Logs. Drei Beispielzeilen je Rueckfall, Fehlermeldungen auf 200 Zeichen. Ein
Bericht, den man scrollen muss, wird ueberflogen, und ueberflogen heisst
ungelesen.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

WURZEL = Path(__file__).resolve().parent.parent
LAUFORDNER = Path(__file__).resolve().parent / "laufberichte"
KENNZAHLEN = LAUFORDNER / "kennzahlen.json"
KENNZAHLEN_VORTAG = LAUFORDNER / "kennzahlen_vortag.json"

MAX_BEISPIELE = 3
MAX_MELDUNG = 200


@dataclass(frozen=True)
class Rueckfallmarke:
    """Ein bekannter Ersatzpfad und die Zeile, an der man ihn erkennt.

    Jede Marke traegt ihren Ort mit: ein Zaehler ohne Ort ist im Bericht
    wertlos, weil man morgens nicht suchen will, sondern nachsehen.
    """
    name: str
    muster: str
    ort: str
    warum: str


# Die Marken sind ECHTE Zeilen aus dem Code, nicht ausgedachte. Jede wurde in
# der Quelle nachgesehen; steht sie nicht mehr dort, faellt der Zaehler auf
# null, ohne dass es auffiele - deshalb prueft
# tests/smoke_test_nachtbetrieb.py, dass jedes Muster im Quelltext noch
# vorkommt. Ein Waechter, der ins Leere zeigt, ist schlimmer als keiner.
RUECKFAELLE = (
    Rueckfallmarke(
        "GPU-Erosion faellt auf CPU",
        r"GPU-Erosion (?:fehlgeschlagen|nicht verfuegbar)",
        "core/erosion_generator.py:775,778",
        "Der CPU-Pfad liefert gemessen Faktor 385 weniger Abtrag "
        "(38,5 m gegen 0,1 m). Das Ergebnis sieht trotzdem plausibel aus."),
    Rueckfallmarke(
        "Kein GL-4.3-Kontext",
        r"kein GL 4\.3\+ Kontext",
        "managers/shader_manager.py:1929",
        "Ohne Kontext laeuft ALLES auf CPU - nicht eine Operation, sondern "
        "die ganze Pipeline."),
    Rueckfallmarke(
        "Adaptives 3D-Netz nicht angewandt",
        r"Adaptives Mesh nicht anwendbar",
        "gui/widgets/map_display_3d.py:1404",
        "Genau diese Zeile war der einzige Hinweis darauf, dass das "
        "adaptive Netz monatelang nie lief."),
    Rueckfallmarke(
        "Wegsuche ohne numba",
        # Der Wortlaut aus core/wegsuche_schnell.py:229, nicht bloss "numba".
        # Das lose Wort traf beim ersten echten Lauf vier Docstring-Zeilen und
        # meldete einen Rueckfall, den es nicht gab. Ein Block 4 mit
        # Fehlalarmen wird nach drei Morgen nicht mehr gelesen, und dann ist
        # er schaedlicher als keiner.
        r"numba fehlt - Wegsuche laeuft im langsamen Python-Pfad",
        "core/wegsuche_schnell.py:229",
        "Ohne den uebersetzten Kern ist die Wegsuche gemessen 14x langsamer, "
        "bei identischem Pfad - also nur an der Uhr zu erkennen."),
    Rueckfallmarke(
        "Biom-Klassifikation auf Ersatzpfad",
        r"(?:CPU|Simple)-Fallback classification|simple fallback classification",
        "core/biome_generator.py:207,213,733",
        "Dreistufige Kette GPU-Shader -> CPU -> einfach. Stufe drei liefert "
        "immer noch eine vollstaendige Biomkarte."),
    Rueckfallmarke(
        "Wetter auf Einzelschicht",
        r"Einzelschicht-Fallback",
        "core/weather_generator.py:1369,1406",
        "Der gekoppelte Loop fiel aus; Wind und Feuchte kommen dann aus "
        "einer Schicht statt aus dem Zusammenspiel."),
    Rueckfallmarke(
        "Geologie mit Ersatzdaten",
        r"[Uu]sing fallback geology data|fallback values",
        "core/geology_generator.py:1028,1338",
        "Ersatzwerte statt der Parameter aus value_default.py - die Karte "
        "entsteht, stimmt aber mit keinem Regler ueberein."),
    Rueckfallmarke(
        "Siedlungs-Kostenflut auf CPU",
        r"terrain-cost-flood fehlgeschlagen",
        "core/settlement_generator.py:2120",
        "Betrifft die Eignungsfelder, auf denen jede Siedlungsplatzierung "
        "aufbaut."),
    Rueckfallmarke(
        "Shader nicht uebersetzt, Drahtgitter",
        r"Shader compilation failed",
        "gui/widgets/map_display_3d.py:826",
        "Die 3D-Ansicht zeigt dann Drahtgitter - sichtbar, aber nur, wenn "
        "jemand hinschaut."),
)


def _kuerze(text, laenge=MAX_MELDUNG):
    text = " ".join(str(text).split())
    return text if len(text) <= laenge else text[:laenge - 1] + "…"


def _prozent(heute, gestern):
    """Veraenderung in Prozent, oder None, wenn sie nicht rechenbar ist.

    Kein stiller Nuller: gestern 0 ergibt keine Prozentzahl, und eine
    erfundene 0 % waere eine Behauptung, die der Bericht nicht decken kann.
    """
    try:
        if gestern in (None, 0) or heute is None:
            return None
        return (float(heute) - float(gestern)) / abs(float(gestern)) * 100.0
    except (TypeError, ValueError):
        return None


def lies_testlauf(pfad):
    """Die JSON-Datei aus tools/testlauf.py --bericht."""
    pfad = Path(pfad)
    if not pfad.exists():
        return None
    return json.loads(pfad.read_text(encoding="utf-8"))


def kennzahlen_aus_testlauf(bericht):
    """Die Zahlen, die wirklich messbar vorliegen - nicht mehr.

    Bewusst wenige: gruen, rot, Gesamtlaufzeit. Die Regionseichwerte
    (Medianhang und Wasseranteil je Region) waeren die interessanteren
    Kennzahlen, aber sie stehen heute nur als Text in der Ausgabe von
    smoke_test_regionen_welt.py. Sie hier aus Text zu fischen hiesse, eine
    Zahl aus einem Format zu ziehen, das sich jederzeit aendern kann - und
    beim naechsten Umformulieren faende der Bericht nichts mehr, ohne es zu
    sagen. Sobald Ticket #29 die Regionsparameter herausloest, werden sie
    ueber kennzahlen_ablegen() eingespeist.
    """
    if not bericht:
        return {}
    ergebnisse = bericht.get("ergebnisse", [])
    gruen = len([e for e in ergebnisse if e.get("code") == 0])
    return {
        "Testdateien gruen": gruen,
        "Testdateien rot": len(ergebnisse) - gruen,
        "Gesamtlaufzeit der Tests (s)": round(
            float(bericht.get("gesamtdauer_s", 0.0)), 1),
    }


def kennzahlen_ablegen(werte, pfad=None):
    """Legt die Kennzahlen dieses Laufs ab und ruecken die alten auf Vortag.

    Die Verschiebung passiert HIER und nicht im Bericht: der Bericht darf
    mehrfach gebaut werden, ohne dass der Vergleichswert wegrutscht.
    """
    pfad = Path(pfad) if pfad else KENNZAHLEN
    pfad.parent.mkdir(parents=True, exist_ok=True)
    vortag = pfad.parent / (pfad.stem + "_vortag.json")
    if pfad.exists():
        vortag.write_text(pfad.read_text(encoding="utf-8"), encoding="utf-8")
    pfad.write_text(json.dumps(werte, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return pfad


def kennzahlen_lesen(pfad=None):
    pfad = Path(pfad) if pfad else KENNZAHLEN
    if not pfad.exists():
        return {}
    return json.loads(pfad.read_text(encoding="utf-8"))


def zaehle_rueckfaelle(texte, marken=RUECKFAELLE):
    """Die Naht von Block 4. Texte rein, gezaehlte Treffer raus.

    Woher die Texte kommen - Testausgaben, ein mitgeschriebenes Konsolenlog,
    eine Handvoll Zeilen aus einem Test - ist dieser Funktion gleich. Genau
    deshalb kann der Smoke-Test Block 4 vorfuehren, ohne eine Nacht laufen zu
    lassen.
    """
    zusammen = []
    for text in texte:
        if text:
            zusammen.extend(str(text).splitlines())
    gefunden = []
    for marke in marken:
        regex = re.compile(marke.muster)
        treffer = [z.strip() for z in zusammen if regex.search(z)]
        if treffer:
            gefunden.append((marke, len(treffer), treffer[:MAX_BEISPIELE]))
    return gefunden, len(zusammen)


def _texte_aus_testlauf(bericht):
    if not bericht:
        return []
    return [e.get("ausgabe", "") for e in bericht.get("ergebnisse", [])]


def _texte_aus_protokollen(pfade):
    texte = []
    for p in pfade or ():
        pfad = Path(p)
        if pfad.exists():
            texte.append(pfad.read_text(encoding="utf-8", errors="replace"))
    return texte


def baue(commits=None, testlauf=None, kennzahlen_heute=None,
         kennzahlen_gestern=None, rueckfaelle=None, durchsuchte_zeilen=0,
         steckengeblieben=None, branch="", protokollpfade=None, datum=None):
    """Der Bericht als Text. Rechnet nichts nach - er bekommt alles fertig.

    Absichtlich ohne eigene Datenbeschaffung: so kann der Test jeden Block
    einzeln vorfuehren, mit ausgedachten Eingaben, und die Beschaffung liegt
    sichtbar in sammle_und_baue().
    """
    commits = commits or []
    kennzahlen_heute = kennzahlen_heute or {}
    kennzahlen_gestern = kennzahlen_gestern or {}
    rueckfaelle = rueckfaelle or []
    steckengeblieben = steckengeblieben or []
    rote = [e for e in (testlauf or {}).get("ergebnisse", [])
            if e.get("code") != 0]

    ruhig = (not rote and not rueckfaelle and not steckengeblieben)

    z = ["# Morgenbericht %s" % (datum or datetime.now().strftime("%Y-%m-%d %H:%M")),
         ""]
    if branch:
        z.append("Branch: %s" % branch)
    if ruhig and commits:
        z.append("**Ruhige Nacht: %d Ticket(s) geschlossen, kein roter Test, "
                 "kein Rueckfall, nichts steckengeblieben.**" % len(commits))
    elif ruhig and not commits:
        z.append("**Nichts passiert: kein Ticket geschlossen, kein roter "
                 "Test, kein Rueckfall. Der Lauf ist nicht gelaufen oder "
                 "sofort abgebrochen - das ist ein Befund, kein Ergebnis.**")
    z.append("")

    # --- 1 Geschlossen ------------------------------------------------------
    z.append("## 1. Geschlossen")
    if not commits:
        z.append("Kein Ticket abgeschlossen.")
    else:
        for c in commits:
            nummer = ("#%s" % c["nummer"]) if c.get("nummer") else "(ohne Nr.)"
            z.append("- %-7s %s — Tests: %s"
                     % (nummer, c.get("titel", ""), c.get("tests", "nicht genannt")))
    z.append("")

    # --- 2 Rot -------------------------------------------------------------
    z.append("## 2. Rot")
    if not testlauf:
        z.append("Kein Testlauf abgelegt. Das ist KEIN gruener Befund - es "
                 "wurde nicht gemessen.")
    elif not rote:
        z.append("Alle %d Testdateien gruen."
                 % len(testlauf.get("ergebnisse", [])))
    else:
        for e in rote:
            z.append("- **%s** (%.1f s)" % (e.get("datei", "?"),
                                            e.get("dauer", 0.0)))
            fails = e.get("fails") or []
            for f in fails[:MAX_BEISPIELE]:
                z.append("    %s" % _kuerze(f))
            if len(fails) > MAX_BEISPIELE:
                rest = len(fails) - MAX_BEISPIELE
                z.append("    ... und %d weitere%s Befund%s in derselben Datei."
                         % (rest, "r" if rest == 1 else "",
                            "" if rest == 1 else "e"))
            if not fails:
                z.append("    Rueckgabewert %s, keine erkannte FAIL-Zeile - "
                         "vermutlich Absturz." % e.get("code"))
    z.append("")

    # --- 3 Kennzahlen -------------------------------------------------------
    z.append("## 3. Kennzahlen")
    if not kennzahlen_heute:
        z.append("Keine Kennzahlen abgelegt.")
    else:
        z.append("| Kennzahl | heute | gestern | Veraenderung |")
        z.append("|---|---:|---:|---:|")
        for name in sorted(kennzahlen_heute):
            heute = kennzahlen_heute[name]
            gestern = kennzahlen_gestern.get(name)
            p = _prozent(heute, gestern)
            if gestern is None:
                bewegung = "erster Wert"
            elif p is None:
                bewegung = "nicht rechenbar"
            else:
                bewegung = "%+.1f %%" % p
            z.append("| %s | %s | %s | %s |"
                     % (name, heute,
                        "—" if gestern is None else gestern, bewegung))
        if not kennzahlen_gestern:
            z.append("")
            z.append("Kein Vortagswert vorhanden - eine Zahl allein sagt "
                     "nichts, die Bewegung sagt alles. Ab morgen steht sie "
                     "daneben.")
    z.append("")

    # --- 4 Stille Rueckfaelle ----------------------------------------------
    z.append("## 4. Stille Rueckfaelle")
    z.append("Durchsucht: %d Zeilen mitgeschriebener Ausgabe gegen %d bekannte "
             "Marken." % (durchsuchte_zeilen, len(RUECKFAELLE)))
    if durchsuchte_zeilen == 0:
        z.append("**Null durchsuchte Zeilen heisst NICHT null Rueckfaelle.** "
                 "Es heisst, der Lauf hat keine Ausgabe mitgeschrieben - "
                 "dieser Block hat dann nichts gemessen.")
    elif not rueckfaelle:
        z.append("Kein bekannter Ersatzpfad hat sich gemeldet.")
    else:
        for marke, anzahl, beispiele in rueckfaelle:
            z.append("- **%s** — %dx — %s"
                     % (marke.name, anzahl, marke.ort))
            z.append("    %s" % marke.warum)
            for b in beispiele:
                z.append("    > %s" % _kuerze(b))
    z.append("")

    # --- Steckengeblieben ---------------------------------------------------
    z.append("## Steckengeblieben an der Zeitgrenze")
    if not steckengeblieben:
        z.append("Kein Ticket an der Zeitgrenze abgebrochen.")
    else:
        for s in steckengeblieben:
            z.append("- **#%d %s** — %.0f von %.0f min, offen geblieben"
                     % (s.nummer, s.titel, s.verstrichen_s / 60.0,
                        s.grenze_s / 60.0))
            z.append("    naechste Vermutung: %s" % _kuerze(s.vermutung))
        z.append("")
        z.append("Diese Tickets sind NICHT geschlossen und NICHT neu "
                 "gestartet worden. Die volle Notiz je Ticket steht in "
                 "`nachtbetrieb/laufberichte/steckenbleib-<nr>.json` und im "
                 "Ticket selbst.")
    z.append("")

    # --- Fussnote: wo die Logs liegen --------------------------------------
    z.append("---")
    quellen = ["`git log %s` (Block 1)" % (branch or "nacht/...")]
    if testlauf:
        quellen.append("Testlauf-JSON (Block 2 und 3)")
    for p in (protokollpfade or ()):
        quellen.append("`%s` (Block 4)" % p)
    z.append("Quellen: " + ", ".join(quellen) +
             ". Der Bericht nennt sie, er enthaelt sie nicht.")
    return "\n".join(z)


def sammle_und_baue(basis="main", testlauf_json=None, protokolle=None,
                    ordner=None, repo=None):
    """Die Beschaffung: alles zusammentragen und baue() fuettern."""
    from nachtbetrieb import branch as branch_modul
    from nachtbetrieb import zeitgrenze

    ordner = Path(ordner) if ordner else LAUFORDNER
    try:
        commits = branch_modul.commits_der_nacht(basis=basis, repo=repo)
        aktueller = branch_modul.aktueller_branch(repo=repo)
    except branch_modul.NachtlaufFehler:
        commits, aktueller = [], ""

    bericht = lies_testlauf(testlauf_json) if testlauf_json else None
    heute = kennzahlen_aus_testlauf(bericht)
    heute.update(kennzahlen_lesen(ordner / "kennzahlen_zusatz.json")
                 if (ordner / "kennzahlen_zusatz.json").exists() else {})
    gestern = kennzahlen_lesen(ordner / "kennzahlen.json")

    texte = _texte_aus_testlauf(bericht) + _texte_aus_protokollen(protokolle)
    treffer, zeilen = zaehle_rueckfaelle(texte)

    text = baue(commits=commits, testlauf=bericht, kennzahlen_heute=heute,
                kennzahlen_gestern=gestern, rueckfaelle=treffer,
                durchsuchte_zeilen=zeilen,
                steckengeblieben=zeitgrenze.sammle(ordner),
                branch=aktueller, protokollpfade=protokolle)
    if heute:
        kennzahlen_ablegen(heute, ordner / "kennzahlen.json")
    return text


def schreibe(text, pfad=None, ordner=None):
    ordner = Path(ordner) if ordner else LAUFORDNER
    pfad = Path(pfad) if pfad else ordner / "morgenbericht.md"
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(text, encoding="utf-8")
    return pfad


if __name__ == "__main__":
    import sys
    print(sammle_und_baue(testlauf_json=sys.argv[1] if len(sys.argv) > 1 else None))
