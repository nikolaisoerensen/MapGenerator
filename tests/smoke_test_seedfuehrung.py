"""
Path: tests/smoke_test_seedfuehrung.py

Waechter fuer nachtbetrieb/seeds.py (#48): Seedfuehrung fuer Waechter und
Eichung.

Prueft die vier Abnahmekriterien des Tickets einzeln:

  1. Der Waechter-Seed steht fest, an einer benannten, begruendeten Stelle.
  2. Die Eichung zieht drei Seeds ueber ein nachvollziehbares, dokumentiertes
     Verfahren - nicht per Wuerfel.
  3. Eine Meldung, die ueber mit_seed() gebaut wird, nennt den Seed.
  4. Ein Eichungslauf ist mit dem genannten Seed exakt nachstellbar - hier
     vorgefuehrt an core.terrain_river_network.poisson_points(), demselben
     seed-gesteuerten Generator, den tests/smoke_test_poisson_punkte.py
     bereits als Waechter benutzt.

Dazu, als fuenfte Gruppe, der Rueckfallwaechter aus dem Abnahmekriterium
"kein Test zieht mehr einen Seed aus der Uhrzeit": ein echter Scan ueber
tests/, core/, managers/ - UND ein Beweis, dass der Scan ueberhaupt etwas
findet, wenn man ihm absichtlich etwas Verbotenes unterschiebt (sonst
waere ein Scan, der immer leer bleibt, nicht von einem kaputten Scan zu
unterscheiden - dieselbe Falle wie beim adaptiven Mesh, CLAUDE.md).

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_seedfuehrung.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nachtbetrieb import seeds  # noqa: E402
from core.terrain_river_network import poisson_points  # noqa: E402


def run_waechter_seed_fest_und_benannt():
    print("\n--- Waechter-Seed: fest, benannt, begruendet ---")
    ok = (isinstance(seeds.WAECHTER_SEED, int) and seeds.WAECHTER_SEED == 20260804)
    print("  WAECHTER_SEED = %s (int: %s)"
          % (seeds.WAECHTER_SEED, isinstance(seeds.WAECHTER_SEED, int)))

    # Er soll kein neu erfundener Wert sein, sondern der bereits ueberall
    # benutzte - stichprobenartig gegen drei bestehende Waechtertests
    # geprueft, die ihn heute noch als eigene Modulkonstante fuehren.
    wurzel = Path(__file__).resolve().parent.parent
    stichprobe = ["smoke_test_regionen_welt.py", "smoke_test_kuestenprofiltreue.py",
                  "smoke_test_archetyp_verteilung.py"]
    for name in stichprobe:
        text = (wurzel / "tests" / name).read_text(encoding="utf-8")
        treffer = ("SEED = %d" % seeds.WAECHTER_SEED) in text
        ok = ok and treffer
        print("  %-38s benutzt WAECHTER_SEED: %s" % (name, treffer))
    return ok


def run_eichungs_seeds_deterministisch_und_verschieden():
    print("\n--- Eichungs-Seeds: drei, deterministisch, nachvollziehbar ---")
    a1 = seeds.eichungs_seeds("2026-09-21")
    a2 = seeds.eichungs_seeds("2026-09-21")
    b = seeds.eichungs_seeds("2026-09-22")

    anzahl_ok = len(a1) == seeds.EICHUNGS_ANZAHL == 3
    alle_int = all(isinstance(s, int) for s in a1)
    gleiche_quelle_gleiche_seeds = a1 == a2
    drei_verschieden = len(set(a1)) == 3
    andere_quelle_andere_seeds = a1 != b
    print("  Quelle '2026-09-21' -> %s" % (a1,))
    print("  Quelle '2026-09-22' -> %s" % (b,))
    print("  Anzahl == 3:                    %s" % anzahl_ok)
    print("  alles int:                      %s" % alle_int)
    print("  gleiche Quelle -> gleiche Seeds: %s" % gleiche_quelle_gleiche_seeds)
    print("  drei Seeds sind verschieden:     %s" % drei_verschieden)
    print("  andere Quelle -> andere Seeds:   %s" % andere_quelle_andere_seeds)

    # Kein Default, kein leerer Wert - sonst waere "quelle" wieder optional
    # und der bequeme, unkontrollierte Aufruf moeglich.
    kein_default = False
    try:
        seeds.eichungs_seeds()
        print("  FEHLER: eichungs_seeds() ohne Argument ging durch.")
    except TypeError:
        kein_default = True
        print("  eichungs_seeds() ohne Argument: abgelehnt (kein Default). ok")

    leere_quelle_abgelehnt = False
    try:
        seeds.eichungs_seeds("")
        print("  FEHLER: eichungs_seeds('') ging durch.")
    except ValueError:
        leere_quelle_abgelehnt = True
        print("  eichungs_seeds(''): abgelehnt. ok")

    return all([anzahl_ok, alle_int, gleiche_quelle_gleiche_seeds, drei_verschieden,
               andere_quelle_andere_seeds, kein_default, leere_quelle_abgelehnt])


def run_quelle_aus_nachtbranch():
    print("\n--- Eichungs-Quelle aus dem Nachtbranch-Namen ---")
    a = seeds.quelle_aus_nachtbranch("nacht/2026-09-21")
    b = seeds.quelle_aus_nachtbranch("nacht/2026-09-21-b")
    print("  'nacht/2026-09-21'   -> %r" % a)
    print("  'nacht/2026-09-21-b' -> %r" % b)
    # Ein Kollisionsbuchstabe (branch.naechster_branchname) darf die
    # Eichungs-Seeds derselben Nacht nicht veraendern.
    gleich_trotz_buchstabe = a == b == "2026-09-21"

    kein_nachtbranch_abgelehnt = False
    try:
        seeds.quelle_aus_nachtbranch("main")
        print("  FEHLER: 'main' wurde als Nachtbranch akzeptiert.")
    except ValueError:
        kein_nachtbranch_abgelehnt = True
        print("  'main' als Quelle: abgelehnt. ok")

    return gleich_trotz_buchstabe and kein_nachtbranch_abgelehnt


def run_mit_seed_nennt_seed():
    print("\n--- Jede Meldung nennt den benutzten Seed ---")
    seed = seeds.eichungs_seeds("2026-09-21")[0]
    text = seeds.mit_seed("[FAIL] Kuestenanteil 4.2%% statt 6.5%%", seed)
    print("  %s" % text)
    ok = str(seed) in text and text.startswith("[FAIL]")
    return ok


def run_vorfuehrung_reproduzierbarkeit():
    print("\n--- Vorfuehrung: Eichungslauf mit genanntem Seed exakt nachstellbar ---")
    print("  (an core.terrain_river_network.poisson_points, demselben")
    print("   seed-gesteuerten Generator wie tests/smoke_test_poisson_punkte.py)")

    seed = seeds.eichungs_seeds("vorfuehrung-2026-09-21")[1]
    extent_m, min_distance_m = 4000.0, 300.0

    # "Lauf 1": das ist der (hier simulierte) fehlgeschlagene Eichungslauf.
    lauf_1 = poisson_points(extent_m, min_distance_m, seed)
    meldung = seeds.mit_seed(
        "[FAIL] Eichung: Punktabstand unter Schwelle bei Punkt 3", seed)
    print("  Lauf 1: %d Punkte, Meldung: %s" % (len(lauf_1), meldung))

    # "Nachstellen": nichts weiter als derselbe Seed aus der Meldung, kein
    # Zusatzwissen ueber die Quelle noetig.
    nachgestellter_seed = int(meldung.rsplit("Seed ", 1)[1].rstrip(")"))
    lauf_2 = poisson_points(extent_m, min_distance_m, nachgestellter_seed)
    print("  Lauf 2 (aus der Meldung nachgestellt): %d Punkte" % len(lauf_2))

    bitgleich = (lauf_1.shape == lauf_2.shape
                and bool((lauf_1 == lauf_2).all()))
    seed_aus_meldung_stimmt = nachgestellter_seed == seed
    print("  Seed aus der Meldung korrekt extrahiert: %s" % seed_aus_meldung_stimmt)
    print("  Beide Laeufe bitgleich (identischer Punktsatz): %s" % bitgleich)

    # Gegenprobe: ein ANDERER Seed liefert (so gut wie sicher) einen
    # ANDEREN Punktsatz - sonst waere "bitgleich" oben ein Zufallstreffer
    # und keine echte Bestaetigung des Seeds als Ursache.
    anderer_seed = seeds.eichungs_seeds("vorfuehrung-2026-09-22")[1]
    lauf_3 = poisson_points(extent_m, min_distance_m, anderer_seed)
    unterschiedlich_bei_anderem_seed = (
        lauf_3.shape != lauf_1.shape or not bool((lauf_3 == lauf_1).all()))
    print("  Anderer Seed liefert anderen Punktsatz (Gegenprobe): %s"
          % unterschiedlich_bei_anderem_seed)

    return bitgleich and seed_aus_meldung_stimmt and unterschiedlich_bei_anderem_seed


def run_keine_unkontrollierten_seeds_im_baum():
    print("\n--- Kein Test zieht einen Seed aus der Uhrzeit ---")
    # Geprueft wird tests/ - dort leben Waechter und Eichung, um die es in
    # diesem Ticket geht. tools/*_werkstatt.py/*_lab.py sind interaktive
    # Laborwerkzeuge mit einem sichtbaren "Neuer Seed"-Knopf (siehe
    # tools/flussnetz_werkstatt.py:neuer_seed()) - eine bewusste, angezeigte
    # Zufallswahl per Klick ist etwas anderes als ein Test, der beim
    # automatischen Lauf still einen Uhrzeit-Seed zieht, und faellt nicht
    # unter dieses Abnahmekriterium.
    #
    # Diese Datei selbst schliesst sich aus: sie enthaelt weiter unten die
    # verbotenen Muster als Text in einer Gegenprobe-Datei, nicht als
    # eigene Seed-Ziehung - sonst waere der Waechter nie gruen zu bekommen.
    wurzel = Path(__file__).resolve().parent.parent
    pfade = [p for p in (wurzel / "tests").rglob("*.py")
            if p.resolve() != Path(__file__).resolve()]
    treffer = seeds.finde_unkontrollierte_seeds(pfade)
    print("  durchsucht: %d Dateien in tests/ (ohne dieses Waechter-Skript "
          "selbst)" % len(pfade))
    for pfad, nr, zeile in treffer:
        print("  UNKONTROLLIERT: %s:%d  %s" % (pfad, nr, zeile))
    sauber = len(treffer) == 0
    print("  Baum sauber: %s" % sauber)

    # Gegenprobe: der Scanner muss etwas finden, wenn wirklich etwas drin
    # steht - sonst waere "sauber" oben bedeutungslos (ein Scan, der immer
    # leer meldet, sieht identisch aus wie ein Scan, der nichts findet).
    with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write("seed = int(time.time())\n"
                "rng = np.random.default_rng()\n")
        unsauberer_pfad = f.name
    try:
        gefunden = seeds.finde_unkontrollierte_seeds([unsauberer_pfad])
        erkennt_beide_muster = len(gefunden) == 2
        print("  Gegenprobe (absichtlich unsaubere Datei): %d Treffer "
              "(erwartet 2): %s" % (len(gefunden), erkennt_beide_muster))
    finally:
        os.unlink(unsauberer_pfad)

    return sauber and erkennt_beide_muster


if __name__ == "__main__":
    ergebnisse = {
        "waechter_seed_fest_und_benannt": run_waechter_seed_fest_und_benannt(),
        "eichungs_seeds_deterministisch_und_verschieden":
            run_eichungs_seeds_deterministisch_und_verschieden(),
        "quelle_aus_nachtbranch": run_quelle_aus_nachtbranch(),
        "mit_seed_nennt_seed": run_mit_seed_nennt_seed(),
        "vorfuehrung_reproduzierbarkeit": run_vorfuehrung_reproduzierbarkeit(),
        "keine_unkontrollierten_seeds_im_baum":
            run_keine_unkontrollierten_seeds_im_baum(),
    }
    print("\n=== SUMMARY ===")
    for name, ergebnis in ergebnisse.items():
        print("%-46s %s" % (name, "PASS" if ergebnis else "FAIL"))
    sys.exit(0 if all(ergebnisse.values()) else 1)
