"""
Path: tests/smoke_test_seedfuehrung.py

Fuehrt die Seed-Infrastruktur aus Tests/seedfuehrung.py vor (Ticket #48) -
ohne selbst schon die Testraenge Waechter/Eichung einzufuehren (das ist
Ticket #46, baut auf #45 auf). Fuenf Gruppen, eine je Abnahmekriterium:

  1. WAECHTER_SEED     ist eine feste, importierbare Zahl.
  2. EICHUNG_DETERMINISTISCH  `eichung_seeds()` liefert bei gleichem Datum
                        immer dieselben drei Zahlen und bei verschiedenen
                        Tagen verschiedene - "wechselnd", aber nie aus der
                        Uhrzeit.
  3. SEED_IN_MELDUNG   `seed_meldung()` nennt die Zahl in der Meldung.
  4. REPRODUZIERBARKEIT  Der ABNAHME-VORFUEHRUNG-Punkt: ein echter
                        Berechnungslauf mit einem Eichungs-Seed wird
                        zweimal gefahren und muss bitgleich sein - genau das
                        macht einen roten Eichungslauf nachstellbar.
  5. KEINE_UHRZEIT_QUELLE  Statische Pruefung: kein `tests/`- oder `core/`-
                        Quelltext zieht einen Seed aus `time.time()`,
                        `datetime.now()` oder einem seedlosen `.seed()`.
                        Muss ROT werden, sobald das wieder vorkommt - siehe
                        CLAUDE.md "Gruene Tests koennen eine tote Funktion
                        verdecken": eine Pruefung, die nichts mehr findet,
                        weil sie nichts mehr sucht, waere nutzlos.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_seedfuehrung.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime

from seedfuehrung import WAECHTER_SEED, eichung_seeds, seed_meldung

WURZEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Verzeichnisse, in denen Testcode bzw. Berechnungscode liegt und in denen ein
# unkontrollierter Seed am teuersten waere.
DURCHSUCHTE_ORDNER = ["tests", "core", "managers", "gui"]

# Dieselben Muster, mit denen vor dem Bau dieser Datei bereits geprueft wurde,
# dass NICHTS im Projekt sie trifft (siehe Ticket-Notiz). Bleibt das so,
# bestaetigt dieser Test es bei jedem Lauf neu, statt es nur einmal behauptet
# zu haben.
UNKONTROLLIERTE_MUSTER = [
    re.compile(r"\.seed\(\s*\)"),                    # random.seed() / np.random.seed() ohne Wert -> OS-Entropie
    re.compile(r"seed\s*=\s*int\(\s*time\.time"),
    re.compile(r"seed\s*=\s*time\.time\(\)"),
    re.compile(r"Random\(\s*time\.time"),
    re.compile(r"Random\(\s*\)"),                    # random.Random() ohne Seed -> Systemzeit/OS-Entropie
    re.compile(r"seed\s*=\s*datetime\.now"),
    re.compile(r"np\.random\.seed\(\s*None\s*\)"),
]

# Eigene Ausnahme: diese Datei hier zaehlt die Muster in ihrer eigenen
# Dokumentation auf (als Text, nicht als Code) - sie muss sich nicht selbst
# meist treffen.
AUSGENOMMENE_DATEIEN = {os.path.abspath(__file__)}


def main():
    fehler = []

    # ---------- 1: Waechter-Seed ----------
    print("1. Waechter-Seed benannt und fest")
    if not isinstance(WAECHTER_SEED, int):
        fehler.append(f"WAECHTER_SEED ist kein int: {WAECHTER_SEED!r}")
    if WAECHTER_SEED != 20260804:
        # Keine harte Notwendigkeit, aber eine Aenderung hier ist eine
        # bewusste Neueichung (siehe Docstring) - sie soll auffallen, nicht
        # lautlos durchlaufen.
        fehler.append(
            f"WAECHTER_SEED hat sich von 20260804 auf {WAECHTER_SEED} "
            "veraendert - falls beabsichtigt, diese Zeile mit anpassen "
            "(Neueichung, keine Routine).")
    print(f"   WAECHTER_SEED = {WAECHTER_SEED}")

    # ---------- 2: Eichung ist deterministisch, nicht uhrzeitbasiert ----------
    print("\n2. Eichungs-Seeds: gleicher Tag -> gleiche Zahlen, anderer Tag -> andere")
    heute = datetime.date(2026, 8, 4)
    a = eichung_seeds(heute)
    b = eichung_seeds(heute)
    if a != b:
        fehler.append(f"eichung_seeds() ist nicht deterministisch: {a} != {b} "
                       "(zweimal derselbe Tag muss dieselben Seeds liefern)")
    if len(set(a)) != 3:
        fehler.append(f"eichung_seeds() liefert keine drei verschiedenen Seeds: {a}")
    morgen = heute + datetime.timedelta(days=1)
    c = eichung_seeds(morgen)
    if c == a:
        fehler.append("eichung_seeds() liefert an zwei verschiedenen Tagen "
                       f"dieselben Seeds ({a}) - das waere kein 'wechselnder' "
                       "Seed mehr, siehe Ticket #48.")
    ohne_argument = eichung_seeds()  # Standardfall: heutiges Kalenderdatum
    if len(ohne_argument) != 3 or len(set(ohne_argument)) != 3:
        fehler.append(f"eichung_seeds() ohne Argument liefert keine drei "
                       f"verschiedenen Seeds: {ohne_argument}")
    print(f"   {heute}: {a}")
    print(f"   {morgen}: {c}")
    print(f"   heute ({datetime.date.today()}): {ohne_argument}")

    # ---------- 3: Fehlermeldung nennt den Seed ----------
    print("\n3. Fehlermeldungen nennen den benutzten Seed")
    beispiel = seed_meldung("Region Morobora ausserhalb Sollhang", a[1])
    if str(a[1]) not in beispiel:
        fehler.append(f"seed_meldung() nennt den Seed nicht: {beispiel!r}")
    if "seed=" not in beispiel:
        fehler.append(f"seed_meldung() benutzt nicht das vereinbarte "
                       f"'seed='-Format: {beispiel!r}")
    print(f"   Beispiel: {beispiel}")

    # ---------- 4: Reproduzierbarkeit EINMAL VORGEFUEHRT ----------
    #
    # Abnahmekriterium: "Ein fehlgeschlagener Eichungslauf ist mit dem
    # genannten Seed exakt nachstellbar - einmal vorgefuehrt." Dafuer ein
    # echter Berechnungslauf (nicht nur die Seed-Zahl selbst), zweimal mit
    # demselben gezogenen Eichungs-Seed, bitgleiches Ergebnis verlangt.
    print("\n4. Reproduzierbarkeit vorgefuehrt: echter Lauf, zweimal, ein Eichungs-Seed")
    import numpy as np

    import core.terrain_weltkarte as rw

    seed_lauf = eichung_seeds(heute)[1]  # einer der drei "heutigen" Seeds
    H1, felder1 = rw.weltfeld(128, seed_lauf)
    H2, felder2 = rw.weltfeld(128, seed_lauf)
    bitgleich = np.array_equal(H1, H2)
    if not bitgleich:
        differenz = float(np.abs(H1 - H2).max())
        fehler.append(seed_meldung(
            "Eichungslauf nicht reproduzierbar - zwei Laeufe mit demselben "
            f"Seed weichen um bis zu {differenz:.4f} m voneinander ab",
            seed_lauf))
    print(f"   weltfeld(128, seed={seed_lauf}) zweimal gerechnet: "
          f"{'bitgleich' if bitgleich else 'ABWEICHUNG - siehe Fehler'}")
    print(seed_meldung("   Vorfuehrung mit Eichungs-Seed", seed_lauf))

    # ---------- 5: Keine Seed-Ziehung aus der Uhrzeit ----------
    print("\n5. Kein Test/Kerncode zieht einen Seed aus der Uhrzeit")
    durchsucht = 0
    treffer = []
    for ordner in DURCHSUCHTE_ORDNER:
        pfad = os.path.join(WURZEL, ordner)
        if not os.path.isdir(pfad):
            continue
        for wurzel, _dirs, dateien in os.walk(pfad):
            for name in dateien:
                if not name.endswith(".py"):
                    continue
                voll = os.path.join(wurzel, name)
                if os.path.abspath(voll) in AUSGENOMMENE_DATEIEN:
                    continue
                try:
                    with open(voll, encoding="utf-8") as f:
                        zeilen = f.readlines()
                except OSError:
                    continue
                for nr, zeile in enumerate(zeilen, 1):
                    durchsucht += 1
                    for muster in UNKONTROLLIERTE_MUSTER:
                        if muster.search(zeile):
                            treffer.append(f"{os.path.relpath(voll, WURZEL)}:{nr}: {zeile.strip()}")
    if durchsucht == 0:
        fehler.append("KEINE Zeilen durchsucht - das ist KEIN gruener Befund, "
                       "die Pruefung lief ins Leere statt etwas zu pruefen.")
    if treffer:
        fehler.append(f"{len(treffer)} unkontrollierte Seed-Ziehung(en) gefunden:")
        fehler.extend(f"    {t}" for t in treffer)
    print(f"   {durchsucht} Zeilen durchsucht ueber {DURCHSUCHTE_ORDNER}, "
          f"{len(treffer)} Treffer")

    print("\n" + "=" * 70)
    if fehler:
        print(f"NICHT IN ORDNUNG - {len(fehler)} Befunde:")
        for f in fehler:
            print(f"  [FAIL] {f}")
        return 1
    print("IN ORDNUNG - Seedfuehrung (Waechter-Seed, Eichungs-Seeds, "
          "Seed-in-Meldung, Reproduzierbarkeit, keine Uhrzeit-Quelle)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
