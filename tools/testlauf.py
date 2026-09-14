"""
Path: tools/testlauf.py

Vollstaendiger Lauf der Testsuite, als Grundlage fuer docs/TESTBERICHT.md.

JEDE DATEI ALS EIGENER PROZESS. Nicht gesammelt in einem Interpreter, und das
aus zwei Gruenden, die beide schon Geld gekostet haben:

  * Ein Absturz reisst sonst die uebrigen mit, und man weiss hinterher nicht,
    welche Tests ueberhaupt gelaufen waeren.
  * Die Laufzeiten waeren nicht vergleichbar. Am 2026-08-24 wurde gemessen,
    dass der parallele Erosionsfilter JEDE nachfolgende numpy-Rechnung im
    selben Prozess dauerhaft 2.4x verlangsamt (Allokator-Fragmentierung,
    siehe docs/PERFORMANCE_2026-08-23.md). Im selben Prozess gemessene
    Laufzeiten waeren also von der Reihenfolge abhaengig.

MESSWARNUNG, die in jeden Bericht gehoert: diese Maschine schwankt um Faktor
2-3. Dieselbe feste Referenzlast mass zwischen 0.80 s und 1.91 s ohne
erkennbare Fremdlast. Laufzeiten sind Groessenordnungen, keine Messwerte.

Aufruf:
    .venv/Scripts/python.exe tools/testlauf.py              # alles
    .venv/Scripts/python.exe tools/testlauf.py --schnell    # nur die kurzen
    .venv/Scripts/python.exe tools/testlauf.py --nur kueste # Namensfilter
"""

import argparse
import json
import os
import subprocess
import sys
import time

WURZEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(WURZEL, ".venv", "Scripts", "python.exe")
TESTS = os.path.join(WURZEL, "tests")

# Tests, die Gelaende oder die volle Pipeline fahren - alles andere ist kurz.
# Die Einstufung kommt aus einer statischen Pruefung (welche Datei `weltfeld`,
# `CALCULATOR_GRAPH`, `flussnetz` o.ae. benutzt), nicht aus Erfahrung.
TEURE_MERKMALE = ("weltfeld", "_pipeline", "CALCULATOR_GRAPH", "flussnetz",
                  "baue_remesh", "weltfluesse", "GPUWorker", "ShaderManager")

# Wieviel Zeit eine einzelne Datei hoechstens bekommt. Grosszuegig: der
# langsamste bekannte Test (smoke_test_regionen_welt.py) braucht rund zwei
# Minuten, und die Maschine schwankt.
ZEITGRENZE_S = 1800


def ist_teuer(pfad):
    try:
        with open(pfad, encoding="utf-8") as f:
            quelle = f.read()
    except OSError:
        return True
    return any(w in quelle for w in TEURE_MERKMALE)


def sammeln(nur=None, schnell=False):
    dateien = sorted(f for f in os.listdir(TESTS)
                     if f.startswith("smoke_test_") and f.endswith(".py"))
    if nur:
        dateien = [f for f in dateien if nur.lower() in f.lower()]
    if schnell:
        dateien = [f for f in dateien
                   if not ist_teuer(os.path.join(TESTS, f))]
    return dateien


def einen_laufen_lassen(datei):
    pfad = os.path.join("tests", datei)
    beginn = time.time()
    try:
        ergebnis = subprocess.run(
            [PYTHON, pfad], cwd=WURZEL, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=ZEITGRENZE_S)
        code, ausgabe = ergebnis.returncode, ergebnis.stdout + ergebnis.stderr
    except subprocess.TimeoutExpired:
        code, ausgabe = -9, f"ZEITGRENZE {ZEITGRENZE_S} s ueberschritten"
    dauer = time.time() - beginn

    # Fehlgeschlagene Zusicherungen einsammeln - der Rueckgabewert allein sagt
    # nur DASS, nicht WAS. Ohne das muesste man jede Datei einzeln nachfahren.
    #
    # DREI FORMATE, weil die Suite historisch gewachsen ist (gezaehlt am
    # 2026-08-25: 26 Dateien mit "[FAIL]", 2 mit "FEHLGESCHLAGEN", dazu
    # Zusammenfassungszeilen "name: FAIL"). Erst mit allen dreien wird aus
    # "Rueckgabe 1" eine Aussage; die erste Fassung hielt
    # smoke_test_regionen_welt.py faelschlich fuer einen Absturz, weil dieser
    # Test "NICHT IN ORDNUNG - N Befunde:" schreibt.
    marken = ("[FAIL]", "FEHLGESCHLAGEN", "NICHT IN ORDNUNG")
    fails = [z.strip() for z in ausgabe.splitlines()
             if any(m in z for m in marken) or z.rstrip().endswith(": FAIL")]
    # Nach "NICHT IN ORDNUNG - N Befunde:" stehen die Befunde in den
    # Folgezeilen, nicht in der Marke selbst - die waere sonst nutzlos.
    zeilen = ausgabe.splitlines()
    for i, z in enumerate(zeilen):
        if "NICHT IN ORDNUNG" in z:
            fails.extend(w.strip() for w in zeilen[i + 1:i + 10] if w.strip())
            break

    return {"datei": datei, "code": code, "dauer": dauer,
            "fails": fails[:12], "zeilen": len(ausgabe.splitlines()),
            "ausgabe": ausgabe[-4000:]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--schnell", action="store_true",
                   help="nur Tests ohne Gelaende-/Pipelinelauf")
    p.add_argument("--nur", default=None, help="Namensfilter")
    p.add_argument("--bericht", default=None,
                   help="JSON-Datei fuer die Rohergebnisse")
    args = p.parse_args()

    dateien = sammeln(args.nur, args.schnell)
    print(f"{len(dateien)} Testdateien, je eigener Prozess, "
          f"Zeitgrenze {ZEITGRENZE_S} s\n")

    ergebnisse = []
    gesamt = time.time()
    for nummer, datei in enumerate(dateien, 1):
        print(f"[{nummer:2d}/{len(dateien)}] {datei:52s}", end="", flush=True)
        e = einen_laufen_lassen(datei)
        ergebnisse.append(e)
        zeichen = "OK  " if e["code"] == 0 else f"FAIL({e['code']})"
        print(f" {zeichen:>10}  {e['dauer']:7.1f}s")
        for zeile in e["fails"][:3]:
            print(f"          {zeile[:110]}")
    dauer_gesamt = time.time() - gesamt

    bestanden = [e for e in ergebnisse if e["code"] == 0]
    gefallen = [e for e in ergebnisse if e["code"] != 0]

    print("\n" + "=" * 78)
    print(f"{len(bestanden)} von {len(ergebnisse)} bestanden, "
          f"Gesamtlaufzeit {dauer_gesamt:.0f} s ({dauer_gesamt/60:.1f} Min)")
    print("=" * 78)

    if gefallen:
        print(f"\nFEHLGESCHLAGEN ({len(gefallen)}):")
        for e in gefallen:
            print(f"\n  {e['datei']}  (Rueckgabe {e['code']}, {e['dauer']:.1f} s)")
            for zeile in e["fails"]:
                print(f"      {zeile[:120]}")
            if not e["fails"]:
                print("      (keine [FAIL]-Zeile - vermutlich Absturz, "
                      "letzte Ausgabe:)")
                for zeile in e["ausgabe"].splitlines()[-6:]:
                    print(f"      {zeile[:120]}")

    print("\nLANGSAMSTE ZEHN:")
    for e in sorted(ergebnisse, key=lambda x: -x["dauer"])[:10]:
        print(f"   {e['dauer']:7.1f}s  {e['datei']}")

    if args.bericht:
        with open(args.bericht, "w", encoding="utf-8") as f:
            json.dump({"gesamtdauer_s": dauer_gesamt,
                       "ergebnisse": ergebnisse}, f, indent=2,
                      ensure_ascii=False)
        print(f"\nRohergebnisse: {args.bericht}")

    return 0 if not gefallen else 1


if __name__ == "__main__":
    sys.exit(main())
