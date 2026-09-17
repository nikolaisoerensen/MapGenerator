"""
Path: tests/smoke_test_fluss_sinuositaet.py

Deckt die beiden in Ticket #33 verlangten Nachweise ab: "ein gerader Lauf
ergibt 1,0, ein bekannter Zickzack einen bekannten Wert". Dazu zwei
strukturelle Pruefungen, die core/fluss_sinuositaet.py's Kernannahme
absichern: Ordnungsstufen (Strahler-Zahl, "river_order") trennen Laeufe an
einem Zusammenfluss, und zu kurze Rasterreste werden nicht mitgezaehlt.

Alle Masken sind von Hand gebaut (keine echte Pipeline noetig) - die
erwarteten Werte sind daher exakt nachrechenbar, nicht nur "plausibel".

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_fluss_sinuositaet.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.fluss_sinuositaet import sinuositaet_je_fluss


def check(label, bedingung, zusatz=""):
    status = "OK" if bedingung else "FAIL"
    print(f"[{status}] {label}{(' - ' + zusatz) if zusatz else ''}")
    return bool(bedingung)


def _leere_karten(groesse=32):
    mask = np.zeros((groesse, groesse), dtype=np.float32)
    order = np.zeros((groesse, groesse), dtype=np.float32)
    return mask, order


def run_gerader_lauf_waagrecht():
    """Ein gerader, waagrechter Lauf muss Sinuositaet 1,0 ergeben."""
    mask, order = _leere_karten()
    mask[5, 2:18] = 1.0
    order[5, 2:18] = 1.0

    ergebnisse = sinuositaet_je_fluss(mask, order, meters_per_pixel=10.0)
    ok = check("waagrechter Lauf: genau ein Ergebnis", len(ergebnisse) == 1,
               f"{len(ergebnisse)} Ergebnisse")
    if not ok:
        return False
    r = ergebnisse[0]
    ok &= check("waagrechter Lauf: Sinuositaet == 1.0",
                math.isclose(r["sinuositaet"], 1.0, abs_tol=1e-9),
                f"gemessen {r['sinuositaet']!r}")
    erwartete_laenge_m = 15 * 10.0  # 16 Pixel, 15 Schritte, je 1 px
    ok &= check("waagrechter Lauf: Pfadlaenge korrekt",
                math.isclose(r["pfadlaenge_m"], erwartete_laenge_m, abs_tol=1e-6),
                f"gemessen {r['pfadlaenge_m']} erwartet {erwartete_laenge_m}")
    return ok


def run_gerader_lauf_diagonal():
    """Ein gerader Lauf entlang der Diagonalen (45 Grad) muss ebenfalls
    Sinuositaet 1,0 ergeben - Schrittweite sqrt(2) kuerzt sich weg."""
    mask, order = _leere_karten()
    for i in range(10):
        mask[i, i] = 1.0
        order[i, i] = 2.0

    ergebnisse = sinuositaet_je_fluss(mask, order, meters_per_pixel=5.0)
    ok = check("diagonaler Lauf: genau ein Ergebnis", len(ergebnisse) == 1)
    if not ok:
        return False
    r = ergebnisse[0]
    ok &= check("diagonaler Lauf: Sinuositaet == 1.0",
                math.isclose(r["sinuositaet"], 1.0, abs_tol=1e-9),
                f"gemessen {r['sinuositaet']!r}")
    return ok


def run_bekannter_zickzack():
    """Dreieckswelle ueber 9 Spalten (Periode 4, Amplitude 2, Steigung 45
    Grad): Pfadlaenge = 8 Diagonalschritte * sqrt(2), Luftlinie = 8 Spalten
    Nettoverschiebung (Anfang und Ende auf derselben Zeile) - das Verhaeltnis
    ist exakt sqrt(2), unabhaengig von meters_per_pixel."""
    mask, order = _leere_karten()
    zeilen_je_spalte = [0, 1, 2, 1, 0, 1, 2, 1, 0]
    for spalte, zeile in enumerate(zeilen_je_spalte):
        mask[zeile, spalte] = 1.0
        order[zeile, spalte] = 1.0

    ergebnisse = sinuositaet_je_fluss(mask, order, meters_per_pixel=1.0)
    ok = check("Zickzack: genau ein Ergebnis", len(ergebnisse) == 1,
               f"{len(ergebnisse)} Ergebnisse")
    if not ok:
        return False
    r = ergebnisse[0]
    erwartet = math.sqrt(2.0)
    ok &= check("Zickzack: Sinuositaet == sqrt(2)",
                math.isclose(r["sinuositaet"], erwartet, abs_tol=1e-9),
                f"gemessen {r['sinuositaet']!r} erwartet {erwartet!r}")
    ok &= check("Zickzack: Luftlinie == 8 Pixel",
                math.isclose(r["luftlinie_m"], 8.0, abs_tol=1e-9),
                f"gemessen {r['luftlinie_m']}")
    ok &= check("Zickzack: Pfadlaenge == 8*sqrt(2) Pixel",
                math.isclose(r["pfadlaenge_m"], 8.0 * math.sqrt(2.0), abs_tol=1e-9),
                f"gemessen {r['pfadlaenge_m']}")
    return ok


def run_ordnungswechsel_trennt_laeufe():
    """Zwei Laeufe unterschiedlicher Ordnung, die pixelweise aneinander
    anschliessen (wie an einem Zusammenfluss), muessen als ZWEI getrennte
    Ergebnisse erscheinen, nicht als einer."""
    mask, order = _leere_karten()
    mask[3, 0:10] = 1.0
    order[3, 0:10] = 1.0
    mask[3, 10:20] = 1.0
    order[3, 10:20] = 2.0  # ab hier hoehere Ordnung - simulierter Zufluss

    ergebnisse = sinuositaet_je_fluss(mask, order, meters_per_pixel=1.0)
    ok = check("Ordnungswechsel: zwei getrennte Laeufe", len(ergebnisse) == 2,
               f"{len(ergebnisse)} Ergebnisse")
    ordnungen = sorted(r["ordnung"] for r in ergebnisse)
    ok &= check("Ordnungswechsel: Ordnungen 1 und 2 vorhanden",
                ordnungen == [1, 2], f"gefunden {ordnungen}")
    return ok


def run_kurze_stummel_werden_ausgelassen():
    """Ein 3-Pixel-Rest (Rasterreste an einer Muendung) darf bei
    min_laenge_px=5 nicht als eigener Fluss auftauchen."""
    mask, order = _leere_karten()
    mask[8, 0:15] = 1.0
    order[8, 0:15] = 1.0
    mask[20, 20:23] = 1.0  # 3 Pixel, isoliert, andere Ordnung
    order[20, 20:23] = 3.0

    ergebnisse = sinuositaet_je_fluss(mask, order, meters_per_pixel=1.0,
                                       min_laenge_px=5)
    ok = check("Stummel ausgelassen: genau ein Ergebnis", len(ergebnisse) == 1,
               f"{len(ergebnisse)} Ergebnisse")
    ok &= check("Stummel ausgelassen: es ist der lange Lauf (Ordnung 1)",
                len(ergebnisse) == 1 and ergebnisse[0]["ordnung"] == 1)
    return ok


def run_region_mehrheit():
    """Region wird als Mehrheitswert entlang des Pfads zugeordnet."""
    mask, order = _leere_karten()
    region_map = np.zeros((32, 32), dtype=np.int16)
    mask[4, 0:10] = 1.0
    order[4, 0:10] = 1.0
    region_map[4, 0:7] = 5   # 7 von 10 Pixeln Region 5
    region_map[4, 7:10] = 9  # 3 von 10 Pixeln Region 9

    ergebnisse = sinuositaet_je_fluss(mask, order, meters_per_pixel=1.0,
                                       region_map=region_map)
    ok = check("Region: genau ein Ergebnis", len(ergebnisse) == 1)
    ok &= check("Region: Mehrheitsregion 5 erkannt",
                ergebnisse and ergebnisse[0]["region_index"] == 5,
                f"gemessen {ergebnisse[0]['region_index'] if ergebnisse else None}")
    return ok


def main():
    gruppen = {
        "gerader Lauf (waagrecht)": run_gerader_lauf_waagrecht(),
        "gerader Lauf (diagonal)": run_gerader_lauf_diagonal(),
        "bekannter Zickzack": run_bekannter_zickzack(),
        "Ordnungswechsel trennt Laeufe": run_ordnungswechsel_trennt_laeufe(),
        "kurze Stummel ausgelassen": run_kurze_stummel_werden_ausgelassen(),
        "Region als Mehrheitswert": run_region_mehrheit(),
    }
    print()
    alles_ok = True
    for name, ok in gruppen.items():
        print(f"{'[OK]' if ok else '[FAIL]'} Gruppe: {name}")
        alles_ok &= ok

    return 0 if alles_ok else 1


if __name__ == "__main__":
    sys.exit(main())
