"""
Path: tests/smoke_test_kuesten_naht_kruemmung.py

Realismus-Test fuer die "Naht" an einer Landzunge/einem Hals (core/vektor_kueste.py,
Klasse VektorKueste, Kernstelle _hoehe_block(), K_KUESTEN-Mischung aus mehreren
unabhaengig berechneten Kuestenabschnitten).

Befund aus der /mattpocock-skills:tdd-Sitzung 2026-09-17: an Stellen, wo zwei
Kuestenarme sich naehern (ein "Hals"), kreuzen sich zwei UNABHAENGIG voneinander
berechnete Distanzfelder (je Segment sein eigener exakter Abstand zur eigenen
Kontur, siehe _auf_strecken()). Weil `w_i` per MAXIMUM und `profil_i` per
gewichtetem MITTELWERT ueber die Segmente kombiniert wird (Zeilen
3123/3156/3160-3162), kippt die Dominanz an der Mittelachse (Ort gleichen
Abstands zu beiden Armen) nicht zwangslaeufig glatt - sichtbar als Kante/Naht,
vom Nutzer treffend mit einer Fraeskante um eine Ecke verglichen ("wie wenn du
eine Fraeskante um die Ecke fuehrst und dann eine Linie hast dazwischen").

DIESER TEST prueft NICHT gegen einen aus der Formel zurueckgerechneten Wert,
sondern VERGLEICHT an derselben Stelle zwei Messungen derselben Groesse: die
Kruemmung (2. Ableitung der Hoehe nach der Bogenlaenge) NAHE der Halsmitte
gegen die Kruemmung WEITER WEG auf demselben Querschnitt. Eine natuerlich
verrauschte Kueste hat an keiner bestimmten Stelle einen um mehr als das
Doppelte ueberhoehten Krummungs-Ausschlag - das ist die unabhaengig begruendete
Erwartung (Vergleich zwischen zwei Messpunkten derselben Groesse, keine
Neuberechnung der Formel - analog zur Monotonie-Pruefung in
smoke_test_erosion_realismus.py).

Die acht Pruefpunkte (Pixel-Koordinaten + Schnittrichtung) liegen auf der
festen Karte size=384, seed=20260804 (core/terrain_weltkarte.py:weltfeld())
und wurden per Diagnoseskript als echte Haelse identifiziert (zwei Kuesten-
abschnitte gleichzeitig in Reichweite, etwa gleicher Abstand zu beiden). Sie
sind fest eingefroren, damit der Test nicht bei jedem Lauf erneut nach einem
Hals suchen muss - das waere eine Abhaengigkeit von einer internen
Auswahlfunktion (_kuesten_waehlen), waehrend die eigentliche PRUEFUNG
ausschliesslich ueber die oeffentliche Methode VektorKueste.hoehe() laeuft.

GEMESSEN 2026-09-17 (aktueller Stand, VOR jeder Kontur-Stoerung): alle acht
Punkte liegen ueber der 2x-Schwelle (Naht-Verhaeltnis 2.6 bis 207, Median
8.05). Der Test ist daher bewusst ROT committet, als dokumentierter echter
Befund - noch ohne Behebung. Ein erster Gegenversuch (d_i je Segment mit
Bogenlaengen-korreliertem Rauschen stoeren, 25 m Amplitude/120 m Wellenlaenge)
senkte den Median auf 2.16, aber nicht bei allen acht Punkten gleich stark -
die eigentliche Behebung/Kalibrierung ist ein eigener, noch offener Schritt.

Aufruf: .venv\\Scripts\\python.exe tests/smoke_test_kuesten_naht_kruemmung.py
"""
import sys

import numpy as np

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

from core.terrain_weltkarte import weltfeld
from core.vektor_kueste import VektorKueste, MESH_MINDEST_SKALA_M

SIZE = 384
SEED = 20260804

# (x_px, y_px, schnitt_richtung_grad) - acht echte Haelse auf der festen Karte,
# per Diagnoseskript gefunden (zwei Kuestenabschnitte gleichzeitig nah, etwa
# gleicher Abstand zu beiden).
HALS_PUNKTE = [
    (311.0, 165.0, 120.0),
    (236.0, 118.0, 75.0),
    (162.0, 290.0, 105.0),
    (211.0, 80.0, 165.0),
    (119.0, 232.0, 150.0),
    (326.0, 179.0, 90.0),
    (194.0, 57.0, 120.0),
    (264.0, 96.0, 0.0),
]

NAHT_SCHWELLE = 2.0


def check(label, condition):
    print(("[OK] " if condition else "[FAIL] ") + label)
    return bool(condition)


def naht_verhaeltnis(vk, x_px, y_px, richtung_grad, halblaenge_m=250.0, n=400):
    """
    Tastet eine Gerade durch (x_px, y_px) in Richtung `richtung_grad` ab -
    NUR ueber VektorKueste.hoehe(), keine internen Felder. Liefert das
    Verhaeltnis der Kruemmung (|2. Ableitung|) nahe der Mitte (|s| <= 40 m)
    zur Kruemmung weit weg (|s| > 150 m) auf demselben Schnitt.
    """
    rad = np.radians(richtung_grad)
    richtung = np.array([np.cos(rad), np.sin(rad)])
    s_m = np.linspace(-halblaenge_m, halblaenge_m, n)
    s_px = s_m / vk.mpp
    pts = np.array([x_px, y_px])[None, :] + s_px[:, None] * richtung[None, :]

    h = vk.hoehe(pts[:, 0], pts[:, 1], MESH_MINDEST_SKALA_M)
    d1 = np.gradient(h, s_m)
    d2 = np.gradient(d1, s_m)

    an_mitte = np.abs(s_m) <= 40
    fern = np.abs(s_m) > 150
    med_mitte = float(np.median(np.abs(d2[an_mitte])))
    med_fern = float(np.median(np.abs(d2[fern])))
    return med_mitte / max(med_fern, 1e-6)


def run_naht_an_landzungen_haelsen_bleibt_unauffaellig():
    """
    Acht echte Haelse auf derselben Karte (size=384, seed=20260804). Fuer
    jeden wird das Naht-Verhaeltnis (Kruemmung Mitte/fern, s.o.) gemessen und
    gegen NAHT_SCHWELLE=2.0 geprueft - der Punkt, ab dem die Kruemmung an der
    Halsmitte nicht mehr als natuerliche Rauschvarianz durchgeht, sondern als
    strukturelle Auffaelligkeit (Kante/Naht) zaehlt.
    """
    print("Generiere Karte (size={}, seed={}) ...".format(SIZE, SEED))
    H, felder = weltfeld(SIZE, SEED)
    H = np.asarray(H, dtype=np.float64)
    vk = VektorKueste(H, felder["regionen"], SEED)

    ok = True
    for x_px, y_px, richtung in HALS_PUNKTE:
        verhaeltnis = naht_verhaeltnis(vk, x_px, y_px, richtung)
        ok &= check(
            "Hals bei ({:.0f},{:.0f}), Schnitt {:.0f} Grad: "
            "Naht-Verhaeltnis {:.2f} <= {:.1f}".format(
                x_px, y_px, richtung, verhaeltnis, NAHT_SCHWELLE),
            verhaeltnis <= NAHT_SCHWELLE)
    return ok


def main():
    tests = [
        ("naht_an_landzungen_haelsen_bleibt_unauffaellig",
         run_naht_an_landzungen_haelsen_bleibt_unauffaellig),
    ]
    results = {}
    for name, func in tests:
        print("\n=== {} ===".format(name))
        results[name] = func()

    print("\n=== SUMMARY ===")
    for name, passed in results.items():
        print("{}: {}".format(name, "PASS" if passed else "FAIL"))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
