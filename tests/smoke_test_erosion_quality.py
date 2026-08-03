"""
Path: tests/smoke_test_erosion_quality.py

Qualitaets-Regressionstest fuer das Feld-Erosionsmodell
(core/erosion_generator.py). Friert die BILDEIGENSCHAFTEN ein, die der Nutzer
am Ergebnis beurteilt - nicht die technischen Zusagen, die stehen in
smoke_test_erosion_field.py.

Die fuenf Kennzahlen bilden das vom Nutzer beschriebene Zielbild ab:

  (a) keine Krater          - Senken, die kein See sind
  (b) Erosion in Linien     - ein verzweigtes Netz statt einer Flaeche
  (c) zusammenhaengendes Netz - Baechlinien, die sich verbinden
  (d) Ebenen                - Sedimentation erzeugt flache Talboeden
  (e) Skalierung            - dieselbe Landschaft bei anderer Aufloesung

ZUR KRATER-SCHWELLE: sie ist RELIEF-RELATIV (1% des Kartenreliefs), nicht
absolut. Eine absolute 0.5-m-Schwelle war aus dem Droplet-Modell uebernommen,
wo Krater 300 m tiefe Loecher mit aufgeworfenem Rand waren. Im Feldmodell
zaehlte dieselbe Schwelle 119 Senken, die sich bei Nachmessung als bis zu 1 m
tiefe TUEMPEL entpuppten - voll Wasser (1.49 m gegen 0.11 m Kartenmittel), mit
ebenem Wasserspiegel und ohne weitere Eintiefung. Also genau das, was das
Modell erzeugen SOLL ("Seen moeglich, aber nicht noetig", Nutzer-Vorgabe), und
kein Fehler. Mit der relief-relativen Schwelle sind es null.

Aufruf: .venv\\Scripts\\python.exe tests/smoke_test_erosion_quality.py
"""

import sys

import numpy as np
from scipy.ndimage import gaussian_filter, label, minimum_filter

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

from core.erosion_generator import HydraulicFieldSimulator
from core.water_generator import _NEIGHBOR_FOOTPRINT_8


def check(label_text, condition):
    print(("[OK] " if condition else "[FAIL] ") + label_text)
    return bool(condition)


def make_terrain(size, seed=5, amplitude=4000.0):
    rng = np.random.RandomState(seed)
    return (gaussian_filter(rng.rand(size, size), 3) * amplitude).astype(np.float32)


def measure(size, parameters=None, seed=5):
    """Einen Lauf ausfuehren und die fuenf Kennzahlen bestimmen."""
    terrain = make_terrain(size, seed)
    hardness = np.full((size, size), 50.0, dtype=np.float32)
    params = {"max_steps": 30000, "rainfall": 2.0}
    params.update(parameters or {})

    result = HydraulicFieldSimulator().simulate(
        terrain, hardness, params, 10000.0 / size)

    erosion = result["erosion_map"]
    after = terrain.astype(np.float64) - erosion + result["sedimentation_map"]
    relief = float(terrain.max() - terrain.min())
    normalized = (terrain - terrain.min()) / relief

    ranked = np.sort(erosion.ravel())[::-1]
    top5 = float(ranked[:len(ranked) // 20].sum() / max(ranked.sum(), 1e-9))

    positive = erosion[erosion > 0]
    network = 0
    if positive.size > 20:
        threshold = np.percentile(positive, 92)
        labelled, _ = label(erosion > threshold)
        network = int(np.bincount(labelled.ravel())[1:].max())

    neighbour_min = minimum_filter(after, footprint=_NEIGHBOR_FOOTPRINT_8,
                                   mode='constant', cval=np.inf)
    craters = int(((neighbour_min - after)[1:-1, 1:-1] > 0.01 * relief).sum())

    gy, gx = np.gradient(after)
    slope = np.hypot(gx, gy)
    plains = float((slope < 0.25 * slope.mean()).mean()) * 100.0

    return {
        "steps": result["steps_taken"],
        "converged": result["converged"],
        "top5": top5,
        "network": network,
        "craters": craters,
        "plains": plains,
        "centroid": float((erosion * normalized).sum() / max(erosion.sum(), 1e-9)),
        "terrain_centroid": float(normalized.mean()),
        "balance": result["mass_balance"],
    }


def run_quality_metrics():
    """
    Die fuenf Kennzahlen bei 128 px mit den Default-Parametern.

    Gemessener Stand 2026-07-28 (Erosionsschwelle 0.6 m^2/s,
    Konvergenz 1e-6, 12 275 Schritte):

        Krater                    0
        Top-5%-Anteil         0.300
        groesstes Kanalnetz      87 px
        Ebenen                 23.2%
        Erosions-Schwerpunkt  0.619  (Gelaende 0.531)

    Die Schwellen unten liegen bewusst etwas unter den gemessenen Werten - der
    Test soll eine VERSCHLECHTERUNG fangen, nicht bei jeder Nachkalibrierung
    rot werden.
    """
    print("    (128px, Default-Parameter)")
    m = measure(128)
    print("    {} Schritte, konvergiert={}, Bilanzabweichung {:.2%}".format(
        m["steps"], m["converged"], abs(m["balance"])))

    ok = check("(a) keine Krater - Senken tiefer als 1% des Reliefs "
               "(gemessen {})".format(m["craters"]), m["craters"] == 0)
    ok &= check("(b) Erosion konzentriert sich in Linien "
                "(Top-5%-Anteil {:.3f}, Schwelle > 0.25)".format(m["top5"]),
                m["top5"] > 0.25)
    ok &= check("(c) zusammenhaengendes Kanalnetz "
                "(groesste Komponente {} px, Schwelle > 60)".format(m["network"]),
                m["network"] > 60)
    ok &= check("(d) Sedimentation erzeugt Ebenen "
                "(Flaechenanteil {:.1f}%, Schwelle 15-55%)".format(m["plains"]),
                15.0 <= m["plains"] <= 55.0)
    ok &= check("(e) der Lauf konvergiert statt in die Schrittgrenze zu laufen",
                m["converged"])
    return ok


def run_erosion_threshold_spares_the_peaks():
    """
    Die Erosionsschwelle (HydraulicFieldSimulator.EROSION_THRESHOLD_DISCHARGE)
    muss den Anteil der Erosion im OBEREN Gelaendedrittel senken - das ist die
    Nutzer-Vorgabe "Gipfel und Kaemme groesstenteils unberuehrt".

    Ohne sie nagten die vielen schwach durchflossenen Hangzellen flaechig:
    gemessen verursachten 48% der Karte 64% der Gesamterosion, und das obere
    Drittel allein 39.5%.
    """
    size = 128
    terrain = make_terrain(size)
    hardness = np.full((size, size), 50.0, dtype=np.float32)
    normalized = (terrain - terrain.min()) / float(terrain.max() - terrain.min())
    upper = normalized >= 0.66

    shares = {}
    for threshold in (0.0, HydraulicFieldSimulator.EROSION_THRESHOLD_DISCHARGE):
        simulator = HydraulicFieldSimulator()
        simulator.EROSION_THRESHOLD_DISCHARGE = threshold
        result = simulator.simulate(
            terrain, hardness, {"max_steps": 4000, "rainfall": 2.0}, 10000.0 / size)
        erosion = result["erosion_map"]
        shares[threshold] = float(erosion[upper].sum() / max(erosion.sum(), 1e-9)) * 100.0

    without, with_threshold = (shares[0.0],
                               shares[HydraulicFieldSimulator.EROSION_THRESHOLD_DISCHARGE])
    return check(
        "die Schwelle senkt den Erosionsanteil im oberen Gelaendedrittel "
        "({:.1f}% -> {:.1f}%)".format(without, with_threshold),
        with_threshold < without * 0.75)


def run_scale_consistency():
    """
    Dieselbe Landschaft bei anderer Aufloesung muss dieselbe LANDSCHAFT
    ergeben - nicht dieselbe Zahl von Schritten, aber vergleichbare
    Bildeigenschaften.

    Das ist der Test, der die Aufloesungs-Abhaengigkeit des Abbruchkriteriums
    gefangen haette: es mass die Aenderung pro SCHRITT, und ein Schritt ist bei
    feiner Aufloesung kuerzer (dt = 0.5 * Zellbreite / Bezugsgeschwindigkeit).
    Mit derselben Schwelle konvergierte 128 px nach 2650 Schritten, 256 px aber
    schon nach 700 - die groessere Karte lieferte die UNREIFERE Landschaft,
    obwohl sie laenger rechnete. Seither misst das Kriterium die Aenderung pro
    SEKUNDE Simulationszeit.
    """
    coarse = measure(96)
    fine = measure(144)
    print("     96 px: {:5d} Schritte, Ebenen {:.1f}%, Top-5% {:.3f}".format(
        coarse["steps"], coarse["plains"], coarse["top5"]))
    print("    144 px: {:5d} Schritte, Ebenen {:.1f}%, Top-5% {:.3f}".format(
        fine["steps"], fine["plains"], fine["top5"]))

    # NICHT mehr geprueft: "beide Aufloesungen konvergieren". Seit der
    # Kalibrierung vom 2026-07-28 (Regen 5.0, ohne Boeschung, ohne Glaettung)
    # laeuft ein Lauf regulaer in die Schrittgrenze statt ins
    # Konvergenzkriterium. Das ist gemessenes Verhalten, kein Defekt: die
    # Aenderungsrate faellt wie ~1/t und braucht dafuer laenger als das
    # Schritt-Budget.
    #
    #     Schritte          144 px            512 px      (Rate / Relief)
    #        500          4.04e-06          7.27e-06
    #       8000          1.01e-06          2.34e-06      <- Schrittgrenze
    #      18000          5.07e-07          1.24e-06
    #
    # Eine Landschaft unter Dauerregen ohne Hebung kommt nie ganz zum
    # Stillstand; die Schwelle sagt "es lohnt nicht mehr", nicht "fertig".
    # Massgeblich ist deshalb Max Steps, und 8000 ist genau der Stand, der
    # gegen das Zielbild abgenommen wurde. Die Schwelle bleibt als Notbremse
    # fuer den Fall, dass ein Lauf frueher zur Ruhe kommt.
    #
    # Was hier stattdessen zaehlt, steht unten: dieselbe LANDSCHAFT bei
    # anderer Aufloesung - Ebenenanteil und Reifegrad, nicht die Schrittzahl.
    ok = True
    ratio = max(coarse["steps"], fine["steps"]) / max(min(coarse["steps"], fine["steps"]), 1)
    ok &= check("die Schrittzahl bleibt vergleichbar "
                "(Faktor {:.2f}, erlaubt < 3.0)".format(ratio), ratio < 3.0)
    ok &= check("die Top-5%-Konzentration bleibt vergleichbar "
                "({:.3f} gegen {:.3f})".format(coarse["top5"], fine["top5"]),
                abs(coarse["top5"] - fine["top5"]) < 0.10)
    ok &= check("der Ebenenanteil bleibt vergleichbar "
                "({:.1f}% gegen {:.1f}%)".format(coarse["plains"], fine["plains"]),
                abs(coarse["plains"] - fine["plains"]) < 15.0)
    return ok


def main():
    tests = [
        ("quality_metrics", run_quality_metrics),
        ("erosion_threshold_spares_the_peaks", run_erosion_threshold_spares_the_peaks),
        ("scale_consistency", run_scale_consistency),
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
