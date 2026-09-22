"""
Ticket #34: Biomkarte in die Siedlungs-Eignungsrechnung einhaengen.

Die Eignungsrechnung fuer Siedlungen sah bis zu diesem Ticket die Biomkarte
gar nicht - die Kante im Rechengraphen (managers/calculator_graph.py) fehlte.
Siedlungen wurden also platziert, ohne zu wissen, ob der Standort Wueste,
Sumpf oder Wiese ist.

Dieser Test prueft die drei Fallen, die das Ticket ausdruecklich benennt:

  A) Die Kante Biomkarte -> Siedlungs-Eignung existiert im Graphen.
  B) Die Reihenfolge stimmt: biome.integrate_layers wird nie NACH
     settlement.suitability bereit (echter CalculatorDispatcher, kein
     Handschema) - sonst griffe still ein leeres Feld.
  C) Die Kante hat tatsaechlich Wirkung: die Eignungswerte unterscheiden
     sich MESSBAR zwischen biome_map=None (vorher) und einer echten
     biome_map (nachher), und ein gutes Biom (grasland) wird an einem sonst
     identischen Standort hoeher bewertet als ein schlechtes (halbwueste).
  D) Greift die Biomkarte einmal nicht (None, oder eine unbekannte Biom-ID),
     wird das laut geloggt (Pipeline-Logger) statt still weiterzurechnen.

Laeuft mit synthetischen Arrays, nicht der vollen Pipeline - schnell genug
fuer jeden Lauf, und isoliert genug, um die Ursache klar zuzuordnen.
"""
import logging
import sys

import numpy as np

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

from core.settlement_generator import TerrainSuitabilityAnalyzer, BIOME_SIEDLUNGSEIGNUNG
from managers.calculator_graph import CALCULATOR_GRAPH, CalculatorDispatcher


class _LogFang(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def check(label, condition):
    print(("[OK]  " if condition else "[FEHLER]") + " " + label)
    return bool(condition)


def main():
    ok = True

    # ---------- A: Graph-Kante ----------
    deps = CALCULATOR_GRAPH["settlement.suitability"].depends_on
    ok &= check("settlement.suitability haengt an biome.integrate_layers: %s" % (deps,),
                "biome.integrate_layers" in deps)

    # ---------- B: Reihenfolge ueber echten Dispatcher ----------
    dispatcher = CalculatorDispatcher({cid: (lambda ctx: None) for cid in CALCULATOR_GRAPH})
    for generator in {spec.generator for spec in CALCULATOR_GRAPH.values()}:
        dispatcher.request(generator, 2)
    erste_runde = {}
    for round_n in range(1, 12):
        while True:
            ready = dispatcher.get_ready_nodes(round_n)
            if not ready:
                break
            for cid in ready:
                erste_runde.setdefault(cid, round_n)
                dispatcher.mark_completed(cid, round_n)
    ok &= check("Pipeline laeuft mit der neuen Kante vollstaendig durch",
                dispatcher.is_fully_done())
    ok &= check(
        "settlement.suitability wird nie vor biome.integrate_layers bereit "
        "(Runde %s gegen %s)" % (erste_runde.get("settlement.suitability"),
                                  erste_runde.get("biome.integrate_layers")),
        erste_runde.get("settlement.suitability", 0)
        >= erste_runde.get("biome.integrate_layers", 0))

    # ---------- C: Vorher/Nachher, synthetisches Feld ----------
    n = 64
    heightmap = np.full((n, n), 100.0, dtype=np.float32)   # ueberall Land, flach
    slopemap = np.zeros((n, n, 2), dtype=np.float32)        # ueberall flach
    water_map = np.zeros((n, n), dtype=np.float32)          # kein Wasser

    analyzer = TerrainSuitabilityAnalyzer(1.0, n)

    vorher = analyzer.create_combined_suitability(heightmap, slopemap, water_map)

    # Linke Haelfte grasland (3, gute Eignung), rechte Haelfte halbwueste
    # (14, schlechte Eignung) - alles andere am Standort identisch.
    biome_map = np.full((n, n), 3, dtype=np.uint8)
    biome_map[:, n // 2:] = 14

    nachher = analyzer.create_combined_suitability(heightmap, slopemap, water_map,
                                                    biome_map=biome_map)

    diff = float(np.mean(np.abs(nachher.astype(np.float64) - vorher.astype(np.float64))))
    print("   mittlere |Nachher-Vorher| Abweichung: %.4f" % diff)
    ok &= check("Eignungswerte aendern sich messbar durch die Biomkarte (diff > 0.01)",
                diff > 0.01)

    grasland_wert = float(np.mean(nachher[:, :n // 2]))
    wueste_wert = float(np.mean(nachher[:, n // 2:]))
    print("   grasland-Mittel=%.4f  halbwueste-Mittel=%.4f" % (grasland_wert, wueste_wert))
    ok &= check("grasland wird hoeher bewertet als halbwueste an sonst gleichem Standort",
                grasland_wert > wueste_wert)

    # ---------- D: stille Rueckfaelle sind verboten ----------
    fang = _LogFang()
    pipeline_logger = logging.getLogger("Pipeline")
    pipeline_logger.addHandler(fang)
    pipeline_logger.setLevel(logging.WARNING)
    try:
        ohne_biom = analyzer.evaluate_biome_suitability(None)
    finally:
        pipeline_logger.removeHandler(fang)
    ok &= check("biome_map=None liefert None (neutraler Faktor, keine Daempfung)",
                ohne_biom is None)
    ok &= check("biome_map=None schreibt eine laute Logzeile statt still weiterzurechnen",
                any("biome_map fehlt" in r for r in fang.records))

    fang2 = _LogFang()
    pipeline_logger.addHandler(fang2)
    pipeline_logger.setLevel(logging.WARNING)
    try:
        seltsame_map = np.full((8, 8), 99, dtype=np.uint8)  # 99 ist keine gueltige Biom-ID
        resultat = analyzer.evaluate_biome_suitability(seltsame_map)
    finally:
        pipeline_logger.removeHandler(fang2)
    ok &= check("unbekannte Biom-ID liefert trotzdem ein Feld (neutral 1.0)",
                resultat is not None and np.allclose(resultat, 1.0))
    ok &= check("unbekannte Biom-ID schreibt ebenfalls eine laute Logzeile",
                any("unbekannte" in r for r in fang2.records))

    # Alle 27 Biom-IDs abgedeckt? (15 Basis + 12 Superbiome)
    ok &= check("BIOME_SIEDLUNGSEIGNUNG deckt alle 27 bekannten Biom-IDs ab (0-26)",
                set(BIOME_SIEDLUNGSEIGNUNG.keys()) == set(range(27)))

    print("")
    if not ok:
        print("NICHT IN ORDNUNG")
        return 1
    print("Alle Zusicherungen erfuellt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
