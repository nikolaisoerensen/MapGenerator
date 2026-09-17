"""
Ticket #35: Die Feldliste der Siedlungsnaht festschreiben.

Prueft die "Naht" aus docs/SIEDLUNGEN_ENTWURF.md §6 - die eine Stelle, an
der das Spiel Siedlungsdaten abholt. Entschieden in Issue #19 (Fragen 19.2
und 19.3): der Editor liefert genau sechs Felder (Kontur der Stadtgrenze,
Anschlusspunkte der Wege, Stadtgroesse, Stadttyp, Rang, Kultur) und NICHT
Parzellen/Innenstrassen/Haeuserformen/Gewerbe - das entsteht im Spiel.

Von den sechs Feldern liefert `settlement_list` heute VIER (Kultur, Rang,
Stadtgroesse ueber house_count, Stadttyp) auf jedem `Location`-Eintrag mit
`location_type == 'settlement'`. Die anderen zwei (Kontur, Anschlusspunkte)
fehlen nachweislich komplett - das ist keine Test-Luecke, sondern der
dokumentierte, mit eigenen Tickets versehene IST-Zustand (§6.3). Dieser Test
haelt genau das fest: schlaegt fehl, wenn eines der VIER gelieferten Felder
verschwindet oder seinen Typ/Wertebereich verlaesst, UND schlaegt fehl, wenn
ploetzlich eine Kontur/Anschlusspunkte-Ausgabe auftaucht, OHNE dass dieser
Test (und §6 der Entwurfsdatei) aktualisiert wurde - das waere ein Signal,
dass eines der beiden Folge-Tickets committet wurde, ohne die Naht-Doku
nachzuziehen.

Faehrt die echte Pipeline (Terrain bis Settlements) bei kleiner Kartengroesse,
ein Seed reicht - hier geht es um Feldvorhandensein/-typ, nicht um
statistische Verteilung (die deckt smoke_test_settlement_placement.py ab).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import smoke_test_pipeline_outputs as sp

SIZE = 96
LOD = 3
KM = 15.0
SEED = 20260804

# Aus docs/SIEDLUNGEN_ENTWURF.md §6.1 - die vier heute gelieferten Naht-Felder.
KULTUREN_PLATZHALTER_ANZAHL = 9  # Issue #19.5: ausdruecklich vorlaeufig, Anzahl pruefen reicht
RAENGE = {"dorf", "siedlung", "stadt"}
STADTTYPEN = {"bergdorf", "marktstadt", "agrarstadt", "sonstige"}
RANG_HAEUSER = {"dorf": (15, 25), "siedlung": (25, 35), "stadt": (35, 50)}


def _lauf(seed):
    from managers.data_lod_manager import DataLODManager
    from managers.calculator_graph import CALCULATOR_GRAPH

    sp._qt()
    manager = DataLODManager()
    manager.set_map_distance_km(KM)
    manager.set_map_seed(seed)
    parameter = dict(sp._parameter())
    parameter["map_size"] = SIZE
    parameter["map_seed"] = seed
    generatoren = sp._generatoren(manager, None)
    for knoten in CALCULATOR_GRAPH:
        manager.set_calculator_target_lod(knoten, LOD)
    for generator in generatoren.values():
        if hasattr(generator, "set_active_parameters"):
            generator.set_active_parameters(parameter)

    for knoten in sp._reihenfolge():
        spec = CALCULATOR_GRAPH[knoten]
        generator = generatoren.get(spec.generator)
        methode = getattr(generator, "_calc_" + knoten.split(".", 1)[1], None)
        if methode is None:
            continue
        methode(knoten, LOD)
        if knoten == "settlement.settlements":
            break

    return manager.get_calculator_output("settlement.settlements", "settlement_list", LOD)


def main():
    fehler = []

    print("Pipeline bis settlement.settlements, Seed %d, %dpx" % (SEED, SIZE))
    siedlungen = _lauf(SEED)
    if not siedlungen:
        print("FEHLER: settlement_list ist leer - kann Naht-Felder nicht pruefen")
        return 1

    orte = [s for s in siedlungen if s.location_type == 'settlement']
    if not orte:
        fehler.append("keine Eintraege mit location_type == 'settlement'")
    print("   %d Siedlungen (von %d Location-Eintraegen gesamt)" % (len(orte), len(siedlungen)))

    # ---------- die vier gelieferten Felder ----------
    print("1. Kultur (str, einer von %d Platzhaltern)" % KULTUREN_PLATZHALTER_ANZAHL)
    kulturen_gesehen = set()
    for s in orte:
        if not isinstance(s.culture, str) or not s.culture:
            fehler.append("Siedlung %d: culture fehlt oder ist kein nicht-leerer str (%r)"
                           % (s.location_id, s.culture))
        else:
            kulturen_gesehen.add(s.culture)
    print("   gesehen: %s" % sorted(kulturen_gesehen))

    print("2. Rang (str, einer von %s)" % sorted(RAENGE))
    for s in orte:
        if s.rank not in RAENGE:
            fehler.append("Siedlung %d: rank=%r ist keiner von %s"
                           % (s.location_id, s.rank, sorted(RAENGE)))

    print("3. Stadtgroesse (house_count: int, im Bereich des Rangs)")
    for s in orte:
        if s.rank not in RANG_HAEUSER:
            continue
        lo, hi = RANG_HAEUSER[s.rank]
        if not isinstance(s.house_count, (int, np.integer)) or not (lo <= s.house_count <= hi):
            fehler.append("Siedlung %d: house_count=%r ausserhalb [%d, %d] fuer Rang %r"
                           % (s.location_id, s.house_count, lo, hi, s.rank))

    print("4. Stadttyp (settlement_type: str, einer von %s)" % sorted(STADTTYPEN))
    for s in orte:
        if s.settlement_type not in STADTTYPEN:
            fehler.append("Siedlung %d: settlement_type=%r ist keiner von %s"
                           % (s.location_id, s.settlement_type, sorted(STADTTYPEN)))

    # ---------- die zwei fehlenden Felder: bewusst als IST-Zustand pruefen ----------
    print("5. Kontur der Stadtgrenze / Anschlusspunkte der Wege: dokumentiert FEHLEND")
    for s in orte:
        if hasattr(s, "boundary_polygon") or (s.properties and "boundary_polygon" in (s.properties or {})):
            fehler.append(
                "Siedlung %d traegt jetzt eine Kontur (boundary_polygon) - "
                "docs/SIEDLUNGEN_ENTWURF.md §6 und dieser Test muessen "
                "aktualisiert werden (Folge-Ticket aus §6.3 wurde offenbar "
                "umgesetzt)" % s.location_id)
        if hasattr(s, "road_entry_points") or (s.properties and "road_entry_points" in (s.properties or {})):
            fehler.append(
                "Siedlung %d traegt jetzt Anschlusspunkte (road_entry_points) - "
                "docs/SIEDLUNGEN_ENTWURF.md §6 und dieser Test muessen "
                "aktualisiert werden (Folge-Ticket aus §6.3 wurde offenbar "
                "umgesetzt)" % s.location_id)

    print()
    if fehler:
        print("FEHLGESCHLAGEN (%d):" % len(fehler))
        for f in fehler:
            print("  - " + f)
        return 1

    print("Alle Naht-Feld-Pruefungen bestanden (§6.4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
