"""
Ticket #35: Die Feldliste der Siedlungsnaht festschreiben.

Prueft die "Naht" aus docs/SIEDLUNGEN_ENTWURF.md §6 - die eine Stelle, an
der das Spiel Siedlungsdaten abholt. Entschieden in Issue #19 (Fragen 19.2
und 19.3): der Editor liefert genau sechs Felder (Kontur der Stadtgrenze,
Anschlusspunkte der Wege, Stadtgroesse, Stadttyp, Rang, Kultur) und NICHT
Parzellen/Innenstrassen/Haeuserformen/Gewerbe - das entsteht im Spiel.

Von den sechs Feldern lieferte `settlement_list` zunaechst VIER (Kultur, Rang,
Stadtgroesse ueber house_count, Stadttyp) auf jedem `Location`-Eintrag mit
`location_type == 'settlement'`. Ticket #73 hat das fuenfte Feld
(Anschlusspunkte der Wege) ergaenzt: `settlement.pathfinding` liefert seither
zusaetzlich `road_entry_points`, ein Dict {settlement_id: [(x, y), ...]} mit
den Schnittpunkten des ueberregionalen Wegenetzes (`roads`) mit der
Stadtgrenzen-Kontur, in Karten-Pixel-Koordinaten - NICHT als Attribut auf
`Location` oder in `s.properties`, sondern als eigener Calculator-Output
(siehe §6.1/§6.2). Das sechste Feld (Kontur der Stadtgrenze) ist weiterhin
offen (Ticket #72, §6.3). Dieser Test haelt fest: die VIER `Location`-Felder
bleiben typ-/wertebereichsgeprueft, `road_entry_points` bekommt eine echte
Typ-/Wertebereichspruefung, und die Kontur bleibt als dokumentiert-fehlend
geprueft - schlaegt also fehl, wenn ploetzlich eine Kontur-Ausgabe auftaucht,
OHNE dass dieser Test (und §6 der Entwurfsdatei) aktualisiert wurde.

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
    # smoke_test_pipeline_outputs.py (Zeile ~43) haengt beim Import
    # sys.path.insert(0, "...\\MapGenerator") an - fest auf den Haupt-Checkout,
    # nicht relativ zu __file__. In einem Git-Worktree (siehe CLAUDE.md,
    # Abschnitt "Git worktrees") landet dieser Eintrag VOR dem Worktree-Pfad,
    # den dieses Testmodul selbst oben eingefuegt hat - "core.*"-Importe
    # wuerden dann lautlos den Stand des Haupt-Checkouts statt den des
    # Worktrees pruefen. Reihenfolge hier vor dem Generatorenaufbau wieder
    # herstellen, ohne die geteilte Datei anzufassen (Merge-Risiko mit
    # parallelen Nachtaufgaben, die dieselbe Datei benutzen).
    worktree_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if sys.path[0] != worktree_root:
        sys.path.insert(0, worktree_root)

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
        if knoten == "settlement.pathfinding":
            break

    settlement_list = manager.get_calculator_output("settlement.settlements", "settlement_list", LOD)
    road_entry_points = manager.get_calculator_output("settlement.pathfinding", "road_entry_points", LOD)
    return settlement_list, road_entry_points


def main():
    fehler = []

    print("Pipeline bis settlement.pathfinding, Seed %d, %dpx" % (SEED, SIZE))
    siedlungen, road_entry_points = _lauf(SEED)
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

    # ---------- Kontur der Stadtgrenze: weiterhin offen (Ticket #72, §6.3) ----------
    print("5. Kontur der Stadtgrenze: dokumentiert FEHLEND")
    for s in orte:
        if hasattr(s, "boundary_polygon") or (s.properties and "boundary_polygon" in (s.properties or {})):
            fehler.append(
                "Siedlung %d traegt jetzt eine Kontur (boundary_polygon) - "
                "docs/SIEDLUNGEN_ENTWURF.md §6 und dieser Test muessen "
                "aktualisiert werden (Folge-Ticket aus §6.3 wurde offenbar "
                "umgesetzt)" % s.location_id)

    # ---------- Anschlusspunkte der Wege: seit Ticket #73 geliefert ----------
    print("6. Anschlusspunkte der Wege (road_entry_points: dict je Siedlung, "
          "Liste von (x, y) im Kartenraster)")
    if not isinstance(road_entry_points, dict):
        fehler.append("road_entry_points ist kein dict (%r)" % (type(road_entry_points),))
    else:
        orte_ids = {s.location_id for s in orte}
        fehlende_ids = orte_ids - set(road_entry_points)
        if fehlende_ids:
            fehler.append("road_entry_points fehlt fuer Siedlungs-IDs %s" % sorted(fehlende_ids))
        gesamt_punkte = 0
        for sid, punkte in road_entry_points.items():
            if sid not in orte_ids:
                continue
            if not isinstance(punkte, list):
                fehler.append("Siedlung %d: road_entry_points-Eintrag ist keine Liste (%r)"
                               % (sid, type(punkte)))
                continue
            for punkt in punkte:
                if not (isinstance(punkt, tuple) and len(punkt) == 2
                        and all(isinstance(k, (int, float, np.floating)) for k in punkt)):
                    fehler.append("Siedlung %d: Anschlusspunkt %r ist kein (x, y)-Zahlenpaar"
                                  % (sid, punkt))
                    continue
                px, py = punkt
                if not (0.0 <= px <= SIZE and 0.0 <= py <= SIZE):
                    fehler.append("Siedlung %d: Anschlusspunkt (%.1f, %.1f) liegt ausserhalb "
                                   "des %dpx-Kartenrasters" % (sid, px, py, SIZE))
                gesamt_punkte += 1
        print("   %d Anschlusspunkte insgesamt ueber %d Siedlungen" % (gesamt_punkte, len(orte_ids)))

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
