"""
Ticket #35: Die Feldliste der Siedlungsnaht festschreiben.
Ticket #72: Kontur der Stadtgrenze als Polygon exportieren.
Ticket #73: Anschlusspunkte der Wege an der Stadtgrenze berechnen.

Prueft die "Naht" aus docs/SIEDLUNGEN_ENTWURF.md §6 - die eine Stelle, an
der das Spiel Siedlungsdaten abholt. Entschieden in Issue #19 (Fragen 19.2
und 19.3): der Editor liefert genau sechs Felder (Kontur der Stadtgrenze,
Anschlusspunkte der Wege, Stadtgroesse, Stadttyp, Rang, Kultur) und NICHT
Parzellen/Innenstrassen/Haeuserformen/Gewerbe - das entsteht im Spiel.

Von den sechs Feldern liefert `settlement_list` VIER (Kultur, Rang,
Stadtgroesse ueber house_count, Stadttyp) auf jedem `Location`-Eintrag mit
`location_type == 'settlement'`. Seit Ticket #72 liefert der Calculator-
Knoten `settlement.city_boundary` zusaetzlich `city_boundary_polygons`
(Kontur der Stadtgrenze, dict location_id -> Liste Polygone). Seit Ticket
#73 liefert `settlement.pathfinding` zusaetzlich `road_entry_points`, ein
Dict {settlement_id: [(x, y), ...]} mit den Schnittpunkten des
ueberregionalen Wegenetzes (`roads`) mit derselben Stadtgrenzen-Kontur -
NICHT als Attribut auf `Location` oder in `s.properties`, sondern als
eigener Calculator-Output (siehe §6.1/§6.2). Damit sind alle sechs Felder
aus §6.1 geliefert; dieser Test haelt fest, dass alle sechs
typ-/wertebereichsgeprueft bleiben.

Faehrt die echte Pipeline (Terrain bis settlement.pathfinding) bei kleiner
Kartengroesse, ein Seed reicht - hier geht es um Feldvorhandensein/-typ,
nicht um statistische Verteilung (die deckt smoke_test_settlement_placement.py
ab).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import smoke_test_pipeline_outputs as sp

# smoke_test_pipeline_outputs.py setzt seinerseits die Projektwurzel an
# Position 0 von sys.path. Sie wird dort aus __file__ abgeleitet, zeigt
# also auf denselben Baum wie hier - der Insert unten ist damit ein
# wirkungsloses Duplikat und steht nur, damit die Reihenfolge auch dann
# stimmt, wenn dieser Import spaeter einmal anders geloest wird.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        # settlement.pathfinding haengt an settlement.city_boundary (siehe
        # calculator_graph.py, Ticket #73) - Abbruch danach reicht, um alle
        # drei Ausgaben (settlement_list, city_boundary_polygons,
        # road_entry_points) zu haben, ohne die restliche, fuer diesen Test
        # unnoetige Pipeline (Plot-Physik) mitzurechnen.
        if knoten == "settlement.pathfinding":
            break

    settlement_list = manager.get_calculator_output("settlement.settlements", "settlement_list", LOD)
    city_boundary_polygons = manager.get_calculator_output(
        "settlement.city_boundary", "city_boundary_polygons", LOD)
    road_entry_points = manager.get_calculator_output("settlement.pathfinding", "road_entry_points", LOD)
    return settlement_list, city_boundary_polygons, road_entry_points


def main():
    fehler = []

    print("Pipeline bis settlement.pathfinding, Seed %d, %dpx" % (SEED, SIZE))
    siedlungen, city_boundary_polygons, road_entry_points = _lauf(SEED)
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

    # ---------- Feld 5: Kontur der Stadtgrenze (Ticket #72) ----------
    print("5. Kontur der Stadtgrenze (city_boundary_polygons: dict location_id -> "
          "Liste[Polygon], Polygon = Liste von (x, y)-Punkten in Karten-Pixel-Koordinaten)")
    if city_boundary_polygons is None:
        fehler.append(
            "settlement.city_boundary liefert city_boundary_polygons=None - "
            "Ticket #72 scheint zurueckgedreht (Ausgabeschluessel fehlt wieder)")
    elif not isinstance(city_boundary_polygons, dict):
        fehler.append(
            "city_boundary_polygons hat den falschen Typ %r (erwartet: dict)"
            % type(city_boundary_polygons))
    else:
        for s in orte:
            polys = city_boundary_polygons.get(s.location_id)
            if polys is None:
                fehler.append(
                    "Siedlung %d fehlt als Schluessel in city_boundary_polygons"
                    % s.location_id)
                continue
            if not isinstance(polys, list) or not polys:
                fehler.append(
                    "Siedlung %d: city_boundary_polygons liefert kein nicht-leeres "
                    "Polygon (%r)" % (s.location_id, polys))
                continue
            for poly in polys:
                if not isinstance(poly, list) or len(poly) < 4:
                    fehler.append(
                        "Siedlung %d: Polygon hat zu wenige Punkte fuer eine "
                        "geschlossene Kontur (%r)" % (s.location_id, poly))
                    continue
                for punkt in poly:
                    if not (isinstance(punkt, tuple) and len(punkt) == 2):
                        fehler.append(
                            "Siedlung %d: Punkt %r ist kein (x, y)-Tupel"
                            % (s.location_id, punkt))
                        break
                    x, y = punkt
                    if not (isinstance(x, (int, float, np.integer, np.floating))
                            and isinstance(y, (int, float, np.integer, np.floating))):
                        fehler.append(
                            "Siedlung %d: Punktkoordinaten %r sind nicht numerisch"
                            % (s.location_id, punkt))
                        break
                    # Grosszuegiger Rand statt exakt [0, SIZE): Marching-Squares
                    # (skimage.measure.find_contours) darf Konturpunkte direkt auf
                    # den Randpixeln liefern - ein zu enger Bereichscheck wuerde
                    # hier Rundungsrauschen als Fehler melden statt echte Ausreisser.
                    if not (-1.0 <= x <= SIZE + 1.0 and -1.0 <= y <= SIZE + 1.0):
                        fehler.append(
                            "Siedlung %d: Punkt %r liegt weit ausserhalb der "
                            "%dpx-Karte" % (s.location_id, punkt, SIZE))
                        break

    # ---------- Feld 6: Anschlusspunkte der Wege (Ticket #73) ----------
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
