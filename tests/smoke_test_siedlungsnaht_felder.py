"""
Die Siedlungsnaht - feste Feldliste zwischen Editor und Spiel
(docs/SIEDLUNGEN_ENTWURF.md §6, Ticket #35).

Was das Spiel von der Siedlungsschicht liest, ist ab jetzt eine feste Liste
von sechs Feldern, nicht "was gerade an internen Feldern existiert". Dieser
Test prueft die Naht von der Abnehmer-Seite: er faehrt eine echte
Pipeline-Ausgabe und zaehlt nach, ob genau diese sechs Felder da sind.

FUENF Felder gibt es heute schon und pruefen entsprechend GRUEN:
Stadtgroesse, Stadttyp, Rang, Kultur (alle vier stehen auf jedem
core.settlement_generator.Location-Eintrag aus
settlement.settlements/settlement_list) sowie seit Ticket #72 die Kontur der
Stadtgrenze (settlement.city_boundary -> city_boundary_polygons, ein
Marching-Squares-Polygon je Siedlung).

EIN Feld fehlt heute noch (Anschlusspunkte der Wege an der Stadtgrenze) -
darauf baut das Folgeticket #73 auf ("Blocked by #72", siehe
docs/SIEDLUNGEN_ENTWURF.md §6.2 - erst seit #72 gibt es ueberhaupt eine
Kontur, gegen die eine Strecke geschnitten werden koennte). Dieser Test
schlaegt fuer GENAU dieses eine Feld ABSICHTLICH und NAMENTLICH fehl - nicht
als Crash, sondern als Platzhalter-Erwartung: "dieses Feld gibt es noch
nicht, und der Test soll das laut sagen, bis #73 es liefert." Ein rotes
Ergebnis hier ist bis zum Schliessen von #73 der ERWARTETE Zustand, keine
Regression - siehe docs/SIEDLUNGEN_ENTWURF.md §6.4 und den Morgenbericht der
Nacht 2026-09-18.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_siedlungsnaht_felder.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import smoke_test_pipeline_outputs as sp

SIZE = 96
LOD = 3
KM = 15.0
SEED = 20260804


def _lauf():
    """Faehrt die echte Pipeline bis settlement.city_boundary UND
    settlement.pathfinding fertig sind - genau wie
    smoke_test_settlement_placement.py._lauf(), nur mit einem spaeteren
    Abbruchpunkt, weil hier zusaetzlich city_mask und roads gebraucht
    werden, nicht nur settlement_list."""
    from managers.data_lod_manager import DataLODManager
    from managers.calculator_graph import CALCULATOR_GRAPH

    sp._qt()
    manager = DataLODManager()
    manager.set_map_distance_km(KM)
    manager.set_map_seed(SEED)
    parameter = dict(sp._parameter())
    parameter["map_size"] = SIZE
    parameter["map_seed"] = SEED
    generatoren = sp._generatoren(manager, None)
    for knoten in CALCULATOR_GRAPH:
        manager.set_calculator_target_lod(knoten, LOD)
    for generator in generatoren.values():
        if hasattr(generator, "set_active_parameters"):
            generator.set_active_parameters(parameter)

    ziel = {"settlement.city_boundary", "settlement.pathfinding"}
    erledigt = set()
    for knoten in sp._reihenfolge():
        spec = CALCULATOR_GRAPH[knoten]
        generator = generatoren.get(spec.generator)
        methode = getattr(generator, "_calc_" + knoten.split(".", 1)[1], None)
        if methode is None:
            continue
        methode(knoten, LOD)
        erledigt.add(knoten)
        if ziel <= erledigt:
            break

    settlement_list = manager.get_calculator_output(
        "settlement.settlements", "settlement_list", LOD)
    city_mask = manager.get_calculator_output(
        "settlement.city_boundary", "city_mask", LOD)
    city_boundary_polygons = manager.get_calculator_output(
        "settlement.city_boundary", "city_boundary_polygons", LOD)
    roads = manager.get_calculator_output(
        "settlement.pathfinding", "roads", LOD)
    return settlement_list, city_mask, city_boundary_polygons, roads


# Die sechs Felder der Naht, docs/SIEDLUNGEN_ENTWURF.md §6.1, in derselben
# Reihenfolge wie dort. `attribut=None` heisst: dieses Feld gibt es auf
# Location (noch) nicht - das ist der beabsichtigt rote Teil dieses Tests.
#
# "Kontur der Stadtgrenze" (Feld 1) ist seit Ticket #72 kein Platzhalter
# mehr: `attribut` traegt jetzt den echten Zugriffspfad
# (settlement.city_boundary -> city_boundary_polygons, ein Dict
# location_id -> Liste von Punktlisten, siehe _calc_city_boundary()). Weil
# das kein Attribut auf Location ist wie die vier folgenden Felder, sondern
# ein eigener Calculator-Output, bekommt es `quelle="city_boundary_polygons"`
# und wird unten in main() ueber einen eigenen Zweig geprueft statt ueber
# den generischen getattr(settlement, attribut)-Zweig.
NAHT_FELDER = [
    dict(
        name="Kontur der Stadtgrenze",
        attribut="city_boundary_polygons",
        quelle="city_boundary_polygons",
        ticket=None,
        begruendung=None,
    ),
    dict(
        name="Anschlusspunkte der Wege",
        attribut=None,
        quelle=None,
        ticket="#73",
        begruendung=(
            "weder settlement.pathfinding (roads/sea_roads) noch "
            "settlement.city_boundary berechnen, wo eine Strecke die "
            "Stadtgrenze schneidet - roads ist nur eine Punktliste "
            "zwischen zwei Siedlungszentren"),
    ),
    dict(name="Stadtgroesse", attribut="house_count", quelle=None, ticket=None, begruendung=None),
    dict(name="Stadttyp", attribut="settlement_type", quelle=None, ticket=None, begruendung=None),
    dict(name="Rang", attribut="rank", quelle=None, ticket=None, begruendung=None),
    dict(name="Kultur", attribut="culture", quelle=None, ticket=None, begruendung=None),
]


def main():
    fehler = []
    settlement_list, city_mask, city_boundary_polygons, roads = _lauf()

    print("Siedlungen in diesem Lauf: %d" % len(settlement_list))
    print("city_mask vorhanden: %s, Wege (roads) in diesem Lauf: %d"
          % (city_mask is not None, len(roads or [])))

    kartengroesse = city_mask.shape[0] if city_mask is not None else SIZE

    if not settlement_list:
        fehler.append(
            "settlement_list ist leer - die Naht laesst sich an diesem "
            "Lauf gar nicht pruefen (Kartengroesse/Seed pruefen)")
        print("\nNICHT IN ORDNUNG - %d Befund(e):" % len(fehler))
        for f in fehler:
            print("   " + f)
        return 1

    print("\nDie sechs Felder der Siedlungsnaht (docs/SIEDLUNGEN_ENTWURF.md §6.1):")
    for feld in NAHT_FELDER:
        name = feld["name"]
        attribut = feld["attribut"]
        quelle = feld.get("quelle")

        if attribut is None:
            # ABSICHTLICH ROT: dieses Feld gibt es noch nicht, siehe Docstring.
            print("   FEHLT   %-28s -> Folgeticket %s (%s)"
                  % (name, feld["ticket"], feld["begruendung"]))
            fehler.append(
                "Naht-Feld '%s' fehlt noch - %s. Folgeticket %s "
                "(docs/SIEDLUNGEN_ENTWURF.md §6.2). Das ist ein "
                "beabsichtigter Platzhalter-Fehlschlag, keine Regression."
                % (name, feld["begruendung"], feld["ticket"]))
            continue

        if quelle == "city_boundary_polygons":
            # Eigener Zweig statt getattr(settlement, attribut): dieses Feld
            # ist ein Calculator-Output (Dict location_id -> Liste von
            # Punktlisten), kein Attribut auf Location. Echte Pruefung
            # (Ticket #72): Polygon vorhanden, mindestens 3 Punkte,
            # Koordinaten liegen im Kartenbereich.
            fehlende_ids = []
            ungueltige = []
            for s in settlement_list:
                polygone = (city_boundary_polygons or {}).get(s.location_id)
                if not polygone or not any(len(p) >= 3 for p in polygone):
                    fehlende_ids.append(s.location_id)
                    continue
                for punktliste in polygone:
                    for (x, y) in punktliste:
                        if not (0 <= x <= kartengroesse and 0 <= y <= kartengroesse):
                            ungueltige.append(s.location_id)
                            break
            if fehlende_ids:
                print("   ROT     %-28s -> bei %d/%d Siedlungen kein Polygon "
                      "mit >=3 Punkten (IDs %s)"
                      % (name, len(fehlende_ids), len(settlement_list), fehlende_ids))
                fehler.append(
                    "Naht-Feld '%s' hat bei %d von %d Siedlungen kein "
                    "gueltiges Polygon (location_id %s)"
                    % (name, len(fehlende_ids), len(settlement_list), fehlende_ids))
            elif ungueltige:
                print("   ROT     %-28s -> bei %s Koordinaten ausserhalb "
                      "des Kartenbereichs [0, %d]" % (name, ungueltige, kartengroesse))
                fehler.append(
                    "Naht-Feld '%s' hat Koordinaten ausserhalb des "
                    "Kartenbereichs (location_id %s)" % (name, ungueltige))
            else:
                beispiel_id = settlement_list[0].location_id
                beispiel = city_boundary_polygons[beispiel_id][0]
                print("   OK      %-28s -> z.B. %d Punkte fuer Siedlung %s "
                      "(city_boundary_polygons)" % (name, len(beispiel), beispiel_id))
            continue

        # "" (leerer String) und 0 (bei house_count ausserhalb 15-50 nie
        # gueltig) gelten als fehlend, nicht nur ein nicht vorhandenes
        # Attribut - ein Feld, das zwar existiert aber nie befuellt wird,
        # waere von aussen genauso wertlos wie ein fehlendes (CLAUDE.md:
        # stille Ruckfaelle brauchen eine laute Zeile, kein .get(key, "")).
        fehlende_ids = [
            s.location_id for s in settlement_list
            if not hasattr(s, attribut) or getattr(s, attribut) in (None, "", 0)
        ]
        if fehlende_ids:
            print("   ROT     %-28s -> bei %d/%d Siedlungen leer/fehlend (IDs %s)"
                  % (name, len(fehlende_ids), len(settlement_list), fehlende_ids))
            fehler.append(
                "Naht-Feld '%s' ist bei %d von %d Siedlungen leer oder "
                "fehlend (location_id %s)"
                % (name, len(fehlende_ids), len(settlement_list), fehlende_ids))
        else:
            beispiel = getattr(settlement_list[0], attribut)
            print("   OK      %-28s -> z.B. %r (Location.%s)"
                  % (name, beispiel, attribut))

    print("\nWas NICHT zur Naht gehoert (§6.3): Parzellen, Innenstrassen, "
          "Haeuserformen, Gewerbe - das entsteht im Spiel, nicht im Editor.")

    print("")
    if fehler:
        print("NICHT IN ORDNUNG - %d Befund(e) (1 davon SOLL heute rot "
              "sein - Ticket #73, siehe Docstring):" % len(fehler))
        for f in fehler:
            print("   " + f)
        return 1
    print("Alle Zusicherungen erfuellt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
