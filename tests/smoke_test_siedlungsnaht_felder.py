"""
Die Siedlungsnaht - feste Feldliste zwischen Editor und Spiel
(docs/SIEDLUNGEN_ENTWURF.md §6, Ticket #35).

Was das Spiel von der Siedlungsschicht liest, ist ab jetzt eine feste Liste
von sechs Feldern, nicht "was gerade an internen Feldern existiert". Dieser
Test prueft die Naht von der Abnehmer-Seite: er faehrt eine echte
Pipeline-Ausgabe und zaehlt nach, ob genau diese sechs Felder da sind.

VIER Felder gibt es heute schon (Stadtgroesse, Stadttyp, Rang, Kultur - alle
vier stehen auf jedem core.settlement_generator.Location-Eintrag aus
settlement.settlements/settlement_list) und pruefen entsprechend GRUEN.

ZWEI Felder fehlen heute noch (Kontur der Stadtgrenze als Polygon,
Anschlusspunkte der Wege an der Stadtgrenze) - darauf bauen die Folgetickets
#72 und #73 auf. Dieser Test schlaegt fuer GENAU diese zwei Felder
ABSICHTLICH und NAMENTLICH fehl - nicht als Crash, sondern als
Platzhalter-Erwartung: "dieses Feld gibt es noch nicht, und der Test soll
das laut sagen, bis #72/#73 es liefern." Ein rotes Ergebnis hier ist bis zum
Schliessen von #72/#73 der ERWARTETE Zustand, keine Regression - siehe
docs/SIEDLUNGEN_ENTWURF.md §6.4 und den Morgenbericht der Nacht 2026-09-18.

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
    roads = manager.get_calculator_output(
        "settlement.pathfinding", "roads", LOD)
    return settlement_list, city_mask, roads


# Die sechs Felder der Naht, docs/SIEDLUNGEN_ENTWURF.md §6.1, in derselben
# Reihenfolge wie dort. `attribut=None` heisst: dieses Feld gibt es auf
# Location (noch) nicht - das ist der beabsichtigt rote Teil dieses Tests.
NAHT_FELDER = [
    dict(
        name="Kontur der Stadtgrenze",
        attribut=None,
        ticket="#72",
        begruendung=(
            "existiert nur als Rastermaske (settlement.city_boundary -> "
            "city_mask), nicht als Polygon; die Polygon-Extraktion selbst "
            "gibt es zwar schon (PlotPhysicsSystem._build_city_boundary_"
            "polygons(), core/settlement_generator.py), aber NICHT als "
            "Ausgabe des Calculator-Knotens settlement.city_boundary"),
    ),
    dict(
        name="Anschlusspunkte der Wege",
        attribut=None,
        ticket="#73",
        begruendung=(
            "weder settlement.pathfinding (roads/sea_roads) noch "
            "settlement.city_boundary berechnen, wo eine Strecke die "
            "Stadtgrenze schneidet - roads ist nur eine Punktliste "
            "zwischen zwei Siedlungszentren"),
    ),
    dict(name="Stadtgroesse", attribut="house_count", ticket=None, begruendung=None),
    dict(name="Stadttyp", attribut="settlement_type", ticket=None, begruendung=None),
    dict(name="Rang", attribut="rank", ticket=None, begruendung=None),
    dict(name="Kultur", attribut="culture", ticket=None, begruendung=None),
]


def main():
    fehler = []
    settlement_list, city_mask, roads = _lauf()

    print("Siedlungen in diesem Lauf: %d" % len(settlement_list))
    print("city_mask vorhanden: %s, Wege (roads) in diesem Lauf: %d"
          % (city_mask is not None, len(roads or [])))

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
        print("NICHT IN ORDNUNG - %d Befund(e) (2 davon SOLLEN heute rot "
              "sein - Ticket #72/#73, siehe Docstring):" % len(fehler))
        for f in fehler:
            print("   " + f)
        return 1
    print("Alle Zusicherungen erfuellt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
