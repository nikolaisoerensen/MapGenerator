"""
Path: tests/smoke_test_siedlungsnaht_felder.py

Prueft die Feldliste der Siedlungsnaht (Ticket #35, docs/SIEDLUNGEN_ENTWURF.md
§6): die eine Stelle, an der das Spiel Siedlungsdaten vom Editor abholt - mit
genau diesen Feldern, nicht mehr.

Von den acht Feldern aus §6 liefert der Editor HEUTE sechs (city_id,
city_center, city_size als house_count+radius, city_type, rank, culture) -
alles Werte der Location-Dataclass (core/settlement_generator.py), gefuellt in
SettlementGenerator.calculate_settlements(). Diese sechs werden hier gegen
eine echte, headless gerechnete Karte geprueft.

Zwei Felder fehlen HEUTE noch und sind Gegenstand der beiden parallel
laufenden Tickets #72 (city_boundary_polygons) und #73 (road_entry_points).
Nach der Nutzervorgabe fuer dieses Ticket wird das NICHT gruen geluegen: die
beiden Pruefungen unten schlagen ABSICHTLICH und LESBAR fehl, mit Verweis auf
das jeweilige Ticket, statt eine nicht existierende Berechnung vorzutaeuschen
oder die Pruefung stillschweigend wegzulassen. Der Gesamt-Exitcode dieses
Skripts ist deshalb, solange #72/#73 offen sind, ABSICHTLICH 1 - das ist der
ehrliche Befund, nicht ein kaputter Test.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_siedlungsnaht_felder.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import smoke_test_pipeline_outputs as sp

# Kleine, billige Karte reicht - diese Pruefung braucht keine statistische
# Verteilung (das leistet bereits tests/smoke_test_settlement_placement.py),
# nur EINEN echten Satz Siedlungen mit allen Feldern befuellt.
SIZE = 96
LOD = 3
KM = 15.0
SEED = 20260804

ERLAUBTE_RAENGE = {"dorf", "siedlung", "stadt"}
ERLAUBTE_TYPEN = {"bergdorf", "marktstadt", "agrarstadt", "sonstige"}
RANG_HAEUSER = {"dorf": (15, 25), "siedlung": (25, 35), "stadt": (35, 50)}


def _hole_siedlungen():
    """Faehrt die echte Pipeline bis einschliesslich settlement.settlements -
    der fruehste Knoten, der settlement_list liefert. 1:1 uebernommenes Muster
    aus tests/smoke_test_settlement_placement.py._lauf(), damit hier keine
    zweite, abweichende Version der Pipeline-Ansteuerung entsteht."""
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


def _pruefe_gelieferte_felder(siedlungen, fehler):
    """Die sechs HEUTE gelieferten Naht-Felder aus §6: city_id, city_center,
    city_size (house_count+radius), city_type, rank, culture. Jedes muss auf
    jeder Siedlung gesetzt UND plausibel sein - nicht nur "nicht None"."""
    if not siedlungen:
        fehler.append("Keine Siedlungen erzeugt - Feldpruefung nicht moeglich "
                      "(Vorbedingung fuer den ganzen Test verletzt)")
        return

    gesehene_ids = set()
    geprueft = 0
    for s in siedlungen:
        if s.location_type != "settlement":
            continue  # Landmarks/Roadsites sind nicht Teil dieser Naht (§6.2)
        geprueft += 1

        # city_id
        if not isinstance(s.location_id, int):
            fehler.append(f"city_id ist kein int bei Siedlung {s.location_id!r}: {type(s.location_id)}")
        elif s.location_id in gesehene_ids:
            fehler.append(f"city_id doppelt vergeben: {s.location_id}")
        gesehene_ids.add(s.location_id)

        # city_center
        if not (isinstance(s.x, float) and isinstance(s.y, float)):
            fehler.append(f"city_center (x,y) nicht als float geliefert bei Siedlung {s.location_id}: "
                          f"({type(s.x)}, {type(s.y)})")
        if not (0.0 <= s.x <= SIZE and 0.0 <= s.y <= SIZE):
            fehler.append(f"city_center ausserhalb des {SIZE}x{SIZE}-Rasters bei Siedlung "
                          f"{s.location_id}: ({s.x}, {s.y})")

        # city_size: house_count (Haeuser) + radius (Pixel), muessen zueinander passen
        if not isinstance(s.house_count, int) or not (15 <= s.house_count <= 50):
            fehler.append(f"city_size/house_count ausserhalb 15-50 bei Siedlung "
                          f"{s.location_id}: {s.house_count!r}")
        elif s.rank in RANG_HAEUSER:
            lo, hi = RANG_HAEUSER[s.rank]
            if not (lo <= s.house_count <= hi):
                fehler.append(f"city_size/house_count {s.house_count} passt nicht zu rank "
                              f"{s.rank!r} ({lo}-{hi}) bei Siedlung {s.location_id}")
        if not (isinstance(s.radius, float) and s.radius > 0.0):
            fehler.append(f"city_size/radius nicht positiv bei Siedlung {s.location_id}: {s.radius!r}")

        # city_type
        if s.settlement_type not in ERLAUBTE_TYPEN:
            fehler.append(f"city_type unbekannt bei Siedlung {s.location_id}: {s.settlement_type!r} "
                          f"(erlaubt: {sorted(ERLAUBTE_TYPEN)})")

        # rank
        if s.rank not in ERLAUBTE_RAENGE:
            fehler.append(f"rank unbekannt bei Siedlung {s.location_id}: {s.rank!r} "
                          f"(erlaubt: {sorted(ERLAUBTE_RAENGE)})")

        # culture
        if not s.culture:
            fehler.append(f"culture leer bei Siedlung {s.location_id}")

    if geprueft == 0:
        fehler.append("settlement_list enthielt keine location_type=='settlement'-Eintraege")
    else:
        print(f"   {geprueft} Siedlung(en) geprueft, alle sechs Felder vorhanden")


def _platzhalter_stadtgrenze():
    """Feld city_boundary_polygons (§6) - Kontur der Stadtgrenze als Polygon.

    HEUTE NICHT geliefert: SettlementData hat kein eigenes Feld dafuer. Die
    Vorstufe existiert nur INTERN in PlotPhysicsSystem._city_polygons
    (core/settlement_generator.py Zeile ~2519, gebaut von
    _build_city_boundary_polygons() Zeile ~2738-2767 per Marching-Squares
    ueber city_mask) und wird dort nur fuer die Plot-Node-Physik benutzt, nie
    nach aussen durchgereicht. SettlementData.city_mask ist ein Pixel-Raster
    (Settlement-ID pro Pixel), keine Polygon-Kontur - das erfuellt die
    Feldvorgabe aus §6 nicht.

    Das ist ABSICHTLICH ein roter Platzhalter statt eines gruen geluegenen
    Tests - siehe Ticket #72."""
    return ("PLATZHALTER city_boundary_polygons: heute NICHT geliefert (nur "
            "SettlementData.city_mask als Pixel-Raster, keine Polygon-Kontur "
            "je Stadt auf SettlementData) - siehe Ticket #72 und "
            "docs/SIEDLUNGEN_ENTWURF.md §6.1")


def _platzhalter_wegeanschluss():
    """Feld road_entry_points (§6) - wo eine Wegverbindung die Stadtgrenze
    schneidet.

    HEUTE NICHT geliefert: SettlementData.roads/sea_roads/landmark_roads
    (core/settlement_generator.py Zeile 144-150, befuellt in
    calculate_road_network() ab Zeile 5705) enthalten nur die volle,
    geglaettete Pfad-Polylinie von Ortszentrum (Location.x/y) zu
    Ortszentrum - keinen gesonderten Schnittpunkt mit der Stadtgrenze. Ein
    Schnittpunkt liesse sich erst berechnen, wenn city_boundary_polygons
    (siehe oben, #72) tatsaechlich existiert - die beiden Platzhalter haengen
    also voneinander ab.

    Das ist ABSICHTLICH ein roter Platzhalter statt eines gruen geluegenen
    Tests - siehe Ticket #73."""
    return ("PLATZHALTER road_entry_points: heute NICHT geliefert (roads/"
            "sea_roads/landmark_roads enden am Ortszentrum, nicht an der "
            "Stadtgrenze; haengt zudem an city_boundary_polygons aus #72) - "
            "siehe Ticket #73 und docs/SIEDLUNGEN_ENTWURF.md §6.1")


def main():
    fehler = []
    platzhalter = []

    print("1. Gelieferte Felder: city_id, city_center, city_size, city_type, rank, culture")
    siedlungen = _hole_siedlungen()
    _pruefe_gelieferte_felder(siedlungen, fehler)

    print("\n2. Platzhalter fuer heute NICHT gelieferte Felder (#72/#73)")
    platzhalter.append(_platzhalter_stadtgrenze())
    platzhalter.append(_platzhalter_wegeanschluss())
    for p in platzhalter:
        print("   " + p)

    print("")
    if fehler:
        print("ECHTE FEHLER - %d Befund(e), das sind keine Platzhalter:" % len(fehler))
        for f in fehler:
            print("   " + f)
        print("")

    if fehler or platzhalter:
        print("Zusammenfassung: %d echte(r) Fehler, %d beabsichtigte(r) Platzhalter "
              "(city_boundary_polygons/#72, road_entry_points/#73)."
              % (len(fehler), len(platzhalter)))
        print("Exitcode 1 ist hier ERWARTET, solange #72 und #73 offen sind - "
              "kein gruen geluegener Test.")
        return 1

    print("Alle acht Naht-Felder aus §6 vollstaendig geliefert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
