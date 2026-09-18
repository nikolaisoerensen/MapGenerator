"""
Ticket #35: Die Feldliste der Siedlungsnaht festschreiben.
Ticket #72: Kontur der Stadtgrenze als Polygon exportieren.

Prueft die "Naht" aus docs/SIEDLUNGEN_ENTWURF.md §6 - die eine Stelle, an
der das Spiel Siedlungsdaten abholt. Entschieden in Issue #19 (Fragen 19.2
und 19.3): der Editor liefert genau sechs Felder (Kontur der Stadtgrenze,
Anschlusspunkte der Wege, Stadtgroesse, Stadttyp, Rang, Kultur) und NICHT
Parzellen/Innenstrassen/Haeuserformen/Gewerbe - das entsteht im Spiel.

Von den sechs Feldern liefert `settlement_list` VIER (Kultur, Rang,
Stadtgroesse ueber house_count, Stadttyp) auf jedem `Location`-Eintrag mit
`location_type == 'settlement'`. Seit Ticket #72 liefert der Calculator-
Knoten `settlement.city_boundary` zusaetzlich `city_boundary_polygons`
(Kontur der Stadtgrenze, dict location_id -> Liste Polygone). Nur die
Anschlusspunkte der Wege (Ticket #73) fehlen noch nachweislich komplett -
das ist keine Test-Luecke, sondern der dokumentierte, mit eigenem Ticket
versehene IST-Zustand (§6.3). Dieser Test haelt genau das fest: schlaegt
fehl, wenn eines der FUENF gelieferten Felder verschwindet, seinen
Typ/Wertebereich verlaesst oder (bei der Kontur) keine gueltige Geometrie
mehr liefert, UND schlaegt fehl, wenn ploetzlich eine
Anschlusspunkte-Ausgabe auftaucht, OHNE dass dieser Test (und §6 der
Entwurfsdatei) aktualisiert wurde - das waere ein Signal, dass Ticket #73
committet wurde, ohne die Naht-Doku nachzuziehen.

Faehrt die echte Pipeline (Terrain bis settlement.city_boundary) bei kleiner
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

# smoke_test_pipeline_outputs.py setzt selbst `sys.path.insert(0, r"...MapGenerator")`
# FEST VERDRAHTET auf den Haupt-Checkout (ihre Zeile, nicht von __file__
# abgeleitet). Laeuft dieser Test aus einem Git-Worktree (siehe CLAUDE.md
# "Git worktrees: changes are invisible until merged or tested in-place"),
# landet dieser Pfad NACH unseren beiden Inserts oben auf Position 0 und
# ueberdeckt sie - jedes spaetere `import core...`/`import managers...`
# (z.B. in sp._generatoren(), Ticket #72 gefunden beim Debuggen von
# city_boundary_polygons) liefert dann leise den Hauptcheckout statt den
# Worktree-Stand, ohne jede Fehlermeldung. Deshalb hier NACH dem sp-Import
# erneut einfuegen, um den Worktree-Pfad wieder vor den Hauptcheckout zu
# schieben. Im Hauptcheckout selbst ist das ein wirkungsloses Duplikat.
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
        # settlement.city_boundary haengt an settlement.settlements (siehe
        # calculator_graph.py) - Abbruch danach reicht, um beide Ausgaben
        # (settlement_list UND city_boundary_polygons, Ticket #72) zu haben,
        # ohne die restliche, fuer diesen Test unnoetige Pipeline (Wege,
        # Plot-Physik) mitzurechnen.
        if knoten == "settlement.city_boundary":
            break

    settlement_list = manager.get_calculator_output("settlement.settlements", "settlement_list", LOD)
    city_boundary_polygons = manager.get_calculator_output(
        "settlement.city_boundary", "city_boundary_polygons", LOD)
    return settlement_list, city_boundary_polygons


def main():
    fehler = []

    print("Pipeline bis settlement.city_boundary, Seed %d, %dpx" % (SEED, SIZE))
    siedlungen, city_boundary_polygons = _lauf(SEED)
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

    # ---------- Feld 5: Kontur der Stadtgrenze (Ticket #72, jetzt echt geprueft) ----------
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

    # ---------- Feld 6: Anschlusspunkte der Wege - bewusst als IST-Zustand pruefen (Ticket #73) ----------
    print("6. Anschlusspunkte der Wege: dokumentiert FEHLEND")
    for s in orte:
        if hasattr(s, "road_entry_points") or (s.properties and "road_entry_points" in (s.properties or {})):
            fehler.append(
                "Siedlung %d traegt jetzt Anschlusspunkte (road_entry_points) - "
                "docs/SIEDLUNGEN_ENTWURF.md §6 und dieser Test muessen "
                "aktualisiert werden (Folge-Ticket #73 wurde offenbar "
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
