"""
Ticket #37: Flussnetz als Linienzuege exportieren.

Prueft `gui/utils/map_export.py` (`_flusslinien_aus_graph()`/`vektordaten()`)
gegen eine echte Pipeline-Ausgabe (WELTKARTE_AKTIV/WELTFLUESSE_AKTIV sind
Standard = True). Drei Pruefungen, wie in docs/NACHTBETRIEB.md fuer neue
Exportpfade verlangt:

    A. ZWEI-ENDEN-VERGLEICH (SPEZIFIKATION-Lektion vom 2026-08-24, siehe
       CLAUDE.md "Wenn alle Einzelpruefungen gruen sind..."): die Vereinigung
       aller exportierten Linienzug-Knoten muss (fast) alle Knoten des
       Graphen abdecken - sonst waere die Astpunkt-Logik in
       _flusslinien_aus_graph() lueckenhaft, ohne dass irgendein Einzeltest
       das gesehen haette.
    B. ZUSAMMENHANG - kein Linienzug darf zwischen zwei benachbarten Punkten
       einen unplausibel grossen Sprung machen (Indiz fuer einen Astpunkt-
       Bug, der zwei unverbundene Aeste zusammenklebt).
    C. STRENGES JSON (RFC 8259, `allow_nan=False`) - der staerkste hier
       pruefbare Ersatzbeleg fuer "von Godot einlesbar", da kein Godot-Binary
       in dieser Umgebung verfuegbar ist (siehe Kommentar in
       gui/utils/map_export.py export_all_layers()).

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_flussexport_vektor.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import smoke_test_pipeline_outputs as sp

SIZE = 128
LOD = 3
KM = 15.0
SEED = 20260730


def _pipeline_bis_river_graph():
    from managers.data_lod_manager import DataLODManager

    sp._qt()
    manager = DataLODManager()
    manager.set_map_distance_km(KM)
    manager.set_map_seed(SEED)
    parameter = sp._parameter()
    generatoren = sp._generatoren(manager, None)
    manager.set_calculator_target_lod("terrain.redistribution", LOD)
    terrain_gen = generatoren["terrain"]
    terrain_gen.set_active_parameters(parameter)

    terrain_gen._calc_noise("terrain.noise", LOD)
    terrain_gen._calc_redistribution("terrain.redistribution", LOD)
    terrain_gen._calc_slope("terrain.slope", LOD)
    terrain_gen._calc_shadow("terrain.shadow", LOD)
    terrain_data = terrain_gen.assemble_terrain_data(LOD, parameter)
    manager.set_terrain_data_complete_lod(terrain_data, LOD, parameter)
    return manager


def main():
    fehler = []
    from gui.utils.map_export import _flusslinien_aus_graph, vektordaten

    print("Pipeline bis river_graph faehrt (Seed %d, %dpx)" % (SEED, SIZE))
    manager = _pipeline_bis_river_graph()
    graph = manager.get_terrain_data("river_graph")
    if graph is None:
        print("FEHLER: river_graph ist None - Vorstufe (smoke_test_flussgraph_propagation.py) pruefen")
        return 1

    eltern = np.asarray(graph["eltern"])
    punkte = np.asarray(graph["punkte"])
    n = punkte.shape[0]
    print("   Graph hat %d Knoten" % n)

    # ---------- A. Zwei-Enden-Vergleich: Abdeckung ----------
    print("A. Abdeckung: Vereinigung aller Linienzug-Knoten gegen Graphgroesse")
    # NICHT gegen n vergleichen: `flussnetz()` liefert auch isolierte
    # Einzelpunkte (Quelle == Muendung am selben Knoten, kein eltern UND kein
    # Kind) - eigentlich kein Fluss, sondern eine Pfuetze ohne Ablauf. Ein
    # einzelner Punkt ergibt keinen Linienzug (>= 2 Punkte) und darf zu Recht
    # fehlen. Der ehrliche Vergleich ist gegen die NICHT-isolierten Knoten -
    # jeder davon haengt an mindestens einem Nachbarn und muss auf einem
    # Linienzug landen.
    kinder_anzahl = np.zeros(n, dtype=np.int64)
    for e in eltern:
        if e >= 0:
            kinder_anzahl[e] += 1
    nicht_isoliert = np.where((eltern >= 0) | (kinder_anzahl >= 1))[0]

    ketten = _flusslinien_aus_graph(graph)
    abgedeckt = set()
    for kette in ketten:
        abgedeckt.update(kette)
    ziel = set(nicht_isoliert.tolist())
    anteil = len(abgedeckt & ziel) / len(ziel) if ziel else 1.0
    print("   %d Linienzuege, %d von %d nicht-isolierten Knoten abgedeckt (%.1f%%), "
          "%d isolierte Einzelpunkte ohne Fluss ausgelassen" %
          (len(ketten), len(abgedeckt & ziel), len(ziel), anteil * 100, n - len(ziel)))
    if anteil < 0.999:
        fehlend = ziel - abgedeckt
        fehler.append("%d nicht-isolierte Knoten fehlen auf jedem Linienzug (z.B. %s) - "
                      "Astpunkt-Logik hat vermutlich eine Luecke"
                      % (len(fehlend), sorted(fehlend)[:5]))
    if not ketten:
        fehler.append("_flusslinien_aus_graph() liefert keine Linienzuege bei einem "
                      "Graphen mit %d Knoten" % n)

    # ---------- B. Zusammenhang ----------
    print("B. Zusammenhang: kein Sprung zwischen benachbarten Kettenpunkten")
    # Grosszuegige Grenze: der Graph liegt in Heightmap-Aufloesung (SIZE px
    # Kante), ein Einzelschritt zwischen Nachbarknoten sollte nie mehr als
    # ein Bruchteil der Kartenkante sein.
    grenze_px = SIZE * 0.25
    groesster_sprung = 0.0
    for kette in ketten:
        for a, b in zip(kette, kette[1:]):
            sprung = float(np.hypot(*(punkte[a] - punkte[b])))
            groesster_sprung = max(groesster_sprung, sprung)
            if sprung > grenze_px:
                fehler.append("Linienzug-Sprung %.1f px zwischen Knoten %d und %d "
                              "(Grenze %.1f px) - zwei Aeste vermutlich verklebt"
                              % (sprung, a, b, grenze_px))
    print("   groesster Einzelschritt: %.2f px (Grenze %.1f px)" % (groesster_sprung, grenze_px))

    # ---------- C. Strenges JSON ----------
    print("C. vektordaten() liefert die Fluesse, Export ist RFC-8259-JSON")
    vektor = vektordaten(manager, meter_pro_pixel=10.4)
    if not vektor["fluesse"]:
        fehler.append("vektordaten()['fluesse'] ist leer, obwohl river_graph %d Knoten hat" % n)
    else:
        for zug in vektor["fluesse"]:
            if len(zug) < 2:
                fehler.append("ein exportierter Flusszug hat weniger als 2 Punkte")
                continue
            for punkt in zug:
                for feld in ("x", "y", "strahler", "breite_m"):
                    if feld not in punkt:
                        fehler.append("Flusspunkt ohne Feld %r: %r" % (feld, punkt))
                if punkt.get("breite_m", 0) <= 0:
                    fehler.append("breite_m <= 0: %r" % (punkt,))
        try:
            text = json.dumps(vektor, allow_nan=False)
            json.loads(text)
        except ValueError as e:
            fehler.append("vektordaten()-Ausgabe ist kein striktes JSON: %s" % e)
        else:
            print("   %d Flusszuege, %d Bytes striktes JSON" % (len(vektor["fluesse"]), len(text)))

    print()
    if fehler:
        print("FEHLGESCHLAGEN (%d):" % len(fehler))
        for f in fehler:
            print("  - " + f)
        return 1

    print("Fluesse als Linienzuege: Abdeckung, Zusammenhang und striktes JSON bestanden (Ticket #37).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
