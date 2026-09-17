"""
Ticket #37: Flussnetz als Linienzuege exportieren - Vorstufe.

Prueft NUR die Weiterleitung des neuen Knotengraphen (`river_graph`) durch die
drei Speicherstufen, BEVOR irgendein Exportcode ihn liest:

    _calc_redistribution() -> set_calculator_output()
        -> assemble_terrain_data() -> TerrainData.river_graph
        -> set_terrain_data_complete_lod() -> get_terrain_data("river_graph")

`river_graph` ist ein dict (punkte/eltern/strahler/flaeche/meter_pro_pixel),
kein np.ndarray wie jede andere Terrain-Ausgabe. Genau das war die Gefahr:
set_terrain_data_lod() lehnt alles ab, was kein numpy-Array ist (WARNING statt
Fehler, siehe managers/data_lod_manager.py _validate_lod_input()) - deshalb
speichert set_terrain_data_complete_lod() river_graph jetzt an
set_terrain_data_lod() VORBEI, direkt im internen Store. Dieser Test faehrt
die echte Pipeline (WELTKARTE_AKTIV/WELTFLUESSE_AKTIV sind Standard = True,
siehe gui/config/value_default.py) und prueft, dass am Ende genau das
ankommt, was _weltfluesse() gebaut hat - keine der drei Stufen darf den Wert
stillschweigend verschlucken.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_flussgraph_propagation.py
"""
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


def main():
    fehler = []

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

    print("1. Pipeline bis terrain.shadow faehrt (Seed %d, %dpx)" % (SEED, SIZE))
    terrain_gen._calc_noise("terrain.noise", LOD)
    terrain_gen._calc_redistribution("terrain.redistribution", LOD)
    # assemble_terrain_data() verlangt auch slopemap/shadowmap (terrain.slope/
    # terrain.shadow) - ohne die bricht Stufe 2 mit einer eigenen, mit
    # river_graph nicht zusammenhaengenden ValueError ab.
    terrain_gen._calc_slope("terrain.slope", LOD)
    terrain_gen._calc_shadow("terrain.shadow", LOD)

    print("2. get_calculator_output('terrain.redistribution', 'river_graph', LOD) - Stufe 1")
    graph_calc = manager.get_calculator_output("terrain.redistribution", "river_graph", LOD)
    if graph_calc is None:
        fehler.append("Stufe 1 (set_calculator_output/get_calculator_output): river_graph ist None")
    elif not isinstance(graph_calc, dict):
        fehler.append("Stufe 1: river_graph ist kein dict, sondern %s" % type(graph_calc).__name__)
    else:
        print("   OK - Keys: %s" % sorted(graph_calc.keys()))

    print("3. assemble_terrain_data() - Stufe 2 (TerrainData.river_graph)")
    terrain_data = terrain_gen.assemble_terrain_data(LOD, parameter)
    graph_td = getattr(terrain_data, "river_graph", "FEHLENDES_ATTRIBUT")
    if graph_td == "FEHLENDES_ATTRIBUT":
        fehler.append("Stufe 2: TerrainData hat gar kein river_graph-Attribut")
    elif graph_td is None:
        fehler.append("Stufe 2 (assemble_terrain_data): TerrainData.river_graph ist None")
    elif not isinstance(graph_td, dict):
        fehler.append("Stufe 2: TerrainData.river_graph ist kein dict, sondern %s" % type(graph_td).__name__)
    else:
        print("   OK - Keys: %s" % sorted(graph_td.keys()))

    print("4. set_terrain_data_complete_lod() + get_terrain_data('river_graph') - Stufe 3")
    manager.set_terrain_data_complete_lod(terrain_data, LOD, parameter)
    graph_store = manager.get_terrain_data("river_graph")
    if graph_store is None:
        fehler.append("Stufe 3 (set_terrain_data_complete_lod/get_terrain_data): river_graph ist None")
    elif not isinstance(graph_store, dict):
        fehler.append("Stufe 3: river_graph ist kein dict, sondern %s" % type(graph_store).__name__)
    else:
        print("   OK - Keys: %s" % sorted(graph_store.keys()))

    print("5. Inhalt pruefen (punkte/eltern/strahler/flaeche/meter_pro_pixel)")
    if not fehler:
        erwartete_keys = {"punkte", "eltern", "strahler", "flaeche", "meter_pro_pixel"}
        fehlende_keys = erwartete_keys - set(graph_store.keys())
        if fehlende_keys:
            fehler.append("river_graph fehlen Keys: %s" % sorted(fehlende_keys))
        else:
            punkte = np.asarray(graph_store["punkte"])
            eltern = np.asarray(graph_store["eltern"])
            strahler = np.asarray(graph_store["strahler"])
            flaeche = np.asarray(graph_store["flaeche"])
            mpp = graph_store["meter_pro_pixel"]

            if punkte.ndim != 2 or punkte.shape[1] != 2:
                fehler.append("punkte hat falsche Form: %s (erwartet Nx2)" % (punkte.shape,))
            n = punkte.shape[0] if punkte.ndim == 2 else 0
            if n == 0:
                fehler.append("punkte ist leer - kein Flussnetz entstanden bei Seed %d" % SEED)
            for name, feld in (("eltern", eltern), ("strahler", strahler), ("flaeche", flaeche)):
                if feld.shape[0] != n:
                    fehler.append("%s hat Laenge %d, punkte hat %d" % (name, feld.shape[0], n))
            if not isinstance(mpp, (int, float, np.floating)) or mpp <= 0:
                fehler.append("meter_pro_pixel ist kein positiver Zahlenwert: %r" % (mpp,))
            else:
                print("   %d Knoten, meter_pro_pixel=%.3f, Strahler max=%s" %
                      (n, mpp, int(strahler.max()) if n else -1))

            # Gegenprobe: river_mask (das Raster) muss zur gleichen Seed/Groesse
            # ebenfalls echte Fluesse zeigen - sonst waere ein leerer Graph nur
            # deshalb "plausibel", weil bei diesem Seed gar kein Fluss entsteht.
            river_mask = manager.get_terrain_data("river_mask")
            if river_mask is not None and np.any(np.asarray(river_mask) > 0) and n == 0:
                fehler.append("river_mask zeigt Fluesse, river_graph aber keine Knoten - "
                               "Graph und Raster widersprechen sich")

    print()
    if fehler:
        print("FEHLGESCHLAGEN (%d):" % len(fehler))
        for f in fehler:
            print("  - " + f)
        return 1

    print("Alle drei Weiterleitungsstufen liefern river_graph unveraendert (Ticket #37, Vorstufe).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
