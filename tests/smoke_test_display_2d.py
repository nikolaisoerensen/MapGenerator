"""
Jede 2D-Darstellung einmal wirklich zeichnen.

ANLASS (2026-08-10). Der Nutzer: "Slope 2D ist neuerdings tot. im 2D geht im
geology nur Rock Outcrop und Cross section. und so weiter." Die Ursache im
Slope-Fall war ein Einzeiler:

    self.current_colorbar = None
    self.current_colorbar.set_label('Slope (°)')     # Aufruf auf None

Jede Slope-Anzeige warf damit einen AttributeError. Auffallen konnte das
niemandem: `TerrainTab.update_display_mode` faengt Fehler ab und schreibt sie
ins Log, die Flaeche blieb einfach leer. In 3D lief es weiter, weil der Weg
dort an dieser Methode vorbeigeht.

KEIN BESTEHENDER TEST KONNTE DAS FINDEN. `smoke_test_pipeline_outputs.py`
prueft, ob die DATEN entstehen - sie entstanden ja. Zwischen Daten und Bild lag
niemand.

Dieser Test schliesst die Luecke: er baut ein echtes MapDisplay2D (offscreen,
ohne Fenster), schickt jede Layer-Art einmal hindurch und prueft, dass
  - kein Fehler fliegt UND
  - danach wirklich etwas auf der Achse steht.
Der zweite Teil ist der wichtige. Ein `try/except`, das den Fehler schluckt,
laesst den ersten bestehen.
"""
import os
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

from gui.widgets.map_display_2d import MapDisplay2D
from core.geology_layers import N_LAYERS

N = 48


def _hoehe():
    y, x = np.mgrid[0:N, 0:N].astype(np.float64)
    return (300.0 * np.sin(x / 9.0) * np.cos(y / 11.0) - 40.0).astype(np.float32)


def _faelle():
    """(Beschriftung, layer_type, Daten) fuer jede Darstellung der Oberflaeche."""
    hoehe = _hoehe()
    rng = np.random.RandomState(4)
    dy, dx = np.gradient(hoehe.astype(np.float64))
    hang = np.dstack([dx, dy]).astype(np.float32)
    return [
        ("Terrain / Height", "heightmap", hoehe),
        ("Terrain / Combined", "heightmap_combined", hoehe),
        ("Terrain / Slope", "slopemap", hang),
        ("Terrain / Regionen", "region_map",
         {"regionen": (rng.randint(0, 9, (N, N))).astype(np.int16),
          "heightmap": hoehe}),
        ("Fluesse / Ordnung", "river_order",
         rng.randint(0, 5, (N, N)).astype(np.float32)),
        ("Geology / Rock Outcrop", "rock_map",
         rng.randint(0, 255, (N, N, 3)).astype(np.uint8)),
        ("Geology / Hardness", "hardness_map",
         (rng.rand(N, N) * 0.9 + 0.05).astype(np.float32)),
        ("Geology / Terrain Hub", "terrain_hub_delta",
         (hoehe * 0.1).astype(np.float32)),
        ("Geology / Tilt", "tilt_delta", (hoehe * 0.05).astype(np.float32)),
        ("Geology / Fold", "fold_delta", (hoehe * 0.03).astype(np.float32)),
        ("Geology / Fault", "fault_delta", (hoehe * 0.07).astype(np.float32)),
        ("Geology / Intrusion", "intrusion_delta",
         (hoehe * 0.02).astype(np.float32)),
        ("Weather / Temperature", "temp_map",
         (12.0 + hoehe / 100.0).astype(np.float32)),
        ("Weather / Precipitation", "precip_map",
         (rng.rand(N, N) * 80.0).astype(np.float32)),
        ("Weather / Humidity", "humid_map", (rng.rand(N, N)).astype(np.float32)),
        ("Weather / Wind", "wind_map",
         np.dstack([rng.rand(N, N) * 8.0 - 4.0,
                    rng.rand(N, N) * 8.0 - 4.0]).astype(np.float32)),
        ("Water / Water Depth", "water_map",
         (np.maximum(-hoehe, 0.0) * 0.02).astype(np.float32)),
        ("Water / Flow", "flow_map", (rng.rand(N, N) * 3.0).astype(np.float32)),
        ("Water / Soil Moisture", "soil_moist_map",
         (rng.rand(N, N)).astype(np.float32)),
        ("Water / Evaporation", "evaporation_map",
         (rng.rand(N, N) * 4.0).astype(np.float32)),
        ("Erosion / Erosion", "erosion_map",
         (rng.rand(N, N) * 2.0).astype(np.float32)),
        ("Erosion / Sedimentation", "sedimentation_map",
         (rng.rand(N, N) * 2.0).astype(np.float32)),
        ("Erosion / Net Change", "net_change_map",
         (rng.rand(N, N) * 4.0 - 2.0).astype(np.float32)),
        ("Erosion / Sediment Load", "sediment_load_map",
         (rng.rand(N, N)).astype(np.float32)),
        ("Erosion / Water Depth", "water_depth_map",
         (rng.rand(N, N)).astype(np.float32)),
        ("Erosion / Flow Velocity", "flow_velocity_map",
         (rng.rand(N, N) * 5.0).astype(np.float32)),
        ("Erosion / Thermal Erosion", "thermal_erosion_map",
         (rng.rand(N, N)).astype(np.float32)),
        ("Erosion / Thermal Deposition", "thermal_deposition_map",
         (rng.rand(N, N)).astype(np.float32)),
        ("Biome / Biome", "biome_map",
         rng.randint(0, 20, (N, N)).astype(np.float32)),
        # Der Querschnitt erwartet GENAU N_LAYERS Schichtgrenzen und eine
        # Position als Bruchteil [0,1] entlang der anderen Achse - nicht als
        # Pixelindex. Beides beim ersten Anlauf falsch geliefert, was einen
        # IndexError gab, der nicht dem Renderer anzulasten war.
        ("Geology / Cross-Section", "geology_cross_section",
         {"layer_boundaries": np.cumsum(
             rng.rand(N_LAYERS, N, N).astype(np.float32) * 40.0, axis=0),
          "terrain_height": hoehe, "axis": "x", "position": 0.5}),
    ]


def main():
    anzeige = MapDisplay2D()
    befunde = []

    print("%-30s %-24s %s" % ("Anzeige", "layer_type", "Ergebnis"))
    print("-" * 78)
    for beschriftung, art, daten in _faelle():
        anzeige.ax.clear()
        anzeige.current_colorbar = None
        fehler = None
        try:
            anzeige.update_display(daten, art)
        except Exception as f:                       # noqa: BLE001
            fehler = "%s: %s" % (type(f).__name__, f)

        gezeichnet = (len(anzeige.ax.images) + len(anzeige.ax.collections)
                      + len(anzeige.ax.lines) + len(anzeige.ax.patches))
        if fehler:
            zustand = "FEHLER  " + fehler
            befunde.append("%s (%s) wirft %s" % (beschriftung, art, fehler))
        elif gezeichnet == 0:
            zustand = "LEER    nichts auf der Achse"
            befunde.append("%s (%s) zeichnet nichts" % (beschriftung, art))
        else:
            zustand = "ok      %d Element(e)" % gezeichnet
        print("%-30s %-24s %s" % (beschriftung, art, zustand))

    print("")
    if befunde:
        print("NICHT IN ORDNUNG - %d Befunde:" % len(befunde))
        for b in befunde:
            print("   " + b)
        return 1
    print("Alle %d Darstellungen zeichnen etwas." % len(_faelle()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
