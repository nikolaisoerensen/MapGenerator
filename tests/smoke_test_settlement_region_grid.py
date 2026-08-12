"""
3x3-Ausschnittsgitter (docs/OFFENE_PUNKTE.md 5.9/5.14/6.2)
und die neue PRO-REGION-Zielzahl fuer Roadsites/Landmarks.

ANLASS (2026-08-10). Nutzer-Vorgabe: "eine box um die insel und teilen das in
maps auf ... die siedlungen und landmarks und road sites sollten dabei etwas
von der grenze entfernt sein ... kannst du machen das road sites und
landmarks erscheinen? also pro region ein paar, so 1-4 jeweils."

Vorher: `calculate_roadsites`/`calculate_landmarks` kannten nur eine einzige,
globale Zielzahl fuer die ganze Karte (bei Standard-Reglerstellung 3) - auf
neun Regionen verteilt blieben am Ende oft nur 3-6 Objekte auf der GESAMTEN
Weltkarte uebrig (siehe docs/OFFENE_PUNKTE.md 5.12, Roadsite-Zahl bei 512 px).

Drei Zusicherungen:
1. Das Gitter selbst ist konsistent (4 Linien, gleicher Abstand, symmetrisch
   um die Kartenmitte).
2. Jede Region mit Landflaeche/Wegen bekommt bei Standard-Reglerstellung
   zwischen 0 und 4 Roadsites bzw. Landmarks (nie mehr - das ist die
   mathematische Grenze aus randint(1,4)*Multiplikator(1.0)*Skala(1.0)).
3. Die Gesamtzahl ueber alle neun Regionen liegt deutlich ueber dem alten
   globalen Ziel (3) - das eigentliche Nutzerziel ("erscheinen").
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pruefe_gitter(fehler):
    import core.terrain_weltkarte as rw

    for size in (128, 256, 512):
        linien = rw.gitterlinien_px(size)
        if len(linien) != 4:
            fehler.append("gitterlinien_px(%d): %d statt 4 Linien" % (size, len(linien)))
            continue
        linien_sortiert = sorted(linien)
        mitte = 0.5 * size
        abstaende = [linien_sortiert[i + 1] - linien_sortiert[i] for i in range(3)]
        if max(abstaende) - min(abstaende) > 1e-6:
            fehler.append("gitterlinien_px(%d): ungleicher Abstand %s" % (size, abstaende))
        schwerpunkt = sum(linien_sortiert) / 4.0
        if abs(schwerpunkt - mitte) > 1e-6:
            fehler.append("gitterlinien_px(%d): nicht symmetrisch um die Mitte "
                          "(Schwerpunkt %.3f statt %.3f)" % (size, schwerpunkt, mitte))

        # Mittlere Region (1,1) ohne Rand muss genau eine Kantenlaenge breit
        # sein, zentriert auf die Kartenmitte.
        x0, x1, y0, y1 = rw.regionsbox_px(1, 1, size, rand_anteil=0.0)
        kante = rw.gitter_kante_px(size)
        if abs((x1 - x0) - kante) > 1e-6 or abs((y1 - y0) - kante) > 1e-6:
            fehler.append("regionsbox_px(1,1,%d): Breite/Hoehe stimmt nicht "
                          "mit gitter_kante_px ueberein" % size)
        # 25% Rand je Seite -> 1.5x Kantenlaenge.
        x0r, x1r, y0r, y1r = rw.regionsbox_px(1, 1, size, rand_anteil=0.25)
        if abs((x1r - x0r) - 1.5 * kante) > 1e-6:
            fehler.append("regionsbox_px(1,1,%d,rand=0.25): Breite %.2f statt %.2f"
                          % (size, x1r - x0r, 1.5 * kante))


def main():
    fehler = []
    pruefe_gitter(fehler)

    from PyQt6.QtGui import QGuiApplication
    _app = QGuiApplication.instance() or QGuiApplication([])

    import core.terrain_weltkarte as rw
    import smoke_test_pipeline_outputs as sp
    from managers.calculator_graph import CALCULATOR_GRAPH
    from managers.data_lod_manager import DataLODManager

    SIZE, LOD, SEED = 256, 6, 20260804

    manager = DataLODManager()
    manager.set_map_seed(SEED)
    manager.set_map_distance_km(rw.WELT_KM)
    par = dict(sp._parameter())
    par["map_size"] = SIZE
    par["map_seed"] = SEED
    par["map_distance_km"] = rw.WELT_KM
    gen = sp._generatoren(manager, None)
    for k in CALCULATOR_GRAPH:
        manager.set_calculator_target_lod(k, LOD)
    for g in gen.values():
        if hasattr(g, "set_active_parameters"):
            g.set_active_parameters(par)

    for knoten in sp._reihenfolge():
        spec = CALCULATOR_GRAPH[knoten]
        erzeuger = gen.get(spec.generator)
        methode = getattr(erzeuger, "_calc_" + knoten.split(".", 1)[1], None)
        if methode is None:
            continue
        methode(knoten, LOD)
        if knoten == "settlement.landmarks":
            break

    region_map = manager.get_calculator_output("terrain.redistribution", "region_map", LOD)
    roadsites = manager.get_calculator_output("settlement.roadsites", "roadsite_list", LOD)
    landmarks = manager.get_calculator_output("settlement.landmarks", "landmark_list", LOD)
    if region_map is None or roadsites is None or landmarks is None:
        fehler.append("region_map/roadsite_list/landmark_list nicht verfuegbar")
        print("NICHT IN ORDNUNG:", fehler)
        return 1

    region_map = np.asarray(region_map)

    def _region_von(punkt):
        x, y = int(round(punkt.x)), int(round(punkt.y))
        x = int(np.clip(x, 0, region_map.shape[1] - 1))
        y = int(np.clip(y, 0, region_map.shape[0] - 1))
        return int(region_map[y, x])

    print("%-22s %10s %10s" % ("Region", "Roadsites", "Landmarks"))
    print("-" * 46)
    for i, (_z, _s, r) in enumerate(rw.alle_regionen()):
        n_road = sum(1 for p in roadsites if _region_von(p) == i)
        n_land = sum(1 for p in landmarks if _region_von(p) == i)
        print("%-22s %10d %10d" % (r["name"], n_road, n_land))
        if n_road > 4:
            fehler.append("%s: %d Roadsites - mehr als die erlaubten 4" % (r["name"], n_road))
        if n_land > 4:
            fehler.append("%s: %d Landmarks - mehr als die erlaubten 4" % (r["name"], n_land))

    print("")
    print("Gesamt: %d Roadsites, %d Landmarks (altes globales Ziel bei "
          "Standardreglern: 3)" % (len(roadsites), len(landmarks)))
    if len(roadsites) <= 4:
        fehler.append("Nur %d Roadsites insgesamt - kaum mehr als das alte "
                      "globale Ziel von 3, PRO-REGION-Verteilung wirkt nicht"
                      % len(roadsites))
    # KEINE Mindestzahl-Zusicherung mehr fuer Landmarks (2026-08-11): seit
    # der Landfilter-Korrektur in calculate_landmarks() (Nutzer-Befund am
    # laufenden Programm: "warum sind landmarks im meer" - "gipfel"/
    # "abgelegen" nahmen vorher jedes Meerespixel mit civ_map~0 an) ist der
    # echte Kandidatenpool kleiner als der vorher faelschlich ozean-
    # aufgeblaehte - eine niedrige, aber LAND-KORREKTE Zahl ist jetzt
    # erwartetes Verhalten, keine Regression. Die harte Zusicherung bleibt
    # stattdessen: KEIN Landmark im Wasser.
    heightmap = manager.get_calculator_output("terrain.redistribution", "heightmap", LOD)
    if heightmap is not None:
        im_wasser = [lm.location_id for lm in landmarks
                    if heightmap[int(round(lm.y)), int(round(lm.x))] <= 0.0]
        if im_wasser:
            fehler.append("%d Landmarks liegen im Wasser: IDs %s" % (len(im_wasser), im_wasser))

    print("")
    if fehler:
        print("NICHT IN ORDNUNG - %d Befunde:" % len(fehler))
        for f in fehler:
            print("   " + f)
        return 1
    print("Alle Zusicherungen erfuellt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
