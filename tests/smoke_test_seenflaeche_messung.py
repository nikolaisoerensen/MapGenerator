"""
Path: tests/smoke_test_seenflaeche_messung.py

Ticket #32 "Seenflaeche neu messen": eine reine Messung, keine Korrektur.
`docs/TESTBERICHT.md` meldete zuletzt eine Binnenseeflaeche von 0,0 % gegen
ein Ziel groesser null. Seither kamen Seegliederung, Seegrad-Voronoi und
Seewege dazu (siehe docs/project_see_voronoi_seegliederung.md) sowie die
Neun-Regionen-Weltkarte mit regionsspezifischen Kuesten-Archetypen
(core/terrain_weltkarte.py:weltfeld()). Dieses Skript erhebt den Ist-Wert
neu, auf echten Kartengroessen (256/512/1024 px, NICHT 129/257/513 - das war
schon einmal die Ursache dafuer, dass zehn gruene Tests eine tote Funktion
verdeckt haben, siehe CLAUDE.md).

WICHTIGE METHODIK-KLARSTELLUNG (fand zwei falsche Faehrten, bevor diese
Fassung stand):

  1. Das Feld `seegrad` (Ausgabe von `terrain.redistribution`, gebaut in
     core/terrain_weltkarte.py um Zeile 1690: `seegrad_roh = np.where(maske,
     0, grad[etikett])`, mit `maske` = Landmaske) ist eine
     Abstand-von-Land-Ringstufe ueber ALLES Wasser, Ozean eingeschlossen -
     nicht Binnenseen-spezifisch. `(seegrad > 0) & (heightmap > 0)` ist
     leer per Konstruktion, unabhaengig vom Kartenzustand: liefert also IMMER
     0,0 %, ganz gleich was das Programm tatsaechlich erzeugt.
  2. Die Kontinent-Silhouette vor der Kuestenverformung (`kontinentform()`)
     gegen die fertige Hoehenkarte zu halten (`maske & (heightmap <= 0)`)
     zaehlt auch jede Stelle mit, an der die Kueste seither zurueckgewichen
     ist (Buchten, Fjorde) - keine Binnenseen. Ergab unplausible 30-48 %
     "Seeflaeche" in mehreren Regionen.

  Die tatsaechlich fuer Binnenseen zustaendige Instanz ist der eigene
  Calculator-Knoten `water.lake_detection` (`core/water_generator.py`,
  `LakeDetectionSystem.detect_lakes()`): Prioritaets-Flutung/Wasserscheide
  auf der Hoehenkarte, Becken bis zum Ueberlaufpunkt gefuellt, nach
  Volumen gefiltert. Sein `lake_map` (-1 = kein See, >=0 = See-ID) ist per
  Konstruktion auf abflusslose Senken im Relief beschraenkt - das ist die
  Definition von "Binnensee", die auch `water._classify_water_bodies()`
  (`is_lake = lake_map >= 0`) im echten Betrieb verwendet.

Gemessen wird deshalb: Seepixel = `lake_map >= 0`, Landpixel = `heightmap >
0` (dieselbe Konvention wie ueberall sonst im Projekt), je Region aus
`region_map` (Ausgabe von `terrain.redistribution`, gefuellt weil
WELTKARTE_AKTIV=True immer die Neun-Regionen-Weltkarte erzeugt).

Gerechnet wird ueber die ECHTE Pipeline (`tools/weather_lab.py:
run_pipeline()`, echter CalculatorDispatcher, keine nachgebaute Reihenfolge)
- nur so lauft auch `water.lake_detection` mit, das von
`terrain.redistribution` abhaengt.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_seenflaeche_messung.py
"""

import sys
import time

import numpy as np

_WURZEL = r"C:\Lokale Dateien\Projects\Python\MapGenerator"
sys.path.insert(0, _WURZEL)

# Die Qt-Anwendung MUSS modulweit gehalten werden - als lokale Variable raeumt
# Python sie ab, waehrend der GL-Kontext noch lebt (Segfault ohne Meldung).
_QT_APP = None


def _qt():
    global _QT_APP
    from PyQt6.QtGui import QGuiApplication
    if _QT_APP is None:
        _QT_APP = QGuiApplication.instance() or QGuiApplication([])
    return _QT_APP


# Dieselben drei Seeds wie in der vorherigen (unverschmolzenen) Messung
# dieses Tickets auf Branch nacht/2026-09-17 (Commit 333d026), damit beide
# Messungen vergleichbar sind.
SEEDS = (424242, 13, 20260917)
GROESSEN = (256, 512, 1024)


def lod_fuer(size, map_size_min=32):
    """Passendes LOD fuer eine Zielaufloesung - calculate_lod_size() in
    managers/data_lod_manager.py ist size = map_size_min * 2**(lod-1), NICHT
    map_size direkt. Ohne diese Umrechnung liefert run_pipeline(map_size=256)
    beim Default-LOD still ein 128x128-Array (gefunden waehrend dieser
    Messung selbst - siehe Nachtrag in docs/TESTBERICHT.md)."""
    lod = 1
    s = map_size_min
    while s < size:
        s *= 2
        lod += 1
    return lod


def messe(size, seed):
    import tools.weather_lab as wl
    import core.terrain_weltkarte as rw

    wl.LAB_SEED = seed
    lod_req = lod_fuer(size)
    t0 = time.time()
    manager, lod = wl.run_pipeline(map_size=size, map_distance_km=rw.WELT_KM,
                                    lod=lod_req)
    dauer = time.time() - t0

    heightmap = manager.get_calculator_output(
        "terrain.redistribution", "heightmap", lod)
    region_map = manager.get_calculator_output(
        "terrain.redistribution", "region_map", lod)
    lake_map = manager.get_calculator_output(
        "water.lake_detection", "lake_map", lod)

    assert heightmap.shape == (size, size), (
        "heightmap hat die falsche Aufloesung: %r statt (%d, %d) - "
        "LOD-Umrechnung kaputt?" % (heightmap.shape, size, size))
    assert lake_map.shape == heightmap.shape, "lake_map/heightmap Formmismatch"
    assert region_map is not None, "region_map fehlt (WELTKARTE_AKTIV aus?)"

    land = heightmap > 0
    see = lake_map >= 0

    regionsnamen = {i: r["name"] for i, (_z, _s, r) in enumerate(rw.alle_regionen())}

    mpp = rw.WELT_KM * 1000.0 / size
    pixelflaeche_km2 = (mpp / 1000.0) ** 2

    from scipy import ndimage
    ergebnis = {
        "size": size, "seed": seed, "dauer": dauer,
        "land_gesamt": int(land.sum()), "see_gesamt": int(see.sum()),
        "regionen": {},
    }

    lake_ids_seen = np.unique(lake_map)
    lake_ids_seen = lake_ids_seen[lake_ids_seen >= 0]
    if lake_ids_seen.size:
        groessen_px = ndimage.sum(np.ones_like(lake_map, dtype=np.float64),
                                   lake_map, index=lake_ids_seen)
        groessen_km2 = groessen_px * pixelflaeche_km2
    else:
        groessen_km2 = np.array([])
    ergebnis["seen_anzahl_global"] = int(lake_ids_seen.size)
    ergebnis["see_groesse_max_km2"] = float(groessen_km2.max()) if groessen_km2.size else 0.0
    ergebnis["see_groesse_median_km2"] = float(np.median(groessen_km2)) if groessen_km2.size else 0.0

    for i in sorted(regionsnamen):
        name = regionsnamen[i]
        reg_maske = (region_map == i)
        reg_land = land & reg_maske
        reg_see = see & reg_maske
        n_land = int(reg_land.sum())
        n_see = int(reg_see.sum())
        anzahl_seen = 0
        if n_see:
            anzahl_seen = int(np.unique(lake_map[reg_see]).size)
        anteil = (100.0 * n_see / n_land) if n_land else float("nan")
        ergebnis["regionen"][name] = {
            "land_px": n_land, "see_px": n_see,
            "anteil_pct": anteil, "anzahl_seen": anzahl_seen,
        }

    return ergebnis


def main():
    _qt()
    alle = []
    t_start = time.time()
    for size in GROESSEN:
        for seed in SEEDS:
            print("messe size=%d seed=%d ..." % (size, seed))
            r = messe(size, seed)
            alle.append(r)
            print("  dauer=%.1fs land=%d see=%d global_seen=%d "
                  "groesster=%.4f km2 median=%.4f km2"
                  % (r["dauer"], r["land_gesamt"], r["see_gesamt"],
                     r["seen_anzahl_global"], r["see_groesse_max_km2"],
                     r["see_groesse_median_km2"]))

    print()
    print("=== Seenanteil (%% Landflaeche) je Region und Groesse, Mittel ueber %d Seeds ==="
          % len(SEEDS))
    regionsnamen_reihenfolge = list(alle[0]["regionen"].keys())
    header = "%-10s" % "Region"
    for size in GROESSEN:
        header += " | %8d px" % size
    header += " | %8s" % "Mittel"
    print(header)
    gesamt_je_groesse = {size: [] for size in GROESSEN}
    for name in regionsnamen_reihenfolge:
        zeile = "%-10s" % name
        werte_region = []
        for size in GROESSEN:
            werte = [r["regionen"][name]["anteil_pct"] for r in alle if r["size"] == size]
            mittel = float(np.nanmean(werte))
            zeile += " | %9.3f%%" % mittel
            werte_region.append(mittel)
            gesamt_je_groesse[size].append(mittel)
        zeile += " | %9.3f%%" % float(np.mean(werte_region))
        print(zeile)

    print()
    print("Mittel ueber alle Regionen je Groesse:")
    for size in GROESSEN:
        print("  %d px: %.3f%%" % (size, float(np.mean(gesamt_je_groesse[size]))))

    nullseen = [(r["size"], r["seed"], name)
                for r in alle for name, d in r["regionen"].items()
                if d["land_px"] > 0 and d["see_px"] == 0]
    print()
    print("Kombinationen ganz ohne See (von %d): %d" % (len(alle) * len(regionsnamen_reihenfolge), len(nullseen)))
    for size, seed, name in nullseen:
        print("  %d px, Seed %d: %s" % (size, seed, name))

    print()
    print("Gesamtdauer: %.1f s" % (time.time() - t_start))
    print()
    print("Dies ist eine Messung, keine Korrektur - siehe Ticket #32.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
