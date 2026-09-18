"""
Path: tests/smoke_test_flussexport_vektor.py

Prueft den Flussnetz-Export als Linienzuege end-to-end:
TerrainGenerator._weltfluesse() -> TerrainData.river_lines ->
DataLODManager -> gui.utils.map_export.vektordaten().

WORUM ES GEHT. Der Knotengraph des Flussnetzes (core/terrain_weltfluesse.py)
wurde bisher nur gerastert (river_mask/river_order) und danach verworfen -
der Vektorexport ("fluesse" in vektor.json) blieb deshalb immer leer. Aus
einem Rasterbild laesst sich eine scharfe Flusslinie nur noch erraten
("ein Fluss, der nur als Raster existiert, ist im Spiel eine Treppe").

Dieser Test faehrt die vier neu verdrahteten Stationen der Reihe nach und
prueft an jeder, dass die Daten wirklich ankommen:

  1. TerrainGenerator.calculate_heightmap() (WELTKARTE_AKTIV, echte
     Kartengroesse) liefert ein TerrainData-Objekt mit befuelltem
     `river_lines`.
  2. Jede Linie hat mindestens zwei Punkte und eine sinnvolle
     Strahler-Ordnung (>= 1).
  3. DataLODManager.set_terrain_data_complete_lod() legt `river_lines` ab,
     OHNE an der np.ndarray-Pruefung zu scheitern (es ist eine Liste von
     Dicts) - get_terrain_data("river_lines") liefert sie unveraendert
     zurueck.
  4. gui.utils.map_export.vektordaten() setzt "fluesse" mit echten
     Meterkoordinaten (nicht den rohen Pixelwerten aus 1./2.) und
     uebernimmt "ordnung" je Linie.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_flussexport_vektor.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reale Kartengroesse (gui/config/value_default.py: MAPSIZEMIN=32,
# MAPSIZEMAX=1024, step=32) - keine ausgedachte Groesse wie 129/257, siehe
# CLAUDE.md ("Gruene Tests koennen eine tote Funktion verdecken").
SIZE = 256
LOD = 3
SEED = 20260918

_QT_APP = None


def _qt():
    # Modulweit halten (nicht lokal), sonst raeumt Python die QGuiApplication
    # ab, waehrend QObject-Instanzen (DataLODManager-Signale) noch leben.
    global _QT_APP
    from PyQt6.QtGui import QGuiApplication
    if _QT_APP is None:
        _QT_APP = QGuiApplication.instance() or QGuiApplication([])
    return _QT_APP


def _parameter():
    """Ein Parametersatz aus allen Vorgaben von value_default (wie in
    tests/smoke_test_pipeline_outputs.py) - map_size/map_seed ueberschrieben,
    alle river_*-Regler bleiben auf ihrem Katalogwert."""
    import gui.config.value_default as vd

    parameter = {"map_size": SIZE, "map_seed": SEED}
    for klassenname in ("TERRAIN", "RIVER_NETWORK"):
        klasse = getattr(vd, klassenname, None)
        if klasse is None:
            continue
        praefix = "river_" if klassenname == "RIVER_NETWORK" else ""
        for name in dir(klasse):
            if not name.isupper():
                continue
            wert = getattr(klasse, name)
            if isinstance(wert, dict) and "default" in wert:
                parameter.setdefault(praefix + name.lower(), wert["default"])
    return parameter


def erzeuge_terrain_data():
    """Schritt 1+2: echte Kartengenerierung, WELTKARTE_AKTIV-Pfad."""
    from managers.data_lod_manager import DataLODManager
    from core.terrain_generator import BaseTerrainGenerator

    dlm = DataLODManager()
    generator = BaseTerrainGenerator(map_seed=SEED, shader_manager=None,
                                     data_lod_manager=dlm)
    terrain_data = generator.calculate_heightmap(_parameter(), LOD)
    return dlm, terrain_data


def pruefe_river_lines(terrain_data):
    fehler = []
    linien = getattr(terrain_data, "river_lines", None)
    if not linien:
        fehler.append("TerrainData.river_lines ist leer/None - "
                      "kein Flussnetz erzeugt oder Anschluss nicht verdrahtet")
        return fehler, []

    print(f"[OK] TerrainData.river_lines: {len(linien)} Linien")

    zu_kurz = [l for l in linien if len(l.get("punkte", [])) < 2]
    if zu_kurz:
        fehler.append(f"{len(zu_kurz)} Linien mit weniger als 2 Punkten")

    ordnungen = [l.get("ordnung") for l in linien]
    unplausibel = [o for o in ordnungen if not isinstance(o, int) or o < 1]
    if unplausibel:
        fehler.append(f"unplausible Strahler-Ordnung(en): {unplausibel[:5]}")
    else:
        print(f"[OK] Strahler-Ordnung liegt zwischen {min(ordnungen)} "
              f"und {max(ordnungen)}")

    return fehler, linien


def pruefe_data_lod_manager(dlm, terrain_data):
    """Schritt 3: set_terrain_data_complete_lod() darf an river_lines
    (eine Liste von Dicts, kein np.ndarray) nicht scheitern."""
    fehler = []
    try:
        dlm.set_terrain_data_complete_lod(terrain_data, LOD, _parameter())
    except Exception as e:                                    # noqa: BLE001
        fehler.append(f"set_terrain_data_complete_lod() ist abgestuerzt: {e}")
        return fehler

    zurueck = dlm.get_terrain_data("river_lines")
    if not zurueck:
        fehler.append("get_terrain_data('river_lines') liefert nichts zurueck "
                      "- Speicherung/Ruecklesen nicht verdrahtet")
    else:
        print(f"[OK] DataLODManager: get_terrain_data('river_lines') "
              f"liefert {len(zurueck)} Linien zurueck")
    return fehler


def pruefe_vektorexport(dlm, roh_linien):
    """Schritt 4: gui.utils.map_export.vektordaten() - Pixel -> Meter,
    genau wie bei 'wege'."""
    fehler = []
    from core.terrain_weltkarte import WELT_KM
    from gui.utils.map_export import vektordaten

    mpp = WELT_KM * 1000.0 / float(SIZE)
    vektor = vektordaten(dlm, mpp)

    fluesse = vektor.get("fluesse")
    if not fluesse:
        fehler.append(f"vektordaten()['fluesse'] ist leer - 'fehlt': "
                      f"{vektor.get('fehlt')}")
        return fehler

    print(f"[OK] vektordaten()['fluesse']: {len(fluesse)} Linien, "
          f"meter_pro_pixel={mpp:.2f}")

    # Groesster Roh-Pixelwert ueber alle Linien - die Meterwerte muessen
    # klar darueber liegen, sonst wurde nur roh durchgereicht statt
    # umgerechnet (mpp liegt bei SIZE=256/WELT_KM deutlich ueber 1).
    roh_max = max(max(abs(x), abs(y)) for l in roh_linien
                  for x, y in l["punkte"])

    zu_kurz = [l for l in fluesse if len(l.get("punkte", [])) < 2]
    if zu_kurz:
        fehler.append(f"{len(zu_kurz)} exportierte Linien mit weniger als "
                      f"2 Punkten")

    alle_koords = [k for l in fluesse for p in l["punkte"] for k in p]
    unrealistisch = [k for k in alle_koords if k < 0 or k > SIZE * mpp * 1.5]
    if unrealistisch:
        fehler.append(f"{len(unrealistisch)} Koordinaten ausserhalb der "
                      f"plausiblen Kartenausdehnung (0..{SIZE * mpp:.0f} m)")

    nicht_umgerechnet = [l for l in fluesse
                        if all(abs(x) <= roh_max and abs(y) <= roh_max
                               for x, y in l["punkte"])]
    if len(nicht_umgerechnet) == len(fluesse) and mpp > 1.5:
        fehler.append("keine exportierte Linie liegt ueber dem groessten "
                      "rohen Pixelwert - sieht nach unkonvertierten "
                      "Pixelkoordinaten statt Metern aus")
    else:
        print(f"[OK] Meterkoordinaten liegen ueber den rohen Pixelwerten "
              f"(roh max {roh_max:.1f}, mpp {mpp:.2f})")

    ordnungen_export = [l.get("ordnung") for l in fluesse]
    if any(not isinstance(o, int) or o < 1 for o in ordnungen_export):
        fehler.append("exportierte 'ordnung' ist nicht durchgaengig ein "
                      "plausibler positiver int")

    breiten = [l["breite_m"] for l in fluesse if "breite_m" in l]
    if breiten:
        print(f"[OK] {len(breiten)} von {len(fluesse)} Linien tragen "
              f"'breite_m' ({min(breiten):.1f}..{max(breiten):.1f} m)")

    return fehler


def lauf():
    _qt()
    print(f"Flussexport-Vektor-Test, {SIZE} px, LOD {LOD}, Seed {SEED}\n")

    alle_fehler = []
    dlm, terrain_data = erzeuge_terrain_data()

    f, roh_linien = pruefe_river_lines(terrain_data)
    alle_fehler.extend(f)
    if not roh_linien:
        print("\nNICHT IN ORDNUNG - kein Flussnetz, weitere Stufen "
              "uebersprungen")
        for e in alle_fehler:
            print(f"   {e}")
        return 1

    alle_fehler.extend(pruefe_data_lod_manager(dlm, terrain_data))
    alle_fehler.extend(pruefe_vektorexport(dlm, roh_linien))

    print()
    if alle_fehler:
        print(f"NICHT IN ORDNUNG - {len(alle_fehler)} Befunde:")
        for e in alle_fehler:
            print(f"   {e}")
        return 1
    print("Alle Stufen (Graph -> TerrainData -> DataLODManager -> "
          "vektordaten) liefern Fluesse als Linienzuege in Metern.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(lauf())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
