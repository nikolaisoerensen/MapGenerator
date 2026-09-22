"""
Path: tests/smoke_test_terrain_erosion_filter.py

Prueft die Einbindung des ATEF-Erosionsfilters in den Terrain-Aufbau
(BaseTerrainGenerator._apply_erosion_filter, SPEZIFIKATION §9).

Fuenf Zusicherungen. Die drei letzten sind Gegenproben - nach §5.1.4 prueft
eine Zusicherung, die auch ohne die Aenderung haelt, nichts.

  1. Der Filter LAEUFT ueberhaupt. §4.2 ist der teuerste Fehlertyp dieses
     Projekts (dreimal an einem Tag), deshalb wird nicht das Ergebnis
     interpretiert, sondern belegt, dass der Zweig betreten wurde: die
     ridge_map existiert nur, wenn er lief, und ihr Wertebereich ist ohne den
     Filter nicht herstellbar.
  2. Die Hoehenspanne ist exakt BASE_ELEVATION_M .. AMPLITUDE. §3.1 fuehrt sie
     als erfuellt, und ein Delta obendrauf reisst sie - _calc_redistribution
     bildet deshalb nach dem Filter zurueck.
  3. GEGENPROBE Reglerwirkung: erosion_filter_strength = 0 muss dieselbe
     Heightmap liefern wie der abgeschaltete Filter. Sonst tut der Regler
     nicht, was sein Name sagt (§4.7).
  4. GEGENPROBE Wirksamkeit: mit Vorgabewerten muss sich die Heightmap
     MESSBAR von der ungefilterten unterscheiden. Ohne diesen Teil wuerden
     die Punkte 1-3 auch bei einem Filter gruen sein, der nichts tut.
  5. GEGENPROBE Regler einzeln: jeder der sieben Regler muss eine sichtbare
     Wirkung haben. Drei tote Water-Slider gab es hier monatelang (§4.7).

Aufruf:
    .venv\\Scripts\\python.exe tests/smoke_test_terrain_erosion_filter.py
"""

import sys

import numpy as np

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

# DER ALTE PFAD WIRD HIER GEPRUEFT, nicht die Weltkarte.
#
# Seit dem 2026-08-05 gibt es WELTKARTE_AKTIV. Steht er, liefert
# _calc_redistribution das Regionengelaende statt Noise -> Potenz ->
# Erosionsfilter -> Flussnetz - und dieser Test misst dann etwas voellig
# anderes als das, was in seinem Namen steht. Beim ersten Lauf danach meldete
# er prompt "Messgeraet unbrauchbar" und "flache Karte ergibt Spanne 1546".
#
# Deshalb wird der Schalter hier hart ausgeschaltet. Er gehoert in denselben
# Rang wie FLUSSNETZ_AKTIV: ein Hauptschalter, den ein Test ueber den von ihm
# geprueften Pfad selbst setzen muss.
def _alten_pfad_erzwingen():
    import gui.config.value_default as vd
    vd.WELTKARTE_AKTIV = False


_alten_pfad_erzwingen()

SIZE = 128
LOD = 3          # 32 -> 64 -> 128


def _heightmap(parameter_overrides=None, filter_aktiv=True):
    """Faehrt terrain.noise + terrain.redistribution und liefert die Outputs."""
    import gui.config.value_default as vd
    from managers.data_lod_manager import DataLODManager
    from gui.config.value_default import TERRAIN, EROSION_FILTER
    from core.terrain_generator import BaseTerrainGenerator

    original = vd.EROSION_FILTER_AKTIV
    vd.EROSION_FILTER_AKTIV = filter_aktiv
    try:
        manager = DataLODManager()
        manager.set_map_distance_km(TERRAIN.MAP_DISTANCE_KM["default"])

        parameters = {key.lower(): getattr(TERRAIN, key)["default"] for key in
                      ("AMPLITUDE", "OCTAVES", "FEATURE_SIZE_M", "PERSISTENCE",
                       "LACUNARITY", "REDISTRIBUTE_POWER", "MAP_SEED")}
        parameters["map_size"] = SIZE
        parameters["map_seed"] = 20260730
        # Die Regler so, wie der Tab sie liefert (Praefix erosion_filter_).
        # Ticket #61: Attributname und Parameterschluessel sind seit der
        # Umbenennung NICHT mehr durch .lower() ineinander umrechenbar (z.B.
        # GULLY_REACH -> "erosion_filter_detail", der ATEF-Quellenname bleibt
        # als Schluessel bestehen). Deshalb hier explizit statt abgeleitet.
        for attr_name, schluessel_suffix in (
                ("STRENGTH", "strength"),
                ("GULLY_SIZE_M", "gully_size_m"),
                ("GULLY_REACH", "detail"),
                ("GULLY_VS_SHARPNESS", "gully_weight"),
                ("RIDGE_ROUNDING", "ridge_rounding"),
                ("VALLEY_ROUNDING", "crease_rounding"),
                ("OCTAVES", "octaves")):
            parameters["erosion_filter_" + schluessel_suffix] = \
                getattr(EROSION_FILTER, attr_name)["default"]
        parameters.update(parameter_overrides or {})

        generator = BaseTerrainGenerator(data_lod_manager=manager)
        generator.set_active_parameters(parameters)
        for node in ("terrain.noise", "terrain.redistribution"):
            manager.set_calculator_target_lod(node, LOD)
        generator._calc_noise("terrain.noise", LOD)
        generator._calc_redistribution("terrain.redistribution", LOD)

        heightmap = manager.get_calculator_output(
            "terrain.redistribution", "heightmap", LOD)
        ridge_map = manager.get_calculator_output(
            "terrain.redistribution", "ridge_map", LOD)
        assert heightmap.shape == (SIZE, SIZE), (
            "Werkzeug liefert %s statt %dx%d - §5.2" % (heightmap.shape, SIZE, SIZE))
        return heightmap.astype(np.float64), ridge_map
    finally:
        vd.EROSION_FILTER_AKTIV = original


def lauf():
    from gui.config.value_default import TERRAIN, EROSION_FILTER

    fehler = []

    # ---------- 1: laeuft der Filter? ----------
    mit, ridge = _heightmap()
    if ridge is None:
        fehler.append("ridge_map fehlt - der Filter lief nicht")
        print("1. Filter lief (ridge_map vorhanden) ........... FEHLER")
    else:
        spanne_ok = -1.6 < float(ridge.min()) and float(ridge.max()) < 1.6
        if not spanne_ok:
            fehler.append("ridge_map ausserhalb -1..1 (%.2f..%.2f)"
                          % (ridge.min(), ridge.max()))
        print("1. Filter lief, ridge_map %.3f .. %.3f ......... %s"
              % (ridge.min(), ridge.max(), "ok" if spanne_ok else "FEHLER"))

    # ---------- 2: Hoehenspanne ----------
    soll_tief = TERRAIN.BASE_ELEVATION_M
    soll_hoch = TERRAIN.AMPLITUDE["default"]
    ist_tief, ist_hoch = float(mit.min()), float(mit.max())
    # float32-Speicherung, deshalb keine Bitgleichheit verlangen.
    spanne_ok = abs(ist_tief - soll_tief) < 0.01 and abs(ist_hoch - soll_hoch) < 0.01
    if not spanne_ok:
        fehler.append("Hoehenspanne %.2f..%.2f statt %.2f..%.2f"
                      % (ist_tief, ist_hoch, soll_tief, soll_hoch))
    print("2. Spanne %.1f .. %.1f m (soll %.1f .. %.1f) ..... %s"
          % (ist_tief, ist_hoch, soll_tief, soll_hoch,
             "ok" if spanne_ok else "FEHLER"))

    # ---------- 3: Staerke 0 == Filter aus ----------
    ohne, _ = _heightmap(filter_aktiv=False)
    null, _ = _heightmap({"erosion_filter_strength": 0.0})
    rest = float(np.abs(null - ohne).max())
    if rest > 0.01:
        fehler.append("Staerke 0 weicht um %.4g m vom abgeschalteten Filter ab"
                      % rest)
    print("3. Staerke 0 == Filter aus (%.4g m Abweichung) .. %s"
          % (rest, "ok" if rest <= 0.01 else "FEHLER"))

    # ---------- 4: wirkt er ueberhaupt? ----------
    wirkung = float(np.abs(mit - ohne).max())
    if wirkung <= 1.0:
        fehler.append("Filter aendert die Heightmap um nur %.4g m - die Punkte "
                      "1-3 belegen dann nichts" % wirkung)
    print("4. Vorgabewerte wirken (%.0f m Unterschied) ..... %s"
          % (wirkung, "ok" if wirkung > 1.0 else "FEHLER"))

    # ---------- 5: jeder Regler einzeln ----------
    print("5. Wirkung jedes einzelnen Reglers:")
    proben = (
        ("erosion_filter_strength", EROSION_FILTER.STRENGTH["max"]),
        ("erosion_filter_gully_size_m", EROSION_FILTER.GULLY_SIZE_M["max"]),
        ("erosion_filter_detail", EROSION_FILTER.GULLY_REACH["min"]),
        ("erosion_filter_gully_weight", EROSION_FILTER.GULLY_VS_SHARPNESS["max"]),
        ("erosion_filter_ridge_rounding", EROSION_FILTER.RIDGE_ROUNDING["max"]),
        ("erosion_filter_crease_rounding", EROSION_FILTER.VALLEY_ROUNDING["max"]),
        ("erosion_filter_octaves", EROSION_FILTER.OCTAVES["min"]),
    )
    for name, wert in proben:
        variante, _ = _heightmap({name: wert})
        unterschied = float(np.abs(variante - mit).max())
        ok = unterschied > 1.0
        if not ok:
            fehler.append("Regler %s = %s aendert nichts (%.4g m)"
                          % (name, wert, unterschied))
        print("   %-34s -> %8.1f m  %s"
              % (name.replace("erosion_filter_", "") + " = " + str(wert),
                 unterschied, "ok" if ok else "TOT"))

    # ---------- 6: flache Karte ----------
    # Ist amplitude gleich BASE_ELEVATION_M, ist die Zielspanne der
    # Redistribution 100..100 und die Karte voellig flach. Das ist kein
    # konstruierter Fall: test_terrain_generator() im Modul selbst stellt genau
    # das ein. filter_heightmap() nimmt dafuer einen frueheren Rueckweg, dem
    # zunaechst ein Schluessel fehlte - der KeyError schickte die ganze
    # Terrain-Generierung in die Fehlerbehandlung, sichtbar nur an
    # validity_state == "error".
    flach, flach_ridge = _heightmap({"amplitude": TERRAIN.BASE_ELEVATION_M})
    flach_ok = (float(flach.max() - flach.min()) < 0.01
                and np.all(np.isfinite(flach)))
    if not flach_ok:
        fehler.append("flache Karte (amplitude == BASE_ELEVATION_M) ergibt "
                      "Spanne %.4g statt 0" % float(flach.max() - flach.min()))
    print("6. Flache Karte ohne Absturz .................... %s"
          % ("ok" if flach_ok else "FEHLER"))

    # ---------- 7: keine Inversion ueber den ganzen Amplitude-Bereich ----------
    # AMPLITUDE ist die GIPFELHOEHE. Solange BASE_ELEVATION_M bei 100 m lag,
    # wurde (amplitude - base_elevation) fuer jedes amplitude < 100 NEGATIV und
    # die Landschaft kippte um - Gipfel wurden Taeler, gemessen r = -0.998.
    #
    # Geprueft wird in BEIDEN Schalterstellungen des Filters. Mit
    # eingeschaltetem Filter bildet _apply_erosion_filter die Spanne ein zweites
    # Mal ab, zwei Inversionen heben sich auf, und der Fehler war unsichtbar -
    # eine Pruefung nur in der Vorgabestellung haette ihn nie gefunden.
    print("7. Keine Inversion ueber den Amplitude-Bereich:")
    import gui.config.value_default as vd
    for filter_an in (True, False):
        referenz, _ = _heightmap(filter_aktiv=filter_an)
        for amplitude in (TERRAIN.AMPLITUDE["min"], 100.0,
                          TERRAIN.AMPLITUDE["max"]):
            probe, _ = _heightmap({"amplitude": float(amplitude)},
                                  filter_aktiv=filter_an)
            r = float(np.corrcoef(referenz.ravel(), probe.ravel())[0, 1])
            ok = r > 0.5
            if not ok:
                fehler.append("amplitude %.0f mit Filter %s: Korrelation %+.3f "
                              "- das Gelaende ist umgedreht"
                              % (amplitude, filter_an, r))
            print("   Filter %-5s amplitude %6.0f -> r = %+.3f  %s"
                  % (filter_an, amplitude, r, "ok" if ok else "INVERTIERT"))

    print()
    if fehler:
        for f in fehler:
            print("  FEHLER: %s" % f)
        return 1
    print("Alle sieben Zusicherungen erfuellt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(lauf())
