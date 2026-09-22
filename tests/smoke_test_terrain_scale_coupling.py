"""
Path: tests/smoke_test_terrain_scale_coupling.py

Verknuepft die drei Skalenebenen des Terrains und prueft, dass sie sich
gegenseitig nicht verstellen:

    map_size            Aufloesung in Pixeln
    map_distance_km     reale Ausdehnung der Karte
    Rinnengroesse       Groesse der Erosionsrinnen

Die Frage, um die es geht: bleibt eine Landschaft DIESELBE Landschaft, wenn man
nur die Aufloesung erhoeht oder nur den Ausschnitt vergroessert? Nach
SPEZIFIKATION §4.4 muss jede Groesse mit Einheit beantworten, gegen WAS sie
bemessen ist - und eine Rinne ist gegen die Wirklichkeit bemessen, nicht gegen
den Bildausschnitt. Zoomt man heraus, muessen mehr Rinnen ins Bild passen, nicht
groessere.

Messgroesse: die charakteristische Wellenlaenge des Erosions-Deltas IN METERN,
aus dem radial aufsummierten Leistungsspektrum. Das Delta ist auf die
Rinnen-Skalen bandbegrenzt und hat deshalb ein echtes Maximum - die Heightmap
selbst waere rot und haette keins.

Gemessen wird mit EINER Rinnen-Oktave. Mit den ueblichen fuenf ueberlagern sich
fuenf Rinnengroessen, und _max_safe_octaves laesst je nach Skala vier oder fuenf
davon zu - das Maximum springt dann auf eine andere Oktave und die Messung wird
unbrauchbar (Gegenprobe kam auf Faktor 1.38 statt 2.0). Eine Oktave isoliert
die Groesse, um die es geht.

Drei Laeufe:

  0. GEGENPROBE MESSGERAET. Wird die eingestellte Rinnengroesse verdoppelt,
     muss die gemessene Wellenlaenge sich verdoppeln. §5.2 fuehrt vier Faelle,
     in denen das Messgeraet selbst falsch war ("die Senkenmessung hatte Rand
     und Innenbereich vertauscht"). Ohne diesen Lauf sagen die anderen nichts.
  1. AUFLOESUNG. map_size 128/256/512 bei fester Ausdehnung. Die grobe Form und
     die Rinnengroesse in Metern muessen gleich bleiben; hoehere Aufloesung
     darf nur FEINERES hinzufuegen.
  2. AUSDEHNUNG. map_distance_km 10/20/40 bei fester Aufloesung. Die
     Rinnengroesse in Metern muss gleich bleiben.

     Warum nicht 5 km: dort passen bei 2250 m Rinnen nur 2.2 davon ueber die
     Karte, und bei einer Wellenzahl von 2 hat die FFT nichts zu messen - der
     erste Durchgang meldete dort 2500 m statt 1889 m, was eine Aussage ueber
     das Messgeraet war und keine ueber die Landschaft. Die drei gewaehlten
     Ausdehnungen liegen alle bei Wellenzahlen von 7 bis 29 und unterhalb der
     Drei-Pixel-Grenze aus _erosion_filter_parameters().

Aufruf:
    .venv\\Scripts\\python.exe tests/smoke_test_terrain_scale_coupling.py
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

SEED = 20260730


# =============================================================================
# MESSGERAET
# =============================================================================

def wellenlaenge_m(feld: np.ndarray, meter_pro_pixel: float) -> float:
    """
    Charakteristische Wellenlaenge eines Feldes in METERN.

    Radial aufsummiertes Leistungsspektrum; das Maximum ueber den Radius ist
    die vorherrschende Wellenzahl k in Zyklen ueber die Karte. Aufsummiert
    statt gemittelt, weil die Ringflaeche mit k waechst und die Summe damit die
    tatsaechlich in dieser Skala steckende Leistung ist.
    """
    n = feld.shape[0]
    spektrum = np.fft.fftshift(np.fft.fft2(feld - float(feld.mean())))
    leistung = np.abs(spektrum) ** 2

    y, x = np.mgrid[0:n, 0:n]
    radius = np.hypot(x - n // 2, y - n // 2).astype(int)
    je_ring = np.bincount(radius.ravel(), leistung.ravel())

    # k = 0 ist der Mittelwert, oberhalb n/2 liegt nur die Spiegelung.
    gueltig = je_ring[1:n // 2]
    if gueltig.size == 0 or not np.any(gueltig > 0):
        return float("nan")

    # SCHWERPUNKT um das Maximum statt des Maximums selbst. k ist ganzzahlig,
    # und bei den hier auftretenden k von 8-11 ist das eine Aufloesung von nur
    # ~10% - zu grob, um einen Faktor 2 sauber von einem Faktor 1.4 zu
    # unterscheiden. Gewichtet wird ueber alle Ringe mit mehr als der halben
    # Spitzenleistung, das ist die Breite der einen Oktave.
    k_achse = np.arange(1, n // 2, dtype=np.float64)
    fenster = gueltig >= 0.5 * gueltig.max()
    k = float(np.sum(k_achse[fenster] * gueltig[fenster])
              / np.sum(gueltig[fenster]))
    return (n * meter_pro_pixel) / k


# =============================================================================
# GELAENDE
# =============================================================================

def baue(map_size, map_distance_km, filter_overrides=None):
    """
    Faehrt terrain.noise + terrain.redistribution wie die echte Pipeline und
    liefert (gefiltert, roh, delta, meter_pro_pixel).

    `delta` kommt DIREKT aus filter_heightmap() und nicht als Differenz der
    fertigen Karten. Gemessen 2026-07-30: die Differenz der fertigen Karten ist
    unbrauchbar, weil beide von _calc_redistribution auf 100..4000 m
    zurueckgebildet werden. Diese Rueckbildung ist eine grossflaechige
    Verschiebung, die das Leistungsspektrum beherrscht - die Gegenprobe des
    Messgeraets kam auf Verhaeltnis 0.50 statt 2.0, und die gemessenen
    Wellenzahlen lagen bei 2-4 statt bei den 7-14 der Rinnen.
    """
    from managers.data_lod_manager import DataLODManager
    from gui.config.value_default import TERRAIN, EROSION_FILTER
    from core.terrain_generator import BaseTerrainGenerator

    manager = DataLODManager()
    manager.set_map_distance_km(float(map_distance_km))

    parameters = {key.lower(): getattr(TERRAIN, key)["default"] for key in
                  ("AMPLITUDE", "OCTAVES", "FEATURE_SIZE_M", "PERSISTENCE",
                   "LACUNARITY", "REDISTRIBUTE_POWER")}
    parameters["map_size"] = int(map_size)
    parameters["map_distance_km"] = float(map_distance_km)
    parameters["map_seed"] = SEED
    # Ticket #61: Attributname und Parameterschluessel sind seit der
    # Umbenennung NICHT mehr durch .lower() ineinander umrechenbar (z.B.
    # GULLY_REACH -> "erosion_filter_detail", der ATEF-Quellenname bleibt als
    # Schluessel bestehen). Deshalb hier explizit statt abgeleitet. "SCALE"
    # gab es auf EROSION_FILTER nie - das hasattr-Auslassen war historisch
    # bereits immer False und bleibt es.
    for attr_name, schluessel_suffix in (
            ("STRENGTH", "strength"),
            ("SCALE", "scale"),
            ("GULLY_REACH", "detail"),
            ("GULLY_VS_SHARPNESS", "gully_weight"),
            ("RIDGE_ROUNDING", "ridge_rounding"),
            ("VALLEY_ROUNDING", "crease_rounding"),
            ("OCTAVES", "octaves")):
        if hasattr(EROSION_FILTER, attr_name):
            parameters["erosion_filter_" + schluessel_suffix] = \
                getattr(EROSION_FILTER, attr_name)["default"]
    if hasattr(EROSION_FILTER, "GULLY_SIZE_M"):
        parameters["erosion_filter_gully_size_m"] = \
            EROSION_FILTER.GULLY_SIZE_M["default"]
    parameters.update(filter_overrides or {})

    lod = int(round(np.log2(max(int(map_size), 32) / 32.0))) + 1
    generator = BaseTerrainGenerator(data_lod_manager=manager)
    generator.set_active_parameters(parameters)
    for node in ("terrain.noise", "terrain.redistribution"):
        manager.set_calculator_target_lod(node, lod)
    generator._calc_noise("terrain.noise", lod)
    generator._calc_redistribution("terrain.redistribution", lod)

    heightmap = manager.get_calculator_output(
        "terrain.redistribution", "heightmap", lod)
    # §5.2: ein Werkzeug, das still etwas anderes liefert als angefragt, macht
    # jede damit gewonnene Zahl wertlos. build_terrain(512) gab einmal 256.
    assert heightmap.shape == (int(map_size), int(map_size)), (
        "angefragt %d px, bekommen %s" % (map_size, heightmap.shape))

    # Dasselbe Gelaende OHNE Filter - das ist der Eingang, auf den der Filter
    # wirkt, und die Grundlage fuer das isoliert gemessene Delta.
    import gui.config.value_default as vd
    original = vd.EROSION_FILTER_AKTIV
    vd.EROSION_FILTER_AKTIV = False
    try:
        manager2 = DataLODManager()
        manager2.set_map_distance_km(float(map_distance_km))
        g2 = BaseTerrainGenerator(data_lod_manager=manager2)
        g2.set_active_parameters(parameters)
        for node in ("terrain.noise", "terrain.redistribution"):
            manager2.set_calculator_target_lod(node, lod)
        g2._calc_noise("terrain.noise", lod)
        g2._calc_redistribution("terrain.redistribution", lod)
        roh = manager2.get_calculator_output(
            "terrain.redistribution", "heightmap", lod).astype(np.float64)
    finally:
        vd.EROSION_FILTER_AKTIV = original

    meter_pro_pixel = float(map_distance_km) * 1000.0 / float(map_size)

    # Das Delta direkt vom Filter, ohne die Spannen-Rueckbildung dazwischen.
    # Genau derselbe Weg, den _apply_erosion_filter geht - sonst waere hier
    # etwas anderes gemessen als in der App laeuft (§4.2).
    filter_parameter = generator._erosion_filter_parameters(
        int(map_size), float(map_distance_km))
    from core.terrain_erosion_filter import filter_heightmap
    delta = filter_heightmap(roh, meter_pro_pixel, filter_parameter)["height_delta"]

    return (heightmap.astype(np.float64), roh, delta.astype(np.float64),
            meter_pro_pixel)


def grobform(feld: np.ndarray, ziel: int = 64) -> np.ndarray:
    """
    Auf ziel x ziel heruntergerechnet und auf 0..1 normiert - die grobe Form
    ohne das Detail, das hoehere Aufloesung zurecht hinzufuegt. Blockmittel,
    damit kein Interpolationsverfahren die Aussage mitbestimmt.
    """
    n = feld.shape[0]
    if n < ziel:
        return np.full((ziel, ziel), np.nan)
    block = n // ziel
    beschnitten = feld[:block * ziel, :block * ziel]
    klein = beschnitten.reshape(ziel, block, ziel, block).mean(axis=(1, 3))
    spanne = klein.max() - klein.min()
    return (klein - klein.min()) / (spanne if spanne > 1e-9 else 1.0)


# =============================================================================
# LAEUFE
# =============================================================================

def lauf():
    fehler = []

    # ---------- 0: Gegenprobe Messgeraet ----------
    print("0. GEGENPROBE MESSGERAET (doppelte Rinnengroesse = doppelte Messung)")
    from gui.config.value_default import EROSION_FILTER
    in_metern = hasattr(EROSION_FILTER, "GULLY_SIZE_M")
    schluessel = ("erosion_filter_gully_size_m" if in_metern
                  else "erosion_filter_scale")
    klein_wert = 1000.0 if in_metern else 0.10
    gross_wert = 2000.0 if in_metern else 0.20
    EINE_OKTAVE = {"erosion_filter_octaves": 1}

    messungen = {}
    for etikett, wert in (("klein", klein_wert), ("gross", gross_wert)):
        _, _, delta, mpp = baue(256, 15.0, dict(EINE_OKTAVE, **{schluessel: wert}))
        messungen[etikett] = wellenlaenge_m(delta, mpp)
    verhaeltnis = messungen["gross"] / messungen["klein"]
    geraet_ok = 1.6 < verhaeltnis < 2.5
    print("   %s %-8s -> %6.0f m ;  %-8s -> %6.0f m ;  Verhaeltnis %.2f  %s"
          % (schluessel.replace("erosion_filter_", ""), klein_wert,
             messungen["klein"], gross_wert, messungen["gross"], verhaeltnis,
             "ok" if geraet_ok else "MESSGERAET UNBRAUCHBAR"))
    if not geraet_ok:
        fehler.append("Messgeraet: Verhaeltnis %.2f statt ~2.0 - die Laeufe 1 "
                      "und 2 belegen dann nichts" % verhaeltnis)
        for f in fehler:
            print("\n  FEHLER: %s" % f)
        return 1

    # ---------- 1: Aufloesung ----------
    print()
    print("1. AUFLOESUNG bei fester Ausdehnung (15 km)")
    print("   %-10s %10s %14s %16s" % ("map_size", "m/px", "Rinne in m", "Grobform-Abw."))
    referenz_grob = None
    wellen = []
    for map_size in (128, 256, 512):
        mit, _, delta, mpp = baue(map_size, 15.0, {"erosion_filter_octaves": 1})
        welle = wellenlaenge_m(delta, mpp)
        wellen.append(welle)
        grob = grobform(mit)
        if referenz_grob is None:
            referenz_grob, abweichung = grob, 0.0
        else:
            abweichung = float(np.abs(grob - referenz_grob).mean())
        print("   %-10d %10.1f %14.0f %15.3f" % (map_size, mpp, welle, abweichung))
        if abweichung > 0.06:
            fehler.append("map_size %d: Grobform weicht um %.3f ab - hoehere "
                          "Aufloesung veraendert die Landschaft, statt nur "
                          "Detail zu ergaenzen" % (map_size, abweichung))
    streuung = max(wellen) / min(wellen)
    print("   Streuung der Rinnengroesse: Faktor %.2f  %s"
          % (streuung, "ok" if streuung < 1.35 else "FEHLER"))
    if streuung >= 1.35:
        fehler.append("Rinnengroesse haengt an der Aufloesung (Faktor %.2f)"
                      % streuung)

    # ---------- 2: Ausdehnung ----------
    print()
    print("2. AUSDEHNUNG bei fester Aufloesung (256 px, Rinne 2000 m)")
    print("   %-16s %10s %14s" % ("map_distance_km", "m/px", "Rinne in m"))
    wellen = []
    for km in (10.0, 20.0, 40.0):
        _, _, delta, mpp = baue(256, km, {"erosion_filter_octaves": 1,
                                          "erosion_filter_gully_size_m": 2000.0})
        welle = wellenlaenge_m(delta, mpp)
        wellen.append(welle)
        print("   %-16.0f %10.1f %14.0f" % (km, mpp, welle))
    streuung = max(wellen) / min(wellen)
    ok = streuung < 1.35
    print("   Streuung der Rinnengroesse: Faktor %.2f  %s"
          % (streuung, "ok" if ok else "FEHLER"))
    if not ok:
        fehler.append(
            "Rinnengroesse haengt an der Kartenausdehnung (Faktor %.2f). Eine "
            "Rinne ist gegen die Wirklichkeit bemessen, nicht gegen den "
            "Bildausschnitt - beim Herauszoomen muessen MEHR Rinnen ins Bild "
            "passen, nicht groessere (§4.4)." % streuung)

    print()
    if fehler:
        for f in fehler:
            print("  FEHLER: %s" % f)
        return 1
    print("Alle Skalenebenen sind entkoppelt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(lauf())
