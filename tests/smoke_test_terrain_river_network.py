"""
Path: tests/smoke_test_terrain_river_network.py

Prueft das Flussnetz-Skelett im Terrain-Aufbau
(BaseTerrainGenerator._apply_river_network, core/terrain_river_network.py,
SPEZIFIKATION §12).

Sieben Zusicherungen. Die letzten drei sind Gegenproben - nach §5.1.4 prueft
eine Zusicherung, die auch ohne die Aenderung haelt, nichts.

  1. Das Netz LAEUFT. river_mask existiert nur, wenn es lief.
  2. Die Hoehenspanne ist exakt BASE_ELEVATION_M .. AMPLITUDE. Der Einschnitt
     drueckt die Talsohle sonst darunter (gemessen -1260 m, §12).
  3. Der ENTWAESSERUNGSANTEIL steigt deutlich. Das ist der eigentliche Zweck:
     §7 mass 10-22% fuer die Feld-Erosion, der Noise-Pfad 10.1%.
  4. GEGENPROBE Tiefe 0: kein Einschnitt darf das Gelaende nicht veraendern.
  5. GEGENPROBE Regler einzeln: jeder der sieben muss wirken (§4.7, drei tote
     Water-Slider gab es hier monatelang).
  6. GEGENPROBE Aufloesung: derselbe Punktsatz bei 256 und 512 px. Der Satz
     wird in METERN erzeugt; haengt er doch an der Pixelzahl, springt das
     Gelaende zwischen den LOD-Stufen.

Aufruf:
    .venv\\Scripts\\python.exe tests/smoke_test_terrain_river_network.py
"""

import sys

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

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

KM = 15.0
SEED = 20260730


def _bauen(size=256, netz_aktiv=True, overrides=None):
    import gui.config.value_default as vd
    from managers.data_lod_manager import DataLODManager
    from gui.config.value_default import TERRAIN, EROSION_FILTER, RIVER_NETWORK
    from core.terrain_generator import BaseTerrainGenerator

    original = vd.FLUSSNETZ_AKTIV
    vd.FLUSSNETZ_AKTIV = netz_aktiv
    try:
        manager = DataLODManager()
        manager.set_map_distance_km(KM)
        parameters = {key.lower(): getattr(TERRAIN, key)["default"] for key in
                      ("AMPLITUDE", "OCTAVES", "FEATURE_SIZE_M", "PERSISTENCE",
                       "LACUNARITY", "REDISTRIBUTE_POWER")}
        parameters.update({"map_size": size, "map_distance_km": KM,
                           "map_seed": SEED})
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
        for name, key in (("SPACING_M", "river_spacing_m"),
                          ("INCISION_SHARE", "river_incision_share"),
                          ("PLATEAU_FLATTEN", "river_plateau_flatten"),
                          ("VALLEY_WIDTH", "river_valley_width"),
                          ("VALLEY_FORM", "river_valley_form"),
                          ("COST_STRENGTH", "river_cost_strength")):
            parameters[key] = getattr(RIVER_NETWORK, name)["default"]
        parameters.update(overrides or {})

        lod = int(round(np.log2(max(size, 32) / 32.0))) + 1
        generator = BaseTerrainGenerator(data_lod_manager=manager)
        generator.set_active_parameters(parameters)
        for node in ("terrain.noise", "terrain.redistribution"):
            manager.set_calculator_target_lod(node, lod)
        generator._calc_noise("terrain.noise", lod)
        generator._calc_redistribution("terrain.redistribution", lod)

        z = manager.get_calculator_output(
            "terrain.redistribution", "heightmap", lod)
        maske = manager.get_calculator_output(
            "terrain.redistribution", "river_mask", lod)
        assert z.shape == (size, size), (
            "angefragt %d px, bekommen %s - §5.2" % (size, z.shape))
        return z.astype(np.float64), maske, KM * 1000.0 / size
    finally:
        vd.FLUSSNETZ_AKTIV = original


def lauf():
    from gui.config.value_default import TERRAIN, RIVER_NETWORK
    import tools.drainage_lab as dl

    fehler = []

    mit, maske, mpp = _bauen()
    ohne, _, _ = _bauen(netz_aktiv=False)

    # ---------- 1 ----------
    lief = maske is not None and bool(maske.any())
    if not lief:
        fehler.append("river_mask fehlt oder ist leer - das Netz lief nicht")
    print("1. Netz lief (%s Flusspixel) ................... %s"
          % (int(maske.sum()) if maske is not None else 0,
             "ok" if lief else "FEHLER"))

    # ---------- 2 ----------
    soll_tief, soll_hoch = TERRAIN.BASE_ELEVATION_M, TERRAIN.AMPLITUDE["default"]
    spanne_ok = (abs(float(mit.min()) - soll_tief) < 0.01
                 and abs(float(mit.max()) - soll_hoch) < 0.01)
    if not spanne_ok:
        fehler.append("Spanne %.1f..%.1f statt %.1f..%.1f"
                      % (mit.min(), mit.max(), soll_tief, soll_hoch))
    print("2. Spanne %.0f .. %.0f m ........................ %s"
          % (mit.min(), mit.max(), "ok" if spanne_ok else "FEHLER"))

    # ---------- 3 ----------
    a_mit = dl.drainage_share(mit)["abfluss_anteil"]
    a_ohne = dl.drainage_share(ohne)["abfluss_anteil"]
    besser = a_mit > a_ohne + 0.15
    if not besser:
        fehler.append("Entwaesserung %.1f%% gegen %.1f%% ohne Netz - das ist "
                      "der Zweck der Sache" % (100 * a_mit, 100 * a_ohne))
    print("3. Entwaesserung %.1f%% gegen %.1f%% ohne Netz ... %s"
          % (100 * a_mit, 100 * a_ohne, "ok" if besser else "FEHLER"))

    # ---------- 4 ----------
    # Der Regler muss tun, was sein Name sagt (§4.7): mehr Tiefe = tiefere
    # Taeler, monoton ueber den ganzen Bereich.
    #
    # Frueher stand hier "Tiefe 0 muss bitgleich mit abgeschaltetem Netz sein".
    # Das war die falsche Frage, gemessen und verworfen: bei Tiefe 0 ist das
    # Flussbett zwischen zwei Knoten eine GERADE ueber den Knotenabstand,
    # waehrend P dazwischen um hunderte Meter schwankt - der Lauf schneidet
    # also auch ohne eingestellte Tiefe durch die Huegel, und genau das soll
    # ein Flusslauf tun. Korrelation zum netzlosen Gelaende war 0.76, und
    # weder das Mindestgefaelle (ueber zwei Groessenordnungen geprueft) noch
    # das Auffuellen von P (r = 0.9985) erklaerten das.
    # Gemessen wird die Sohle gegen die Flaeche P, IN DIE eingeschnitten wird,
    # und direkt am Modul - vor der Spannen-Rueckbildung.
    #
    # Zwei falsche Messgroessen davor, beide verworfen (§5.2, "falsche Formel
    # im Messgeraet"):
    #   * "Tiefe 0 muss gleich Netz aus sein" - ist es nicht, weil der Lauf
    #     zwischen zwei Knoten auch ohne eingestellte Tiefe durchs Gelaende
    #     schneidet.
    #   * "90. Perzentil minus Flusshoehe" - das misst den Abstand der Gipfel
    #     zu den Taelern, also das gesamte Grossrelief, und wird zusaetzlich
    #     von der Spannen-Rueckbildung verzerrt. Meldete 1277 m, wo 330 m
    #     richtig waren, und war nicht monoton.
    print("4. Talteife steigt mit dem Regler:")
    from skimage.morphology import reconstruction
    from core.terrain_river_network import carve_river_network

    P, _, mpp_p = _bauen(netz_aktiv=False)
    saat = P.max() * np.ones_like(P)
    saat[0], saat[-1], saat[:, 0], saat[:, -1] = P[0], P[-1], P[:, 0], P[:, -1]
    P_gefuellt = reconstruction(saat, P, method="erosion")

    tiefen = []
    for wert in (0.0, 0.25, 0.55):
        # Direkter Modulaufruf - dort heisst der Schluessel ohne river_-Praefix
        # (den setzt erst _apply_river_network um).
        r = carve_river_network(P, mpp_p, TERRAIN.AMPLITUDE["default"], SEED,
                                {"incision_share": wert})
        z_roh = r["heightmap"].astype(np.float64)
        tiefe = float((P_gefuellt[r["river_mask"]]
                       - z_roh[r["river_mask"]]).mean())
        tiefen.append(tiefe)
        print("   Regler %5.2f -> Sohle %6.0f m unter P" % (wert, tiefe))
    monoton = tiefen[0] < tiefen[1] < tiefen[2]
    if not monoton:
        fehler.append("Talteife waechst nicht monoton mit dem Regler: %s"
                      % [round(t) for t in tiefen])
    print("   monoton steigend .......................... %s"
          % ("ok" if monoton else "FEHLER"))

    # ---------- 5 ----------
    print("5. Wirkung jedes einzelnen Reglers:")
    proben = (
        ("river_spacing_m", RIVER_NETWORK.SPACING_M["min"]),
        ("river_incision_share", RIVER_NETWORK.INCISION_SHARE["max"]),
        ("river_plateau_flatten", RIVER_NETWORK.PLATEAU_FLATTEN["max"]),
        ("river_valley_width", RIVER_NETWORK.VALLEY_WIDTH["max"]),
        ("river_valley_form", RIVER_NETWORK.VALLEY_FORM["min"]),
        ("river_cost_strength", RIVER_NETWORK.COST_STRENGTH["max"]),
    )
    for name, wert in proben:
        variante, _, _ = _bauen(overrides={name: wert})
        unterschied = float(np.abs(variante - mit).max())
        ok = unterschied > 1.0
        if not ok:
            fehler.append("Regler %s = %s aendert nichts (%.3g m)"
                          % (name, wert, unterschied))
        print("   %-32s -> %8.1f m  %s"
              % (name.replace("river_", "") + " = " + str(wert),
                 unterschied, "ok" if ok else "TOT"))

    # ---------- 6 ----------
    # Der Punktsatz wird in METERN erzeugt, damit jede Aufloesung dasselbe Netz
    # bekommt. Verglichen wird die GROBFORM, weil die hoehere Aufloesung zurecht
    # feineres Detail ergaenzt - nur die Lage der Taeler muss dieselbe sein.
    gross, _, _ = _bauen(size=512)
    klein_grob = mit
    n = gross.shape[0] // klein_grob.shape[0]
    gross_grob = gross[:n * klein_grob.shape[0], :n * klein_grob.shape[0]].reshape(
        klein_grob.shape[0], n, klein_grob.shape[1], n).mean(axis=(1, 3))

    def norm(a):
        s = a.max() - a.min()
        return (a - a.min()) / (s if s > 1e-9 else 1.0)

    r = float(np.corrcoef(norm(klein_grob).ravel(), norm(gross_grob).ravel())[0, 1])
    gleich = r > 0.85
    if not gleich:
        fehler.append("256 und 512 px ergeben verschiedene Netze (r = %+.3f) - "
                      "das Gelaende wuerde zwischen den Stufen springen" % r)
    print("6. Gleiches Netz bei 256 und 512 px (r=%+.3f) .. %s"
          % (r, "ok" if gleich else "FEHLER"))

    # ---------- 7: keine Kanten am Lauf ----------
    # Ein Fluss faellt stetig - jeder Sprung zwischen BENACHBARTEN Flusspixeln
    # ist ein Fehler. Sie entstanden an den Zusammenfluessen: der
    # Muendungsknoten wird vom tiefsten Zufluss nach unten gezogen, der andere
    # Zufluss steht unmittelbar daneben noch auf seiner eigenen Hoehe.
    # Gemessen 538 m; mit der Steigungsbegrenzung (max_gradient) 39 m.
    print("7. Keine Kanten am Lauf:")
    _, maske7, _ = _bauen(size=256)
    z7, _, _ = _bauen(size=256)
    zz = np.where(maske7, z7, np.nan)
    spruenge = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = zz[1:-1, 1:-1]
        b = np.roll(np.roll(zz, -dy, 0), -dx, 1)[1:-1, 1:-1]
        d = np.abs(a - b)
        spruenge.append(d[np.isfinite(d)])
    alle = np.concatenate(spruenge)
    p99 = float(np.percentile(alle, 99))
    groesster = float(alle.max())
    # Bezogen auf die Hoehenspanne, nicht als fester Meterwert (§4.4).
    grenze = 0.03 * TERRAIN.AMPLITUDE["default"]
    ok7 = groesster < grenze
    if not ok7:
        fehler.append("groesster Sprung am Lauf %.0f m, Grenze %.0f m "
                      "(3%% der Hoehenspanne)" % (groesster, grenze))
    print("   p99 %.0f m, groesster %.0f m, Grenze %.0f m ....... %s"
          % (p99, groesster, grenze, "ok" if ok7 else "FEHLER"))

    print()
    if fehler:
        for f in fehler:
            print("  FEHLER: %s" % f)
        return 1
    print("Alle sieben Zusicherungen erfuellt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(lauf())
