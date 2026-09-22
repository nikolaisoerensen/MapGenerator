"""
Path: tests/smoke_test_fluss_export.py

Prueft `core/fluss_export.py` (Ticket #37: Flussnetz als Linienzuege
exportieren).

Vier Gruppen, vom Handbaum zur echten Karte - gegen die fuenf
Abnahmekriterien des Tickets:

  1. `gebiet_je_knoten`/`fluss_segmente` an einem von Hand gebauten
     kleinen Y-Baum - Kanten-Identitaet: JEDE Baumkante gehoert zu GENAU
     einem Segment (Kriterium c, im Kleinen).
  2. `fluss_linien` auf demselben Handbaum - Endpunkte, Ordnung, Breite
     und Zusammenhang je exportierter Linie (Kriterien b, d).
  3. Eine echte kleine Weltkarte (`weltfeld` -> `_weltfluesse`) - der Baum
     UEBERLEBT (Kriterium a), und die Menge der Flussordnungen im Export
     deckt sich mit der Menge der Flussordnungen im tatsaechlich
     gezeichneten Raster (Kriterium c, gegen das Raster gemessen, nicht
     nur behauptet - "zwei Enden der Kette gegeneinander halten", siehe
     CLAUDE.md).
  4. Format-Gueltigkeit: das Export-Ergebnis ist verlustfrei JSON-
     serialisierbar (Kriterium e). Ein echter Godot-Lauf ist headless
     nicht moeglich (kein Godot-Projekt in diesem Repo) - dokumentierte
     Begruendung im Test selbst statt eines Live-Nachweises.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_fluss_export.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 20260804


def _y_baum():
    """
    Vier Knoten, zwei Quellen muenden in einen gemeinsamen Lauf:

        0 (Quelle A) --\
                         2 (Zusammenfluss) -- 3 (Auslass)
        1 (Quelle B) --/

    eltern zeigt auf den flussABwaertigen Nachbarn (-1 = Auslass).
    flaeche waechst stromabwaerts (0,1 je 1.0; am Zusammenfluss 3.0; am
    Auslass 4.0) - realistisch fuer akkumulierte Wasserflaeche.
    """
    punkte = np.array([
        [20.0, 0.0],    # 0: Quelle A
        [20.0, 20.0],   # 1: Quelle B
        [10.0, 10.0],   # 2: Zusammenfluss
        [0.0, 10.0],    # 3: Auslass
    ])
    eltern = np.array([2, 2, 3, -1], dtype=np.int64)
    flaeche = np.array([1.0, 1.0, 3.0, 4.0])
    # strahler_order braucht Kinder-vor-Eltern in reihenfolge[::-1]:
    # Kinder-vor-Eltern waere [0,1,2,3], reihenfolge also die Umkehrung.
    reihenfolge = np.array([3, 2, 1, 0])
    return punkte, eltern, flaeche, reihenfolge


def kanten_identitaet_am_handbaum():
    """1. Jede Baumkante gehoert zu genau einem Segment - keine Kante geht
    im Export verloren und keine wird doppelt gezaehlt."""
    import core.terrain_river_network as rn
    from core.fluss_sinuositaet import fluss_segmente
    from core.fluss_export import gebiet_je_knoten

    fehler = []
    punkte, eltern, flaeche, reihenfolge = _y_baum()
    order = rn.strahler_order(eltern, reihenfolge)
    print(f"Strahler-Ordnung: {order.tolist()} (erwartet [1, 1, 2, 2])")
    ok_order = order.tolist() == [1, 1, 2, 2]
    if not ok_order:
        fehler.append(f"Strahler-Ordnung {order.tolist()} statt [1, 1, 2, 2]")

    segmente = fluss_segmente(eltern, order)
    ketten = sorted(tuple(k) for k in segmente)
    erwartet = sorted([(0, 2), (1, 2), (2, 3)])
    print(f"Segmente: {ketten} (erwartet {erwartet})")
    if ketten != erwartet:
        fehler.append(f"Segmentierung {ketten} statt {erwartet}")

    kanten_gesamt = int(np.sum(eltern >= 0))
    kanten_in_segmenten = sum(len(k) - 1 for k in segmente)
    ok_kanten = kanten_gesamt == kanten_in_segmenten
    print(f"[{'OK' if ok_kanten else 'FEHLER'}] Kanten im Baum: {kanten_gesamt}, "
          f"Kanten in Segmenten aufsummiert: {kanten_in_segmenten}")
    if not ok_kanten:
        fehler.append(f"{kanten_in_segmenten} Kanten in Segmenten statt {kanten_gesamt} im Baum - "
                      f"Kriterium (c) verletzt: nicht ALLE Kanten exportiert")

    gebiet = gebiet_je_knoten(eltern, flaeche)
    ok_gebiet = gebiet[3] >= gebiet[2] >= gebiet[0] and abs(gebiet[3] - 1.0) < 1e-9
    print(f"gebiet_je_knoten: {gebiet.tolist()} (Auslass muss 1.0 sein, stromauf abnehmend)")
    if not ok_gebiet:
        fehler.append(f"gebiet_je_knoten nicht monoton/normiert: {gebiet.tolist()}")
    return fehler


def linien_am_handbaum():
    """2. fluss_linien() auf dem Y-Baum: Endpunkte, Ordnung, Breite,
    Zusammenhang (kein Sprung, keine Fragmentierung)."""
    import core.terrain_river_network as rn
    from core.fluss_sinuositaet import fluss_segmente
    from core.fluss_export import fluss_linien
    from core.terrain_weltfluesse import BEZUGSFORM_M

    fehler = []
    punkte, eltern, flaeche, reihenfolge = _y_baum()
    order = rn.strahler_order(eltern, reihenfolge)
    size = 25
    felder = {"formgroesse_m": np.full((size, size), BEZUGSFORM_M)}
    netz = {"punkte": punkte, "eltern": eltern, "flaeche": flaeche,
            "reihenfolge": reihenfolge}
    mpp = 100.0

    linien = fluss_linien(netz, order, felder, mpp,
                          breite_faktor=0.35, abstand_makro_m=1200.0)

    ok_anzahl = len(linien) == 3
    print(f"[{'OK' if ok_anzahl else 'FEHLER'}] {len(linien)} Linien exportiert (erwartet 3)")
    if not ok_anzahl:
        fehler.append(f"{len(linien)} Linien statt 3")

    segmente = {tuple(k): k for k in fluss_segmente(eltern, order)}
    for kette in segmente.values():
        # Zugehoerige Linie ueber ihre Ordnung + Endpunkte identifizieren
        soll_start_xy = [float(punkte[kette[0]][1]), float(punkte[kette[0]][0])]
        soll_ende_xy = [float(punkte[kette[-1]][1]), float(punkte[kette[-1]][0])]
        treffer = [ln for ln in linien
                  if ln["punkte"][0] == soll_start_xy or
                     np.allclose(ln["punkte"][0], soll_start_xy, atol=1e-6)]
        if not treffer:
            fehler.append(f"Keine Linie mit Startpunkt {soll_start_xy} fuer Kette {kette} gefunden")
            continue
        ln = treffer[0]

        # Ordnung muss zur Kette passen
        ok_ord = ln["ordnung"] == int(order[kette[0]])
        if not ok_ord:
            fehler.append(f"Kette {kette}: Ordnung {ln['ordnung']} statt {int(order[kette[0]])}")

        # Breite positiv und mindestens 2.5 Pixel (in Metern)
        ok_breite = ln["breite_m"] >= 2.5 * mpp - 1e-6
        if not ok_breite:
            fehler.append(f"Kette {kette}: Breite {ln['breite_m']} unter Mindestbreite {2.5*mpp}")

        # Endpunkt (Muendungsende der Kette) muss erreicht werden
        ende_erreicht = np.allclose(ln["punkte"][-1], soll_ende_xy, atol=1e-6)
        if not ende_erreicht:
            fehler.append(f"Kette {kette}: Endpunkt {ln['punkte'][-1]} statt {soll_ende_xy}")

        # ZUSAMMENHANG: kein Sprung groesser als das Doppelte der groessten
        # Knotenkante dieser Kette - sonst waere die Kurve fragmentiert
        # (Kriterium d).
        pts = np.asarray(ln["punkte"])
        gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        knoten_xy = np.array([[punkte[k][1], punkte[k][0]] for k in kette])
        max_kante = np.max(np.linalg.norm(np.diff(knoten_xy, axis=0), axis=1))
        ok_zusammenhang = gaps.max() < max_kante * 2.0 if len(gaps) else True
        print(f"  Kette {kette}: Ordnung {ln['ordnung']}, Breite {ln['breite_m']:.1f} m, "
              f"{len(pts)} Stuetzpunkte, groesster Sprung {gaps.max() if len(gaps) else 0:.2f} "
              f"(Grenze {max_kante*2.0:.2f})")
        if not ok_zusammenhang:
            fehler.append(f"Kette {kette}: Sprung {gaps.max():.2f} ueberschreitet Grenze "
                          f"{max_kante*2.0:.2f} - Linie wirkt fragmentiert")

    return fehler


def echte_karte_vollstaendig_und_ueberlebt():
    """
    3. Echte kleine Weltkarte: `_weltfluesse()` liefert Raster UND Linien
    aus demselben Aufruf. Kriterium (a): der Baum ueberlebt (Linien nicht
    leer). Kriterium (c): die Menge der Flussordnungen im Export deckt
    sich mit der Menge der Flussordnungen, die tatsaechlich im Raster
    gezeichnet wurden - gemessen gegen das Raster, nicht nur behauptet.
    """
    import logging
    from core.terrain_weltkarte import weltfeld
    from core.terrain_generator import BaseTerrainGenerator

    fehler = []
    size = 128
    H, felder = weltfeld(size, SEED)
    H = np.asarray(H, dtype=np.float64)

    gen = BaseTerrainGenerator.__new__(BaseTerrainGenerator)
    gen.shader_manager = None
    gen.data_lod_manager = None
    gen.logger = logging.getLogger("smoke_test_fluss_export")
    gen._current_parameters = {}

    (_geschnitten, _maske, ordnung, _generation, _wasser,
     linien) = gen._weltfluesse(H, felder, size, SEED)

    ok_ueberlebt = len(linien) > 0
    print(f"[{'OK' if ok_ueberlebt else 'FEHLER'}] {len(linien)} Linien aus dem echten Netz "
          f"exportiert (Kriterium a: der Baum ueberlebt die Rasterisierung)")
    if not ok_ueberlebt:
        fehler.append("fluss_linien() liefert nichts auf einer echten Karte - "
                      "der Baum ueberlebt NICHT (Kriterium a verletzt)")
        return fehler

    for ln in linien:
        if not (isinstance(ln.get("ordnung"), int) and ln["ordnung"] >= 1):
            fehler.append(f"Linie ohne gueltige Ordnung: {ln.get('ordnung')}")
        if not (isinstance(ln.get("breite_m"), float) and ln["breite_m"] > 0):
            fehler.append(f"Linie ohne gueltige Breite: {ln.get('breite_m')}")
        if len(ln.get("punkte", [])) < 2:
            fehler.append(f"Linie mit weniger als 2 Punkten: {ln}")

    ordnungen_export = sorted(set(int(ln["ordnung"]) for ln in linien))
    ordnungen_raster = sorted(int(o) for o in np.unique(ordnung) if o > 0)
    ok_deckung = ordnungen_export == ordnungen_raster
    print(f"Ordnungen im Export: {ordnungen_export}")
    print(f"Ordnungen im Raster: {ordnungen_raster}")
    print(f"[{'OK' if ok_deckung else 'FEHLER'}] Export- und Raster-Ordnungen decken sich "
          f"(Kriterium c, gegen das Raster gemessen)")
    if not ok_deckung:
        fehler.append(f"Export-Ordnungen {ordnungen_export} != Raster-Ordnungen {ordnungen_raster} - "
                      f"mindestens eine Flussordnung fehlt im Export oder im Raster")

    # Zusammenhang je Linie, wie in Gruppe 2, hier an der echten Karte.
    for ln in linien:
        pts = np.asarray(ln["punkte"])
        if len(pts) < 3:
            continue
        gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        # Grosszuegige Grenze: die Kartenkantenlaenge selbst kann als
        # Obergrenze fuer einen Einzelsprung gelten - jeder groessere
        # Sprung waere sicher ein Fehler, keine reale Geometrie.
        if gaps.max() > size:
            fehler.append(f"Linie Ordnung {ln['ordnung']}: Sprung {gaps.max():.1f} px "
                          f"groesser als die Karte selbst ({size} px) - fragmentiert")

    return fehler


def format_ist_json_gueltig():
    """
    4. Kriterium (e): Godot-Lesbarkeit.

    Es gibt in diesem Repository kein Godot-Projekt und keine .gd-Dateien -
    ein echter Godot-Importlauf ist headless nicht moeglich und wird hier
    NICHT vorgetaeuscht. Stattdessen wird geprueft, was tatsaechlich
    pruefbar ist: das Export-Format ist reines JSON (Liste von Dicts mit
    "punkte" [[x,y],...], "ordnung" int, "breite_m" float) und ueberlebt
    einen JSON-Hin-und-Rueck-Lauf ohne Praezisionsverlust.

    Begruendung, warum das fuer Godot reicht (dokumentiert statt getestet):
    Godot 4 laedt beliebiges JSON ueber die eingebaute `JSON`-Klasse
    (`JSON.parse_string()`); die entstehenden Arrays/Dictionaries lassen
    sich direkt in `Curve3D.add_point()`/`Path3D` einspeisen (GDScript-
    Kernfunktionalitaet, keine Zusatz-Bibliothek). Ein `[[x,y],...]`-
    Polygonzug ist exakt das Format, das dieses Projekt fuer "wege" und
    "seewege" bereits verwendet (gui/utils/map_export.py) - dieselbe
    Struktur, nur mit den zwei zusaetzlichen Feldern "ordnung"/"breite_m".
    """
    import json
    import core.terrain_river_network as rn
    from core.fluss_export import fluss_linien
    from core.terrain_weltfluesse import BEZUGSFORM_M

    fehler = []
    punkte, eltern, flaeche, reihenfolge = _y_baum()
    order = rn.strahler_order(eltern, reihenfolge)
    size = 25
    felder = {"formgroesse_m": np.full((size, size), BEZUGSFORM_M)}
    netz = {"punkte": punkte, "eltern": eltern, "flaeche": flaeche,
            "reihenfolge": reihenfolge}
    linien = fluss_linien(netz, order, felder, 100.0)

    try:
        text = json.dumps(linien)
        zurueck = json.loads(text)
    except (TypeError, ValueError) as fehlermeldung:
        fehler.append(f"nicht JSON-serialisierbar: {fehlermeldung}")
        return fehler

    ok = zurueck == linien
    print(f"[{'OK' if ok else 'FEHLER'}] JSON-Hin-und-Rueck-Lauf verlustfrei "
          f"({len(text)} Zeichen fuer {len(linien)} Linien)")
    if not ok:
        fehler.append("JSON-Rundlauf veraendert die Daten (Praezisions- oder Typverlust)")
    return fehler


def lauf():
    gruppen = [
        ("kanten_identitaet_am_handbaum", kanten_identitaet_am_handbaum),
        ("linien_am_handbaum", linien_am_handbaum),
        ("echte_karte_vollstaendig_und_ueberlebt", echte_karte_vollstaendig_und_ueberlebt),
        ("format_ist_json_gueltig", format_ist_json_gueltig),
    ]
    ergebnis, alle = {}, []
    for name, fn in gruppen:
        print(f"\n--- {name} ---")
        f = fn()
        ergebnis[name] = not f
        alle.extend(f)

    print("\n=== SUMMARY ===")
    for name, ok in ergebnis.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if alle:
        print(f"\nNICHT IN ORDNUNG - {len(alle)} Befunde:")
        for f in alle:
            print(f"   {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(lauf())
