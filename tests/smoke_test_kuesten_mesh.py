"""
Path: tests/smoke_test_kuesten_mesh.py

Prueft gui/widgets/kuesten_mesh.py - Terrain-Vernetzung mit FREI gesetzten
Punkten entlang der Kuestenlinie (docs/OFFENE_PUNKTE.md 6.20, Nutzeridee:
"ein remesh kann auf jeder tangente passieren").

DIE DREI ZUSICHERUNGEN, AN DENEN BEIM BAUEN ETWAS SCHIEFLIEF, ZUERST:

  1. **Die Kuestenvertices muessen auf Hoehe 0 liegen.** Drei Anlaeufe waren
     noetig: Ausduennen+Nachverdichten (97 m daneben), Ausduennen+Abtasten
     (88 m), Hoehe nachschlagen statt setzen (75 m). Erst das SETZEN der
     Hoehe war richtig - der Punkt kommt ja aus der 0-Kontur.
  2. **Die Vertices duerfen NICHT auf Pixelecken sitzen.** Genau das ist der
     Unterschied zum Quadtree-Mesh (6.16), dessen Vertices gemessen 0.000004
     px von der Ecke abweichen, also exakt darauf liegen.
  3. **Die Kuestenkanten muessen tatsaechlich Dreieckskanten sein.** Ohne
     Constrained Delaunay ist das nur wahrscheinlich, nicht garantiert -
     deshalb wird der Anteil gemessen statt angenommen.

Laeuft mit echtem Weltgelaende und den echten Kartengroessen.
"""
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, ".")

from core.terrain_weltkarte import weltfeld
from gui.widgets.kuesten_mesh import (
    baue_kuesten_mesh, kuestenpunkte, kuestenlinien, stuetzpunkte,
    kuestentreue, KUESTE_ABSTAND_PX, INNEN_ABSTAND_PX)
from gui.widgets.adaptive_terrain_mesh import build_adaptive_mesh

WELT_KM = 21.3


def check(label, bedingung, zusatz=""):
    status = "OK" if bedingung else "FAIL"
    print(f"[{status}] {label}{(' - ' + zusatz) if zusatz else ''}")
    return bool(bedingung)


def _skalen(size):
    return 10.0 / size, 10.0 / (WELT_KM * 1000.0)


def run_kuestenvertices_auf_null():
    """Zusicherung 1 - die, an der drei Anlaeufe scheiterten."""
    ok = True
    for size in (256, 384, 512):
        H, _f = weltfeld(size, 20260804)
        tsf, ths = _skalen(size)
        kueste, _linien = kuestenpunkte(H)
        v, _i, _st = baue_kuesten_mesh(H, tsf, ths)
        V = v.reshape(-1, 8)
        # Kuestenpunkte stehen garantiert vorne (siehe stuetzpunkte())
        hoehen_m = V[:len(kueste), 1] / ths
        ok &= check(f"{size} px: alle Kuestenvertices auf Hoehe 0",
                    float(np.abs(hoehen_m).max()) < 1e-3,
                    f"max |{np.abs(hoehen_m).max():.5f}| m")
    return ok


def run_vertices_liegen_frei():
    """Zusicherung 2 - der eigentliche Zweck des ganzen Verfahrens."""
    size = 384
    H, _f = weltfeld(size, 20260804)
    kueste, _l = kuestenpunkte(H)
    abweichung = np.abs(kueste - np.round(kueste))
    frei = float((abweichung > 0.02).any(axis=1).mean())
    ok = check("Kuestenpunkte liegen NICHT auf Pixelecken", frei > 0.9,
               f"{frei:.1%} frei, mittlere Abweichung "
               f"{abweichung.mean():.3f} px")
    # Gegenprobe mit 512 px, NICHT mit 384: `build_adaptive_mesh()` verlangt
    # eine Kantenlaenge, die eine Zweierpotenz ist, und liefert sonst None.
    # 384 ist keine - das Kuesten-Mesh oben laeuft dort trotzdem, was
    # nebenbei ein Vorteil ist: es kennt diese Einschraenkung nicht.
    v_size = 512
    H2, _f2 = weltfeld(v_size, 20260804)
    tsf, ths = _skalen(v_size)
    v, _i, _st = build_adaptive_mesh(H2, tsf, ths, 6.0)
    V = v.reshape(-1, 8)
    px = (V[:, 0] / tsf / v_size + 0.5) * (v_size - 1)
    quadtree_abw = float(np.abs(px - np.round(px)).max())
    ok &= check("  Gegenprobe: Quadtree-Mesh liegt exakt auf dem Raster",
                quadtree_abw < 1e-3, f"max {quadtree_abw:.6f} px")
    ok &= check("  Kuesten-Mesh braucht KEINE Zweierpotenz (384 px lief oben)",
                build_adaptive_mesh(H, _skalen(size)[0], _skalen(size)[1], 6.0) is None,
                "Quadtree liefert bei 384 px None")
    return ok


def run_kuestentreue():
    """Zusicherung 3 - ohne Constrained Delaunay nur wahrscheinlich."""
    ok = True
    for size in (256, 384, 512):
        H, _f = weltfeld(size, 20260804)
        tsf, ths = _skalen(size)
        _v, _i, st = baue_kuesten_mesh(H, tsf, ths)
        ok &= check(f"{size} px: Kuestenkanten sind Dreieckskanten",
                    st["kuestentreue"] > 0.95, f"{st['kuestentreue']:.1%}")
    return ok


def run_topologie():
    """Keine Kante darf von mehr als zwei Dreiecken geteilt werden - sonst
    ist das Netz kaputt und es entstehen Risse oder Ueberlappungen."""
    ok = True
    for size in (256, 512):
        H, _f = weltfeld(size, 20260804)
        tsf, ths = _skalen(size)
        v, i, _st = baue_kuesten_mesh(H, tsf, ths)
        kanten = Counter()
        for a, b, c in i.reshape(-1, 3):
            for u, w in ((a, b), (b, c), (c, a)):
                kanten[(min(u, w), max(u, w))] += 1
        verteilung = Counter(kanten.values())
        ok &= check(f"{size} px: jede Kante von hoechstens 2 Dreiecken",
                    all(n in (1, 2) for n in kanten.values()),
                    f"{dict(verteilung)}")
        ok &= check(f"  Vertexdaten endlich, Indizes gueltig",
                    bool(np.all(np.isfinite(v))) and int(i.max()) < len(v) // 8)
        normalen = v.reshape(-1, 8)[:, 3:6]
        ok &= check("  Normalen normiert",
                    bool(np.allclose(np.linalg.norm(normalen, axis=1), 1.0,
                                     atol=1e-4)))
    return ok


def run_abdeckung_und_groesse():
    size = 512
    H, _f = weltfeld(size, 20260804)
    tsf, ths = _skalen(size)
    punkte, _linien, anzahl_kueste = stuetzpunkte(H)
    ok = check("Stuetzpunkte decken die ganze Karte ab",
               punkte[:, 0].max() >= size - 1.5 and punkte[:, 1].max() >= size - 1.5
               and punkte[:, 0].min() <= 0.5 and punkte[:, 1].min() <= 0.5)
    ok &= check("Kuestenpunkte stehen VORNE im Array "
                "(baue_kuesten_mesh verlaesst sich darauf)",
                anzahl_kueste > 0 and anzahl_kueste < len(punkte))

    _v, _i, sk = baue_kuesten_mesh(H, tsf, ths)
    _v2, _i2, sq = build_adaptive_mesh(H, tsf, ths, 6.0)
    ok &= check("weniger Dreiecke als das Quadtree-Mesh",
                sk["dreiecke"] < sq["dreiecke"],
                f"{sk['dreiecke']} gegen {sq['dreiecke']} "
                f"({sq['dreiecke'] / sk['dreiecke']:.1f}x)")
    ok &= check("deutlich weniger als das volle Gitter",
                sk["dreiecke"] < sq["voll_dreiecke"] * 0.1,
                f"{sk['dreiecke']} gegen {sq['voll_dreiecke']}")
    return ok


def run_randfaelle():
    ok = True
    # Karte ganz ohne Kueste (alles Land) - darf nicht abstuerzen
    H = np.full((128, 128), 100.0, dtype=np.float32)
    tsf, ths = _skalen(128)
    kueste, linien = kuestenpunkte(H)
    ok &= check("Karte ohne Kueste: keine Kuestenpunkte, kein Absturz",
                len(kueste) == 0 and linien == [])
    ergebnis = baue_kuesten_mesh(H, tsf, ths)
    ok &= check("  Mesh entsteht trotzdem (nur Innengitter)",
                ergebnis is not None and ergebnis[2]["dreiecke"] > 0,
                f"{ergebnis[2]['dreiecke']} Dreiecke" if ergebnis else "None")
    # Alles Wasser
    H2 = np.full((128, 128), -50.0, dtype=np.float32)
    ergebnis = baue_kuesten_mesh(H2, tsf, ths)
    ok &= check("Karte ganz unter Wasser: kein Absturz",
                ergebnis is None or ergebnis[2]["dreiecke"] > 0)
    # Treue-Messung ohne Linien
    ok &= check("Treue ohne Kuestenlinie ist 1.0 (nichts zu verfehlen)",
                kuestentreue(np.zeros((3, 2)), [(0, 1, 2)], []) == 1.0)
    return ok


def run_klippenband():
    """Senkrechte Klippenwaende als eigene Geometrie (docs/OFFENE_PUNKTE.md
    6.20, Variante 3 aus der Nutzerliste).

    Das ist die eigentliche Antwort auf "heightmap soll senkrechte flaechen
    ... darstellen koennen": eine Heightmap kann das per Definition nicht
    (ein z-Wert je xy), zwei Vertexreihen koennen es.
    """
    from gui.widgets.kuesten_mesh import (klippenband, KLIPPE_MINDESTHOEHE_M,
                                           KLIPPE_VERSATZ_PX)
    ok = True
    for size in (256, 512):
        H, _f = weltfeld(size, 20260804)
        tsf, ths = _skalen(size)
        v, i, st = klippenband(H, tsf, ths)
        ok &= check(f"{size} px: Waende entstanden", st["dreiecke"] > 0,
                    f"{st['dreiecke']} Dreiecke, {st['abschnitte']} Abschnitte")
        if st["dreiecke"] == 0:
            continue
        ok &= check("  Werte endlich, Indizes gueltig",
                    bool(np.all(np.isfinite(v))) and int(i.max()) < len(v))

        y_m = v[:, 1] / ths
        auf_null = np.abs(y_m) < 1e-3
        ok &= check("  untere Reihe liegt exakt auf der Wasserlinie",
                    bool(auf_null.any()) and
                    float(np.abs(y_m[auf_null]).max()) < 1e-3)
        ok &= check(f"  keine Wand unter {KLIPPE_MINDESTHOEHE_M:.0f} m "
                    "(sonst Streifen am Flachstrand)",
                    float(y_m[~auf_null].min()) >= KLIPPE_MINDESTHOEHE_M - 1e-3,
                    f"niedrigste {y_m[~auf_null].min():.1f} m")
        # Gleich viele untere wie obere Vertices - sonst fehlt einer Wand
        # ihre Ober- oder Unterkante.
        ok &= check("  gleich viele Wasserlinien- wie Oberkanten-Vertices",
                    int(auf_null.sum()) == int((~auf_null).sum()),
                    f"{auf_null.sum()} / {(~auf_null).sum()}")
        # KEIN toter Ballast: jeder Vertex muss in einem Dreieck vorkommen.
        ok &= check("  keine ungenutzten Vertices im Puffer",
                    len(np.unique(i)) == len(v),
                    f"{len(np.unique(i))} genutzt von {len(v)}")

    # Flachkarte ohne Kueste -> kein Band, kein Absturz
    flach = np.full((128, 128), 50.0, dtype=np.float32)
    tsf, ths = _skalen(128)
    v, i, st = klippenband(flach, tsf, ths)
    ok &= check("Karte ohne Kueste: kein Band, kein Absturz",
                len(v) == 0 and len(i) == 0 and st["dreiecke"] == 0)
    return ok


if __name__ == "__main__":
    ergebnisse = {
        "kuestenvertices_auf_null": run_kuestenvertices_auf_null(),
        "vertices_liegen_frei": run_vertices_liegen_frei(),
        "kuestentreue": run_kuestentreue(),
        "topologie": run_topologie(),
        "abdeckung_und_groesse": run_abdeckung_und_groesse(),
        "randfaelle": run_randfaelle(),
        "klippenband": run_klippenband(),
    }
    print("\n=== SUMMARY ===")
    for name, bestanden in ergebnisse.items():
        print(f"{name}: {'PASS' if bestanden else 'FAIL'}")
    sys.exit(0 if all(ergebnisse.values()) else 1)
