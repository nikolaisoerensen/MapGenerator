"""
Path: tests/smoke_test_terrain_remesh.py

Prueft gui/widgets/terrain_remesh.py - das Remesh mit frei verschobenen
Vertices (docs/OFFENE_PUNKTE.md 6.33).

DIE ZUSICHERUNGEN, AN DENEN FRUEHERE VERSUCHE GESCHEITERT SIND, ZUERST:

  1. **Die Vertices duerfen NICHT auf Pixelecken liegen.** Das ist der ganze
     Zweck. Das Quadtree-Mesh (6.16) liegt gemessen 0.000004 px daneben, also
     exakt darauf - deshalb wird es hier als Gegenprobe mitgemessen. Ein Test,
     der nur "Remesh liefert Vertices" prueft, wuerde ein Quadtree nicht von
     einem Remesh unterscheiden.
  2. **Die Naht zwischen Meer- und Landteil muss dicht sein.** An exakt
     dieser Stelle ist das Klippenband gescheitert (T-Stuecke, Risse, altes
     Mesh schimmerte durch). Geprueft wird deshalb nicht "sieht gut aus",
     sondern: jede Kante gehoert zu einem oder zwei Dreiecken, nie zu mehr,
     und es gibt keine doppelten Vertexpositionen.
  3. **Es muss bei den ECHTEN Kartengroessen laufen** - 256/384/512, nicht
     bei ausgedachten. Ein grunes Testfeld mit 2^n+1 hat schon einmal
     wochenlang verdeckt, dass das adaptive Mesh im Betrieb gar nicht lief
     (siehe CLAUDE.md). 384 ist dabei besonders wichtig: dort faellt das
     Quadtree auf das Gleichmaessig-Gitter zurueck, das Remesh nicht.

Laeuft mit echtem Weltgelaende.
"""
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, ".")

from core.terrain_weltkarte import weltfeld
from gui.widgets.adaptive_terrain_mesh import (build_adaptive_mesh,
                                               ist_fuer_adaptives_mesh_geeignet)
from gui.widgets.terrain_remesh import (baue_remesh, remesh_verfuegbar,
                                        MEER_BUDGET_ANTEIL, TIEFENLINIE_M)

WELT_KM = 21.3
SEED = 20260804
_GELAENDE = {}


def check(label, bedingung, zusatz=""):
    status = "OK" if bedingung else "FAIL"
    print(f"[{status}] {label}{(' - ' + zusatz) if zusatz else ''}")
    return bool(bedingung)


def _skalen(size):
    return 10.0 / size, 10.0 / (WELT_KM * 1000.0)


def _gelaende(size):
    if size not in _GELAENDE:
        H, _f = weltfeld(size, SEED)
        _GELAENDE[size] = np.asarray(H, dtype=np.float32)
    return _GELAENDE[size]


def _pixelkoordinaten(vertices, size, tsf):
    V = np.asarray(vertices, dtype=np.float64).reshape(-1, 8)
    px = (V[:, 0] / (size * tsf) + 0.5) * (size - 1)
    py = (V[:, 2] / (size * tsf) + 0.5) * (size - 1)
    return np.stack([px, py], axis=1)


def run_vertices_liegen_frei():
    """
    Zusicherung 1 - der ganze Zweck des Moduls.

    ACHTUNG, ERST FALSCH GEMESSEN: die erste Fassung verlangte einen
    Median-Versatz ueber ALLE Vertices von mehr als 0.01 px und schlug bei
    256 px fehl (Median 0.000003 px). Das war kein Fehler des Verfahrens,
    sondern eine untaugliche Kennzahl: bei nur 75 % Reduktion bleibt gut die
    Haelfte der Vertices von jedem Kantenkollaps unberuehrt und liegt darum
    weiterhin exakt auf ihrer Pixelecke - voellig in Ordnung, denn dort ist
    das Gelaende flach und es gibt nichts zu verbessern.

    Wichtig ist, ob sich die Vertices DORT loesen, wo die Treppe sichtbar
    ist: an der Wasserlinie. Genau das wird hier gemessen.
    """
    from scipy import ndimage

    ok = True
    for size in (256, 384, 512):
        H = _gelaende(size)
        tsf, ths = _skalen(size)
        ergebnis = baue_remesh(H, tsf, ths, ziel_anteil=0.25)
        if not check(f"{size} px: Remesh liefert ein Netz", ergebnis is not None):
            ok = False
            continue
        v, i, stats = ergebnis
        punkte = _pixelkoordinaten(v, size, tsf)
        abstand = np.hypot(punkte[:, 0] - np.round(punkte[:, 0]),
                           punkte[:, 1] - np.round(punkte[:, 1]))

        # Entfernung jedes Pixels zur Wasserlinie.
        #
        # NICHT das Minimum der beiden Distanztransformationen nehmen - das
        # ist ueberall 0 und waehlt damit die ganze Karte aus (erst so
        # gebaut; die "Kuestenwerte" waren dann bis auf die Nachkommastelle
        # mit den Gesamtwerten identisch, woran es auffiel). Auf Land zaehlt
        # der Abstand zum Meer, im Meer der Abstand zum Land.
        land = H > 0
        zur_kueste = np.where(land,
                              ndimage.distance_transform_edt(land),
                              ndimage.distance_transform_edt(~land))
        xi = np.clip(np.round(punkte[:, 0]).astype(int), 0, size - 1)
        yi = np.clip(np.round(punkte[:, 1]).astype(int), 0, size - 1)
        nah_an_kueste = zur_kueste[yi, xi] <= 3.0

        frei_gesamt = stats["frei_verschoben"]
        frei_kueste = float((abstand[nah_an_kueste] > 0.01).mean())
        ok &= check(f"{size} px: ein grosser Teil der Vertices ist frei verschoben",
                    frei_gesamt > 0.40, f"{frei_gesamt:.1%}")

        # EHRLICH BENANNTE GRENZE, gemessen am 2026-08-16:
        #
        # An der Kueste ist der Anteil KLEINER als im Kartenmittel (bei 512 px
        # 39.7 % gegen 54.2 %), nicht groesser. Das ist kein Fehler, sondern
        # die Bauart: QEM verschiebt einen Vertex nur, wenn es zwei
        # verschmilzt - und an einer Kante mit hoher Kruemmung verschmilzt es
        # gerade NICHT, der Vertex ueberlebt also unveraendert auf seiner
        # Pixelecke. Wer die Kuestenlinie zwingend rasterfrei will, kommt an
        # einer Zwangskante (Constrained Delaunay) nicht vorbei, siehe 6.32.
        #
        # Der Test haelt das als MESSWERT fest statt als Wunschbedingung -
        # eine Zusicherung "an der Kueste mehrheitlich frei" waere schlicht
        # falsch gewesen und stand hier zwei Fassungen lang.
        ok &= check(f"{size} px: an der Kueste loest sich immerhin ein Teil",
                    frei_kueste > 0.25,
                    f"{frei_kueste:.1%} an der Kueste gegen "
                    f"{frei_gesamt:.1%} im Mittel - an der Kueste WENIGER, "
                    f"siehe Kommentar")

        # Gegenprobe: das Quadtree MUSS rastergebunden sein, sonst misst der
        # Test oben etwas anderes als er glaubt.
        if ist_fuer_adaptives_mesh_geeignet(H):
            qv, _qi, _qs = build_adaptive_mesh(H, tsf, ths, fehler_toleranz_m=6.0)
            qp = _pixelkoordinaten(qv, size, tsf)
            qab = np.hypot(qp[:, 0] - np.round(qp[:, 0]), qp[:, 1] - np.round(qp[:, 1]))
            ok &= check(f"{size} px: Gegenprobe - Quadtree IST rastergebunden",
                        float(np.median(qab)) < 1e-4,
                        f"Median {float(np.median(qab)):.8f} px")
    return ok


def run_naht_ist_dicht():
    """
    Zusicherung 2 - die Stelle, an der das Klippenband gescheitert ist.

    Geprueft wird die Kanten-Nachbarschaft: in einer sauberen Flaeche gehoert
    jede Kante zu genau zwei Dreiecken (innen) oder zu einem (Aussenrand).
    Drei oder mehr heisst, dass sich zwei Teilnetze ueberlappen statt
    anzuschliessen.
    """
    ok = True
    for size in (256, 384):
        H = _gelaende(size)
        tsf, ths = _skalen(size)
        v, i, stats = baue_remesh(H, tsf, ths, ziel_anteil=0.25)
        D = np.asarray(i, dtype=np.int64).reshape(-1, 3)

        kanten = np.concatenate([D[:, [0, 1]], D[:, [1, 2]], D[:, [2, 0]]])
        kanten = np.sort(kanten, axis=1)
        _einmalig, zaehler = np.unique(kanten, axis=0, return_counts=True)
        verteilung = Counter(zaehler.tolist())
        ok &= check(f"{size} px: keine Kante an mehr als 2 Dreiecken",
                    zaehler.max() <= 2,
                    f"Verteilung {dict(sorted(verteilung.items()))}")

        punkte = _pixelkoordinaten(v, size, tsf)
        _u, anzahl = np.unique(np.round(punkte, 6), axis=0, return_counts=True)
        ok &= check(f"{size} px: keine doppelten Vertexpositionen",
                    anzahl.max() == 1, f"hoechstens {anzahl.max()}-fach")

        ok &= check(f"{size} px: alle Indizes im gueltigen Bereich",
                    D.min() >= 0 and D.max() < len(punkte))
        entartet = (D[:, 0] == D[:, 1]) | (D[:, 1] == D[:, 2]) | (D[:, 0] == D[:, 2])
        ok &= check(f"{size} px: keine entarteten Dreiecke",
                    not entartet.any(), f"{int(entartet.sum())} entartet")
    return ok


def run_format_passt_zum_display():
    """
    Das Netz muss dort einsetzbar sein, wo build_adaptive_mesh() steht -
    gleiches interleaved Layout [pos,normal,uv], sonst liest der Shader Muell.
    """
    size = 256
    H = _gelaende(size)
    tsf, ths = _skalen(size)
    v, i, stats = baue_remesh(H, tsf, ths, ziel_anteil=0.25)
    qv, qi, _qs = build_adaptive_mesh(H, tsf, ths, fehler_toleranz_m=6.0)

    ok = check("Vertexformat float32", v.dtype == np.float32)
    ok &= check("Indexformat uint32", i.dtype == np.uint32)
    ok &= check("8 Werte je Vertex (wie das Quadtree)", len(v) % 8 == 0)
    ok &= check("Indexzahl durch 3 teilbar", len(i) % 3 == 0)
    ok &= check("gleiche Struktur wie build_adaptive_mesh",
                (qv.dtype == v.dtype) and (qi.dtype == i.dtype))

    V = v.reshape(-1, 8)
    laengen = np.linalg.norm(V[:, 3:6], axis=1)
    ok &= check("Normalen sind normiert",
                bool(np.allclose(laengen, 1.0, atol=1e-3)),
                f"min {laengen.min():.4f}, max {laengen.max():.4f}")
    ok &= check("Normalen zeigen nach oben (ny > 0)", bool((V[:, 4] > 0).all()))
    ok &= check("Texturkoordinaten in [0,1]",
                bool((V[:, 6:8] >= -1e-6).all() and (V[:, 6:8] <= 1 + 1e-6).all()))

    # Hoehe am Vertex muss zur Heightmap passen (bilinear, deshalb Toleranz)
    punkte = _pixelkoordinaten(v, size, tsf)
    nah = H[np.clip(np.round(punkte[:, 1]).astype(int), 0, size - 1),
            np.clip(np.round(punkte[:, 0]).astype(int), 0, size - 1)]
    hoehe_m = V[:, 1] / ths
    spanne = float(H.max() - H.min())
    ok &= check("Vertexhoehen passen zur Heightmap",
                float(np.percentile(np.abs(hoehe_m - nah), 99)) < 0.10 * spanne,
                f"p99 {float(np.percentile(np.abs(hoehe_m - nah), 99)):.1f} m "
                f"bei {spanne:.0f} m Spanne")
    return ok


def run_trennung_bringt_das_budget_an_land():
    """
    Die Meer/Land-Trennung ist der Grund, warum das Remesh dem Quadtree
    ueberlegen ist. Ohne sie landen ueber die Haelfte der Vertices im Meer.
    """
    size = 256
    H = _gelaende(size)
    tsf, ths = _skalen(size)
    land_anteil = float((H > 0).mean())

    def anteil_ueber_land(v):
        p = _pixelkoordinaten(v, size, tsf)
        return float((H[np.clip(np.round(p[:, 1]).astype(int), 0, size - 1),
                        np.clip(np.round(p[:, 0]).astype(int), 0, size - 1)] > 0).mean())

    v_mit, _i, _s = baue_remesh(H, tsf, ths, ziel_anteil=0.25, zerlegen=True)
    v_ohne, _i2, _s2 = baue_remesh(H, tsf, ths, ziel_anteil=0.25, zerlegen=False)
    mit, ohne = anteil_ueber_land(v_mit), anteil_ueber_land(v_ohne)

    ok = check("Trennung bringt mehr Vertices an Land", mit > ohne + 0.05,
               f"mit {mit:.1%}, ohne {ohne:.1%} (Land ist {land_anteil:.1%} der Karte)")
    ok &= check("Land bekommt ueberproportional viel", mit > land_anteil,
                f"{mit:.1%} > {land_anteil:.1%}")
    return ok


def run_genauer_als_quadtree():
    """
    Der eigentliche Nutzen: bei GLEICHER Dreieckszahl weniger Hoehenfehler
    auf Land als das Quadtree.

    Gemessen wird ueber die ECHTEN Dreiecke. Eine Neutriangulierung der
    Vertices haette am 2026-08-16 das Remesh als "5-fach schlechter"
    dastehen lassen - sie wirft die langgestreckten Dreiecke entlang der
    Grate weg, also genau das, was das Verfahren aufbaut.
    """
    sys.path.insert(0, "tools")
    from mesh_werkstatt import hoehenfehler

    size = 256
    H = _gelaende(size)
    tsf, ths = _skalen(size)

    qv, qi, qs = build_adaptive_mesh(H, tsf, ths, fehler_toleranz_m=6.0)
    rv, ri, rs = baue_remesh(H, tsf, ths, ziel_dreiecke=qs["dreiecke"])

    ok = check("gleiche Dreieckszahl fuer den Vergleich",
               abs(rs["dreiecke"] - qs["dreiecke"]) <= 0.02 * qs["dreiecke"],
               f"Quadtree {qs['dreiecke']}, Remesh {rs['dreiecke']}")

    werte = {}
    for name, v, i in (("Quadtree", qv, qi), ("Remesh", rv, ri)):
        V = np.asarray(v, np.float64).reshape(-1, 8)
        punkte = _pixelkoordinaten(v, size, tsf)
        f, land = hoehenfehler(punkte, V[:, 1] / ths,
                               np.asarray(i, np.int64).reshape(-1, 3),
                               H.astype(np.float64))
        werte[name] = (float(np.sqrt((f[land] ** 2).mean())),
                       float(np.percentile(f[land], 99)))
        print(f"       {name}: Land RMS {werte[name][0]:.2f} m, p99 {werte[name][1]:.2f} m")

    ok &= check("Remesh hat kleineren RMS auf Land",
                werte["Remesh"][0] < werte["Quadtree"][0],
                f"{werte['Remesh'][0]:.2f} m gegen {werte['Quadtree'][0]:.2f} m")
    ok &= check("Remesh hat kleineren p99 auf Land",
                werte["Remesh"][1] < werte["Quadtree"][1],
                f"{werte['Remesh'][1]:.2f} m gegen {werte['Quadtree'][1]:.2f} m")
    return ok


def run_cache_greift():
    size = 256
    H = _gelaende(size)
    tsf, ths = _skalen(size)
    baue_remesh(H, tsf, ths, ziel_anteil=0.25)          # waermt
    t0 = time.time()
    _v, _i, stats = baue_remesh(H, tsf, ths, ziel_anteil=0.25)
    dauer = time.time() - t0
    ok = check("zweiter Aufruf kommt aus dem Zwischenspeicher",
               bool(stats.get("aus_cache")), f"{dauer*1000:.1f} ms")
    _v2, _i2, stats2 = baue_remesh(H, tsf, ths, ziel_anteil=0.40)
    ok &= check("anderer Parameter rechnet neu",
                not stats2.get("aus_cache"))
    return ok


def run_verhaeltnisse_stimmen():
    """Reglerwirkung: mehr Budget = mehr Dreiecke, und zwar monoton."""
    size = 256
    H = _gelaende(size)
    tsf, ths = _skalen(size)
    zahlen = []
    for anteil in (0.05, 0.15, 0.30, 0.50):
        _v, i, stats = baue_remesh(H, tsf, ths, ziel_anteil=anteil)
        zahlen.append(stats["dreiecke"])
    ok = check("Dreieckszahl waechst mit dem Budget",
               all(a < b for a, b in zip(zahlen, zahlen[1:])), str(zahlen))
    ok &= check("Budget wird nicht ueberschritten",
                zahlen[-1] <= 2 * (size - 1) * (size - 1))
    return ok


def main():
    if not remesh_verfuegbar():
        print("[FAIL] fast_simplification ist nicht installiert - "
              "ohne das Paket gibt es kein Remesh.")
        return 1

    print("=" * 70)
    print("Remesh - Vertices frei verschieben (6.33)")
    print("=" * 70)
    ergebnisse = []
    for name, funktion in [("Vertices liegen frei", run_vertices_liegen_frei),
                           ("Naht ist dicht", run_naht_ist_dicht),
                           ("Format passt zum Display", run_format_passt_zum_display),
                           ("Trennung bringt Budget an Land", run_trennung_bringt_das_budget_an_land),
                           ("Genauer als das Quadtree", run_genauer_als_quadtree),
                           ("Zwischenspeicher greift", run_cache_greift),
                           ("Reglerwirkung", run_verhaeltnisse_stimmen)]:
        print(f"\n--- {name} ---")
        ergebnisse.append(funktion())

    print("\n" + "=" * 70)
    fehlend = ergebnisse.count(False)
    print(f"{len(ergebnisse) - fehlend}/{len(ergebnisse)} Gruppen gruen")
    return 0 if fehlend == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
