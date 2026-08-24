"""
Path: tests/smoke_test_adaptive_mesh_vectorized.py

Vergleicht die vektorisierte Fassung von gui/widgets/adaptive_terrain_mesh.py
(2026-08-13, OFFENE_PUNKTE 6.18b) gegen eine eingebettete Kopie der ALTEN,
schleifenbasierten Fassung - auf ECHTEN Kartengroessen dieses Projekts.

Warum eine eingebettete Kopie und kein Import: die alte Fassung existiert
nach dem Umbau nicht mehr im Projekt. Sie steht deshalb hier, unveraendert
uebernommen, als Referenz - genau die Rolle, die eine Referenz-Implementierung
in einem Aequivalenztest hat.

Warum echte Groessen: der Vorlaeufer-Fehler bei 6.16 (siehe CLAUDE.md) war,
dass zehn gruene Tests alle mit 129/257/513 gebaut waren - Groessen, die im
Programm nie vorkommen. Dieser Test benutzt 128/256/512, plus 1024 als
optionalen Langlauf.

Verglichen wird die GEOMETRIE, nicht die Index-Nummerierung: die neue Fassung
fasst Vertices ueber numpy.unique() zusammen statt ueber ein Python-dict, die
Vertex-Indizes sind deshalb anders sortiert. Ein Dreieck ist identisch, wenn
seine drei KOORDINATEN-Tripel identisch sind (Reihenfolge innerhalb des
Dreiecks bleibt erhalten - sie bestimmt die Wicklung und damit das
Backface-Culling, siehe adaptive_terrain_mesh.py).
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from gui.widgets.adaptive_terrain_mesh import (
    _quad_fehler,
    _blaetter_sammeln,
    _gepolsterte_hoehen,
    baue_adaptives_mesh_roh,
)


# ---------------------------------------------------------------------------
# ALTE, SCHLEIFENBASIERTE FASSUNG - unveraendert uebernommen als Referenz
# ---------------------------------------------------------------------------

def _alt_besitzer_gitter(blaetter, zellen, min_size):
    besitzer = np.zeros((zellen, zellen), dtype=np.int64)
    for (x0, y0), size in blaetter.items():
        c = size // min_size
        cx, cy = x0 // min_size, y0 // min_size
        besitzer[cy:cy + c, cx:cx + c] = size
    return besitzer


def _alt_blaetter_balancieren(blaetter, N, min_size):
    zellen = N // min_size
    while True:
        besitzer = _alt_besitzer_gitter(blaetter, zellen, min_size)
        neue_blaetter = {}
        musste_teilen = False

        for (x0, y0), size in blaetter.items():
            c = size // min_size
            cx, cy = x0 // min_size, y0 // min_size

            kleinster_nachbar = size
            if cx > 0:
                kleinster_nachbar = min(kleinster_nachbar, int(besitzer[cy:cy + c, cx - 1].min()))
            if cx + c < zellen:
                kleinster_nachbar = min(kleinster_nachbar, int(besitzer[cy:cy + c, cx + c].min()))
            if cy > 0:
                kleinster_nachbar = min(kleinster_nachbar, int(besitzer[cy - 1, cx:cx + c].min()))
            if cy + c < zellen:
                kleinster_nachbar = min(kleinster_nachbar, int(besitzer[cy + c, cx:cx + c].min()))

            if size > min_size and kleinster_nachbar < size // 2:
                musste_teilen = True
                half = size // 2
                xm, ym = x0 + half, y0 + half
                neue_blaetter[(x0, y0)] = half
                neue_blaetter[(xm, y0)] = half
                neue_blaetter[(x0, ym)] = half
                neue_blaetter[(xm, ym)] = half
            else:
                neue_blaetter[(x0, y0)] = size

        blaetter = neue_blaetter
        if not musste_teilen:
            return blaetter


def _alt_dreiecke_aus_blaettern(blaetter, N, min_size):
    zellen = N // min_size
    besitzer = _alt_besitzer_gitter(blaetter, zellen, min_size)

    def nachbar_groesse(cx, cy, c, richtung):
        if richtung == "links":
            return int(besitzer[cy:cy + c, cx - 1].min()) if cx > 0 else None
        if richtung == "rechts":
            return int(besitzer[cy:cy + c, cx + c].min()) if cx + c < zellen else None
        if richtung == "oben":
            return int(besitzer[cy - 1, cx:cx + c].min()) if cy > 0 else None
        return int(besitzer[cy + c, cx:cx + c].min()) if cy + c < zellen else None

    vertex_index = {}
    positionen = []

    def vidx(punkt):
        i = vertex_index.get(punkt)
        if i is None:
            i = len(positionen)
            vertex_index[punkt] = i
            positionen.append(punkt)
        return i

    dreiecke = []
    for (x0, y0), size in blaetter.items():
        c = size // min_size
        cx, cy = x0 // min_size, y0 // min_size
        x1, y1 = x0 + size, y0 + size
        xm, ym = x0 + size // 2, y0 + size // 2

        n_links = nachbar_groesse(cx, cy, c, "links")
        n_rechts = nachbar_groesse(cx, cy, c, "rechts")
        n_oben = nachbar_groesse(cx, cy, c, "oben")
        n_unten = nachbar_groesse(cx, cy, c, "unten")

        polygon = [(x0, y1)]
        if size > min_size and n_unten is not None and n_unten < size:
            polygon.append((xm, y1))
        polygon.append((x1, y1))
        if size > min_size and n_rechts is not None and n_rechts < size:
            polygon.append((x1, ym))
        polygon.append((x1, y0))
        if size > min_size and n_oben is not None and n_oben < size:
            polygon.append((xm, y0))
        polygon.append((x0, y0))
        if size > min_size and n_links is not None and n_links < size:
            polygon.append((x0, ym))

        idxs = [vidx(p) for p in polygon]
        for i in range(1, len(idxs) - 1):
            dreiecke.append((idxs[0], idxs[i], idxs[i + 1]))

    return positionen, dreiecke


def alte_fassung(heightmap, fehler_toleranz_m, min_leaf_size=1):
    """Kompletter alter Pfad bis zu (positionen, dreiecke, blaetter, N).

    Nutzt bewusst die im Modul verbliebenen rekursiven `_quad_fehler()`/
    `_blaetter_sammeln()` (dict-/Rekursions-Fassung) als Referenz - die neue
    Fassung baut denselben Fehlerbaum ebenenweise als numpy-Pyramide
    (`_fehler_pyramide()`/`_blaetter_sammeln_pyramide()`). Dass beide Wege
    dieselbe Blattmenge liefern, ist damit Teil dessen, was dieser Test
    prueft, nicht nur die Balancierung/Triangulierung."""
    H, N = _gepolsterte_hoehen(heightmap)
    cache = {}
    _quad_fehler(H, 0, 0, N, cache)
    blaetter = {}
    _blaetter_sammeln(cache, 0, 0, N, fehler_toleranz_m, min_leaf_size, blaetter)
    blaetter = _alt_blaetter_balancieren(blaetter, N, min_leaf_size)
    positionen, dreiecke = _alt_dreiecke_aus_blaettern(blaetter, N, min_leaf_size)
    return positionen, dreiecke, blaetter, N


# ---------------------------------------------------------------------------
# VERGLEICH
# ---------------------------------------------------------------------------

def als_koordinaten_dreiecke(positionen, dreiecke):
    """Dreiecke als Koordinaten-Tripel statt Index-Tripel - macht den Vergleich
    unabhaengig von der Vertex-Nummerierung (die sich bewusst geaendert hat)."""
    return {tuple(positionen[i] for i in tri) for tri in dreiecke}


def realistisches_gelaende(size, seed):
    """Geglaettetes Rauschen plus eine scharfe Klippe - dieselbe Bauart wie in
    smoke_test_adaptive_terrain_mesh.py, damit die Blattzahl in derselben
    Groessenordnung liegt wie bei echtem Weltgelaende."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.RandomState(seed)
    roh = rng.uniform(0, 1, size=(size, size)).astype(np.float32)
    H = gaussian_filter(roh, sigma=max(2.0, size / 64.0)) * 1000.0
    viertel = size // 4
    H[viertel:viertel + size // 8, viertel:viertel + size // 8] += 400.0
    return H


def vergleiche(size, toleranz, seed):
    H = realistisches_gelaende(size, seed)

    t0 = time.time()
    alt_pos, alt_tri, alt_blaetter, alt_N = alte_fassung(H, toleranz)
    t_alt = time.time() - t0

    t0 = time.time()
    neu = baue_adaptives_mesh_roh(H, toleranz)
    t_neu = time.time() - t0
    assert neu is not None, f"{size}px: neue Fassung lieferte None"
    neu_pos, neu_tri, neu_blaetter, neu_N = neu

    ok = True

    if alt_N != neu_N:
        print(f"  FAIL N: alt {alt_N} vs neu {neu_N}")
        ok = False

    if alt_blaetter != neu_blaetter:
        nur_alt = set(alt_blaetter.items()) - set(neu_blaetter.items())
        nur_neu = set(neu_blaetter.items()) - set(alt_blaetter.items())
        print(f"  FAIL Blaetter: {len(nur_alt)} nur alt, {len(nur_neu)} nur neu, "
              f"Beispiele alt={list(nur_alt)[:3]} neu={list(nur_neu)[:3]}")
        ok = False

    alt_geo = als_koordinaten_dreiecke(alt_pos, alt_tri)
    neu_geo = als_koordinaten_dreiecke(neu_pos, neu_tri)

    if len(alt_tri) != len(alt_geo):
        print(f"  WARN: alte Fassung hat {len(alt_tri) - len(alt_geo)} doppelte Dreiecke")
    if len(neu_tri) != len(neu_geo):
        print(f"  WARN: neue Fassung hat {len(neu_tri) - len(neu_geo)} doppelte Dreiecke")

    if alt_geo != neu_geo:
        nur_alt = alt_geo - neu_geo
        nur_neu = neu_geo - alt_geo
        print(f"  FAIL Dreiecke: {len(nur_alt)} nur alt, {len(nur_neu)} nur neu")
        print(f"    Beispiel nur alt: {list(nur_alt)[:2]}")
        print(f"    Beispiel nur neu: {list(nur_neu)[:2]}")
        ok = False

    if set(alt_pos) != set(neu_pos):
        print(f"  FAIL Vertices: alt {len(alt_pos)}, neu {len(neu_pos)}, "
              f"Schnittmenge {len(set(alt_pos) & set(neu_pos))}")
        ok = False

    status = "OK  " if ok else "FAIL"
    print(f"[{status}] {size}px tol={toleranz}: {len(neu_blaetter)} Blaetter, "
          f"{len(neu_tri)} Dreiecke | alt {t_alt:.2f}s -> neu {t_neu:.2f}s "
          f"({t_alt / max(t_neu, 1e-9):.1f}x)")
    return ok, t_alt, t_neu


if __name__ == "__main__":
    # ECHTE map_size-Werte dieses Projekts, mehrere Toleranzen (die Toleranz
    # steuert die Blattzahl und damit, wie stark die Balancierung arbeiten muss).
    # Die Toleranz steuert die Blattzahl. Der 1024px-Fall mit tol=0.5 stellt
    # das ECHTE Szenario aus dem Nutzerlog vom 2026-08-13 nach (194068
    # Blaetter, 15.7 s Netzaufbau) - glattes Testgelaende allein ergibt nur
    # rund 3000 Blaetter und wuerde damit genau den Fall verfehlen, der im
    # Programm langsam war.
    faelle = [
        (128, 6.0, 1),
        (128, 1.0, 2),
        (256, 6.0, 3),
        (256, 2.0, 4),
        (512, 6.0, 5),
        (512, 2.0, 6),
        (1024, 6.0, 7),
        (1024, 0.5, 8),
    ]

    alle_ok = True
    summe_alt = 0.0
    summe_neu = 0.0
    for size, toleranz, seed in faelle:
        ok, t_alt, t_neu = vergleiche(size, toleranz, seed)
        alle_ok &= ok
        summe_alt += t_alt
        summe_neu += t_neu

    print(f"\nGesamt: alt {summe_alt:.2f}s -> neu {summe_neu:.2f}s "
          f"({summe_alt / max(summe_neu, 1e-9):.1f}x schneller)")
    print("Alle Faelle bit-identisch." if alle_ok else "ABWEICHUNGEN GEFUNDEN.")
    sys.exit(0 if alle_ok else 1)
