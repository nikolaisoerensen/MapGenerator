"""
Path: gui/widgets/delatin_mesh.py

Fehler-getriebenes Terrain-Remesh nach Garland & Heckbert (1995), "Fast
Polygonal Approximation of Terrains and Height Fields" - derselbe Algorithmus,
den `hmm`/`pydelatin` implementieren (docs/OFFENE_PUNKTE.md 6.30).

WARUM SELBST GEBAUT

`pip install pydelatin` scheitert auf diesem Rechner: das Paket bringt kein
Windows-Wheel mit und verlangt zum Bauen "Microsoft Visual C++ 14.0 or
greater". Das ungefragt nachzuinstallieren waere ein Eingriff in die
Arbeitsumgebung des Nutzers. Der Kern des Verfahrens ist ueberschaubar, also
hier direkt.

DER UNTERSCHIED ZUM QUADTREE (6.16)

Der Quadtree kann nur RASTERZELLEN zusammenfassen - jeder Vertex bleibt auf
einer Pixelecke (gemessen 0.000004 px Abweichung), und deshalb folgt die
Kuestensilhouette dem Raster und wird als Treppe sichtbar.

Dieses Verfahren waehlt stattdessen PUNKTE aus: es beginnt mit zwei Dreiecken
ueber die ganze Karte und fuegt immer wieder den Punkt mit dem groessten
Hoehenfehler ein. Die Vertices landen dadurch von selbst dort, wo das
Gelaende sie braucht - an Graten, Klippen und der Wasserlinie, weil dort die
Abweichung zwischen Dreieck und Wirklichkeit am groessten ist.

BATCHWEISE STATT EINZELN

Das Original fuegt EINEN Punkt ein und trianguliert lokal nach. Eine lokale
Delaunay-Insertion ist ohne C++-Bibliothek aufwendig; `scipy.spatial.Delaunay`
kann nur komplett neu rechnen. Deshalb werden hier je Runde MEHRERE Punkte
auf einmal eingefuegt (die schlechtesten, mit Mindestabstand zueinander) und
danach einmal neu trianguliert. Das Ergebnis ist nicht bitgleich mit dem
Original, verfolgt aber dieselbe Zielgroesse - und ist messbar (siehe
`kuestentreue()` in kuesten_mesh.py sowie die Fehlerauswertung im Test).
"""

import numpy as np


# Wie viele Punkte je Runde eingefuegt werden, als Anteil der bereits
# vorhandenen. Gross = wenige Runden, aber groebere Auswahl; klein = naeher
# am Original, aber mehr Triangulierungen.
BATCH_ANTEIL = 0.5

# Mindestabstand zwischen zwei in DERSELBEN Runde eingefuegten Punkten, in
# Pixeln. Ohne ihn landet die halbe Runde auf demselben Grat, weil dort alle
# Fehler gleichzeitig gross sind.
BATCH_MINDESTABSTAND_PX = 2.0


def _fehlerfeld(punkte, z, dreiecke, H, stichprobe=1):
    """
    Je Rasterpixel: |interpolierte Hoehe - echte Hoehe|, und zu welchem
    Dreieck es gehoert. NaN ausserhalb der Triangulierung.
    """
    from scipy.spatial import Delaunay
    hoehe, breite = H.shape
    ys, xs = np.mgrid[0:hoehe:stichprobe, 0:breite:stichprobe]
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)

    tri = Delaunay(punkte)
    tri.simplices  # sicherstellen, dass sie gebaut ist
    s = tri.find_simplex(pts)
    gut = s >= 0
    fehler = np.full(len(pts), np.nan)
    if not gut.any():
        return fehler.reshape(xs.shape), pts, s, tri

    b = tri.transform[s[gut], :2].transpose(1, 0, 2)
    d = pts[gut] - tri.transform[s[gut], 2]
    bary = np.einsum('ijk,ik->ij', b.transpose(1, 0, 2) if b.shape[0] == 2 else b, d)
    bary = np.c_[bary, 1 - bary.sum(axis=1)]
    ecken = tri.simplices[s[gut]]
    interp = (z[ecken] * bary).sum(axis=1)
    echt = H[pts[gut, 1].astype(np.int32), pts[gut, 0].astype(np.int32)]
    fehler[gut] = np.abs(interp - echt)
    return fehler.reshape(xs.shape), pts, s, tri


def waehle_punkte(H, max_fehler_m=None, max_punkte=None, stichprobe=1,
                  batch_anteil=BATCH_ANTEIL,
                  mindestabstand=BATCH_MINDESTABSTAND_PX,
                  bericht=None, zeitgrenze_s=None, fortschritt=None):
    """
    Die Punktauswahl nach Garland & Heckbert: immer dort verdichten, wo die
    Abweichung am groessten ist.

    Genau EINE der beiden Grenzen sollte gesetzt sein - `max_fehler_m` ("so
    genau will ich es haben") oder `max_punkte` ("so viel darf es kosten").

    `stichprobe` ist nur ein Rechenspar-Raster fuer die Fehlersuche, KEINE
    Grenze fuer das Ergebnis: laeuft es leer, wird es hier automatisch
    verfeinert (siehe unten). `mindestabstand` wird mitverkleinert.

    `bericht` (optionales dict) wird mit dem Abbruchgrund gefuellt - ohne das
    ist "hat 28479 statt 65954 Punkte geliefert" von "war fertig" nicht zu
    unterscheiden.

    `zeitgrenze_s` bricht nach so vielen Sekunden mit dem ab, was da ist.
    DAS BRAUCHT DIESES VERFAHREN WIRKLICH: jede Runde rechnet eine komplette
    Delaunay-Triangulierung neu, gemessen 126 s fuer 65954 Punkte bei 512 px.
    In `tools/mesh_werkstatt.py` fror das Fenster dadurch minutenlang ohne
    jede Rueckmeldung ein - es sah aus wie ein Absturz.

    `fortschritt(runde, punkte, groesster_fehler_m)` wird nach jeder Runde
    gerufen, damit ein Aufrufer anzeigen kann, dass ueberhaupt etwas passiert.

    Rueckgabe: (P,2)-Array der gewaehlten Punkte in (x, y).
    """
    import time

    from scipy.spatial import cKDTree

    beginn = time.time()
    runde = 0

    H = np.asarray(H, dtype=np.float64)
    hoehe, breite = H.shape
    # Start: die vier Kartenecken, also zwei Dreiecke ueber alles.
    punkte = np.array([[0, 0], [breite - 1, 0],
                       [0, hoehe - 1], [breite - 1, hoehe - 1]], dtype=np.float64)

    if max_punkte is None:
        max_punkte = 200000
    if max_fehler_m is None:
        max_fehler_m = 0.0

    grund = "max_punkte erreicht"
    letzte_runde_s = 0.0
    while len(punkte) < max_punkte:
        # VORAUSSCHAUEND pruefen, nicht nur rueckblickend. Die Runden werden
        # immer teurer (jede rechnet eine Delaunay ueber mehr Punkte), deshalb
        # ueberzieht ein rein rueckblickender Test um eine ganze Runde -
        # gemessen 27.4 s bei einer Grenze von 20 s. Mit der Schaetzung aus
        # der letzten Rundendauer bleibt es dicht an der Vorgabe.
        vergangen = time.time() - beginn
        if zeitgrenze_s is not None and (vergangen + letzte_runde_s) > zeitgrenze_s:
            grund = (f"Zeitgrenze {zeitgrenze_s:.0f} s erreicht bei "
                     f"{len(punkte)} von {max_punkte} Punkten")
            break
        runde += 1
        runde_beginn = time.time()
        z = H[np.clip(punkte[:, 1].astype(np.int32), 0, hoehe - 1),
              np.clip(punkte[:, 0].astype(np.int32), 0, breite - 1)]
        feld, pts, _s, _tri = _fehlerfeld(punkte, z, None, H, stichprobe)
        flach = feld.ravel()
        gueltig = np.isfinite(flach)
        if not gueltig.any():
            grund = "kein Punkt innerhalb der Triangulierung"
            break
        groesster = float(np.nanmax(flach))
        if groesster <= max_fehler_m:
            grund = f"Fehlergrenze erreicht (max {groesster:.2f} m)"
            break

        # Die schlechtesten Kandidaten dieser Runde, absteigend
        wieviele = max(1, int(len(punkte) * batch_anteil))
        wieviele = min(wieviele, max_punkte - len(punkte))
        ordnung = np.argsort(np.where(gueltig, -flach, np.inf))
        baum = cKDTree(punkte)
        neu = []
        for k in ordnung[:wieviele * 20]:
            if len(neu) >= wieviele:
                break
            if not gueltig[k] or flach[k] <= max_fehler_m:
                break
            kandidat = pts[k]
            # Nicht zu nah an vorhandenen Punkten...
            if baum.query(kandidat)[0] < mindestabstand:
                continue
            # ...und nicht zu nah an denen dieser Runde (sonst landet alles
            # auf demselben Grat, wo die Fehler gleichzeitig gross sind).
            if neu and min(np.hypot(*(kandidat - np.array(neu)).T)) < mindestabstand:
                continue
            neu.append(kandidat)

        letzte_runde_s = time.time() - runde_beginn
        if fortschritt is not None:
            fortschritt(runde, len(punkte) + len(neu), groesster)

        if not neu:
            # Das Kandidatenraster ist erschoepft: bei `stichprobe`=s gibt es
            # nur (H/s)*(W/s) moegliche Positionen, und die belegten fallen
            # laufend weg. Gemessen am 2026-08-16: mit s=3 auf 512x512 war bei
            # 28479 von angeforderten 65954 Punkten Schluss - die Funktion gab
            # das kleinere Netz kommentarlos zurueck und sah dabei aus wie ein
            # fertiges Ergebnis. Also verfeinern statt aufgeben.
            if stichprobe > 1:
                stichprobe = max(1, stichprobe // 2)
                mindestabstand = max(1.0, mindestabstand * 0.5)
                continue
            if mindestabstand > 1.0:
                mindestabstand = 1.0
                continue
            grund = "keine freien Kandidatenpositionen mehr"
            break
        punkte = np.concatenate([punkte, np.array(neu)])

    if bericht is not None:
        bericht["abbruch"] = grund
        bericht["stichprobe_ende"] = stichprobe
        bericht["punkte"] = len(punkte)
    return punkte


def baue_delatin_mesh(heightmap, terrain_scale_factor, terrain_height_scale,
                      max_fehler_m=None, max_punkte=None, stichprobe=2,
                      zeitgrenze_s=None, fortschritt=None):
    """
    Terrain-Mesh mit frei gesetzten Vertices.

    Rueckgabe wie `adaptive_terrain_mesh.build_adaptive_mesh()`:
    (vertices float32 interleaved [x,y,z,nx,ny,nz,u,v], indices uint32,
    stats dict) - damit es an derselben Stelle einsetzbar ist.
    """
    from scipy.spatial import Delaunay
    from gui.widgets.adaptive_terrain_mesh import _normalen_voll

    H = np.asarray(heightmap, dtype=np.float32)
    hoehe_px, breite_px = H.shape
    bericht = {}
    punkte = waehle_punkte(H, max_fehler_m=max_fehler_m,
                           max_punkte=max_punkte, stichprobe=stichprobe,
                           bericht=bericht, zeitgrenze_s=zeitgrenze_s,
                           fortschritt=fortschritt)
    if len(punkte) < 4:
        return None

    tri = Delaunay(punkte)
    dreiecke = tri.simplices
    xs, ys = punkte[:, 0], punkte[:, 1]
    xi = np.clip(np.round(xs).astype(np.int32), 0, breite_px - 1)
    yi = np.clip(np.round(ys).astype(np.int32), 0, hoehe_px - 1)
    z = H[yi, xi].astype(np.float64)

    pos_x = (np.clip(xs, 0, breite_px - 1) / (breite_px - 1) - 0.5) * breite_px * terrain_scale_factor
    pos_z = (np.clip(ys, 0, hoehe_px - 1) / (hoehe_px - 1) - 0.5) * hoehe_px * terrain_scale_factor
    pos_y = z * terrain_height_scale

    nx, ny, nz = _normalen_voll(H, terrain_height_scale, terrain_scale_factor)
    tex_u = np.clip(xs, 0, breite_px - 1) / (breite_px - 1)
    tex_v = np.clip(ys, 0, hoehe_px - 1) / (hoehe_px - 1)

    vertex_array = np.stack([pos_x, pos_y, pos_z,
                              nx[yi, xi], ny[yi, xi], nz[yi, xi],
                              tex_u, tex_v], axis=-1).astype(np.float32)

    # Wicklung wie beim Quadtree-Mesh - Delaunay liefert gegen den
    # Uhrzeigersinn in (x,y), die Anzeige nutzt (x,z) mit Backface-Culling.
    dreiecke = dreiecke[:, ::-1]

    stats = {"vertices": len(punkte),
             "dreiecke": len(dreiecke),
             "voll_dreiecke": 2 * (breite_px - 1) * (hoehe_px - 1),
             "voll_vertices": breite_px * hoehe_px,
             "abbruch": bericht.get("abbruch"),
             "stichprobe_ende": bericht.get("stichprobe_ende")}
    return vertex_array.reshape(-1), dreiecke.reshape(-1).astype(np.uint32), stats
