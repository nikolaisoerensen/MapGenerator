"""
Path: gui/widgets/terrain_remesh.py

ECHTES REMESH - Vertices, die NICHT auf Pixelecken sitzen.

Nutzervorgabe 2026-08-16: *"wenn wir die treppen loswerden wollen muessen wir
die vertices verschieben."* Genau das leistet dieses Modul, und das adaptive
Quadtree (adaptive_terrain_mesh.py) leistet es prinzipiell nicht - dessen
Vertices liegen gemessen 0.000004 px von einer Pixelecke entfernt, also
exakt darauf, und deshalb folgt die Kuestensilhouette dem Raster.

VON OBEN STATT VON UNTEN

Alle frueheren Versuche gingen von unten: grob anfangen und Punkte einfuegen
(Delatin/greedy insertion, siehe delatin_mesh.py, docs/OFFENE_PUNKTE.md 6.30).
Sie scheitern daran, dass inkrementelles Delaunay in reinem Python nicht
bezahlbar ist - der noetige Stapel-Kompromiss kostete am Ende Faktor 5 an
Genauigkeit.

Hier wird umgekehrt vorgegangen: Start ist das VOLLE Gitter (bei 512 px also
262144 Vertices - die feinste Darstellung, die es gibt, ganz ohne Treppen),
dann fallen Kanten zusammen nach der Quadric Error Metric (Garland & Heckbert
1997). Der verschmolzene Vertex landet dort, wo der Quadric-Fehler minimal
ist - also frei im Raum, nicht auf einer Pixelecke. An einer Klippe kostet
das Zusammenfallen quer zur Kante viel, die Kante bleibt also stehen und die
Dreiecksseiten legen sich von selbst an sie an.

Gerechnet wird von `fast_simplification` (MIT, C++), gemessen am 2026-08-16
bei gleicher Vertexzahl gegen das Quadtree: Median-Abstand zur Pixelecke
0.296685 px statt 0.000000 px, 80.5 % der Vertices frei verschoben.

DAS MEER BEKOMMT EIN EIGENES BUDGET

`fast_simplification` ist ein Schwellendurchlauf, keine
Prioritaetswarteschlange (`Simplify.h`: `threshold = 1e-9*(iteration+3)^agg`,
`if (t.err[3] > threshold) continue;`). Es verteilt das Budget deshalb grob
gleichmaessig ueber die Flaeche - und weil das Meer 71 % der Karte ausmacht,
landete dort mehr als die Haelfte aller Vertices, bei einer Genauigkeit von
0.42 m, die unter Wasser niemand sieht.

Der Versuch, das ueber eine gestauchte Hoehenachse unter Wasser zu steuern,
schlug messbar fehl (Landanteil 43.7 -> 43.5 % bei Stauchung bis 0.03) - eben
weil die Schwelle nicht nach Fehlergroesse sortiert.

Was wirkt: das Gelaende an einer Tiefenlinie in zwei Teilnetze zerlegen und
JEDES FUER SICH dezimieren, mit `preserve_border=True`. Die Trennlinie ist
dann fuer beide Teile Rand, beide behalten dort exakt dieselben Vertices, die
Naht ist dicht - kein T-Stueck, kein Riss (die Falle, an der das Klippenband
gescheitert war). Gemessen bei gleicher Vertexzahl:

    Quadtree            Land RMS 1.59 m   p99 6.05 m   Meer RMS 2.01 m
    QEM ohne Zerlegung  Land RMS 2.33 m   p99 7.81 m   Meer RMS 0.42 m
    QEM, Meer 10 %      Land RMS 1.07 m   p99 3.51 m   Meer RMS 0.94 m

Mit Zerlegung ist es auf Land UND im Meer zugleich besser als das Quadtree.

Preis, ehrlich benannt: die Vertices auf der Trennlinie bleiben rastergebunden
(`preserve_border` haelt sie fest), der Anteil frei verschobener Vertices
sinkt von 80.5 auf 54.2 %. Die Trennlinie liegt aber unter Wasser; die
eigentliche Kuestenlinie bei 0 m liegt im feinen Teil und wird frei
verschoben.
"""

import numpy as np

from gui.widgets.adaptive_terrain_mesh import _normalen_voll


# Ab dieser Tiefe gilt es als "offenes Meer" und bekommt das kleine Budget.
# Nicht 0 m: der Schelf direkt vor der Kueste ist durch flaches Wasser
# sichtbar und gehoert zum feinen Teil.
TIEFENLINIE_M = -25.0

# Anteil des Dreiecksbudgets fuer das offene Meer. 0.10 ist der gemessene
# Punkt, an dem Land UND Meer besser sind als beim Quadtree; darunter kippt
# der Meeresfehler (bei 0.05 auf 2.63 m, schlechter als das Quadtree).
MEER_BUDGET_ANTEIL = 0.10

# Steuert, wie schnell die Schwelle in `fast_simplification` waechst.
# 7.0 ist die Vorgabe der Bibliothek; 0 waere langsam und formtreu.
AGGRESSIVITAET = 7.0

# Wie beim Quadtree ueber den INHALT zwischengespeichert - alle Reiter teilen
# sich dasselbe Netz, statt es je Reiter neu zu bauen.
_CACHE = {}
_CACHE_MAX = 3


def remesh_verfuegbar():
    """
    Ist `fast_simplification` installiert?

    Getrennte Funktion, damit der Aufrufer die Antwort BEKOMMT statt sie aus
    einem `None` zu erraten - ein stiller Rueckfall auf das Quadtree waere von
    Erfolg nicht zu unterscheiden (dieselbe Falle wie bei den GPU-Fallbacks
    und der 2^n+1-Bedingung, siehe CLAUDE.md).
    """
    try:
        import fast_simplification  # noqa: F401
        return True
    except ImportError:
        return False


def _volles_gitter(heightmap, terrain_scale_factor, terrain_height_scale):
    """Jeder Pixel ein Vertex - Ausgangspunkt der Dezimierung."""
    hoehe, breite = heightmap.shape
    ys, xs = np.mgrid[0:hoehe, 0:breite]
    pos_x = (xs / (breite - 1) - 0.5) * breite * terrain_scale_factor
    pos_z = (ys / (hoehe - 1) - 0.5) * hoehe * terrain_scale_factor
    pos_y = heightmap * terrain_height_scale
    punkte = np.stack([pos_x.ravel(), pos_y.ravel(), pos_z.ravel()],
                      axis=1).astype(np.float64)

    i = np.arange(hoehe * breite).reshape(hoehe, breite)
    oben_links, oben_rechts = i[:-1, :-1].ravel(), i[:-1, 1:].ravel()
    unten_links, unten_rechts = i[1:, :-1].ravel(), i[1:, 1:].ravel()
    # Wicklung wie das Gleichmaessig-Gitter in map_display_3d.py:
    # (oben_links, unten_links, oben_rechts) und (oben_rechts, unten_links, unten_rechts)
    dreiecke = np.concatenate([
        np.stack([oben_links, unten_links, oben_rechts], axis=1),
        np.stack([oben_rechts, unten_links, unten_rechts], axis=1)]).astype(np.int32)
    return punkte, dreiecke


def _teilnetz(dreiecke, behalten):
    """Dreiecksauswahl zu einem eigenstaendigen Netz umindizieren."""
    ausgewaehlt = dreiecke[behalten]
    benutzt = np.unique(ausgewaehlt)
    um = np.full(int(benutzt.max()) + 1, -1, np.int64)
    um[benutzt] = np.arange(len(benutzt))
    return benutzt, um[ausgewaehlt].astype(np.int32)


def _zusammenfuegen(teile):
    """
    Teilnetze zu einem Netz vereinen und die Naht schliessen.

    Die Randvertices sind in beiden Teilen KOORDINATENGLEICH (dafuer sorgt
    `preserve_border`), also lassen sie sich ueber die Koordinate
    zusammenfuehren. Ohne diesen Schritt laege an der Naht eine doppelte
    Vertexreihe - unsichtbar im Bild, aber jede spaetere Kantenoperation
    haette dort ein Loch gesehen.
    """
    alle = np.concatenate([p for p, _f in teile])
    _einmalig, erste, zurueck = np.unique(np.round(alle, 9), axis=0,
                                          return_index=True, return_inverse=True)
    punkte = alle[erste]
    versatz = 0
    umgeschrieben = []
    for p, f in teile:
        umgeschrieben.append(zurueck[f + versatz])
        versatz += len(p)
    return punkte, np.concatenate(umgeschrieben)


def remesh_roh(heightmap, terrain_scale_factor, terrain_height_scale,
               ziel_dreiecke, meer_anteil=MEER_BUDGET_ANTEIL,
               tiefenlinie_m=TIEFENLINIE_M, aggressivitaet=AGGRESSIVITAET,
               zerlegen=True):
    """
    Die Dezimierung selbst. Rueckgabe (punkte (N,3) Welt, dreiecke (M,3)).

    `zerlegen=False` dezimiert ohne Meer/Land-Trennung - nur fuer den
    Vergleich in der Werkstatt gedacht, im Betrieb ist die Trennung besser.
    """
    import fast_simplification as fs

    H = np.asarray(heightmap, dtype=np.float64)
    punkte, dreiecke = _volles_gitter(H, terrain_scale_factor, terrain_height_scale)

    if not zerlegen:
        p, f = fs.simplify(punkte, dreiecke, target_count=int(ziel_dreiecke),
                           agg=aggressivitaet)
        return np.asarray(p, np.float64), np.asarray(f, np.int64)

    tief = (H <= tiefenlinie_m).ravel()
    ist_meer = tief[dreiecke].all(axis=1)

    # Liegt alles auf einer Seite, ist die Zerlegung sinnlos - dann in einem
    # Stueck rechnen, statt ein leeres Teilnetz zu bauen.
    if ist_meer.all() or not ist_meer.any():
        p, f = fs.simplify(punkte, dreiecke, target_count=int(ziel_dreiecke),
                           agg=aggressivitaet)
        return np.asarray(p, np.float64), np.asarray(f, np.int64)

    d_meer = max(int(ziel_dreiecke * meer_anteil), 100)
    d_land = max(int(ziel_dreiecke) - d_meer, 100)

    teile = []
    for maske, ziel in ((ist_meer, d_meer), (~ist_meer, d_land)):
        idx, f_teil = _teilnetz(dreiecke, maske)
        p_teil = punkte[idx]
        if len(f_teil) > ziel:
            p_teil, f_teil = fs.simplify(p_teil, f_teil, target_count=ziel,
                                         agg=aggressivitaet, preserve_border=True)
        teile.append((np.asarray(p_teil, np.float64), np.asarray(f_teil, np.int64)))
    return _zusammenfuegen(teile)


def baue_remesh(heightmap, terrain_scale_factor, terrain_height_scale,
                ziel_dreiecke=None, ziel_anteil=0.25,
                meer_anteil=MEER_BUDGET_ANTEIL, tiefenlinie_m=TIEFENLINIE_M,
                aggressivitaet=AGGRESSIVITAET, zerlegen=True):
    """
    Terrain-Mesh mit frei verschobenen Vertices.

    Rueckgabe wie `adaptive_terrain_mesh.build_adaptive_mesh()`:
    (vertices float32 interleaved [x,y,z,nx,ny,nz,u,v], indices uint32,
    stats dict) - damit es an derselben Stelle einsetzbar ist. `None`, wenn
    `fast_simplification` fehlt.

    `ziel_dreiecke` absolut, sonst `ziel_anteil` als Bruchteil des vollen
    Gitters (0.25 = ein Viertel der 2*(w-1)*(h-1) Dreiecke).

    Die Normalen werden aus dem HOEHENFELD bilinear an der Vertexposition
    abgetastet, nicht aus den Dreiecken gemittelt. Sonst saehe das Remesh
    facettiert aus, waehrend das Quadtree-Mesh glatt schattiert ist - der
    Vergleich waere dann nicht der zwischen den Netzen, sondern der zwischen
    zwei Beleuchtungen.
    """
    if not remesh_verfuegbar():
        return None

    H = np.asarray(heightmap, dtype=np.float32)
    hoehe_px, breite_px = H.shape
    voll_dreiecke = 2 * (breite_px - 1) * (hoehe_px - 1)
    if ziel_dreiecke is None:
        ziel_dreiecke = max(int(voll_dreiecke * ziel_anteil), 200)
    ziel_dreiecke = int(min(ziel_dreiecke, voll_dreiecke))

    schluessel = (H.tobytes(), H.shape, float(terrain_scale_factor),
                  float(terrain_height_scale), ziel_dreiecke, float(meer_anteil),
                  float(tiefenlinie_m), float(aggressivitaet), bool(zerlegen))
    treffer = _CACHE.get(schluessel)
    if treffer is not None:
        v, i, s = treffer
        return v, i, dict(s, aus_cache=True)

    punkte, dreiecke = remesh_roh(H, terrain_scale_factor, terrain_height_scale,
                                  ziel_dreiecke, meer_anteil, tiefenlinie_m,
                                  aggressivitaet, zerlegen)
    if len(punkte) < 3 or len(dreiecke) < 1:
        return None

    # Weltkoordinaten zurueck ins Pixelraster fuer Normalen und Texturkoordinaten
    px = (punkte[:, 0] / (breite_px * terrain_scale_factor) + 0.5) * (breite_px - 1)
    py = (punkte[:, 2] / (hoehe_px * terrain_scale_factor) + 0.5) * (hoehe_px - 1)
    px = np.clip(px, 0.0, breite_px - 1)
    py = np.clip(py, 0.0, hoehe_px - 1)

    nx_feld, ny_feld, nz_feld = _normalen_voll(H, terrain_height_scale,
                                               terrain_scale_factor)
    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    x1 = np.minimum(x0 + 1, breite_px - 1)
    y1 = np.minimum(y0 + 1, hoehe_px - 1)
    fx = (px - x0)[:, None]
    fy = (py - y0)[:, None]

    def _bilinear(feld):
        oben = feld[y0, x0] * (1 - fx[:, 0]) + feld[y0, x1] * fx[:, 0]
        unten = feld[y1, x0] * (1 - fx[:, 0]) + feld[y1, x1] * fx[:, 0]
        return oben * (1 - fy[:, 0]) + unten * fy[:, 0]

    nx = _bilinear(nx_feld)
    ny = _bilinear(ny_feld)
    nz = _bilinear(nz_feld)
    laenge = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    sicher = np.where(laenge > 0, laenge, 1.0)
    nx, ny, nz = nx / sicher, ny / sicher, nz / sicher

    vertices = np.stack([punkte[:, 0], punkte[:, 1], punkte[:, 2],
                         nx, ny, nz,
                         px / (breite_px - 1), py / (hoehe_px - 1)],
                        axis=-1).astype(np.float32)

    abstand = np.hypot(px - np.round(px), py - np.round(py))
    stats = {"vertices": len(punkte),
             "dreiecke": len(dreiecke),
             "voll_dreiecke": voll_dreiecke,
             "voll_vertices": breite_px * hoehe_px,
             "frei_verschoben": float((abstand > 0.01).mean()),
             "versatz_median_px": float(np.median(abstand)),
             "aus_cache": False}

    ergebnis = (vertices.reshape(-1), dreiecke.reshape(-1).astype(np.uint32), stats)
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[schluessel] = ergebnis
    return ergebnis
