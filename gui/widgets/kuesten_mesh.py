"""
Path: gui/widgets/kuesten_mesh.py

Terrain-Vernetzung mit FREI GESETZTEN Punkten entlang der Kuestenlinie
statt eines reinen Rastergitters (docs/OFFENE_PUNKTE.md 6.20, Nutzeridee:
"kann man das mesh nicht aus dem hoehenmodell erstellen und dann in ein mesh
umwandeln das mit dem hoehenmodell nichts mehr gemein hat ... ein remesh kann
auf jeder tangente passieren").

DAS PROBLEM

Das adaptive Quadtree-Mesh (6.16) reduziert die Dreieckszahl, aber **jeder
Vertex sitzt weiterhin auf einer Pixelecke** - nachgemessen liegt die
Abweichung bei 0.000004. Die Kuestensilhouette folgt deshalb dem Raster und
wird als TREPPE sichtbar. Das laesst sich mit einem Quadtree nicht beheben,
egal wie fein er unterteilt: eine Stufe im Raster bleibt eine Stufe.

DER ANSATZ

1. **Kuestenlinie als Kontur ziehen** (`skimage.measure.find_contours` auf
   der 0-Linie, also Marching Squares). Das ergibt einen Linienzug mit
   Zwischenwerten - die Punkte liegen NICHT auf Pixelecken, sondern dort, wo
   die Hoehe tatsaechlich durch null geht.
2. **Linienzug ausduennen** (Douglas-Peucker), sonst hat man mehr
   Kuestenpunkte als das ganze bisherige Mesh Vertices hatte.
3. **Delaunay ueber Kuestenpunkte + ein gestreutes Innengitter.**

WARUM KEIN "CONSTRAINED" DELAUNAY

Sauber waere eine Triangulierung, die die Kuestenkanten ERZWINGT (Constrained
Delaunay / "TIN mit Breaklines"). Dafuer gibt es in dieser Umgebung keine
Bibliothek: `scipy.spatial.Delaunay` kann nur unconstrained, `triangle` ist
nicht installiert (geprueft).

Der Ausweg ohne neue Abhaengigkeit: die Kuestenpunkte DICHTER setzen als die
umgebenden Rasterpunkte. Delaunay verbindet dann bevorzugt Nachbarn auf der
Kontur, weil sie naeher beieinander liegen als alles andere - die Kuestenkante
entsteht praktisch von selbst. **Das ist eine Naeherung, keine Garantie**, und
genau deshalb misst `kuestentreue()` unten nach, wie viele Konturkanten
tatsaechlich als Dreieckskanten vorkommen. Ohne diese Messung waere der
Unterschied zum echten Verfahren nicht zu bemerken.
"""

import numpy as np


# Wie stark der Kuesten-Linienzug ausgeduennt wird, in PIXELN. Punkte, die
# weniger als das von der Verbindungslinie ihrer Nachbarn abweichen, fallen
# weg. Klein = treuer, aber mehr Vertices.
KUESTE_TOLERANZ_PX = 0.35

# Abstand der Kuestenpunkte zueinander nach dem Ausduennen, in Pixeln. Sie
# muessen DICHTER liegen als das Innengitter, sonst greift der Trick oben
# nicht (siehe Modulkopf).
KUESTE_ABSTAND_PX = 1.2

# Kantenlaenge des gestreuten Innengitters in Pixeln. Groesser = weniger
# Dreiecke, aber groebere Gelaendeform abseits der Kueste.
INNEN_ABSTAND_PX = 6.0


def kuestenlinien(heightmap, pegel=0.0):
    """
    Die Kuestenlinie(n) als Liste von (M,2)-Linienzuegen in (x, y).

    `skimage.measure.find_contours` arbeitet mit Marching Squares und
    interpoliert LINEAR zwischen den Rasterwerten - die zurueckgegebenen
    Punkte liegen daher zwischen den Pixeln, genau dort, wo die Hoehe durch
    `pegel` geht. Das ist der eigentliche Gewinn gegenueber jedem
    Rasterverfahren.
    """
    from skimage import measure

    H = np.asarray(heightmap, dtype=np.float64)
    linien = []
    for kontur in measure.find_contours(H, pegel):
        # find_contours liefert (zeile, spalte) = (y, x) - gedreht, damit es
        # zur uebrigen (x, y)-Konvention des Projekts passt.
        linien.append(np.stack([kontur[:, 1], kontur[:, 0]], axis=1))
    return linien


def _douglas_peucker(punkte, toleranz):
    """Linienzug ausduennen - iterativ statt rekursiv, damit eine lange
    Kuestenlinie nicht die Rekursionsgrenze sprengt.

    WIRD FUER DIE KUESTENPUNKTE NICHT MEHR BENUTZT (siehe
    _entlang_abtasten()): jede Ausduennung ersetzt die Kontur durch Sehnen,
    und die liegen nicht auf der Wasserlinie. Bleibt als Werkzeug erhalten,
    falls anderswo ein Linienzug vereinfacht werden soll - dort aber bitte
    denselben Fehler mitdenken."""
    n = len(punkte)
    if n < 3:
        return punkte
    behalten = np.zeros(n, dtype=bool)
    behalten[0] = behalten[-1] = True
    stapel = [(0, n - 1)]
    while stapel:
        a, b = stapel.pop()
        if b <= a + 1:
            continue
        start, ende = punkte[a], punkte[b]
        richtung = ende - start
        laenge = float(np.hypot(*richtung))
        teil = punkte[a + 1:b]
        if laenge < 1e-12:
            abstand = np.hypot(teil[:, 0] - start[0], teil[:, 1] - start[1])
        else:
            # senkrechter Abstand zur Geraden start-ende
            abstand = np.abs(np.cross(richtung, teil - start)) / laenge
        if not len(abstand):
            continue
        k = int(np.argmax(abstand))
        if abstand[k] > toleranz:
            behalten[a + 1 + k] = True
            stapel.append((a, a + 1 + k))
            stapel.append((a + 1 + k, b))
    return punkte[behalten]


def _entlang_abtasten(punkte, abstand):
    """
    Tastet einen Linienzug in gleichmaessigen Schritten AB - entlang seiner
    eigenen Bogenlaenge, nicht auf Sehnen.

    ZWEI GEMESSENE FEHLVERSUCHE, BEIDE MIT DERSELBEN URSACHE - festgehalten,
    weil der zweite genau wie der erste aussah und ich ihn deshalb fast
    uebersehen haette:

      1. Douglas-Peucker ausduennen, dann Zwischenpunkte auf der GERADEN
         zwischen zwei behaltenen Punkten einfuegen. In einer gekruemmten
         Bucht liegt diese Sehne weit im Land oder im Wasser - gemessen bis
         **97 m** von der Wasserlinie entfernt.
      2. Douglas-Peucker ausduennen, dann entlang der AUSGEDUENNTEN Linie
         abtasten. Klingt richtig, ist aber derselbe Fehler: die
         ausgeduennte Linie besteht ja gerade aus langen Sehnen. Gemessen
         immer noch **88 m**.

    Beides hat denselben Kern: **jede Ausduennung ersetzt die Kontur durch
    Sehnen, und Sehnen liegen nicht auf der Kueste.** Deshalb wird hier gar
    nicht ausgeduennt, sondern direkt auf der Originalkontur abgetastet -
    deren Punkte stehen nur Bruchteile eines Pixels auseinander, die Sehnen
    dazwischen sind also unerheblich. Die Punktzahl regelt allein `abstand`.
    """
    punkte = np.asarray(punkte, dtype=np.float64)
    if len(punkte) < 2:
        return punkte
    schritte = np.hypot(*(np.diff(punkte, axis=0).T))
    bogen = np.concatenate([[0.0], np.cumsum(schritte)])
    gesamt = float(bogen[-1])
    if gesamt < abstand:
        return punkte[[0, -1]]
    ziele = np.arange(0.0, gesamt, abstand)
    # Endpunkt mitnehmen, damit eine geschlossene Kontur geschlossen bleibt
    if gesamt - ziele[-1] > abstand * 0.25:
        ziele = np.append(ziele, gesamt)
    x = np.interp(ziele, bogen, punkte[:, 0])
    y = np.interp(ziele, bogen, punkte[:, 1])
    return np.stack([x, y], axis=1)


def kuestenpunkte(heightmap, toleranz=KUESTE_TOLERANZ_PX,
                  abstand=KUESTE_ABSTAND_PX, mindestlaenge=8):
    """
    Alle Kuestenlinien, ausgeduennt und wieder gleichmaessig verdichtet.

    Rueckgabe: (P,2)-Array aller Punkte plus die Liste der einzelnen
    Linienzuege (fuer die Treue-Messung unten).
    """
    linien = []
    for roh in kuestenlinien(heightmap):
        if len(roh) < mindestlaenge:
            # Winzige Konturen sind einzelne Felsen im Wasser - sie wuerden
            # nur Vertices kosten und im Bild nicht auffallen.
            continue
        # DIREKT auf der Originalkontur abtasten - KEIN Ausduennen davor.
        # Siehe _entlang_abtasten() fuer die zwei gemessenen Fehlversuche.
        linien.append(_entlang_abtasten(roh, abstand))
    if not linien:
        return np.zeros((0, 2)), []
    return np.concatenate(linien), linien


def stuetzpunkte(heightmap, innen_abstand=INNEN_ABSTAND_PX, **kwargs):
    """
    Alle Punkte, ueber die trianguliert wird: die Kuestenlinie plus ein
    gestreutes Innengitter plus die vier Kartenecken.

    Das Innengitter laesst Punkte weg, die zu nah an der Kueste liegen -
    sonst entstehen dort winzige, schlecht geformte Dreiecke, und die
    Kuestenpunkte verlieren ihren Dichtevorteil.

    Rueckgabe: (punkte, linien, anzahl_kueste). Die Kuestenpunkte stehen
    VORNE - `baue_kuesten_mesh()` setzt deren Hoehe auf 0 und verlaesst sich
    auf diese Reihenfolge.
    """
    from scipy.spatial import cKDTree

    H = np.asarray(heightmap, dtype=np.float64)
    hoehe, breite = H.shape
    kueste, linien = kuestenpunkte(H, **kwargs)

    gx, gy = np.meshgrid(
        np.arange(0, breite, innen_abstand, dtype=np.float64),
        np.arange(0, hoehe, innen_abstand, dtype=np.float64))
    innen = np.stack([gx.ravel(), gy.ravel()], axis=1)

    if len(kueste):
        baum = cKDTree(kueste)
        abstand, _ = baum.query(innen)
        innen = innen[abstand > innen_abstand * 0.75]

    ecken = np.array([[0, 0], [breite - 1, 0],
                      [0, hoehe - 1], [breite - 1, hoehe - 1]], dtype=np.float64)
    # REIHENFOLGE IST BINDEND: Kueste zuerst. `baue_kuesten_mesh()` setzt die
    # Hoehe der ersten `anzahl_kueste` Punkte auf 0 und verlaesst sich darauf.
    alle = np.concatenate([p for p in (kueste, innen, ecken) if len(p)])
    return alle, linien, len(kueste)


def kuestentreue(punkte, dreiecke, linien, toleranz=1e-6):
    """
    Wie viele Kuesten-Konturkanten kommen tatsaechlich als Dreieckskante vor?

    **Der Prueflauf, ohne den der Unterschied zum echten Constrained Delaunay
    nicht zu bemerken waere** (siehe Modulkopf): das Verfahren hier ERZWINGT
    die Kuestenkanten nicht, es macht sie nur sehr wahrscheinlich. Ein Wert
    deutlich unter 1.0 heisst, dass Dreiecke ueber die Kueste hinweglaufen -
    genau die Treppe, die beseitigt werden sollte.

    Rueckgabe: Anteil 0..1.
    """
    if not linien:
        return 1.0
    from scipy.spatial import cKDTree

    baum = cKDTree(punkte)
    kanten = set()
    for a, b, c in dreiecke:
        for u, v in ((a, b), (b, c), (c, a)):
            kanten.add((min(u, v), max(u, v)))

    gesamt = treffer = 0
    for linie in linien:
        _abstand, index = baum.query(linie)
        for u, v in zip(index[:-1], index[1:]):
            if u == v:
                continue
            gesamt += 1
            if (min(u, v), max(u, v)) in kanten:
                treffer += 1
    return treffer / gesamt if gesamt else 1.0


def baue_kuesten_mesh(heightmap, terrain_scale_factor, terrain_height_scale,
                      innen_abstand=INNEN_ABSTAND_PX, **kwargs):
    """
    Vollstaendiges Terrain-Mesh mit freier Kuestenvernetzung.

    Rueckgabe wie `adaptive_terrain_mesh.build_adaptive_mesh()`:
    (vertices float32 interleaved [x,y,z,nx,ny,nz,u,v], indices uint32,
    stats dict) - damit es an derselben Stelle eingesetzt werden kann.

    Die Hoehe an einem Kuestenpunkt wird bilinear aus der Heightmap geholt;
    auf der Kontur ist sie per Definition rund 0, was genau richtig ist - der
    Kuestenpunkt SOLL auf Meereshoehe liegen.
    """
    from scipy.spatial import Delaunay
    from gui.widgets.wege_geometrie import _hoehe_an

    H = np.asarray(heightmap, dtype=np.float32)
    hoehe_px, breite_px = H.shape
    punkte, linien, anzahl_kueste = stuetzpunkte(
        H, innen_abstand=innen_abstand, **kwargs)
    if len(punkte) < 4:
        return None

    tri = Delaunay(punkte)
    dreiecke = tri.simplices
    if not len(dreiecke):
        return None

    xs, ys = punkte[:, 0], punkte[:, 1]
    z = _hoehe_an(H, xs, ys)

    # KUESTENPUNKTE BEKOMMEN HOEHE 0 GESETZT, NICHT NACHGESCHLAGEN.
    #
    # Sie liegen per Definition auf der Nullhoehe - dafuer wurden sie ja
    # gerade aus der 0-Kontur gewonnen. Ein Nachschlagen ist nicht nur
    # ueberfluessig, es ist MESSBAR FALSCH: `find_contours` interpoliert
    # linear entlang der Pixelkanten, `_hoehe_an` bilinear ueber die Flaeche.
    # An einer Steilklippe (bis 800 m Hoehenunterschied ueber wenige Pixel)
    # laufen beide auseinander - gemessen bis **75 m** Hoehe an einem Punkt,
    # der auf der Wasserlinie liegen sollte, bei einem Median von 0.38 m. Der
    # Kuestenvertex waere dort aus dem Wasser geragt, also genau die Kante,
    # die dieses Verfahren beseitigen soll.
    #
    # Die Kuestenpunkte stehen durch `stuetzpunkte()` garantiert VORNE im
    # Array - deshalb genuegt ein Schnitt.
    if anzahl_kueste:
        z[:anzahl_kueste] = 0.0

    pos_x = (np.clip(xs, 0, breite_px - 1) / (breite_px - 1) - 0.5) * breite_px * terrain_scale_factor
    pos_z = (np.clip(ys, 0, hoehe_px - 1) / (hoehe_px - 1) - 0.5) * hoehe_px * terrain_scale_factor
    pos_y = z * terrain_height_scale

    # Normalen aus dem Gelaende, wie beim Quadtree-Mesh - so bleibt die
    # Beleuchtung identisch, egal welches Verfahren das Netz gebaut hat.
    from gui.widgets.adaptive_terrain_mesh import _normalen_voll
    nx_feld, ny_feld, nz_feld = _normalen_voll(H, terrain_height_scale,
                                                terrain_scale_factor)
    xi = np.clip(np.round(xs).astype(np.int32), 0, breite_px - 1)
    yi = np.clip(np.round(ys).astype(np.int32), 0, hoehe_px - 1)

    tex_u = np.clip(xs, 0, breite_px - 1) / (breite_px - 1)
    tex_v = np.clip(ys, 0, hoehe_px - 1) / (hoehe_px - 1)

    vertex_array = np.stack([
        pos_x, pos_y, pos_z,
        nx_feld[yi, xi], ny_feld[yi, xi], nz_feld[yi, xi],
        tex_u, tex_v], axis=-1).astype(np.float32)

    # WICKLUNG ANGLEICHEN. Delaunay liefert die Ecken gegen den Uhrzeigersinn
    # in einem RECHTShaendigen (x,y)-System. Die Anzeige nutzt (x, z) mit
    # nach unten wachsendem z und Backface-Culling auf GL_CW - ohne Umdrehen
    # waere das halbe Netz von hinten und damit unsichtbar.
    dreiecke = dreiecke[:, ::-1]

    stats = {
        "vertices": len(punkte),
        "dreiecke": len(dreiecke),
        "kuestenpunkte": int(sum(len(l) for l in linien)),
        "kuestentreue": float(kuestentreue(punkte, dreiecke, linien)),
        "voll_dreiecke": 2 * (breite_px - 1) * (hoehe_px - 1),
        "voll_vertices": breite_px * hoehe_px,
    }
    return vertex_array.reshape(-1), dreiecke.reshape(-1).astype(np.uint32), stats


# ---------------------------------------------------------------------------
# KLIPPENBAND (docs/OFFENE_PUNKTE.md 6.20, Variante 3)
# ---------------------------------------------------------------------------
#
# Das Terrain-Mesh bleibt UNANGETASTET - es ist als Hoehenmodell gut (gemessen
# p99 5.4 m Abweichung) und soll es bleiben. Darueber liegt ein eigenes Band
# entlang der 0-Kontur, das die Steilkante zeigt.
#
# WARUM DAS DIE EIGENTLICHE ANTWORT AUF DIE FRAGE IST: eine Heightmap kann
# per Definition keine senkrechte Flaeche - ein z-Wert je (x,y). Ein Band aus
# zwei Vertexreihen kann es sehr wohl: die untere Reihe liegt auf der
# Wasserlinie (z=0), die obere minimal landeinwaerts auf der dortigen
# Gelaendehoehe. Dazwischen spannen zwei Dreiecke eine ECHTE Wand auf, so
# steil wie die Hoehendifferenz es vorgibt - unabhaengig von der
# Rasteraufloesung, ohne Treppe.
#
# Dieselbe Mechanik wie die Wegbaender (wege_geometrie.py): zwei Vertexreihen
# entlang eines Linienzugs. Dort ist sie bereits im Einsatz und geprueft.

# Wie weit landeinwaerts die Oberkante der Wand liegt, in Pixeln. Klein
# genug, dass die Wand steil wirkt; gross genug, dass die Hoehe dort schon
# ueber Null liegt.
KLIPPE_VERSATZ_PX = 1.0

# Ab welcher Hoehe an der Oberkante ueberhaupt eine Wand gezeichnet wird.
# An einem Flachstrand gibt es keine Klippe - dort waere ein Band nur ein
# stoerender Streifen.
KLIPPE_MINDESTHOEHE_M = 12.0


def klippenband(heightmap, terrain_scale_factor, terrain_height_scale,
                versatz_px=KLIPPE_VERSATZ_PX,
                mindesthoehe_m=KLIPPE_MINDESTHOEHE_M,
                abstand_px=KUESTE_ABSTAND_PX):
    """
    Senkrechte Klippenwaende entlang der Kuestenlinie als eigene Geometrie.

    Rueckgabe (vertices (N,3) float32 in WELTkoordinaten, indices uint32,
    stats dict) - oder (leer, leer, stats) wenn es keine Klippen gibt.

    Die Landseite wird aus der Gelaendehoehe bestimmt, nicht aus der
    Umlaufrichtung der Kontur: `find_contours` garantiert keine einheitliche
    Orientierung, und eine falsch geratene Seite haette die Waende ins Meer
    gestellt - ein Fehler, der auf einem Uebersichtsbild leicht durchgeht.
    """
    H = np.asarray(heightmap, dtype=np.float32)
    hoehe_px, breite_px = H.shape
    from gui.widgets.wege_geometrie import _hoehe_an

    alle_v, alle_i = [], []
    versatz = 0
    abschnitte = 0
    for roh in kuestenlinien(H):
        if len(roh) < 8:
            continue
        linie = _entlang_abtasten(roh, abstand_px)
        if len(linie) < 2:
            continue

        # Laufrichtung und Normale in der Kartenebene
        richtung = np.zeros_like(linie)
        richtung[1:-1] = linie[2:] - linie[:-2]
        richtung[0] = linie[1] - linie[0]
        richtung[-1] = linie[-1] - linie[-2]
        laenge = np.hypot(richtung[:, 0], richtung[:, 1])
        richtung /= np.where(laenge > 1e-9, laenge, 1.0)[:, None]
        normale = np.stack([-richtung[:, 1], richtung[:, 0]], axis=1)

        # WELCHE SEITE IST LAND? Beide Seiten abtasten und die hoehere nehmen.
        probe_a = linie + normale * versatz_px
        probe_b = linie - normale * versatz_px
        h_a = _hoehe_an(H, probe_a[:, 0], probe_a[:, 1])
        h_b = _hoehe_an(H, probe_b[:, 0], probe_b[:, 1])
        landseite = np.where((h_a >= h_b)[:, None], normale, -normale)
        oben_xy = linie + landseite * versatz_px
        oben_h = np.maximum(_hoehe_an(H, oben_xy[:, 0], oben_xy[:, 1]), 0.0)

        # Nur dort eine Wand, wo es wirklich steil hochgeht
        hoch_genug = oben_h >= mindesthoehe_m
        if not hoch_genug.any():
            continue

        def nach_welt(xy, z):
            x = np.clip(xy[:, 0], 0, breite_px - 1)
            y = np.clip(xy[:, 1], 0, hoehe_px - 1)
            return np.stack([
                (x / (breite_px - 1) - 0.5) * breite_px * terrain_scale_factor,
                z * terrain_height_scale,
                (y / (hoehe_px - 1) - 0.5) * hoehe_px * terrain_scale_factor,
            ], axis=1)

        unten = nach_welt(linie, np.zeros(len(linie)))
        oben = nach_welt(oben_xy, oben_h)

        n = len(linie)
        v = np.empty((2 * n, 3), dtype=np.float32)
        v[0::2] = unten
        v[1::2] = oben

        idx = []
        for k in range(n - 1):
            # Nur Abschnitte zeichnen, wo BEIDE Enden hoch genug sind - sonst
            # entstehen Dreiecksfetzen dort, wo die Klippe ausläuft.
            if not (hoch_genug[k] and hoch_genug[k + 1]):
                continue
            a, b = 2 * k, 2 * k + 1
            c, d = 2 * (k + 1), 2 * (k + 1) + 1
            idx.extend((a, c, b, b, c, d))
            abschnitte += 1
        if not idx:
            continue
        # NUR TATSAECHLICH VERWENDETE VERTICES BEHALTEN. Oben entstehen zwei
        # Vertices je Konturpunkt, Dreiecke aber nur fuer Abschnitte ueber
        # der Mindesthoehe - an einem Flachstrand bliebe sonst die halbe
        # Kuestenlinie als toter Ballast im Puffer stehen (gemessen: die
        # niedrigste mitgefuehrte Oberkante lag bei 1 m, obwohl erst ab 12 m
        # eine Wand gezeichnet wird).
        idx = np.asarray(idx, dtype=np.int64)
        genutzt, neu_idx = np.unique(idx, return_inverse=True)
        alle_v.append(v[genutzt])
        alle_i.append((neu_idx + versatz).astype(np.uint32))
        versatz += len(genutzt)

    stats = {"abschnitte": abschnitte,
             "linien": len(alle_v),
             "vertices": versatz,
             "dreiecke": sum(len(i) for i in alle_i) // 3}
    if not alle_v:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros(0, dtype=np.uint32), stats)
    return (np.concatenate(alle_v).astype(np.float32),
            np.concatenate(alle_i).astype(np.uint32), stats)
