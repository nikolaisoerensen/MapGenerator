"""
Path: gui/widgets/kuesten_schnitt.py

DIE KUESTENLINIE ALS ECHTE MESHKANTE - Gitter entlang der 0-Kontur schneiden.

Nutzerbefund 2026-08-17: *"die kuestenlinie sollte nicht rastergebunden sein
oder? das sollten wir fixen."* Richtig - `core/vektor_kueste.py` loest bis
hierher nur die HOEHE vom Raster, nicht die LINIE: die Vertices sassen
weiterhin auf Gitterpunkten, Uebersampling machte die Treppe nur kleiner.

DAS VERFAHREN, UND WARUM ES OHNE CONSTRAINED DELAUNAY AUSKOMMT

Jede Gitterzelle wird einzeln betrachtet. Wo die 0-Linie eine Zellkante
kreuzt, entsteht ein NEUER Vertex an der linear interpolierten Stelle -
genau wie bei Marching Squares, also zwischen den Pixeln. Die Zelle wird
dann so zerlegt, dass diese Punkte Ecken von Dreiecken sind und die
Kuestenlinie eine echte Dreieckskante wird.

Der entscheidende Punkt fuer die Dichtheit: **auf einer gemeinsamen Kante
rechnen beide Nachbarzellen denselben Schnittpunkt aus denselben zwei
Eckhoehen.** Die Naht ist damit per Konstruktion dicht - es braucht weder
eine Zwangskanten-Triangulierung (6.20 scheiterte daran, dass scipy kein CDT
kann) noch `preserve_border` mit seiner rastergebundenen Trennlinie (6.33).

Der frueher gescheiterte Weg war ein AUFGESETZTES Band (6.20, "Klippenband"):
zwei Flaechen an fast derselben Stelle, die um dieselben Tiefenwerte
konkurrieren - es flackerte und das Raster schimmerte durch. Hier wird
stattdessen das Gelaendenetz SELBST zerschnitten; es gibt nur eine Flaeche.

ZELLZERLEGUNG

Die Zellecken heissen c00 c10 c11 c01 (gegen den Uhrzeigersinn ab unten
links), die Kanten dazwischen e0 e1 e2 e3. Der Umlauf

    c00  e0  c10  e1  c11  e2  c01  e3

liefert die Land- und die See-Teilflaeche direkt: man laeuft ihn ab und
nimmt jede Ecke mit, die auf der gesuchten Seite liegt, sowie jeden
Kantenpunkt, an dem die Linie wirklich kreuzt. Beide Teilflaechen werden
faecherfoermig trianguliert. Das ersetzt eine 16-Faelle-Tabelle durch eine
Regel - bis auf einen Fall:

DER SATTEL. Liegen die beiden DIAGONALEN Ecken auf derselben Seite
(c00+c11 gegen c10+c01), ist die Zerlegung mehrdeutig: entweder sind die
beiden Landecken durch eine Landbruecke verbunden, oder sie sind zwei
getrennte Zipfel. Der Umlauf allein liefert dort ein Sechseck, das die
Mitte einschliesst - was nur in einem der beiden Faelle stimmt. Entschieden
wird ueber den Mittelwert der vier Ecken; dieselbe Regel benutzt auch
Marching Squares. Ohne diese Sonderbehandlung entstuenden an Sattelzellen
ueberlappende Dreiecke.
"""

import numpy as np

# Umlauf der acht Plaetze einer Zelle: Ecke, Kante, Ecke, Kante, ...
# 0..3 sind die Ecken c00 c10 c11 c01, 4..7 die Kanten e0 e1 e2 e3.
_UMLAUF = (0, 4, 1, 5, 2, 6, 3, 7)

# Welche zwei Ecken jede Kante verbindet.
_KANTE_ECKEN = {4: (0, 1), 5: (1, 2), 6: (2, 3), 7: (3, 0)}

# Die beiden Sattel-Belegungen (diagonale Ecken auf derselben Seite).
_SATTEL = (0b0101, 0b1010)


def _polygone_je_belegung():
    """
    Fuer jede der 16 Belegungen die Land- und die See-Teilflaeche als
    Plätzeliste (siehe _UMLAUF). Einmal beim Import gerechnet.

    Sattelbelegungen bleiben hier ausgespart - sie haengen vom Mittelwert
    der Zelle ab und werden erst zur Laufzeit entschieden.
    """
    tabelle = {}
    for belegung in range(16):
        ist_land = [(belegung >> k) & 1 for k in range(4)]
        if belegung in _SATTEL:
            tabelle[belegung] = None
            continue
        land, see = [], []
        for platz in _UMLAUF:
            if platz < 4:
                (land if ist_land[platz] else see).append(platz)
            else:
                a, b = _KANTE_ECKEN[platz]
                if ist_land[a] != ist_land[b]:      # Linie kreuzt hier
                    land.append(platz)
                    see.append(platz)
        tabelle[belegung] = (land, see)
    return tabelle


_POLYGONE = _polygone_je_belegung()


def _faecher(polygon):
    """
    Faechertriangulierung einer Plätzeliste: (0,2,1), (0,3,2), ...

    RUECKWAERTS, und das ist Absicht. _UMLAUF laeuft gegen den Uhrzeigersinn
    um die Zelle; die naive Reihenfolge (polygon[0], polygon[i],
    polygon[i+1]) erbt diese Wicklung und liefert Dreiecke mit POSITIVER
    signierter Flaeche.

    Das Gitter in map_display_3d._generate_terrain_mesh() wickelt aber
    (top_left, bottom_left, top_right), also IM Uhrzeigersinn, signierte
    Flaeche -0.5. Und dort steht glFrontFace(GL_CW) mit glCullFace(GL_BACK).

    Ein Netz mit der anderen Wicklung wird deshalb exakt verkehrt herum
    gecullt: alles, was zur Kamera zeigt, verschwindet, sichtbar bleiben nur
    die abgewandten Rueckhaenge der Grate. Von unten ist die Karte dann
    vollstaendig - das ist die Signatur (Nutzerbefund 2026-08-22, das Gelaende
    sah aus wie in Streifen zerschnitten).

    Der Smoke-Test hatte nur geprueft, dass die Wicklung EINHEITLICH ist,
    nicht dass sie zum Gitter PASST - siehe Gruppe `wicklung_wie_gitter`
    dort, die genau diese Luecke schliesst.
    """
    return [(polygon[0], polygon[i + 1], polygon[i])
            for i in range(1, len(polygon) - 1)]


def schnitt_netz(H, hoehen_fn=None, pegel=0.0):
    """
    Zerlegt das Gitter von `H` so, dass die `pegel`-Kontur Dreieckskante ist.

    `H` bestimmt NUR den Verlauf der Kontur und die Vorzeichen. Die HOEHEN der
    Vertices kommen aus `hoehen_fn(x, y)` (Pixelkoordinaten, beliebige
    Fliesskommawerte) - dort wird `core.vektor_kueste.an_punkten` uebergeben,
    damit die Klippen ihre volle, nicht rastergedeckelte Steilheit bekommen.
    Ohne `hoehen_fn` wird bilinear aus `H` abgetastet.

    Die Konturvertices bekommen die Hoehe `pegel` GESETZT, nicht abgetastet -
    sie liegen per Definition darauf. (Dieselbe Lehre wie in
    `kuesten_mesh.py`: das Nachschlagen der Hoehe an einem Konturpunkt ergab
    dort bis zu 75 m Abweichung, weil Marching Squares laengs der Pixelkante
    interpoliert und die Hoehenabfrage bilinear ueber die Flaeche.)

    Rueckgabe (punkte, dreiecke, ist_land_dreieck):
      punkte  (N,2) float64 - Pixelkoordinaten (x, y), NICHT gerundet
      hoehen  (N,)  float64 - Meter
      dreiecke (M,3) int32
      ist_land (M,) bool    - je Dreieck, ob es auf der Landseite liegt
    """
    H = np.asarray(H, dtype=np.float64)
    hoehe_px, breite = H.shape
    zellen_y, zellen_x = hoehe_px - 1, breite - 1

    land = H > pegel

    # ---- Vertexnummern in drei Bloecken, damit gemeinsame Kanten
    # ---- zwangslaeufig denselben Index bekommen (Dichtheit).
    n_ecken = hoehe_px * breite
    n_hkanten = hoehe_px * zellen_x          # waagerechte Kanten
    n_vkanten = zellen_y * breite            # senkrechte Kanten

    def ecke_index(x, y):
        return y * breite + x

    def hkante_index(x, y):                  # Kante (x,y)-(x+1,y)
        return n_ecken + y * zellen_x + x

    def vkante_index(x, y):                  # Kante (x,y)-(x,y+1)
        return n_ecken + n_hkanten + y * breite + x

    # ---- Kreuzungspunkte auf allen Kanten vorrechnen (vektorisiert) -----
    gy, gx = np.mgrid[0:hoehe_px, 0:breite]

    hx = np.full((hoehe_px, zellen_x), np.nan)
    ha, hb = H[:, :-1], H[:, 1:]
    kreuzt_h = (land[:, :-1] != land[:, 1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_h = (pegel - ha) / (hb - ha)
    t_h = np.clip(np.nan_to_num(t_h, nan=0.5, posinf=0.5, neginf=0.5), 0.0, 1.0)
    hx = gx[:, :-1] + t_h

    vy = np.full((zellen_y, breite), np.nan)
    va, vb = H[:-1, :], H[1:, :]
    kreuzt_v = (land[:-1, :] != land[1:, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_v = (pegel - va) / (vb - va)
    t_v = np.clip(np.nan_to_num(t_v, nan=0.5, posinf=0.5, neginf=0.5), 0.0, 1.0)
    vy = gy[:-1, :] + t_v

    # ---- Vertexliste: Ecken, dann H-Kanten, dann V-Kanten --------------
    punkte = np.empty((n_ecken + n_hkanten + n_vkanten, 2), dtype=np.float64)
    punkte[:n_ecken, 0] = gx.ravel()
    punkte[:n_ecken, 1] = gy.ravel()
    punkte[n_ecken:n_ecken + n_hkanten, 0] = hx.ravel()
    punkte[n_ecken:n_ecken + n_hkanten, 1] = np.repeat(
        np.arange(hoehe_px, dtype=np.float64), zellen_x)
    punkte[n_ecken + n_hkanten:, 0] = np.tile(
        np.arange(breite, dtype=np.float64), zellen_y)
    punkte[n_ecken + n_hkanten:, 1] = vy.ravel()

    # ---- Hoehen ---------------------------------------------------------
    if hoehen_fn is None:
        def hoehen_fn(x, y):
            xi = np.clip(x, 0, breite - 1.001)
            yi = np.clip(y, 0, hoehe_px - 1.001)
            x0 = np.floor(xi).astype(np.int32)
            y0 = np.floor(yi).astype(np.int32)
            fx, fy = xi - x0, yi - y0
            x1 = np.minimum(x0 + 1, breite - 1)
            y1 = np.minimum(y0 + 1, hoehe_px - 1)
            return ((H[y0, x0] * (1 - fx) + H[y0, x1] * fx) * (1 - fy)
                    + (H[y1, x0] * (1 - fx) + H[y1, x1] * fx) * fy)

    hoehen = np.asarray(hoehen_fn(punkte[:, 0], punkte[:, 1]), dtype=np.float64)
    # Konturvertices liegen per Definition auf `pegel` - setzen, nicht
    # abtasten (siehe Docstring).
    hoehen[n_ecken:] = pegel

    # ---- Zellen nach Belegung gruppieren und je Gruppe triangulieren ----
    zy, zx = np.mgrid[0:zellen_y, 0:zellen_x]
    belegung = (land[:-1, :-1].astype(np.int32)
                | (land[:-1, 1:].astype(np.int32) << 1)
                | (land[1:, 1:].astype(np.int32) << 2)
                | (land[1:, :-1].astype(np.int32) << 3))

    # Plätze -> Vertexindex je Zelle
    platz_index = {
        0: ecke_index(zx, zy),
        1: ecke_index(zx + 1, zy),
        2: ecke_index(zx + 1, zy + 1),
        3: ecke_index(zx, zy + 1),
        4: hkante_index(zx, zy),
        5: vkante_index(zx + 1, zy),
        6: hkante_index(zx, zy + 1),
        7: vkante_index(zx, zy),
    }

    dreiecke, ist_land = [], []

    def _sammeln(maske, polygon, land_flag):
        if polygon is None or len(polygon) < 3 or not maske.any():
            return
        for a, b, c in _faecher(polygon):
            dreiecke.append(np.stack([platz_index[a][maske],
                                      platz_index[b][maske],
                                      platz_index[c][maske]], axis=1))
            ist_land.append(np.full(int(maske.sum()), land_flag, dtype=bool))

    for wert in range(16):
        maske = belegung == wert
        if not maske.any():
            continue
        if wert in _SATTEL:
            continue                                   # unten, mit Mittelwert
        land_poly, see_poly = _POLYGONE[wert]
        _sammeln(maske, land_poly, True)
        _sammeln(maske, see_poly, False)

    # ---- Sattelzellen: Mittelwert entscheidet ---------------------------
    mitte = 0.25 * (H[:-1, :-1] + H[:-1, 1:] + H[1:, 1:] + H[1:, :-1])
    for wert in _SATTEL:
        maske_all = belegung == wert
        if not maske_all.any():
            continue
        ecken_land = [(wert >> k) & 1 for k in range(4)]
        # Die zwei Zipfel: je eine Ecke mit ihren beiden Kantenpunkten.
        zipfel = []
        for k in range(4):
            vor = 4 + ((k + 3) % 4)      # Kante VOR der Ecke im Umlauf
            nach = 4 + k                 # Kante NACH der Ecke
            zipfel.append((k, nach, vor))
        for mitte_ist_land in (True, False):
            maske = maske_all & ((mitte > pegel) == mitte_ist_land)
            if not maske.any():
                continue
            # Die Seite, auf der die Mitte liegt, wird zum Sechseck
            # verbunden; die andere zerfaellt in ihre zwei Zipfel.
            verbunden = [p for p in _UMLAUF
                         if (p >= 4) or (ecken_land[p] == mitte_ist_land)]
            getrennt_ecken = [k for k in range(4)
                              if ecken_land[k] != mitte_ist_land]
            _sammeln(maske, verbunden, mitte_ist_land)
            for k in getrennt_ecken:
                ecke, nach, vor = zipfel[k]
                _sammeln(maske, [ecke, nach, vor], not mitte_ist_land)

    if not dreiecke:
        return (punkte, hoehen, np.zeros((0, 3), dtype=np.int32),
                np.zeros(0, dtype=bool))

    return (punkte, hoehen,
            np.concatenate(dreiecke).astype(np.int32),
            np.concatenate(ist_land))


def baue_schnitt_mesh(heightmap, terrain_scale_factor, terrain_height_scale,
                      hoehen_fn=None, pegel=0.0):
    """
    Rueckgabe wie `adaptive_terrain_mesh.build_adaptive_mesh()`:
    (vertices float32 interleaved [x,y,z,nx,ny,nz,u,v], indices uint32,
    stats dict) - damit an derselben Stelle einsetzbar, insbesondere ueber
    `MapDisplay3D.setze_mesh_bauer()`.

    Normalen bilinear aus dem HOEHENFELD abgetastet, nicht aus den Dreiecken
    gemittelt: sonst saehe das geschnittene Netz an der Kueste facettiert aus,
    waehrend das uebrige Gelaende glatt schattiert ist - der Vergleich waere
    dann der zweier Beleuchtungen statt zweier Netze (dieselbe Ueberlegung
    wie in terrain_remesh.py).
    """
    from gui.widgets.adaptive_terrain_mesh import _normalen_voll

    H = np.asarray(heightmap, dtype=np.float32)
    hoehe_px, breite_px = H.shape
    punkte, hoehen, dreiecke, ist_land = schnitt_netz(H, hoehen_fn, pegel)
    if len(dreiecke) == 0:
        return None

    # Nur tatsaechlich benutzte Vertices behalten - die Kreuzungspunkte auf
    # Kanten ohne Schnitt sind angelegt, aber unbenutzt (sie stehen im
    # Vertexblock, damit die Indexrechnung einfach bleibt).
    benutzt = np.zeros(len(punkte), dtype=bool)
    benutzt[dreiecke.ravel()] = True
    um = np.full(len(punkte), -1, dtype=np.int64)
    um[benutzt] = np.arange(int(benutzt.sum()))
    punkte = punkte[benutzt]
    hoehen = hoehen[benutzt]
    dreiecke = um[dreiecke].astype(np.int32)

    px = np.clip(punkte[:, 0], 0.0, breite_px - 1)
    py = np.clip(punkte[:, 1], 0.0, hoehe_px - 1)

    nx_feld, ny_feld, nz_feld = _normalen_voll(H, terrain_height_scale,
                                               terrain_scale_factor)
    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    x1 = np.minimum(x0 + 1, breite_px - 1)
    y1 = np.minimum(y0 + 1, hoehe_px - 1)
    fx, fy = px - x0, py - y0

    def _bilinear(feld):
        oben = feld[y0, x0] * (1 - fx) + feld[y0, x1] * fx
        unten = feld[y1, x0] * (1 - fx) + feld[y1, x1] * fx
        return oben * (1 - fy) + unten * fy

    nx, ny, nz = _bilinear(nx_feld), _bilinear(ny_feld), _bilinear(nz_feld)
    laenge = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    sicher = np.where(laenge > 0, laenge, 1.0)
    nx, ny, nz = nx / sicher, ny / sicher, nz / sicher

    pos_x = (px / (breite_px - 1) - 0.5) * breite_px * terrain_scale_factor
    pos_z = (py / (hoehe_px - 1) - 0.5) * hoehe_px * terrain_scale_factor
    pos_y = hoehen * terrain_height_scale

    vertices = np.stack([pos_x, pos_y, pos_z, nx, ny, nz,
                         px / (breite_px - 1), py / (hoehe_px - 1)],
                        axis=-1).astype(np.float32)

    n_ecken = hoehe_px * breite_px
    kontur = benutzt.copy()
    kontur[:n_ecken] = False
    versatz = np.hypot(px - np.round(px), py - np.round(py))
    ist_kontur = np.zeros(len(punkte), dtype=bool)
    ist_kontur[um[np.flatnonzero(kontur)]] = True

    stats = {
        "vertices": len(punkte),
        "dreiecke": len(dreiecke),
        "voll_dreiecke": 2 * (breite_px - 1) * (hoehe_px - 1),
        "voll_vertices": n_ecken,
        "konturvertices": int(ist_kontur.sum()),
        "frei_verschoben": float((versatz > 0.01).mean()),
        "versatz_median_px": float(np.median(versatz[ist_kontur]))
        if ist_kontur.any() else 0.0,
        "aus_cache": False,
    }
    return vertices.reshape(-1), dreiecke.reshape(-1).astype(np.uint32), stats
