"""
Path: gui/widgets/adaptive_terrain_mesh.py

Funktionsweise: Fehler-getriebene adaptive Terrain-Triangulierung (restricted
quadtree) als Ersatz fuer das gleichfoermige 1-Vertex-pro-Pixel-Gitter in
_generate_terrain_mesh() (map_display_3d.py). Flache Bereiche (offenes Meer,
Ebenen) bekommen wenige grosse Dreiecke, Klippen/Detailbereiche bleiben bei
voller Pixel-Aufloesung.

Algorithmus (bewusst KEIN RTIN/Martini-Bit-Trick - siehe Begruendung unten):
1. Quadtree ueber die Heightmap, rekursiv geteilt nach Hoehen-Abweichung
   (bilineare Eckpunkt-Interpolation vs. tatsaechliche Zwischenwerte).
2. "Restricted"/balancierter Quadtree: kein Blatt darf mehr als eine Stufe
   feiner/groeber sein als seine Nachbarn (klassische, gut verstandene
   Technik aus Terrain-LOD-Engines).
3. Pro Blatt: 2 Dreiecke (Standardfall) bis 6 Dreiecke (alle 4 Kanten haben
   einen feineren Nachbarn) durch Faecher-Triangulierung mit eingefuegten
   Kanten-Mittelpunkten - das ist die Riss-Vermeidung (T-Junction-Fix).

Warum kein RTIN/Martini: dessen Bit-Indizierung (ein Fehlerwert pro
Gitterpunkt statt pro Dreieck, ueber die Hypotenusen-Halbierung kodiert) ist
ohne Referenz-Implementierung zum Gegenpruefen fehleranfaellig - ein Fehler
dort erzeugt Risse, die nur im laufenden 3D-Fenster sichtbar wuerden (dieses
Projekt kann Compute-Shader headless testen, aber NICHT das gerenderte Bild).
Der Quadtree-Ansatz hier ist dagegen mit einer expliziten, headless
lauffaehigen Wasserdichtigkeits-Pruefung verifizierbar (siehe
tests/smoke_test_adaptive_terrain_mesh.py: jede innere Kante muss von genau
2 Dreiecken geteilt werden) - Korrektheit vor Optimalitaet.

Bewusst KEINE entfernungsbasierte LOD (mehrere Meshes je nach Kameraabstand):
dieses Tool zeigt ein einzelnes, kleines (~21km) Gelaende in einer Vorschau
mit Orbit-Kamera, kein offenes/stroemendes Terrain. Die adaptive Triangulierung
platziert Detail bereits nach der Komplexitaet des Gelaendes selbst - genau
das eigentliche Ziel (wenige Dreiecke auf offener See, viele an Klippen).
Mehrstufiges LOD wuerde zusaetzliches Nahtstellen-Risiko einbringen, ohne bei
dieser Groessenordnung einen echten Vorteil zu bieten.
"""

import numpy as np


def ist_fuer_adaptives_mesh_geeignet(heightmap):
    """
    Funktionsweise: Prueft die Quadtree-Voraussetzung - quadratisch, Kantenlaenge
    (map_size) selbst eine Zweierpotenz.
    Aufgabe: Alle in diesem Projekt tatsaechlich verwendeten map_size-Werte
    (128/256/512/1024/2048) sind SELBST Zweierpotenzen (NICHT "Zweierpotenz+1" -
    das war ein erster, falscher Anlauf hier: er liess die Bedingung fuer JEDE
    reale Kartengroesse scheitern, siehe [[project_adaptive_terrain_mesh_2026_08_12]]).
    Der Quadtree braucht N+1 Gitterpunkte mit N=Zweierpotenz - build_adaptive_mesh()
    polstert deshalb intern um genau eine Zeile/Spalte (Kantenwert dupliziert),
    N ist dann exakt die urspruengliche map_size.
    """
    if heightmap is None:
        return False
    height, width = heightmap.shape
    if height != width:
        return False
    return height > 0 and (height & (height - 1)) == 0


def _quad_fehler(H, x0, y0, size, cache):
    """
    Funktionsweise: Rekursiver, bottom-up gecachter Fehlerwert eines
    Quadranten (x0,y0,size) - wie stark weicht eine bilineare Interpolation
    seiner 4 Eckpunkte von den tatsaechlichen Zwischenwerten ab, inklusive
    aller Nachkommen (max-Propagation nach oben).
    Aufgabe: Ein einziger Aufruf an der Wurzel (0,0,N) fuellt den gesamten
    Cache bottom-up - nachfolgende Lookups sind O(1).
    """
    key = (x0, y0, size)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if size <= 1:
        cache[key] = 0.0
        return 0.0

    x1, y1 = x0 + size, y0 + size
    h00 = H[y0, x0]
    h10 = H[y0, x1]
    h01 = H[y1, x0]
    h11 = H[y1, x1]
    half = size // 2
    xm, ym = x0 + half, y0 + half

    eigen = abs(float(H[ym, xm]) - 0.25 * float(h00 + h10 + h01 + h11))
    eigen = max(
        eigen,
        abs(float(H[y0, xm]) - 0.5 * float(h00 + h10)),
        abs(float(H[y1, xm]) - 0.5 * float(h01 + h11)),
        abs(float(H[ym, x0]) - 0.5 * float(h00 + h01)),
        abs(float(H[ym, x1]) - 0.5 * float(h10 + h11)),
    )

    kinder_max = max(
        _quad_fehler(H, x0, y0, half, cache),
        _quad_fehler(H, xm, y0, half, cache),
        _quad_fehler(H, x0, ym, half, cache),
        _quad_fehler(H, xm, ym, half, cache),
    )
    fehler = max(eigen, kinder_max)
    cache[key] = fehler
    return fehler


def _blaetter_sammeln(cache, x0, y0, size, toleranz, min_size, blaetter):
    """Top-down Auswahl: Blatt wird zu einem Blatt, wenn seine Groesse die
    Mindestgroesse erreicht hat oder sein (gecachter) Fehler die Toleranz
    unterschreitet - sonst rekursiv in 4 Kinder teilen."""
    fehler = cache[(x0, y0, size)]
    if size <= min_size or fehler <= toleranz:
        blaetter[(x0, y0)] = size
        return
    half = size // 2
    xm, ym = x0 + half, y0 + half
    _blaetter_sammeln(cache, x0, y0, half, toleranz, min_size, blaetter)
    _blaetter_sammeln(cache, xm, y0, half, toleranz, min_size, blaetter)
    _blaetter_sammeln(cache, x0, ym, half, toleranz, min_size, blaetter)
    _blaetter_sammeln(cache, xm, ym, half, toleranz, min_size, blaetter)


def _besitzer_gitter(blaetter, zellen, min_size):
    """(zellen x zellen) Gitter in min_size-Einheiten: welche Blattgroesse
    besitzt jede kleinste Zelle - die Nachschlage-Struktur fuer Nachbar-Abfragen."""
    besitzer = np.zeros((zellen, zellen), dtype=np.int64)
    for (x0, y0), size in blaetter.items():
        c = size // min_size
        cx, cy = x0 // min_size, y0 // min_size
        besitzer[cy:cy + c, cx:cx + c] = size
    return besitzer


def _blaetter_balancieren(blaetter, N, min_size):
    """
    Funktionsweise: Erzwingt die "restricted quadtree"-Eigenschschaft - kein
    Blatt darf mehr als doppelt so gross sein wie sein feinster angrenzender
    Nachbar. Iteriert bis zum Fixpunkt (garantiert terminierend, da jede
    Runde nur Groessen verkleinert, min_size als Untergrenze).
    Aufgabe: Ist Voraussetzung fuer die Kanten-Faecher-Triangulierung unten -
    ohne diese Balance koennte ein Blatt einen um 2+ Stufen feineren
    Nachbarn haben, was ein einzelner Kanten-Mittelpunkt nicht mehr flicken kann.
    """
    zellen = N // min_size
    while True:
        besitzer = _besitzer_gitter(blaetter, zellen, min_size)
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


def _dreiecke_aus_blaettern(blaetter, N, min_size):
    """
    Funktionsweise: Baut Vertex-Liste (Gitterkoordinaten) + Dreiecks-Liste
    (Vertex-Indices) aus den balancierten Blaettern. Fasst geteilte
    Eckpunkte/Kanten-Mittelpunkte ueber ein Koordinaten->Index-Dict zusammen
    (dadurch automatisch identische Vertices auf beiden Seiten einer
    gemeinsamen Kante - Voraussetzung fuer Risslosigkeit).
    Aufgabe: Fächer-Reihenfolge (BL,BR,TR,TL) reproduziert exakt die Diagonale
    und Wicklung des bisherigen Gleichmaessig-Gitters (dort: Dreieck1 =
    TL,BL,TR; Dreieck2 = TR,BL,BR - beide nutzen die BL-TR-Diagonale), was bei
    aktivem Backface-Culling (glCullFace(GL_BACK), glFrontFace(GL_CW) in
    map_display_3d.py) zwingend ist, sonst wuerden neue Dreiecke von hinten
    weggeschnitten.
    """
    zellen = N // min_size
    besitzer = _besitzer_gitter(blaetter, zellen, min_size)

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

        polygon = [(x0, y1)]  # BL - Faecher-Ursprung
        if size > min_size and n_unten is not None and n_unten < size:
            polygon.append((xm, y1))
        polygon.append((x1, y1))  # BR
        if size > min_size and n_rechts is not None and n_rechts < size:
            polygon.append((x1, ym))
        polygon.append((x1, y0))  # TR
        if size > min_size and n_oben is not None and n_oben < size:
            polygon.append((xm, y0))
        polygon.append((x0, y0))  # TL
        if size > min_size and n_links is not None and n_links < size:
            polygon.append((x0, ym))

        idxs = [vidx(p) for p in polygon]
        for i in range(1, len(idxs) - 1):
            dreiecke.append((idxs[0], idxs[i], idxs[i + 1]))

    return positionen, dreiecke


def _gepolsterte_hoehen(heightmap):
    """
    Funktionsweise: Dupliziert die letzte Zeile/Spalte (Kantenwert, kein neuer
    Wert), damit aus einer (map_size, map_size)-Heightmap ein (N+1,N+1)-Gitter
    mit N=map_size wird - genau die Form, die der Quadtree braucht.
    Aufgabe: Rueckgabe (H_gepolstert, N). Der zusaetzliche Rand ist geometrisch
    eine reine Fortsetzung der letzten Reihe (Hoehe unveraendert), betrifft also
    nur eine unsichtbare Nullbreiten-Kante am Kartenrand, keine echte Flaeche.
    """
    H = heightmap.astype(np.float32)
    N = H.shape[0]
    H_gepolstert = np.pad(H, ((0, 1), (0, 1)), mode="edge")
    return H_gepolstert, N


def baue_adaptives_mesh_roh(heightmap, fehler_toleranz_m, min_leaf_size=1):
    """
    Funktionsweise: Reine Gitterkoordinaten-Fassung des adaptiven Meshes
    (keine Welt-Koordinaten/Normalen) - fuer die headless Wasserdichtigkeits-
    Pruefung im Smoke-Test direkt nutzbar. Arbeitet auf der um 1 gepolsterten
    Heightmap (siehe _gepolsterte_hoehen) - Gitterkoordinaten laufen deshalb
    0..N mit N=urspruengliche map_size.
    Aufgabe: Rueckgabe (positionen, dreiecke, blaetter, N) oder None wenn die
    Heightmap die Quadtree-Voraussetzung nicht erfuellt.
    """
    if not ist_fuer_adaptives_mesh_geeignet(heightmap):
        return None

    H, N = _gepolsterte_hoehen(heightmap)

    cache = {}
    _quad_fehler(H, 0, 0, N, cache)

    blaetter = {}
    _blaetter_sammeln(cache, 0, 0, N, fehler_toleranz_m, min_leaf_size, blaetter)
    blaetter = _blaetter_balancieren(blaetter, N, min_leaf_size)

    positionen, dreiecke = _dreiecke_aus_blaettern(blaetter, N, min_leaf_size)
    if not dreiecke:
        return None
    return positionen, dreiecke, blaetter, N


def _normalen_voll(H, terrain_height_scale, terrain_scale_factor):
    """Exakt dieselbe Formel wie _generate_terrain_mesh() in map_display_3d.py
    (zentrale Differenzen, einseitig am Rand) - fuer identisches Shading
    unabhaengig von der Mesh-Topologie."""
    height, width = H.shape
    dz_dx = np.empty_like(H)
    dz_dx[:, 1:-1] = H[:, 2:] - H[:, :-2]
    dz_dx[:, 0] = H[:, 1] - H[:, 0]
    dz_dx[:, -1] = H[:, -1] - H[:, -2]

    dz_dy = np.empty_like(H)
    dz_dy[1:-1, :] = H[2:, :] - H[:-2, :]
    dz_dy[0, :] = H[1, :] - H[0, :]
    dz_dy[-1, :] = H[-1, :] - H[-2, :]

    dx = dz_dx * terrain_height_scale
    dy = dz_dy * terrain_height_scale

    step_size = terrain_scale_factor
    normal_x = -dx * step_size
    normal_y = np.full((height, width), step_size ** 2, dtype=np.float32)
    normal_z = -dy * step_size

    length = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
    safe_length = np.where(length > 0, length, 1.0)
    normal_x = np.where(length > 0, normal_x / safe_length, 0.0)
    normal_y = np.where(length > 0, normal_y / safe_length, 1.0)
    normal_z = np.where(length > 0, normal_z / safe_length, 0.0)
    return normal_x, normal_y, normal_z


def build_adaptive_mesh(heightmap, terrain_scale_factor, terrain_height_scale,
                         fehler_toleranz_m, min_leaf_size=1):
    """
    Funktionsweise: Oeffentlicher Einstiegspunkt - baut das adaptive Mesh und
    liefert es im selben interleaved Vertex-Format wie das bisherige
    Gleichmaessig-Gitter ([pos_x,pos_y,pos_z,nx,ny,nz,u,v] pro Vertex), damit
    _create_mesh_buffers()/das Shader-Attribut-Layout in map_display_3d.py
    unveraendert bleiben kann.
    Aufgabe: Rueckgabe (vertices float32, indices uint32, stats dict) oder
    None wenn ungeeignet/leer - der Aufrufer faellt dann auf das
    Gleichmaessig-Gitter zurueck.
    """
    roh = baue_adaptives_mesh_roh(heightmap, fehler_toleranz_m, min_leaf_size)
    if roh is None:
        return None
    positionen, dreiecke, blaetter, N = roh

    height, width = heightmap.shape
    H_gepolstert, _ = _gepolsterte_hoehen(heightmap)
    normal_x, normal_y, normal_z = _normalen_voll(H_gepolstert, terrain_height_scale, terrain_scale_factor)

    gx = np.array([p[0] for p in positionen], dtype=np.int64)
    gy = np.array([p[1] for p in positionen], dtype=np.int64)

    # Fuer Position/Texturkoordinaten auf die ORIGINALE Kartengroesse geklemmt
    # (dieselbe Formel wie das bisherige Gleichmaessig-Gitter, Nenner width-1/
    # height-1) - nur die Hoehen-/Normalen-Lookups nutzen das gepolsterte
    # Gitter direkt (unverfaelscht, da der Polster-Rand ohnehin der duplizierte
    # Kantenwert ist).
    gx_klemm = np.minimum(gx, width - 1)
    gy_klemm = np.minimum(gy, height - 1)

    pos_x = (gx_klemm.astype(np.float32) / (width - 1) - 0.5) * width * terrain_scale_factor
    pos_z = (gy_klemm.astype(np.float32) / (height - 1) - 0.5) * height * terrain_scale_factor
    pos_y = H_gepolstert[gy, gx] * terrain_height_scale

    nx = normal_x[gy, gx]
    ny = normal_y[gy, gx]
    nz = normal_z[gy, gx]

    tex_u = gx_klemm.astype(np.float32) / (width - 1)
    tex_v = gy_klemm.astype(np.float32) / (height - 1)

    vertex_array = np.stack([pos_x, pos_y, pos_z, nx, ny, nz, tex_u, tex_v], axis=-1).astype(np.float32)
    vertices = vertex_array.reshape(-1)
    indices = np.array(dreiecke, dtype=np.uint32).reshape(-1)

    stats = {
        "vertices": len(positionen),
        "dreiecke": len(dreiecke),
        "blaetter": len(blaetter),
        "voll_dreiecke": 2 * (width - 1) * (height - 1),
        "voll_vertices": width * height,
    }
    return vertices, indices, stats
