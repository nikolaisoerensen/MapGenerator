"""
Path: gui/widgets/adaptive_terrain_mesh.py

Funktionsweise: Fehler-getriebene adaptive Terrain-Triangulierung (restricted
quadtree) als Ersatz fuer das gleichfoermige 1-Vertex-pro-Pixel-Gitter in
_generate_terrain_mesh() (map_display_3d.py). Flache Bereiche (offenes Meer,
Ebenen) bekommen wenige grosse Dreiecke, Klippen/Detailbereiche bleiben bei
voller Pixel-Aufloesung.

WAS DIESES MODUL NICHT KANN, UND WARUM DAS KEIN FEHLER IST
(docs/OFFENE_PUNKTE.md 6.19, Nutzerbefund 2026-08-13: "die kuesten sind
immernoch 90 Grad ... ich habe nicht das gefuehl dass das mesh ausser x und y
auch z in betracht zieht")

Die Beobachtung stimmt, die Ursache liegt aber eine Ebene tiefer: **eine
Heightmap speichert je (x,y) GENAU EINEN Hoehenwert.** Zwei benachbarte Pixel
mit 180 m und -3 m ergeben zwangslaeufig eine senkrechte Flaeche von einem
Pixel Breite. Und weil die Kuestenlinie dem Pixelraster folgt, wird daraus
die sichtbare Treppe.

Dieses Modul kann das PRINZIPIELL nicht beheben und war nie dafuer gedacht -
es fasst Rasterzellen zusammen, seine Vertices liegen gemessen 0.000004 px
von einer Pixelecke entfernt, also exakt darauf. Wer die Treppe loswerden
will, braucht Vertices, die frei liegen duerfen:

  * `gui/widgets/terrain_remesh.py` (6.33) verschiebt sie per
    QEM-Decimation - hilft ueber die Flaeche, loest die Kuestenlinie aber
    NICHT vom Raster (dort nachgemessen).
  * Wirklich rasterfrei wird die Kueste erst mit ihr als Zwangskante
    (Constrained Delaunay) oder ueber den Vektorweg
    (`core/vektor_kueste.py`), der die Kueste als Polylinie mit Stationen in
    Metern fuehrt statt als Pixelmaske.

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


def _fehler_pyramide(H, N):
    """
    Funktionsweise: Bit-identisches Ergebnis zu `_quad_fehler()`, aber
    ebenenweise bottom-up als numpy-Arrays statt rekursiv je Quadrant
    (2026-08-13, OFFENE_PUNKTE 6.18b). Rueckgabe: dict Groesse -> (N/s, N/s)
    float64-Array mit dem Fehlerwert jedes Quadranten dieser Groesse.

    Aufgabe: `_quad_fehler()` war nach der Vektorisierung von Balancierung und
    Triangulierung der verbliebene Engpass - gemessen 0.835 s von 0.883 s
    Gesamtzeit bei 1024 px, weil es fuer jeden der rund 1.4 Mio.
    Quadtree-Knoten einen Python-Funktionsaufruf samt dict-Zugriff macht. Hier
    stattdessen: je Groessenebene EIN Satz Slices ueber das ganze Gitter.

    ZUR BIT-IDENTITAET: die alte Fassung rechnet `float(h00+h10+h01+h11)` -
    die Summe entsteht also in float32 (numpy-Skalare), erst danach wird auf
    Python-float (float64) erweitert und mit 0.25 multipliziert. Genau diese
    Reihenfolge wird hier nachgebaut (Summe in float32, dann `.astype(float64)`),
    sonst weicht das Ergebnis in den letzten Stellen ab und die
    Blatt-Auswahl an der Toleranzgrenze koennte kippen.
    """
    ebenen = {1: np.zeros((N, N), dtype=np.float64)}
    size = 2
    while size <= N:
        s = size
        half = s // 2
        # Eckpunkte des Quadranten - H hat (N+1, N+1) Punkte, die Slices
        # liefern je (N/s, N/s) Werte, einen je Quadrant dieser Ebene.
        h00 = H[0:N:s, 0:N:s]
        h10 = H[0:N:s, s::s]
        h01 = H[s::s, 0:N:s]
        h11 = H[s::s, s::s]

        mitte = H[half:N:s, half:N:s]
        kante_oben = H[0:N:s, half:N:s]
        kante_unten = H[s::s, half:N:s]
        kante_links = H[half:N:s, 0:N:s]
        kante_rechts = H[half:N:s, s::s]

        # float32-Summen zuerst (wie die alte Fassung), dann auf float64
        eigen = np.abs(mitte.astype(np.float64)
                       - 0.25 * (h00 + h10 + h01 + h11).astype(np.float64))
        eigen = np.maximum(eigen, np.abs(kante_oben.astype(np.float64)
                                         - 0.5 * (h00 + h10).astype(np.float64)))
        eigen = np.maximum(eigen, np.abs(kante_unten.astype(np.float64)
                                         - 0.5 * (h01 + h11).astype(np.float64)))
        eigen = np.maximum(eigen, np.abs(kante_links.astype(np.float64)
                                         - 0.5 * (h00 + h01).astype(np.float64)))
        eigen = np.maximum(eigen, np.abs(kante_rechts.astype(np.float64)
                                         - 0.5 * (h10 + h11).astype(np.float64)))

        kinder = ebenen[half]
        kinder_max = np.maximum(
            np.maximum(kinder[0::2, 0::2], kinder[0::2, 1::2]),
            np.maximum(kinder[1::2, 0::2], kinder[1::2, 1::2]))

        ebenen[s] = np.maximum(eigen, kinder_max)
        size *= 2
    return ebenen


def _blaetter_sammeln_pyramide(ebenen, N, toleranz, min_size):
    """
    Funktionsweise: Top-down Blattauswahl wie `_blaetter_sammeln()`, aber
    ebenenweise ueber die Fehler-Pyramide statt rekursiv je Quadrant.
    Ein Quadrant wird zum Blatt, wenn er die Mindestgroesse erreicht hat oder
    sein Fehler die Toleranz unterschreitet - sonst wandern seine vier Kinder
    eine Ebene tiefer.
    Rueckgabe: (x0s, y0s, sizes) als drei parallele int64-Arrays.
    """
    x0s_teile, y0s_teile, sizes_teile = [], [], []

    # Startebene: der eine Wurzelquadrant (0,0,N), in Quadrant-Koordinaten (0,0).
    offen_i = np.zeros(1, dtype=np.int64)
    offen_j = np.zeros(1, dtype=np.int64)
    size = N

    while True:
        fehler = ebenen[size][offen_j, offen_i]
        ist_blatt = (size <= min_size) | (fehler <= toleranz)

        if ist_blatt.any():
            x0s_teile.append(offen_i[ist_blatt] * size)
            y0s_teile.append(offen_j[ist_blatt] * size)
            sizes_teile.append(np.full(int(ist_blatt.sum()), size, dtype=np.int64))

        weiter_i = offen_i[~ist_blatt]
        weiter_j = offen_j[~ist_blatt]
        if len(weiter_i) == 0:
            break

        # Vier Kinder je offenem Quadranten, in den Koordinaten der naechsten Ebene
        offen_i = np.concatenate([weiter_i * 2, weiter_i * 2 + 1,
                                  weiter_i * 2, weiter_i * 2 + 1])
        offen_j = np.concatenate([weiter_j * 2, weiter_j * 2,
                                  weiter_j * 2 + 1, weiter_j * 2 + 1])
        size //= 2

    if not sizes_teile:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64))
    return (np.concatenate(x0s_teile), np.concatenate(y0s_teile),
            np.concatenate(sizes_teile))


def _blaetter_sammeln(cache, x0, y0, size, toleranz, min_size, blaetter):
    """Top-down Auswahl: Blatt wird zu einem Blatt, wenn seine Groesse die
    Mindestgroesse erreicht hat oder sein (gecachter) Fehler die Toleranz
    unterschreitet - sonst rekursiv in 4 Kinder teilen.
    Rekursive dict-Fassung, seit 2026-08-13 nur noch vom Vergleichstest
    (tests/smoke_test_adaptive_mesh_vectorized.py) als Referenz benutzt - der
    Aufbau laeuft ueber `_blaetter_sammeln_pyramide()`."""
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
    besitzt jede kleinste Zelle - die Nachschlage-Struktur fuer Nachbar-Abfragen.
    Dict-Fassung, nur noch fuer den headless-Vergleichstest gegen die
    vektorisierte Fassung unten (`_besitzer_gitter_aus_arrays`) gebraucht -
    der eigentliche Aufbau (`_blaetter_balancieren`/`_dreiecke_aus_blaettern`)
    nutzt seit 2026-08-13 nur noch die Array-Fassung, siehe dort."""
    besitzer = np.zeros((zellen, zellen), dtype=np.int64)
    for (x0, y0), size in blaetter.items():
        c = size // min_size
        cx, cy = x0 // min_size, y0 // min_size
        besitzer[cy:cy + c, cx:cx + c] = size
    return besitzer


def _besitzer_gitter_aus_arrays(x0s, y0s, sizes, zellen, min_size):
    """
    Funktionsweise: Bit-identisches Ergebnis zu `_besitzer_gitter()`, aber EIN
    Durchgang je VORKOMMENDER Blattgroesse statt je Blatt (2026-08-13,
    OFFENE_PUNKTE 6.18b - bei echtem 1024px-Gelaende 194068 Blaetter, aber
    nur ~11 verschiedene Groessen, da jede eine Zweierpotenz zwischen
    min_size und N ist). Fuer jede Groesse werden alle ihre Blaetter in EINEM
    numpy-Aufruf per Broadcasting eingetragen (kein Python-Loop ueber
    einzelne Blaetter) - Ueberlappungen zwischen Gruppen sind ausgeschlossen,
    weil Blaetter den Raum ueberlappungsfrei zerlegen.
    """
    besitzer = np.zeros((zellen, zellen), dtype=np.int64)
    if len(sizes) == 0:
        return besitzer
    cxs = x0s // min_size
    cys = y0s // min_size
    cs = sizes // min_size
    for c in np.unique(cs):
        c = int(c)
        maske = cs == c
        cx = cxs[maske]
        cy = cys[maske]
        s = int(sizes[maske][0])
        off = np.arange(c)
        zeilen = cy[:, None, None] + off[None, :, None]
        spalten = cx[:, None, None] + off[None, None, :]
        besitzer[zeilen, spalten] = s
    return besitzer


def _kanten_minima_je_seite(besitzer, cx, cy, c, zellen, eigene_groesse):
    """
    Funktionsweise: Fuer eine Gruppe gleich grosser Blaetter (Zellkoordinaten
    `cx`/`cy`, Ausdehnung `c` Zellen) die kleinste angrenzende Blattgroesse je
    Seite - vier Arrays der Laenge len(cx), in der Reihenfolge
    (links, rechts, oben, unten). Wo es keinen Nachbarn gibt (Kartenrand),
    steht `eigene_groesse` (neutral - kann nie kleiner sein als man selbst).

    Aufgabe: Ersetzt die Slice-Abfrage `besitzer[cy:cy+c, cx-1].min()` der
    alten, blattweisen Fassung durch EIN Gather ueber alle Blaetter der Gruppe.
    Bewusst gezieltes Gathern (`besitzer[zeilen, spalten]`, K*c Elemente)
    statt eines gleitenden Minimums ueber das ganze Gitter - letzteres war der
    erste Anlauf und skalierte mit `zellen**2` statt mit der beruehrten
    Kantenlaenge, gemessen halb so schnell wie die alte Fassung bei wenigen
    grossen Blaettern (siehe _blaetter_balancieren()-Docstring).
    """
    anzahl = len(cx)
    off = np.arange(c)
    zeilen = cy[:, None] + off[None, :]      # (K, c) - Zeilenband des Blattes
    spalten = cx[:, None] + off[None, :]     # (K, c) - Spaltenband des Blattes

    def seite(hat_nachbar, zeilen_idx, spalten_idx):
        werte = np.full(anzahl, eigene_groesse, dtype=np.int64)
        if hat_nachbar.any():
            werte[hat_nachbar] = besitzer[zeilen_idx, spalten_idx].min(axis=1)
        return werte

    hat_links = cx > 0
    links = seite(hat_links, zeilen[hat_links], (cx[hat_links] - 1)[:, None])

    hat_rechts = (cx + c) < zellen
    rechts = seite(hat_rechts, zeilen[hat_rechts], (cx[hat_rechts] + c)[:, None])

    hat_oben = cy > 0
    oben = seite(hat_oben, (cy[hat_oben] - 1)[:, None], spalten[hat_oben])

    hat_unten = (cy + c) < zellen
    unten = seite(hat_unten, (cy[hat_unten] + c)[:, None], spalten[hat_unten])

    return links, rechts, oben, unten


def _kanten_minima(besitzer, cx, cy, c, zellen, eigene_groesse):
    """Kleinste angrenzende Blattgroesse ueber ALLE vier Seiten - genau der
    `kleinster_nachbar`-Wert der alten, blattweisen Balancier-Schleife."""
    links, rechts, oben, unten = _kanten_minima_je_seite(
        besitzer, cx, cy, c, zellen, eigene_groesse)
    return np.minimum(np.minimum(links, rechts), np.minimum(oben, unten))


def _blaetter_balancieren(x0s, y0s, sizes, N, min_size):
    """
    Funktionsweise: Erzwingt die "restricted quadtree"-Eigenschschaft - kein
    Blatt darf mehr als doppelt so gross sein wie sein feinster angrenzender
    Nachbar. Iteriert bis zum Fixpunkt (garantiert terminierend, da jede
    Runde nur Groessen verkleinert, min_size als Untergrenze).
    Aufgabe: Ist Voraussetzung fuer die Kanten-Faecher-Triangulierung unten -
    ohne diese Balance koennte ein Blatt einen um 2+ Stufen feineren
    Nachbarn haben, was ein einzelner Kanten-Mittelpunkt nicht mehr flicken kann.

    VEKTORISIERT (2026-08-13, OFFENE_PUNKTE 6.18b, Nachricht des Nutzers "dann
    weiter"): die alte Fassung lief in einem Python-Loop ueber jedes einzelne
    Blatt - bei 194068 Blaettern (echtes 1024px-Gelaende, Nutzerlog
    2026-08-13) der gemessene Hauptanteil der 15.7s/33.3s Netzaufbauzeit.
    Ersetzt durch: Blaetter nach Groesse gruppiert (Python-Loop nur ueber die
    ~11 VORKOMMENDEN Groessen), pro Gruppe die vier Kanten-Minima fuer ALLE
    Blaetter der Gruppe gleichzeitig ueber `_kanten_minima()` geholt.

    ERSTER ANLAUF WAR MESSBAR LANGSAMER und wurde verworfen: er nutzte
    `sliding_window_view(besitzer, c).min(axis=-1)`, was ein gleitendes
    Minimum ueber das GESAMTE Gitter rechnet - also mit `zellen**2` skaliert,
    unabhaengig davon, wie wenige Blaetter dieser Groesse es gibt. Gemessen
    bei 1024px/3019 Blaettern: 1.63s alt gegen 3.24s neu, also **halb so
    schnell**. Aufgefallen nur, weil der Vergleichstest die Zeiten beider
    Fassungen nebeneinander ausgibt statt nur die Gleichheit zu pruefen. Die
    jetzige Fassung gathert stattdessen gezielt die `K*c` Randzellen der
    Gruppe (siehe `_kanten_minima()`) und skaliert damit mit der tatsaechlich
    beruehrten Kantenlaenge statt mit der Gitterflaeche.

    Ergebnis bit-identisch zur alten Fassung, siehe
    tests/smoke_test_adaptive_mesh_vectorized.py (Vergleich alt/neu auf
    echten Kartengroessen, nicht nur synthetischen Testgroessen - siehe
    CLAUDE.md-Lehre zu 6.16, dieselbe Falle sollte hier nicht wiederholt werden).
    """
    zellen = N // min_size

    while True:
        besitzer = _besitzer_gitter_aus_arrays(x0s, y0s, sizes, zellen, min_size)
        cxs = x0s // min_size
        cys = y0s // min_size
        cs = sizes // min_size

        muss_teilen = np.zeros(len(sizes), dtype=bool)

        for c in np.unique(cs):
            c = int(c)
            if c >= zellen:
                continue  # einziges Blatt deckt die ganze Karte, kein Nachbar moeglich

            idx = np.nonzero(cs == c)[0]
            s = int(sizes[idx[0]])
            if s <= min_size:
                continue  # kann nicht weiter geteilt werden, Nachbarwert irrelevant

            kleinster = _kanten_minima(besitzer, cxs[idx], cys[idx], c, zellen, s)
            teilen_lokal = kleinster < (s // 2)
            if teilen_lokal.any():
                muss_teilen[idx[teilen_lokal]] = True

        if not muss_teilen.any():
            return x0s, y0s, sizes

        bleiben = ~muss_teilen
        halbe = sizes[muss_teilen] // 2
        x0_teil = x0s[muss_teilen]
        y0_teil = y0s[muss_teilen]

        x0s = np.concatenate([x0s[bleiben], x0_teil, x0_teil + halbe, x0_teil, x0_teil + halbe])
        y0s = np.concatenate([y0s[bleiben], y0_teil, y0_teil, y0_teil + halbe, y0_teil + halbe])
        sizes = np.concatenate([sizes[bleiben], halbe, halbe, halbe, halbe])


def _dreiecke_aus_blaettern(x0s, y0s, sizes, N, min_size):
    """
    Funktionsweise: Baut Vertex-Liste (Gitterkoordinaten) + Dreiecks-Liste
    (Vertex-Indices) aus den balancierten Blaettern. Fasst geteilte
    Eckpunkte/Kanten-Mittelpunkte automatisch zu identischen Vertices auf
    beiden Seiten einer gemeinsamen Kante zusammen - Voraussetzung fuer
    Risslosigkeit.
    Aufgabe: Fächer-Reihenfolge (BL,BR,TR,TL) reproduziert exakt die Diagonale
    und Wicklung des bisherigen Gleichmaessig-Gitters (dort: Dreieck1 =
    TL,BL,TR; Dreieck2 = TR,BL,BR - beide nutzen die BL-TR-Diagonale), was bei
    aktivem Backface-Culling (glCullFace(GL_BACK), glFrontFace(GL_CW) in
    map_display_3d.py) zwingend ist, sonst wuerden neue Dreiecke von hinten
    weggeschnitten.

    VEKTORISIERT (2026-08-13, OFFENE_PUNKTE 6.18b): die alte Fassung baute
    Polygon und Dreiecke in einem Python-Loop ueber jedes einzelne Blatt samt
    einem dict-basierten Koordinaten->Index-Cache fuer die Vertex-
    Zusammenfassung - bei 194068 Blaettern der zweite Hauptanteil der
    gemessenen Netzaufbauzeit. Ersetzt durch: Blaetter zunaechst nach den
    VIER Nachbar-Flags gruppiert (max. 16 Kombinationen: hat ein Blatt an
    Unten/Rechts/Oben/Links einen feineren Nachbarn oder nicht - bestimmt,
    welche Kanten-Mittelpunkte ins Polygon kommen). Jede Kombination hat eine
    FESTE Polygon-Vertex-Schablone (z.B. nur BL/BR/TR/TL ohne jeden
    Mittelpunkt fuer ein min_size-Blatt), deshalb kann die Fächer-
    Triangulierung fuer alle Blaetter EINER Kombination gleichzeitig gebaut
    werden statt Blatt fuer Blatt. Die Vertex-Zusammenfassung selbst laeuft
    am Ende in einem einzigen `numpy.unique()` ueber die kodierten (x,y)-
    Koordinaten aller Dreieckspunkte, statt ueber ein Python-dict.
    Reihenfolge der Vertex-Indizes weicht dadurch von der alten (dict-
    Einfuegereihenfolge-basierten) Fassung ab - das ist folgenlos, kein
    Downstream-Code haengt an konkreten Index-Werten, nur an der
    Dreiecksgeometrie selbst. Bit-identische Geometrie zur alten Fassung
    verifiziert in tests/smoke_test_adaptive_mesh_vectorized.py.
    """
    n_blaetter = len(sizes)
    if n_blaetter == 0:
        return [], []

    zellen = N // min_size
    besitzer = _besitzer_gitter_aus_arrays(x0s, y0s, sizes, zellen, min_size)

    x1s = x0s + sizes
    y1s = y0s + sizes
    xms = x0s + sizes // 2
    yms = y0s + sizes // 2

    hat_unten = np.zeros(n_blaetter, dtype=bool)
    hat_rechts = np.zeros(n_blaetter, dtype=bool)
    hat_oben = np.zeros(n_blaetter, dtype=bool)
    hat_links = np.zeros(n_blaetter, dtype=bool)

    cxs = x0s // min_size
    cys = y0s // min_size
    cs = sizes // min_size

    for c in np.unique(cs):
        c = int(c)
        idx = np.nonzero(cs == c)[0]
        s = int(sizes[idx[0]])
        if s <= min_size or c >= zellen:
            continue  # min_size-Blaetter bekommen nie einen Mittelpunkt

        links, rechts, oben, unten = _kanten_minima_je_seite(
            besitzer, cxs[idx], cys[idx], c, zellen, s)
        # `< s` entspricht exakt der alten Bedingung `n_richtung < size` -
        # wo es keinen Nachbarn gibt, steht `s` selbst und die Bedingung ist
        # damit False, genau wie das alte `n_richtung is not None`.
        hat_links[idx] = links < s
        hat_rechts[idx] = rechts < s
        hat_oben[idx] = oben < s
        hat_unten[idx] = unten < s

    # Feste Vertex-Formeln je Schablonen-Position (siehe Docstring-Grafik:
    # BL -> [mid_unten] -> BR -> [mid_rechts] -> TR -> [mid_oben] -> TL -> [mid_links]).
    schablonen_positionen = {
        "BL": (x0s, y1s), "mid_unten": (xms, y1s), "BR": (x1s, y1s),
        "mid_rechts": (x1s, yms), "TR": (x1s, y0s), "mid_oben": (xms, y0s),
        "TL": (x0s, y0s), "mid_links": (x0s, yms),
    }

    kombi = (hat_unten.astype(np.int8) | (hat_rechts.astype(np.int8) << 1) |
             (hat_oben.astype(np.int8) << 2) | (hat_links.astype(np.int8) << 3))

    alle_dreieck_xy = []
    for k in np.unique(kombi):
        idx = np.nonzero(kombi == k)[0]
        reihenfolge = ["BL"]
        if k & 1:
            reihenfolge.append("mid_unten")
        reihenfolge.append("BR")
        if k & 2:
            reihenfolge.append("mid_rechts")
        reihenfolge.append("TR")
        if k & 4:
            reihenfolge.append("mid_oben")
        reihenfolge.append("TL")
        if k & 8:
            reihenfolge.append("mid_links")

        polygon_x = np.stack([schablonen_positionen[name][0][idx] for name in reihenfolge], axis=1)
        polygon_y = np.stack([schablonen_positionen[name][1][idx] for name in reihenfolge], axis=1)

        for i in range(1, len(reihenfolge) - 1):
            dreieck_xy = np.stack([
                np.stack([polygon_x[:, 0], polygon_y[:, 0]], axis=1),
                np.stack([polygon_x[:, i], polygon_y[:, i]], axis=1),
                np.stack([polygon_x[:, i + 1], polygon_y[:, i + 1]], axis=1),
            ], axis=1)  # (K, 3, 2)
            alle_dreieck_xy.append(dreieck_xy)

    dreieck_xy = np.concatenate(alle_dreieck_xy, axis=0)  # (Dreiecke, 3, 2)
    flach = dreieck_xy.reshape(-1, 2)  # (Dreiecke*3, 2)

    schluessel = flach[:, 0].astype(np.int64) * (N + 2) + flach[:, 1].astype(np.int64)
    eindeutige_schluessel, inverse = np.unique(schluessel, return_inverse=True)

    positionen = list(zip((eindeutige_schluessel // (N + 2)).tolist(),
                          (eindeutige_schluessel % (N + 2)).tolist()))
    dreieck_indizes = inverse.reshape(-1, 3)
    dreiecke = [tuple(t) for t in dreieck_indizes.tolist()]

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

    # Durchgehend als parallele Arrays statt als dicts (2026-08-13,
    # OFFENE_PUNKTE 6.18b) - Fehlerberechnung, Blattauswahl, Balancierung und
    # Triangulierung arbeiten alle vektorisiert.
    ebenen = _fehler_pyramide(H, N)
    x0s, y0s, sizes = _blaetter_sammeln_pyramide(ebenen, N, fehler_toleranz_m, min_leaf_size)
    if len(sizes) == 0:
        return None

    x0s, y0s, sizes = _blaetter_balancieren(x0s, y0s, sizes, N, min_leaf_size)

    positionen, dreiecke = _dreiecke_aus_blaettern(x0s, y0s, sizes, N, min_leaf_size)
    if not dreiecke:
        return None
    blaetter = dict(zip(zip(x0s.tolist(), y0s.tolist()), sizes.tolist()))
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


# ZWISCHENSPEICHER UEBER ALLE REITER (2026-08-13, OFFENE_PUNKTE 6.18).
# Jeder Reiter hat ein eigenes MapDisplay3D-Widget und baute deshalb dasselbe
# Netz erneut - im Nutzerlog meldeten Terrain und Geologie exakt dieselben
# Zahlen (450193 Dreiecke, 194068 Blaetter), bei 15.7 s bzw. 33.3 s. Bei zehn
# Reitern zehnmal dieselbe Rechnung. Der Schluessel geht ueber den INHALT der
# Heightmap, nicht ueber die Objektidentitaet.
_MESH_CACHE = {}
_MESH_CACHE_MAX = 3   # Terrain-Rohform, kombinierte Form, eine Reserve


def _cache_schluessel(heightmap, terrain_scale_factor, terrain_height_scale,
                      fehler_toleranz_m, min_leaf_size):
    """
    Funktionsweise: Inhaltsschluessel aus der Heightmap plus allen Groessen,
    die das Ergebnis beeinflussen.
    Aufgabe: `hash(bytes)` ueber ein 1024x1024-float32-Feld kostet rund 1 ms -
    gegenueber 15 s Netzaufbau vernachlaessigbar. Bewusst der INHALT und nicht
    `id()`: `get_terrain_data_combined()` liefert bei jedem Aufruf ein neues
    Array (`.copy()`), eine Identitaetspruefung ginge also immer daneben.
    """
    h = heightmap
    if not h.flags["C_CONTIGUOUS"]:
        h = np.ascontiguousarray(h)
    return (h.shape, h.dtype.str, hash(h.tobytes()),
            round(float(terrain_scale_factor), 9),
            round(float(terrain_height_scale), 12),
            round(float(fehler_toleranz_m), 6), int(min_leaf_size))


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

    Ergebnisse werden ueber den Heightmap-INHALT zwischengespeichert (siehe
    _MESH_CACHE) - alle Reiter teilen sich dasselbe Netz, statt es je Reiter
    neu zu bauen. `stats["aus_cache"]` sagt, ob gerechnet wurde.
    """
    if ist_fuer_adaptives_mesh_geeignet(heightmap):
        schluessel = _cache_schluessel(heightmap, terrain_scale_factor,
                                       terrain_height_scale, fehler_toleranz_m,
                                       min_leaf_size)
        treffer = _MESH_CACHE.get(schluessel)
        if treffer is not None:
            vertices, indices, stats = treffer
            stats = dict(stats, aus_cache=True)
            return vertices, indices, stats
    else:
        schluessel = None

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
        "aus_cache": False,
    }

    if schluessel is not None:
        # Aeltesten Eintrag verwerfen, wenn voll (einfaches FIFO - bei maximal
        # drei Eintraegen lohnt keine echte LRU-Buchfuehrung). Die Arrays
        # werden von den Reitern nur gelesen und an OpenGL uebergeben.
        if len(_MESH_CACHE) >= _MESH_CACHE_MAX:
            _MESH_CACHE.pop(next(iter(_MESH_CACHE)))
        _MESH_CACHE[schluessel] = (vertices, indices, stats)

    return vertices, indices, stats
