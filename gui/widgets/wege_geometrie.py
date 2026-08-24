"""
Path: gui/widgets/wege_geometrie.py

Wege als echte BANDGEOMETRIE statt als Textur auf dem Gelaende
(Nutzerwunsch 2026-08-13: "es geht mir darum das ich in der karte was
auswaehlen kann und es nicht so schoen aussieht mit den strassen als textur.
ich will es etwas schoener haben ... ich will fuer den editor der nur in
python ist ein bisschen schoenere optik").

WARUM DIE TEXTUR NICHT REICHT

Der bisherige Weg (6.23) zeichnet die Strassen in eine RGBA-Textur und legt
sie ueber das Terrain-Mesh. Das hat zwei Grenzen, die sich nicht wegstellen
lassen:

  * **Aufloesung.** Die Textur hat feste Kantenlaenge (map_size). Zoomt man
    heran, wird die Strasse pixelig - sie hat schlicht nicht mehr Bildpunkte.
  * **Anklickbarkeit.** Eine Textur ist kein Objekt. Es gibt nichts, was ein
    Mausklick treffen koennte.

Ein Band aus echten Dreiecken hat beide Probleme nicht: es bleibt bei jedem
Zoom scharf, weil es Geometrie ist, und jedes Band traegt eine ID.

AUFBAU

Je Wegpunkt werden zwei Vertices erzeugt - einer links, einer rechts der
Laufrichtung. Zwischen zwei aufeinanderfolgenden Punkten entsteht daraus ein
Viereck (zwei Dreiecke). Die Hoehe kommt aus der Heightmap, plus einem kleinen
Aufschlag: das Band soll knapp UEBER dem Gelaende liegen, sonst kaempft es mit
der Terrainflaeche um dieselben Tiefenwerte (Z-Fighting) und flackert.

Die Breite ist bewusst kartografisch ueberhoeht, nicht realistisch - genau wie
die Hoehen (Nutzer: "ist ja auch bei den bergen, das 1000m berge hohe 4000er
symbolisieren"). Eine real 5 m breite Strasse waere auf dieser Karte
unsichtbar.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


# Breite eines Weges in METERN, kartografisch ueberhoeht (siehe Modul-Kopf).
# 2026-08-16 verschmaelert (Nutzer: "koennen im uebrigen noch schmaler
# sein") - die echte Bandgeometrie wirkt schon bei weniger Breite praesent,
# anders als der pixelige Textur-Skin, gegen den die alten 60/45 m geeicht
# waren.
# 2026-08-24 auf 60 % gesetzt (Nutzer: "nur 60% von der dicke die es jetzt
# hat") - vorher 35/26 m. Mit dem Querprofil und dem weichen Rand wirkt das
# Band praesenter als das alte flache Rechteck, es braucht weniger Breite fuer
# dieselbe Lesbarkeit.
WEG_BREITE_M = 21.0
SEEWEG_BREITE_M = 16.0

# Glaettungsfenster fuer das Hoehenprofil entlang eines Weges, in Wegpunkten
# (2026-08-16, Nutzer: "glaetten den weg etwas, so dass wir hier immer etwas
# horizontale wege haben"). Das A* Routing waehlt zwar schon die guenstigste
# Trasse (5.19, Steigungskosten), aber die GEZEICHNETE Hoehe folgt bisher
# roh dem Pixel-fuer-Pixel-Gelaende - eine echte Strasse ist dagegen planiert
# und federt kleine Bodenwellen ab. Ein gleitendes Mittel auf dem bereits
# berechneten `hoehe_band` bildet genau das nach, ohne das Routing selbst
# anzufassen.
GLAETTUNG_FENSTER_PUNKTE = 7

# Glaettung des VERLAUFS in der Kartenebene, nicht nur der Hoehe
# (Nutzerbefund 2026-08-24: "wege sind sehr windy obwohl sie recht dick
# dargestellt werden. koennten also weniger wiggly sein").
#
# URSACHE: das A*-Routing laeuft auf einer 8er-Nachbarschaft im Pixelraster.
# Eine schraege Strecke entsteht dort als Treppe aus abwechselnd geraden und
# diagonalen Schritten - Richtungswechsel um 45 Grad bei jedem zweiten Punkt.
# Auf dem duennen Textur-Skin fiel das kaum auf; als breites Band mit
# sichtbaren Raendern schon.
#
# ZWEI DURCHGAENGE MIT KLEINEM FENSTER statt eines grossen: ein Fenster von 13
# glaettet zwar staerker, schneidet aber echte Kurven ab (der Weg wandert aus
# seinem Korridor). Zweimal 7 kommt auf aehnliche Glaette bei rund halbem
# Versatz.
#
# Die ENDPUNKTE bleiben fest - dort steht eine Siedlung, und der Weg muss sie
# treffen, nicht in ihre Naehe kommen.
GLAETTUNG_XY_FENSTER = 7
GLAETTUNG_XY_DURCHGAENGE = 2

# Untergrenze der Bandbreite in Pixeln - siehe baue_wegbaender().
MINDEST_BREITE_PX = 1.5

# Wie weit das Band ueber dem Gelaende schwebt, als Anteil der Weltgroesse.
#
# 2026-08-24 von 0.0006 auf ein Sechstel gesenkt: gegen Z-Fighting sorgt jetzt
# `glPolygonOffset` im Renderer (map_display_3d._render_wegbaender), und das
# wirkt NUR im Tiefenpuffer, nicht in der Welt. Ein Weltversatz dagegen hebt
# das Band bei flachem Blickwinkel sichtbar vom Boden ab - man sieht unter die
# Strasse. Der kleine Rest bleibt als Sicherheitsnetz fuer Gelaende, das
# zwischen den drei Stuetzstellen des Querprofils hoeher liegt.
SCHWEBE_ANTEIL = 0.0001

# --- Querprofil (Nutzerwunsch 2026-08-24, Methoden 2 und 3) ----------------
#
# Bis dahin war ein Wegband ein flaches Rechteck aus zwei Vertexreihen: quer
# eben, eine Farbe, harte Polygonkante am Rand. Jetzt fuenf Reihen, die
# zusammen ein Strassenprofil bilden:
#
#     Kante   Schulter   Scheitel   Schulter   Kante
#     t=-1     t=-0.68     t=0       t=+0.68    t=+1
#     Deckung 0 ------- 1 --- 1 --- 1 ------- 0      <- weicher Rand
#     Hoehe   +0 ----- woelbung(1-t^2) ------ +0     <- Woelbung
#
# `t` ist die Querkoordinate in Einheiten der halben Bandbreite und liegt als
# Vertexattribut `deckung` (bereits ausgewertet) mit im Puffer.
SCHULTER_ANTEIL = 0.68

# Geometrische Woelbung der Fahrbahnmitte, als Anteil der GANZEN Bandbreite.
# Bewusst klein - eine sichtbare Wulst im Umriss will niemand.
WOELBUNG_ANTEIL = 0.02

# Neigung, mit der die NORMALEN quer zur Fahrtrichtung verkippt werden.
#
# ABSICHTLICH GROESSER ALS DIE GEOMETRISCHE WOELBUNG. Die Schattierung soll
# die Woelbung deutlich zeigen, der Umriss sie nur andeuten - waere beides
# gleich, muesste man fuer sichtbares Relief eine Wulst bauen, die von der
# Seite als Wurst auffaellt. Derselbe Kniff wie bei den kartografisch
# ueberhoehten Berghoehen, nur fuer die Beleuchtung.
NORMALEN_WOELBUNG = 0.40

# Die fuenf Bahnen des Querprofils, als Vielfache der halben Bandbreite.
PROFIL_T = (-1.0, -SCHULTER_ANTEIL, 0.0, SCHULTER_ANTEIL, 1.0)

# Wie viele Stuetzstellen quer zur Fahrtrichtung fuer die HOEHE abgetastet
# werden. Nicht dasselbe wie die fuenf Profilbahnen: die Hoehe braucht
# eine dichtere Abtastung, damit kein Grat zwischen zwei Bahnen das Band
# untertaucht (siehe band_aus_pfad).
QUER_STUETZSTELLEN = 9


def _hoehe_an(heightmap, x, y):
    """Bilinear interpolierte Gelaendehoehe an einer Bruchkoordinate."""
    h, w = heightmap.shape
    x = np.clip(x, 0.0, w - 1.001)
    y = np.clip(y, 0.0, h - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    oben = heightmap[y0, x0] * (1 - fx) + heightmap[y0, x1] * fx
    unten = heightmap[y1, x0] * (1 - fx) + heightmap[y1, x1] * fx
    return oben * (1 - fy) + unten * fy


def _normalen_an(nx_feld, ny_feld, nz_feld, x, y):
    """Bilinear interpolierte, renormierte Normale an einer Bruchkoordinate."""
    h, w = nx_feld.shape
    x = np.clip(x, 0.0, w - 1.001)
    y = np.clip(y, 0.0, h - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0

    def _abtasten(feld):
        oben = feld[y0, x0] * (1 - fx) + feld[y0, x1] * fx
        unten = feld[y1, x0] * (1 - fx) + feld[y1, x1] * fx
        return oben * (1 - fy) + unten * fy

    nx, ny, nz = _abtasten(nx_feld), _abtasten(ny_feld), _abtasten(nz_feld)
    laenge = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    sicher = np.where(laenge > 1e-9, laenge, 1.0)
    return nx / sicher, ny / sicher, nz / sicher


def band_aus_pfad(pfad, heightmap, breite_px, terrain_scale_factor,
                  terrain_height_scale, schwebe, normalenfeld):
    """
    Ein Wegband als (vertices, indices) in WELTkoordinaten.

    `pfad` ist eine Liste von (x, y) in Pixelkoordinaten der Karte. Rueckgabe:
      vertices (N,7) float32 - [Weltposition xyz, Normale xyz, Deckung] je
               Vertex; `Deckung` ist 1 auf der Fahrbahn und laeuft zu den
               Kanten auf 0 - daraus macht wegband.frag den weichen Rand.
      indices  (M,)  uint32  - Dreiecke

    Die Umrechnung Pixel -> Welt ist EXAKT dieselbe wie im Terrain-Mesh
    (adaptive_terrain_mesh.build_adaptive_mesh) - eine zweite Formel hier
    wuerde das Band gegen das Gelaende verschieben, und zwar unauffaellig
    genug, um lange nicht bemerkt zu werden.

    Die NORMALE kommt bilinear aus dem TERRAIN-Normalenfeld
    (`normalenfeld` = (nx, ny, nz)-Tupel gleicher Form wie `heightmap`, aus
    `adaptive_terrain_mesh._normalen_voll()`, EINMAL je Heightmap berechnet
    und von `baue_wegbaender()` an alle Baender durchgereicht), nicht aus den
    Banddreiecken selbst (2026-08-16, Nutzerfeedback: die Baender sollen sich
    "schoen auf die Textur schmiegen" statt unlit/flach zu wirken) - exakt
    dasselbe Vorgehen wie beim Remesh (terrain_remesh.py): eine aus den
    eigenen (fast ebenen) Dreiecken berechnete Normale saehe facettiert aus,
    waehrend das Gelaende glatt schattiert ist.
    """
    punkte = np.asarray([(float(p[0]), float(p[1])) for p in pfad],
                        dtype=np.float64)
    if len(punkte) < 2:
        return (np.zeros((0, 7), dtype=np.float32),
                np.zeros(0, dtype=np.uint32))

    hoehe_px, breite_karte = heightmap.shape

    # VERLAUF GLAETTEN, bevor irgendetwas daraus gebaut wird (siehe
    # GLAETTUNG_XY_FENSTER). Muss VOR der Richtungsberechnung stehen - sonst
    # stehen die Bandkanten senkrecht auf der ungeglaetteten Zickzacklinie und
    # das Band flattert, obwohl seine Mittellinie ruhig laeuft.
    if len(punkte) > 4 and GLAETTUNG_XY_FENSTER >= 3:
        fenster_xy = min(GLAETTUNG_XY_FENSTER, len(punkte))
        if fenster_xy % 2 == 0:
            fenster_xy -= 1
        if fenster_xy >= 3:
            anfang = punkte[0].copy()
            ende = punkte[-1].copy()
            for _ in range(max(1, GLAETTUNG_XY_DURCHGAENGE)):
                punkte[:, 0] = uniform_filter1d(punkte[:, 0], size=fenster_xy,
                                                mode="nearest")
                punkte[:, 1] = uniform_filter1d(punkte[:, 1], size=fenster_xy,
                                                mode="nearest")
            punkte[0] = anfang
            punkte[-1] = ende

    # Laufrichtung je Punkt: zentrale Differenz, an den Enden einseitig.
    richtung = np.zeros_like(punkte)
    richtung[1:-1] = punkte[2:] - punkte[:-2]
    richtung[0] = punkte[1] - punkte[0]
    richtung[-1] = punkte[-1] - punkte[-2]
    laenge = np.hypot(richtung[:, 0], richtung[:, 1])
    laenge = np.where(laenge > 1e-9, laenge, 1.0)
    richtung /= laenge[:, None]

    # Normale in der Kartenebene (senkrecht zur Laufrichtung)
    normale = np.stack([-richtung[:, 1], richtung[:, 0]], axis=1)
    halbe = breite_px * 0.5

    links = punkte + normale * halbe
    rechts = punkte - normale * halbe

    # HOEHE VOM PFADPUNKT, NICHT VOM BANDRAND - das Band ist quer EBEN.
    #
    # Zwei Gruende, und der erste wurde beim Bauen gemessen: nimmt jede
    # Bandkante ihre eigene Gelaendehoehe, kippt das Band am Querhang mit -
    # und weil der Rand bis zu einer halben Bandbreite entfernt liegt, kann
    # er dort TIEFER liegen als das Gelaende unter der Wegmitte. Gemessen
    # lag ein Randvertex dadurch 0.002 Welteinheiten UNTER dem Terrain,
    # obwohl der Schwebeaufschlag 0.006 betrug: das Band waere im Hang
    # versunken. Zweitens ist eine quer ebene Fahrbahn ohnehin das
    # realistischere Bild - eine echte Strasse wird in den Hang geschnitten
    # und folgt ihm nicht seitlich.
    # UEBER DIE GANZE BREITE ABTASTEN, nicht nur an drei Stellen.
    #
    # Bis 2026-08-24 war es das Maximum aus Mitte, linkem und rechtem Rand.
    # Das uebersieht einen Grat ZWISCHEN diesen Stellen - und je breiter das
    # Band im Verhaeltnis zum Raster ist, desto wahrscheinlicher wird das.
    # Gemessen bei 256 px (Band 2.5 px = 208 m breit) lagen danach 9 von 550
    # Vertices unter dem Gelaende. Mit `QUER_STUETZSTELLEN` Abtastungen ueber
    # die volle Breite verschwindet der Fall.
    quer_t = np.linspace(-1.0, 1.0, QUER_STUETZSTELLEN)
    hoehe_band = None
    for t in quer_t:
        stelle = punkte + normale * (halbe * -t)
        h = _hoehe_an(heightmap, stelle[:, 0], stelle[:, 1])
        hoehe_band = h if hoehe_band is None else np.maximum(hoehe_band, h)

    # HOEHENPROFIL GLAETTEN (2026-08-16, Nutzer: "glaetten den weg etwas, so
    # dass wir hier immer etwas horizontale wege haben"). Ohne das folgt die
    # gezeichnete Strasse jeder einzelnen Bodenwelle des Rasters - eine
    # echte Trasse ist dagegen planiert. Gleitendes Mittel entlang der
    # Wegpunktreihe, Fenster an kurze Wege angepasst; 'nearest' haelt Start-
    # und Endhoehe nahe am Original, statt sie gegen den jeweils anderen
    # Wegrand zu verwischen.
    fenster = min(GLAETTUNG_FENSTER_PUNKTE, len(hoehe_band))
    if fenster >= 3:
        if fenster % 2 == 0:
            fenster -= 1
        hoehe_band = uniform_filter1d(hoehe_band, size=fenster, mode="nearest")

    def nach_welt(pkt):
        x = np.clip(pkt[:, 0], 0, breite_karte - 1)
        y = np.clip(pkt[:, 1], 0, hoehe_px - 1)
        pos_x = (x / (breite_karte - 1) - 0.5) * breite_karte * terrain_scale_factor
        pos_z = (y / (hoehe_px - 1) - 0.5) * hoehe_px * terrain_scale_factor
        pos_y = hoehe_band * terrain_height_scale + schwebe
        return np.stack([pos_x, pos_y, pos_z], axis=1)

    # --- Querprofil aufbauen (fuenf Bahnen, siehe PROFIL_T oben) ----------
    #
    # Die Fahrbahn bleibt quer EBEN auf `hoehe_band` - der oben gemessene
    # Versink-Fehler am Querhang bleibt damit behoben. Was dazukommt, ist ein
    # kleiner Aufschlag zur Mitte hin (Woelbung) und ein Deckungswert, der zu
    # den Kanten hin auf 0 laeuft. Der weiche Rand entsteht also in der
    # ALPHA, nicht in der Geometrie: haetten die Kanten ihre eigene
    # Gelaendehoehe, waere genau der alte Fehler wieder da.
    breite_welt = breite_px * terrain_scale_factor
    woelbung_welt = WOELBUNG_ANTEIL * breite_welt

    # Querrichtung in Weltkoordinaten, von links nach rechts. `normale` zeigt
    # zur linken Seite (links = punkte + normale * halbe), also negiert. x und
    # y der Karte skalieren beide mit terrain_scale_factor, die Richtung
    # bleibt dadurch normiert.
    q_x = -normale[:, 0]
    q_z = -normale[:, 1]

    nx_feld, ny_feld, nz_feld = normalenfeld

    n = len(punkte)
    bahnen = len(PROFIL_T)
    vertices = np.empty((bahnen * n, 7), dtype=np.float32)

    for j, t in enumerate(PROFIL_T):
        quer = punkte + normale * (halbe * -t)   # t=-1 ist die linke Kante
        welt = nach_welt(quer)
        # Woelbung: h(t) = woelbung * (1 - t^2), Scheitel in der Mitte
        welt[:, 1] += woelbung_welt * (1.0 - t * t)

        # Normale des Gelaendes an dieser Bahn, quer verkippt.
        #
        # Fuer eine Flaeche h(t) ist der Quer-Anteil der Normalen -dh/dt, und
        # mit h = w(1-t^2) ist dh/dt = -2wt, der Anteil also proportional zu
        # +t. Am linken Rand (t=-1) kippt die Normale nach links - genau das,
        # was man an einer gewoelbten Fahrbahn sieht.
        nx_b, ny_b, nz_b = _normalen_an(nx_feld, ny_feld, nz_feld,
                                        quer[:, 0], quer[:, 1])
        neigung = NORMALEN_WOELBUNG * t
        nx_b = nx_b + neigung * q_x
        nz_b = nz_b + neigung * q_z
        laenge_n = np.sqrt(nx_b ** 2 + ny_b ** 2 + nz_b ** 2)
        laenge_n = np.where(laenge_n > 1e-9, laenge_n, 1.0)

        # Deckung: bis zur Schulter voll, dahinter linear auf 0.
        rand = max(1.0 - SCHULTER_ANTEIL, 1e-6)
        deckung = min(max((1.0 - abs(t)) / rand, 0.0), 1.0)

        vertices[j::bahnen, :3] = welt
        vertices[j::bahnen, 3] = nx_b / laenge_n
        vertices[j::bahnen, 4] = ny_b / laenge_n
        vertices[j::bahnen, 5] = nz_b / laenge_n
        vertices[j::bahnen, 6] = deckung

    # Je Segment und Bahnpaar zwei Dreiecke - also 4 Vierecke quer statt
    # einem. Wicklung wie beim Terrain-Mesh.
    segmente = np.arange(n - 1, dtype=np.int64)
    streifen = []
    for j in range(bahnen - 1):
        a = segmente * bahnen + j
        b = a + 1
        c = a + bahnen
        d = c + 1
        streifen.append(np.stack([a, c, b, b, c, d], axis=1))
    if not streifen:
        return (np.zeros((0, 7), dtype=np.float32), np.zeros(0, dtype=np.uint32))
    indices = np.concatenate(streifen).reshape(-1)
    return vertices, indices.astype(np.uint32)


def baue_wegbaender(wege, heightmap, welt_km, terrain_scale_factor,
                    terrain_height_scale, breite_m=WEG_BREITE_M):
    """
    Alle Wege zu EINEM Vertex-/Indexpuffer zusammenfassen, plus eine
    Zuordnung, welcher Indexbereich zu welchem Weg gehoert.

    Ein einziger Puffer statt eines Draw-Calls je Weg: bei einem Weltnetz mit
    dutzenden Strecken waere sonst der Aufruf-Aufwand groesser als das
    Zeichnen selbst.

    Rueckgabe (vertices, indices, bereiche) mit
      vertices (N,7): [Weltposition xyz, Normale xyz, Deckung] je Vertex - die Farbe
      ist KEIN Vertex-Attribut mehr (siehe wegband.frag: `wegFarbe`-Uniform),
      weil die Beleuchtung jetzt echte Normalen braucht und Vertex-Slots
      knapp sind - eine Kategorie (Land-/Seeweg) hat ohnehin nur eine Farbe.
      bereiche: Liste von (weg_index, index_start, index_anzahl) - die
      Grundlage fuer das spaetere Anklicken einzelner Strassen.
    """
    if heightmap is None or not wege:
        return (np.zeros((0, 7), dtype=np.float32),
                np.zeros(0, dtype=np.uint32), [])

    heightmap = np.asarray(heightmap, dtype=np.float32)
    size = heightmap.shape[0]
    meter_pro_pixel = welt_km * 1000.0 / max(size, 1)
    # MINDESTBREITE IN PIXELN, nicht nur in Metern: auf einer kleinen Karte
    # (256 px = 83 m/px) waere ein 60-m-Weg nur 0.72 Pixel breit und damit
    # praktisch unsichtbar - gemessen beim Bauen. Die Metervorgabe bestimmt
    # die Breite auf grossen Karten, die Pixeluntergrenze haelt sie auf
    # kleinen sichtbar.
    breite_px = max(MINDEST_BREITE_PX, breite_m / meter_pro_pixel)
    # Schwebehoehe in Renderkoordinaten - relativ zur Weltausdehnung, damit
    # sie bei jeder Kartengroesse gleich wirkt.
    schwebe = SCHWEBE_ANTEIL * size * terrain_scale_factor

    # EINMAL je Heightmap, nicht je Weg - dieselbe Normale wird von jedem
    # Band dieser Karte bilinear abgetastet (siehe band_aus_pfad()).
    from gui.widgets.adaptive_terrain_mesh import _normalen_voll
    normalenfeld = _normalen_voll(heightmap, terrain_height_scale, terrain_scale_factor)

    alle_v, alle_i, bereiche = [], [], []
    vertex_versatz = 0
    index_versatz = 0
    for weg_index, weg in enumerate(wege):
        v, i = band_aus_pfad(weg, heightmap, breite_px, terrain_scale_factor,
                             terrain_height_scale, schwebe, normalenfeld)
        if len(v) == 0:
            continue
        bereiche.append((weg_index, index_versatz, len(i)))
        alle_v.append(v)
        alle_i.append(i + vertex_versatz)
        vertex_versatz += len(v)
        index_versatz += len(i)

    if not alle_v:
        return (np.zeros((0, 7), dtype=np.float32),
                np.zeros(0, dtype=np.uint32), [])
    return (np.concatenate(alle_v).astype(np.float32),
            np.concatenate(alle_i).astype(np.uint32), bereiche)
