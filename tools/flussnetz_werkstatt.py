"""
Path: tools/flussnetz_werkstatt.py

WERKSTATT fuer das Flussnetz - ein eigenes Fenster, live bedienbar.

Der Nutzer am 2026-07-30: "ich will das live verstellen koennen und die karten
live sehen, als radio buttons verstellbar ... im grunde will ich nur sehen wie
die fluesse verzweigen und sich aufbauen."

Deshalb: kein Skript, das ein Bild schreibt, sondern ein Fenster mit Reglern
links und der Karte rechts. Jede Reglerbewegung rechnet neu und zeichnet neu.
Radio-Buttons schalten zwischen den vier Stufen um, aus denen das Netz
entsteht:

    Senkenwert   wo Wasser hinwill - Bergspitze niedrig, Mulde hoch
    Dreiecke     die Delaunay-Triangulierung ueber dem Punktsatz
    Flussnetz    der Spannbaum, Linienstaerke nach Einzugsgebiet
    Gelaende     die Flaeche P, in die eingeschnitten wird
    Winkel       Verteilung der Einmuendungswinkel

DER SENKENWERT ist die neue Zutat. Er ist bewusst der RANG innerhalb der
Nachbarschaft, nicht die absolute Hoehe - sonst bekaeme jedes Hochtal den Wert
"will kein Wasser", was falsch ist. Wasser laeuft ins lokal Tiefste.

Damit greifen zwei Gewichte auf die Wegwahl, und der Unterschied ist
wesentlich:

    Rivers Follow Lowland   meidet ANSTIEG (bergauf teuer, oben entlang billig)
    Drain Value Weight      zieht zu MULDEN hin

Das Gelaende kommt aus denselben Vorgabewerten wie die App
(gui/config/value_default.py), damit hier nicht an einer anderen Landschaft
getuned wird als spaeter laeuft.

Aufruf:
    .venv/Scripts/python.exe tools/flussnetz_werkstatt.py
"""

import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

SIZE = 256
KM = 15.0

# =============================================================================
# DIE WELT
# =============================================================================
# Eine Insel. Rauschen minus Kastengradient - der Vorschlag des Nutzers vom
# 2026-08-04, und er loest mehr als er sollte:
#
#   * Die Kueste IST der Auslass. Kein Auslasswinkel aus dem Seed, kein
#     Randabfluss-Preis, kein Ring am Kartenrand - es gibt keinen Kartenrand
#     mehr, an dem sich Wasser sammeln muesste.
#   * Die Richtung, in die Fluesse tendieren, entsteht aus der Inselform.
#   * Wo das lokale Fenster liegt, entscheidet die Landschaft. Das sind die
#     "Regionen" aus dem allerersten Gespraech, ohne eigenen Mechanismus.
#
# BEIDE Anteile sind reine Funktionen der WELTKOORDINATE: das Rauschen ueber
# noise_coord = weltmeter / grundform, der Gradient analytisch. Jede Zoomstufe
# rechnet also exakt dasselbe Feld aus, ohne Interpolation. Nachgemessen am
# 2026-08-04: Ueberlappung zweier Stufen r = 0.997 bzw. 0.983, dieselbe
# Rechnung an der falschen Weltstelle r = -0.03.
WELT_KM = 240.0            # Kantenlaenge der Welt
GRUNDFORM_M = 60000.0      # groesste Rauschform - ein Viertel der Welt
SPREIZUNG = 1.35           # feste Umrechnung Rauschen -> [0,1], NICHT je Fenster
WELT_METER = 3200.0        # Hoehe je Welteinheit, global gleich

# PERSISTENZ - der Regler, der ueber sichtbares Detail entscheidet.
#
# Der Nutzer am 2026-08-04: "die aufloesung der berge wird nicht groesser ...
# die mikroebene sieht aus wie ein grosser gradient". Das lag NICHT an der
# Oktavenzahl - die steigt nach innen korrekt von 8 auf 10 - sondern am
# Amplitudenspektrum. Gemessen als Relief, das die zusaetzlichen Oktaven im
# 15-km-Fenster beitragen:
#
#     Persistenz   0.50    0.58    0.65    0.72
#     davon neu     5 m    18 m    45 m   105 m
#     Anteil       2.7 %   8.3 %  19.7 %  43.1 %
#
# Bei 0.50 halbiert jede Oktave, die zehnte traegt noch 3 m zu ueber 1000 m
# Relief bei - unsichtbar. 0.65 gibt sichtbares Detail, ohne die Weltstufe in
# Rauschen aufzuloesen. (Fraktal gesprochen: 0.5 heisst Hurst-Exponent 1, also
# ungewoehnlich glatt; echtes Gelaende liegt eher bei 0.5 bis 0.8.)
PERSISTENZ = 0.65

# INSELFORM - Kastengradient, Exponent ueber dem Abstand zur Mitte.
#
# <1 laesst den Abfall schon in der Mitte steil ansteigen: viel Meer, kleine
# Inseln. >1 haelt die Mitte lange hoch und faellt erst zum Rand hin steil ab -
# groessere zusammenhaengende Kontinente. Der Nutzer wollte ausdruecklich
# "etwas quadratisch ... so haben wir groessere kontinente, weniger meer in der
# mitte", deshalb 2.0.
INSEL_FORM = 2.0
INSEL_STAERKE = 1.2        # wie tief der Gradient den Rand absenkt

# Farbe je Generation: Makro rot, Meso gruen, Mikro gelb - so benannt vom
# Nutzer. Ein Lauf behaelt die Farbe der Stufe, auf der er ENTSTANDEN ist,
# auch wenn man weiter hineinzoomt.
STUFEN_FARBE = ("#e03030", "#25a03a", "#e8c020")

# Wie billig eine geerbte Kette ist, als Anteil ihrer normalen Kosten.
#
# ZUERST STAND HIER ~0, UND DAS WAR FALSCH. Ein gratis Trog zieht jeden
# Nebenfluss an seinen NAECHSTGELEGENEN Knoten, weil das Weiterfliessen im Trog
# nichts kostet - die Richtung der Einmuendung wird damit beliebig. Gemessen
# als Anteil der Einmuendungen, die flussaufwaerts zeigen (Winkel ueber 90
# Grad), auf der Mikrostufe: 29 % bei gratis. Mit einem echten, nur kleinen
# Preis muss ein Zufluss die Strecke im Trog mitbezahlen und trifft ihn
# deshalb dort, wo es insgesamt am kuerzesten ist - also spitz und
# flussabwaerts zeigend.
ERBE_KOSTEN = 0.12

# Zoomstufen: Ausdehnung x4, Pixel x2 je Stufe. Damit verdoppelt sich die
# Zellgroesse (eine Oktave weniger) und das innere Fenster liegt auf exakten
# Zellgrenzen - 15 km sind auf Meso 128 px und auf Makro 64 px.
STUFEN = (("Makro", 1024, 240.0),
          ("Meso", 512, 60.0),
          ("Mikro", 256, 15.0))


def oktaven_fuer(mpp):
    """So viele Oktaven, wie die Zellgroesse traegt (Wellenlaenge >= 2 px)."""
    k = 0
    while GRUNDFORM_M / (2.0 ** k) >= 2.0 * mpp:
        k += 1
    return max(k, 1)


def weltfeld(size, extent_km, mitte_x_km, mitte_y_km, seed):
    """
    Ein Ausschnitt der Welt, zentriert auf (mitte_x, mitte_y) in km.

    Rueckgabe: (H in Metern ueber Meer, m/px, Oktavenzahl). H < 0 ist See.

    KEINE Normierung auf das Fenster. Genau daran haengt die ganze Pyramide:
    wuerde jede Stufe ihr eigenes Minimum und Maximum auf 0..Amplitude
    strecken, bekaeme dieselbe Weltstelle je nach Zoom eine andere Hoehe. Ein
    Kuestenfenster mit 80 m echtem Relief wuerde auf 1800 m aufgeblasen.
    Gemessen: die Spannen zweier Stufen stehen wie 0.214 zu 1.031, Faktor 4.8.
    """
    from core.terrain_generator import SimplexNoiseGenerator

    mpp = extent_km * 1000.0 / size
    oktaven = oktaven_fuer(mpp)
    links_x = mitte_x_km * 1000.0 - 0.5 * extent_km * 1000.0
    links_y = mitte_y_km * 1000.0 - 0.5 * extent_km * 1000.0

    gen = SimplexNoiseGenerator()
    gen.set_seed(seed)
    n = gen.generate_noise_grid(
        size=size, frequency=mpp / GRUNDFORM_M, octaves=oktaven,
        persistence=PERSISTENZ, lacunarity=2.0,
        offset_x=links_x / mpp, offset_y=links_y / mpp)

    # Weltkoordinate je Pixelmitte - daraus der Gradient, analytisch.
    wx = links_x + (np.arange(size) + 0.5) * mpp
    wy = links_y + (np.arange(size) + 0.5) * mpp
    WX, WY = np.meshgrid(wx, wy, indexing="xy")
    halb = 0.5 * WELT_KM * 1000.0
    kasten = np.maximum(np.abs(WX), np.abs(WY)) / halb   # 0 Mitte, 1 Rand
    abfall = np.power(np.clip(kasten, 0.0, 1.0), INSEL_FORM)

    roh = np.clip(0.5 + SPREIZUNG * n, 0.0, 1.0) - INSEL_STAERKE * abfall
    return roh * WELT_METER, mpp, oktaven


def gelaende_farben():
    """
    Die Gelaendefarben OHNE Blau - ab Gruen aufwaerts.

    matplotlib.terrain beginnt bei Blau (Meer) und wird erst ab etwa 0.25
    gruen. Diese Karten haben aber kein Meer (SPEZIFIKATION §1), also waere
    jede blaue Flaeche eine Falschaussage: sie sieht aus wie Wasser und ist
    doch nur tiefes Land. Deshalb wird die Skala bei 0.25 abgeschnitten.
    """
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    quelle = matplotlib.colormaps["terrain"]
    return LinearSegmentedColormap.from_list(
        "gelaende_ohne_blau", quelle(np.linspace(0.25, 1.0, 256)))


def schummerung(P, mpp, azimut=315.0, hoehe=40.0):
    """Reliefschattierung - macht das Gelaende unter den Linien lesbar."""
    dy, dx = np.gradient(P, mpp)
    neigung = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspekt = np.arctan2(-dx, dy)
    az, hh = np.deg2rad(360.0 - azimut + 90.0), np.deg2rad(hoehe)
    return np.clip(np.sin(hh) * np.sin(neigung)
                   + np.cos(hh) * np.cos(neigung) * np.cos(az - aspekt), 0, 1)


# =============================================================================
# SENKENWERT
# =============================================================================

def senkenwert(P, fenster_px):
    """
    Wie sehr will Wasser hier hinlaufen? 0 = Bergspitze, 1 = Mulde.

    Als RANG innerhalb der Nachbarschaft:

        wert = (P_max_lokal - P) / (P_max_lokal - P_min_lokal)

    Damit bekommt eine Mulde auf 2000 m denselben hohen Wert wie eine auf
    200 m. Ein Senkenwert aus der absoluten Hoehe wuerde jedes Hochtal als
    "will kein Wasser" einstufen - Wasser laeuft aber ins LOKAL Tiefste.
    """
    fenster = max(int(fenster_px), 3)
    hoch = ndimage.maximum_filter(P, size=fenster)
    tief = ndimage.minimum_filter(P, size=fenster)
    return np.clip((hoch - P) / np.maximum(hoch - tief, 1e-9), 0.0, 1.0)


# =============================================================================
# GELAENDE UND NETZ
# =============================================================================

def gelaende(seed, size=SIZE, km=KM):
    """Die Flaeche P - dieselben Vorgaben wie die App, nur ohne Flussnetz."""
    import gui.config.value_default as vd
    from managers.data_lod_manager import DataLODManager
    from core.terrain_generator import BaseTerrainGenerator

    alt = vd.FLUSSNETZ_AKTIV
    vd.FLUSSNETZ_AKTIV = False
    try:
        m = DataLODManager()
        m.set_map_distance_km(km)
        p = {k.lower(): getattr(vd.TERRAIN, k)["default"] for k in
             ("AMPLITUDE", "OCTAVES", "FEATURE_SIZE_M", "PERSISTENCE",
              "LACUNARITY", "REDISTRIBUTE_POWER")}
        p.update({"map_size": size, "map_distance_km": km, "map_seed": seed})
        # Ticket #61: Attributname und Parameterschluessel sind seit der
        # Umbenennung NICHT mehr durch .lower() ineinander umrechenbar (z.B.
        # GULLY_REACH -> "erosion_filter_detail", der ATEF-Quellenname bleibt
        # als Schluessel bestehen). Deshalb hier explizit statt abgeleitet.
        for attr_name, schluessel_suffix in (
                ("STRENGTH", "strength"),
                ("GULLY_SIZE_M", "gully_size_m"),
                ("GULLY_REACH", "detail"),
                ("GULLY_VS_SHARPNESS", "gully_weight"),
                ("RIDGE_ROUNDING", "ridge_rounding"),
                ("VALLEY_ROUNDING", "crease_rounding"),
                ("OCTAVES", "octaves")):
            p["erosion_filter_" + schluessel_suffix] = \
                getattr(vd.EROSION_FILTER, attr_name)["default"]
        lod = int(round(np.log2(max(size, 32) / 32.0))) + 1
        g = BaseTerrainGenerator(data_lod_manager=m)
        g.set_active_parameters(p)
        for n in ("terrain.noise", "terrain.redistribution"):
            m.set_calculator_target_lod(n, lod)
        g._calc_noise("terrain.noise", lod)
        g._calc_redistribution("terrain.redistribution", lod)
        P = m.get_calculator_output("terrain.redistribution", "heightmap", lod)
        return P.astype(np.float64), km * 1000.0 / size
    finally:
        vd.FLUSSNETZ_AKTIV = alt


def in_senken_ziehen(punkte, P, radius_px, mindest_px):
    """
    Jeden Knoten auf den tiefsten Punkt seiner Umgebung ziehen.

    WARUM DAS NOETIG IST. Ein Fluss kann nur von Knoten zu Knoten laufen. Die
    Poisson-Punkte liegen aber gleichmaessig verteilt, also groesstenteils NEBEN
    den Rinnen - liegt kein Knoten in der Talsohle, kann der Lauf dort auch
    nicht verlaufen, egal wie die Kantenkosten aussehen. Die Wegwahl war also
    nie das eigentliche Problem.

    Nach dem Ziehen fallen Knoten zusammen, die in dieselbe Mulde rutschen.
    Die werden ausgeduennt, sonst entstehen entartete Dreiecke.
    """
    from scipy.spatial import cKDTree

    size = P.shape[0]
    r = max(int(round(radius_px)), 1)
    neu = punkte.copy()
    for k in range(len(punkte)):
        py, px = int(punkte[k, 0]), int(punkte[k, 1])
        y0, y1 = max(py - r, 0), min(py + r + 1, size)
        x0, x1 = max(px - r, 0), min(px + r + 1, size)
        fenster = P[y0:y1, x0:x1]
        if fenster.size == 0:
            continue
        j = int(np.argmin(fenster))
        neu[k] = (y0 + j // fenster.shape[1], x0 + j % fenster.shape[1])

    # Ausduennen: wer einem schon behaltenen Knoten zu nahe kommt, faellt weg.
    behalten = []
    baum = cKDTree(neu)
    vergeben = np.zeros(len(neu), dtype=bool)
    for k in range(len(neu)):
        if vergeben[k]:
            continue
        behalten.append(k)
        for j in baum.query_ball_point(neu[k], mindest_px):
            if j != k:
                vergeben[j] = True
    return neu[behalten], np.array(behalten, dtype=np.int64)


def baue_netz(P, mpp, punktzahl, drain_gewicht, lowland, seed, umlenkung_grad,
              randabfluss=1.0, senken_zug=0.0, erbe=None, stufe_index=0):
    """
    Punktsatz, Delaunay, Spannbaum - mit Senkengewicht und Umlenkgrenze.

    `erbe` bringt die Knoten der naechstgroesseren Zoomstufe mit, die in
    dieses Fenster fallen:

        punkte   (m,2)  Lage in Pixeln DIESER Karte
        eltern   (m,)   Index des Knotens flussabwaerts, -1 = verlaesst das
                        Fenster (und ist damit hier ein Auslass)
        stufe    (m,)   auf welcher Generation der Knoten entstanden ist
        zusatz   (m,)   Einzugsgebiet, das von ausserhalb hereinkommt

    DIE GROBEN LAEUFE WERDEN UEBERNOMMEN, NICHT NEU ERFUNDEN. Der Nutzer am
    2026-08-04: "ich will ja das die groben fluesse aus der hoeheren generation
    uebernommen werden und sollen mit jeder genauigkeitsstufe mehr fluesse zu
    sehen sein, die dann ineinander fliessen."

    Dafuer genuegt es nicht, die geerbte Kante einfach in den Graphen zu legen -
    zwischen zwei Makroknoten liegen jetzt viele feine Knoten, die Delaunay
    dazwischenschiebt, und eine direkte Kante waere eine schnurgerade Linie
    quer durchs Gelaende. Stattdessen wird jede geerbte Verbindung durch den
    FEINEN Graphen geroutet und die gefundene Kette auf Kosten ~0 gesetzt. Der
    grobe Lauf bleibt damit als Korridor erhalten, folgt aber dem feineren
    Gelaende, und alles Neue fliesst von selbst hinein, weil die Kette gratis
    ist.
    """
    from scipy.spatial import Delaunay
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    import core.terrain_river_network as rn
    from gui.config.value_default import flussnetz_auslaesse

    size = P.shape[0]
    extent_m = mpp * size
    # Aus der Zielpunktzahl den Mindestabstand: ein Poisson-Satz mit Abstand d
    # belegt rund 0.7 * Flaeche / d^2 Punkte.
    #
    # Gezaehlt wird die LANDFLAECHE, nicht die Kartenflaeche. Ohne das bekam die
    # Weltstufe bei 31 % Land nur 181 statt 625 Knoten - dreimal duenner als die
    # lokale Karte, wo fast alles Land ist. Im Bild waren die Fluesse dort nur
    # noch kurze Stummel statt eines Netzes.
    landanteil = max(float((P > 0.0).mean()), 0.05)
    abstand_m = np.sqrt(0.7 * landanteil * extent_m * extent_m
                        / max(punktzahl, 4))
    punkte = rn.poisson_points(extent_m, abstand_m, seed) / mpp

    # NUR LAND. Ueber See laeuft kein Fluss - und ohne diesen Filter setzt
    # Delaunay Knoten mitten ins Meer.
    yi = np.clip(np.round(punkte[:, 0]).astype(int), 0, size - 1)
    xi = np.clip(np.round(punkte[:, 1]).astype(int), 0, size - 1)
    land = P[yi, xi] > 0.0
    if land.sum() < 8:
        return None
    punkte = punkte[land]

    # KNOTEN IN DIE RINNEN ZIEHEN. Nur die NEUEN - geerbte behalten ihre Lage
    # exakt, sonst wandert ein Makroknoten auf jeder feineren Stufe woanders
    # hin und die Schachtelung bricht.
    if senken_zug > 0.0 and len(punkte) > 8:
        punkte, _ = in_senken_ziehen(
            punkte, P, senken_zug * abstand_m / mpp,
            0.55 * abstand_m / mpp)
    stufe = np.full(len(punkte), stufe_index, dtype=np.int64)

    # GEERBTE KNOTEN VORANSTELLEN. Sie behalten ihre Lage exakt - dadurch
    # liegen die groben Knoten in jeder feineren Stufe an derselben Weltstelle,
    # und der Punktsatz ist geschachtelt statt jedesmal neu gewuerfelt.
    erbe_n = 0
    if erbe is not None and len(erbe["punkte"]):
        eigene = punkte
        geerbt = np.asarray(erbe["punkte"], dtype=np.float64)
        erbe_n = len(geerbt)
        # Neue Punkte verwerfen, die einem geerbten zu nahe kommen - sonst
        # entstehen Doppelknoten und entartete Dreiecke.
        from scipy.spatial import cKDTree
        zu_nah = cKDTree(geerbt).query(eigene)[0] < 0.75 * abstand_m / mpp
        punkte = np.vstack([geerbt, eigene[~zu_nah]])
        stufe = np.concatenate([np.asarray(erbe["stufe"], dtype=np.int64),
                                np.full(int((~zu_nah).sum()), stufe_index,
                                        dtype=np.int64)])

    tri = Delaunay(punkte)
    kanten = set()
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            kanten.add((min(s[a], s[b]), max(s[a], s[b])))
    kanten = np.array(sorted(kanten))

    # KANTEN UEBER SEE VERWERFEN. Der Punktsatz ist auf Land gefiltert, die
    # Kanten sind es nicht - Delaunay verbindet quer ueber Meerengen hinweg,
    # im Bild als gerade Linie zwischen zwei Inseln. Geprueft wird an
    # Stuetzstellen entlang der Kante, nicht nur an den Enden.
    ueber_see = np.zeros(len(kanten), dtype=bool)
    for t in np.linspace(0.0, 1.0, 9)[1:-1]:
        m = punkte[kanten[:, 0]] * (1.0 - t) + punkte[kanten[:, 1]] * t
        my = np.clip(np.round(m[:, 0]).astype(int), 0, size - 1)
        mx = np.clip(np.round(m[:, 1]).astype(int), 0, size - 1)
        ueber_see |= P[my, mx] <= 0.0
    kanten = kanten[~ueber_see]
    if len(kanten) < 4:
        return None

    yi = np.clip(np.round(punkte[:, 0]).astype(int), 0, size - 1)
    xi = np.clip(np.round(punkte[:, 1]).astype(int), 0, size - 1)
    glatt = ndimage.gaussian_filter(P, max(0.25 * abstand_m / mpp, 1.0))
    h = glatt[yi, xi]
    h_norm = (h - h.min()) / max(h.max() - h.min(), 1e-9)

    drain = senkenwert(P, abstand_m / mpp)
    d_knoten = drain[yi, xi]

    laenge = np.linalg.norm(punkte[kanten[:, 0]] - punkte[kanten[:, 1]], axis=1)
    delta = h_norm[kanten[:, 1]] - h_norm[kanten[:, 0]]
    bezug = float(np.median(np.abs(delta) / np.maximum(laenge, 1e-9))) or 1e-6

    def kosten(anstieg, ziel):
        hoch = lowland * np.square(
            np.maximum(anstieg, 0.0) / np.maximum(laenge, 1e-9) / bezug)
        senke = drain_gewicht * (1.0 - d_knoten[ziel])
        return laenge * (1.0 + hoch + senke)

    n = len(punkte)
    kosten_hin = kosten(delta, kanten[:, 1])
    kosten_zurueck = kosten(-delta, kanten[:, 0])

    # ------------------------------------------------- geerbte Laeufe erzwingen
    #
    # Jede Verbindung der groberen Stufe wird durch den FEINEN Graphen geroutet
    # und die gefundene Kette auf Kosten ~0 gesetzt. Damit bleibt der grobe
    # Lauf erhalten, folgt aber dem feineren Gelaende - und alles Neue fliesst
    # von selbst hinein, weil die Kette gratis ist.
    kanten_stufe = np.full(len(kanten), stufe_index, dtype=np.int64)
    fest = set()          # Kanten geerbter Laeufe - die Umlenkung laesst sie
    ketten = []           # je geerbte Verbindung die Kette (oben -> unten)
    if erbe_n:
        vor_matrix = csr_matrix(
            (np.concatenate([kosten_hin, kosten_zurueck]),
             (np.concatenate([kanten[:, 0], kanten[:, 1]]),
              np.concatenate([kanten[:, 1], kanten[:, 0]]))), shape=(n, n))
        kante_nr = {}
        for nr, (a, b) in enumerate(kanten):
            kante_nr[(int(a), int(b))] = nr
            kante_nr[(int(b), int(a))] = nr
        erbe_eltern = np.asarray(erbe["eltern"], dtype=np.int64)
        erbe_stufe = np.asarray(erbe["stufe"], dtype=np.int64)
        quellen = [i for i in range(erbe_n) if 0 <= erbe_eltern[i] < erbe_n]

        # REIHENFOLGE: von der Muendung nach oben. Die Tiefe im geerbten Baum
        # sagt, wie weit ein Knoten vom Ausgang des Fensters entfernt ist -
        # die Troege kommen damit zuerst, die Quellaeste zuletzt.
        erbe_tiefe = np.zeros(erbe_n, dtype=np.int64)
        for i in range(erbe_n):
            k, d = i, 0
            while 0 <= erbe_eltern[k] < erbe_n and d < erbe_n:
                k, d = int(erbe_eltern[k]), d + 1
            erbe_tiefe[i] = d
        # GROBE GENERATION ZUERST. Sonst legt eine Mesokette ihren Weg, und der
        # Makrotrog muendet anschliessend in SIE hinein statt umgekehrt.
        quellen.sort(key=lambda i: (erbe_stufe[i], erbe_tiefe[i]))

        if quellen:
            _, vorg = dijkstra(vor_matrix, indices=quellen,
                               return_predecessors=True)
            # Jeder feine Knoten bekommt HOECHSTENS EINEN Kettenelternknoten.
            #
            # Ohne diese Regel liefen mehrere Ketten durch dieselben feinen
            # Knoten und widersprachen sich in der Richtung - gemessen lagen
            # einzelne Knoten "auf 3 Ketten", und der Baum stand danach genau
            # verkehrt herum (229->12 gefordert, 229->263->248 vorhanden).
            # Trifft eine Kette auf eine schon gelegte, MUENDET sie dort. Das
            # ist ohnehin die richtige Aussage: ein Nebenfluss, der den Trog
            # erreicht, laeuft nicht daneben her.
            ketten_eltern = -np.ones(n, dtype=np.int64)
            for reihe, i in enumerate(quellen):
                ziel = int(erbe_eltern[i])
                pfad, k, schutz = [], ziel, 0
                while k != i and k >= 0 and schutz < 4 * n:
                    pfad.append(k)
                    k, schutz = int(vorg[reihe, k]), schutz + 1
                if k != i:
                    continue                       # kein Weg gefunden
                pfad.append(i)
                pfad.reverse()                     # jetzt i ... ziel
                for oben, unten in zip(pfad[:-1], pfad[1:]):
                    if ketten_eltern[oben] >= 0:
                        break                      # muendet in eine bestehende
                    ketten_eltern[oben] = unten
                    nr = kante_nr.get((oben, unten))
                    if nr is not None:
                        kosten_hin[nr] *= ERBE_KOSTEN
                        kosten_zurueck[nr] *= ERBE_KOSTEN
                        # Der Lauf behaelt die Generation, auf der er
                        # entstanden ist - nicht die des Fensters.
                        kanten_stufe[nr] = min(kanten_stufe[nr],
                                               int(erbe_stufe[i]))
                        fest.add((oben, unten))
                        fest.add((unten, oben))
            ketten = [(int(a), int(b)) for a, b in enumerate(ketten_eltern)
                      if b >= 0]

    # ---------------------------------------------------------------- Auslass
    #
    # DREI FAELLE, und nur der dritte ist noch der alte.
    #
    # 1. Geerbte Knoten, deren Lauf das Fenster verlaesst. Dann sitzt der
    #    Auslass dort, wo die groebere Stufe das Wasser hinausschickt.
    # 2. Sonst: JEDE Kuestenzelle ist Auslass. Das ist die Weltstufe. Kein
    #    Auslasswinkel aus dem Seed noetig, kein Randabfluss-Preis.
    # 3. Weder noch - dann bleibt es beim Winkel aus dem Seed plus Randabfluss.
    haupt = []
    if erbe_n:
        haupt = [i for i in range(erbe_n)
                 if int(np.asarray(erbe["eltern"])[i]) < 0]
    if not haupt:
        meer_nah = ndimage.binary_dilation(
            P <= 0.0, iterations=max(int(round(0.6 * abstand_m / mpp)), 1))
        haupt = np.flatnonzero(meer_nah[yi, xi]).tolist()
    if not haupt:
        anzahl = flussnetz_auslaesse(extent_m / 1000.0)
        start = np.random.default_rng(seed ^ 0x5EED).random() * 2.0 * np.pi
        mitte = 0.5 * size
        for i in range(anzahl):
            w = start + 2.0 * np.pi * i / anzahl
            # Auf den KARTENRAND projizieren, nicht auf einen Kreis - sonst
            # liegt der Auslass fuer die meisten Winkel mitten in der Karte.
            ry, rx = np.cos(w), np.sin(w)
            streckung = mitte / max(abs(ry), abs(rx), 1e-9)
            zy, zx = mitte + streckung * ry, mitte + streckung * rx
            k = int(np.argmin((punkte[:, 0] - zy) ** 2
                              + (punkte[:, 1] - zx) ** 2))
            if k not in haupt:
                haupt.append(k)

    # RANDABFLUSS. Mit nur einem Auslass muss JEDER Knoten dorthin - fuer die
    # gegenueberliegende Ecke ist der Weg am Kartenrand entlang oft billiger
    # als quer durch das Gebirge. Ergebnis war ein geschlossener Ring aus
    # Fluessen laengs des Randes, der ueber die Bergruecken lief.
    #
    # Abhilfe: ein virtueller Knoten n als gemeinsame Wurzel. Er haengt mit
    # Kosten 0 an den Hauptauslaessen und mit einem festen Preis an jedem
    # Randknoten. Dijkstra entscheidet dann selbst - ein Randknoten verlaesst
    # die Karte an Ort und Stelle, sobald der Umweg zum Hauptauslass teurer
    # ist als dieser Preis. Wer den Preis auf 0 setzt, bekommt lauter kurze
    # Laeufe zum naechsten Rand; wer ihn sehr hoch setzt, bekommt den Ring
    # zurueck.
    rand_px = 0.75 * abstand_m / mpp
    # `size`, nicht `size - 1` - die Punkte laufen bis size. Mit der -1 lag der
    # Streifen je nach Aufloesung verschieden weit innen (siehe Kernmodul).
    rand = np.flatnonzero(
        (punkte[:, 0] < rand_px) | (punkte[:, 0] > size - rand_px) |
        (punkte[:, 1] < rand_px) | (punkte[:, 1] > size - rand_px))
    # Mit geerbten Auslaessen ist der Randabfluss ueberfluessig: sie stehen
    # dort, wo das Wasser die Karte wirklich verlaesst. Ein zusaetzlicher Preis
    # am Rand wuerde nur wieder erfundene Auslaesse dazwischenschieben.
    if erbe_n:
        rand = np.empty(0, dtype=np.int64)
    # Ein Hauptauslass liegt selbst am Rand. Bliebe er in beiden Listen,
    # summierte csr_matrix die zwei Eintraege und er verloere seinen Vorrang.
    rand = np.setdiff1d(rand, np.asarray(haupt, dtype=np.int64))
    # Preis relativ zum Netz, nicht in absoluten Zahlen (§4.4): 1.0 heisst
    # "so teuer wie ein Lauf quer ueber die halbe Karte".
    quer = 0.5 * size / max(float(np.median(laenge)), 1e-9)
    preis = randabfluss * quer * float(np.median(kosten_hin))

    quellen = np.concatenate([kanten[:, 0], kanten[:, 1],
                              np.full(len(haupt), n),
                              np.full(len(rand), n)])
    ziele = np.concatenate([kanten[:, 1], kanten[:, 0],
                            np.asarray(haupt, dtype=np.int64), rand])
    # ACHTUNG: scipy.csgraph liest eine 0 in der Matrix als "keine Kante".
    # Die Hauptauslaesse brauchen deshalb einen winzigen positiven Wert statt
    # der 0 - sonst haengen sie ueberhaupt nicht an der Wurzel, und dann ist
    # jeder Randknoten ein Auslass, egal wie hoch der Preis steht.
    werte = np.concatenate([kosten_hin, kosten_zurueck,
                            np.full(len(haupt), 1e-9),
                            np.full(len(rand), max(preis, 1e-9))])
    matrix = csr_matrix((werte, (quellen, ziele)), shape=(n + 1, n + 1))

    erg = dijkstra(matrix, indices=n, return_predecessors=True)
    eltern = erg[1][:n].astype(np.int64)
    entfernung = erg[0][:n]
    # Wer direkt an der Wurzel haengt, ist selbst ein Auslass.
    eltern[eltern == n] = -1
    auslaesse = np.flatnonzero(eltern < 0).tolist()
    reihenfolge = np.argsort(np.where(np.isfinite(entfernung), entfernung, np.inf))
    reihenfolge = reihenfolge[np.isfinite(entfernung[reihenfolge])]

    # ---------------------------------------------- geerbte Kette erzwingen
    #
    # Billig genuegt NICHT. Dijkstra darf eine Kette jederzeit umgehen, wenn
    # ein kurzer Weg daneben insgesamt weniger kostet - der geerbte Lauf reisst
    # dann mittendrin ab und laeuft als feinere Generation weiter. Der Nutzer
    # im Bild: "Fluss rot ist unterbrochen und hat keinen abfluss mehr."
    #
    # Gemessen: 7 Generationsbrueche im Meso, 5 im Mikro - und sie blieben
    # auch, als der Umlenk-Pass ganz abgeschaltet war. Es ist also Dijkstra
    # selbst, nicht die Nachbearbeitung.
    #
    # Deshalb wird die Kette hinterher in den Baum GESCHRIEBEN. Von der
    # Muendung nach aussen, damit das Geruest von unten waechst, und mit
    # Zyklusprobe vor jedem Umhaengen.
    if ketten:
        # Kein Zyklus moeglich: jeder Knoten hat hoechstens einen
        # Kettenelternknoten, und jede Kette endet an einem Knoten, der
        # entweder schon zur Muendung fuehrt oder selbst Auslass ist.
        aus_menge = set(auslaesse)
        for oben, unten in ketten:
            if oben not in aus_menge:
                eltern[oben] = unten
        # Die Reihenfolge haengt jetzt nicht mehr an der Dijkstra-Entfernung -
        # ein erzwungener Elternknoten kann weiter weg liegen. Neu aus der
        # Tiefe im Baum, sonst rechnet die Einzugsgebiets-Summe falsch.
        tiefe = np.zeros(n, dtype=np.int64)
        for i in range(n):
            k, d = i, 0
            while eltern[k] >= 0 and d < n:
                k, d = eltern[k], d + 1
            tiefe[i] = d
        reihenfolge = np.argsort(tiefe, kind="stable")
        auslaesse = np.flatnonzero(eltern < 0).tolist()

    # Umlenkung begrenzen. Nur Nachbarn mit KLEINERER Auslassentfernung sind
    # als neuer Elternknoten erlaubt - damit kann kein Kreis entstehen.
    nachbarn = {i: set() for i in range(n)}
    for a, b in kanten:
        nachbarn[a].add(b)
        nachbarn[b].add(a)
    grenze = np.deg2rad(umlenkung_grad)

    def winkel_bei(kind, mitte_k):
        """Winkel zwischen der Fliessrichtung bei mitte_k und dem Zufluss."""
        if eltern[mitte_k] < 0:
            return 0.0
        r1 = punkte[mitte_k] - punkte[eltern[mitte_k]]
        r2 = punkte[kind] - punkte[mitte_k]
        nn = np.linalg.norm(r1) * np.linalg.norm(r2)
        if nn < 1e-9:
            return 0.0
        return float(np.arccos(np.clip(np.dot(r1, r2) / nn, -1.0, 1.0)))

    # UMLENKUNG BEGRENZEN - fuer ALLE Einmuendungen, auch zwischen den
    # Generationen. Nur Nachbarn mit KLEINERER Auslassentfernung sind als neuer
    # Elternknoten erlaubt, damit kein Kreis entstehen kann.
    #
    # ZWEI FEHLER STECKTEN HIER, beide vom Nutzer im Bild gesehen ("die fluesse
    # sollen alle die regeln befolgen mit den winkeln, auch untereinander"):
    #
    # 1. Die Schwelle wurde mit (1 + h_norm) multipliziert - "steileres Gelaende
    #    darf staerker umlenken". Damit stand sie im Gebirge bei bis zu 150
    #    Grad, also genau dort, wo die meisten Nebenfluesse liegen. Von 557
    #    Einmuendungen wurden 12 umgehaengt, obwohl 21 % ueber der nominellen
    #    Schwelle von 75 Grad lagen. Die Regel lief praktisch leer.
    # 2. Ein Durchgang genuegt nicht: jedes Umhaengen aendert die Winkel der
    #    Zufluesse weiter oben. Jetzt bis zu vier Durchgaenge, oder bis sich
    #    nichts mehr aendert.
    umgehaengt = 0
    for _durchgang in range(4):
        geaendert = 0
        for i in reihenfolge:
            e = eltern[i]
            if e < 0 or eltern[e] < 0:
                continue
            # Geerbte Laeufe bleiben, wie sie sind - sonst haette das Erzwingen
            # keinen Zweck: die Umlenkung wuerde den groben Lauf gleich wieder
            # aufbrechen.
            if (int(i), int(e)) in fest:
                continue
            w = winkel_bei(i, e)
            if w <= grenze:
                continue
            bester, bester_w = None, w
            for k in nachbarn[i]:
                if entfernung[k] >= entfernung[i] or eltern[k] < 0:
                    continue
                wk = winkel_bei(i, k)
                if wk < bester_w:
                    bester, bester_w = k, wk
            if bester is not None:
                eltern[i] = bester
                geaendert += 1
        umgehaengt += geaendert
        if not geaendert:
            break

    ordnung = rn.strahler_order(eltern, reihenfolge)
    # ZUFLUESSE VON AUSSEN. Ein Trog, der von der groesseren Stufe hereinkommt,
    # bringt sein Einzugsgebiet mit - sonst waere er im Fenster nur ein
    # Baechlein mit dem Gebiet, das innerhalb entsteht, und bekaeme kein
    # breites Tal. Der Nutzer dazu: "diese erzeugen dann auch je nach stufe
    # breite taeler."
    flaeche = np.ones(n)
    if erbe_n:
        flaeche[:erbe_n] += np.asarray(erbe["zusatz"], dtype=np.float64)
    for i in reihenfolge[::-1]:
        if eltern[i] >= 0:
            flaeche[eltern[i]] += flaeche[i]

    winkel = []
    for i in range(n):
        e = eltern[i]
        if e < 0 or eltern[e] < 0:
            continue
        r1, r2 = punkte[e] - punkte[eltern[e]], punkte[i] - punkte[e]
        nn = np.linalg.norm(r1) * np.linalg.norm(r2)
        if nn > 1e-9:
            winkel.append(np.rad2deg(np.arccos(
                np.clip(np.dot(r1, r2) / nn, -1, 1))))

    # AUSLAUF BIS INS MEER.
    #
    # Der Auslassknoten ist der letzte Knoten auf Land - zwischen ihm und der
    # Kueste bleibt bis zu ein knapper Knotenabstand Luecke, und der Fluss
    # endete sichtbar im Nichts. Der Nutzer: "ist es moeglich das fluesse auch
    # ins meer fliessen und nicht in der naehe enden? das sieht komisch aus."
    #
    # Die Distanztransformation liefert zu jedem Landpixel gleich den INDEX
    # des naechsten Meerpixels mit - daraus wird das letzte Stueck. Es ist
    # kurz genug, dass eine gerade Strecke genuegt.
    auslauf = {}
    if (P <= 0.0).any():
        _, naechstes = ndimage.distance_transform_edt(P > 0.0,
                                                      return_indices=True)
        for a in auslaesse:
            zy = float(naechstes[0][yi[a], xi[a]])
            zx = float(naechstes[1][yi[a], xi[a]])
            if np.hypot(zy - punkte[a][0],
                        zx - punkte[a][1]) < 2.5 * abstand_m / mpp:
                auslauf[int(a)] = (zy, zx)

    # Kein Meer im Fenster? Dann fuehrt der Lauf zum KARTENRAND weiter, und
    # zwar in die Richtung, in der die groebere Stufe ihn fortsetzt.
    if erbe_n and erbe is not None and "ziel" in erbe:
        erbe_ziel = np.asarray(erbe["ziel"], dtype=np.float64)
        for a in auslaesse:
            if int(a) in auslauf or a >= erbe_n:
                continue
            z = erbe_ziel[a]
            if not np.isfinite(z).all():
                continue
            start = punkte[a]
            letzter = None
            for t in np.linspace(0.0, 1.0, 65):
                p = start * (1.0 - t) + z * t
                if -0.5 <= p[0] <= size - 0.5 and -0.5 <= p[1] <= size - 0.5:
                    letzter = p
                else:
                    break
            if letzter is not None:
                auslauf[int(a)] = (float(letzter[0]), float(letzter[1]))

    # EINLAUF: das Stueck vom Kartenrand bis zum Eintrittsknoten. Gegenstueck
    # zum Auslauf - ohne beides beginnt und endet ein Trog irgendwo im Feld.
    einlauf = {}
    if erbe_n and erbe is not None and "einlauf" in erbe:
        quelle = np.asarray(erbe["einlauf"], dtype=np.float64)
        for k in range(erbe_n):
            if not np.isfinite(quelle[k]).all():
                continue
            start, z = punkte[k], quelle[k]
            letzter = None
            for t in np.linspace(0.0, 1.0, 65):
                p = start * (1.0 - t) + z * t
                if -0.5 <= p[0] <= size - 0.5 and -0.5 <= p[1] <= size - 0.5:
                    letzter = p
                else:
                    break
            if letzter is not None:
                einlauf[int(k)] = (float(letzter[0]), float(letzter[1]))

    # Generation je Baumkante: die des Knotens flussaufwaerts, aber nie groeber
    # als das, was das Erzwingen gesetzt hat.
    kante_stufe_von = {}
    for nr, (a, b) in enumerate(kanten):
        kante_stufe_von[(int(a), int(b))] = kanten_stufe[nr]
        kante_stufe_von[(int(b), int(a))] = kanten_stufe[nr]
    lauf_stufe = np.full(n, stufe_index, dtype=np.int64)
    for i in range(n):
        e = eltern[i]
        if e >= 0:
            lauf_stufe[i] = kante_stufe_von.get((int(i), int(e)), stufe[i])
    # FLUSSABWAERTS DARF DIE GENERATION NIE FEINER WERDEN. Sobald sich ein
    # Makrolauf mit einem Mesolauf vereinigt, ist alles darunter Makro - ein
    # grosser Fluss wird flussabwaerts nicht wieder klein. Ohne diesen Durchgang
    # riss die Farbe an jedem solchen Zusammenfluss ab, und im Bild sah es aus,
    # als endete der rote Fluss mitten in der Karte.
    #
    # Von oben nach unten, damit jeder Knoten seine Zufluesse schon kennt. Der
    # Durchgang deckt zugleich die Auslaesse ab: die haben kein Elternteil und
    # fielen sonst auf die Farbe der aktuellen Stufe zurueck.
    for i in reihenfolge[::-1]:
        e = eltern[i]
        if e >= 0:
            lauf_stufe[e] = min(lauf_stufe[e], lauf_stufe[i])

    return dict(punkte=punkte, kanten=kanten, eltern=eltern, ordnung=ordnung,
                flaeche=flaeche, drain=drain, d_knoten=d_knoten,
                auslaesse=auslaesse, abstand_m=abstand_m, size_px=size,
                umgehaengt=umgehaengt, winkel=np.array(winkel),
                stufe=stufe, lauf_stufe=lauf_stufe, erbe_n=erbe_n,
                auslauf=auslauf, einlauf=einlauf, ketten=ketten)


# =============================================================================
# TAELER EINGRABEN
# =============================================================================

def taeler_eingraben(H, mpp, netz, breite_m=900.0, form=1.3, tiefe_anteil=0.35,
                     rand_weich=1.0, max_hang=1.0):
    """
    Aus dem Liniennetz ein Gelaende mit Taelern machen.

    WARUM DAS IN DIE WERKSTATT GEHOERT. Bis hierher wurden die Laeufe auf ROHES
    RAUSCHEN gezeichnet - und dort gibt es noch keine Taeler, die Rinnen
    entstehen ja erst durch das Eingraben. Ein Lauf sieht deshalb zwangslaeufig
    aus, als laege er quer im Gelaende.

    Gemessen am 2026-08-04: das Knotennetz liegt im Mittel 65 m ueber dem
    tiefsten Punkt seiner Umgebung, ein ECHTES Abflussnetz auf demselben
    Rauschen 49 m - nur Faktor 1.3. An der Wegwahl war also fast nichts mehr zu
    holen; es fehlte das Tal.

        z = P - (P - z_fluss) * (1 - profil(abstand / breite))

    An der Sohle steht z_fluss, weit weg bleibt P unveraendert. Die Breite
    waechst mit dem Einzugsgebiet (hydraulische Geometrie, ~A^0.4), die
    Eintiefung ebenso schwaecher (~A^0.3).
    """
    size = H.shape[0]
    pk, el, fl = netz["punkte"], netz["eltern"], netz["flaeche"]
    reihenfolge = np.argsort(-fl)          # grosse Gebiete zuerst rastern

    # --- 1. Flusshoehen: streng fallend flussabwaerts ---------------------
    yi = np.clip(np.round(pk[:, 0]).astype(int), 0, size - 1)
    xi = np.clip(np.round(pk[:, 1]).astype(int), 0, size - 1)
    z = H[yi, xi].astype(np.float64)
    # Von den Quellen abwaerts: der Empfaenger muss tiefer liegen als jeder
    # Zufluss. GESENKT wird, nicht angehoben - ein angehobener Lauf laege ueber
    # dem Gelaende.
    ordnung = np.argsort(fl)               # kleine Gebiete = weit oben
    gefaelle = 0.001                       # 1 m je km, nur damit es faellt
    for i in ordnung:
        e = el[i]
        if e < 0:
            continue
        strecke_m = float(np.linalg.norm(pk[i] - pk[e])) * mpp
        z[e] = min(z[e], z[i] - gefaelle * strecke_m)

    # --- 2. Einzugsgebiet ENTLANG DES BAUMES glaetten ---------------------
    # Ohne das springt das Einzugsgebiet an jedem Zusammenfluss schlagartig -
    # der Nebenarm hat dort ein viel kleineres Gebiet als der Hauptlauf. Da
    # Breite und Tiefe daran haengen, sah der Trog aus wie eine Wurstkette.
    glatt_fl = fl.astype(np.float64).copy()
    for _ in range(4):
        neu = glatt_fl.copy()
        for i in range(len(glatt_fl)):
            e = el[i]
            if e >= 0:
                neu[i] = 0.5 * glatt_fl[i] + 0.5 * glatt_fl[e]
        glatt_fl = neu
    gebiet = glatt_fl / max(float(glatt_fl.max()), 1.0)

    # --- 3. Eintiefung nach Einzugsgebiet ---------------------------------
    relief = float(H[H > 0].max() - H[H > 0].min()) if (H > 0).any() else 1.0
    z_fluss = z - tiefe_anteil * relief * np.power(gebiet, 0.30)

    # --- 4. Rastern: Sohlenhoehe und Breite je Flusspixel -----------------
    #
    # Die Breite darf NICHT rein proportional zu A^0.4 laufen. Ein Quellast mit
    # Gebiet 1 von 3500 kaeme damit auf 0.036 x 600 m = 22 m, also unter einen
    # Pixel - im Bild waren die Nebentaeler nur noch Kratzer. Deshalb eine
    # Untergrenze: das kleinste Tal ist ein Fuenftel des groessten, nicht ein
    # Dreissigstel.
    sohle = np.full((size, size), np.nan)
    breite = np.zeros((size, size))
    breite_px = breite_m / mpp
    untergrenze = 0.22
    for i in reihenfolge:
        e = el[i]
        if e < 0:
            continue
        strecke = float(np.linalg.norm(pk[i] - pk[e]))
        # Drei Stuetzstellen je Pixel Strecke. Bei 1.5 rissen diagonale
        # Abschnitte auf, die Distanztransformation fand dann Luecken - im
        # Bild als gestrichelte Textur laengs der Talhaenge.
        schritte = max(int(strecke * 3.0), 3)
        w = breite_px * (untergrenze + (1.0 - untergrenze)
                         * gebiet[i] ** 0.40)
        # Mindestens drei Pixel breit, damit ein Quellast eine Rinne ist und
        # kein Einzelpixel-Riss.
        w = max(w, 3.0)
        for t in np.linspace(0.0, 1.0, schritte):
            p = pk[e] * (1.0 - t) + pk[i] * t
            y = int(np.clip(round(p[0]), 0, size - 1))
            x = int(np.clip(round(p[1]), 0, size - 1))
            sohle[y, x] = z_fluss[e] * (1.0 - t) + z_fluss[i] * t
            breite[y, x] = max(breite[y, x], w)

    ist_fluss = np.isfinite(sohle)
    if not ist_fluss.any():
        return H

    # --- 5. Abstand zum naechsten Lauf, samt dessen Sohle und Breite ------
    abstand, index = ndimage.distance_transform_edt(
        ~ist_fluss, return_indices=True)
    z_nah = sohle[index[0], index[1]]
    b_nah = np.maximum(breite[index[0], index[1]], 1e-6)

    # WASSERSCHEIDEN-NAHT ENTSCHAERFEN.
    #
    # "Der naechste Lauf" teilt die Karte in Voronoi-Zellen. An deren Grenze
    # springt z_nah von einem Fluss auf den anderen - im Bild als schnurgerade
    # Kante quer ueber den Hang. Genau in der Mitte zwischen zwei Laeufen zieht
    # das Profil aber immer noch mit rund 20 % am Gelaende, also bleibt der
    # Sprung sichtbar.
    #
    # Deshalb: eine geglaettete Fassung beider Felder, und ueberblendet nach
    # dem Profil - an der Sohle zaehlt der exakte Wert, weit weg der glatte.
    # Dort, wo die Naht liegt, ist das Profil nahe 1, also gewinnt die glatte
    # Fassung und die Kante verschwindet.
    # MESSERSCHNEIDEN ZWISCHEN ZWEI LAEUFEN.
    #
    # Zwei Laeufe koennen sich auf wenige Pixel naehern, obwohl sie im Netz
    # weit auseinanderliegen. Die Eintiefung haengt am Einzugsgebiet, also
    # schneidet der Trog tief und der Quellast daneben flach - gemessen an
    # (36,159): rohes Gelaende glatt (6 m Unterschied), eingegraben 958 gegen
    # 509 m auf drei Pixel. Dazwischen stand eine 83-Grad-Wand.
    #
    # Abhilfe als KEGEL-EROSION auf dem Sohlenfeld: keine Sohle darf ueber
    # einer nahen tieferen Sohle stehen, steiler als max_hang. Formal
    #     z(p) := min ueber q von ( z(q) + max_hang * |p-q| )
    # was grey_erosion mit einem Kegel als Struktur genau leistet. Physikalisch
    # ist das die richtige Aussage: ein Tal, das 450 m ueber einem 300 m
    # entfernten Tal haengt, waere laengst angezapft worden.
    if max_hang > 0.0:
        r = max(int(round(2.0 * breite_px)), 2)
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        kegel = -max_hang * mpp * np.hypot(yy, xx)
        z_nah = ndimage.grey_erosion(
            z_nah, footprint=np.ones((2 * r + 1, 2 * r + 1), dtype=bool),
            structure=kegel)

    sigma = max(0.35 * netz["abstand_m"] / mpp, 1.0)
    z_glatt = ndimage.gaussian_filter(z_nah, sigma)
    b_glatt = ndimage.gaussian_filter(b_nah, sigma)

    # --- 6. Querprofil ----------------------------------------------------
    # 1 - exp(-t^a) laeuft asymptotisch aus, statt an der Talkante hart auf 1
    # zu springen - sonst steht dort eine sichtbare Stufe.
    t = abstand / np.maximum(0.5 * (b_nah + b_glatt), 1e-6)
    d = 1.0 - np.exp(-np.power(np.maximum(t, 0.0), max(rand_weich, 0.2)))
    profil = np.power(d, form)

    z_eff = (1.0 - profil) * z_nah + profil * z_glatt
    neu = H - (H - z_eff) * (1.0 - profil)
    # Meer bleibt Meer.
    return np.where(H > 0.0, neu, H)


# =============================================================================
# UEBERGABE VON EINER ZOOMSTUFE ZUR NAECHSTEN
# =============================================================================

def erbe_bilden(netz, mpp_aussen, mitte_aussen_km, mitte_innen_km,
                extent_innen_km, mpp_innen, innen_abstand_m):
    """
    Die Knoten der groeberen Stufe, die in das innere Fenster fallen.

    Sie werden UEBERNOMMEN, nicht nachgebildet: dieselbe Weltstelle, dieselbe
    Generation, dieselbe Reihenfolge flussabwaerts. Damit ist der Punktsatz
    geschachtelt - ein Makroknoten bleibt auf jeder feineren Stufe derselbe
    Knoten, und der grobe Lauf bleibt derselbe Lauf.

    Rueckgabe: {"punkte": (m,2) in Pixeln der INNEREN Karte,
                "eltern": (m,) Index flussabwaerts, -1 = verlaesst das Fenster,
                "stufe":  (m,) Generation,
                "zusatz": (m,) Einzugsgebiet, das von aussen hereinkommt}
    """
    punkte, eltern, flaeche = netz["punkte"], netz["eltern"], netz["flaeche"]

    # WEITERGEGEBEN WIRD DIE GENERATION DES LAUFS, NICHT DIE DES KNOTENS.
    #
    # `stufe` sagt, auf welcher Zoomstufe ein Knoten ENTSTANDEN ist.
    # `lauf_stufe` sagt, welche Generation der Fluss hat, der durch ihn
    # hindurchgeht - und nur das interessiert die naechste Stufe.
    #
    # Der Unterschied ist gross: im Meso-Netz sind 54 Knoten auf der
    # Makrostufe entstanden, aber 157 liegen auf einem Makro-Lauf. Mit `stufe`
    # erbte das Mikrofenster nur 9.0 der 48.4 km Makro-Lauf, die durch es
    # hindurchfuehren - im Bild ein roter Trog, der auf der Meso-Karte quer
    # durch das Fenster laeuft und auf der Mikro-Karte als kurzer Stummel
    # endet. Der Nutzer: "da geht keine rote linie durch bei mikro."
    stufe = netz["lauf_stufe"]

    # Weltkoordinate (in Metern) je aeusserem Knoten. Zeile 0 ist Norden, also
    # y aus mitte_y, x aus mitte_x.
    size_aussen = netz["size_px"]
    links = mitte_aussen_km[0] * 1000.0 - 0.5 * mpp_aussen * size_aussen
    oben = mitte_aussen_km[1] * 1000.0 - 0.5 * mpp_aussen * size_aussen
    welt = np.stack([oben + punkte[:, 0] * mpp_aussen,
                     links + punkte[:, 1] * mpp_aussen], axis=1)

    # Fensterrechteck in Weltmetern.
    #
    # SAUM: Der Saum war urspruenglich ein halber Knotenabstand, damit kein
    # Lauf genau auf der Kante endet. Mit Ein- und Auslauf ist das nicht mehr
    # noetig, und der Preis war hoch - gemessen an einem 15-km-Fenster lagen
    # 14.6 der 48.4 km Makro-Lauf in diesem 930-m-Streifen und gingen der
    # inneren Stufe verloren:
    #
    #     Saum (x Knotenabstand)   0.50   0.25   0.10   0.00
    #     Makro-Lauf im Mikro       28.3   40.9   48.7   48.7 km
    #
    # Die Meso-Stufe meldet 48.4 km fuer dasselbe Fenster - ab 0.1 stimmen die
    # Stufen also ueberein. Ganz auf 0 bringt nichts mehr.
    halb = 0.5 * extent_innen_km * 1000.0
    x0 = mitte_innen_km[0] * 1000.0 - halb
    y0 = mitte_innen_km[1] * 1000.0 - halb
    saum = 0.1 * netz["abstand_m"]
    drin = ((welt[:, 1] >= x0 + saum) & (welt[:, 1] <= x0 + 2 * halb - saum) &
            (welt[:, 0] >= y0 + saum) & (welt[:, 0] <= y0 + 2 * halb - saum))
    innen = np.flatnonzero(drin)
    if len(innen) < 2:
        return None

    neu_index = -np.ones(len(punkte), dtype=np.int64)
    neu_index[innen] = np.arange(len(innen))

    # Umrechnung der Einzugsgebiete. Ein Knoten steht je Stufe fuer eine andere
    # Landflaeche - aussen fuer (Knotenabstand aussen)^2, innen entsprechend
    # weniger. Ohne die Umrechnung waere ein Trog im Fenster genauso "gross"
    # wie ein lokaler Bach.
    faktor = (netz["abstand_m"] / max(innen_abstand_m, 1e-9)) ** 2

    p_innen = np.stack([(welt[innen, 0] - y0) / mpp_innen,
                        (welt[innen, 1] - x0) / mpp_innen], axis=1)
    e_innen = np.array([neu_index[eltern[i]] if eltern[i] >= 0 else -1
                        for i in innen], dtype=np.int64)

    # WOHIN der Lauf weitergeht, wenn er das Fenster verlaesst. Ohne diese
    # Angabe endet der Fluss beim letzten Knoten - und der liegt einen halben
    # Knotenabstand INNERHALB des Randes, im Bild also sichtbar im Nichts.
    ziel = np.full((len(innen), 2), np.nan)
    for k, i in enumerate(innen):
        e = eltern[i]
        if e >= 0 and not drin[e]:
            ziel[k] = [(welt[e, 0] - y0) / mpp_innen,
                       (welt[e, 1] - x0) / mpp_innen]

    # Was von aussen hereinfliesst, dem Eintrittsknoten gutschreiben.
    #
    # UND SEINE GENERATION UEBERNEHMEN. Ohne das bekam der Eintrittsknoten nur
    # das Einzugsgebiet, nicht den Rang - ein Makrotrog lief ab der Fenstergrenze
    # als Mesofluss weiter. Gemessen: die Meso-Stufe meldete 7 Kreuzungen des
    # Makrolaufs mit dem Fensterrand, die Mikro-Stufe kannte davon 1.
    st = stufe[innen].copy()
    zusatz = np.zeros(len(innen))
    einlauf = np.full((len(innen), 2), np.nan)
    # Je Eintrittsknoten zaehlt der GROEBSTE Zufluss - nicht nur einer, der
    # groeber ist als der Knoten selbst. Sonst bleibt das Randstueck genau dort
    # ungezeichnet, wo Trog und Eintrittsknoten schon dieselbe Generation
    # haben, also im haeufigsten Fall.
    bester = np.full(len(innen), len(STUFEN) + 1, dtype=np.int64)
    for j in np.flatnonzero(~drin):
        e = eltern[j]
        if e < 0 or not drin[e]:
            continue
        k = neu_index[e]
        zusatz[k] += float(flaeche[j]) * faktor
        if stufe[j] < bester[k]:
            bester[k] = stufe[j]
            st[k] = min(st[k], stufe[j])
            # Wo der Trog den Rand kreuzt - fuer das Stueck vom Kartenrand bis
            # zum Eintrittsknoten, das sonst gar nicht gezeichnet wuerde.
            einlauf[k] = [(welt[j, 0] - y0) / mpp_innen,
                          (welt[j, 1] - x0) / mpp_innen]

    return {"punkte": p_innen, "eltern": e_innen, "ziel": ziel,
            "stufe": st, "zusatz": zusatz, "einlauf": einlauf}


# =============================================================================
# FENSTER
# =============================================================================

def _regler(name, minimum, maximum, wert, schritt, nachkomma=2):
    """Ein Schieberegler mit Beschriftung. Qt kann nur ganze Zahlen."""
    from PyQt6.QtWidgets import QSlider, QLabel, QVBoxLayout, QWidget
    from PyQt6.QtCore import Qt

    behaelter = QWidget()
    layout = QVBoxLayout(behaelter)
    layout.setContentsMargins(0, 2, 0, 2)
    layout.setSpacing(1)
    beschriftung = QLabel()
    schieber = QSlider(Qt.Orientation.Horizontal)
    schieber.setMinimum(int(round(minimum / schritt)))
    schieber.setMaximum(int(round(maximum / schritt)))
    schieber.setValue(int(round(wert / schritt)))
    schieber.faktor = schritt
    schieber.beschriftung = beschriftung
    schieber.titel = name
    schieber.nachkomma = nachkomma

    def zeige():
        beschriftung.setText("%s: %.*f" % (name, nachkomma,
                                           schieber.value() * schritt))
    schieber.valueChanged.connect(lambda _: zeige())
    zeige()
    layout.addWidget(beschriftung)
    layout.addWidget(schieber)
    return behaelter, schieber


class Werkstatt:
    """Fenster mit Reglern links und Karte rechts."""

    ANSICHTEN = ("Senkenwert", "Dreiecke", "Flussnetz", "Gelaende",
                 "Mit Taelern", "Winkel")

    def __init__(self):
        from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout,
                                     QRadioButton, QButtonGroup, QPushButton,
                                     QLabel, QGroupBox)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.fenster = QWidget()
        self.fenster.setWindowTitle("Flussnetz-Werkstatt")
        self.fenster.resize(1250, 850)
        aussen = QHBoxLayout(self.fenster)

        # --- linke Spalte ---
        links = QVBoxLayout()
        links.setSpacing(4)

        # --- Zoomstufe. Alle drei zeigen DIESELBE Weltstelle, nur weiter
        #     herausgezoomt. Makro ist die ganze Insel.
        zoom_box = QGroupBox("Zoom")
        zoom_layout = QVBoxLayout(zoom_box)
        self.zoom_gruppe = QButtonGroup(self.fenster)
        for i, (name, size, km) in enumerate(STUFEN):
            knopf = QRadioButton("%s  %.0f km  %d px" % (name, km, size))
            if name == "Mikro":
                knopf.setChecked(True)
            self.zoom_gruppe.addButton(knopf, i)
            zoom_layout.addWidget(knopf)
        self.zoom_gruppe.idToggled.connect(lambda _i, an: an and self.zeichne())
        links.addWidget(zoom_box)

        ansicht_box = QGroupBox("Ansicht")
        ansicht_layout = QVBoxLayout(ansicht_box)
        self.ansicht_gruppe = QButtonGroup(self.fenster)
        for i, name in enumerate(self.ANSICHTEN):
            knopf = QRadioButton(name)
            if name == "Flussnetz":
                knopf.setChecked(True)
            self.ansicht_gruppe.addButton(knopf, i)
            ansicht_layout.addWidget(knopf)
        self.ansicht_gruppe.idToggled.connect(lambda _i, an: an and self.zeichne())
        links.addWidget(ansicht_box)

        regler_box = QGroupBox("River Network")
        regler_layout = QVBoxLayout(regler_box)
        self.regler = {}
        for name, lo, hi, wert, schritt, nk in (
                ("Punkte", 100, 2000, 625, 25, 0),
                ("Drain Value Weight", 0.0, 4.0, 1.0, 0.1, 1),
                ("Rivers Follow Lowland", 0.0, 12.0, 6.0, 0.5, 1),
                # 2026-08-04 von 75 auf 60 gesenkt: die Schwelle wurde vorher
                # mit (1 + h_norm) aufgeweicht, 75 wirkte also wie 75 bis 150.
                # Ohne die Aufweichung ist 60 etwa so streng wie frueher
                # gemeint.
                ("Max. Umlenkung (Grad)", 20, 180, 60, 5, 0),
                ("Randabfluss-Kosten", 0.0, 4.0, 1.0, 0.1, 1),
                ("Knoten in Rinnen ziehen", 0.0, 0.7, 0.4, 0.05, 2)):
            behaelter, schieber = _regler(name, lo, hi, wert, schritt, nk)
            schieber.sliderReleased.connect(self.neu_rechnen)
            regler_layout.addWidget(behaelter)
            self.regler[name] = schieber
        links.addWidget(regler_box)

        # --- Wo auf der Insel liegt das lokale Fenster? Genau hier entstehen
        #     die "Regionen": Kuestenebene, Hochmassiv oder Haupttal - je
        #     nachdem, wohin man die 15 km legt.
        ort_box = QGroupBox("Ort auf der Insel")
        ort_layout = QVBoxLayout(ort_box)
        grenze = 0.42 * WELT_KM
        for name, wert in (("Ost (km)", 0.0), ("Nord (km)", 0.0)):
            behaelter, schieber = _regler(name, -grenze, grenze, wert, 2.0, 0)
            schieber.sliderReleased.connect(self.neu_rechnen)
            ort_layout.addWidget(behaelter)
            self.regler[name] = schieber
        links.addWidget(ort_box)

        seed_box = QGroupBox("Map Seed")
        seed_layout = QVBoxLayout(seed_box)
        self.seed_label = QLabel()
        knopf = QPushButton("Neuer Seed")
        knopf.clicked.connect(self.neuer_seed)
        seed_layout.addWidget(self.seed_label)
        seed_layout.addWidget(knopf)
        links.addWidget(seed_box)

        self.status = QLabel()
        self.status.setWordWrap(True)
        links.addWidget(self.status)
        links.addStretch(1)
        aussen.addLayout(links, 0)

        # --- Karte ---
        self.figur = Figure(figsize=(8, 8))
        self.leinwand = FigureCanvasQTAgg(self.figur)
        aussen.addWidget(self.leinwand, 1)

        self.seed = 20260730
        self.stufen = {}
        self._welt_schluessel, self._welt = None, None
        self.neu_rechnen()

    # ------------------------------------------------------------------
    def neuer_seed(self):
        self.seed = int(np.random.default_rng().integers(1, 10 ** 6))
        self.neu_rechnen()

    def wert(self, name):
        s = self.regler[name]
        return s.value() * s.faktor

    def _stufe(self):
        return STUFEN[max(self.zoom_gruppe.checkedId(), 0)]

    def neu_rechnen(self, *_):
        """
        Die ganze Kette von aussen nach innen.

        Makro zuerst, weil nur dort die Kueste liegt und damit ueberhaupt ein
        echter Auslass. Jede engere Stufe bekommt von der vorigen ihre
        Randbedingungen - und rechnet den Verlauf dazwischen selbst.
        """
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ort = (self.wert("Ost (km)"), self.wert("Nord (km)"))
            punkte = int(self.wert("Punkte"))
            gemeinsam = (self.wert("Drain Value Weight"),
                         self.wert("Rivers Follow Lowland"), self.seed,
                         self.wert("Max. Umlenkung (Grad)"),
                         self.wert("Randabfluss-Kosten"),
                         self.wert("Knoten in Rinnen ziehen"))

            self.stufen = {}
            erbe, vorige = None, None
            meldungen = []
            for index, (name, size, km) in enumerate(STUFEN):
                # Makro ist die ganze Welt und steht immer mittig; die engeren
                # Stufen sitzen am gewaehlten Ort.
                mitte = (0.0, 0.0) if km >= WELT_KM else ort
                # DIE WELTSTUFE HAENGT NUR AM SEED UND AN DEN FLUSSREGLERN,
                # nicht am Ort. Gelaende und Netz kosten dort zusammen 5 s von
                # 7.7 s je Durchgang - ohne Zwischenspeicher ist beim Schieben
                # der Ortsregler nichts mehr live bedienbar.
                if km >= WELT_KM:
                    schluessel = (self.seed, punkte, gemeinsam)
                    if schluessel == self._welt_schluessel:
                        self.stufen[name] = vorige = self._welt
                        continue
                H, mpp, okt = weltfeld(size, km, mitte[0], mitte[1], self.seed)
                erbe = None
                if vorige is not None:
                    landanteil = max(float((H > 0).mean()), 0.05)
                    abstand_m = np.sqrt(0.7 * landanteil * (mpp * size) ** 2
                                        / max(punkte, 4))
                    erbe = erbe_bilden(
                        vorige["netz"], vorige["mpp"], vorige["mitte"],
                        mitte, km, mpp, abstand_m)
                netz = baue_netz(H, mpp, punkte, *gemeinsam, erbe=erbe,
                                 stufe_index=index)
                if netz is None:
                    meldungen.append("%s: kein Land im Fenster" % name)
                    vorige = None
                    continue
                eintrag = dict(H=H, mpp=mpp, netz=netz, mitte=mitte,
                               km=km, size=size, oktaven=okt)
                if km >= WELT_KM:
                    self._welt_schluessel = (self.seed, punkte, gemeinsam)
                    self._welt = eintrag
                self.stufen[name] = eintrag
                vorige = eintrag

            self.seed_label.setText("Seed %d" % self.seed)
            self._status_text(meldungen)
            self.zeichne()
        finally:
            QApplication.restoreOverrideCursor()

    def _status_text(self, meldungen):
        name = self._stufe()[0]
        eintrag = self.stufen.get(name)
        if eintrag is None:
            self.status.setText("\n".join(meldungen)
                                or "%s: nichts zu zeigen" % name)
            return
        netz, H = eintrag["netz"], eintrag["H"]
        w = netz["winkel"]
        je_stufe = " / ".join(
            "%s %d" % (STUFEN[s][0], int((netz["lauf_stufe"] == s).sum()))
            for s in range(len(STUFEN)) if (netz["lauf_stufe"] == s).any())
        self.status.setText(
            "%s  %.0f km  %d px  %.0f m/px  %d Oktaven\n"
            "Land %.0f %%, hoechster Punkt %.0f m\n"
            "%d Knoten, davon %d geerbt, Abstand %.0f m\n"
            "Laeufe je Generation: %s\n"
            "%d Auslaesse, groesstes Gebiet %d\n"
            "Einmuendung: Median %.0f Grad, ueber 90 Grad %.0f %%\n%s"
            % (name, eintrag["km"], eintrag["size"], eintrag["mpp"],
               eintrag["oktaven"], 100.0 * float((H > 0).mean()), H.max(),
               len(netz["punkte"]), netz["erbe_n"], netz["abstand_m"],
               je_stufe, len(netz["auslaesse"]), int(netz["flaeche"].max()),
               np.median(w) if len(w) else 0,
               100.0 * np.mean(w > 90) if len(w) else 0,
               "\n".join(meldungen)))

    # ------------------------------------------------------------------
    def _hintergrund(self, ax, deckkraft=1.0):
        """
        Das Gelaende unter jede Kartenansicht legen, Meer in Blau.

        Der Nutzer will die Punkte, Dreiecke und Laeufe IM Gelaende sehen, nicht
        auf weissem Grund - sonst laesst sich nicht beurteilen, ob ein Lauf in
        einer Rinne liegt oder quer ueber einen Ruecken geht.
        """
        H, mpp = self.P, self.mpp
        if (H <= 0).any():
            ax.imshow(np.zeros_like(H), cmap=_meerfarbe(), vmin=0, vmax=1,
                      alpha=deckkraft)
        ax.imshow(np.where(H > 0, H, np.nan), cmap=gelaende_farben(),
                  alpha=deckkraft)
        ax.imshow(np.where(H > 0, schummerung(np.where(H > 0, H, 0.0), mpp),
                           np.nan), cmap="gray", alpha=0.35 * deckkraft)

    def _innere_fenster(self, ax):
        """Die engeren Zoomstufen als Rechteck eintragen."""
        from matplotlib.patches import Rectangle
        eintrag = self.stufen[self._stufe()[0]]
        mpp, size, mitte = eintrag["mpp"], eintrag["size"], eintrag["mitte"]
        for name, _s, km in STUFEN:
            if km >= eintrag["km"] or name not in self.stufen:
                continue
            innen = self.stufen[name]
            b = km * 1000.0 / mpp
            mx = 0.5 * size + (innen["mitte"][0] - mitte[0]) * 1000.0 / mpp
            my = 0.5 * size + (innen["mitte"][1] - mitte[1]) * 1000.0 / mpp
            ax.add_patch(Rectangle((mx - b / 2, my - b / 2), b, b, fill=False,
                                   ec="red", lw=1.4, zorder=8))
            ax.text(mx, my - b / 2 - 3, name, color="red", ha="center",
                    va="bottom", fontsize=8, weight="bold", zorder=8)

    def zeichne(self):
        name = self._stufe()[0]
        eintrag = self.stufen.get(name)
        if eintrag is None:
            return
        self.P, self.mpp, self.netz = eintrag["H"], eintrag["mpp"], eintrag["netz"]
        ansicht = self.ANSICHTEN[self.ansicht_gruppe.checkedId()]
        self.figur.clear()
        ax = self.figur.add_subplot(111)
        pk = self.netz["punkte"]
        el = self.netz["eltern"]
        fl = self.netz["flaeche"]

        if ansicht == "Winkel":
            w = self.netz["winkel"]
            ax.hist(w, bins=36, range=(0, 180), color="tab:blue")
            ax.axvline(90, color="red", ls="--", lw=1)
            ax.set_xlabel("Umlenkung von einem Ast zum naechsten (Grad)")
            ax.set_ylabel("Anzahl")
            ax.grid(alpha=0.3)
            ax.set_title("Einmuendungswinkel - spitz ist natuerlich")
            self.leinwand.draw()
            return

        if ansicht == "Senkenwert":
            self._hintergrund(ax, deckkraft=0.55)
            bild = ax.scatter(pk[:, 1], pk[:, 0], s=26,
                              c=self.netz["d_knoten"], cmap="YlOrBr_r",
                              vmin=0, vmax=1,
                              edgecolors="black", linewidths=0.4)
            self.figur.colorbar(bild, ax=ax, shrink=0.8,
                                label="Senkenwert je Punkt")
            ax.set_title("Senkenwert - hell heisst: Wasser will hierhin")
        elif ansicht == "Dreiecke":
            self._hintergrund(ax)
            for a, c in self.netz["kanten"]:
                ax.plot([pk[a][1], pk[c][1]], [pk[a][0], pk[c][0]],
                        color="black", lw=0.3, alpha=0.55)
            ax.scatter(pk[:, 1], pk[:, 0], s=5, c="black")
            ax.set_title("Delaunay-Dreiecke (%d Kanten)"
                         % len(self.netz["kanten"]))
        elif ansicht == "Gelaende":
            self._hintergrund(ax)
            bild = ax.imshow(np.where(self.P > 0, self.P, np.nan),
                             cmap=gelaende_farben())
            self.figur.colorbar(bild, ax=ax, shrink=0.8, label="Hoehe (m)")
            ax.set_title("Das Weltfeld ohne Fluesse - %.0f m/px, %d Oktaven"
                         % (eintrag["mpp"], eintrag["oktaven"]))
        elif ansicht == "Mit Taelern":
            if "geschnitten" not in eintrag:
                eintrag["geschnitten"] = taeler_eingraben(
                    self.P, self.mpp, self.netz,
                    breite_m=1.2 * self.netz["abstand_m"])
            G = eintrag["geschnitten"]
            if (G <= 0).any():
                ax.imshow(np.zeros_like(G), cmap=_meerfarbe(), vmin=0, vmax=1)
            bild = ax.imshow(np.where(G > 0, G, np.nan), cmap=gelaende_farben())
            ax.imshow(np.where(G > 0, schummerung(np.where(G > 0, G, 0.0),
                                                  self.mpp), np.nan),
                      cmap="gray", alpha=0.4)
            self.figur.colorbar(bild, ax=ax, shrink=0.8, label="Hoehe (m)")
            ax.set_title("Gelaende NACH dem Eingraben - %d m Talbreite"
                         % int(1.2 * self.netz["abstand_m"]))
        else:
            self._hintergrund(ax)
            breiteste = float(fl.max())
            ls = self.netz["lauf_stufe"]
            # NACH GENERATION FAERBEN. Ein Lauf behaelt die Farbe der Stufe,
            # auf der er entstanden ist - Makro rot, Meso gruen, Mikro gelb.
            # So ist auf einen Blick zu sehen, was uebernommen wurde und was
            # diese Stufe neu dazugelegt hat.
            for i in range(len(pk)):
                e = el[i]
                if e < 0:
                    continue
                lw = 0.4 + 3.4 * (fl[i] / breiteste) ** 0.45
                ax.plot([pk[e][1], pk[i][1]], [pk[e][0], pk[i][0]],
                        color="#101010", lw=lw + 1.0, solid_capstyle="round",
                        zorder=3 + int(2 - ls[i]))
                ax.plot([pk[e][1], pk[i][1]], [pk[e][0], pk[i][0]],
                        color=STUFEN_FARBE[int(ls[i])], lw=lw,
                        solid_capstyle="round", zorder=4 + int(2 - ls[i]))
            # Die Auslaesse nach Einzugsgebiet groesser zeichnen: seit dem
            # Randabfluss gibt es viele kleine und wenige grosse, und der
            # Unterschied ist genau das, was man sehen will.
            # Das erste Stueck vom Kartenrand herein, in der Farbe des Laufs.
            for a, (zy, zx) in self.netz.get("einlauf", {}).items():
                lw = 0.4 + 3.4 * (fl[a] / breiteste) ** 0.45
                ax.plot([pk[a][1], zx], [pk[a][0], zy], color="#101010",
                        lw=lw + 1.0, solid_capstyle="round", zorder=8)
                ax.plot([pk[a][1], zx], [pk[a][0], zy],
                        color=STUFEN_FARBE[int(ls[a])], lw=lw,
                        solid_capstyle="round", zorder=9)
            # Das letzte Stueck bis ins Meer, in der Farbe des Laufs.
            auslauf = self.netz.get("auslauf", {})
            for a, (zy, zx) in auslauf.items():
                lw = 0.4 + 3.4 * (fl[a] / breiteste) ** 0.45
                ax.plot([pk[a][1], zx], [pk[a][0], zy], color="#101010",
                        lw=lw + 1.0, solid_capstyle="round", zorder=8)
                ax.plot([pk[a][1], zx], [pk[a][0], zy],
                        color=STUFEN_FARBE[int(ls[a])], lw=lw,
                        solid_capstyle="round", zorder=9)
            for a in self.netz["auslaesse"]:
                if a in auslauf:
                    continue      # muendet ins Meer, braucht keine Marke
                ax.scatter([pk[a][1]], [pk[a][0]],
                           s=25 + 110 * (fl[a] / breiteste) ** 0.5,
                           marker="v", c="white", edgecolors="black",
                           linewidths=1.0, zorder=10)
            teile = " ".join("%s=%d" % (STUFEN[s][0], int((ls == s).sum()))
                             for s in range(len(STUFEN)) if (ls == s).any())
            ax.set_title("Flussnetz nach Generation - %s" % teile)

        if ansicht != "Winkel":
            self._innere_fenster(ax)
        ax.set_xlim(0, self.P.shape[1])
        ax.set_ylim(self.P.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])
        self.figur.tight_layout()
        self.leinwand.draw()


def _meerfarbe():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("meer", ["#1b4f9c", "#1b4f9c"])


def main():
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    werkstatt = Werkstatt()
    werkstatt.fenster.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
