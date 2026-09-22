"""
Path: core/terrain_river_network.py

Flussnetz-Skelett mit Hochebene (90_MESSPROTOKOLLE.md §12).

Die Hoehe wird nicht vom Fluss AUFGEBAUT, sondern zwischen dem Fluss und einer
globalen Flaeche GEBLENDET:

    z = P - (P - z_Fluss) * (1 - Profil(d~))

P ist das Gelaende, das ohne Fluesse da waere (fBm + ATEF-Filter, also das
Ergebnis der bisherigen Terrain-Kette). Am Fluss (d~=0, Profil=0) ergibt das
z_Fluss, an der Wasserscheide (d~=1, Profil=1) ergibt es P.

WARUM DIESE UMKEHRUNG. Die naheliegende Form `z = z_Fluss(naechster) + Profil*H`
kann keine Hochebene erzeugen: jenseits der Talbreite haengt sie nur noch am
NAECHSTEN Fluss, und dessen Hoehe springt an jeder Wasserscheide. Ergebnis war
ein Flickwerk exakt ebener Terrassen mit Nahtstellen (90_MESSPROTOKOLLE.md
§8: 62% aller Zellen ohne streng tieferen Nachbarn). Hier laufen an der
Wasserscheide beide Seiten gegen denselben Wert P, der einwertig ist - die
Naht faellt weg, und zwischen den Taelern kann eine echte Hochflaeche liegen.

Aufbau in vier Schritten, alle nicht-iterativ:

  1. Poisson-Disk-Punktsatz mit Mindestabstand IN METERN. Deckt die Karte per
     Konstruktion ab - das war die offene Baustelle aus
     90_MESSPROTOKOLLE.md §8, wo ein gewachsener Baum bei 192 px nur 139 px
     weit kam.
  2. Delaunay-Graph. Planar, deshalb kann sich kein Teilgraph selbst kreuzen.
  3. Kuerzeste-Wege-Baum vom Auslass am Kartenrand, Kantenkosten steigen mit
     der Hoehe von P. Die Fluesse suchen sich dadurch das Tiefe, ohne an den
     zufaelligen Minima des Rauschens zu haengen. Ein Spannbaum hat keine
     Schleifen und erreicht jeden Punkt - Haupt- und Nebenfluesse in einem
     Aufruf, ohne dass "Generationen" konstruiert werden muessten.
  4. Strahler-Ordnung rueckwaerts abgelesen.

KEIN LOD-BEZUG. Der Punktsatz wird in METERKOORDINATEN erzeugt und erst danach
auf das Pixelgitter abgebildet. Dadurch liefert jede Aufloesung DASSELBE Netz -
sonst haette jede LOD-Stufe ein anderes Gelaende und die Karte spraenge
waehrend der progressiven Darstellung.

Gemessener Stand (90_MESSPROTOKOLLE.md §12), 25 km, 384 px, ohne jede
Erosionsiteration: Entwaesserungsanteil 61-97% gegen 10.1% im reinen
Noise-Pfad und 10-22% der Feld-Erosion (90_MESSPROTOKOLLE.md §7).
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


DEFAULTS: Dict[str, Any] = {
    # Abstand benachbarter Talsohlen in Metern. Bestimmt zugleich die Dichte
    # des Netzes und - ueber valley_width_fraction - die Talbreite.
    "river_spacing_m": 2500.0,
    # Tiefe des groessten Tales als ANTEIL der Hoehenspanne (siehe
    # RIVER_NETWORK.INCISION_SHARE in value_default).
    "incision_share": 0.55,
    # Anteil der Talbreite am Flussabstand. 0.5 heisst: das Tal reicht genau
    # bis zur Mitte zwischen zwei Laeufen, es bleibt keine Hochflaeche uebrig.
    # Kleinere Werte lassen zwischen den Taelern eine Flaeche stehen.
    "valley_width_fraction": 0.55,
    # Wie stark die Strahlweite mit dem Einzugsgebiet waechst. 0.4 entspricht
    # der hydraulischen Geometrie echter Fluesse (Breite ~ Wurzel des
    # Durchflusses, Durchfluss ~ Einzugsgebiet).
    "valley_width_exponent": 0.40,
    # Wie stark die Eintiefung mit dem Einzugsgebiet waechst.
    "incision_area_exponent": 0.30,
    # Exponent der Abstandsabbildung 1 - exp(-t^a), siehe carve_river_network.
    #
    # Vorgabe 2026-07-30 von 2.0 auf 1.0 gesenkt. Nahe am Fluss verhaelt sich
    # 1 - exp(-t^a) wie t^a - bei a = 2 ist die Kurve dort also FLACH, und
    # genau das erzeugte die breiten ebenen Talsohlen: auf 30 % der Talbreite
    # standen noch 84 % Flusseinfluss. Bei a = 1 faellt der Einfluss von
    # Anfang an (30 % Breite -> 53 %).
    #
    # Damit sind die beiden Rollen entkoppelt: die FORM des Querschnitts macht
    # allein valley_form, das knickfreie Auslaufen nach aussen die
    # Exponentialfunktion. Vorher formte dieser Regler beides zugleich.
    "edge_softness": 1.0,
    # Verzerrung des Abstandsfeldes durch die vorhandene Gelaendestruktur.
    # 0 ergibt geometrisch saubere Talraender (das "ausgeschnittene" Aussehen),
    # hoehere Werte lassen die Talgrenze mit dem Gelaende wandern.
    "edge_warp": 0.45,
    # Querprofil: <1 Schlucht, 1 V, >1 U (glazial).
    #
    # Vorgabe 2026-07-30 von 1.6 auf 1.1 gesenkt. 1.6 ist ein ausgepraegtes
    # U-Tal und hat per Definition eine flache Sohle - genau die breiten
    # ebenen Boeden, die im Querschnitt als "aufgelegte Baender" auffielen.
    # Gemessene mittlere Neigung in der Sohle: 0.48 bei 1.6, 0.88 bei 1.0.
    # Das U-Tal bleibt einstellbar, es ist nur nicht mehr der Normalfall
    # (glazial ueberformte Landschaften wie Wallis oder Skerrheim).
    "valley_form": 1.1,
    # Wie stark die Fluesse hohes Gelaende meiden - wirkt an ZWEI Stellen:
    # gerichtet auf die Kantenkosten des Baumes (spanning_tree) und auf die
    # Wegsuche zwischen zwei Knoten (densify).
    #
    # Vorgabe 2026-07-30 von 2.5 auf 6.0 angehoben. Gemessen als Ueberhoehung
    # des Laufs ueber dem tiefsten Punkt seiner Umgebung: 476 m geradeaus,
    # 352 m bei 2.5, 300 m bei 8.0 - bei gleichzeitig steigendem
    # Entwaesserungsanteil (67 / 76 / 80 %).
    "cost_strength": 6.0,
    # Zahl der Auslaesse am Kartenrand. Mit nur einem muss JEDER Punkt der
    # Karte dorthin entwaessern, das Netz also jeden Ruecken dazwischen
    # ueberqueren - und das Eintiefen schneidet ihn anschliessend durch.
    # Gemessen (15 km, 2500 m Abstand): groesster erzwungener Abtrag 1426 m bei
    # einem Auslass gegen 644 m bei dreien.
    "outlet_count": 3,
    # Preis dafuer, die Karte an einer beliebigen Randstelle zu verlassen -
    # relativ zu einem Lauf ueber die halbe Karte. Ohne diesen Ausweg legt sich
    # ein Fluss einmal um den Kartenrand herum, weil der Rand eine durchgehende
    # billige Kette ist; siehe die Messreihe in spanning_tree().
    "border_outflow": 1.0,
    # Glaettung der Hoehen und Einzugsgebiete ENTLANG des Netzes. Ohne sie
    # springt beides an jedem Zusammenfluss (der Nebenarm hat dort schlagartig
    # ein viel kleineres Einzugsgebiet als der Hauptlauf), und im Gelaende
    # steht eine Kante.
    "confluence_smoothing": 3,
    # Mindestabstand zweier Laeufe in Pixeln. Bereits gezeichnete Laeufe sind
    # fuer spaetere Wege gesperrt - so koennen sich Fluesse nicht kreuzen und
    # Taeler nicht ineinanderrutschen.
    "separation_px": 2.0,
    # Wie weit der gesuchte Weg von der geraden Verbindung abweichen darf,
    # als Anteil des Talabstands. Haelt die Laeufe in ihrem Korridor und damit
    # kreuzungsfrei (der Delaunay-Baum ist planar).
    "corridor_fraction": 0.45,
    # Glaettung des gefundenen Weges (Chaikin). Nimmt dem Gitterweg die
    # Treppenstufen. Wird nur uebernommen, wenn sie keine Sperre verletzt.
    "path_smoothing": 1,
    # Seitliche Auslenkung der Zwischenpunkte - seit der Wegsuche ueber das
    # Hoehenfeld ohne Wirkung, der Maeander kommt jetzt aus dem Gelaende.
    "meander": 0.18,
    # Abstand der Zwischenpunkte als Anteil des Flussabstands.
    "densify_fraction": 0.22,
    # Wie weich das Tal in die Umgebung uebergeht (0 = harte Kante an der
    # Wasserscheide, 1 = voll geglaettet). Siehe valley_profile().
    "divide_blend": 0.75,
    # HOECHSTE Steigung eines Laufs, in Meter je Meter. Ohne sie steht an
    # jedem Zusammenfluss eine Kante: der Muendungsknoten wird vom TIEFSTEN
    # Zufluss nach unten gezogen, der andere Zufluss steht unmittelbar daneben
    # noch auf seiner eigenen Hoehe. Gemessen 574 m Sprung zwischen zwei direkt
    # aufeinanderfolgenden Punkten desselben Laufs.
    #
    # In der Natur hat sich ein Nebenarm auf dem Weg zur Muendung laengst
    # eingeschliffen. Die Obergrenze verteilt den Sprung ueber eine Strecke,
    # statt ihn auf einem Pixel stehen zu lassen. 0.12 = 12 %, ein steiler,
    # aber vorkommender Gebirgsbach.
    "max_gradient": 0.12,
    # Mindestgefaelle des Laufs, als Anteil der Gesamthoehe ueber die
    # KARTENBREITE (siehe dense_heights). 0.06 gemessen als guter Kompromiss:
    # deutlich bessere Entwaesserung als 0.002 (76% statt 41% bei 192 px) bei
    # nur wenig zusaetzlicher Eintiefung (417 statt 330 m).
    "min_rise_fraction": 0.06,
}


# =============================================================================
# 1. Punktsatz
# =============================================================================

def poisson_points(extent_m: float, min_distance_m: float, seed: int,
                   attempts: int = 30) -> np.ndarray:
    """
    Poisson-Disk-Punktsatz (Bridson) in METERKOORDINATEN, [0, extent_m)^2.

    In Metern und nicht in Pixeln, damit jede Aufloesung dasselbe Netz
    bekommt - dieselbe Ueberlegung wie bei FEATURE_SIZE_M und GULLY_SIZE_M in
    90_MESSPROTOKOLLE.md §10.
    """
    # DIE ZUFALLSREIHENFOLGE IST UNANTASTBAR (2026-08-23).
    #
    # Jeder rng-Aufruf steht in derselben Reihenfolge wie vorher: einmal
    # `random(2)` fuer den Startpunkt, dann je Runde ein `integers` und je
    # Versuch zwei `random`. Wuerde man auch nur einen Aufruf zusammenfassen
    # oder vorziehen, kaeme ein anderer Punktsatz heraus - und damit ein
    # anderes Flussnetz, andere Taeler, ein anderes Gelaende. Beschleunigt
    # ist ausschliesslich, was ZWISCHEN den Zufallsaufrufen passiert.
    #
    # GEMESSEN (1024 px, Mikrostufe mit 150 m Abstand): 12 708 Punkte,
    # 465 662 Versuche, 20.4 s - also 43.8 Mikrosekunden je Versuch fuer im
    # Mittel 11 Zellpruefungen. Das ist fast reiner Aufrufaufwand.
    #
    # DIE URSACHE WAR numpy AN DER FALSCHEN STELLE. `grid[yy, xx]` auf einem
    # numpy-Array kostet rund 1.5 Mikrosekunden - es baut je Zugriff ein
    # numpy-Skalarobjekt. Bei 5.2 Millionen Zellpruefungen sind das etwa
    # 8 Sekunden fuer Indexzugriffe, die als Python-Listenzugriff je 0.05
    # Mikrosekunden kosten. numpy ist schnell auf GANZEN Feldern und langsam
    # auf EINZELWERTEN; hier wird ausschliesslich einzeln zugegriffen.
    #
    # Geaendert: `grid` als verschachtelte Python-Liste, `points` als Liste
    # von Tupeln aus Python-Floats, und Skalararithmetik statt eines
    # 2-Element-Arrays je Versuch. Die Rechnung selbst ist unveraendert -
    # IEEE-double bleibt IEEE-double, ob in numpy oder in Python.
    # tests/smoke_test_poisson_punkte.py prueft die Bitgleichheit.
    #
    # NICHT vektorisiert, obwohl das naheliegt: ein erster Anlauf holte den
    # 5x5-Block in einem Slice und verglich alle Nachbarn auf einmal. Das war
    # GEMESSEN 1.7- bis 3-mal LANGSAMER - es sind typisch nur ein bis fuenf
    # belegte Nachbarn, und die Schleife bricht beim ersten Treffer ab,
    # waehrend die Vektorfassung immer alle prueft und dafuer mehrere
    # Zwischenarrays anlegt.
    rng = np.random.default_rng(seed)
    cell = min_distance_m / np.sqrt(2.0)
    n = int(np.ceil(extent_m / cell)) + 1
    grid = [[-1] * n for _ in range(n)]

    points = []
    active = []
    grenze2 = min_distance_m * min_distance_m

    def insert(p0, p1):
        points.append((p0, p1))
        grid[int(p0 / cell)][int(p1 / cell)] = len(points) - 1
        active.append(len(points) - 1)

    start = rng.random(2) * extent_m
    insert(float(start[0]), float(start[1]))

    while active:
        k = int(rng.integers(0, len(active)))
        c0, c1 = points[active[k]]
        found = False
        for _ in range(attempts):
            angle = rng.random() * 2.0 * np.pi
            radius = min_distance_m * (1.0 + rng.random())
            p0 = c0 + radius * np.cos(angle)
            p1 = c1 + radius * np.sin(angle)
            if not (0.0 <= p0 < extent_m and 0.0 <= p1 < extent_m):
                continue
            gy, gx = int(p0 / cell), int(p1 / cell)
            free = True
            for dy in range(-2, 3):
                yy = gy + dy
                if yy < 0 or yy >= n:
                    continue
                zeile = grid[yy]
                for dx in range(-2, 3):
                    xx = gx + dx
                    if 0 <= xx < n:
                        j = zeile[xx]
                        if j >= 0:
                            q0, q1 = points[j]
                            if ((q0 - p0) * (q0 - p0)
                                    + (q1 - p1) * (q1 - p1) < grenze2):
                                free = False
                                break
                if not free:
                    break
            if free:
                insert(p0, p1)
                found = True
                break
        if not found:
            active.pop(k)

    return np.array(points)


# =============================================================================
# 2.+3. Graph und Spannbaum
# =============================================================================

def spanning_tree(points_px: np.ndarray, P: np.ndarray, size: int,
                  cost_strength: float, seed: int, smoothing_px: float = 0.0,
                  outlet_count: int = 1, border_outflow: float = 1.0):
    """
    Delaunay-Graph, dann kuerzeste Wege vom Auslass am Kartenrand.

    Die Kantenkosten steigen mit der Hoehe von P. Das ist der brauchbare Kern
    der naheliegenden Idee "verbinde die lokalen Minima": nicht die Minima als
    KNOTEN nehmen - die haengen an Frequenz und Aufloesung und sind nach dem
    Eingraben ohnehin keine Minima mehr - sondern die Hoehe als KOSTENFELD.

    Returns: (parents, order_by_distance, auslaesse). parents[i] ist der
    Knoten FLUSSABWAERTS, -1 an jedem Auslass.
    """
    from scipy.spatial import Delaunay
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    tri = Delaunay(points_px)
    edges = set()
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges.add((min(s[a], s[b]), max(s[a], s[b])))
    edges = np.array(sorted(edges))

    # Die Knotenhoehen fuer das Kostenfeld werden aus einem GEGLAETTETEN P
    # genommen. Fluesse folgen der grossraeumigen Topografie, nicht dem
    # Pixelrauschen - und ohne die Glaettung haengt der Verlauf an der
    # Aufloesung: bei 512 px liefert der ATEF-Filter feinere Oktaven, Dijkstra
    # waehlt andere Wege, und die Landschaft aendert sich zwischen den Stufen
    # (gemessen als 0.074 Grobform-Abweichung gegen 0.008 ohne Netz,
    # smoke_test_terrain_scale_coupling Lauf 1). Die Glaettungsbreite ist in
    # METERN vorgegeben und damit selbst aufloesungsunabhaengig.
    P_kosten = P
    if smoothing_px > 0.5:
        P_kosten = ndimage.gaussian_filter(P.astype(np.float64), smoothing_px)

    yi = np.clip(np.round(points_px[:, 0]).astype(int), 0, size - 1)
    xi = np.clip(np.round(points_px[:, 1]).astype(int), 0, size - 1)
    h = P_kosten[yi, xi].astype(np.float64)
    span = float(h.max() - h.min())
    h_norm = (h - h.min()) / (span if span > 1e-9 else 1.0)

    length = np.linalg.norm(points_px[edges[:, 0]] - points_px[edges[:, 1]], axis=1)

    # KOSTEN DES ANSTIEGS, NICHT DER HOEHE.
    #
    # Vorher: cost = laenge * (1 + staerke * mittlere_hoehe). Zwei Fehler darin.
    #
    # Erstens war es die HOEHE. Ein Lauf, der auf einer Hochflaeche entlang
    # geht, wurde damit genauso bestraft wie einer, der eine Wand hochklettert -
    # obwohl nur das Klettern unnatuerlich ist.
    #
    # Zweitens war es SYMMETRISCH, also fuer beide Richtungen gleich. Dijkstra
    # laeuft vom Auslass nach aussen, jede Kantenrichtung u->v ist also
    # FLUSSAUFWAERTS. Damit laesst sich der Anstieg gerichtet bestrafen: hinauf
    # ist teuer, auf gleicher Hoehe entlang billig.
    #
    # Ohne das frass sich ein Nebenarm quer durch einen 3100-m-Gipfel, statt
    # daran vorbeizulaufen: eine Ueberquerung kostete hoechstens
    # (1 + staerke) = 3.5 mal so viel wie der flache Weg, und sobald der
    # Umweg laenger als das Vierfache war, ging der Baum drueber.
    #
    # Der Anstieg wird an der MITTLEREN Kantensteigung des Netzes bemessen,
    # nicht an einem festen Meterwert - damit wirkt der Regler in flachem wie
    # in steilem Gelaende gleich (02_INVARIANTEN.md 4).
    delta = h_norm[edges[:, 1]] - h_norm[edges[:, 0]]
    steigung = np.abs(delta) / np.maximum(length, 1e-9)
    bezug = float(np.median(steigung)) or 1e-6

    def gerichtete_kosten(anstieg):
        """anstieg > 0 heisst bergauf; bergab kostet nur die Laenge."""
        return length * (1.0 + cost_strength
                         * np.square(np.maximum(anstieg, 0.0)
                                     / np.maximum(length, 1e-9) / bezug))

    kosten_hin = gerichtete_kosten(delta)      # edges[:,0] -> edges[:,1]
    kosten_rueck = gerichtete_kosten(-delta)   # edges[:,1] -> edges[:,0]

    n = len(points_px)

    # AUSLASS: Lage aus dem SEED, nicht aus der Hoehe.
    #
    # Vorher war es der tiefste Punkt in Randnaehe - und damit haengte er an P,
    # also an JEDEM Regler, der das Gelaende beeinflusst. Beim erneuten
    # Generieren mit anderer Talform oder Eintiefung wanderte der Auslass und
    # mit ihm das gesamte Netz, obwohl der Seed derselbe war.
    #
    # Jetzt: ein Winkel aus dem Seed legt eine Himmelsrichtung fest, der
    # naechstgelegene Randpunkt wird Auslass. Punktsatz und Winkel haengen
    # beide nur am Seed, der Auslass liegt damit bei jedem Lauf an derselben
    # Stelle. Nur einer - endorheische Becken sind zurueckgestellt.
    # MEHRERE AUSLAESSE, Lage aus dem SEED.
    #
    # Mit einem einzigen Auslass muss JEDER Punkt der Karte dorthin
    # entwaessern - das Netz ueberquert also zwangslaeufig jeden Ruecken, der
    # dazwischen liegt, und das Eintiefen schneidet ihn danach durch. Genau das
    # zeigte der Nutzer an einem Nebenarm, der sich quer durch einen
    # 3100-m-Gipfel frass. In der Natur fliesst das Wasser hinter einem Kamm
    # einfach zu einem ANDEREN Fluss.
    #
    # Die Richtungen sind gleichmaessig ueber den Vollkreis verteilt, der
    # Startwinkel kommt aus dem Seed. Punktsatz und Winkel haengen beide nur am
    # Seed - die Auslaesse liegen damit bei jedem Lauf an derselben Stelle,
    # unabhaengig von jedem Regler, der das Gelaende beeinflusst.
    start = np.random.default_rng(seed ^ 0x5EED).random() * 2.0 * np.pi
    mitte = 0.5 * size
    auslaesse = []
    for i in range(max(int(outlet_count), 1)):
        w = start + 2.0 * np.pi * i / max(int(outlet_count), 1)
        # AUF DEN KARTENRAND PROJIZIEREN, nicht auf einen Kreis.
        #
        # `mitte + mitte*cos(w)` beschreibt einen Kreis, der das Quadrat nur an
        # vier Punkten beruehrt - fuer alle anderen Winkel lag das Ziel INNEN,
        # und damit der Auslass mitten in der Karte statt am Rand. Bei 45 Grad
        # etwa auf (213, 213) einer 250er Karte.
        #
        # Der Strahl wird deshalb so weit gestreckt, bis die groessere der
        # beiden Komponenten den Rand erreicht.
        richtung_y, richtung_x = np.cos(w), np.sin(w)
        streckung = mitte / max(abs(richtung_y), abs(richtung_x), 1e-9)
        ziel_y = mitte + streckung * richtung_y
        ziel_x = mitte + streckung * richtung_x
        kandidat = int(np.argmin((points_px[:, 0] - ziel_y) ** 2
                                 + (points_px[:, 1] - ziel_x) ** 2))
        if kandidat not in auslaesse:
            auslaesse.append(kandidat)

    # RANDABFLUSS - gegen den Ring am Kartenrand.
    #
    # Mit einer festen, kleinen Zahl von Auslaessen muss jeder Knoten dorthin.
    # Fuer die gegenueberliegende Ecke ist der Weg AM KARTENRAND ENTLANG oft
    # billiger als quer durchs Gebirge, weil der Rand eine durchgehende Kette
    # von Delaunay-Kanten ist. Das Ergebnis war ein Fluss, der sich einmal um
    # die halbe Karte legte und dabei ueber die Ruecken stieg - der Nutzer hat
    # es am 2026-08-04 im Bild gezeigt.
    #
    # Abhilfe: ein virtueller Knoten n als gemeinsame Wurzel, an dem die
    # Hauptauslaesse gratis und JEDER Randknoten zu einem festen Preis haengen.
    # Dijkstra entscheidet dann selbst - ein Randknoten verlaesst die Karte an
    # Ort und Stelle, sobald der Umweg zum Hauptauslass mehr kostet als dieser
    # Preis. Das ist auch physikalisch die richtigere Aussage: aus einem
    # 15-km-Fenster fliesst Wasser an vielen Randstellen hinaus, nicht an einer.
    #
    # Gemessen an einem 256er Ausschnitt, 625 Punkte, als LAENGSTE Kette, die
    # den Randsaum nie verlaesst, in Prozent einer Kantenlaenge:
    #
    #     Preis      6.0    3.0    2.0    1.5    1.0    0.5
    #     Randlauf   212%   158%   107%    93%    61%    30%
    #     Auslaesse    1     21     49     54     66     73
    #
    # 6.0 entspricht dem Verhalten davor. Ab 1.0 ist der Ring weg.
    typisch = float(np.median(length))
    saum_px = 0.75 * typisch
    # `size`, NICHT `size - 1`: die Punkte kommen aus poisson_points in Metern
    # und laufen nach der Teilung durch mpp bis size, nicht bis size-1. Mit der
    # -1 lag der Streifen bei 256 und 512 px an verschieden weit innen
    # liegenden Stellen, andere Knoten fielen hinein, und das Netz sprang
    # zwischen den Aufloesungen (r = +0.768 statt +0.990, Test 6).
    rand = np.flatnonzero(
        (points_px[:, 0] < saum_px) | (points_px[:, 0] > size - saum_px) |
        (points_px[:, 1] < saum_px) | (points_px[:, 1] > size - saum_px))
    # Ein Hauptauslass liegt selbst am Rand. Bliebe er in beiden Listen,
    # summierte csr_matrix die zwei Eintraege und er verloere seinen Vorrang.
    rand = np.setdiff1d(rand, np.asarray(auslaesse, dtype=np.int64))
    # Preis relativ zum Netz, nicht als absolute Zahl (02_INVARIANTEN.md 4):
    # 1.0 heisst "so teuer wie ein Lauf quer ueber die halbe Karte".
    quer = 0.5 * size / max(typisch, 1e-9)
    preis = max(float(border_outflow), 0.0) * quer * float(np.median(kosten_hin))

    # ACHTUNG: scipy.csgraph liest eine 0 in der Matrix als "keine Kante". Die
    # Hauptauslaesse brauchen deshalb einen winzigen positiven Wert statt der 0,
    # sonst haengen sie ueberhaupt nicht an der Wurzel - und dann wird jeder
    # Randknoten zum Auslass, voellig unabhaengig vom eingestellten Preis.
    quellen = np.concatenate([edges[:, 0], edges[:, 1],
                              np.full(len(auslaesse), n), np.full(len(rand), n)])
    ziele = np.concatenate([edges[:, 1], edges[:, 0],
                            np.asarray(auslaesse, dtype=np.int64), rand])
    werte = np.concatenate([kosten_hin, kosten_rueck,
                            np.full(len(auslaesse), 1e-9),
                            np.full(len(rand), max(preis, 1e-9))])
    matrix = csr_matrix((werte, (quellen, ziele)), shape=(n + 1, n + 1))

    ergebnis = dijkstra(matrix, indices=n, return_predecessors=True)
    distance = ergebnis[0][:n]
    parents = ergebnis[1][:n].astype(np.int64)
    # Wer direkt an der Wurzel haengt, ist selbst ein Auslass.
    parents[parents == n] = -1
    auslaesse = np.flatnonzero(parents < 0).tolist()

    finite = np.isfinite(distance)
    order = np.argsort(np.where(finite, distance, np.inf))
    order = order[finite[order]]
    return parents, order, auslaesse


def strahler_order(parents: np.ndarray, order: np.ndarray) -> np.ndarray:
    """
    Strahler-Ordnung, ein Rueckwaertslauf ueber `order`.

    Die "Generationen" (Haupt-, Neben-, Nebennebenfluss) muessen nicht
    konstruiert werden - sie stehen bereits im Baum und werden hier nur
    abgelesen. Rueckwaerts ueber `order` heisst: jedes Kind vor seinem
    Elternknoten, weil der Elternknoten naeher am Auslass liegt.
    """
    n = len(parents)
    result = np.ones(n, dtype=np.int32)
    best = np.zeros(n, dtype=np.int32)
    count = np.zeros(n, dtype=np.int32)

    for i in order[::-1]:
        if count[i] == 0:
            result[i] = 1
        elif count[i] == 1:
            result[i] = best[i]
        else:
            result[i] = best[i] + 1
        e = parents[i]
        if e < 0:
            continue
        if result[i] > best[e]:
            best[e], count[e] = result[i], 1
        elif result[i] == best[e]:
            count[e] += 1
    return result


def river_heights(points_px, parents, order, strahler, P, size,
                  peak_m, incision_m, min_rise_fraction):
    """
    Hoehe je Knoten - MONOTON fallend flussabwaerts, und nie ueber dem
    umgebenden Gelaende.

    Zuerst folgt das Flussbett dem Gelaende: z = P - Einschnitt(Ordnung).
    Grosse Fluesse schneiden tiefer, kleine flacher.

    Danach wird die Monotonie durch EINTIEFEN hergestellt, nicht durch
    Anheben: von den Quellen abwaerts wird jeder Knoten mindestens
    `min_drop` unter seinen hoechsten Zufluss gedrueckt.

        z(eltern) = min( z(eltern), z(kind) - Mindestgefaelle )

    WARUM EINTIEFEN UND NICHT ANHEBEN. Die erste Fassung rechnete
    `s(i) = max(s(eltern) + Anstieg, P(i))`, hob den Fluss also ueber das
    Gelaende, wo P eine Delle hat. Folge: bei kleiner Einschnitttiefe lag das
    "Flussbett" UEBER der Umgebung und die Blend-Formel zog das Umland zu
    einem Ruecken hinauf - aus Taelern wurden Grate. Gemessen mit Tiefe 0
    gegen abgeschaltetes Netz: Korrelation nur +0.76 bei 275 m
    Medianabweichung, obwohl "keine Eintiefung" praktisch nichts aendern
    sollte.

    Durch Eintiefen ist beides zugleich erfuellt: das Bett liegt IMMER unter
    P (es wird nur gesenkt, nie gehoben), und es faellt streng flussabwaerts.
    Der Preis ist, dass der Auslass bei langen Laeufen tiefer rutscht - das
    faengt die Spannen-Rueckbildung in _calc_redistribution auf.

    Das Mindestgefaelle ist am Gesamtrelief bemessen, nicht als Steigung pro
    Meter. Als 3%-Steigung angesetzt ergab es bei 3000 m langen Kanten 90 m
    PRO KANTE, und das Flachland kam auf 707 m Relief statt 90 m
    (02_INVARIANTEN.md 4). Das Laengsprofil kommt aus P, nicht aus dieser
    Konstanten.
    """
    yi = np.clip(np.round(points_px[:, 0]).astype(int), 0, size - 1)
    xi = np.clip(np.round(points_px[:, 1]).astype(int), 0, size - 1)
    p_at_node = P[yi, xi].astype(np.float64)

    max_order = max(int(strahler.max()), 1)
    incision = incision_m * (strahler.astype(np.float64) / max_order) ** 0.7
    z = p_at_node - incision

    # Eintiefen von den Quellen abwaerts. `order` ist nach Entfernung vom
    # Auslass aufsteigend sortiert, rueckwaerts also von den Quellen her -
    # jeder Knoten ist fertig, bevor sein Elternknoten an die Reihe kommt.
    min_drop = min_rise_fraction * peak_m
    for i in order[::-1]:
        e = parents[i]
        if e < 0:
            continue
        if z[e] > z[i] - min_drop:
            z[e] = z[i] - min_drop
    return z


# =============================================================================
# Verdichtung: Zwischenpunkte, Maeander, Spline
# =============================================================================

def _scheibe(radius: float):
    """Offset-Liste einer Scheibe - fuer das Stempeln der Sperrzone."""
    r = int(np.ceil(max(radius, 0.0)))
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    drin = (dy * dy + dx * dx) <= radius * radius + 1e-9
    return dy[drin].ravel(), dx[drin].ravel()


def _chaikin(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """
    Chaikin-Glaettung mit festgehaltenen Endpunkten - naehert einen
    quadratischen B-Spline an. Zwei Durchlaeufe reichen; mehr rundet die
    Maeander wieder weg.
    """
    for _ in range(iterations):
        neu = [points[0]]
        for a, b in zip(points[:-1], points[1:]):
            neu.append(0.75 * a + 0.25 * b)
            neu.append(0.25 * a + 0.75 * b)
        neu.append(points[-1])
        points = np.array(neu)
    return points


def densify(points_px, parents, order, strahler, step_px, meander, seed,
            P=None, size=None, cost_strength=2.5, separation_px=2.0,
            smoothing=1, korridor_px=0.0):
    """
    Fuehrt jede Kante als WEG GERINGSTER KOSTEN ueber das Hoehenfeld statt
    geradeaus, und haelt die Laeufe dabei kreuzungsfrei.

    WARUM. Der Knoten-Baum meidet die Ruecken bereits (gerichtete
    Anstiegskosten in spanning_tree). Die Verbindung ZWISCHEN zwei Knoten lief
    aber geradlinig plus Maeander und schnitt dabei ueber alles, was dazwischen
    lag - bei 2500 m Knotenabstand ueber einen ganzen Berg. Gemessen: der
    groesste Abtrag sass am HOECHSTEN Punkt der Karte, das Flussbett wurde dort
    auf -397 m gedrueckt, und 16 % aller Flusspixel hatten ueber 1000 m Abtrag
    (90_MESSPROTOKOLLE.md §15).

    Jetzt sucht sich jede Kante ihren Weg durch das Gelaende. Der Maeander
    entsteht dabei von selbst - der Lauf geht um den Berg herum, nicht
    hindurch.

    KEINE UEBERSCHNEIDUNGEN. Bereits gezeichnete Laeufe werden fuer spaetere
    Wege gesperrt (mit `separation_px` Sicherheitsabstand), ausser in der
    Umgebung des eigenen Elternknotens - dort MUSS der Nebenarm ja muenden.
    Die Kanten werden in der Reihenfolge vom Auslass nach aussen abgearbeitet,
    der Hauptlauf hat also Vorrang.

    Returns: (dense_px, dense_parents, dense_order). dense_parents[j] < j.
    """
    from skimage.graph import route_through_array

    if P is None or size is None:
        raise ValueError("densify braucht P und size fuer die Wegsuche")

    spanne = float(P.max() - P.min())
    hoehe_norm = (P - P.min()) / (spanne if spanne > 1e-9 else 1.0)
    # Kosten je Zelle: Grundkosten 1 plus Hoehenaufschlag. Der Weg bevorzugt
    # damit tiefes Gelaende und laeuft um Erhebungen herum. Quadratisch, damit
    # ein Gipfel wirklich gemieden wird und nicht nur leicht teurer ist.
    grundkosten = 1.0 + cost_strength * np.square(hoehe_norm)
    GESPERRT = 1.0e6

    # Statt einer reinen Sperrmaske wird vermerkt, WELCHER Knoten eine Zelle
    # belegt. Der Unterschied ist wesentlich: beruehrt ein Nebenarm seinen
    # eigenen Hauptlauf, ist das ein ZUSAMMENFLUSS und voellig richtig. Nur
    # die Beruehrung eines FREMDEN Astes waere eine Ueberschneidung.
    #
    # Eine Sperrmaske ohne diese Unterscheidung meldete 3-5 "Kreuzungen", die
    # in Wahrheit Muendungen waren - und blockierte zugleich Geschwister, die
    # denselben Elternknoten verlassen wollten.
    belegt = -np.ones((size, size), dtype=np.int64)
    scheibe_dy, scheibe_dx = _scheibe(max(separation_px, 0.0))

    def sperren(ys, xs, knoten):
        yy = np.clip(ys[:, None] + scheibe_dy[None, :], 0, size - 1)
        xx = np.clip(xs[:, None] + scheibe_dx[None, :], 0, size - 1)
        frei = belegt[yy.ravel(), xx.ravel()] < 0
        belegt[yy.ravel()[frei], xx.ravel()[frei]] = knoten

    dense = []
    dense_parents = []
    dense_order = []
    node_to_dense = {}
    umwege = 0

    for i in order:
        e = parents[i]
        if e < 0:
            node_to_dense[i] = len(dense)
            dense.append(points_px[i])
            dense_parents.append(-1)
            dense_order.append(int(strahler[i]))
            y0 = int(np.clip(round(points_px[i][0]), 0, size - 1))
            x0 = int(np.clip(round(points_px[i][1]), 0, size - 1))
            sperren(np.array([y0]), np.array([x0]), i)
            continue
        if e not in node_to_dense:
            continue

        # Vorfahren von i: deren Zellen sind erlaubt (Muendung in den eigenen
        # Hauptlauf), alle anderen nicht.
        vorfahren = set()
        k = e
        while k >= 0 and k not in vorfahren:
            vorfahren.add(int(k))
            k = parents[k]
        erlaubt = np.array(sorted(vorfahren), dtype=np.int64)

        ya = int(np.clip(round(points_px[e][0]), 0, size - 1))
        xa = int(np.clip(round(points_px[e][1]), 0, size - 1))
        yb = int(np.clip(round(points_px[i][0]), 0, size - 1))
        xb = int(np.clip(round(points_px[i][1]), 0, size - 1))
        if (ya, xa) == (yb, xb):
            node_to_dense[i] = node_to_dense[e]
            continue

        # Fenster um beide Endpunkte, mit Rand zum Ausweichen. Kommt der Weg
        # nicht an einer Sperre vorbei, wird das Fenster vergroessert - ein
        # zu kleines Fenster war die einzige Ursache der wenigen
        # Ueberschneidungen, die bei dichtem Netz auftraten (4 bei 1200 m
        # Talabstand, 0 bei 2500 m).
        pfad = None
        for versuch, faktor in enumerate((4.0, 10.0, 1e9)):
            rand = int(max(step_px * faktor, 12.0))
            oy = max(min(ya, yb) - rand, 0)
            ox = max(min(xa, xb) - rand, 0)
            uy = min(max(ya, yb) + rand + 1, size)
            ux = min(max(xa, xb) + rand + 1, size)

            kosten = grundkosten[oy:uy, ox:ux].copy()
            fremd = belegt[oy:uy, ox:ux]
            gy, gx = np.mgrid[oy:uy, ox:ux]
            # STRENG: jede schon belegte Zelle ist gesperrt, ausser in der
            # unmittelbaren Umgebung des eigenen Elternknotens - dort MUSS
            # gemuendet werden.
            #
            # Eine mildere Regel ("nur fremde Aeste sperren, Muendung in den
            # eigenen Hauptlauf erlauben") ist begrifflich richtiger, war aber
            # gemessen SCHLECHTER: bei 900 m Talabstand 12 Ueberschneidungen
            # statt 5, weil die Wege dann an den Hauptlaeufen entlang
            # abkuerzen und dabei fremde Aeste treffen.
            muendung = ((gy - ya) ** 2 + (gx - xa) ** 2) <= (separation_px + 2.0) ** 2
            kosten[(fremd >= 0) & ~muendung] = GESPERRT

            # KORRIDOR um die gerade Verbindung. Der Delaunay-Baum ist planar,
            # gerade Kanten koennen sich also nicht kreuzen. Sobald ein
            # gesuchter Weg weit ausschert, verlaesst er dieses Korsett und
            # kann dem Nachbarast den Weg abschneiden - genau daher kamen die
            # 3-4 Ueberschneidungen bei dichtem Netz, und ein groesseres
            # Suchfenster half nicht (es machte das Ausscheren nur leichter).
            #
            # Der Korridor laesst dem Weg genug Raum, um einen Berg zu
            # umgehen, ohne in fremdes Gebiet zu laufen.
            if korridor_px > 0:
                vy, vx = float(yb - ya), float(xb - xa)
                ll = vy * vy + vx * vx
                if ll > 1e-9:
                    ty = (gy - ya) * vy + (gx - xa) * vx
                    tt = np.clip(ty / ll, 0.0, 1.0)
                    ny = ya + tt * vy
                    nx = xa + tt * vx
                    abstand_segment = np.hypot(gy - ny, gx - nx)
                    kosten[abstand_segment > korridor_px] = GESPERRT

            try:
                roh, _ = route_through_array(
                    kosten, (ya - oy, xa - ox), (yb - oy, xb - ox),
                    fully_connected=True, geometric=True)
                kandidat = np.array(roh, dtype=np.float64) + np.array([oy, ox])
            except Exception:
                kandidat = np.array([[ya, xa], [yb, xb]], dtype=np.float64)

            ky = np.clip(np.round(kandidat[:, 0]).astype(int), 0, size - 1)
            kx = np.clip(np.round(kandidat[:, 1]).astype(int), 0, size - 1)
            weit = ((ky - ya) ** 2 + (kx - xa) ** 2) > (separation_px + 2.0) ** 2
            frei = not np.any(belegt[ky[weit], kx[weit]] >= 0)
            pfad, py, px_ = kandidat, ky, kx
            if frei:
                break
        else:
            umwege += 1
        if pfad is None:
            pfad = np.array([[ya, xa], [yb, xb]], dtype=np.float64)
            py = np.array([ya, yb]); px_ = np.array([xa, xb])

        if smoothing > 0 and len(pfad) > 3:
            geglaettet = _chaikin(pfad, int(smoothing))
            gy2 = np.clip(np.round(geglaettet[:, 0]).astype(int), 0, size - 1)
            gx2 = np.clip(np.round(geglaettet[:, 1]).astype(int), 0, size - 1)
            weit2 = ((gy2 - ya) ** 2 + (gx2 - xa) ** 2) > (separation_px + 2.0) ** 2
            # Nur uebernehmen, wenn die Glaettung keine Sperre verletzt -
            # sonst waere die Kreuzungsfreiheit nur noch ungefaehr.
            if not np.any(belegt[gy2[weit2], gx2[weit2]] >= 0):
                pfad = geglaettet
                py, px_ = gy2, gx2

        sperren(py, px_, i)

        vorher = node_to_dense[e]
        for q in pfad[1:]:
            dense.append(q)
            dense_parents.append(vorher)
            dense_order.append(int(strahler[i]))
            vorher = len(dense) - 1
        node_to_dense[i] = vorher

    if umwege:
        logger.warning("Flussnetz: %d Kanten mussten eine Sperre queren - "
                       "dort koennen sich Laeufe beruehren", umwege)

    return (np.array(dense), np.array(dense_parents, dtype=np.int64),
            np.array(dense_order, dtype=np.int32), umwege)


def catchment(dense_parents: np.ndarray) -> np.ndarray:
    """
    Einzugsgebiet je verdichtetem Punkt, als Anzahl der Punkte flussaufwaerts
    (einschliesslich sich selbst). Ein Rueckwaertslauf, weil
    dense_parents[j] < j gilt.

    WARUM NICHT DIE STRAHLER-ORDNUNG. Sie ist eine STUFE, und bei den hier
    auftretenden Netzgroessen reicht sie nur bis 3-5. Damit sind fast alle
    Taeler gleich breit, das Ergebnis sieht aus wie ein Schlauchsystem und die
    Quellen enden in runden Kappen (der Abstand zu einem Segmentende ist ein
    Halbkreis).

    Das Einzugsgebiet ist dagegen STETIG und geht an der Quelle gegen 1. Die
    Strahlweite des Flusseinflusses waechst damit gleichmaessig von der Quelle
    zum Auslass, und die Kappen verschwinden von selbst.
    """
    n = len(dense_parents)
    flaeche = np.ones(n, dtype=np.float64)
    for j in range(n - 1, 0, -1):
        e = dense_parents[j]
        if e >= 0:
            flaeche[e] += flaeche[j]
    return flaeche


def glaette_entlang_baum(werte, parents, iterationen=3, gewicht=0.5):
    """
    Glaettet Werte ENTLANG des Netzes (jeder Knoten mit Eltern und Kindern).

    Einzugsgebiet und damit Eintiefung und Talbreite springen sonst an jedem
    Zusammenfluss: der Nebenarm hat unmittelbar oberhalb der Muendung ein viel
    kleineres Einzugsgebiet als der Hauptlauf unmittelbar darunter, und im
    Gelaende steht dort eine Kante.

    Bewusst entlang des BAUMES und nicht in der Flaeche - eine
    Flaechenglaettung wuerde benachbarte, aber unverbundene Laeufe vermischen.
    """
    werte = np.asarray(werte, dtype=np.float64).copy()
    n = len(werte)
    for _ in range(max(int(iterationen), 0)):
        summe = np.zeros(n)
        anzahl = np.zeros(n)
        kind = np.arange(1, n)
        eltern = parents[1:]
        gueltig = eltern >= 0
        np.add.at(summe, eltern[gueltig], werte[kind[gueltig]])
        np.add.at(anzahl, eltern[gueltig], 1.0)
        np.add.at(summe, kind[gueltig], werte[eltern[gueltig]])
        np.add.at(anzahl, kind[gueltig], 1.0)
        nachbar = np.where(anzahl > 0, summe / np.maximum(anzahl, 1.0), werte)
        werte = (1.0 - gewicht) * werte + gewicht * nachbar
    return werte


def dense_heights(dense_px, dense_parents, dense_order, P, size,
                  peak_m, incision_m, min_drop_fraction, meters_per_pixel,
                  area_norm=None, area_exponent=0.30, max_gradient=0.0):
    """
    Hoehe je verdichtetem Punkt: dem Gelaende folgen, dann flussabwaerts
    eintiefen (nie anheben - siehe Modulkopf und 90_MESSPROTOKOLLE.md §12).

    Das Mindestgefaelle wird PRO METER LAUFLAENGE angesetzt, nicht pro
    Segment. Als Wert je Segment gerechnet haengt es an der Zahl der
    Stuetzpunkte: die Verdichtung machte aus jeder Kante rund siebzig
    Segmente, und der erzwungene Abtrag stieg damit um denselben Faktor - die
    Talsohle lag danach 1222 m unter dem Umland, obwohl 400 m eingestellt
    waren. Bezogen auf die Kartenbreite ist der gesamte erzwungene Abfall
    dagegen `min_drop_fraction * peak_m`, unabhaengig davon, wie fein der Lauf
    aufgeloest ist.
    """
    yi = np.clip(np.round(dense_px[:, 0]).astype(int), 0, size - 1)
    xi = np.clip(np.round(dense_px[:, 1]).astype(int), 0, size - 1)
    # Eintiefung waechst mit dem EINZUGSGEBIET, nicht mit der Strahler-Stufe:
    # stetig statt in 3-5 Spruengen. An der Quelle geht sie gegen null, der
    # Lauf setzt dort also weich an statt mit einer Stufe.
    if area_norm is None:
        max_order = max(int(dense_order.max()), 1)
        anteil = (dense_order.astype(np.float64) / max_order) ** 0.7
    else:
        anteil = np.power(np.clip(area_norm, 0.0, 1.0), area_exponent)
    incision = incision_m * anteil
    z = P[yi, xi].astype(np.float64) - incision

    # Gefaelle je Meter: ueber die gesamte Kartenbreite summiert sich der
    # erzwungene Abfall auf min_drop_fraction * peak_m.
    extent_m = float(meters_per_pixel) * size
    gefaelle = min_drop_fraction * peak_m / max(extent_m, 1.0)

    strecken = np.zeros(len(z))
    for j in range(1, len(z)):
        e = dense_parents[j]
        if e >= 0:
            strecken[j] = float(np.hypot(dense_px[j][0] - dense_px[e][0],
                                         dense_px[j][1] - dense_px[e][1])) * meters_per_pixel

    # 1. EINTIEFEN von den Quellen abwaerts - stellt das Mindestgefaelle her.
    #    dense_parents[j] < j, rueckwaerts also von den Quellen her.
    for j in range(len(z) - 1, 0, -1):
        e = dense_parents[j]
        if e < 0:
            continue
        drop = max(gefaelle * strecken[j], 1e-6)
        if z[e] > z[j] - drop:
            z[e] = z[j] - drop

    # 2. STEIGUNG BEGRENZEN, von der Muendung aufwaerts.
    #
    # Schritt 1 zieht einen Muendungsknoten auf die Hoehe seines TIEFSTEN
    # Zuflusses. Der andere Zufluss steht dann unmittelbar daneben noch auf
    # seiner eigenen Hoehe - gemessen 574 m Sprung zwischen zwei direkt
    # aufeinanderfolgenden Punkten. Genau die Kante, die an Zusammenfluessen
    # zu sehen war.
    #
    # Hier wird der Sprung ueber eine Strecke verteilt, statt ihn auf einem
    # Pixel stehen zu lassen: kein Lauf darf steiler ansteigen als
    # max_gradient. Vorwaerts, weil dense_parents[j] < j - der Elternknoten
    # ist dann schon endgueltig.
    #
    # Bricht die Monotonie NICHT: die Obergrenze senkt nur, und sie liegt
    # ueber dem Mindestgefaelle.
    if max_gradient > 0.0:
        for j in range(1, len(z)):
            e = dense_parents[j]
            if e < 0:
                continue
            hoechstens = z[e] + max_gradient * strecken[j]
            if z[j] > hoechstens:
                z[j] = hoechstens
    return z


# =============================================================================
# Talformer
# =============================================================================

def valley_profile(d_norm, form, steps=0, step_wall=0.35, step_tilt=0.18,
                   divide_blend=0.0):
    """
    profil(0) = 0 an der Talsohle, profil(1) = 1 an der Wasserscheide.
    form < 1 Schlucht, = 1 V, > 1 U (glazial). `steps` legt Klippenbaender
    darueber.

    `step_tilt` gibt den ABSAETZEN eine leichte Neigung statt sie exakt
    waagerecht zu lassen. Gemessen: mit ebenen Absaetzen faellt der
    Entwaesserungsanteil von 64% auf 7.3% und die Senkenzahl steigt von 363 auf
    2601 - eine exakt ebene Flaeche ist fuer D8 dasselbe wie eine Senke. Echte
    Schichtstufen sind ohnehin nie waagerecht.
    """
    profile = np.power(np.clip(d_norm, 0.0, 1.0), form)
    steps = int(steps)
    if steps > 0:
        wall = float(np.clip(step_wall, 0.05, 1.0))
        tilt = float(np.clip(step_tilt, 0.0, 1.0))
        s = profile * steps
        k = np.floor(s)
        rest = s - k
        wall_part = np.clip((rest - (1.0 - wall)) / wall, 0.0, 1.0)
        profile = (k + (1.0 - tilt) * wall_part + tilt * rest) / steps
    profile = np.clip(profile, 0.0, 1.0)

    # WEICHER UEBERGANG AN DER WASSERSCHEIDE.
    #
    # `profile` laeuft mit einer Steigung ungleich null gegen 1. Das ergibt
    # genau dort einen scharfen Grat, wo zwei Taeler zusammenstossen - und weil
    # das Abstandsfeld an der Mittelachse zwischen zwei Laeufen ohnehin eine
    # Knickkante hat, addieren sich beide zu dem kantigen, facettierten Bild,
    # das ein reines Voronoi-Gelaende kennzeichnet.
    #
    # smoothstep auf den Mischfaktor bringt die Ableitung an BEIDEN Enden auf
    # null: weicher Uebergang in die Hochflaeche, und zugleich eine flache
    # Talsohle statt einer Kerbe. divide_blend regelt, wieviel davon wirkt -
    # scharfe Grate (Alpen) bleiben damit einstellbar.
    # EINSEITIG glaetten: weich an der Wasserscheide, NICHT an der Talsohle.
    #
    # Vorher stand hier smoothstep, w*w*(3-2w). Das hat die Ableitung null an
    # BEIDEN Enden - also auch am Fluss, und damit wurde die Talsohle eben.
    # Genau die breiten flachen Boeden, die im Querschnitt zu sehen waren.
    # Eingebaut war es, um die Wasserscheide zu glaetten; seit der Talrand
    # exponentiell auslaeuft (siehe carve_river_network), wird dort ohnehin
    # nichts mehr abgeschnitten.
    #
    # w^p mit p > 1 hat die Ableitung null bei w = 0 (Wasserscheide, weich)
    # und p bei w = 1 (Talsohle, also eine echte Neigung statt einer Ebene).
    if divide_blend > 0.0:
        w = 1.0 - profile
        profile = 1.0 - np.power(w, 1.0 + 2.0 * divide_blend)
    return np.clip(profile, 0.0, 1.0)


# =============================================================================
# Hauptfunktion
# =============================================================================

def carve_river_network(P: np.ndarray, meters_per_pixel: float, peak_m: float,
                        seed: int, parameters: Optional[Dict[str, Any]] = None
                        ) -> Optional[Dict[str, Any]]:
    """
    Legt ein Flussnetz in die Flaeche P und liefert das geblendete Gelaende.

    Parameter: P - Flaeche ohne Fluesse (Meter), meters_per_pixel, peak_m -
        Gipfelhoehe (fuer relative Groessen), seed, parameters - siehe DEFAULTS.

    Returns: dict mit heightmap, river_mask, river_order, valley_distance -
        oder None, wenn die Karte fuer den eingestellten Flussabstand zu klein
        ist (dann bleibt P unveraendert).
    """
    p = dict(DEFAULTS)
    p.update(parameters or {})

    size = int(P.shape[0])
    extent_m = float(meters_per_pixel) * size
    spacing_m = float(p["river_spacing_m"])

    # Voellig flache Flaeche: es gibt kein Gefaelle, dem ein Fluss folgen
    # koennte, und ein Netz hineinzuschneiden erzeugt Relief, wo per Vorgabe
    # keines sein soll. Tritt auf, wenn amplitude gleich der Talsohlenhoehe
    # ist - test_terrain_generator() stellt genau das ein.
    if float(P.max() - P.min()) < 1e-9:
        logger.debug("Flussnetz uebersprungen: Flaeche ist eben")
        return None

    # Untergrenzen: unter vier Laeufen je Kartenkante ist ein "Netz" kein Netz
    # mehr, und unter drei Pixeln Abstand ist es nicht darstellbar. Bewusst
    # geometrisch statt als fester Meterwert - dieselbe Ueberlegung wie bei der
    # Drei-Pixel-Grenze der Rinnengroesse in 90_MESSPROTOKOLLE.md §10.
    if extent_m / max(spacing_m, 1e-6) < 4.0:
        logger.debug("Flussnetz uebersprungen: Abstand %.0f m auf %.0f m Karte "
                     "ergibt weniger als vier Laeufe", spacing_m, extent_m)
        return None
    spacing_m = max(spacing_m, 3.0 * meters_per_pixel)

    # AUFLOESUNG GEGEN TALABSTAND. Zwischen zwei Laeufen braucht es genug
    # Pixel, damit das Querprofil ueberhaupt Stufen hat - sonst entstehen
    # ebene Flaechen, und eben ist fuer D8 dasselbe wie eine Senke.
    # Gemessen bei 2500 m Talabstand: 128 px auf 15 km (21 px je Tal) ergeben
    # 17% Entwaesserung, 256 px (43 px je Tal) dagegen 87%.
    # Schwelle 2026-07-30 von 30 auf 40 angehoben: unterhalb von etwa 40
    # Pixeln je Tal treten zusaetzlich zu den Entwaesserungsproblemen einzelne
    # Ueberschneidungen der Laeufe auf (90_MESSPROTOKOLLE.md §16).
    pixel_je_tal = spacing_m / float(meters_per_pixel)
    if pixel_je_tal < 40.0:
        logger.warning(
            "Flussnetz: nur %.0f Pixel je Tal (Talabstand %.0f m bei %.0f m/px). "
            "Unter etwa 40 wird das Querprofil nicht mehr aufgeloest, die "
            "Entwaesserung bricht ein und einzelne Laeufe beruehren sich - "
            "hoehere Map Size oder groesserer Talabstand.",
            pixel_je_tal, spacing_m, meters_per_pixel)

    points_m = poisson_points(extent_m, spacing_m, seed)
    if len(points_m) < 8:
        logger.debug("Flussnetz uebersprungen: nur %d Punkte", len(points_m))
        return None
    points_px = points_m / float(meters_per_pixel)

    # Glaettung fuer das Kostenfeld: ein Viertel des Flussabstands, in Metern
    # vorgegeben und in Pixel umgerechnet.
    parents, order, auslaesse = spanning_tree(
        points_px, P, size, float(p["cost_strength"]), int(seed),
        smoothing_px=0.25 * spacing_m / float(meters_per_pixel),
        outlet_count=int(p["outlet_count"]),
        border_outflow=float(p["border_outflow"]))
    strahler = strahler_order(parents, order)

    # Kanten verdichten: Zwischenpunkte mit Maeander, danach Spline-Glaettung.
    step_px = max(float(p["densify_fraction"]) * spacing_m
                  / float(meters_per_pixel), 2.0)
    dense_px, dense_parents, dense_order, kreuzungen = densify(
        points_px, parents, order, strahler, step_px,
        float(p["meander"]), seed,
        P=P, size=size, cost_strength=float(p["cost_strength"]),
        separation_px=float(p["separation_px"]),
        smoothing=int(p["path_smoothing"]),
        korridor_px=float(p["corridor_fraction"]) * spacing_m
        / float(meters_per_pixel))
    # Einzugsgebiet je Punkt - loest die Strahler-Ordnung als Groessenmass ab.
    flaeche = catchment(dense_parents)
    # Entlang des Netzes glaetten, damit Talbreite und Eintiefung an den
    # Zusammenfluessen nicht springen.
    flaeche = glaette_entlang_baum(flaeche, dense_parents,
                                   int(p["confluence_smoothing"]))
    flaeche_norm = flaeche / max(float(flaeche.max()), 1.0)

    # EINTIEFUNG ALS ANTEIL DER HOEHENSPANNE.
    #
    # Vorher ein Meterwert - und der bedeutete in einer 4000-m-Landschaft etwas
    # voellig anderes als in einer 200-m-Landschaft. Bei 30 m Amplitude schnitt
    # die Vorgabe von 400 m dreizehnmal tiefer als die Landschaft hoch war.
    # Nach oben bei 0.6 gedeckelt: tief genug fuer einen Canyon, flach genug,
    # dass die Landschaft erkennbar bleibt (02_INVARIANTEN.md 7).
    einschnitt = float(np.clip(p["incision_share"], 0.0, 0.6)) * float(peak_m)

    z_dense = dense_heights(dense_px, dense_parents, dense_order, P, size,
                            peak_m, einschnitt,
                            float(p["min_rise_fraction"]), meters_per_pixel,
                            flaeche_norm,
                            float(p["incision_area_exponent"]),
                            float(p["max_gradient"]))

    # Hoehen entlang des Netzes glaetten - das nimmt die Kante an den
    # Muendungen. Danach ERNEUT eintiefen, weil die Glaettung die strenge
    # Monotonie brechen kann und ohne sie die Entwaesserung wegbricht.
    glatt_n = int(p["confluence_smoothing"])
    if glatt_n > 0:
        z_dense = glaette_entlang_baum(z_dense, dense_parents, glatt_n, 0.4)
        gefaelle = (float(p["min_rise_fraction"]) * peak_m
                    / max(extent_m, 1.0))
        for j in range(len(z_dense) - 1, 0, -1):
            e = dense_parents[j]
            if e < 0:
                continue
            strecke = float(np.hypot(dense_px[j][0] - dense_px[e][0],
                                     dense_px[j][1] - dense_px[e][1])) * meters_per_pixel
            drop = max(gefaelle * strecke, 1e-6)
            if z_dense[e] > z_dense[j] - drop:
                z_dense[e] = z_dense[j] - drop

    # --- Netz rasterisieren ---
    mask = np.zeros((size, size), dtype=bool)
    z_raster = np.full((size, size), np.inf)
    order_raster = np.zeros((size, size), dtype=np.int32)
    area_raster = np.zeros((size, size), dtype=np.float64)

    def zeichne(y0, x0, y1, x1, za, zb, ordnung, gebiet=0.0):
        steps = int(max(np.hypot(y1 - y0, x1 - x0) * 2.0, 2))
        t = np.linspace(0.0, 1.0, steps)
        ys = np.clip(np.round(y0 + t * (y1 - y0)).astype(int), 0, size - 1)
        xs = np.clip(np.round(x0 + t * (x1 - x0)).astype(int), 0, size - 1)
        zs = za + t * (zb - za)
        # Tiefster Wert gewinnt - ein Zusammenfluss darf keine Schwelle im
        # Flussbett erzeugen.
        better = zs < z_raster[ys, xs]
        z_raster[ys[better], xs[better]] = zs[better]
        order_raster[ys, xs] = np.maximum(order_raster[ys, xs], ordnung)
        area_raster[ys, xs] = np.maximum(area_raster[ys, xs], gebiet)
        mask[ys, xs] = True

    for j in range(len(dense_px)):
        e = dense_parents[j]
        if e < 0:
            continue
        zeichne(dense_px[e][0], dense_px[e][1],
                dense_px[j][0], dense_px[j][1],
                z_dense[e], z_dense[j], int(dense_order[j]),
                float(flaeche_norm[j]))

    # AUSLAUF BIS ZUM KARTENRAND. Der Auslassknoten liegt in Randnaehe, aber
    # nicht AUF dem Rand - ohne diesen Anschluss ist er der tiefste Punkt der
    # Karte und das gesamte Netz endet in einer Grube wenige Pixel vor dem
    # Ziel. Gemessen: 0.1% Senken bei 1.0% Entwaesserung, nach dem Anschluss
    # 79.9%. Die Senkenzahl allein zeigt diesen Fehler nicht.
    # Gefaelle des Auslaufs: dasselbe Mindestgefaelle wie im Netz.
    gefaelle_auslauf = (float(p["min_rise_fraction"]) * peak_m
                        / max(extent_m, 1.0))
    for w in np.where(dense_parents < 0)[0]:
        ay, ax = dense_px[w]
        target = min(((0.0, ax), (size - 1.0, ax), (ay, 0.0), (ay, size - 1.0)),
                     key=lambda q: (q[0] - ay) ** 2 + (q[1] - ax) ** 2)
        # Der Abfall des Auslaufs richtet sich nach seiner LAENGE, nicht nach
        # einem festen Betrag. Seit die Auslaesse wirklich am Rand liegen, ist
        # dieser letzte Abschnitt oft nur ein, zwei Pixel lang - ein fester
        # Abfall stand dann als Stufe im Flussbett (gemessen 83 m bei 1800 m
        # Hoehenspanne, waehrend p99 bei 11 m lag). 02_INVARIANTEN.md 4,
        # wieder.
        strecke_px = float(np.hypot(target[0] - ay, target[1] - ax))
        abfall = max(gefaelle_auslauf * strecke_px * meters_per_pixel, 0.5)
        zeichne(ay, ax, target[0], target[1], z_dense[w],
                z_dense[w] - abfall, int(dense_order[w]),
                float(flaeche_norm[w]))

    distance, index = ndimage.distance_transform_edt(
        ~mask, sampling=(meters_per_pixel, meters_per_pixel),
        return_indices=True)
    z_near = z_raster[index[0], index[1]]
    order_near = order_raster[index[0], index[1]]
    area_near = area_raster[index[0], index[1]]

    # P SENKENFREI MACHEN. Auf der Hochflaeche gilt z = P, also erbt sie jede
    # geschlossene Delle des Rauschens. Auffuellen bis zum Ueberlauf, danach
    # eine winzige Neigung ZUM NETZ HIN, damit die aufgefuellten Flaechen nicht
    # eben bleiben - eben ist fuer D8 dasselbe wie eine Senke.
    from skimage.morphology import reconstruction
    seed_img = P.max() * np.ones_like(P)
    seed_img[0], seed_img[-1] = P[0], P[-1]
    seed_img[:, 0], seed_img[:, -1] = P[:, 0], P[:, -1]
    P_filled = reconstruction(seed_img, P, method="erosion")
    P_filled = P_filled + (0.004 * peak_m) * (
        distance / max(float(distance.max()), 1e-9))

    # STRAHLWEITE DES FLUSSEINFLUSSES.
    #
    # Grundmass ist der Flussabstand - die Wasserscheide liegt zwangslaeufig
    # etwa in der Mitte zwischen zwei Laeufen, ein absoluter Meterwert waere
    # vom Netz entkoppelt (02_INVARIANTEN.md 4). Darauf skaliert das
    # EINZUGSGEBIET: ein Hauptfluss strahlt weit, ein Quellbach kaum.
    #
    # Vorher stand hier die Strahler-Ordnung. Sie ist eine Stufe und reicht bei
    # diesen Netzgroessen nur bis 3-5 - alle Taeler wurden dadurch gleich
    # breit ("Schlaeuche") und die Quellen endeten in runden Kappen. Das
    # Einzugsgebiet ist stetig und geht an der Quelle gegen null.
    width = (float(p["valley_width_fraction"]) * spacing_m
             * np.power(np.maximum(area_near, 1e-6),
                        float(p["valley_width_exponent"])))

    # VERZERRUNG DES ABSTANDSFELDES.
    #
    # Der euklidische Abstand zu einem Liniennetz erzeugt geometrisch saubere
    # Talraender - Parallelkurven zum Lauf, plus eine Knickkante auf der
    # Mittelachse. Genau das laesst das Tal wie AUSGESCHNITTEN wirken.
    #
    # Hier wird der Abstand mit der vorhandenen Gelaendestruktur multipliziert,
    # nicht addiert: wo P lokal tiefer liegt als seine Umgebung, zaehlt der
    # Abstand WENIGER, das Tal greift also weiter aus; auf lokalen Erhebungen
    # umgekehrt. Der Talrand wandert dadurch mit dem Gelaende, statt ihm eine
    # Kurve aufzuzwingen - und weil es eine Streckung des Abstands ist, bleibt
    # der Uebergang stetig (der Effekt am Fluss selbst bleibt bei d=0 exakt
    # 100 %, egal wie stark verzerrt wird).
    warp = float(p["edge_warp"])
    if warp > 0.0:
        glatt = ndimage.gaussian_filter(P, max(0.5 * pixel_je_tal, 1.0))
        rauheit = P - glatt
        streuung = float(rauheit.std())
        if streuung > 1e-9:
            rauheit = rauheit / streuung
            distance = distance * np.clip(1.0 + warp * rauheit, 0.15, 3.0)

    # AUSKLINGEN STATT ABSCHNEIDEN.
    #
    # `clip(distance / width, 0, 1)` gibt dem Flusseinfluss einen EXAKTEN
    # Radius: innerhalb wirkt er, ausserhalb gar nicht. Genau diese Grenze ist
    # im Bild als Talrand zu sehen, und sie laesst die Taeler wie aufgelegte
    # Baender wirken - unabhaengig davon, wie weich das Querprofil selbst ist.
    #
    # Stattdessen wird der Abstand ohne Obergrenze auf [0,1) abgebildet:
    #
    #     u = 1 - exp(-t^a)        mit t = Abstand / Talbreite
    #
    # u erreicht die 1 nie, der Einfluss klingt also aus statt zu enden - und
    # zwar an JEDER Stelle beliebig oft differenzierbar, es gibt also nirgends
    # einen Knick.
    #
    # Zuerst mit t^a/(1+t^a) versucht und verworfen: die Form klingt zu
    # langsam aus (bei dreifacher Talbreite noch 10 % Flusseinfluss), und der
    # sichtbare Anteil des Noise-Gelaendes fiel dadurch von 43 % auf 18 % -
    # genau gegen die Vorgabe "die Berge gehoeren dem Noise-Gelaende".
    # 1 - exp(-t^a) faellt dagegen bei doppelter Talbreite schon unter 2 %.
    #
    # `edge_softness` (= a) steuert, wie abrupt: grosse Werte kommen dem
    # frueheren harten Schnitt nahe, kleine ergeben einen langen weichen
    # Auslauf ins Noise-Gelaende.
    t = distance / np.maximum(width, 1e-9)
    a = max(float(p["edge_softness"]), 0.2)
    d_norm = 1.0 - np.exp(-np.power(t, a))
    profile = valley_profile(d_norm, float(p["valley_form"]),
                             divide_blend=float(p["divide_blend"]))

    heightmap = P_filled - (P_filled - z_near) * (1.0 - profile)

    return {
        "heightmap": heightmap.astype(np.float32),
        "river_mask": mask,
        "river_order": order_raster.astype(np.float32),
        "valley_distance": d_norm.astype(np.float32),
        "node_count": int(len(dense_px)),
        "crossings": int(kreuzungen),
        "max_order": int(strahler.max()),
    }
