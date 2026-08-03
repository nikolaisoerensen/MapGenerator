"""
Path: core/terrain_river_network.py

Flussnetz-Skelett mit Hochebene (SPEZIFIKATION §12).

Die Hoehe wird nicht vom Fluss AUFGEBAUT, sondern zwischen dem Fluss und einer
globalen Flaeche GEBLENDET:

    z = P - (P - z_Fluss) * (1 - Profil(d~))

P ist das Gelaende, das ohne Fluesse da waere (fBm + ATEF-Filter, also das
Ergebnis der bisherigen Terrain-Kette). Am Fluss (d~=0, Profil=0) ergibt das
z_Fluss, an der Wasserscheide (d~=1, Profil=1) ergibt es P.

WARUM DIESE UMKEHRUNG. Die naheliegende Form `z = z_Fluss(naechster) + Profil*H`
kann keine Hochebene erzeugen: jenseits der Talbreite haengt sie nur noch am
NAECHSTEN Fluss, und dessen Hoehe springt an jeder Wasserscheide. Ergebnis war
ein Flickwerk exakt ebener Terrassen mit Nahtstellen (§8: 62% aller Zellen ohne
streng tieferen Nachbarn). Hier laufen an der Wasserscheide beide Seiten gegen
denselben Wert P, der einwertig ist - die Naht faellt weg, und zwischen den
Taelern kann eine echte Hochflaeche liegen.

Aufbau in vier Schritten, alle nicht-iterativ:

  1. Poisson-Disk-Punktsatz mit Mindestabstand IN METERN. Deckt die Karte per
     Konstruktion ab - das war die offene Baustelle aus §8, wo ein gewachsener
     Baum bei 192 px nur 139 px weit kam.
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

Gemessener Stand (§12), 25 km, 384 px, ohne jede Erosionsiteration:
Entwaesserungsanteil 61-97% gegen 10.1% im reinen Noise-Pfad und 10-22% der
Feld-Erosion (§7).
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
    # Tiefe des groessten Tales unter der umgebenden Flaeche.
    "river_incision_m": 400.0,
    # Anteil der Talbreite am Flussabstand. 0.5 heisst: das Tal reicht genau
    # bis zur Mitte zwischen zwei Laeufen, es bleibt keine Hochflaeche uebrig.
    # Kleinere Werte lassen zwischen den Taelern eine Flaeche stehen.
    "valley_width_fraction": 0.55,
    # Wie stark die Talbreite mit der Flussgroesse waechst.
    "valley_width_exponent": 0.40,
    # Querprofil: <1 Schlucht, 1 V, >1 U (glazial).
    "valley_form": 1.6,
    # Klippenbaender im Querprofil (0 = keine).
    "valley_steps": 0,
    "valley_step_wall": 0.35,
    # Absaetze leicht geneigt statt eben - siehe valley_profile().
    "valley_step_tilt": 0.18,
    # Wie stark die Fluesse hohes Gelaende meiden.
    "cost_strength": 2.5,
    # Seitliche Auslenkung der Zwischenpunkte, als ANTEIL der Kantenlaenge.
    "meander": 0.18,
    # Abstand der Zwischenpunkte als Anteil des Flussabstands.
    "densify_fraction": 0.22,
    # Wie weich das Tal in die Umgebung uebergeht (0 = harte Kante an der
    # Wasserscheide, 1 = voll geglaettet). Siehe valley_profile().
    "divide_blend": 0.75,
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
    §10.
    """
    rng = np.random.default_rng(seed)
    cell = min_distance_m / np.sqrt(2.0)
    n = int(np.ceil(extent_m / cell)) + 1
    grid = -np.ones((n, n), dtype=np.int64)

    points = []
    active = []

    def insert(p):
        points.append(p)
        grid[int(p[0] / cell), int(p[1] / cell)] = len(points) - 1
        active.append(len(points) - 1)

    insert(rng.random(2) * extent_m)

    while active:
        k = int(rng.integers(0, len(active)))
        centre = points[active[k]]
        found = False
        for _ in range(attempts):
            angle = rng.random() * 2.0 * np.pi
            radius = min_distance_m * (1.0 + rng.random())
            p = centre + radius * np.array([np.cos(angle), np.sin(angle)])
            if not (0.0 <= p[0] < extent_m and 0.0 <= p[1] < extent_m):
                continue
            gy, gx = int(p[0] / cell), int(p[1] / cell)
            free = True
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    yy, xx = gy + dy, gx + dx
                    if 0 <= yy < n and 0 <= xx < n and grid[yy, xx] >= 0:
                        q = points[grid[yy, xx]]
                        if ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
                                < min_distance_m ** 2):
                            free = False
                            break
                if not free:
                    break
            if free:
                insert(p)
                found = True
                break
        if not found:
            active.pop(k)

    return np.array(points)


# =============================================================================
# 2.+3. Graph und Spannbaum
# =============================================================================

def spanning_tree(points_px: np.ndarray, P: np.ndarray, size: int,
                  cost_strength: float, smoothing_px: float = 0.0):
    """
    Delaunay-Graph, dann kuerzeste Wege vom Auslass am Kartenrand.

    Die Kantenkosten steigen mit der Hoehe von P. Das ist der brauchbare Kern
    der naheliegenden Idee "verbinde die lokalen Minima": nicht die Minima als
    KNOTEN nehmen - die haengen an Frequenz und Aufloesung und sind nach dem
    Eingraben ohnehin keine Minima mehr - sondern die Hoehe als KOSTENFELD.

    Returns: (parents, order_by_distance, outlet). parents[i] ist der Knoten
    FLUSSABWAERTS, -1 am Auslass.
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
    cost = length * (1.0 + cost_strength * 0.5
                     * (h_norm[edges[:, 0]] + h_norm[edges[:, 1]]))

    n = len(points_px)
    matrix = csr_matrix(
        (np.concatenate([cost, cost]),
         (np.concatenate([edges[:, 0], edges[:, 1]]),
          np.concatenate([edges[:, 1], edges[:, 0]]))), shape=(n, n))

    # Auslass: tiefster Punkt in Randnaehe. Nur einer - endorheische Becken
    # sind bewusst zurueckgestellt.
    border = ((points_px[:, 0] < size * 0.05) | (points_px[:, 0] > size * 0.95)
              | (points_px[:, 1] < size * 0.05) | (points_px[:, 1] > size * 0.95))
    if not np.any(border):
        border = np.zeros(n, dtype=bool)
        border[int(np.argmin(points_px[:, 0]))] = True
    candidates = np.where(border)[0]
    outlet = int(candidates[np.argmin(h[candidates])])

    distance, predecessors = dijkstra(matrix, indices=outlet,
                                      return_predecessors=True)
    parents = predecessors.astype(np.int64)
    parents[outlet] = -1

    finite = np.isfinite(distance)
    order = np.argsort(np.where(finite, distance, np.inf))
    order = order[finite[order]]
    return parents, order, outlet


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
    PRO KANTE, und das Flachland kam auf 707 m Relief statt 90 m (§4.4).
    Das Laengsprofil kommt aus P, nicht aus dieser Konstanten.
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


def densify(points_px, parents, order, strahler, step_px, meander, seed):
    """
    Ersetzt jede gerade Kante durch einen gemaeanderten, geglaetteten Lauf.

    WARUM. Eine Kante ueberspannt den vollen Knotenabstand - bei 2500 m
    Talabstand also 2500 m Luftlinie. Das hat zwei sichtbare Folgen:

      * Der Lauf schneidet geradlinig durch alles, was dazwischen liegt.
        Gemessen lag die Talsohle dadurch schon bei Einschnitttiefe 0 rund
        1160 m unter dem Umland - der Talterm ueberdeckte die gesamte
        Gelaendetextur, statt sich mit ihr zu mischen.
      * Gerade Linien sehen nicht wie Fluesse aus.

    Mit Zwischenpunkten folgt der Lauf dem Gelaende (jeder Punkt nimmt seine
    Hoehe aus P), und der Einschnitt ist wieder das, was der Regler sagt.

    Die Zwischenpunkte werden quer zur Laufrichtung ausgelenkt (Maeander,
    Amplitude als ANTEIL der Kantenlaenge, nicht in Metern - §4.4) und danach
    per Chaikin geglaettet.

    Returns: (dense_px, dense_parents, dense_order). dense_parents[j] < j,
    der Elternknoten kommt also immer vorher.
    """
    rng = np.random.default_rng(seed + 7919)
    dense = []
    dense_parents = []
    dense_order = []
    node_to_dense = {}

    for i in order:
        e = parents[i]
        if e < 0:
            node_to_dense[i] = len(dense)
            dense.append(points_px[i])
            dense_parents.append(-1)
            dense_order.append(int(strahler[i]))
            continue
        if e not in node_to_dense:
            continue

        p0 = points_px[e]
        p1 = points_px[i]
        richtung = p1 - p0
        laenge = float(np.hypot(richtung[0], richtung[1]))
        if laenge < 1e-9:
            node_to_dense[i] = node_to_dense[e]
            continue
        quer = np.array([-richtung[1], richtung[0]]) / laenge

        anzahl = max(int(round(laenge / max(step_px, 1e-6))), 1)
        stuetzen = [p0]
        for j in range(1, anzahl):
            t = j / float(anzahl)
            versatz = rng.normal(0.0, meander * laenge * 0.5)
            stuetzen.append(p0 + t * richtung + quer * versatz)
        stuetzen.append(p1)
        lauf = _chaikin(np.array(stuetzen), 2)

        vorher = node_to_dense[e]
        for q in lauf[1:]:
            dense.append(q)
            dense_parents.append(vorher)
            dense_order.append(int(strahler[i]))
            vorher = len(dense) - 1
        node_to_dense[i] = vorher

    return (np.array(dense), np.array(dense_parents, dtype=np.int64),
            np.array(dense_order, dtype=np.int32))


def dense_heights(dense_px, dense_parents, dense_order, P, size,
                  peak_m, incision_m, min_drop_fraction, meters_per_pixel):
    """
    Hoehe je verdichtetem Punkt: dem Gelaende folgen, dann flussabwaerts
    eintiefen (nie anheben - siehe Modulkopf und §12).

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
    max_order = max(int(dense_order.max()), 1)
    incision = incision_m * (dense_order.astype(np.float64) / max_order) ** 0.7
    z = P[yi, xi].astype(np.float64) - incision

    # Gefaelle je Meter: ueber die gesamte Kartenbreite summiert sich der
    # erzwungene Abfall auf min_drop_fraction * peak_m.
    extent_m = float(meters_per_pixel) * size
    gefaelle = min_drop_fraction * peak_m / max(extent_m, 1.0)

    # dense_parents[j] < j, rueckwaerts also von den Quellen her.
    for j in range(len(z) - 1, 0, -1):
        e = dense_parents[j]
        if e < 0:
            continue
        strecke = float(np.hypot(dense_px[j][0] - dense_px[e][0],
                                 dense_px[j][1] - dense_px[e][1])) * meters_per_pixel
        drop = max(gefaelle * strecke, 1e-6)
        if z[e] > z[j] - drop:
            z[e] = z[j] - drop
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
    if divide_blend > 0.0:
        w = 1.0 - profile
        weich = w * w * (3.0 - 2.0 * w)
        profile = 1.0 - ((1.0 - divide_blend) * w + divide_blend * weich)
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
    # Drei-Pixel-Grenze der Rinnengroesse in §10.
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
    pixel_je_tal = spacing_m / float(meters_per_pixel)
    if pixel_je_tal < 30.0:
        logger.warning(
            "Flussnetz: nur %.0f Pixel je Tal (Talabstand %.0f m bei %.0f m/px). "
            "Unter etwa 30 wird das Querprofil nicht mehr aufgeloest und die "
            "Entwaesserung bricht ein - hoehere Map Size oder groesserer "
            "Talabstand.", pixel_je_tal, spacing_m, meters_per_pixel)

    points_m = poisson_points(extent_m, spacing_m, seed)
    if len(points_m) < 8:
        logger.debug("Flussnetz uebersprungen: nur %d Punkte", len(points_m))
        return None
    points_px = points_m / float(meters_per_pixel)

    # Glaettung fuer das Kostenfeld: ein Viertel des Flussabstands, in Metern
    # vorgegeben und in Pixel umgerechnet.
    parents, order, outlet = spanning_tree(
        points_px, P, size, float(p["cost_strength"]),
        smoothing_px=0.25 * spacing_m / float(meters_per_pixel))
    strahler = strahler_order(parents, order)

    # Kanten verdichten: Zwischenpunkte mit Maeander, danach Spline-Glaettung.
    step_px = max(float(p["densify_fraction"]) * spacing_m
                  / float(meters_per_pixel), 2.0)
    dense_px, dense_parents, dense_order = densify(
        points_px, parents, order, strahler, step_px,
        float(p["meander"]), seed)
    z_dense = dense_heights(dense_px, dense_parents, dense_order, P, size,
                            peak_m, float(p["river_incision_m"]),
                            float(p["min_rise_fraction"]), meters_per_pixel)

    # --- Netz rasterisieren ---
    mask = np.zeros((size, size), dtype=bool)
    z_raster = np.full((size, size), np.inf)
    order_raster = np.zeros((size, size), dtype=np.int32)

    def zeichne(y0, x0, y1, x1, za, zb, ordnung):
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
        mask[ys, xs] = True

    for j in range(len(dense_px)):
        e = dense_parents[j]
        if e < 0:
            continue
        zeichne(dense_px[e][0], dense_px[e][1],
                dense_px[j][0], dense_px[j][1],
                z_dense[e], z_dense[j], int(dense_order[j]))

    # AUSLAUF BIS ZUM KARTENRAND. Der Auslassknoten liegt in Randnaehe, aber
    # nicht AUF dem Rand - ohne diesen Anschluss ist er der tiefste Punkt der
    # Karte und das gesamte Netz endet in einer Grube wenige Pixel vor dem
    # Ziel. Gemessen: 0.1% Senken bei 1.0% Entwaesserung, nach dem Anschluss
    # 79.9%. Die Senkenzahl allein zeigt diesen Fehler nicht.
    ay, ax = dense_px[0]
    target = min(((0.0, ax), (size - 1.0, ax), (ay, 0.0), (ay, size - 1.0)),
                 key=lambda q: (q[0] - ay) ** 2 + (q[1] - ax) ** 2)
    zeichne(ay, ax, target[0], target[1], z_dense[0],
            z_dense[0] - (0.01 * peak_m + 1.0), int(dense_order[0]))

    distance, index = ndimage.distance_transform_edt(
        ~mask, sampling=(meters_per_pixel, meters_per_pixel),
        return_indices=True)
    z_near = z_raster[index[0], index[1]]
    order_near = order_raster[index[0], index[1]]

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

    # Talbreite als Anteil des FLUSSABSTANDS - die Wasserscheide liegt
    # zwangslaeufig etwa in der Mitte zwischen zwei Laeufen. Als absoluter
    # Meterwert waere sie vom Netz entkoppelt (§4.4).
    width = (float(p["valley_width_fraction"]) * spacing_m
             * np.power(np.maximum(order_near, 1),
                        float(p["valley_width_exponent"])))
    d_norm = np.clip(distance / width, 0.0, 1.0)
    profile = valley_profile(d_norm, float(p["valley_form"]),
                             int(p["valley_steps"]),
                             float(p["valley_step_wall"]),
                             float(p["valley_step_tilt"]),
                             float(p["divide_blend"]))

    heightmap = P_filled - (P_filled - z_near) * (1.0 - profile)

    return {
        "heightmap": heightmap.astype(np.float32),
        "river_mask": mask,
        "river_order": order_raster.astype(np.float32),
        "valley_distance": d_norm.astype(np.float32),
        "node_count": int(len(dense_px)),
        "max_order": int(strahler.max()),
    }
