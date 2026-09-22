"""
Path: tools/flussnetz_lab.py

WERKZEUG fuer das Flussnetz mit Hochebenen (Entwurf vom 2026-07-30).

Loest den Ansatz aus SPEZIFIKATION §8 ab. Der Unterschied in einem Satz: die
Hoehe wird nicht mehr VOM FLUSS AUFGEBAUT, sondern zwischen dem Fluss und einer
globalen Flaeche GEBLENDET.

    z = P - (P - z_Fluss) * (1 - Profil(d~))

P ist das Gelaende, das ohne Fluesse da waere (fBm plus ATEF-Filter). Am Fluss
(d~=0, Profil=0) ergibt das z_Fluss, an der Wasserscheide (d~=1, Profil=1)
ergibt es P.

WARUM DIESE UMKEHRUNG. Die alte Form `z = z_Fluss(naechster) + Profil * H`
konnte keine Hochebene erzeugen: jenseits der Talbreite haengt sie nur noch am
NAECHSTEN Fluss, und dessen Hoehe springt an jeder Wasserscheide. Das Ergebnis
war ein Flickwerk aus exakt ebenen Terrassen mit Nahtstellen - gemessen in §8
als 62% aller Zellen ohne streng tieferen Nachbarn. In der neuen Form laufen an
der Wasserscheide beide Seiten gegen denselben Wert P, der einwertig ist.

Drei Groessen tragen damit den Charakter einer Landschaft:

    punktabstand_m   Abstand der Taeler
    einschnitt_m     wie tief die Taeler unter die Umgebung schneiden
    P-Relief         wie bewegt die Flaeche ZWISCHEN den Taelern ist
                     (glatt = Hochebene wie Norwegen, kraeftig = Alpen)

Der Bau des Netzes in vier Schritten:

  1. Poisson-Disk-Punktsatz mit Mindestabstand in METERN. Deckt die Karte per
     Konstruktion ab - das war die offene Baustelle aus §8, wo der gewachsene
     Baum bei 192 px Karte nur 139 px weit kam.
  2. Delaunay-Graph darueber. Planar, deshalb kann sich kein Teilgraph davon
     selbst kreuzen.
  3. Kuerzeste-Wege-Baum vom Auslass am Kartenrand, mit Kantenkosten, die mit
     der Hoehe von P steigen. Dadurch suchen sich die Fluesse das Tiefe, ohne
     an den zufaelligen Minima des Rauschens zu haengen. Ein Spannbaum hat
     keine Schleifen und erreicht jeden Punkt - Haupt- und Nebenfluesse in
     einem Aufruf.
  4. Strahler-Ordnung rueckwaerts abgelesen. Die "Generationen" muessen nicht
     konstruiert werden, sie stehen im Baum.

KEIN LOD. Der Nutzer entfernt sich davon (2026-07-30); nichts hier haengt an
einer LOD-Stufe.

Aufruf:
    .venv\\Scripts\\python.exe tools/flussnetz_lab.py
    .venv\\Scripts\\python.exe tools/flussnetz_lab.py 512
"""

import os
import sys
import time

import numpy as np
from scipy import ndimage

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "lab_output", "flussnetz_lab")

AUSSCHNITT_KM = 25.0
SEED = 20260730


# =============================================================================
# DIE FUENF LANDSCHAFTEN
# =============================================================================
# ALLE ZAHLEN ZU BESTAETIGEN - aus dem Augenschein der fuenf Referenzbilder des
# Nutzers, nicht aus Hoehenmodellen.
#
# gipfel_m        Hoehe der hoechsten Stelle (Talsohle liegt bei 0, siehe
#                 TERRAIN.BASE_ELEVATION_M)
# p_relief_anteil wie viel des Gipfels die Flaeche ZWISCHEN den Taelern
#                 ausmacht. Klein = Hochebene, gross = Bergland.
# einschnitt_m    Tiefe des groessten Tales unter der Umgebung

LANDSCHAFTEN = {
    "1_alpen": {
        "bemerkung": "Oetztal: kraeftige Flaeche, tiefe Troege, scharfe Grate",
        "gipfel_m": 3500.0,
        "punktabstand_m": 3500.0,
        "einschnitt_m": 1500.0,
        "p_feature_m": 9000.0, "p_relief_anteil": 1.00, "p_strength": 0.30,
        "p_ridge_rounding": 0.0,
        "talform_exponent": 2.0,          # U, glazial
        "tal_breite_anteil": 0.55, "tal_breite_exponent": 0.40,
        "kosten_staerke": 3.0,
        "stufen": 0,
    },
    "2_mittelgebirge": {
        "bemerkung": "Plateau mit dichter dendritischer Zertalung",
        "gipfel_m": 550.0,
        "punktabstand_m": 1400.0,         # dicht
        "einschnitt_m": 220.0,
        "p_feature_m": 7000.0, "p_relief_anteil": 0.55, "p_strength": 0.18,
        "p_ridge_rounding": 0.5,
        "talform_exponent": 1.5,
        "tal_breite_anteil": 0.70, "tal_breite_exponent": 0.45,
        "kosten_staerke": 2.5,
        "stufen": 0,
    },
    "3_plattland": {
        "bemerkung": "Norddeutsche Tiefebene: fast kein Einschnitt",
        "gipfel_m": 90.0,
        "punktabstand_m": 3000.0,
        "einschnitt_m": 22.0,             # kaum eingetieft
        "p_feature_m": 12000.0, "p_relief_anteil": 0.80, "p_strength": 0.10,
        "p_ridge_rounding": 0.9,
        "talform_exponent": 1.2,
        "tal_breite_anteil": 0.90, "tal_breite_exponent": 0.5,
        "kosten_staerke": 1.5,
        "stufen": 0,
    },
    "4_fjordland": {
        "bemerkung": "Westnorwegen: GLATTE Hochflaeche, tief eingeschnittene Troege",
        "gipfel_m": 1400.0,
        "punktabstand_m": 4500.0,         # weit auseinander
        "einschnitt_m": 1150.0,           # sehr tief
        "p_feature_m": 11000.0,
        "p_relief_anteil": 0.28,          # DAS ist die Hochebene
        "p_strength": 0.10,
        "p_ridge_rounding": 0.85,
        "talform_exponent": 2.4,          # U-Trog
        "tal_breite_anteil": 0.40, "tal_breite_exponent": 0.35,
        "kosten_staerke": 2.0,
        "stufen": 0,
    },
    "5_vietnam": {
        "bemerkung": "Kalkbergland: enge dichte Kegel, steile Waende",
        "gipfel_m": 1100.0,
        "punktabstand_m": 1100.0,         # sehr dicht
        "einschnitt_m": 620.0,
        "p_feature_m": 5000.0, "p_relief_anteil": 0.85, "p_strength": 0.28,
        "p_ridge_rounding": 0.05,
        "talform_exponent": 0.55,         # Wand direkt am Fluss
        "tal_breite_anteil": 0.55, "tal_breite_exponent": 0.3,
        "kosten_staerke": 2.0,
        "stufen": 2,
    },
}


# =============================================================================
# 1. PUNKTSATZ - Poisson-Disk (Bridson)
# =============================================================================

def poisson_punkte(size, min_abstand, rng, versuche=30):
    """
    Punktsatz mit garantiertem Mindestabstand, der die Flaeche fuellt.

    Genau die Eigenschaft, die dem gewachsenen Baum aus §8 fehlte: dort blieb
    der maximale Flussabstand bei 139 px auf einer 192-px-Karte, weil das
    Wachstum auslief. Ein Poisson-Satz deckt die Karte ab, BEVOR irgendetwas
    verbunden wird.
    """
    zelle = min_abstand / np.sqrt(2.0)
    n = int(np.ceil(size / zelle)) + 1
    gitter = -np.ones((n, n), dtype=np.int64)

    punkte = []
    aktiv = []

    def einfuegen(p):
        punkte.append(p)
        i = len(punkte) - 1
        gitter[int(p[0] / zelle), int(p[1] / zelle)] = i
        aktiv.append(i)

    einfuegen(rng.random(2) * size)

    while aktiv:
        k = int(rng.integers(0, len(aktiv)))
        mitte = punkte[aktiv[k]]
        gefunden = False
        for _ in range(versuche):
            winkel = rng.random() * 2.0 * np.pi
            radius = min_abstand * (1.0 + rng.random())
            p = mitte + radius * np.array([np.cos(winkel), np.sin(winkel)])
            if not (0.0 <= p[0] < size and 0.0 <= p[1] < size):
                continue
            gy, gx = int(p[0] / zelle), int(p[1] / zelle)
            frei = True
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    yy, xx = gy + dy, gx + dx
                    if 0 <= yy < n and 0 <= xx < n and gitter[yy, xx] >= 0:
                        q = punkte[gitter[yy, xx]]
                        if (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 < min_abstand ** 2:
                            frei = False
                            break
                if not frei:
                    break
            if frei:
                einfuegen(p)
                gefunden = True
                break
        if not gefunden:
            aktiv.pop(k)

    return np.array(punkte)


# =============================================================================
# 2.+3. GRAPH UND SPANNBAUM
# =============================================================================

def spannbaum(punkte, P, size, kosten_staerke):
    """
    Delaunay-Graph ueber den Punktsatz, dann kuerzeste Wege vom Auslass.

    Die Kantenkosten steigen mit der Hoehe von P - dadurch suchen sich die
    Fluesse das Tiefe. Das ist der brauchbare Kern der Nutzer-Idee "verbinde
    die lokalen Minima": nicht die Minima als KNOTEN nehmen (die haengen an
    Frequenz und Aufloesung und sind nach dem Eingraben keine Minima mehr),
    sondern die Hoehe als KOSTENFELD.

    Rueckgabe: eltern (Index flussabwaerts, -1 am Auslass), reihenfolge
    (Knoten nach Entfernung vom Auslass aufsteigend), auslass.
    """
    from scipy.spatial import Delaunay
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    tri = Delaunay(punkte)
    kanten = set()
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            kanten.add((min(s[a], s[b]), max(s[a], s[b])))
    kanten = np.array(sorted(kanten))

    # Hoehe je Punkt aus P, auf 0..1 normiert.
    yi = np.clip(np.round(punkte[:, 0]).astype(int), 0, size - 1)
    xi = np.clip(np.round(punkte[:, 1]).astype(int), 0, size - 1)
    h = P[yi, xi]
    spanne = float(h.max() - h.min())
    h_norm = (h - h.min()) / (spanne if spanne > 1e-9 else 1.0)

    laenge = np.linalg.norm(punkte[kanten[:, 0]] - punkte[kanten[:, 1]], axis=1)
    kosten = laenge * (1.0 + kosten_staerke * 0.5
                       * (h_norm[kanten[:, 0]] + h_norm[kanten[:, 1]]))

    n = len(punkte)
    matrix = csr_matrix(
        (np.concatenate([kosten, kosten]),
         (np.concatenate([kanten[:, 0], kanten[:, 1]]),
          np.concatenate([kanten[:, 1], kanten[:, 0]]))),
        shape=(n, n))

    # Auslass: der TIEFSTE Punkt am Kartenrand. Nur einer - "Rand, Becken
    # spaeter" (Nutzer 2026-07-30).
    rand = ((punkte[:, 0] < size * 0.03) | (punkte[:, 0] > size * 0.97)
            | (punkte[:, 1] < size * 0.03) | (punkte[:, 1] > size * 0.97))
    if not np.any(rand):
        rand = np.zeros(n, dtype=bool)
        rand[int(np.argmin(punkte[:, 0]))] = True
    kandidaten = np.where(rand)[0]
    auslass = int(kandidaten[np.argmin(h[kandidaten])])

    entfernung, vorgaenger = dijkstra(matrix, indices=auslass,
                                      return_predecessors=True)
    eltern = vorgaenger.astype(np.int64)
    eltern[auslass] = -1

    # Reihenfolge nach Entfernung: der Elternknoten kommt immer vorher.
    endlich = np.isfinite(entfernung)
    reihenfolge = np.argsort(np.where(endlich, entfernung, np.inf))
    reihenfolge = reihenfolge[endlich[reihenfolge]]
    return eltern, reihenfolge, auslass, entfernung


def strahler(eltern, reihenfolge):
    """
    Strahler-Ordnung, ein Rueckwaertslauf ueber die Reihenfolge.

    Die "Generationen" des Nutzer-Entwurfs muessen nicht konstruiert werden -
    sie stehen bereits im Baum und werden hier nur abgelesen. Rueckwaerts ueber
    reihenfolge heisst: von den Quellen zum Auslass, jedes Kind vor seinem
    Elternknoten.
    """
    n = len(eltern)
    ordnung = np.ones(n, dtype=np.int32)
    beste = np.zeros(n, dtype=np.int32)
    anzahl = np.zeros(n, dtype=np.int32)

    for i in reihenfolge[::-1]:
        if anzahl[i] == 0:
            ordnung[i] = 1
        elif anzahl[i] == 1:
            ordnung[i] = beste[i]
        else:
            ordnung[i] = beste[i] + 1
        e = eltern[i]
        if e < 0:
            continue
        if ordnung[i] > beste[e]:
            beste[e], anzahl[e] = ordnung[i], 1
        elif ordnung[i] == beste[e]:
            anzahl[e] += 1
    return ordnung


# =============================================================================
# 4. HOEHEN
# =============================================================================

def flusshoehen(punkte, eltern, reihenfolge, ordnung, P, size, p_param, mpp):
    """
    Hoehe je Knoten - MONOTON steigend flussaufwaerts, und trotzdem dem
    Gelaende folgend.

        s(i) = max( s(eltern) + Anstieg, P(i) )

    Der Anstieg haengt an der Strahler-Ordnung, nicht am Abstand: kleine
    Baeche steil, der Hauptlauf traege. Das ergibt das KONKAVE Laengsprofil
    echter Fluesse; ein konstanter Wert pro Meter gaebe eine Gerade.

    Die Monotonie gewinnt gegen P, nicht umgekehrt. Genau diese Entscheidung
    hat in §8 die Entwaesserung ueberhaupt erst hergestellt, und der
    umgekehrte Weg ist der, an dem §7 gescheitert ist.

    Danach wird der Einschnitt abgezogen: z_Fluss = s - Einschnitt(Ordnung).
    Grosse Fluesse schneiden tiefer, und weil die Ordnung flussabwaerts
    steigt, verstaerkt das die Monotonie zusaetzlich.
    """
    yi = np.clip(np.round(punkte[:, 0]).astype(int), 0, size - 1)
    xi = np.clip(np.round(punkte[:, 1]).astype(int), 0, size - 1)
    p_am_knoten = P[yi, xi].astype(np.float64)

    s = p_am_knoten.copy()
    s[reihenfolge[0]] = p_am_knoten[reihenfolge[0]]

    # MINDESTANSTIEG je Kante, nicht je Meter.
    #
    # Vorher stand hier eine Steigung von 3% mal Kantenlaenge. Bei 3000 m
    # langen Kanten sind das 90 m ANSTIEG PRO KANTE, ueber zehn Kanten also
    # 900 m - das Flachland kam damit auf 707 m Relief statt 90 m. Wieder der
    # Fehlertyp aus §4.4: eine absolute Groesse, wo eine relative hingehoert.
    #
    # Der Anstieg hat hier nur EINE Aufgabe: strenge Monotonie herstellen, wo
    # P selbst eine Delle hat. Er ist deshalb winzig und am Gesamtrelief
    # bemessen. Das Laengsprofil kommt aus P, nicht aus dieser Konstanten.
    mindestanstieg = p_param["mindestanstieg_anteil"] * p_param["gipfel_m"]
    for i in reihenfolge[1:]:
        e = eltern[i]
        if e < 0:
            continue
        s[i] = max(s[e] + mindestanstieg, p_am_knoten[i])

    max_ordnung = max(int(ordnung.max()), 1)
    einschnitt = p_param["einschnitt_m"] * (
        ordnung.astype(np.float64) / max_ordnung) ** 0.7
    return s - einschnitt, einschnitt


# =============================================================================
# TALFORMER (unveraendert aus dem Skelett-Labor)
# =============================================================================

def talformer(d_norm, exponent, stufen=0, stufen_wandanteil=0.35,
              stufen_neigung=0.18):
    """
    profil(0) = 0 an der Talsohle, profil(1) = 1 an der Wasserscheide.
    Exponent < 1 Schlucht, = 1 V, > 1 U. `stufen` legt Klippenbaender darueber.

    `stufen_neigung` gibt den ABSAETZEN eine leichte Neigung, statt sie exakt
    waagerecht zu lassen. Gemessen 2026-07-30 an Vietnam: mit ebenen Absaetzen
    faellt der Entwaesserungsanteil von 64% auf 7.3% und die Senkenzahl steigt
    von 363 auf 2601 - eine exakt ebene Flaeche ist fuer D8 dasselbe wie eine
    Senke. Echte Schichtstufen sind ohnehin nie waagerecht.
    """
    profil = np.power(np.clip(d_norm, 0.0, 1.0), exponent)
    if stufen > 0:
        wand = float(np.clip(stufen_wandanteil, 0.05, 1.0))
        neigung = float(np.clip(stufen_neigung, 0.0, 1.0))
        s = profil * stufen
        k = np.floor(s)
        rest = s - k
        wandanteil = np.clip((rest - (1.0 - wand)) / wand, 0.0, 1.0)
        # Absatz leicht geneigt statt eben: der Rest traegt anteilig bei.
        profil = (k + (1.0 - neigung) * wandanteil + neigung * rest) / stufen
    return np.clip(profil, 0.0, 1.0)


# =============================================================================
# ZUSAMMENBAU
# =============================================================================

def baue_P(name, size):
    """
    Die Flaeche OHNE Fluesse: das heutige Terrain (fBm + ATEF-Filter), auf den
    gewuenschten Reliefanteil skaliert.

    p_relief_anteil ist der Regler, der Norwegen von den Alpen trennt: klein
    ergibt eine glatte Hochflaeche zwischen den Taelern, gross ein Bergland.
    """
    from managers.data_lod_manager import DataLODManager
    from gui.config.value_default import TERRAIN
    from core.terrain_generator import BaseTerrainGenerator

    L = LANDSCHAFTEN[name]
    manager = DataLODManager()
    manager.set_map_distance_km(AUSSCHNITT_KM)

    parameters = {
        "map_size": size, "map_distance_km": AUSSCHNITT_KM, "map_seed": SEED,
        "amplitude": L["gipfel_m"],
        "feature_size_m": L["p_feature_m"],
        "octaves": 2,
        "persistence": TERRAIN.PERSISTENCE["default"],
        "lacunarity": TERRAIN.LACUNARITY["default"],
        "redistribute_power": 1.0,
        "erosion_filter_strength": L["p_strength"],
        "erosion_filter_gully_size_m": max(L["punktabstand_m"] * 0.45, 200.0),
        "erosion_filter_detail": 1.5,
        "erosion_filter_gully_weight": 0.5,
        "erosion_filter_ridge_rounding": L["p_ridge_rounding"],
        "erosion_filter_crease_rounding": 0.0,
        "erosion_filter_octaves": 5,
    }
    lod = int(round(np.log2(max(size, 32) / 32.0))) + 1
    generator = BaseTerrainGenerator(data_lod_manager=manager)
    generator.set_active_parameters(parameters)
    for node in ("terrain.noise", "terrain.redistribution"):
        manager.set_calculator_target_lod(node, lod)
    generator._calc_noise("terrain.noise", lod)
    generator._calc_redistribution("terrain.redistribution", lod)
    P = manager.get_calculator_output(
        "terrain.redistribution", "heightmap", lod).astype(np.float64)
    assert P.shape == (size, size), "angefragt %d px, bekommen %s" % (size, P.shape)

    # Auf den gewuenschten Reliefanteil stauchen und so anheben, dass der
    # hoechste Punkt bei gipfel_m bleibt.
    anteil = L["p_relief_anteil"]
    P = P * anteil + L["gipfel_m"] * (1.0 - anteil)
    return P


def baue(name, size):
    L = LANDSCHAFTEN[name]
    mpp = AUSSCHNITT_KM * 1000.0 / size
    rng = np.random.default_rng(SEED)

    P = baue_P(name, size)

    min_abstand_px = max(L["punktabstand_m"] / mpp, 3.0)
    punkte = poisson_punkte(size, min_abstand_px, rng)
    eltern, reihenfolge, auslass, _ = spannbaum(
        punkte, P, size, L["kosten_staerke"])
    ordnung = strahler(eltern, reihenfolge)

    p_param = {
        "einschnitt_m": L["einschnitt_m"],
        "gipfel_m": L["gipfel_m"],
        # Nur zur Herstellung strenger Monotonie - siehe flusshoehen().
        "mindestanstieg_anteil": 0.002,
    }
    z_fluss, einschnitt_knoten = flusshoehen(
        punkte, eltern, reihenfolge, ordnung, P, size, p_param, mpp)

    # --- Netz rasterisieren: Linien zwischen Kind und Elternknoten ---
    maske = np.zeros((size, size), dtype=bool)
    z_raster = np.full((size, size), np.inf)
    ordnung_raster = np.zeros((size, size), dtype=np.int32)

    for i in reihenfolge:
        e = eltern[i]
        if e < 0:
            continue
        schritte = int(max(np.linalg.norm(punkte[i] - punkte[e]) * 2.0, 2))
        t = np.linspace(0.0, 1.0, schritte)
        ys = np.clip(np.round(punkte[e][0] + t * (punkte[i][0] - punkte[e][0])
                              ).astype(int), 0, size - 1)
        xs = np.clip(np.round(punkte[e][1] + t * (punkte[i][1] - punkte[e][1])
                              ).astype(int), 0, size - 1)
        zs = z_fluss[e] + t * (z_fluss[i] - z_fluss[e])
        # Tiefster Wert gewinnt - ein Zusammenfluss darf keine Schwelle im
        # Flussbett erzeugen.
        besser = zs < z_raster[ys, xs]
        z_raster[ys[besser], xs[besser]] = zs[besser]
        ordnung_raster[ys, xs] = np.maximum(ordnung_raster[ys, xs], ordnung[i])
        maske[ys, xs] = True

    # AUSLAUF BIS ZUM KARTENRAND.
    #
    # Der Auslassknoten ist ein Poisson-Punkt in Randnaehe, aber nicht AUF dem
    # Rand. Ohne diesen Anschluss ist er der tiefste Punkt der Karte und das
    # gesamte Netz endet in einer Grube wenige Pixel vor dem Ziel.
    #
    # Gemessen im ersten Durchgang, und es war nicht das, was ich vermutet
    # hatte: nur 0.1% der Zellen waren Senken, der Abfluss lag trotzdem bei
    # 1.0%. Fast alles floss also zusammen - nur nicht von der Karte herunter.
    # Die Senkenzahl allein haette diesen Fehler nie gezeigt; erst der
    # Entwaesserungsanteil aus §7 macht ihn sichtbar.
    ay, ax = punkte[auslass]
    ziel = min(((0.0, ax), (size - 1.0, ax), (ay, 0.0), (ay, size - 1.0)),
               key=lambda q: (q[0] - ay) ** 2 + (q[1] - ax) ** 2)
    schritte = int(max(np.hypot(ziel[0] - ay, ziel[1] - ax) * 2.0, 2))
    t_lauf = np.linspace(0.0, 1.0, schritte)
    ys = np.clip(np.round(ay + t_lauf * (ziel[0] - ay)).astype(int), 0, size - 1)
    xs = np.clip(np.round(ax + t_lauf * (ziel[1] - ax)).astype(int), 0, size - 1)
    # Weiter fallend, damit der Randpixel der tiefste Punkt der Karte ist.
    z_aus = z_fluss[auslass] - t_lauf * (0.01 * L["gipfel_m"] + 1.0)
    besser = z_aus < z_raster[ys, xs]
    z_raster[ys[besser], xs[besser]] = z_aus[besser]
    ordnung_raster[ys, xs] = np.maximum(ordnung_raster[ys, xs],
                                        ordnung[auslass])
    maske[ys, xs] = True

    abstand, index = ndimage.distance_transform_edt(
        ~maske, sampling=(mpp, mpp), return_indices=True)

    # P SENKENFREI MACHEN. Auf der Hochflaeche gilt z = P, also erbt sie jede
    # geschlossene Delle des Rauschens - gemessen als 1-17% Entwaesserung im
    # ersten Durchgang. Zwei Schritte:
    #   1. Auffuellen bis zum Ueberlauf (Priority-Flood ueber die
    #      Grauwert-Rekonstruktion, wie in tools.drainage_lab.fill_requirement)
    #   2. eine winzige Neigung ZUM NETZ HIN addieren, damit die aufgefuellten
    #      Flaechen nicht eben bleiben. Eben ist fuer D8 dasselbe wie eine
    #      Senke - genau diese Verwechslung steckte hinter den "62% Senken" in
    #      §8, die in Wahrheit Plateaus waren.
    from skimage.morphology import reconstruction
    saat = P.max() * np.ones_like(P)
    saat[0], saat[-1], saat[:, 0], saat[:, -1] = P[0], P[-1], P[:, 0], P[:, -1]
    P = reconstruction(saat, P, method="erosion")
    P = P + (0.004 * L["gipfel_m"]) * (abstand / max(float(abstand.max()), 1e-9))
    z_nah = z_raster[index[0], index[1]]
    ordnung_nah = ordnung_raster[index[0], index[1]]

    # Talbreite als Anteil des PUNKTABSTANDS - die Wasserscheide liegt
    # zwangslaeufig etwa in der Mitte zwischen zwei Laeufen (§8, Fehlertyp
    # "absolute Groesse, wo eine relative hingehoert").
    breite = (L["tal_breite_anteil"] * L["punktabstand_m"]
              * np.power(np.maximum(ordnung_nah, 1), L["tal_breite_exponent"]))
    d_norm = np.clip(abstand / breite, 0.0, 1.0)
    profil = talformer(d_norm, L["talform_exponent"], L.get("stufen", 0))

    # DIE BLEND-FORMEL
    z = P - (P - z_nah) * (1.0 - profil)

    teile = {
        "P": P, "punkte": punkte, "eltern": eltern, "ordnung": ordnung,
        "maske": maske, "d_norm": d_norm, "profil": profil,
        "auslass": auslass, "z_fluss": z_fluss, "mpp": mpp,
        "einschnitt": P - z_nah,
    }
    return z, mpp, teile


# =============================================================================
# BILDER UND KENNZAHLEN
# =============================================================================

def _hillshade(z, mpp, azimut=315.0, hoehe=42.0):
    dy, dx = np.gradient(z, mpp)
    neigung = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspekt = np.arctan2(-dx, dy)
    az, hh = np.deg2rad(360.0 - azimut + 90.0), np.deg2rad(hoehe)
    return np.clip(np.sin(hh) * np.sin(neigung)
                   + np.cos(hh) * np.cos(neigung) * np.cos(az - aspekt), 0, 1)


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 384
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tools.drainage_lab as dl

    print("Fuenf Landschaften, %.0f x %.0f km, %d px (%.0f m/px)\n"
          % (AUSSCHNITT_KM, AUSSCHNITT_KM, size, AUSSCHNITT_KM * 1000.0 / size))
    print("%-17s %7s %7s %8s %7s %7s %7s %6s"
          % ("Landschaft", "Knoten", "Ordn.", "Abfluss", "Senken", "Netz",
             "Relief", "Zeit"))
    print("-" * 78)

    fig, ax = plt.subplots(len(LANDSCHAFTEN), 3,
                           figsize=(16, 5.0 * len(LANDSCHAFTEN)), squeeze=False)
    for i, name in enumerate(LANDSCHAFTEN):
        L = LANDSCHAFTEN[name]
        start = time.time()
        z, mpp, t = baue(name, size)
        dauer = time.time() - start
        w = dl.bewerte(z, mpp)
        print("%-17s %7d %7d %7.1f%% %7d %7d %7.0f %5.1fs"
              % (name, len(t["punkte"]), t["ordnung"].max(),
                 100 * w["abfluss_anteil"], w["senken"], w["netz"],
                 w["relief"], dauer))

        b = ax[i, 0].imshow(z, cmap="terrain")
        ax[i, 0].imshow(_hillshade(z, mpp), cmap="gray", alpha=0.35)
        ax[i, 0].set_title("%s   %.0f - %.0f m" % (name, z.min(), z.max()),
                           fontsize=10)
        fig.colorbar(b, ax=ax[i, 0], shrink=0.8)

        ax[i, 1].imshow(_hillshade(z, mpp), cmap="gray")
        ax[i, 1].set_title("%s\nTalabstand %.0f m, Einschnitt %.0f m, "
                           "P-Relief %.0f%%"
                           % (L["bemerkung"], L["punktabstand_m"],
                              L["einschnitt_m"], 100 * L["p_relief_anteil"]),
                           fontsize=8)

        ordnung = t["ordnung"]
        ax[i, 2].imshow(t["P"], cmap="Greys_r", alpha=0.5)
        for o in range(1, int(ordnung.max()) + 1):
            m = ordnung == o
            ax[i, 2].scatter(t["punkte"][m, 1], t["punkte"][m, 0],
                             s=0.4 * o * o + 0.5,
                             c=[plt.cm.viridis(o / max(ordnung.max(), 1))],
                             linewidths=0)
        ax[i, 2].set_xlim(0, size); ax[i, 2].set_ylim(size, 0)
        ax[i, 2].set_title("Netz: %d Knoten, Ordnung bis %d"
                           % (len(t["punkte"]), ordnung.max()), fontsize=9)
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])

    fig.suptitle("Flussnetz + Hochebene: z = P - (P - z_Fluss) * (1 - Profil)",
                 fontsize=12)
    fig.tight_layout()
    ziel = os.path.join(OUTPUT_DIR, "landschaften_%dpx.png" % size)
    fig.savefig(ziel, dpi=95)
    plt.close(fig)
    print("\nBild -> %s" % ziel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
