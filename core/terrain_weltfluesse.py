"""
Path: core/terrain_weltfluesse.py

STUFE B - das Flussnetz der Regionenwelt (Kern, seit 2026-08-05).

Drei RECHENSTUFEN auf EINER Karte, nicht drei Zoomfenster. Der Nutzer dazu:
"am ende sind mikro etc ja nicht mehr notwendig, weil wir alles einmal in
hoechster qualitaet berechnen. hoechstens sind makro zu mikro die stufen in
denen die rechnung ablaeuft. also damit wir erst die grossen stroeme und am
ende die baeche generieren."

Damit entfaellt die gesamte Fensterlogik der Flussnetz-Werkstatt
(erbe_bilden, Ein- und Auslauf zwischen Stufen, Randsaum). Was bleibt, ist der
Kern, der dort die abgerissenen Laeufe behoben hat:

    * GESCHACHTELTER PUNKTSATZ. Ein Makroknoten ist auch auf der Mikrostufe
      derselbe Knoten an derselben Stelle.
    * GEERBTE KETTEN WERDEN ERZWUNGEN. Billig genuegt nicht - Dijkstra umgeht
      eine billige Kette, sobald ein kurzer Weg daneben insgesamt weniger
      kostet, und der Strom reisst mittendrin ab.
    * HOECHSTENS EIN KETTENELTERNKNOTEN je Punkt. Trifft eine Kette auf eine
      bestehende, MUENDET sie dort statt daneben herzulaufen.


FLUESSE LAUFEN BIS -50 m. Vorgabe des Nutzers: "fluesse sollten am besten bis
zB -50 m wasserhoehe fliessen. damit fliessen die fluesse deutlich ins meer
hinein und werden danach an der wasserkante abgeschnitten." Die Auslaesse
liegen also UNTER Wasser; gezeichnet und eingegraben wird nur oberhalb von 0.
Das ersetzt den Auslauf-Behelf der Werkstatt (gerade Strecke zum naechsten
Meerpixel) ersatzlos.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import core.terrain_weltkarte as rw

# Knotenabstand je Rechenstufe, in Metern. Makro macht die Stroeme, Mikro die
# Baeche. Der Sprung ist jeweils rund Faktor 3 - genug, dass eine Stufe etwas
# Neues beitraegt, wenig genug, dass die Ketten sich sauber einordnen.
STUFEN = (("Makro", 1200.0), ("Meso", 420.0), ("Mikro", 150.0))
STUFEN_FARBE = ("#e03030", "#25a03a", "#e8c020")

MUENDUNGSTIEFE_M = -50.0   # bis hierhin laufen die Laeufe weiter
ERBE_KOSTEN = 0.12         # geerbte Kette kostet so viel wie sonst

# Untergrenze fuer das Wassergewicht eines Knotens (Block 1.1). Verhindert,
# dass eine sehr trockene Region Knoten mit Gewicht ~0 bekommt und ihre
# Laeufe dadurch voellig verschwinden - auch in der Steppe fliesst etwas.
WASSER_MINDEST = 0.15

# Die drei Stufenabstaende haengen fest aneinander: Meso ist rund ein Drittel
# des Makroabstands, Mikro ein Achtel. Aus 1200 werden so 420 und 150 - genau
# die Werte, mit denen das Netz eingemessen wurde.
#
# WARUM EIN REGLER UND NICHT DREI: die Verhaeltnisse sind das, was die
# Schachtelung traegt. Waeren sie einzeln einstellbar, koennte man Meso groeber
# als Makro stellen, und die Vererbung der Laeufe haette keinen Sinn mehr.
STUFEN_VERHAELTNIS = (1.0, 1.0 / 2.857, 1.0 / 8.0)

# Mittlere Formgroesse der Regionentabelle, rund 2260 m. Sie dient als BEZUG
# fuer die Talbreite: `formgroesse_m / BEZUGSFORM_M` liegt damit zwischen 0.5
# (Griechische Inseln) und 2.2 (Alpenland) und moduliert nur noch, statt die
# Groessenordnung zu setzen. Siehe `taeler_eingraben`.
BEZUGSFORM_M = float(np.mean([r["formgroesse_m"]
                              for _z, _s, r in rw.alle_regionen()]))


def stufenabstaende(abstand_makro_m=None):
    """Die drei Knotenabstaende in Metern, aus dem Makroabstand abgeleitet."""
    if abstand_makro_m is None:
        return [wert for _name, wert in STUFEN]
    return [float(abstand_makro_m) * f for f in STUFEN_VERHAELTNIS]


# =============================================================================
# PUNKTSATZ
# =============================================================================

def geschachtelte_punkte(H, seed, abstand_makro_m=None,
                         muendungstiefe_m=MUENDUNGSTIEFE_M):
    """
    Punkte auf Land (und im Flachwasser bis zur Muendungstiefe), mit Generation.

    Erzeugt wird EIN Poisson-Satz im feinsten Abstand; die groberen Stufen sind
    Teilmengen davon. Damit ist ein Makroknoten auf jeder feineren Stufe
    derselbe Knoten - genau die Schachtelung, ohne die ein grober Lauf sich
    nicht erhalten laesst.
    """
    import core.terrain_river_network as rn

    size = H.shape[0]
    mpp = rw.WELT_KM * 1000.0 / size
    extent_m = rw.WELT_KM * 1000.0
    abstaende = stufenabstaende(abstand_makro_m)
    fein = abstaende[-1]

    # KEIN KNOTEN FEINER ALS VIER PIXEL.
    #
    # Der Knotenabstand steht in Metern, damit die Landschaft nicht an der
    # Pixelzahl haengt (SPEZIFIKATION §10). Unterhalb von etwa vier Pixeln
    # traegt ein weiterer Knoten aber keine Information mehr - das Gelaende
    # zwischen ihm und dem Nachbarn hat gar keine Stuetzstellen. Gemessen bei
    # 256 px: 150 m Abstand sind dort 1.8 Pixel, es entstanden 8575 Knoten fuer
    # 65536 Bildpunkte, und das Netz brauchte 64 Sekunden - laenger als bei
    # 512 px. Die Grenze macht die grobe Stufe wieder schnell, ohne die feine
    # zu beschneiden.
    fein = max(fein, 4.0 * mpp)

    punkte = rn.poisson_points(extent_m, fein, seed) / mpp
    yi = np.clip(np.round(punkte[:, 0]).astype(int), 0, size - 1)
    xi = np.clip(np.round(punkte[:, 1]).astype(int), 0, size - 1)
    brauchbar = H[yi, xi] > muendungstiefe_m
    punkte = punkte[brauchbar]

    # GENERATION DURCH ANHEFTEN, NICHT DURCH GIERIGES AUSDUENNEN.
    #
    # Zuerst wurde je Stufe eine Teilmenge gierig ausgewaehlt und dabei nach
    # JEDEM angenommenen Punkt der k-d-Baum neu gebaut - quadratischer Aufwand,
    # und der groesste Posten der 64 Sekunden.
    #
    # Jetzt wird je Stufe ein eigener Poisson-Satz im groberen Abstand erzeugt
    # und jeder seiner Punkte an den naechstgelegenen FEINEN Punkt geheftet.
    # Die Schachtelung bleibt exakt erhalten - ein Makroknoten IST ein feiner
    # Knoten - und der Aufwand ist n log n statt n^2.
    stufe = np.full(len(punkte), len(abstaende) - 1, dtype=np.int64)
    if len(punkte) == 0:
        return punkte, stufe
    baum = cKDTree(punkte)
    for index, abstand_m in enumerate(abstaende[:-1]):
        grob = rn.poisson_points(extent_m, max(abstand_m, fein), seed + index) / mpp
        if len(grob) == 0:
            continue
        _, nahe = baum.query(grob)
        nahe = np.unique(nahe)
        # Nur nach GROB umwidmen, nie zurueck: eine Stufe darf einen bereits
        # groberen Knoten nicht wieder verfeinern.
        stufe[nahe] = np.minimum(stufe[nahe], index)
    return punkte, stufe


# =============================================================================
# EIN BAUM JE RECHENSTUFE
# =============================================================================

def _kanten_und_kosten(punkte, H, mpp, felder_hang, kosten_staerke,
                       muendungstiefe_m=MUENDUNGSTIEFE_M):
    """Delaunay, Kanten ueber tiefem Wasser raus, gerichtete Kosten."""
    size = H.shape[0]
    tri = Delaunay(punkte)
    menge = set()
    for simplex in tri.simplices:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            menge.add((min(simplex[a], simplex[b]), max(simplex[a], simplex[b])))
    kanten = np.array(sorted(menge))

    # Kanten, die ueber TIEFES Wasser laufen, verwerfen. Flachwasser bis zur
    # Muendungstiefe bleibt erlaubt - dort liegen ja die Auslaesse.
    tief = np.zeros(len(kanten), dtype=bool)
    for t in np.linspace(0.0, 1.0, 7)[1:-1]:
        m = punkte[kanten[:, 0]] * (1.0 - t) + punkte[kanten[:, 1]] * t
        my = np.clip(np.round(m[:, 0]).astype(int), 0, size - 1)
        mx = np.clip(np.round(m[:, 1]).astype(int), 0, size - 1)
        tief |= H[my, mx] <= muendungstiefe_m

    # PUNKTE, DIE DADURCH JEDE KANTE VERLIEREN, BLEIBEN EINE BRUECKE.
    #
    # Ein Punkt auf einer kleinen Insel kann so liegen, dass ALLE seine
    # Delaunay-Kanten ueber tiefes Wasser fuehren - dann wird er restlos aus
    # dem Graphen entfernt und ist fuer Dijkstra unerreichbar, obwohl er selbst
    # ueber Wasser liegt. Gemessen bei 256 px: in 2 von 8 Laeufen genau ein
    # solcher Knoten, mitten an Land ohne jede Verbindung.
    #
    # Der Rest des Netzes soll nicht schwimmen lernen: es bleibt bei je einer
    # KUERZESTEN Verbindung zum Festland-Graphen, nicht bei allen. Das reicht,
    # um den Punkt erreichbar zu machen, ohne das Wasser als generelle
    # Abkuerzung zu oeffnen.
    bleibt = ~tief
    isoliert = np.ones(len(punkte), dtype=bool)
    isoliert[kanten[bleibt, 0]] = False
    isoliert[kanten[bleibt, 1]] = False
    for i in np.flatnonzero(isoliert):
        kandidaten = np.flatnonzero((kanten[:, 0] == i) | (kanten[:, 1] == i))
        if len(kandidaten) == 0:
            continue
        laengen = np.linalg.norm(
            punkte[kanten[kandidaten, 0]] - punkte[kanten[kandidaten, 1]], axis=1)
        bleibt[kandidaten[int(np.argmin(laengen))]] = True

    kanten = kanten[bleibt]

    yi = np.clip(np.round(punkte[:, 0]).astype(int), 0, size - 1)
    xi = np.clip(np.round(punkte[:, 1]).astype(int), 0, size - 1)
    hoehe = H[yi, xi].astype(np.float64)
    spanne = float(hoehe.max() - hoehe.min()) or 1.0
    h_norm = (hoehe - hoehe.min()) / spanne

    laenge = np.linalg.norm(punkte[kanten[:, 0]] - punkte[kanten[:, 1]], axis=1)
    delta = h_norm[kanten[:, 1]] - h_norm[kanten[:, 0]]
    bezug = float(np.median(np.abs(delta) / np.maximum(laenge, 1e-9))) or 1e-6

    def kosten(anstieg):
        """Dijkstra laeuft vom Auslass nach AUSSEN, also flussaufwaerts -
        bergauf ist teuer, auf gleicher Hoehe entlang billig."""
        return laenge * (1.0 + kosten_staerke * np.square(
            np.maximum(anstieg, 0.0) / np.maximum(laenge, 1e-9) / bezug))

    return kanten, kosten(delta), kosten(-delta), laenge


def baue_stufe(punkte, H, mpp, kosten_staerke, erbe=None, stufe_index=0,
               erbe_kosten=ERBE_KOSTEN, muendungstiefe_m=MUENDUNGSTIEFE_M,
               wasser_feld=None):
    """
    Ein Spannbaum ueber `punkte`. `erbe` = (eltern, stufe) der vorigen Stufe,
    auf DIESELBEN Indizes bezogen (die Punkte sind geschachtelt).
    """
    size = H.shape[0]
    n = len(punkte)
    kanten, kosten_hin, kosten_rueck, laenge = _kanten_und_kosten(
        punkte, H, mpp, None, kosten_staerke, muendungstiefe_m)
    if len(kanten) < 4:
        return None

    kante_nr = {}
    for nr, (a, b) in enumerate(kanten):
        kante_nr[(int(a), int(b))] = nr
        kante_nr[(int(b), int(a))] = nr
    kanten_stufe = np.full(len(kanten), stufe_index, dtype=np.int64)
    fest = set()
    ketten_eltern = -np.ones(n, dtype=np.int64)

    # ---------------------------------------------------- geerbte Ketten
    if erbe is not None:
        erbe_eltern, erbe_stufe = erbe
        vor = csr_matrix(
            (np.concatenate([kosten_hin, kosten_rueck]),
             (np.concatenate([kanten[:, 0], kanten[:, 1]]),
              np.concatenate([kanten[:, 1], kanten[:, 0]]))), shape=(n, n))
        quellen = [i for i in range(n) if erbe_eltern[i] >= 0]
        # Grobe Generation zuerst - sonst muendet der Strom in den Nebenfluss
        # statt umgekehrt.
        quellen.sort(key=lambda i: (erbe_stufe[i], i))
        if quellen:
            _, vorg = dijkstra(vor, indices=quellen, return_predecessors=True)
            for reihe, i in enumerate(quellen):
                ziel = int(erbe_eltern[i])
                pfad, k, schutz = [], ziel, 0
                while k != i and k >= 0 and schutz < 4 * n:
                    pfad.append(k)
                    k, schutz = int(vorg[reihe, k]), schutz + 1
                if k != i:
                    continue
                pfad.append(i)
                pfad.reverse()
                for oben, unten in zip(pfad[:-1], pfad[1:]):
                    if ketten_eltern[oben] >= 0:
                        break              # muendet in eine bestehende Kette
                    ketten_eltern[oben] = unten
                    nr = kante_nr.get((oben, unten))
                    if nr is not None:
                        kosten_hin[nr] *= erbe_kosten
                        kosten_rueck[nr] *= erbe_kosten
                        kanten_stufe[nr] = min(kanten_stufe[nr],
                                               int(erbe_stufe[i]))
                        fest.add((oben, unten))
                        fest.add((unten, oben))

    # ---------------------------------------------------- Auslaesse
    #
    # Jeder Knoten unterhalb der Muendungstiefe ist Auslass. Kein Auslasswinkel
    # aus dem Seed, kein Randabfluss-Preis - das Meer ist der Auslass.
    yi = np.clip(np.round(punkte[:, 0]).astype(int), 0, size - 1)
    xi = np.clip(np.round(punkte[:, 1]).astype(int), 0, size - 1)
    unter_wasser = np.flatnonzero(H[yi, xi] <= 0.0)
    if len(unter_wasser) == 0:
        unter_wasser = np.argsort(H[yi, xi])[:max(len(punkte) // 200, 1)]

    quellen_v = np.concatenate([kanten[:, 0], kanten[:, 1],
                                np.full(len(unter_wasser), n)])
    ziele_v = np.concatenate([kanten[:, 1], kanten[:, 0], unter_wasser])
    werte_v = np.concatenate([kosten_hin, kosten_rueck,
                              np.full(len(unter_wasser), 1e-9)])
    matrix = csr_matrix((werte_v, (quellen_v, ziele_v)), shape=(n + 1, n + 1))

    erg = dijkstra(matrix, indices=n, return_predecessors=True)
    entfernung = erg[0][:n]
    eltern = erg[1][:n].astype(np.int64)
    eltern[eltern == n] = -1

    # VOM MEER ABGESCHNITTENE KOMPONENTEN AN DEN NAECHSTEN ERREICHTEN KNOTEN
    # BRUECKEN (2026-08-13, Nutzerbefund "so schwer ist das flusssystem ja
    # nicht - wo sind verbesserungen moeglich"). scipy markiert einen von der
    # Ueberquelle aus NIE erreichten Knoten mit dem Sentinel -9999 in `eltern`
    # UND mit `entfernung == inf` - beides wird bisher nirgends abgefragt.
    # `eltern[eltern==n]=-1` faengt nur den echten Meeresausgang ab, -9999
    # rutschte durch und wurde ueberall, wo nur `eltern<0` geprueft wird
    # (Kettenerzwingung, `netz["auslaesse"]`, `taeler_eingraben`), als
    # gueltiger Ausgang gezaehlt - ein Knoten mitten im Land erschien als
    # Fluss-Muendung. Ursache der Trennung: `_kanten_und_kosten()`s
    # Bruecken-Reparatur haelt fuer einen isolierten Punkt nur EINE, die
    # kuerzeste Kante - garantiert nicht, dass die am ANDEREN Ende auch zurueck
    # zum Festland fuehrt. Zwei einander nur GEGENSEITIG haltende Punkte bilden
    # so eine eigene, vom Meer getrennte Mini-Komponente. Nachgemessen
    # (384px, Seed 20260804): exakt 5 Knoten betroffen, Hoehen 63-306 m - deckt
    # sich exakt mit den von `smoke_test_river_reaches_sea.py` gemeldeten 5
    # Sackgassen.
    #
    # Selbes Bruecken-Prinzip wie dort, nur auf GANZE abgeschnittene
    # Komponenten angewendet statt auf einzelne Punkte: jeder unerreichte
    # Knoten wird an den naechstgelegenen ERREICHTEN Knoten gehaengt (echte
    # Position, nicht ueber den Tiefwasser-gefilterten Kantensatz - dieselbe
    # pragmatische Abkuerzung, die die Punkt-Bruecken-Reparatur oben schon
    # nutzt). `taeler_eingraben()` schneidet ohnehin nur oberhalb von 0 m
    # (`np.where(H > 0.0, neu, H)`), eine ueber Wasser fuehrende Bruecke bleibt
    # also unsichtbar, stellt aber die Kette bis zum Meer her.
    unerreicht = np.flatnonzero(~np.isfinite(entfernung))
    if len(unerreicht) > 0:
        erreicht = np.flatnonzero(np.isfinite(entfernung))
        if len(erreicht) > 0:
            baum_erreicht = cKDTree(punkte[erreicht])
            _, naechste = baum_erreicht.query(punkte[unerreicht])
            eltern[unerreicht] = erreicht[naechste]
        else:
            # Kein einziger Knoten erreicht das Meer (praktisch nie, da
            # `unter_wasser` immer mindestens einen Auslass stellt) - dann
            # bleibt nur die ehrliche Kennzeichnung als eigener Ausgang.
            eltern[unerreicht] = -1

    # ---------------------------------------------------- Ketten erzwingen
    #
    # HIER ENTSTANDEN DIE RINGE (Nutzermeldung 2026-08-10: "die fliessen oft
    # nicht zum meer").
    #
    # Nach Dijkstra ist `eltern` ein sauberer Baum: die Ueberquelle haengt an
    # allen Knoten unter Wasser, jede Elternkette endet deshalb im Meer - die
    # gleiche Garantie durch Konstruktion, die redblobgames' Flusswachstum aus
    # dem Vorgehen "von aussen nach innen" zieht.
    #
    # Dieser Block schrieb danach Eltern aus der VORIGEN Stufe zurueck. Die
    # stammen aus einem anderen Baum, und die Mischung aus beiden ist keiner
    # mehr: zeigt A neu auf B, waehrend B ueber den alten Baum noch auf A
    # zeigt, laeuft die Kette im Kreis. Gemessen ueber drei Seeds bei 384 px
    # hingen 29 bis 47 Prozent ALLER Knoten in einem solchen Ring; nur 53 bis
    # 71 Prozent erreichten das Meer. Auffallen konnte es nicht: die
    # Tiefenschleife darunter bricht bei `d < n` ab und meldet nichts.
    #
    # Die Absicht bleibt - ein geerbter Strom soll seinen Lauf behalten -, aber
    # eine Zuweisung wird nur noch uebernommen, wenn sie keinen Ring schliesst.
    # Dafuer wird vom vorgesehenen Elternteil aus aufwaerts gegangen: taucht
    # `oben` dabei auf, waere es einer, und die Zuweisung entfaellt. Der
    # Dijkstra-Elternteil bleibt dann stehen, und der ist per Bau meerwaerts.
    #
    # Zusaetzlich sind die geerbten Kanten schon ueber `erbe_kosten` verbilligt
    # (Faktor 0.12) - Dijkstra bevorzugt sie also ohnehin. Das Erzwingen ist
    # nur noch der Nachdruck, nicht die einzige Wirkung.
    aus = set(np.flatnonzero(eltern < 0).tolist())
    for oben in np.flatnonzero(ketten_eltern >= 0):
        oben = int(oben)
        if oben in aus:
            continue
        neuer = int(ketten_eltern[oben])
        k, schutz, ring = neuer, 0, False
        while k >= 0 and schutz <= n:
            if k == oben:
                ring = True
                break
            k, schutz = int(eltern[k]), schutz + 1
        if not ring:
            eltern[oben] = neuer

    tiefe = np.zeros(n, dtype=np.int64)
    for i in range(n):
        k, d = i, 0
        while eltern[k] >= 0 and d < n:
            k, d = eltern[k], d + 1
        tiefe[i] = d
    reihenfolge = np.argsort(tiefe, kind="stable")

    # WASSERMENGE STATT KNOTENZAHL (Block 1.1, docs/FLUESSE_UND_WASSER.md).
    #
    # Hier stand `flaeche = np.ones(n)`: jeder Knoten trug 1 bei, egal ob
    # dort 471 mm oder 1967 mm im Jahr fallen. Was das Modell
    # "Einzugsgebiet" nennt, war damit die FLAECHE in Knoten - nicht die
    # Wassermenge. Der Niederschlag lag als volles Feld bereit
    # (`felder["niederschlag_mm"]`) und wurde nie gelesen.
    #
    # GEMESSEN, was das anrichtete (512 px, Seed 20260804): das Fjordland
    # hat mit 1967 mm den hoechsten Niederschlag aller Regionen und die
    # meiste Wassermenge (Niederschlag mal Landflaeche, 13.0 Mio) - und
    # mit 123 Knoten den KLEINSTEN Hauptfluss. Die Steppe hat mit 4.3 Mio
    # die wenigste Wassermenge und einen dreimal groesseren Fluss. Die
    # Korrelation war negativ.
    #
    # `wasser_feld` ist der Niederschlag, auf den Kartenmittelwert
    # bezogen: 1.0 heisst durchschnittlich, das Fjordland liegt bei rund
    # 2.2, die Steppe bei 0.5. Ein Knoten traegt damit sein Wasser bei,
    # nicht seine Existenz.
    #
    # OHNE FELD bleibt es bei `ones` - Werkzeuge und Tests, die
    # `flussnetz()` ohne Regionsdaten aufrufen, sollen unveraendert
    # weiterlaufen. Das ist KEIN stiller Rueckfall: die Wirkung ist
    # dieselbe wie vor dem Umbau, und wer Regionsdaten hat, gibt sie mit.
    if wasser_feld is not None:
        yq = np.clip(np.round(punkte[:, 0]).astype(np.int64),
                     0, wasser_feld.shape[0] - 1)
        xq = np.clip(np.round(punkte[:, 1]).astype(np.int64),
                     0, wasser_feld.shape[1] - 1)
        flaeche = np.maximum(wasser_feld[yq, xq].astype(np.float64),
                             WASSER_MINDEST)
    else:
        flaeche = np.ones(n)
    for i in reihenfolge[::-1]:
        if eltern[i] >= 0:
            flaeche[eltern[i]] += flaeche[i]

    kante_stufe_von = {}
    for nr, (a, b) in enumerate(kanten):
        kante_stufe_von[(int(a), int(b))] = kanten_stufe[nr]
        kante_stufe_von[(int(b), int(a))] = kanten_stufe[nr]
    lauf_stufe = np.full(n, stufe_index, dtype=np.int64)
    for i in range(n):
        if eltern[i] >= 0:
            lauf_stufe[i] = kante_stufe_von.get((int(i), int(eltern[i])),
                                                stufe_index)
    # Flussabwaerts darf die Generation nie feiner werden.
    for i in reihenfolge[::-1]:
        if eltern[i] >= 0:
            lauf_stufe[eltern[i]] = min(lauf_stufe[eltern[i]], lauf_stufe[i])

    return dict(punkte=punkte, eltern=eltern, flaeche=flaeche,
                lauf_stufe=lauf_stufe, reihenfolge=reihenfolge,
                auslaesse=np.flatnonzero(eltern < 0).tolist(),
                entfernung=entfernung)


def flussnetz(H, seed, kosten_staerke=6.0, abstand_makro_m=None,
              muendungstiefe_m=MUENDUNGSTIEFE_M, erbe_kosten=ERBE_KOSTEN,
              niederschlag_mm=None, region_map=None):
    """
    Die drei Rechenstufen nacheinander. Rueckgabe: das Netz der letzten.

    Alle Stellschrauben sind seit dem 2026-08-06 Parameter statt Konstanten -
    vorher nahm diese Funktion ueberhaupt keine entgegen, und die neun
    `river_*`-Regler der Oberflaeche bewegten deshalb nichts (gemessen: alle
    neun 0.00 m Hoehenaenderung, 0 Flusspixel anders).

    Die Vorgaben sind genau die frueheren Konstanten, damit ein Aufruf ohne
    Argumente dieselbe Welt liefert wie zuvor.
    """
    size = H.shape[0]
    mpp = rw.WELT_KM * 1000.0 / size
    abstaende = stufenabstaende(abstand_makro_m)
    # WASSERGEWICHT JE PIXEL (Block 1.1, docs/FLUESSE_UND_WASSER.md).
    #
    # Der Niederschlag, bezogen auf den Mittelwert ueber LAND - nicht ueber
    # die ganze Karte. Ueber See faellt zwar auch Regen, aber er speist
    # keinen Fluss; naehme man ihn in den Mittelwert, haenge das Gewicht
    # am Wasseranteil der Karte statt an der Region.
    wasser_feld = None
    if niederschlag_mm is not None:
        n_feld = np.asarray(niederschlag_mm, dtype=np.float64)
        land = H > 0.0
        mittel = float(np.mean(n_feld[land])) if land.any() else 0.0
        if mittel > 1e-9:
            wasser_feld = n_feld / mittel

    punkte, stufe = geschachtelte_punkte(H, seed, abstand_makro_m,
                                         muendungstiefe_m)

    netz, erbe = None, None
    for index in range(len(abstaende)):
        gehoert = stufe <= index
        auswahl = np.flatnonzero(gehoert)
        rueck = -np.ones(len(punkte), dtype=np.int64)
        rueck[auswahl] = np.arange(len(auswahl))

        teil_erbe = None
        if netz is not None:
            # Eltern der vorigen Stufe auf die neue Indizierung umschreiben.
            alt_eltern = netz["eltern"]
            alt_index = netz["auswahl"]
            e = -np.ones(len(auswahl), dtype=np.int64)
            s = np.full(len(auswahl), index, dtype=np.int64)
            for alt_i, welt_i in enumerate(alt_index):
                neu_i = rueck[welt_i]
                s[neu_i] = netz["lauf_stufe"][alt_i]
                if alt_eltern[alt_i] >= 0:
                    e[neu_i] = rueck[alt_index[alt_eltern[alt_i]]]
            teil_erbe = (e, s)

        neu = baue_stufe(punkte[auswahl], H, mpp, kosten_staerke,
                         erbe=teil_erbe, stufe_index=index,
                         erbe_kosten=erbe_kosten,
                         muendungstiefe_m=muendungstiefe_m,
                         wasser_feld=wasser_feld)
        if neu is None:
            break
        neu["auswahl"] = auswahl
        neu["stufe"] = stufe[auswahl]
        netz, erbe = neu, teil_erbe

    if netz is not None and region_map is not None:
        _hauptstrom_erzwingen(netz, region_map, H, seed)
    return netz


# Wie wahrscheinlich eine Region einen erzwungenen Hauptstrom bekommt, und
# wie gross er mindestens werden soll (Nutzervorgabe 2026-08-24):
#
#   *"dann werden grosse fluesse zumindest in Fjordland (100% chance),
#   Taiga (66%) und Atlantik (66% chance) generiert. der rest generiert
#   weiterhin wie jetzt. nur ein kleiner nudge erstmal."*
#   *"Fjordland soll es groesser sein als jetzt der groesste fluss."*
#
# WARUM ES DIESE REGEL UEBERHAUPT BRAUCHT. Block 1.1 hat den Niederschlag
# ins Netz gebracht, und die Richtung stimmt seither: nasse Regionen
# wachsen, trockene schrumpfen (Fjordland 123 -> 215, Steppe 374 -> 226).
# Das Fjordland fuehrt trotzdem nicht - 215 gegen 737 an der
# Atlantikkueste - und die Ursache ist ungeklaert. Drei Hypothesen wurden
# gemessen und widerlegt (docs/FLUESSE_UND_WASSER.md 1.1).
#
# Diese Regel ist deshalb eine bewusste SETZUNG, kein Modell: sie behebt
# nicht die unbekannte Ursache, sie ueberstimmt sie an genau den drei
# Stellen, an denen der Nutzer grosse Fluesse sehen will.
#
# Der Zielwert 700 liegt ueber den 654, die vor Block 1.1 der groesste
# Fluss der ganzen Karte waren.
HAUPTSTROM_QUOTE = {
    "Fjordland": (1.00, 700.0),
    "Taiga": (0.66, 500.0),
    "Atlantikkueste": (0.66, 500.0),
}


def _hauptstrom_erzwingen(netz, region_map, H, seed):
    """
    Je Region mit Quote: den groessten Lauf auf sein Sollmass anheben.

    NICHT DEN BAUM UMHAENGEN. Der naheliegende Weg waere, Nachbarlaeufe in
    den groessten umzuleiten - das kann aber Zyklen erzeugen und die
    muehsam gesicherte Kettenlogik zerstoeren (siehe Modulkopf: "HOECHSTENS
    EIN KETTENELTERNKNOTEN je Punkt"). Stattdessen wird nur das GEWICHT
    entlang der bestehenden Hauptkette angehoben, so als flosse dort mehr
    Wasser zu. Die Geometrie bleibt unberuehrt, nur die Talbreite und
    -tiefe folgen (`gebiet` in taeler_eingraben).
    """
    eltern, fl, pk = netz["eltern"], netz["flaeche"], netz["punkte"]
    n = len(pk)
    if n == 0:
        return
    ys = np.clip(np.round(pk[:, 0]).astype(np.int64), 0, region_map.shape[0] - 1)
    xs = np.clip(np.round(pk[:, 1]).astype(np.int64), 0, region_map.shape[1] - 1)
    rk = region_map[ys, xs]
    hk = H[ys, xs]

    namen = [r["name"] for _z, _s, r in rw.alle_regionen()]
    for i, name in enumerate(namen):
        eintrag = HAUPTSTROM_QUOTE.get(name)
        if eintrag is None:
            continue
        wahrscheinlich, ziel = eintrag

        # AUS DEM SEED, nicht aus `random`: dieselbe Karte muss dasselbe
        # Ergebnis geben, und zwar unabhaengig davon, wie viele
        # Zufallszahlen vorher gezogen wurden.
        wuerfel = np.random.default_rng(
            (int(seed) << 8) ^ (hash(name) & 0xFFFF)).random()
        if wuerfel >= wahrscheinlich:
            continue

        eigene = (rk == i) & (hk > 0.0)
        if eigene.sum() < 5:
            continue
        groesster = int(np.flatnonzero(eigene)[np.argmax(fl[eigene])])
        if fl[groesster] >= ziel:
            continue

        # NUR DER HAUPTLAUF, vom groessten Knoten ABWAERTS bis zur
        # Muendung. Zwei Fehler, beide beim ersten Anlauf gemessen:
        #
        #   * Die Zufluesse OBERHALB mitzuskalieren ist falsch - dort
        #     fliesst ja nicht mehr Wasser. Ein grosser Strom hat normale
        #     Nebenfluesse, keine ebenfalls vergroesserten.
        #   * Kette und Zufluesse als getrennte Masken multiplizierten den
        #     Hauptknoten ZWEIMAL: 215 * 3.26 * 3.26 = 2275 statt der
        #     angepeilten 700.
        #
        # Der Strom schwillt damit am Zusammenfluss an, nicht allmaehlich -
        # das ist an einem grossen Fluss auch real so, wenn ein
        # Haupteinzugsgebiet dazukommt.
        #
        # ADDITIV, NICHT MULTIPLIKATIV - dritter gemessener Fehler dieser
        # Regel. Ein Faktor haette jeden Knoten flussabwaerts im selben
        # VERHAELTNIS vergroessert, auch die, die ohnehin schon gross sind:
        # der Fjordland-Strom muendet in der Atlantikkueste, und deren
        # Maximum sprang dadurch von 737 auf 2396, das Alpenland von 676
        # auf 2197 - obwohl beide gar keine Quote haben.
        #
        # Mehr Wasser im Oberlauf heisst flussabwaerts eine KONSTANTE
        # Zugabe, kein konstantes Verhaeltnis. Der Zielknoten trifft damit
        # genau sein Sollmass, und die Abschnitte darunter wachsen um
        # denselben absoluten Betrag - so, als waere oben ein weiteres
        # Einzugsgebiet dazugekommen.
        zugabe = ziel - float(fl[groesster])
        kette = np.zeros(n, dtype=bool)
        k, schutz = groesster, 0
        while k >= 0 and schutz <= n:
            kette[k] = True
            k, schutz = int(eltern[k]), schutz + 1
        fl[kette] = fl[kette] + zugabe


# =============================================================================
# STUFE C - TAELER EINGRABEN
# =============================================================================

def taeler_eingraben(H, netz, felder, breite_faktor=0.35, tiefe_anteil=0.30,
                     form=1.3, max_hang=1.0, abstand_makro_m=None):
    """
    Aus dem Liniennetz ein Gelaende mit Taelern machen.

        z = P - (P - z_fluss) * (1 - profil(abstand / breite))

    An der Sohle steht z_fluss, weit weg bleibt P unveraendert.

    JE REGION VERSCHIEDEN: Tiefe kommt aus `relief_m`, die Breite wird mit
    `formgroesse_m` moduliert. Im Alpenland tiefe Troege, im Huegelland flache
    Sohlen - dieselbe Rechnung, ortsabhaengige Zahlen.

    DIE BREITE HAENGT AM NETZ, NICHT AM RAUSCHEN (2026-08-10).
    ---------------------------------------------------------
    Bis dahin war die Talbreite schlicht `breite_faktor * formgroesse_m`, also
    1200 bis 5500 m. Die feinsten Baeche stehen aber nur 150 m auseinander
    (STUFEN), auf Land betrug der mittlere Abstand zum naechsten Lauf 111 m.
    Jedes Talprofil reichte damit ueber Dutzende Nachbarlaeufe hinweg, und weil
    in der Ferne auf eine GEGLAETTETE Sohle gezogen wird, wirkte das Eingraben
    nicht als Kerbe, sondern als Tiefpass ueber das ganze Land.

    Gemessen am 2026-08-10, 384 px: das Netz nahm dem Land im Mittel 4.9 Grad
    Hang ab, der Mittelmeerkueste 8.9 von 14.8. Die Regionen waren damit auf
    ein Gelaende geeicht, das die Anzeige nie zu sehen bekam - die Eichung traf
    `weltfeld()`, gezeigt wurde das Ergebnis dieser Funktion.

    Ein Tal ist ein Merkmal des ENTWAESSERUNGSNETZES. Seine Breite gehoert
    deshalb an den Knotenabstand, nicht an die Formgroesse des Rauschens:

        breite = breite_faktor * abstand_makro_m * (formgroesse_m / BEZUG)

    `formgroesse_m` bleibt als Modulation drin - grosse Formen, breitere
    Sohlen - aber auf den Mittelwert der Tabelle bezogen, damit der Faktor um 1
    herum liegt statt die Groessenordnung zu setzen. Dreht man am Regler
    `river_spacing_m`, wachsen die Taeler jetzt mit dem Netz mit.

    NUR UEBER WASSER. Die Laeufe reichen bis MUENDUNGSTIEFE_M, eingegraben wird
    aber nur oberhalb von 0 - "werden danach an der wasserkante abgeschnitten".
    """
    size = H.shape[0]
    mpp = rw.WELT_KM * 1000.0 / size
    pk, el, fl = netz["punkte"], netz["eltern"], netz["flaeche"]
    reihenfolge = netz["reihenfolge"]

    yi = np.clip(np.round(pk[:, 0]).astype(int), 0, size - 1)
    xi = np.clip(np.round(pk[:, 1]).astype(int), 0, size - 1)
    z = H[yi, xi].astype(np.float64)

    # Streng fallend flussabwaerts. GESENKT wird, nicht angehoben - ein
    # angehobener Lauf laege ueber dem Gelaende.
    gefaelle = 0.001
    for i in reihenfolge[::-1]:
        e = el[i]
        if e < 0:
            continue
        z[e] = min(z[e], z[i] - gefaelle * float(np.linalg.norm(pk[i] - pk[e])) * mpp)

    # Einzugsgebiet entlang des Baumes glaetten, sonst springt die Breite an
    # jedem Zusammenfluss und der Trog sieht aus wie eine Wurstkette.
    glatt_fl = fl.astype(np.float64).copy()
    for _ in range(4):
        neu = glatt_fl.copy()
        for i in range(len(glatt_fl)):
            if el[i] >= 0:
                neu[i] = 0.5 * glatt_fl[i] + 0.5 * glatt_fl[el[i]]
        glatt_fl = neu
    gebiet = glatt_fl / max(float(glatt_fl.max()), 1.0)

    # Ortsabhaengige Talbreite und -tiefe aus dem Regionenfeld. Der Massstab
    # der Breite ist der Knotenabstand des Netzes, `formgroesse_m` moduliert
    # nur noch - siehe Kopf der Funktion.
    abstand = float(abstand_makro_m if abstand_makro_m else STUFEN[0][1])
    breite_feld = (breite_faktor * abstand
                   * (felder["formgroesse_m"] / BEZUGSFORM_M) / mpp)
    tiefe_feld = tiefe_anteil * felder["relief_m"]

    sohle = np.full((size, size), np.nan)
    breite = np.zeros((size, size))
    untergrenze = 0.22
    for i in np.argsort(-fl):
        e = el[i]
        if e < 0:
            continue
        strecke = float(np.linalg.norm(pk[i] - pk[e]))
        schritte = max(int(strecke * 3.0), 3)
        anteil = untergrenze + (1.0 - untergrenze) * gebiet[i] ** 0.40
        # DIE STUETZSTELLEN AUF EINMAL, nicht einzeln (2026-08-23).
        #
        # Hier stand eine Schleife ueber `schritte` Stuetzstellen je Kante,
        # und in ihr zwei `np.clip`-Aufrufe auf SKALAREN. Gemessen per
        # cProfile auf einer 1024-px-Karte: 215 761 clip-Aufrufe, 6.5 s von
        # 12.6 s allein dafuer - `np.clip` auf einem einzelnen Wert ist
        # praktisch nur Aufrufaufwand, die eigentliche Rechnung ist ein
        # Vergleich. Dazu 215 759 `round()`-Aufrufe mit weiteren 1.1 s.
        #
        # DIE SCHREIBREIHENFOLGE BLEIBT DIESELBE, und darauf kommt es an:
        #   * `sohle` wird ZUGEWIESEN. Treffen mehrere Stuetzstellen
        #     dasselbe Pixel, gewann in der Schleife die letzte; bei
        #     `sohle[ys, xs] = werte` gewinnt ebenfalls der letzte Eintrag.
        #   * `breite` wird MAXIMIERT. `np.maximum.at` ist dafuer da und
        #     von der Reihenfolge ohnehin unabhaengig.
        # `np.round` rundet wie Pythons `round` zur geraden Zahl, die
        # Pixelwahl ist also unveraendert. Geprueft in
        # tests/smoke_test_weltfluesse_vektor.py.
        t = np.linspace(0.0, 1.0, schritte)
        ps = pk[e][None, :] * (1.0 - t)[:, None] + pk[i][None, :] * t[:, None]
        ys = np.clip(np.round(ps[:, 0]), 0, size - 1).astype(np.int64)
        xs = np.clip(np.round(ps[:, 1]), 0, size - 1).astype(np.int64)
        w = np.maximum(anteil * breite_feld[ys, xs], 2.5)
        sohle_roh = z[e] * (1.0 - t) + z[i] * t
        # EROSIONSBASIS: ein Fluss kann nicht unter den Meeresspiegel
        # schneiden. Ohne diese Schranke grub sich ein Lauf auf 20 m Hoehe
        # 60 m tief ein und legte die halbe Kueste unter Wasser - gemessen
        # fiel der Landanteil von 65 auf 54 Prozent. Die Eintiefung ist
        # deshalb hoechstens ein Teil der Hoehe ueber Null.
        tief = np.minimum(tiefe_feld[ys, xs] * gebiet[i] ** 0.30,
                          0.55 * np.maximum(sohle_roh, 0.0))
        sohle[ys, xs] = sohle_roh - tief
        np.maximum.at(breite, (ys, xs), w)

    ist_fluss = np.isfinite(sohle)
    if not ist_fluss.any():
        return H

    abstand, index = ndimage.distance_transform_edt(~ist_fluss,
                                                    return_indices=True)
    z_nah = sohle[index[0], index[1]]
    b_nah = np.maximum(breite[index[0], index[1]], 1e-6)

    # Messerschneiden zwischen zwei nahen Laeufen auf verschiedener Hoehe:
    # keine Sohle darf ueber einer nahen tieferen stehen, steiler als max_hang.
    if max_hang > 0.0:
        # RADIUS IN PIXELN BEGRENZT.
        #
        # Der Kegel soll Messerschneiden zwischen zwei nahen Laeufen brechen -
        # dafuer genuegen wenige Pixel. Ohne Deckel waechst er mit der Talbreite
        # in Pixeln, also mit der Aufloesung: bei 1024 px war er viermal so
        # gross wie bei 256, und grey_erosion kostet quadratisch mit dem
        # Radius. Gemessen am 2026-08-06: das Eingraben brauchte bei 1024 px
        # 66 der 82 Sekunden, bei 2048 px brach es mit MemoryError ab.
        #
        # Zehn Pixel reichen: was weiter auseinanderliegt, ist keine Schneide
        # mehr, sondern ein Bergruecken.
        r = int(np.clip(round(2.0 * np.median(b_nah)), 2, 10))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        kegel = -max_hang * mpp * np.hypot(yy, xx)
        z_nah = ndimage.grey_erosion(
            z_nah, footprint=np.ones((2 * r + 1, 2 * r + 1), dtype=bool),
            structure=kegel)

    sigma = max(0.5 * float(np.median(b_nah)), 1.0)
    z_glatt = ndimage.gaussian_filter(z_nah, sigma)
    b_glatt = ndimage.gaussian_filter(b_nah, sigma)

    t = abstand / np.maximum(0.5 * (b_nah + b_glatt), 1e-6)
    d = 1.0 - np.exp(-np.maximum(t, 0.0))

    # DIE TALFORM JE REGION (Nutzervorgabe 2026-08-24: *"ja flusstypen
    # sollte es geben, nach region. zB alpen eher V. Fjordland U und im
    # Atlantik irgendwas dazwischen"*).
    #
    # `form` war bis hierher ein Festwert (1.3) fuer die ganze Karte. Der
    # Exponent bestimmt den Querschnitt: klein heisst, das Profil steigt
    # sofort - schmale Sohle, steile Flanken, also ein fluvial
    # eingeschnittenes V-Tal. Gross heisst, es steigt traege - breite
    # flache Sohle, also ein glazial ausgeschuerftes U-Tal.
    #
    # `felder["talform"]` ist wie alle Regionsparameter ueber die
    # Voronoi-Gewichte weich ueberblendet; an einer Regionsgrenze geht die
    # Talform also allmaehlich ueber, statt zu springen.
    form_feld = felder.get("talform")
    if form_feld is not None:
        form = np.clip(np.asarray(form_feld, dtype=np.float64), 0.3, 6.0)
    profil = np.power(d, form)
    z_eff = (1.0 - profil) * z_nah + profil * z_glatt
    neu = H - (H - z_eff) * (1.0 - profil)
    return np.where(H > 0.0, neu, H)
