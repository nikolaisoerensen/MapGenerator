"""
Path: core/tektonik.py

TEKTONISCHE PLATTEN als Formengrammatik fuer die grossen Gebirgszuege.

Nutzer-Vorgabe 2026-08-16: *"wir haben voronois und wir spawnen tektonische
seeds d.h. 2-3 voronois mit mindestabstand werden unterschiedlichen platten
zugewiesen und diese wachsen an bis alle voronois zugewiesen sind. dann
bewegen wir die platten ineinander und erzeugen damit falten fuer gebirge."*

WOZU DAS GUT IST, und was es NICHT ist.

Heute entstehen Berge aus Rauschen mal Regionsparameter. Rauschen hat keine
Richtung - deshalb wirken die Grate beliebig, und ein zusammenhaengender
Gebirgszug wie die Alpen oder die Pyrenaeen entsteht nur zufaellig. Der
ATEF-Erosionsfilter legt zwar Grate und Rinnen hinein, aber das sind
Entwaesserungsstrukturen im Kleinen; die grossraeumige Richtung fehlt.

Eine Plattengrenze liefert genau diese Richtung: eine LINIE ueber die halbe
Karte, an der das Gelaende aufgefaltet wird. Das ist der Gewinn.

Die Welt ist WELT_KM breit und stellt einen Kontinent im verkleinerten
Massstab dar (Europa-Nachbau, Nutzer-Vorgabe). Die Platten hier sind also
massstabsgetreu zur Welt, nicht zur Erde - sie erzeugen die FORM eines
Faltengebirges, sie simulieren keine Geophysik. Wer spaeter Geschwindigkeiten
in cm/Jahr o.ae. daraus ableiten will, rechnet mit den falschen Zahlen.

ABLAUF

  1. `platten_zerlegung()`  Poisson-Zellen ueber die Karte, 2-5 Saatzellen mit
                            Mindestabstand, Wachstum per Breitensuche ueber
                            den Zellnachbarschaftsgraphen bis alles zugeteilt
                            ist. Konkurrierende Fronten, also runde Buchten
                            statt gerader Schnitte.
  2. `plattenbewegung()`    Je Platte ein Geschwindigkeitsvektor aus dem Seed.
  3. `kollisionsfeld()`     Je Grenzpixel die Konvergenz (Skalarprodukt der
                            Relativgeschwindigkeit mit der Grenznormalen):
                            > 0 aufeinander zu (Faltung), < 0 auseinander
                            (Graben). Daraus ein Hebungsfeld mit
                            Faltenzuegen parallel zur Grenze.

Die Zellmaschinerie ist bewusst DIESELBE wie in `seegliederung()`
(Poisson-Punkte -> cKDTree-Etiketten -> Pixel-Adjazenz -> Breitensuche) - ein
zweiter Zellbau waere eine zweite Wahrheit (SPEZIFIKATION §4.5).
"""

from collections import deque

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from core.terrain_river_network import poisson_points
from core.terrain_weltkarte import WELT_KM


# Zellen, in die die Karte zerlegt wird, bevor Platten daraus wachsen. Mehr
# Zellen = feiner gezackte Plattengrenzen. 200 liegt in derselben
# Groessenordnung wie die Regionszellen (punktzahl=200).
ZELLZAHL = 200

# Mindestabstand zweier Plattensaaten, als Anteil der Kartenkante. Ohne ihn
# koennen zwei Saaten benachbart liegen und eine Platte bleibt ein Fleck.
SAAT_MINDESTABSTAND = 0.28

# Wie weit die Auffaltung von der Grenze ins Landesinnere reicht, in Metern.
# Ein Faltengebirge ist breiter als seine Naht - die Alpen sind rund 150 km
# breit bei einer Sutur von wenigen km.
FALTEN_REICHWEITE_M = 2600.0

# Wellenlaenge der einzelnen Faltenzuege (Parallelketten) in Metern.
#
# 2026-08-17 von 900 auf 3600 m VERVIERFACHT (Nutzer: "nur 1/4 so viele
# falten. also wir deuten das alles ja nur an, auf 21 km brauchen wir also
# weniger viel von allem"). Bei 900 m lagen auf der 2600-m-Reichweite fast
# drei Parallelketten - auf einer 21-km-Karte ist das Detail, das die
# Oktaven ohnehin liefern. Mit 3600 m bleibt eine deutliche Hauptkette und
# eine schwach angedeutete zweite, die der Reichweitenabfall schon
# weitgehend wegnimmt.
FALTEN_WELLENLAENGE_M = 3600.0

# Hoehe der Auffaltung bei voller Konvergenz, in Metern.
HEBUNG_M = 900.0

# Tiefe eines Grabens bei voller Divergenz, in Metern. Deutlich kleiner als
# die Hebung: ein Riftgraben ist eine Senke, kein Antigebirge.
GRABEN_M = 220.0


def platten_zerlegung(size, seed, plattenzahl=3, zellzahl=ZELLZAHL,
                      mindestabstand=SAAT_MINDESTABSTAND):
    """
    Zerlegt die Karte in `plattenzahl` Platten.

    Rueckgabe (platte, etikett, punkte):
      platte  (size,size) int16 - Plattenindex je Pixel, 0..plattenzahl-1
      etikett (size,size) int32 - Voronoi-Zellindex je Pixel
      punkte  (n,2) float      - Zellmittelpunkte in Pixelkoordinaten (y,x)

    Das Wachstum ist eine Breitensuche mit KONKURRIERENDEN FRONTEN: alle
    Saaten starten gleichzeitig, jede Runde waechst jede Platte um einen
    Ring. Dadurch treffen sich zwei Platten dort, wo ihre Laufzeiten gleich
    sind - eine unregelmaessige, aber zusammenhaengende Naht. Waechst man
    stattdessen Platte fuer Platte nacheinander, umschliesst die erste die
    anderen (derselbe Effekt wie beim "single front growth" in
    Huftier et al. 2026, Fig. 12).
    """
    mpp = WELT_KM * 1000.0 / size
    flaeche_m2 = (WELT_KM * 1000.0) ** 2
    abstand_m = np.sqrt(0.7 * flaeche_m2 / max(zellzahl, 8))
    punkte = poisson_points(WELT_KM * 1000.0, abstand_m, int(seed) ^ 0x71EC) / mpp
    if len(punkte) < plattenzahl:
        punkte = np.array([[size * 0.5, size * 0.5]], dtype=np.float64)

    n = len(punkte)
    gy, gx = np.mgrid[0:size, 0:size]
    _, etikett = cKDTree(punkte).query(np.stack([gy.ravel(), gx.ravel()], axis=1))
    etikett = etikett.reshape(size, size).astype(np.int32)

    nachbarn = _nachbarschaft(etikett, n)
    saaten = _saaten_waehlen(punkte, plattenzahl, size, mindestabstand, seed)

    # KONKURRIERENDE FRONTEN: eine gemeinsame Warteschlange, in die alle
    # Saaten am Anfang hineingehen - nicht eine Schleife je Platte.
    zuteilung = np.full(n, -1, dtype=np.int16)
    schlange = deque()
    for platte_index, zelle in enumerate(saaten):
        zuteilung[zelle] = platte_index
        schlange.append(zelle)
    while schlange:
        k = schlange.popleft()
        for m in nachbarn[k]:
            if zuteilung[m] == -1:
                zuteilung[m] = zuteilung[k]
                schlange.append(m)
    # Zellen ohne Nachbarschaftsverbindung (isolierte Punkte am Kartenrand)
    # der raeumlich naechsten Saat zuschlagen, statt sie auf -1 zu lassen -
    # ein -1 im Plattenfeld waere ein stiller Loch-Index im Bild.
    if np.any(zuteilung == -1):
        offen = np.flatnonzero(zuteilung == -1)
        _abst, nahe = cKDTree(punkte[saaten]).query(punkte[offen])
        zuteilung[offen] = nahe.astype(np.int16)

    return zuteilung[etikett].astype(np.int16), etikett, punkte


def _nachbarschaft(etikett, n):
    """Zellnachbarn aus PIXEL-Adjazenz - wie in seegliederung()."""
    nachbarn = [set() for _ in range(n)]
    unterschied_h = etikett[:, :-1] != etikett[:, 1:]
    for a, b in zip(etikett[:, :-1][unterschied_h].tolist(),
                    etikett[:, 1:][unterschied_h].tolist()):
        nachbarn[a].add(b)
        nachbarn[b].add(a)
    unterschied_v = etikett[:-1, :] != etikett[1:, :]
    for a, b in zip(etikett[:-1, :][unterschied_v].tolist(),
                    etikett[1:, :][unterschied_v].tolist()):
        nachbarn[a].add(b)
        nachbarn[b].add(a)
    return nachbarn


def _saaten_waehlen(punkte, plattenzahl, size, mindestabstand, seed):
    """
    `plattenzahl` Zellen mit Mindestabstand, deterministisch aus dem Seed.

    Gierig statt per Zufallsziehung mit Verwerfen: bei drei Saaten und 28 %
    Mindestabstand ist die Ablehnungsrate hoch genug, dass eine Ziehschleife
    gelegentlich gar nichts findet und dann still weniger Platten liefert -
    dieselbe Falle wie bei `waehle_punkte()` im Delatin-Nachbau.
    """
    rng = np.random.default_rng(int(seed) ^ 0x71ED)
    schranke_px = mindestabstand * size
    reihenfolge = rng.permutation(len(punkte))

    gewaehlt = [int(reihenfolge[0])]
    for kandidat in reihenfolge[1:]:
        if len(gewaehlt) >= plattenzahl:
            break
        abstaende = np.hypot(punkte[gewaehlt, 0] - punkte[kandidat, 0],
                             punkte[gewaehlt, 1] - punkte[kandidat, 1])
        if abstaende.min() >= schranke_px:
            gewaehlt.append(int(kandidat))

    # Nicht genug Punkte mit vollem Abstand gefunden: Schranke schrittweise
    # senken, statt mit weniger Platten zurueckzukommen als angefordert. Das
    # LAUT zu melden waere hier falsch - es ist der erwartete Fall bei vielen
    # Platten auf kleiner Karte -, aber still WENIGER Platten zu liefern waere
    # die Falle aus CLAUDE.md.
    faktor = 0.85
    while len(gewaehlt) < plattenzahl and schranke_px > 1.0:
        schranke_px *= faktor
        for kandidat in reihenfolge:
            if len(gewaehlt) >= plattenzahl:
                break
            if int(kandidat) in gewaehlt:
                continue
            abstaende = np.hypot(punkte[gewaehlt, 0] - punkte[kandidat, 0],
                                 punkte[gewaehlt, 1] - punkte[kandidat, 1])
            if abstaende.min() >= schranke_px:
                gewaehlt.append(int(kandidat))
    return gewaehlt


def plattenbewegung(plattenzahl, seed):
    """
    Ein Geschwindigkeitsvektor je Platte, Einheitslaenge mal Betrag 0.4..1.0.

    Rueckgabe (plattenzahl, 2) in (dy, dx) - dieselbe Achsenreihenfolge wie
    die Pixelkoordinaten, damit beim Skalarprodukt mit der Grenznormalen
    nichts vertauscht werden kann.
    """
    rng = np.random.default_rng(int(seed) ^ 0x71EE)
    winkel = rng.random(plattenzahl) * 2.0 * np.pi
    betrag = 0.4 + 0.6 * rng.random(plattenzahl)
    return np.stack([np.sin(winkel) * betrag, np.cos(winkel) * betrag], axis=1)


def kollisionsfeld(platte, geschwindigkeit, size,
                   reichweite_m=FALTEN_REICHWEITE_M,
                   wellenlaenge_m=FALTEN_WELLENLAENGE_M,
                   hebung_m=HEBUNG_M, graben_m=GRABEN_M):
    """
    Hebungsfeld in METERN aus den Plattenbewegungen.

    Je Pixel wird die naechstgelegene Plattengrenze gesucht (exakte
    euklidische Distanztransformation mit `return_indices`, wie in
    `_seetiefe_aus_archetyp()` - keine ringweise Naeherung, die sich als
    Diagonalmuster raecht, siehe 3.13).

    An der Grenze steht die KONVERGENZ: das Skalarprodukt der
    Relativgeschwindigkeit beider Platten mit der Grenznormalen.

        > 0   die Platten laufen aufeinander zu  -> Auffaltung
        < 0   sie laufen auseinander             -> Graben
        ~ 0   sie schrammen aneinander vorbei    -> fast nichts

    Rueckgabe (hebung, konvergenz_karte, grenzmaske):
      hebung           (size,size) float - Meter, positiv = angehoben
      konvergenz_karte (size,size) float - -1..1, je Pixel von der naechsten
                                           Grenze uebernommen (fuer die Anzeige)
      grenzmaske       (size,size) bool  - die Grenzpixel selbst
    """
    mpp = WELT_KM * 1000.0 / size

    # Grenzpixel: ein Pixel, dessen rechter oder unterer Nachbar zu einer
    # anderen Platte gehoert. Beide Seiten markieren, sonst liegt die Naht
    # systematisch um einen halben Pixel versetzt.
    grenze = np.zeros((size, size), dtype=bool)
    anders_h = platte[:, :-1] != platte[:, 1:]
    grenze[:, :-1] |= anders_h
    grenze[:, 1:] |= anders_h
    anders_v = platte[:-1, :] != platte[1:, :]
    grenze[:-1, :] |= anders_v
    grenze[1:, :] |= anders_v

    if not grenze.any():
        leer = np.zeros((size, size), dtype=np.float64)
        return leer, leer.copy(), grenze

    # NORMALE DER GRENZE aus dem geglaetteten Plattenfeld: der Gradient von
    # "Plattenindex als Zahl" steht senkrecht auf der Grenze. Geglaettet,
    # weil der rohe Index eine Treppenfunktion ist und sein Gradient nur an
    # der Kante ueberhaupt existiert.
    weich = ndimage.gaussian_filter(platte.astype(np.float64), sigma=2.0)
    ny, nx = np.gradient(weich)
    laenge = np.hypot(ny, nx)
    sicher = np.where(laenge > 1e-9, laenge, 1.0)
    ny, nx = ny / sicher, nx / sicher

    # Relativgeschwindigkeit an jedem Grenzpixel. Als "andere Platte" gilt
    # die des naechsten Pixels in Normalenrichtung.
    gy, gx = np.mgrid[0:size, 0:size]
    nachbar_y = np.clip(np.round(gy + ny * 2.0).astype(int), 0, size - 1)
    nachbar_x = np.clip(np.round(gx + nx * 2.0).astype(int), 0, size - 1)
    platte_gegen = platte[nachbar_y, nachbar_x]

    v_eigen = geschwindigkeit[platte]
    v_gegen = geschwindigkeit[platte_gegen]
    relativ = v_gegen - v_eigen
    # Positiv, wenn die Gegenplatte auf uns zulaeuft.
    konvergenz = -(relativ[:, :, 0] * ny + relativ[:, :, 1] * nx)
    konvergenz = np.where(platte == platte_gegen, 0.0, konvergenz)

    # Grenzwerte auf die ganze Karte ausbreiten: je Pixel den Wert der
    # naechstgelegenen Grenze.
    abstand_px, index = ndimage.distance_transform_edt(~grenze, return_indices=True)
    konvergenz_karte = konvergenz[index[0], index[1]]
    # Ueber die Naht glaetten - an einem Dreiplattenpunkt treffen sonst drei
    # verschiedene Konvergenzwerte hart aufeinander.
    konvergenz_karte = ndimage.gaussian_filter(konvergenz_karte, sigma=max(2.0, 900.0 / mpp))
    konvergenz_karte = np.clip(konvergenz_karte, -1.0, 1.0)

    abstand_m = abstand_px * mpp

    # PROFIL QUER ZUR GRENZE: exponentiell abklingend statt hart begrenzt -
    # eine harte Reichweite gaebe einen sichtbaren Ring um jedes Gebirge.
    quer = np.exp(-abstand_m / reichweite_m)

    # FALTENZUEGE: Parallelketten neben der Hauptkette. cos ueber den Abstand
    # zur Grenze, mit dem Abstand gedaempft - die erste Kette ist die
    # hoechste, die weiteren klingen ab.
    falten = 0.65 + 0.35 * np.cos(2.0 * np.pi * abstand_m / wellenlaenge_m)

    hebung = np.where(
        konvergenz_karte >= 0.0,
        hebung_m * konvergenz_karte * quer * falten,
        graben_m * konvergenz_karte * quer)

    return hebung, konvergenz_karte, grenze


# Wieviel des Grenzverlaufs mindestens konvergent sein muss, damit die Karte
# ueberhaupt ein Gebirge bekommt - siehe _bewegung_mit_gebirge().
MINDEST_KONVERGENZ_ANTEIL = 0.18


def _bewegung_mit_gebirge(platte, size, seed, plattenzahl):
    """
    Geschwindigkeiten, die GARANTIERT ein Gebirge erzeugen.

    GEMESSENER GRUND (2026-08-17, beim Bau): bei zwei Platten lieferten die
    Seeds 20260804 und 12345 `Hebung max = 0 m` - die Platten liefen
    auseinander oder schrammten aneinander vorbei, die Karte bekam KEINEN
    einzigen Berg. Physikalisch ist das ein voellig gueltiger Grenztyp
    (divergent bzw. Transformstoerung), als Gelaendegenerator aber der
    entartete Fall: ein plausibel aussehendes Hoehenfeld, das in Wahrheit
    nichts von dem enthaelt, wofuer die Tektonik gebaut wurde. Genau die
    stille Ruecklauf-Falle aus CLAUDE.md.

    Hier wird deshalb gewuerfelt, bis ein ausreichender Anteil des
    Grenzverlaufs konvergent ist - deterministisch ueber abgeleitete Seeds,
    also weiterhin reproduzierbar. Findet sich nach `versuche` Runden nichts,
    wird die beste gefundene Runde genommen UND das im Rueckgabewert
    vermerkt, damit der Aufrufer es melden kann statt es zu verschlucken.
    """
    bester_anteil, beste_bewegung = -1.0, None
    versuche = 12
    for runde in range(versuche):
        bewegung = plattenbewegung(plattenzahl, int(seed) + runde * 104729)
        _hebung, konvergenz, grenze = kollisionsfeld(platte, bewegung, size)
        if not grenze.any():
            return bewegung, 0.0, True
        anteil = float((konvergenz[grenze] > 0.15).mean())
        if anteil > bester_anteil:
            bester_anteil, beste_bewegung = anteil, bewegung
        if anteil >= MINDEST_KONVERGENZ_ANTEIL:
            return bewegung, anteil, True
    return beste_bewegung, bester_anteil, False


def tektonik(size, seed, plattenzahl=3, zellzahl=ZELLZAHL,
             mindestabstand=SAAT_MINDESTABSTAND,
             reichweite_m=FALTEN_REICHWEITE_M,
             wellenlaenge_m=FALTEN_WELLENLAENGE_M,
             hebung_m=HEBUNG_M, graben_m=GRABEN_M):
    """
    Alles zusammen. Rueckgabe als dict, damit das Labor jede Zwischenstufe
    einzeln anzeigen kann.

    `konvergenz_gefunden=False` heisst: auch nach mehreren Runden kam kein
    ordentliches Gebirge zustande (siehe _bewegung_mit_gebirge). Das Feld ist
    trotzdem gueltig - der Aufrufer soll es aber MELDEN und nicht so tun, als
    waere alles in Ordnung.
    """
    platte, etikett, punkte = platten_zerlegung(
        size, seed, plattenzahl, zellzahl, mindestabstand)
    plattenzahl_echt = int(platte.max()) + 1
    geschwindigkeit, konvergenz_anteil, gefunden = _bewegung_mit_gebirge(
        platte, size, seed, plattenzahl_echt)
    hebung, konvergenz, grenze = kollisionsfeld(
        platte, geschwindigkeit, size, reichweite_m, wellenlaenge_m,
        hebung_m, graben_m)
    return {
        "platte": platte,
        "etikett": etikett,
        "punkte": punkte,
        "geschwindigkeit": geschwindigkeit,
        "hebung": hebung,
        "konvergenz": konvergenz,
        "grenze": grenze,
        "plattenzahl": plattenzahl_echt,
        "konvergenz_anteil": konvergenz_anteil,
        "konvergenz_gefunden": gefunden,
        "plattenanteile": [float((platte == i).mean())
                           for i in range(plattenzahl_echt)],
    }
