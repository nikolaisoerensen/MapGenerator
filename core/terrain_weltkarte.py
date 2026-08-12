"""
Path: tools/regionen_welt.py

DIE REGIONENWELT - Rechenkern, ohne Fenster.

Neun Regionen in einem 3x3-Gitter, jede 4 x 4 km, Kontinent also 12 x 12 km,
Welt 15 x 15 km mit Wasser ringsum (docs/INTEGRATIONSPLAN.md).

Anordnung wie die Gegenden wirklich liegen, Spalte West -> Ost:

    Nord    Huegelland (Kelten)   Fjordland (Wikinger)   Taiga (Slawen)
    Mitte   Atlantikkueste        Alpenland              Mittelgebirge
    Sued    Steppe (Andalus)      Mittelmeer (Italien)   Griech. Inseln


DER KERNGEDANKE: KEIN MOSAIK, SONDERN EIN PARAMETERFELD.

Jeder Regler wird als 3x3-Gitter angegeben und auf volle Aufloesung gebracht -
mit verzerrten Grenzen und weichem Uebergang. Danach gibt es an JEDEM Pixel
einen vollstaendigen Parametersatz, und die Regionen laufen von selbst
ineinander. Es gibt keine Nahtlogik, weil es keine Naht gibt.


WARUM EIN OKTAVENSTAPEL. Formgroesse und Rauheit sollen sich raeumlich aendern:
im Alpenland grosse Massive, im Huegelland kleine Wellen. Mit einem einzigen
Rauschaufruf geht das nicht - dessen Frequenz ist fuer die ganze Karte
dieselbe. Deshalb werden die Oktaven EINZELN erzeugt und je Pixel verschieden
gewichtet. Eine Region mit grosser Formgroesse bekommt nur die groben Oktaven,
eine mit kleiner auch die feinen.


DIE KUESTE ENTSTEHT AUS DEMSELBEN FELD. Kein aufgesetzter Inselgradient:
Atlantikkueste, Mittelmeer und Griechische Inseln haben NEGATIVE Basishoehen,
also liegt die Kueste dort, wo Basis plus Relief unter 0 faellt. Buchten und
Archipel entstehen an den richtigen Stellen, statt dass eine Form sie
aufdrueckt. Nur ringsum zieht ein Randabfall die aeusseren 1.5 km ins Meer.
"""

import numpy as np
from scipy import ndimage

# =============================================================================
# MASSE DER WELT
# =============================================================================

WELT_KM = 21.3            # Kantenlaenge der ganzen Karte
KONTINENT_KM = 13.11      # NUR ALS FLAECHENMASS: der Kontinent bedeckt so viel
                          # wie ein 13.11x13.11-km-Quadrat, hat aber eine
                          # unregelmaessige Form. 172 von 454 km2, also 38 %.
                          #
                          # 2026-08-07 von 12.8 erhoeht: die Atlantikkueste
                          # bekam +40 % Flaechenanteil, die Summe stieg von
                          # 10.25 auf 10.75. Ohne diese Anhebung haette
                          # `flaeche_soll` nur UMVERTEILT - die anderen acht
                          # Regionen waeren geschrumpft. Der Nutzer wollte
                          # das Gegenteil: "das wird aus dem meer allokiert,
                          # so das die gesamte inselngroesse waechst."
REGION_KM = 4.0           # Bezugsgroesse einer Region

# 2026-08-07: ATLANTIKKUESTE VON 1.25 AUF 1.75 (+40 %). Nutzer: "die
# franzoesische region der atlantikkueste sieht insgesamt gut aus, aber ist zu
# klein (wir verlieren zu viel ans meer) ... das wird aus dem meer allokiert,
# so das die gesamte inselngroesse waechst." Die Summe der Flaechenanteile
# steigt damit von 10.25 auf 10.75, also um 4.9 % - die Welt waechst
# entsprechend mit, siehe WELT_KM.

# WARUM DIE WELT GEWACHSEN IST (2026-08-05). Mittelmeer und Griechische Inseln
# verlieren 40 bzw. 65 Prozent ihrer Flaeche ans Wasser, die Atlantikkueste 45.
# Sie bekommen deshalb mehr GRUNDflaeche, damit am Ende genug Land uebrig
# bleibt: Faktor 1.5 / 1.5 / 1.25 gegen 1.0 der uebrigen. Die Summe steigt
# damit von 9.00 auf 10.25 Regionsflaechen, also um 13.9 Prozent - Kontinent
# von 144 auf 164 km2, Welt von 20.0 auf 21.3 km bei gleichem Landanteil.

# PLAETZCHENFORM: so viele ueberlappende Scheiben bilden den Kontinent. In den
# Zwickeln zwischen ihnen entstehen die Buchten, die Raender bleiben rund.
SCHEIBEN = 7

# Groesste Rauschwelle = Kontinentbreite. Alles Groebere waere ein Kipp der
# ganzen Karte und gehoert in die Basishoehe, nicht ins Relief.
GRUNDFORM_M = 12000.0
OKTAVEN = 9               # feinste Welle 12000 / 2^8 = 47 m

MEERESBODEN_M = -200.0    # wohin der Randabfall zieht

# Bis zu welcher HOEHE die Kuestenform wirkt (Meter, beiderseits der
# Wasserlinie). Darueber bleibt das Gelaende unveraendert, damit das
# Landesinnere und die Regionseichung unberuehrt bleiben.
#
# 2026-08-07 von 150 auf 450: gemessen stehen Taiga und Mittelgebirge schon
# 100 m vor der Kueste bei 172 bzw. 268 m Hoehe - GENAU die Klippe, die der
# Nutzer weghaben wollte. Mit der 150-m-Deckelung lag sie ausserhalb des
# Wirkbereichs, und die beiden Regionen aenderten sich um 0 m. Die Deckelung
# muss den ganzen Absturz umfassen, sonst formt sie nur den Saum darunter.
KUESTENHOEHE_M = 450.0


# =============================================================================
# DIE NEUN REGIONEN
# =============================================================================
# Zeilen Nord -> Sued, Spalten West -> Ost.
#
# HOEHEN SIND AUF 4 KM UMGERECHNET, NICHT ABGESCHRIEBEN. Echte Alpen haben
# 2500 m Relief auf 10 km; dieselbe Zahl auf 4 km waere eine Wand mit 60 Grad
# Durchschnittshang. Uebernommen ist das VERHAELTNIS von Relief zu Breite.
#
#   hoehe_m       MITTLERE Hoehe der Region, negativ = ueberwiegend Wasser
#   relief_m      Hoehenspanne um diese Mitte, also hoehe_m +/- relief_m/2
#   formgroesse_m Groesse der groessten Gelaendeform (Massiv, Ruecken, Becken)
#   rauheit       wie stark die feinen Oktaven mitreden (0.4 glatt .. 0.75 rau)
#   potenz        <1 hebt an (Hochflaeche), >1 drueckt herunter (Ebene + Gipfel)
#
# HOEHE IST DIE MITTE, NICHT DER TIEFSTE PUNKT. Zuerst stand hier die
# Grundflaeche, ueber der sich das Relief erhebt. Das war unbrauchbar: das
# normierte Relief hat nur std 0.140 (unabhaengige Oktaven mitteln sich
# heraus), t ueberstrich nur 0.21 bis 0.76 - die Grundflaeche wurde NIE
# erreicht, und jede Kuestenregion blieb zu 100 % trocken. Mit der Mitte als
# Bezug ist der Wasseranteil dagegen unmittelbar einstellbar.

# SPREIZUNG: feste Umrechnung Relief -> [0,1].
#
# Gemessen hat das normierte Relief std 0.140. Mit 1.5 liegen rund +/- 2.4
# Standardabweichungen in [0,1], die Spanne wird also wirklich ausgeschoepft,
# ohne dass grosse Teile der Karte an der Klemmgrenze kleben. FEST, nicht je
# Fenster - eine Normierung auf das jeweilige Bild waere genau der Fehler,
# den die Hoehenskala schon einmal hatte.
SPREIZUNG = 1.5

# HOEHEN NEU GEEICHT AM 2026-08-06, nach der Nord-Sued-Korrektur.
#
# Die alten Werte waren gegen einen Kontinent eingestellt, der auf dem Kopf
# stand (siehe voronoi_regionen). Nach der Umkehr sitzt jede Region auf einem
# anderen Stueck der unregelmaessigen Form, und der Nordlappen traegt mehr
# Wasser als der Suedlappen - das Huegelland sprang von 6 auf 26 Prozent.
#
# Geeicht wurde NUR `hoehe_m`, gemittelt ueber FUENF Seeds, damit die Werte
# nicht auf eine einzelne Kontinentform passen. `relief_m` blieb bewusst
# unberuehrt: der gemessene Hang ist zu einem grossen Teil gar nicht der
# eigene. Ein als Taiga gefuehrtes Pixel traegt im Mittel 33 % FREMDES
# Gewicht und damit 372 m Relief statt der eingetragenen 138 (Atlantikkueste
# +181 %). Bei hoher Reinheit (Gewicht > 0.95) trifft die Taiga ihren
# Sollhang exakt - 3.4 gegen 3.0. Das Relief auf die Mischung zu eichen
# haette es auf 30 m gedrueckt: eine Region ohne Charakter, nur damit eine
# Zahl stimmt, die etwas anderes misst.
#
# ZWEI SEEDS WAREN ZU WENIG (nachkorrigiert am 2026-08-06). Der erste Durchgang
# stellte Fjordland auf 196.6 m; ueber fuenf Seeds gemessen waren das 10
# Prozentpunkte zu wenig Wasser, richtig sind 113.1 m. Die Streuung EINER
# Region ueber Seeds betraegt bis zu 37 Prozentpunkte Wasseranteil und 11 Grad
# Hang - sie sitzt je nach Kontinentform auf einem anderen Stueck Land und hat
# andere Nachbarn. Weniger als vier Seeds eichen auf eine Form, nicht auf die
# Regel.

# DREI KULTURNAMEN GEAENDERT AM 2026-08-06 (docs/KULTUREN_UND_ORTE.md):
#
#   Alpenland           "-"          -> Alemannen    hatte gar keine Kultur und
#                                                    bekam damit keine Siedlungen
#   Mittelgebirge       Franken      -> Sachsen      war doppelt mit der
#                                                    Atlantikkueste belegt
#   Griechische Inseln  Phoenizier   -> Byzantiner   die phoenizischen Stadt-
#                                                    staaten enden rund 1500
#                                                    Jahre vor dem Zeitschnitt
#
# `volk` ist ab jetzt ein SCHLUESSEL, kein Schmuck: der Siedlungsgenerator
# gruppiert danach (2-5 Orte je Kultur, Zusammenhang innerhalb einer Kultur
# erzwungen) und waehlt danach die Landmark- und Roadsite-Arten aus. Zwei
# Regionen mit demselben `volk` waeren eine Kultur mit doppelter Flaeche.

# `farbe` faerbt die Region im Terrain- und im Regional-Reiter. NEUN
# UNTERSCHEIDBARE FARBEN, UND KEIN GELB: Gelb ist fuer das 3x3-Ausschnittsgitter
# reserviert, das im selben Bild liegt. Kein reines Blau, weil das Meer blau
# ist. Die Farbe steht HIER und nicht in der Anzeige - Terrain-Reiter,
# Regional-Reiter und Legende sollen dieselbe benutzen.

# DIE ZIELWERTE - das, was auf der Karte HERAUSKOMMEN soll.
#
# `temp_mittel_m0` und `temp_spanne` in REGIONEN sind NICHT diese Zahlen,
# sondern gegen die Regionsmischung vorkompensierte Eingabewerte. Der Grund
# ist derselbe wie beim Relief: ein Pixel, das als Steppe gefuehrt wird,
# traegt rund 40 % fremdes Gewicht und wird von den kuehleren Nachbarn
# heruntergezogen - gemessen -2.2 K, die Griechischen Inseln -1.8 K.
#
# Damit die Klimatabelle eine FESTLEGUNG bleibt und nicht nur eine Hoffnung,
# wurden die Eingabewerte so geeicht, dass die GEMESSENEN Werte hier landen
# (3 Seeds, 7 Runden). Die Eingabe ist dadurch nicht mehr als "Klima von
# Bergen" lesbar - deshalb steht die lesbare Fassung hier, und
# tests/smoke_test_regionen_welt.py prueft beide gegeneinander.
#
# NACHGEEICHT AM 2026-08-07 nach Einfuehrung von KLIMA_SCHAERFE. Mit der
# schaerferen Klimamischung faellt die Verduennung kleiner aus, und die
# Kompensation entsprechend: Fjordlands Jahresspanne stand vorher auf 5.2 und
# jetzt auf 10.4 bei einem Ziel von 13.0. Die Eingabewerte sind damit wieder
# annaehernd lesbar.
#
# ACHTUNG BEIM NACHEICHEN: das Eichskript muss gegen KLIMA_ZIEL messen, NICHT
# gegen die aktuellen REGIONEN-Werte. Beim ersten Versuch las es die bereits
# kompensierten Werte als Ziel, und die Kompensation schaukelte sich selbst
# auf - die Taiga-Jahresspanne lief von 29 ueber 41.9 auf 56.2.
# Der Jahresniederschlag ist NICHT vorkompensiert - er steht hier so, wie er
# auf der Karte herauskommen soll. Das Wettersystem normiert das fertige
# Niederschlagsfeld direkt auf NIEDERSCHLAG_ZIEL (siehe
# weather_generator._je_region_auf_mittel), womit jede Region ihren Wert per
# Konstruktion trifft.
#
# ZWEI EICHVERSUCHE WAREN VORHER NOETIG UND BEIDE FALSCH: geeicht wurde der
# Eingabewert, gemessen aber das Endergebnis - und die Glaettung der Normierung
# verschiebt das Mittel dazwischen erneut. Der zweite Versuch machte es
# schlechter statt besser (4 statt 3 Regionen daneben). Direkt auf das Ziel zu
# normieren macht die Eichung ueberfluessig; die Taiga stand zwischenzeitlich
# auf 79 mm, damit 600 ankamen.
NIEDERSCHLAG_ZIEL = {
    "Huegelland": 1200.0, "Fjordland": 2250.0, "Taiga": 600.0,
    "Atlantikkueste": 780.0, "Alpenland": 850.0, "Mittelgebirge": 640.0,
    "Steppe": 430.0, "Mittelmeer": 800.0, "Griechische Inseln": 480.0,
}

KLIMA_ZIEL = {
    "Huegelland": (10.9, 9.5),
    "Fjordland": (8.6, 13.0),
    "Taiga": (3.8, 29.0),
    "Atlantikkueste": (13.6, 14.0),
    "Alpenland": (12.8, 18.5),
    "Mittelgebirge": (11.2, 18.5),
    "Steppe": (20.0, 19.0),
    "Mittelmeer": (16.9, 17.5),
    "Griechische Inseln": (19.7, 14.0),
}

# KLIMA JE REGION (2026-08-07). Drei Werte, alle auf MEERESHOEHE:
#
#   temp_mittel_m0    Jahresmittel in Grad, auf 0 m zurueckgerechnet
#   temp_spanne       Jahresspanne (Juli minus Januar) in Kelvin
#   niederschlag_mm   Jahresniederschlag
#
# Abgeleitet aus Bezugsorten, die der Nutzer vorgegeben hat: Cork, Bergen,
# Wologda, La Rochelle, Chur, Bamberg, Madrid, Rom, Iraklio. Die Rueckrechnung
# auf Meereshoehe benutzt 0.6 K je 100 m; Herleitung in docs/BIOME_MATRIX.md.
#
# WARUM MEERESHOEHE UND NICHT REGIONSHOEHE. Eine Regionshoehe ist ein
# GEEICHTER Wert - `hoehe_m` wurde in dieser Woche zweimal nachgezogen. Waere
# das Klima darauf bezogen, waere es stillschweigend mitgewandert. Meereshoehe
# ist der einzige Bezug, der nicht mitwandert.
#
# Die Jahresspanne traegt den Unterschied zwischen See- und Kontinentalklima
# von selbst: Huegelland 9.5 K, Taiga 29.0 K. Niemand muss das modellieren.

REGIONEN = [
    [   # ---------------------------------------------------------- NORD
        dict(name="Huegelland", farbe="#8ab661", volk="Kelten",
             bemerkung="sanfte Wellen, breite Sohlen, dichtes Bachnetz",
             hoehe_m=165.3, relief_m=79.5, formgroesse_m=1600.0,
             rauheit=0.52, potenz=1.0, wasser_soll=0.0, kuestenform=1.45,
             temp_mittel_m0=10.9, temp_spanne=9.5,
             niederschlag_mm=1200, wind_mittel_ms=4.5),
        dict(name="Fjordland", farbe="#5fa8a0", volk="Wikinger",
             bemerkung="EIN Hauptfjord, Hochflaeche, steile Waende",
             hoehe_m=-52.0, relief_m=484.9, formgroesse_m=1400.0,
             rauheit=0.45, potenz=0.55, wasser_soll=20.0, kuestenform=1.90,
             temp_mittel_m0=8.6, temp_spanne=13.0,
             niederschlag_mm=2250, wind_mittel_ms=3.0),
        dict(name="Taiga", farbe="#3f6b4a", volk="Slawen",
             bemerkung="flaches Hochland, weite Mulden, traege Maeander",
             hoehe_m=293.6, relief_m=118.1, formgroesse_m=3000.0,
             rauheit=0.42, potenz=0.9, wasser_soll=0.0, kuestenform=0.45,
             temp_mittel_m0=3.8, temp_spanne=29.0,
             niederschlag_mm=600, wind_mittel_ms=3.2),
    ],
    [   # ---------------------------------------------------------- MITTE
        dict(name="Atlantikkueste", farbe="#9b5fb5", volk="Franken",
             bemerkung="Kuestenebene mit Aestuar, Kliff im Norden",
             hoehe_m=-80.9, relief_m=147.1, formgroesse_m=2400.0,
             rauheit=0.50, potenz=1.3, wasser_soll=45.0, flaeche_soll=1.75, kuestenform=1.00,
             temp_mittel_m0=13.6, temp_spanne=14.0,
             niederschlag_mm=780, wind_mittel_ms=4.5),
        dict(name="Alpenland", farbe="#b5aca0", volk="Alemannen",
             bemerkung="Trogtaeler, scharfe Grate, grosse Massive",
             hoehe_m=1000.0, relief_m=1050.0, formgroesse_m=3800.0,
             rauheit=0.68, potenz=1.5, wasser_soll=0.0, kuestenform=1.00,
             temp_mittel_m0=12.8, temp_spanne=18.5,
             niederschlag_mm=850, wind_mittel_ms=2.2),
        dict(name="Mittelgebirge", farbe="#8a5a33", volk="Sachsen",
             bemerkung="dichte dendritische Zertalung",
             hoehe_m=350.0, relief_m=134.7, formgroesse_m=1400.0,
             rauheit=0.62, potenz=1.0, wasser_soll=0.0, kuestenform=0.45,
             temp_mittel_m0=11.2, temp_spanne=18.5,
             niederschlag_mm=640, wind_mittel_ms=3.0),
    ],
    [   # ---------------------------------------------------------- SUED
        dict(name="Steppe", farbe="#d9a05b", volk="Andalusier",
             bemerkung="Trockentaeler, weite Flaechen, wenig Netz",
             hoehe_m=230.0, relief_m=109.4, formgroesse_m=2600.0,
             rauheit=0.48, potenz=1.4, wasser_soll=0.0, kuestenform=1.10,
             temp_mittel_m0=20.0, temp_spanne=19.0,
             niederschlag_mm=430, wind_mittel_ms=3.0),
        dict(name="Mittelmeer", farbe="#d1603d", volk="Italiener",
             bemerkung="Kuestengebirge direkt am Meer, kurze steile Laeufe",
             hoehe_m=0.6, relief_m=312.9, formgroesse_m=1800.0,
             rauheit=0.60, potenz=1.2, wasser_soll=40.0, flaeche_soll=1.50, kuestenform=1.00,
             temp_mittel_m0=16.9, temp_spanne=17.5,
             niederschlag_mm=800, wind_mittel_ms=3.5),
        dict(name="Griechische Inseln", farbe="#a8447e", volk="Byzantiner",
             bemerkung="Archipel, viel Wasser, kleine steile Inseln",
             hoehe_m=-67.8, relief_m=403.6, formgroesse_m=1100.0,
             rauheit=0.58, potenz=1.1, wasser_soll=65.0, flaeche_soll=1.50, kuestenform=1.00,
             temp_mittel_m0=19.7, temp_spanne=14.0,
             niederschlag_mm=480, wind_mittel_ms=4.5),
    ],
]

REGLER = ("hoehe_m", "relief_m", "formgroesse_m", "rauheit",
          "potenz", "kuestenform",
          "temp_mittel_m0", "temp_spanne", "niederschlag_mm", "wind_mittel_ms")


def regionsname(zeile, spalte):
    return REGIONEN[zeile][spalte]["name"]


def alle_regionen():
    """[(zeile, spalte, dict), ...] von Nordwest nach Suedost."""
    return [(z, s, REGIONEN[z][s]) for z in range(3) for s in range(3)]


def gitter_kante_px(size):
    """Kantenlaenge einer Region in Pixeln - fuer das gelbe 3x3-Ausschnittsgitter
    (docs/OFFENE_PUNKTE.md 6.2/5.14). Dieses Gitter ist ein
    fester Kasten um die Kontinentmitte, unabhaengig von der WEICHEN,
    verzogenen Regionszugehoerigkeit aus `regionsgewichte()` - es dient allein
    der Kartenaufteilung (paginierbare "Regionalkarten"), nicht der Eichung."""
    return REGION_KM * 1000.0 / (WELT_KM * 1000.0 / size)


def gitterlinien_px(size):
    """Die 4 senkrechten und 4 waagrechten Linien des 3x3-Ausschnittsgitters,
    in Pixeln - dieselben Linien fuer Anzeige (gelbes Gitter), Zuschnitt
    (Regional-Reiter) und die weiche Randstrafe der Siedlungsplatzierung."""
    kante = gitter_kante_px(size)
    mitte = 0.5 * size
    return [mitte + versatz * kante for versatz in (-1.5, -0.5, 0.5, 1.5)]


def regionsbox_px(zeile, spalte, size, rand_anteil=0.0):
    """Pixelgrenzen (x0, x1, y0, y1) einer Region als fester Kasten, optional
    mit ueberlappendem Rand (`rand_anteil` = Anteil der Kantenlaenge je Seite,
    z.B. 0.25 fuer 25 %) - docs/OFFENE_PUNKTE.md 5.9, urspr. TODO §D2 Vorschlag 1: "Ueberlappender
    Ausschnitt ... ein Ort auf der Grenze ist dann in beiden Nachbarfeldern
    mit seiner Umgebung zu sehen"."""
    kante = gitter_kante_px(size)
    mitte_y = 0.5 * size - (zeile - 1) * kante
    mitte_x = 0.5 * size + (spalte - 1) * kante
    halb = 0.5 * kante * (1.0 + 2.0 * rand_anteil)
    return (mitte_x - halb, mitte_x + halb, mitte_y - halb, mitte_y + halb)


# =============================================================================
# OKTAVENSTAPEL
# =============================================================================

_STAPEL_CACHE = {}


def oktavenstapel(size, seed, shader_manager=None):
    """
    Die einzelnen Rauschoktaven als (OKTAVEN, size, size).

    Getrennt erzeugt, damit sie je Pixel verschieden gewichtet werden koennen -
    das ist die Voraussetzung dafuer, dass Formgroesse und Rauheit sich
    raeumlich aendern duerfen.

    Wellenlaenge der Oktave k: GRUNDFORM_M / 2^k, in METERN. Damit haengt das
    Ergebnis an der Wirklichkeit und nicht an der Pixelzahl (SPEZIFIKATION §10).
    """
    schluessel = (size, seed, shader_manager is not None)
    if schluessel in _STAPEL_CACHE:
        return _STAPEL_CACHE[schluessel]

    mpp = WELT_KM * 1000.0 / size
    stapel = np.zeros((OKTAVEN, size, size), dtype=np.float32)
    for k in range(OKTAVEN):
        wellenlaenge = GRUNDFORM_M / (2.0 ** k)
        frequenz = mpp / wellenlaenge
        # Je Oktave ein eigener Seed - sonst sind die Oktaven verschobene
        # Fassungen desselben Musters und die Landschaft bekommt Streifen.
        okt_seed = int(seed) + k * 7919
        if shader_manager is not None and shader_manager.gpu_available:
            stapel[k] = shader_manager.process_noise_generation(
                size=size, octaves=1, frequency=frequenz, persistence=0.5,
                lacunarity=2.0, seed=okt_seed)
        else:
            from opensimplex import OpenSimplex
            gen = OpenSimplex(seed=okt_seed)
            achse = np.arange(size, dtype=np.float64) * frequenz
            stapel[k] = gen.noise2array(achse, achse).astype(np.float32)

    _STAPEL_CACHE[schluessel] = stapel
    return stapel


def _wellenlaengen():
    return np.array([GRUNDFORM_M / (2.0 ** k) for k in range(OKTAVEN)])


# =============================================================================
# DIE KONTINENTFORM - "PLAETZCHEN"
# =============================================================================

def kontinentform(size, seed, shader_manager=None, unruhe_m=700.0):
    """
    Eine unregelmaessige, gerundete Landmasse mit Buchten und OHNE Seen.

    Rueckgabe: (maske, sdf) - Boolesche Landmaske und die vorzeichenbehaftete
    Abstandsfunktion in Metern (positiv im Land, negativ auf See).

    WIE DIE FORM ENTSTEHT. Sieben ueberlappende Scheiben, weich vereinigt:

        feld(x) = softmax_k( radius_k - abstand_k(x) )

    Der weiche Maximumoperator rundet die Uebergaenge zwischen den Scheiben ab,
    statt sie als Kante stehen zu lassen - in den Zwickeln entstehen genau die
    konkaven Buchten der Vorlage. Ein Kastengradient kann so etwas nicht, weil
    er nur einen Abstand zum Kartenrand kennt.

    KEINE SEEN, KONSTRUKTIONSBEDINGT. Nach dem Schwellwert werden erst alle
    Loecher geschlossen und dann nur die groesste zusammenhaengende Flaeche
    behalten. Ein eingeschlossenes Wasserstueck kann es danach nicht mehr geben
    - der Nutzer wollte "buchten aber keine seen".

    DIE FLAECHE IST GEEICHT, NICHT GERATEN. Der Schwellwert wird per
    Intervallhalbierung so gesucht, dass die Landflaeche KONTINENT_KM^2
    entspricht. Damit bleibt der Kontinent gleich gross, egal wie
    unregelmaessig die Form ausfaellt - sonst haengt jede Eichung der Regionen
    an der Tagesform des Seeds.
    """
    mpp = WELT_KM * 1000.0 / size
    achse = (np.arange(size) + 0.5) * mpp - 0.5 * WELT_KM * 1000.0
    WX, WY = np.meshgrid(achse, achse, indexing="xy")

    # Die Koordinate verziehen, damit die Scheibenraender nicht als Kreisboegen
    # zu erkennen sind.
    if unruhe_m > 1.0:
        stapel = oktavenstapel(size, int(seed) ^ 0x0FF5, shader_manager)
        WX = WX + unruhe_m * (0.7 * stapel[3] + 0.3 * stapel[5])
        WY = WY + unruhe_m * (0.7 * stapel[4] + 0.3 * stapel[2])

    rng = np.random.default_rng(int(seed) ^ 0xC00C)
    halb = 0.5 * KONTINENT_KM * 1000.0

    # LAPPEN. Der Kern bleibt klein und die Lappen sitzen weit aussen - sonst
    # verschmilzt alles zu einem Klecks. Mit einer Zentralscheibe von 0.62 und
    # Lappen bei 0.45 bis 0.85 kam genau das heraus: ein Kreis mit
    # gekraeuseltem Rand, ohne eine einzige Bucht.
    felder = []
    for k in range(SCHEIBEN):
        if k == 0:
            mx, my, radius = 0.0, 0.0, 0.34 * halb
        else:
            winkel = 2.0 * np.pi * (k - 1) / (SCHEIBEN - 1) \
                + rng.uniform(-0.55, 0.55)
            weite = halb * rng.uniform(0.55, 0.95)
            mx, my = weite * np.cos(winkel), weite * np.sin(winkel)
            radius = halb * rng.uniform(0.26, 0.52)
        felder.append(radius - np.hypot(WX - mx, WY - my))

    # Weiche Vereinigung. Klein gewaehlt, damit die Zwickel zwischen den Lappen
    # spitz bleiben - sie SIND die Buchten.
    weich = 0.055 * halb
    feld = weich * np.log(np.sum(np.exp(np.stack(felder) / weich), axis=0))

    # ABGEZOGENE SCHEIBEN. Die Zwickel allein geben nur flache Einbuchtungen.
    # Zwei bis drei Scheiben, die Material WEGNEHMEN, schneiden tief herein -
    # das sind die langen Buchten der Vorlage. Sie sitzen am Rand und zeigen
    # nach innen.
    for _ in range(rng.integers(2, 4)):
        winkel = rng.uniform(0.0, 2.0 * np.pi)
        weite = halb * rng.uniform(0.75, 1.15)
        bx, by = weite * np.cos(winkel), weite * np.sin(winkel)
        radius = halb * rng.uniform(0.30, 0.55)
        bucht = radius - np.hypot(WX - bx, WY - by)
        # weiches Minimum aus Land und "nicht Bucht"
        feld = -weich * np.log(np.exp(-feld / weich) + np.exp(bucht / weich))

    ziel = (KONTINENT_KM * 1000.0) ** 2
    unten, oben = float(feld.min()), float(feld.max())
    maske = feld > 0.0
    for _ in range(40):
        schwelle = 0.5 * (unten + oben)
        probe = ndimage.binary_fill_holes(feld > schwelle)
        marken, anzahl = ndimage.label(probe)
        if anzahl > 1:
            groessen = ndimage.sum(probe, marken, range(1, anzahl + 1))
            probe = marken == (int(np.argmax(groessen)) + 1)
        flaeche = float(probe.sum()) * mpp * mpp
        maske = probe
        if flaeche > ziel:
            unten = schwelle
        else:
            oben = schwelle

    innen = ndimage.distance_transform_edt(maske) * mpp
    aussen = ndimage.distance_transform_edt(~maske) * mpp
    return maske, innen - aussen


def voronoi_regionen(maske, seed, punktzahl=200, zweitanteil=0.35,
                     glaettung_m=260.0, verzerrung=0.22, shader_manager=None):
    """
    Der Kontinent wird in Voronoi-Zellen zerlegt und diese den Regionen
    zugeteilt. Rueckgabe: Gewichte (9, size, size).

    WARUM VORONOI STATT ABSTANDSFELD. Reine Abstandsgewichte ergeben runde,
    ineinander verlaufende Flecken. Der Nutzer will "nicht runde aber halbwegs
    kompakte regionen mit unregelmaessiger grenze" - und genau das leistet eine
    Zellzerlegung: die Zellen sind kompakt, ihre gemeinsamen Raender aber
    zackig und unvorhersehbar.

    HIMMELSRICHTUNG STIMMT TROTZDEM. Jede Zelle bekommt ihre Region nach der
    Lage im UMSCHLIESSENDEN RECHTECK des Kontinents, nicht der Karte. Damit
    liegt Fjordland auch dann im Norden, wenn die Landmasse dort einen Lappen
    hat und anderswo eine Bucht.

    BIS ZU ZWEI REGIONEN JE ZELLE, wie vom Nutzer vorgegeben: "jedes voronoi
    kann bis zu zwei regionen enthalten ... dort werden dann die mittelwerte
    genommen". Die zweite kommt nur zum Zug, wenn sie mindestens
    `zweitanteil` des Gewichts der ersten hat - sonst bliebe von der klaren
    Zuteilung nichts uebrig.
    """
    import core.terrain_river_network as rn

    size = maske.shape[0]
    mpp = WELT_KM * 1000.0 / size

    # Punkte gleichmaessig im Kontinent. Der Poisson-Abstand folgt aus der
    # Zielzahl und der LANDflaeche, nicht der Kartenflaeche.
    land_m2 = float(maske.sum()) * mpp * mpp
    abstand_m = np.sqrt(0.7 * land_m2 / max(punktzahl, 8))
    punkte = rn.poisson_points(WELT_KM * 1000.0, abstand_m, seed) / mpp
    yi = np.clip(np.round(punkte[:, 0]).astype(int), 0, size - 1)
    xi = np.clip(np.round(punkte[:, 1]).astype(int), 0, size - 1)
    punkte = punkte[maske[yi, xi]]
    if len(punkte) < 9:
        punkte = np.array([[size * 0.5, size * 0.5]])

    # Voronoi als naechster-Nachbar-Etikett. Polygone braucht niemand.
    from scipy.spatial import cKDTree
    gy, gx = np.mgrid[0:size, 0:size]
    _, etikett = cKDTree(punkte).query(
        np.stack([gy.ravel(), gx.ravel()], axis=1))
    etikett = etikett.reshape(size, size)

    # Lage jeder Zelle IM KONTINENT, nicht in der Karte.
    zeilen, spalten = np.nonzero(maske)
    y0, y1 = zeilen.min(), zeilen.max()
    x0, x1 = spalten.min(), spalten.max()
    mitten = np.array([1.0 / 6.0, 0.5, 5.0 / 6.0])

    # DIE ZUTEILUNGSKOORDINATE WIRD VERZOGEN. Ohne das laufen die Grenzen
    # zwischen den Regionen im Inneren fast gerade - die Voronoi-Zacken sitzen
    # nur auf einer geraden Linie. Mit einem glatten Rauschversatz wandert die
    # Grenze selbst, und die Regionen greifen unregelmaessig ineinander.
    stapel = oktavenstapel(size, int(seed) ^ 0x2B0B, shader_manager)
    versatz_u = 0.7 * stapel[2] + 0.3 * stapel[4]
    versatz_v = 0.7 * stapel[3] + 0.3 * stapel[1]

    lage = np.zeros((len(punkte), 2))
    for k in range(len(punkte)):
        py = int(np.clip(round(punkte[k, 0]), 0, size - 1))
        px = int(np.clip(round(punkte[k, 1]), 0, size - 1))
        lage[k, 0] = (punkte[k, 1] - x0) / max(x1 - x0, 1) \
            + verzerrung * float(versatz_u[py, px])          # u, West -> Ost
        # v = 0 IST NORDEN, UND NORDEN IST DIE HOHE ZEILENNUMMER.
        #
        # Ohne die Umkehr stand die Welt auf dem Kopf: Steppe und Mittelmeer
        # oben, Fjordland und Taiga unten. Aufgefallen erst, als die Regionen
        # BESCHRIFTET wurden - an einer namenlosen Hoehenkarte sieht man nicht,
        # wo Norden ist. Genau das ist der Grund, warum die Regionenkarte
        # gebaut wurde: sie macht eine Himmelsrichtung ueberhaupt pruefbar.
        #
        # Die Konvention kommt nicht von hier, sie gilt im ganzen Projekt: die
        # Anzeige zeichnet mit origin='lower' (Zeile 0 unten), und
        # core/weather_generator.py haelt bei Wind und Schattenwurf ausdruecklich
        # "Zeile height-1 = Norden" fest. Ein Regionsgitter, das dem
        # widerspricht, wuerde Fjordland in die Sonne und die Steppe in den
        # Schatten legen.
        lage[k, 1] = 1.0 - (punkte[k, 0] - y0) / max(y1 - y0, 1) \
            + verzerrung * float(versatz_v[py, px])          # v, Nord -> Sued

    d2 = np.stack([(lage[:, 0] - mitten[s]) ** 2 + (lage[:, 1] - mitten[z]) ** 2
                   for z in range(3) for s in range(3)], axis=1)

    # FLAECHENEICHUNG.
    #
    # Mittelmeer und Griechische Inseln verlieren 40 bzw. 65 Prozent ihrer
    # Flaeche ans Wasser, die Atlantikkueste 45. Sie brauchen deshalb mehr
    # GRUNDflaeche - `flaeche_soll` gibt an, wieviel.
    #
    # Ein Faktor auf das Gewicht taete es nicht: er verschiebt eine Grenze, und
    # wieviel Flaeche dabei wandert, haengt an der Form des Kontinents. Deshalb
    # wird ein additiver Vorteil je Region nachgezogen, bis die GEMESSENEN
    # Landanteile den Sollwerten entsprechen. Zwoelf Runden auf 200 Zellen
    # kosten nichts und machen die Eichung unabhaengig von Seed und Form.
    zell_land = np.bincount(etikett[maske].ravel(), minlength=len(punkte))
    soll = np.array([r.get("flaeche_soll", 1.0)
                     for _z, _s, r in alle_regionen()], dtype=np.float64)
    soll /= soll.sum()
    # Der Schritt ist LOGARITHMISCH. Ein linearer Schritt (0.9 * relativer
    # Fehler) schwang: Alpenland landete bei 0.049 statt 0.098, Mittelmeer bei
    # 0.211 statt 0.146. Im Logarithmus entspricht ein Vorteil genau einem
    # Faktor auf das Gewicht, und die Regelung wird stabil.
    vorteil = np.zeros(9)
    for _runde in range(60):
        fuehrt = np.argmax(-d2 / 0.055 + vorteil, axis=1)
        ist = np.array([zell_land[fuehrt == i].sum() for i in range(9)],
                       dtype=np.float64)
        ist = np.maximum(ist / max(ist.sum(), 1.0), 1e-4)
        vorteil += 0.35 * np.log(soll / ist)

    zell_gewichte = np.zeros((len(punkte), 9))
    for k in range(len(punkte)):
        roh = np.exp(-d2[k] / 0.055 + vorteil)
        ordnung = np.argsort(-roh)
        erste, zweite = ordnung[0], ordnung[1]
        gewicht = np.zeros(9)
        gewicht[erste] = roh[erste]
        if roh[zweite] >= zweitanteil * roh[erste]:
            gewicht[zweite] = roh[zweite]
        zell_gewichte[k] = gewicht / gewicht.sum()

    gewichte = zell_gewichte[etikett].transpose(2, 0, 1).copy()

    # Die Zellraender bleiben als FORM sichtbar, ihre Parametersprunge nicht:
    # ohne diese Glaettung stuende an jeder Zellkante eine Stufe im Gelaende.
    sigma = max(glaettung_m / mpp, 0.5)
    for i in range(9):
        gewichte[i] = ndimage.gaussian_filter(gewichte[i], sigma)
    gewichte /= np.maximum(gewichte.sum(axis=0, keepdims=True), 1e-12)
    return gewichte


# Zieltiefe je Seegrad, docs/KLIMA_UND_SEE.md §2 - eine TABELLE statt einer
# Formel (§0: "keine Simulationskreise mehr, sondern in jedem Kreis
# Festlegungen"). 4+ ist der Meeresboden.
#
# GRAD 0 NICHT MEHR 0.0 (2026-08-11, Nutzer-Befund am laufenden Programm):
# "das meer ist nicht vertieft ... es gibt land um die regionen herum
# (wahrscheinlich auf 0m oder gerundet um die 0m, aber hier als gruen
# gezeichnet) ... soll an der kueste immer auf -3m seetiefe abfallen (muss
# keine kante sein)". Ohne erzwungene Mindesttiefe blieb ein breiter Saum
# knapp unter/ueber 0m dem puren Gelaenderauschen ueberlassen - genau dieser
# Saum erschien als unentschiedenes Gruen/Blau-Gemisch statt einer klaren
# Kuestenlinie, UND machte den gesamten kuestennahen (= grossen) Teil der
# See optisch flach. -3m ist ein MINIMUM (`np.minimum`, siehe
# `seegliederung()`), kein fester Wert - eine bereits tiefere Kuestenzelle
# (z.B. Fjordland/Klippenkueste) wird dadurch nicht angehoben. `kuestenform`
# formt weiterhin die LAND-Seite; hier geht es nur um die See-Seite.
TIEFE_JE_SEEGRAD = {0: -3.0, 1: -40.0, 2: -90.0, 3: -150.0, 4: MEERESBODEN_M}

# Seetyp-Abweichungen von der Standardtabelle, docs/OFFENE_PUNKTE.md 3.6.
# Nutzer 2026-08-07: "das faellt vor dem fjordland steil ab (seetyp
# fjordland), das huegelland ab seegrad 2". Nur die GENANNTEN Regionen
# weichen ab - jede nicht aufgefuehrte Region (auch die anderen sieben)
# benutzt weiterhin TIEFE_JE_SEEGRAD unveraendert.
SEETYP_TIEFENTABELLE = {
    # FJORDLAND: schon Grad 1 auf Fjordtiefe statt der sanften Standardkurve -
    # ein Fjord hat keinen flachen Schelf, das Ufer bricht steil weg.
    "Fjordland": {0: -3.0, 1: -90.0, 2: -150.0, 3: -180.0, 4: MEERESBODEN_M},
    # HUEGELLAND: bleibt bis Grad 1 flach ("ab Seegrad 2" - erst dort beginnt
    # die eigentliche Vertiefung), dann schneller auf Meeresbodenniveau.
    "Huegelland": {0: -3.0, 1: -10.0, 2: -60.0, 3: -140.0, 4: MEERESBODEN_M},
}

# Seeeis-Wahrscheinlichkeit je Seegrad vor der Taiga (Nutzer 2026-08-11,
# Nachbesserung der ersten Fassung "nur Grad 1 und 2" - hart an/aus wirkte zu
# kuenstlich): "grade 0 ist 100% eis grade 1 75% chance und grade 2 ist 50%
# chance. und grade 3 ist 25%." Grad 4+ bewusst NICHT gelistet (0 %, .get()
# faellt darauf zurueck) - dort verlaufen Seewege, die frei bleiben sollen.
# NOCH OFFEN, vom Nutzer selbst vertagt: "es wird spaeter im spiel nur im
# winter erscheinen" - diese Karte hier bleibt EIN statischer Schnappschuss
# ohne Jahreszeit (siehe KLIMA_UND_SEE.md §0), die Saisonalitaet gehoert in
# das spaetere Zeitmodell, nicht in diese Funktion.
EISWAHRSCHEINLICHKEIT_JE_SEEGRAD = {0: 1.0, 1: 0.75, 2: 0.5, 3: 0.25}


def seegliederung(maske, gewichte, seed, punktzahl_land=200, punktzahl_see=400,
                  glaettung_m=260.0, shader_manager=None):
    """
    Die See als EIGENE Voronoi-Gliederung, docs/KLIMA_UND_SEE.md §2. Nutzer:
    "wenn wir das inland als voronoi kacheln haben, dann koennen wir ja auch
    das gleiche bei der see machen ... aber auch zB dass die voronois an der
    kueste nicht staerker vertieft werden, aber dass die voronois mit dem
    grad 1 etwas vertiefter sind, grad 2 noch vertiefter etc."

    SEEGRAD per Breitensuche ueber den Zellnachbarschaftsgraphen: Grad 0 sind
    alle Zellen, die Land enthalten, Grad 1 grenzt an eine Grad-0-Zelle, Grad 2
    an Grad 1, und so weiter (docs/KLIMA_UND_SEE.md §2). Ersetzt den alten
    Kuestenschelf `-t*(1-exp(-d/L))`, der rein auf dem EUKLIDISCHEN Abstand zur
    Kueste beruhte (an der Aufloesung der Distanztransformation haengend) durch
    eine Zellstruktur, die spaeter auch fuer Seewege/Seemonster/Fischgruende
    gebraucht wird - "eine Karte, die spaeter etwas bedeutet".

    UFERREGIONEN: je Zelle die bis zu zwei naechstgelegenen LAND-Punkte, ueber
    deren fuehrende Region nachgeschlagen ("bis zu zwei Uferregionen ... damit
    faellt das Ufer vor dem Fjordland steil ab, das Huegelland ab Seegrad 2,
    und die Taiga bekommt Seeeis" - diese drei konkreten Regeln selbst sind
    NICHT Teil dieser Funktion, nur das Datenfundament dafuer).

    EIGENE PUNKTMENGE, unabhaengig von `voronoi_regionen()` - jene ist auf die
    KULTURFLAECHEN-Sollwerte geeicht (Land-only), der Seegrad braucht davon
    unabhaengige, ueber Land UND Wasser verteilte Zellen mit eigener Dichte je
    Seite (die See ist groesser als das Land, siehe Vorbehalt unten).

    VORBEHALT AUS DEM ENTWURF: "wenn das im Bild als Kachelung sichtbar wird,
    muss die Punktzahl fuer die See hoeher liegen als fuer das Land (etwa 400
    statt 200)" - deshalb defaultet `punktzahl_see` auf 400 waehrend
    `punktzahl_land` (nur fuer die Punktdichte hier, nicht fuer die Region-
    Voronoi) bei 200 bleibt.

    Rueckgabe (H,W)-Felder:
        seegrad        int16    0 auf Land, 1..4 auf See, UNGEGLAETTET (fuer
                                 Schwellwertentscheidungen wie "ab Grad 1")
        seegrad_tiefe   float64 Zieltiefe in Metern, GEGLAETTET ueber die
                                 Zellgrenzen - ersetzt den alten Schelf
        ufer_region_a   int16   naechstgelegene Region; auf Land trivial die
                                 eigene Region (wie region_map)
        ufer_region_b   int16   zweitnaechste Region auf See, sonst -1
    """
    import core.terrain_river_network as rn
    from scipy.spatial import cKDTree
    from collections import deque

    size = maske.shape[0]
    mpp = WELT_KM * 1000.0 / size

    land_m2 = float(maske.sum()) * mpp * mpp
    see_m2 = float((~maske).sum()) * mpp * mpp
    abstand_land_m = np.sqrt(0.7 * land_m2 / max(punktzahl_land, 8))
    abstand_see_m = np.sqrt(0.7 * max(see_m2, 1.0) / max(punktzahl_see, 8))

    punkte_land_alle = rn.poisson_points(WELT_KM * 1000.0, abstand_land_m, int(seed) ^ 0x5EE0) / mpp
    punkte_see_alle = rn.poisson_points(WELT_KM * 1000.0, abstand_see_m, int(seed) ^ 0x5EE1) / mpp

    def _behalte(punkte, will_maske):
        yi = np.clip(np.round(punkte[:, 0]).astype(int), 0, size - 1)
        xi = np.clip(np.round(punkte[:, 1]).astype(int), 0, size - 1)
        return punkte[will_maske[yi, xi]]

    punkte_land = _behalte(punkte_land_alle, maske)
    punkte_see = _behalte(punkte_see_alle, ~maske)
    if len(punkte_land) == 0:
        punkte_land = np.array([[size * 0.5, size * 0.5]])
    if len(punkte_see) == 0:
        # Keine offene See auf der Karte (z.B. winziger Testausschnitt) -
        # unveraendert zurueckgeben, nichts zu vertiefen.
        return {
            "seegrad": np.zeros((size, size), dtype=np.int16),
            "seegrad_tiefe": np.zeros((size, size), dtype=np.float64),
            "ufer_region_a": np.argmax(gewichte, axis=0).astype(np.int16),
            "ufer_region_b": np.full((size, size), -1, dtype=np.int16),
            "see_eis": np.zeros((size, size), dtype=bool),
        }

    n_land = len(punkte_land)
    punkte = np.concatenate([punkte_land, punkte_see], axis=0)
    n = len(punkte)

    gy, gx = np.mgrid[0:size, 0:size]
    _, etikett = cKDTree(punkte).query(np.stack([gy.ravel(), gx.ravel()], axis=1))
    etikett = etikett.reshape(size, size)

    hat_land = np.zeros(n, dtype=bool)
    if np.any(maske):
        hat_land[np.unique(etikett[maske])] = True

    # Nachbarschaftsgraph aus PIXEL-Adjazenz der Zellbeschriftung - billiger
    # als echte Voronoi-Ridges und fuer eine Breitensuche voellig ausreichend.
    nachbarn = [set() for _ in range(n)]
    unterschied_h = etikett[:, :-1] != etikett[:, 1:]
    for a, b in zip(etikett[:, :-1][unterschied_h].tolist(), etikett[:, 1:][unterschied_h].tolist()):
        nachbarn[a].add(b)
        nachbarn[b].add(a)
    unterschied_v = etikett[:-1, :] != etikett[1:, :]
    for a, b in zip(etikett[:-1, :][unterschied_v].tolist(), etikett[1:, :][unterschied_v].tolist()):
        nachbarn[a].add(b)
        nachbarn[b].add(a)

    grad = np.full(n, -1, dtype=np.int32)
    warteschlange = deque()
    for k in range(n):
        if hat_land[k]:
            grad[k] = 0
            warteschlange.append(k)
    while warteschlange:
        k = warteschlange.popleft()
        for m in nachbarn[k]:
            if grad[m] == -1:
                grad[m] = grad[k] + 1
                warteschlange.append(m)
    # Isolierte Zellen ohne Landverbindung im Punktnetz (praktisch nur bei
    # sehr wenigen Punkten) als tiefste Stufe behandeln, nicht als Fehler.
    grad[grad == -1] = 4

    seegrad_roh = np.where(maske, 0, grad[etikett]).astype(np.int16)

    # UFERREGIONEN: fuehrende Region JE PUNKT nachschlagen (nicht je Zelle -
    # die Zellform ist fuer diese Suche irrelevant), dann fuer jeden Seepunkt
    # die 1-2 naechsten LAND-Punkte per eigenem KDTree. VOR der Zieltiefe
    # berechnet, weil die SEETYP-ABHAENGIGE Tiefentabelle (siehe unten) sie
    # braucht.
    fuehrende_region = np.argmax(gewichte, axis=0)
    land_regionen = np.array([
        fuehrende_region[int(np.clip(round(py), 0, size - 1)),
                         int(np.clip(round(px), 0, size - 1))]
        for py, px in punkte_land
    ], dtype=np.int16)

    region_a_je_zelle = np.full(n, -1, dtype=np.int16)
    region_b_je_zelle = np.full(n, -1, dtype=np.int16)
    region_a_je_zelle[:n_land] = land_regionen

    k_nachbarn = min(2, n_land)
    _abstaende, idx = cKDTree(punkte_land).query(punkte_see, k=k_nachbarn)
    idx = np.atleast_2d(idx.reshape(len(punkte_see), k_nachbarn))
    region_a_je_zelle[n_land:] = land_regionen[idx[:, 0]]
    if k_nachbarn > 1:
        region_b_je_zelle[n_land:] = land_regionen[idx[:, 1]]

    ufer_region_a = region_a_je_zelle[etikett]
    ufer_region_b = region_b_je_zelle[etikett]

    # SEETYP (docs/OFFENE_PUNKTE.md 3.6, Nutzer 2026-08-07): "das Ufer vor dem
    # Fjordland faellt steil ab ... das Huegelland ab Seegrad 2 ... die Taiga
    # bekommt Seeeis". Die Zieltiefe haengt jetzt an ZWEI Dingen statt einem -
    # dem Seegrad UND der naechsten Uferregion jeder Zelle (`region_a_je_zelle`,
    # nicht region_b - der Seetyp folgt dem NAECHSTEN Ufer, nicht dem zweiten).
    regionsnamen = [r["name"] for _z, _s, r in alle_regionen()]
    tabelle_je_zelle = [
        SEETYP_TIEFENTABELLE.get(regionsnamen[region_a_je_zelle[k]], TIEFE_JE_SEEGRAD)
        if region_a_je_zelle[k] >= 0 else TIEFE_JE_SEEGRAD
        for k in range(n)
    ]
    tiefe_je_zelle = np.array([tabelle_je_zelle[k].get(min(int(grad[k]), 4), MEERESBODEN_M)
                               for k in range(n)])
    sigma = max(glaettung_m / mpp, 0.5)
    seegrad_tiefe = ndimage.gaussian_filter(tiefe_je_zelle[etikett], sigma)

    # SEEEIS: eine WAHRSCHEINLICHKEIT je Seegrad statt eines harten Schnitts -
    # Nutzer 2026-08-11, Nachbesserung der ersten (harten Grad-1/2-)Fassung:
    # "grade 0 ist 100% eis grade 1 75% chance und grade 2 ist 50% chance.
    # und grade 3 ist 25%." Grad 4+ bleibt 0 % (EISWAHRSCHEINLICHKEIT_JE_
    # SEEGRAD.get() faellt darauf zurueck) - dort verlaufen Seewege.
    #
    # Der Wuerfel faellt JE ZELLE, nicht je Pixel - sonst waere Eis ein
    # Salz-und-Pfeffer-Rauschen statt zusammenhaengender Schollen, die der
    # Zellform folgen. Deterministisch aus dem Kartenseed (eigener
    # XOR-Offset wie ueberall in dieser Datei), reproduzierbar.
    #
    # NOCH OFFEN, vom Nutzer selbst vertagt ("es wird spaeter im spiel nur im
    # winter erscheinen ... aber geh erstmal so rein"): diese Karte bleibt EIN
    # statischer Schnappschuss ohne Jahreszeit - die Saisonalitaet gehoert ins
    # spaetere Zeitmodell (KLIMA_UND_SEE.md §0), nicht in diese Funktion.
    try:
        taiga_index = regionsnamen.index("Taiga")
        ist_taiga_ufer = region_a_je_zelle == taiga_index
        wahrscheinlichkeit_je_zelle = np.array(
            [EISWAHRSCHEINLICHKEIT_JE_SEEGRAD.get(int(g), 0.0) for g in grad])
        zufall_eis = np.random.RandomState(int(seed) ^ 0x5EE2)
        wuerfel_je_zelle = zufall_eis.random_sample(n)
        eis_je_zelle = ist_taiga_ufer & (wuerfel_je_zelle < wahrscheinlichkeit_je_zelle)
        see_eis = np.where(maske, False, eis_je_zelle[etikett])
    except ValueError:
        see_eis = np.zeros((size, size), dtype=bool)

    return {"seegrad": seegrad_roh, "seegrad_tiefe": seegrad_tiefe,
            "ufer_region_a": ufer_region_a, "ufer_region_b": ufer_region_b,
            "see_eis": see_eis}


# =============================================================================
# PARAMETERFELD
# =============================================================================

def regionsgewichte(size, seed, uebergang_m, verzerrung_m,
                    shader_manager=None):
    """
    Wie stark jede der neun Regionen an jedem Pixel mitredet. (9, size, size).

    KEIN GITTER MEHR, SONDERN ABSTANDSGEWICHTE.
    ============================================
    Zuerst wurde je Pixel die Gitterzelle bestimmt (floor) und das Ergebnis
    weichgezeichnet. Das ergibt zwangslaeufig ein Schachbrett mit weichen
    Kanten: die Grenzen bleiben Geraden, egal wie stark man glaettet, weil sie
    aus einer Rasterung stammen. Der Nutzer dazu: "die regionen sollen etwas
    weniger stark rechteckig abgetrennt sein ... keine geraden Grenzen".

    Jetzt gibt es nur noch neun MITTELPUNKTE. Jedes Pixel bekommt zu jedem
    Mittelpunkt ein Gewicht exp(-(d/breite)^2), und alle Parameter werden damit
    gemischt. Damit gibt es prinzipiell keine Grenze mehr - nur noch ein Feld,
    in dem eine Region langsam die Oberhand gewinnt.

    Die Flaechen bleiben im Mittel richtig, weil die Mittelpunkte weiter auf
    dem 3x3-Raster liegen; nur die Zugehoerigkeit ist jetzt fliessend.

    Die Weltkoordinate wird zusaetzlich mit drei Rauschoktaven VERZOGEN. Ohne
    das waeren die Uebergaenge zwar weich, aber immer noch achsenparallel
    angeordnet - mit der Verzerrung greifen die Regionen unregelmaessig
    ineinander, so wie Landschaften es tun.
    """
    mpp = WELT_KM * 1000.0 / size
    achse = (np.arange(size) + 0.5) * mpp - 0.5 * WELT_KM * 1000.0
    WX, WY = np.meshgrid(achse, achse, indexing="xy")

    if verzerrung_m > 1.0:
        stapel = oktavenstapel(size, int(seed) ^ 0x5A17, shader_manager)
        # Drei Oktaven: die grobe verschiebt ganze Regionen, die feineren
        # fransen die Uebergaenge aus.
        vx = 0.55 * stapel[1] + 0.30 * stapel[2] + 0.15 * stapel[4]
        vy = 0.55 * stapel[2] + 0.30 * stapel[3] + 0.15 * stapel[5]
        WX = WX + verzerrung_m * vx
        WY = WY + verzerrung_m * vy

    breite = max(uebergang_m, 1.0)
    gewichte = np.empty((9, size, size), dtype=np.float64)
    for i, (zeile, spalte, _r) in enumerate(alle_regionen()):
        mx = (spalte - 1) * REGION_KM * 1000.0
        my = (zeile - 1) * REGION_KM * 1000.0
        d2 = (WX - mx) ** 2 + (WY - my) ** 2
        gewichte[i] = np.exp(-d2 / (breite * breite))
    gewichte /= np.maximum(gewichte.sum(axis=0, keepdims=True), 1e-12)
    return gewichte


# Wie stark die KLIMAfelder zur fuehrenden Region hin geschaerft werden.
#
# Gemessen am 2026-08-07: das Mittelgebirge hat 12.1 K Klimaspanne INNERHALB
# seines eigenen Gebiets, waehrend seine ganze Hoehenspanne nur 3.8 K liefert.
# Die Regionsmischung reicht so tief, dass sie die Hoehenabnahme voellig
# ueberdeckt - der Nutzer sah genau das: "ist die Hoehentemperatur vorhanden,
# augenscheinlich nicht so deutlich zu erkennen".
#
# In einer reinen Stichprobe (Klimasockel konstant gehalten) trifft die Taiga
# ihre -0.60 K/100 m exakt. Die Hoehenabnahme ist also richtig, sie geht nur
# unter.
#
# Das GELAENDE braucht die breite Mischung - ohne sie stuenden Stufen im
# Relief. Das KLIMA nicht: eine Region darf innen klimatisch einheitlich sein,
# solange der Uebergang nach aussen weich bleibt. Genau das leistet ein
# Exponent auf die Gewichte: 2.5 laesst einen Uebergang von rund einem Drittel
# der bisherigen Breite, aber immer noch ohne Kante.
KLIMA_SCHAERFE = 2.5


def parameterfeld(name, gewichte, schaerfe=1.0):
    """
    Ein Regler als volles Feld - gewichtete Mischung der neun Werte.

    `schaerfe` > 1 zieht die Mischung zur fuehrenden Region hin, ohne eine
    harte Kante zu erzeugen. Siehe KLIMA_SCHAERFE.
    """
    werte = np.array([r[name] for _z, _s, r in alle_regionen()],
                     dtype=np.float64)
    if schaerfe != 1.0:
        gewichte = np.power(np.maximum(gewichte, 0.0), schaerfe)
        gewichte = gewichte / np.maximum(gewichte.sum(axis=0, keepdims=True), 1e-12)
    return np.tensordot(werte, gewichte, axes=(0, 0))


def randabfall(size, seed=0, unruhe_m=1100.0, shader_manager=None):
    """
    Die aeusseren 1.5 km ins Meer ziehen.

    Anteil 0 auf dem Kontinent, 1 am Kartenrand - als glatte Kurve, damit keine
    Stufe an der Kontinentgrenze steht.

    DIE KUESTENLINIE WIRD VERRAUSCHT. Ohne `unruhe_m` ist der Kastenabstand
    exakt, und der Kontinent bekommt eine schnurgerade Kueste mit vier runden
    Ecken - im Bild sofort als Rechteck zu erkennen. Das Rauschen verschiebt die
    Abfallkante um bis zu `unruhe_m` nach innen oder aussen und ergibt Kaps,
    Buchten und vorgelagerte Inseln.
    """
    mpp = WELT_KM * 1000.0 / size
    achse = np.abs((np.arange(size) + 0.5) * mpp - 0.5 * WELT_KM * 1000.0)
    AX, AY = np.meshgrid(achse, achse, indexing="xy")
    abstand = np.maximum(AX, AY)

    if unruhe_m > 1.0:
        stapel = oktavenstapel(size, int(seed) ^ 0x0C0A, shader_manager)
        # Zwei mittlere Oktaven: gross genug fuer Buchten, fein genug fuer
        # eine ausgefranste Linie.
        kante = 0.65 * stapel[3] + 0.35 * stapel[5]
        abstand = abstand + unruhe_m * kante

    innen = 0.5 * KONTINENT_KM * 1000.0
    aussen = 0.5 * WELT_KM * 1000.0
    t = np.clip((abstand - innen) / max(aussen - innen, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)          # smoothstep


# =============================================================================
# KUESTEN-ARCHETYPEN (docs/OFFENE_PUNKTE.md, Nutzer-Vorgabe 2026-08-12)
# =============================================================================
# Drei an realen Kuesten orientierte Auspraegungen je Region, die laengs der
# Kuestenlinie deterministisch wechseln - siehe _kuesten_umformen(). Laeuft
# NACH kuestenform() (die regionsweite Grundform bleibt unveraendert
# bestehen) und VOR der Seegrad-Tiefenzuweisung (die behaelt das letzte
# Wort ueber die eigentliche Meerestiefe - dieser Pass reicht nur bis knapp
# unter die Kuestenlinie).
#
# hoehe_faktor skaliert KUESTENHOEHE_M (450 m), winkel_grad ist der
# Zielwinkel der Klippenfront (nie 90 - siehe MAX_KLIPPENWINKEL_GRAD, "eher
# 80 Grad statt 90" war die ausdrueckliche Vorgabe), kantig=True bricht die
# glatte Anstiegskurve in Facetten auf (Fjord-/Schaerencharakter statt
# einer glatten Wand - Norwegen/Albanien gegen Irland/Portugal, beide vom
# Nutzer namentlich genannt), strand_anteil ist der Anteil INNERHALB der
# eigenen Zone, der als Strand-Luecke ausgespart wird (nie eine
# durchgehende Klippenwand), max_anteil die Obergrenze am gesamten
# Kuestenumfang der Region - garantiert per Quote, nicht nur wahrscheinlich.
KUESTEN_ARCHETYPEN = {
    "Huegelland": (
        dict(name="Moher-Klippen", hoehe_faktor=1.4, winkel_grad=82, kantig=False, strand_anteil=0.10, max_anteil=0.25),
        dict(name="West-Cork-Buchten", hoehe_faktor=0.8, winkel_grad=65, kantig=False, strand_anteil=0.35, max_anteil=0.40),
        dict(name="Dingle-Straende", hoehe_faktor=0.4, winkel_grad=45, kantig=False, strand_anteil=0.55, max_anteil=0.35),
    ),
    "Fjordland": (
        dict(name="Fjordwand", hoehe_faktor=1.8, winkel_grad=78, kantig=True, strand_anteil=0.05, max_anteil=0.30),
        dict(name="Schaerenkueste", hoehe_faktor=0.5, winkel_grad=55, kantig=True, strand_anteil=0.30, max_anteil=0.40),
        dict(name="Fjordbucht", hoehe_faktor=0.3, winkel_grad=40, kantig=False, strand_anteil=0.60, max_anteil=0.30),
    ),
    "Taiga": (
        dict(name="Kola-Steilkueste", hoehe_faktor=1.1, winkel_grad=70, kantig=True, strand_anteil=0.15, max_anteil=0.30),
        dict(name="Weissmeer-Flachkueste", hoehe_faktor=0.25, winkel_grad=30, kantig=False, strand_anteil=0.65, max_anteil=0.40),
        dict(name="Labrador-Buchten", hoehe_faktor=0.7, winkel_grad=55, kantig=False, strand_anteil=0.35, max_anteil=0.30),
    ),
    "Atlantikkueste": (
        dict(name="Bretagne-Klippen", hoehe_faktor=0.9, winkel_grad=72, kantig=False, strand_anteil=0.20, max_anteil=0.25),
        dict(name="Vendee-Straende", hoehe_faktor=0.3, winkel_grad=35, kantig=False, strand_anteil=0.70, max_anteil=0.45),
        dict(name="Ile-de-Re-Watt", hoehe_faktor=0.35, winkel_grad=30, kantig=False, strand_anteil=0.60, max_anteil=0.30),
    ),
    "Alpenland": (
        dict(name="Kotor-Steilfjord", hoehe_faktor=1.7, winkel_grad=80, kantig=False, strand_anteil=0.05, max_anteil=0.25),
        dict(name="Dalmatien-Klippen", hoehe_faktor=1.2, winkel_grad=75, kantig=False, strand_anteil=0.20, max_anteil=0.40),
        dict(name="Alpine-Flussmuendung", hoehe_faktor=0.4, winkel_grad=40, kantig=False, strand_anteil=0.55, max_anteil=0.35),
    ),
    "Mittelgebirge": (
        dict(name="Ruegen-Kreidekueste", hoehe_faktor=0.85, winkel_grad=70, kantig=False, strand_anteil=0.25, max_anteil=0.25),
        dict(name="Ostsee-Flachkueste", hoehe_faktor=0.3, winkel_grad=30, kantig=False, strand_anteil=0.65, max_anteil=0.50),
        dict(name="Foerdenkueste", hoehe_faktor=0.35, winkel_grad=35, kantig=False, strand_anteil=0.60, max_anteil=0.25),
    ),
    "Steppe": (
        dict(name="Algarve-Klippen", hoehe_faktor=1.3, winkel_grad=80, kantig=False, strand_anteil=0.15, max_anteil=0.30),
        dict(name="Costa-Brava-Buchten", hoehe_faktor=0.9, winkel_grad=68, kantig=True, strand_anteil=0.35, max_anteil=0.35),
        dict(name="San-Sebastian-Bucht", hoehe_faktor=0.5, winkel_grad=40, kantig=False, strand_anteil=0.55, max_anteil=0.35),
    ),
    "Mittelmeer": (
        dict(name="Amalfi-Steilkueste", hoehe_faktor=1.6, winkel_grad=82, kantig=False, strand_anteil=0.10, max_anteil=0.30),
        dict(name="Cinque-Terre-Buchten", hoehe_faktor=1.0, winkel_grad=70, kantig=False, strand_anteil=0.30, max_anteil=0.35),
        dict(name="Toskana-Straende", hoehe_faktor=0.4, winkel_grad=35, kantig=False, strand_anteil=0.60, max_anteil=0.35),
    ),
    "Griechische Inseln": (
        dict(name="Santorini-Kliff", hoehe_faktor=1.7, winkel_grad=84, kantig=False, strand_anteil=0.05, max_anteil=0.25),
        dict(name="Kreta-Buchten", hoehe_faktor=0.8, winkel_grad=65, kantig=False, strand_anteil=0.35, max_anteil=0.40),
        dict(name="Kykladen-Strand", hoehe_faktor=0.45, winkel_grad=40, kantig=False, strand_anteil=0.55, max_anteil=0.35),
    ),
}

MAX_KLIPPENWINKEL_GRAD = 85.0   # nie eine reine 90-Grad-Wand
KUESTEN_ARCHETYP_PUNKTE_JE_REGION = 24  # Saatpunkte laengs der Kueste je Region
KUESTEN_BAND_KM = 1.2            # Reichweite der Umformung ab der Kuestenlinie, beidseitig


def _kuesten_rauschen_lokal(zone, seed, sigma):
    """
    Geglaettetes Zufallsfeld NUR im Begrenzungsrahmen von `zone` (plus
    Filterrand) statt auf der ganzen Karte - dieselbe Zahl an sinnvollen
    Werten, ein Bruchteil der Rechenzeit, weil jede Kuesten-Archetyp-Zone
    typischerweise < 1 % der Kartenflaeche einnimmt. Ausserhalb des
    Ausschnitts ueberall 0 - unschaedlich, da nur `ergebnis[zone]` gelesen
    wird und `zone` per Konstruktion komplett im Ausschnitt liegt.
    Deterministisch aus `seed`, aber NICHT bitgleich zu einer Vollkarten-
    Ziehung (andere Ziehreihenfolge) - hier ohne Belang, siehe Aufrufstelle.
    """
    ys, xs = np.nonzero(zone)
    rand_px = int(np.ceil(sigma * 4.0)) + 1
    y0 = max(0, int(ys.min()) - rand_px)
    y1 = min(zone.shape[0], int(ys.max()) + rand_px + 1)
    x0 = max(0, int(xs.min()) - rand_px)
    x1 = min(zone.shape[1], int(xs.max()) + rand_px + 1)

    rng = np.random.RandomState(seed)
    ausschnitt = ndimage.gaussian_filter(rng.rand(y1 - y0, x1 - x0), sigma=sigma)

    ergebnis = np.zeros(zone.shape, dtype=np.float64)
    ergebnis[y0:y1, x0:x1] = ausschnitt
    return ergebnis


def _kuesten_umformen(H, felder, seed, size):
    """
    Formt die Kuestenzone je Region nach KUESTEN_ARCHETYPEN um. Ablauf je
    Region: (1) Saatpunkte nahe der 0-Linie ausduennen, (2) jedem Saatpunkt
    einen Archetyp zuteilen - anspruchsvollster (hoechster hoehe_faktor)
    zuerst, jeweils die zum vorhandenen Rohgelaende best passenden freien
    Punkte plus etwas Seed-Jitter (siehe Docstring oben, "Uebereinstimmung
    mit dem vorhandenen Rohgelaende, aber trotzdem einzigartig je Karte"),
    Quote strikt aus max_anteil - garantiert Vorkommen, nicht nur
    Wahrscheinlichkeit, (3) Naechster-Saatpunkt-Zuordnung je Pixel im Band
    (gleiches Muster wie seegliederung()), (4) Zielhoehe aus dem
    zugeordneten Archetyp-Profil, (5) weich zur bestehenden Hoehe
    hinueberblenden, Blendstaerke faellt zum Bandrand hin auf 0.

    NICHT no-op-sicher bei sehr kleinen Karten (< 8 Kuestenpixel je Region) -
    solche Regionen werden übersprungen, kein Fehler.
    """
    region_map = felder["regionen"]
    mpp = WELT_KM * 1000.0 / size
    band_m = KUESTEN_BAND_KM * 1000.0
    band_px = max(3.0, band_m / mpp)

    # Signierte Distanz zur Kuestenlinie: positiv an Land, negativ auf See,
    # 0 an der Linie selbst (gleiche Bauform wie
    # biome_generator._calculate_beach_probabilities()).
    land = H > 0.0
    dist_land = ndimage.distance_transform_edt(land) * mpp
    dist_see = ndimage.distance_transform_edt(~land) * mpp
    distanz_m = np.where(land, dist_land, -dist_see)

    im_band = np.abs(distanz_m) <= band_m * 1.5
    # DIAGNOSE-FELDER FUERS 2D-ANZEIGE (2026-08-12, Nutzer-Vorgabe: "kann man
    # die Kuestentypen auf der 2D-Karte darstellen"). `kuesten_archetyp`:
    # lokaler Index (0..2) INNERHALB der Region - zusammen mit dem schon
    # vorhandenen `region_map` ergibt das den vollen Archetyp (Name/Werte
    # ueber KUESTEN_ARCHETYPEN[region_name][index] nachschlagbar), -1 = kein
    # Archetyp hier zugewiesen. `kuesten_staerke`: die Blendstaerke selbst
    # (0..1) - direkt die vom Nutzer angefragte "Strahlungstiefe", wie stark
    # dieser Pass an diesem Pixel gegenueber dem Rohgelaende gewichtet wurde.
    felder["kuesten_archetyp"] = np.full(H.shape, -1, dtype=np.int8)
    felder["kuesten_staerke"] = np.zeros(H.shape, dtype=np.float32)
    if not np.any(im_band):
        return H

    ziel_hoehe = H.copy()
    veraendert = np.zeros(H.shape, dtype=bool)
    basis_seed = int(seed) ^ 0x4B57

    for i, (_z, _s, r) in enumerate(alle_regionen()):
        archetypen = KUESTEN_ARCHETYPEN.get(r["name"])
        if not archetypen:
            continue
        region_maske = (region_map == i) & im_band
        if not np.any(region_maske):
            continue

        linien_maske = region_maske & (np.abs(distanz_m) < mpp * 3.0)
        ys, xs = np.nonzero(linien_maske)
        if len(ys) < 8:
            continue

        rng_region = np.random.RandomState(basis_seed + i * 97)
        n_punkte = min(KUESTEN_ARCHETYP_PUNKTE_JE_REGION, len(ys))
        auswahl = rng_region.choice(len(ys), size=n_punkte, replace=False)
        saat_y, saat_x = ys[auswahl], xs[auswahl]

        lokale_hoehe = np.array([
            float(np.mean(H[max(0, y - 2):y + 3, max(0, x - 2):x + 3]))
            for y, x in zip(saat_y, saat_x)])
        hoehe_norm = np.clip(lokale_hoehe / KUESTENHOEHE_M, 0.0, 2.0)

        ziel_anzahl = {a["name"]: max(1, int(round(a["max_anteil"] * n_punkte)))
                       for a in archetypen}

        frei = set(range(n_punkte))
        zuordnung = {}
        for archetyp in sorted(archetypen, key=lambda a: -a["hoehe_faktor"]):
            if not frei:
                break
            jitter = rng_region.normal(0.0, 0.15, size=n_punkte)
            score = -(np.abs(hoehe_norm - archetyp["hoehe_faktor"])) + jitter
            kandidaten = sorted(frei, key=lambda idx: -score[idx])
            n_ziel = min(ziel_anzahl[archetyp["name"]], len(kandidaten))
            gewaehlt = kandidaten[:n_ziel]
            zuordnung.update({idx: archetyp for idx in gewaehlt})
            frei -= set(gewaehlt)
        if frei:
            rest_typ = max(archetypen, key=lambda a: a["max_anteil"])
            zuordnung.update({idx: rest_typ for idx in frei})

        name_zu_index = {a["name"]: k for k, a in enumerate(archetypen)}
        punkte_xy = np.column_stack([saat_x, saat_y]).astype(np.float64)

        idx_pixel_y, idx_pixel_x = np.nonzero(region_maske)
        pixel_xy = np.column_stack([idx_pixel_x, idx_pixel_y]).astype(np.float64)
        from scipy.spatial import cKDTree
        baum = cKDTree(punkte_xy)
        _abst, naechster = baum.query(pixel_xy)

        archetyp_id_karte = np.full(H.shape, -1, dtype=np.int16)
        for pixel_idx, punkt_idx in enumerate(naechster):
            y, x = idx_pixel_y[pixel_idx], idx_pixel_x[pixel_idx]
            archetyp_id_karte[y, x] = name_zu_index[zuordnung[punkt_idx]["name"]]
        felder["kuesten_archetyp"][region_maske] = archetyp_id_karte[region_maske].astype(np.int8)

        # Strand-Luecken: eigener grober, seed-fester Rauschanteil markiert
        # einen Teil jeder Zone als "Strand" statt "volle Klippenhoehe".
        #
        # NUR AUF DEM AUSSCHNITT DER ZONE gerechnet, nicht auf der ganzen
        # Karte (2026-08-12, Performance-Nachbesserung - siehe Docstring
        # oben "Wie laesst sich das optimieren" in der Session-Historie):
        # jede Zone ist ein schmaler Kuestenabschnitt, oft < 1% der
        # Kartenflaeche - ein volles (H,W)-Zufallsfeld + Gauss-Filter je
        # Archetyp war reine Verschwendung. `_kuesten_rauschen_lokal()`
        # generiert und glaettet nur den Begrenzungsrahmen der Zone (plus
        # Rand fuer den Filterkern), bleibt aber bitgleich deterministisch
        # aus demselben Seed - nur eben nicht mehr bitgleich zu einer
        # Vollkarten-Ziehung (andere Ziehreihenfolge), was hier egal ist.
        strand_feld = np.zeros(H.shape, dtype=bool)
        for archetyp in archetypen:
            zone = (archetyp_id_karte == name_zu_index[archetyp["name"]]) & region_maske
            if not np.any(zone) or archetyp["strand_anteil"] <= 0.0:
                continue
            rauschen_seed = basis_seed + i * 97 + name_zu_index[archetyp["name"]] * 13
            rauschen = _kuesten_rauschen_lokal(zone, rauschen_seed, sigma=max(2.0, band_px * 0.5))
            schwelle = np.percentile(rauschen[zone], (1.0 - archetyp["strand_anteil"]) * 100.0)
            strand_feld |= zone & (rauschen >= schwelle)

        for archetyp in archetypen:
            zone = (archetyp_id_karte == name_zu_index[archetyp["name"]]) & region_maske
            if not np.any(zone):
                continue
            winkel = min(archetyp["winkel_grad"], MAX_KLIPPENWINKEL_GRAD)
            # Reichweite aus dem gewuenschten Neigungswinkel abgeleitet, statt
            # aus einer festen, hoehenunabhaengigen Konstante (Nutzerbefund
            # 2026-08-12: gezackte "Mauer" an jeder Kueste in 3D). Fuer
            # `H0*(1-exp(-d/L))` ist die Anfangssteigung bei d=0 genau H0/L -
            # `L = H0/tan(winkel)` setzt also direkt die tatsaechliche Steigung
            # an der Kuestenlinie auf den gewuenschten Winkel, statt einer
            # Formel ohne physikalischen Bezug zur Zielhoehe. Die alte feste
            # Mindestreichweite (15 m) war kleiner als ein Kartenpixel (mpp bei
            # 512px ~42 m) - jede hohe, steile Klippe (KUESTENHOEHE_M *
            # hoehe_faktor bis ~800 m) erreichte dadurch ihre volle Zielhoehe
            # INNERHALB EINES EINZIGEN Pixels (gemessen: 63 m Sprung im Median,
            # bis 530 m). Zusaetzlicher Mindestwert an der Pixelgroesse
            # ausgerichtet, damit auch bei sehr niedriger Aufloesung nie unter
            # rund 2 Pixel Reichweite gefallen wird.
            ziel_hoehe_m = KUESTENHOEHE_M * archetyp["hoehe_faktor"]
            skala_m = max(2.0 * mpp, ziel_hoehe_m / max(np.tan(np.radians(winkel)), 0.05))
            hoehe_faktor_lokal = np.where(
                strand_feld[zone], archetyp["hoehe_faktor"] * 0.25, archetyp["hoehe_faktor"])
            d = distanz_m[zone]
            d_land = np.clip(d, 0.0, None)
            profil = KUESTENHOEHE_M * hoehe_faktor_lokal * (1.0 - np.exp(-d_land / skala_m))
            # Auf der Seeseite sanft zur Kuestenlinie hin auslaufen - die
            # eigentliche Tiefe uebernimmt gleich danach die Seegrad-Tabelle
            # (siehe Aufrufstelle in weltfeld()), dieser Pass soll die
            # Kuestenlinie selbst nur nicht zerreissen.
            profil = np.where(d < 0, d * 0.5, profil)
            if archetyp["kantig"]:
                facetten_seed = basis_seed + i * 97 + name_zu_index[archetyp["name"]] * 13 + 1
                facetten = _kuesten_rauschen_lokal(zone, facetten_seed, sigma=max(1.0, band_px * 0.15))
                profil = profil + (facetten[zone] - 0.5) * 2.0 * KUESTENHOEHE_M * hoehe_faktor_lokal * 0.15
            ziel_hoehe[zone] = profil
            veraendert[zone] = True

    if not np.any(veraendert):
        return H

    # Zonengrenzen (Naechster-Punkt-Zuordnung hat harte Kanten) entschaerfen -
    # gleiche Ueberlegung wie bei den geglaetteten Klimafeldern.
    ziel_hoehe_glatt = ndimage.gaussian_filter(ziel_hoehe, sigma=max(1.0, band_px * 0.2))

    staerke = np.clip(1.0 - np.abs(distanz_m) / band_m, 0.0, 1.0)
    staerke = np.where(im_band & veraendert, staerke * 0.85, 0.0)
    felder["kuesten_staerke"] = staerke.astype(np.float32)

    ergebnis = H * (1.0 - staerke) + ziel_hoehe_glatt * staerke

    # VORZEICHEN DER KUESTENLINIE ERHALTEN (Regressions-Fund 2026-08-12,
    # tests/smoke_test_seegliederung.py: "Seeeis liegt auch auf Land"). Die
    # Glaettung oben laeuft ueber ziel_hoehe HINWEG UEBER DIE KUESTENLINIE -
    # ein steiler, hoher Klippen-Zielwert auf der Landseite kann dadurch in
    # ein direkt benachbartes Seepixel hineinverschmieren und dessen
    # geblendetes Ergebnis ueber 0 heben. seegrad/see_eis/ufer_region wurden
    # bereits VORHER aus der urspruenglichen Land/See-Form berechnet
    # (seegliederung(), weiter oben in weltfeld() aufgerufen) und kennen
    # diese Verschiebung nicht - dieser rein kosmetische Kuesten-Pass darf
    # See nicht zu Land machen oder umgekehrt. Kleiner Sicherheitsabstand
    # (0.5 m) statt eines harten 0-Schnitts, damit keine neue Nahtkante am
    # exakten Nulldurchgang entsteht.
    war_see = H <= 0.0
    war_land = ~war_see
    ergebnis = np.where(war_see & (ergebnis > -0.5), -0.5, ergebnis)
    ergebnis = np.where(war_land & (ergebnis < 0.5), 0.5, ergebnis)

    return ergebnis


# =============================================================================
# DAS GELAENDE
# =============================================================================

def weltfeld(size, seed, punktzahl=200, tiefe_skala_m=1400.0, shader_manager=None):
    """
    Die Hoehenkarte der ganzen Welt, in Metern. Unter 0 ist Meer.

    Rueckgabe: (H, felder) - felder enthaelt jeden Regler als volles Feld,
    damit spaetere Stufen (Fluesse, Taeler) ortsabhaengig arbeiten koennen.
    """
    maske, sdf = kontinentform(size, seed, shader_manager)
    gewichte = voronoi_regionen(maske, seed, punktzahl=punktzahl,
                                shader_manager=shader_manager)

    felder = {}
    KLIMAFELDER = ("temp_mittel_m0", "temp_spanne", "niederschlag_mm", "wind_mittel_ms")
    for name in REGLER:
        # Die Klimafelder schaerfer mischen als die Gelaendefelder - siehe
        # KLIMA_SCHAERFE. Das Relief braucht die breite Ueberblendung, das
        # Klima verliert dadurch seine Hoehenabhaengigkeit.
        felder[name] = parameterfeld(
            name, gewichte,
            KLIMA_SCHAERFE if name in KLIMAFELDER else 1.0)

    # DIE FUEHRENDE REGION JE PIXEL - 0..8 in der Reihenfolge von alle_regionen().
    #
    # Sie wird HIER gebildet und nicht spaeter noch einmal: der Siedlungs-
    # generator, der Terrain-Reiter und der Regional-Reiter brauchen alle
    # dieselbe Zuordnung, und `voronoi_regionen` ein zweites Mal aufzurufen
    # waere eine zweite Wahrheit (SPEZIFIKATION §4.5). Der Regional-Reiter tat
    # bis 2026-08-06 genau das.
    #
    # NACH der Glaettung, nicht davor: geglaettet sind die Gewichte, aus denen
    # das Gelaende entsteht. Eine Regionsgrenze, die anders laeuft als der
    # Parameterwechsel, waere im Bild als Versatz sichtbar.
    #
    # AUCH AUF SEE gueltig. Die Gewichte decken die ganze Karte, nicht nur den
    # Kontinent - das Wasser vor den Griechischen Inseln gehoert zu ihnen. Die
    # Seewege des Siedlungsnetzes brauchen genau das.
    felder["regionen"] = np.argmax(gewichte, axis=0).astype(np.int16)

    # SEEGLIEDERUNG (docs/KLIMA_UND_SEE.md §2) - braucht `maske` (Kontinentform)
    # und `gewichte` (fuer die Uferregionen), beide stehen jetzt. Das Ergebnis
    # (seegrad_tiefe) wird weiter unten anstelle des alten Kuestenschelfs
    # angewandt - siehe "SEEGRAD-SCHELF" dort.
    see = seegliederung(maske, gewichte, seed, punktzahl_land=punktzahl,
                        shader_manager=shader_manager)
    felder["seegrad"] = see["seegrad"]
    felder["seegrad_tiefe"] = see["seegrad_tiefe"]
    felder["ufer_region_a"] = see["ufer_region_a"]
    felder["ufer_region_b"] = see["ufer_region_b"]
    felder["see_eis"] = see["see_eis"]

    stapel = oktavenstapel(size, seed, shader_manager)
    wellen = _wellenlaengen()

    # OKTAVENGEWICHTE JE PIXEL.
    #
    # Eine Oktave zaehlt nur, wenn ihre Wellenlaenge in die Formgroesse der
    # Region passt. Der Uebergang ist weich - eine harte Grenze wuerde beim
    # Wandern ueber eine Regionsgrenze schlagartig eine ganze Oktave zu- oder
    # abschalten, und das saehe man als Kante.
    relief = np.zeros((size, size), dtype=np.float64)
    summe = np.zeros((size, size), dtype=np.float64)
    form = felder["formgroesse_m"]
    rauheit = np.clip(felder["rauheit"], 0.2, 0.9)
    for k in range(OKTAVEN):
        tor = 1.0 / (1.0 + np.exp(-(form - wellen[k]) / (0.35 * wellen[k])))
        gewicht = np.power(rauheit, k) * tor
        relief += gewicht * stapel[k]
        summe += gewicht
    relief /= np.maximum(summe, 1e-9)

    # Auf 0..1 mit FESTER Spreizung, dann die Potenzkurve. <1 hebt an
    # (Hochflaeche), >1 drueckt herunter (weite Ebene mit einzelnen Gipfeln).
    t = np.clip(0.5 + SPREIZUNG * relief, 0.0, 1.0)
    potenz = np.clip(felder["potenz"], 0.2, 4.0)

    # DIE POTENZ UM DEN MEDIAN DREHEN, nicht um die Null.
    #
    # t^p verschiebt den Median von 0.5 auf 0.5^p - die Potenz aenderte damit
    # nicht nur die FORM, sondern auch die mittlere Hoehe. Gemessen: die
    # Atlantikkueste stand bei 73 % Wasser statt 45, das Fjordland bei 0 statt
    # 20, obwohl an hoehe_m nichts falsch war. Zwei Regler, die sich
    # gegenseitig verstellen, sind nicht eichbar.
    #
    # Mit der Rueckverschiebung um 0.5^p - 0.5 bleibt der Median bei 0.5:
    # hoehe_m ist dann wirklich die mittlere Hoehe, und potenz formt nur noch.
    t = np.clip(np.power(t, potenz) - np.power(0.5, potenz) + 0.5, 0.0, 1.0)

    # DIE KUESTENFORM (2026-08-07).
    #
    # Der Nutzer hat je Region beschrieben, WIE das Land ins Meer uebergehen
    # soll: das Mittelgebirge "keine klippe ins meer sondern in das meer
    # abfallen", die Taiga "sanft abfallend mit kleinen inseln", das Fjordland
    # "hohe huegel und tiefe graeben die ins meer gehen", die Steppe "nur
    # teilweise klippen und diese gering", das Huegelland Klippen mit Buchten.
    #
    # Aus `hoehe_m`, `relief_m` und `potenz` allein laesst sich das nicht
    # trennen: sie beschreiben die Region als GANZES, und ein steiles Ufer
    # bekaeme man nur, indem man die ganze Region steiler macht.
    #
    # DIE KUESTENFORM WIRKT NUR NAHE NULL. Ein Faktor auf die Hoehe, der in
    # einem Band um den Meeresspiegel wirkt und weiter weg auf 1 auslaeuft:
    #
    #     H' = H * (1 + (kuestenform - 1) * exp(-(H/band)^2))
    #
    #   > 1  streckt die Hoehen dicht am Wasser: das Land steigt schnell an,
    #        die Uferzone wird schmal - eine KLIPPE.
    #   < 1  staucht sie: flache Boeschungen, breite Watten und einzelne
    #        Kuppen, die knapp ueber Wasser stehen bleiben - INSELN.
    #
    # DAS VORZEICHEN BLEIBT ERHALTEN, also auch die Kuestenlinie und damit der
    # Wasseranteil jeder Region. Die Eichung aus hoehe_m wird nicht entwertet -
    # sonst muesste jede Formaenderung eine neue Eichrunde nach sich ziehen.
    # Angewandt wird sie weiter unten, sobald H steht - siehe "KUESTENFORM
    # ANWENDEN" nach dem Meeresgradienten.

    # hoehe_m ist die MITTE, nicht der Boden - deshalb t - 0.5.
    H = felder["hoehe_m"] + (t - 0.5) * felder["relief_m"]

    # DAS MEER AUSSERHALB DER FORM.
    #
    # Kein Kastengradient mehr: abgesenkt wird nach dem ABSTAND ZUR KUESTE, und
    # die Kueste ist der Rand der Plaetzchenform. Damit folgt der Meeresboden
    # der Landmasse statt dem Kartenrand.
    #
    # Innerhalb der Form bleibt das Regionengelaende unveraendert - die Buchten
    # von Mittelmeer und Griechischen Inseln entstehen weiter aus deren
    # negativer Hoehe und nicht aus der Form.
    aussen = np.maximum(-sdf, 0.0)
    H = H - (0.0 - MEERESBODEN_M) * (1.0 - np.exp(-aussen / tiefe_skala_m))
    # Auf See darf kein Landgipfel stehenbleiben.
    H = np.where(maske, H, np.minimum(H, -1.0 * (aussen / tiefe_skala_m)))

    # KUESTENFORM ANWENDEN (Begruendung weiter oben).
    #
    # NACH dem Meeresgradienten, damit auch der Uebergang ins offene Meer
    # mitgeformt wird, und VOR dem Schelf, damit dessen Mindesttiefe das letzte
    # Wort behaelt - eine flache Kueste soll flach aussehen, aber nicht in
    # zentimetertiefem Wasser enden.
    # EINE POTENZKURVE AUF DIE HOEHE, KEIN ORTSBAND.
    #
    # Zwei Entwuerfe davor waren falsch, und beide auf lehrreiche Weise:
    #
    #  1. Ein HOEHENband exp(-(H/relief)^2) griff genau dort nicht, wo es
    #     gebraucht wurde - das Mittelgebirge steigt binnen 300 m auf 163 m,
    #     und ein 63-m-Band erreichte das mit Gewicht 0.001.
    #  2. Ein ABSTANDSband exp(-(d/500 m)^2) griff, machte die flachen
    #     Regionen aber STEILER statt flacher (Taiga 8.9 -> 12.1 Grad): ein
    #     Faktor, der mit dem Abstand ansteigt, hat eine eigene Ableitung und
    #     fuegt damit selbst Gefaelle hinzu.
    #
    # Richtig ist eine Abbildung der Hoehe auf sich selbst:
    #
    #     H' = KUESTENHOEHE_M * (|H| / KUESTENHOEHE_M) ** p * sign(H),  p = 1/kf
    #
    # also H' = H * u^(p-1) mit u = |H|/KUESTENHOEHE_M, geklemmt auf 1.
    #
    # Warum das genau das Gewuenschte tut: der Hang wird mit p * u^(p-1)
    # skaliert.
    #
    #     p > 1 (kf < 1)  -> nahe der Wasserlinie geht der Faktor gegen 0.
    #                        Das Land steigt zunaechst kaum an: eine breite,
    #                        flache Uferzone, aus der einzelne Kuppen als
    #                        INSELN stehenbleiben.
    #     p < 1 (kf > 1)  -> der Faktor waechst zur Wasserlinie hin. Das Land
    #                        springt sofort an: eine KLIPPE.
    #
    # Ueber KUESTENHOEHE_M hinaus ist u = 1 und der Faktor 1 - das
    # Landesinnere bleibt unberuehrt, die Regionseichung gueltig.
    #
    # DAS VORZEICHEN BLEIBT ERHALTEN, also auch die Kuestenlinie und der
    # Wasseranteil jeder Region.
    kuestenform = np.clip(felder["kuestenform"], 0.2, 3.0)
    p = 1.0 / kuestenform
    # Die Untergrenze verhindert, dass der Faktor bei p < 1 direkt an der
    # Wasserlinie ins Unendliche laeuft: bei u = 0.05 und p = 0.53 sind es
    # Faktor 4.2, eine steile, aber endliche Wand.
    u = np.clip(np.abs(H) / KUESTENHOEHE_M, 0.05, 1.0)
    H = H * np.power(u, p - 1.0)

    # KUESTEN-ARCHETYPEN (docs/OFFENE_PUNKTE.md 5, Nutzer-Vorgabe
    # 2026-08-12) - NACH der regionsweiten Kuestenform, VOR der Seegrad-
    # Tiefe: formt die Kuestenzone lokal nach realen Vorbildern um
    # (Klippenhoehe/-winkel/Kantigkeit/Strandhaeufigkeit wechseln laengs der
    # Kueste), ohne dass die Seegrad-Tabelle gleich danach uebersteuert wird.
    H = _kuesten_umformen(H, felder, seed, size)

    # SEEGRAD-SCHELF (docs/OFFENE_PUNKTE.md 3.2) - eine Mindesttiefe, die mit
    # dem SEEGRAD waechst statt mit dem blossen euklidischen Kuestenabstand.
    #
    # Der Gradient oben wirkt nur AUSSERHALB der Kontinentform. In Buchten und
    # hinter Klippen, die innerhalb der Form unter 0 fallen, blieb das Wasser
    # so flach, wie das Regionenfeld es zufaellig machte. Gemessen am
    # 2026-08-06: 14 % der Wasserflaeche flacher als 1 m, und noch 800 m vor
    # der Kueste waren 39 % flacher als 2 m. Der Nutzer sah das als gruene
    # Flecken im Wasser - bei Tiefen um 0 liegt die Farbe genau auf der Grenze
    # zwischen Meer und Land, und die bilineare Glaettung der Anzeige macht
    # daraus Gruen.
    #
    # URSPRUENGLICH ein Distanzgradient `-t*(1-exp(-d/L))` mit zwei freien
    # Konstanten, dessen Ergebnis an der Aufloesung der Distanztransformation
    # hing - eine Formel statt einer Festlegung (docs/KLIMA_UND_SEE.md §0).
    # Ersetzt durch `felder["seegrad_tiefe"]`: eine TABELLE (0/-40/-90/-150/
    # -200 m je Seegrad, `seegliederung()`), ueber den Zellnachbarschaftsgraphen
    # der See-Voronoi-Gliederung statt einer reinen Abstandsmetrik gebildet.
    # Setzt weiterhin nur eine Untergrenze - `minimum` kann nur vertiefen.
    #
    # ROHES RAUSCHEN VORHER AUF -10 M ANHEBEN (2026-08-12, Nutzer-Vorgabe):
    # `minimum(H, seegrad_tiefe)` LAESST tiefere Rauschwerte unangetastet -
    # steht die rohe Kontinuitaets-/Redistribution-Hoehe an einer Zelle
    # zufaellig bei -180 m, obwohl ihr Seegrad nur -90 m verlangt, gewinnt
    # weiterhin die tiefere Zahl. Das erzeugt unregelmaessige, kleinraeumige
    # Vertiefungen quer durch die geglaettete Seegrad-Zonierung - sichtbar an
    # zusaetzlichen Dreiecken im 3D-Mesh, besonders bei vielen kleinen Inseln
    # dicht beieinander (Nutzer-Beispiel: Griechische Inseln). Indem das rohe
    # H VOR der Seegrad-Anwendung auf hoechstens -10 m angehoben wird, bleibt
    # ausschliesslich die glatte Seegrad-Tabelle als Tiefenquelle bestehen -
    # eine flache Vorstufe, aus der `minimum` mit `felder["seegrad_tiefe"]`
    # praktisch immer den Seegrad-Wert zieht.
    H = np.where(H <= 0.0, np.maximum(H, -10.0), H)

    # WEICHER UEBERGANG STATT HARTEM SCHNITT AN DER KUESTE (Nutzerbefund
    # 2026-08-12, 3D-Ansicht: gezackte "Mauer" ringsum jede Kueste). Ein
    # direktes `minimum(H, seegrad_tiefe)` liess unmittelbar am Ufer die grobe
    # Seegrad-Tabelle (Sprünge 0/-40/-90/-150/-200 m je Grad) gegen den sanften
    # Strand-/Klippen-Verlauf aus `_kuesten_umformen()` gewinnen - gemessen 63 m
    # Hoehensprung (Median, teils >500 m) auf einem einzigen Pixel direkt an
    # der Kuestenlinie. `kuesten_staerke` (oben, 0..0.85) sagt bereits genau,
    # wie stark ein Pixel vom Kuesten-Pass geformt wurde - je naeher an der
    # Kueste, desto hoeher. Denselben Wert hier als Blendgewicht wiederverwendet
    # (kein neues Feld, keine neue Distanztransformation): nahe der Kueste
    # bleibt ueberwiegend die glatte Kuestenform bestehen, mit wachsendem
    # Abstand uebernimmt zunehmend die Seegrad-Tiefe - wie zuvor, nur nicht
    # mehr als harter Schnitt an der Nulllinie.
    seegrad_ziel = np.minimum(H, felder["seegrad_tiefe"])
    kuesten_schutz = felder["kuesten_staerke"]
    H = np.where(H <= 0.0, H * kuesten_schutz + seegrad_ziel * (1.0 - kuesten_schutz), H)

    return H.astype(np.float64), felder


def fingerabdruck(H, size):
    """
    Fuenf Zahlen, die den Charakter einer Landschaft messbar machen.

    "Sieht gut aus" ist nicht pruefbar, das hier schon - und damit wird eine
    spaetere Verschlechterung messbar statt nur gefuehlt
    (docs/INTEGRATIONSPLAN.md, Abschnitt 7).
    """
    mpp = WELT_KM * 1000.0 / size
    land = H > 0.0
    if land.sum() < 16:
        return dict(relief_m=0.0, median_hang=0.0, ueber30=0.0,
                    landanteil=0.0, hoehe_median=float(np.median(H)))
    dy, dx = np.gradient(H, mpp)
    hang = np.rad2deg(np.arctan(np.hypot(dx, dy)))
    return dict(
        relief_m=float(H[land].max() - H[land].min()),
        median_hang=float(np.median(hang[land])),
        ueber30=float(100.0 * np.mean(hang[land] > 30.0)),
        landanteil=float(100.0 * land.mean()),
        hoehe_median=float(np.median(H[land])),
    )


def regionskern(feld, zeile, spalte, size, anteil=0.55):
    """
    Der innere Teil einer Region, ohne den Rand.

    Gemessen wird bewusst NICHT der volle 4-km-Kasten: dort blendet die
    Nachbarregion bereits herein, und das Alpenland macht dann das Relief der
    Taiga kaputt. `anteil` gibt an, wieviel der Kantenlaenge der Kern hat.
    """
    kante = REGION_KM * 1000.0 / (WELT_KM * 1000.0 / size)
    # MINUS, nicht plus: `zeile` 0 ist Norden, und Norden ist die HOHE
    # Zeilennummer (siehe die Begruendung in voronoi_regionen). Mit einem Plus
    # haette diese Pruefung ab dem 2026-08-06 die jeweils gegenueberliegende
    # Region gemessen und den Fjordland-Wasseranteil am Mittelmeer geprueft.
    mitte_y = 0.5 * size - (zeile - 1) * kante
    mitte_x = 0.5 * size + (spalte - 1) * kante
    halb = 0.5 * anteil * kante
    y0, y1 = int(mitte_y - halb), int(mitte_y + halb)
    x0, x1 = int(mitte_x - halb), int(mitte_x + halb)
    return feld[max(y0, 0):y1, max(x0, 0):x1]


def pruefe_regionen(H, size):
    """Fingerabdruck je Region gegen den Sollwasseranteil."""
    zeilen = []
    for zeile, spalte, r in alle_regionen():
        kern = regionskern(H, zeile, spalte, size)
        f = fingerabdruck(kern, size)
        wasser = 100.0 - f["landanteil"]
        zeilen.append(dict(name=r["name"], soll=r["wasser_soll"], ist=wasser,
                           relief=f["relief_m"], hang=f["median_hang"],
                           ueber30=f["ueber30"], hoehe=f["hoehe_median"]))
    return zeilen
