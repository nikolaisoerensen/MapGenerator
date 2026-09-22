"""
Path: core/terrain_weltkarte.py

DIE REGIONENWELT - Rechenkern, ohne Fenster.

Neun Regionen in einem 3x3-Gitter, jede 4 x 4 km, Kontinent also 12 x 12 km,
Welt 15 x 15 km mit Wasser ringsum (docs/archiv/2026-08-04_INTEGRATIONSPLAN.md).

Anordnung wie die Gegenden wirklich liegen, Spalte West -> Ost:

    Nord    Clonagh (Kelten)   Skerrheim (Wikinger)   Morobora (Slawen)
    Mitte   Estrande        Nevadin              Nebelrode
    Sued    Samarcia (Andalus)      Macchia (Italien)   Griech. Inseln


DER KERNGEDANKE: KEIN MOSAIK, SONDERN EIN PARAMETERFELD.

Jeder Regler wird als 3x3-Gitter angegeben und auf volle Aufloesung gebracht -
mit verzerrten Grenzen und weichem Uebergang. Danach gibt es an JEDEM Pixel
einen vollstaendigen Parametersatz, und die Regionen laufen von selbst
ineinander. Es gibt keine Nahtlogik, weil es keine Naht gibt.


WARUM EIN OKTAVENSTAPEL. Formgroesse und Rauheit sollen sich raeumlich aendern:
im Nevadin grosse Massive, im Clonagh kleine Wellen. Mit einem einzigen
Rauschaufruf geht das nicht - dessen Frequenz ist fuer die ganze Karte
dieselbe. Deshalb werden die Oktaven EINZELN erzeugt und je Pixel verschieden
gewichtet. Eine Region mit grosser Formgroesse bekommt nur die groben Oktaven,
eine mit kleiner auch die feinen.


DIE KUESTE ENTSTEHT AUS DEMSELBEN FELD. Kein aufgesetzter Inselgradient:
Estrande, Macchia und Thalassia haben NEGATIVE Basishoehen,
also liegt die Kueste dort, wo Basis plus Relief unter 0 faellt. Buchten und
Archipel entstehen an den richtigen Stellen, statt dass eine Form sie
aufdrueckt. Nur ringsum zieht ein Randabfall die aeusseren 1.5 km ins Meer.
"""

import logging
import tomllib
from pathlib import Path

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
                          # 2026-08-07 von 12.8 erhoeht: die Estrande
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

# WARUM DIE WELT GEWACHSEN IST (2026-08-05). Macchia und Thalassia
# verlieren 40 bzw. 65 Prozent ihrer Flaeche ans Wasser, die Estrande 45.
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

# WIE VIEL FEINER ALS DIE FORMGROESSE DAS RAUSCHEN NOCH MITREDEN DARF.
#
# Untere Kante des zweiseitigen Oktaventors (siehe `weltfeld()`):
# Wellenlaengen unter formgroesse_m / FEINHEIT_TEILER werden gedaempft.
# Bei 8 heisst das: eine Region mit 3800 m Massiven laesst Wellen bis
# rund 475 m voll durch und daempft darunter, eine mit 1100 m Huegeln
# entsprechend bis 138 m. Das begrenzt das VERHAELTNIS von feinster zu
# groebster Form, nicht eine absolute Groesse.
FEINHEIT_TEILER = 8.0

MEERESBODEN_M = -200.0    # wohin der Randabfall zieht

# Bis zu welcher HOEHE die Kuestenform wirkt (Meter, beiderseits der
# Wasserlinie). Darueber bleibt das Gelaende unveraendert, damit das
# Landesinnere und die Regionseichung unberuehrt bleiben.
#
# 2026-08-07 von 150 auf 450: gemessen stehen Morobora und Nebelrode schon
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
# Wasser als der Suedlappen - das Clonagh sprang von 6 auf 26 Prozent.
#
# Geeicht wurde NUR `hoehe_m`, gemittelt ueber FUENF Seeds, damit die Werte
# nicht auf eine einzelne Kontinentform passen. `relief_m` blieb bewusst
# unberuehrt: der gemessene Hang ist zu einem grossen Teil gar nicht der
# eigene. Ein als Morobora gefuehrtes Pixel traegt im Mittel 33 % FREMDES
# Gewicht und damit 372 m Relief statt der eingetragenen 138 (Estrande
# +181 %). Bei hoher Reinheit (Gewicht > 0.95) trifft die Morobora ihren
# Sollhang exakt - 3.4 gegen 3.0. Das Relief auf die Mischung zu eichen
# haette es auf 30 m gedrueckt: eine Region ohne Charakter, nur damit eine
# Zahl stimmt, die etwas anderes misst.
#
# ZWEI SEEDS WAREN ZU WENIG (nachkorrigiert am 2026-08-06). Der erste Durchgang
# stellte Skerrheim auf 196.6 m; ueber fuenf Seeds gemessen waren das 10
# Prozentpunkte zu wenig Wasser, richtig sind 113.1 m. Die Streuung EINER
# Region ueber Seeds betraegt bis zu 37 Prozentpunkte Wasseranteil und 11 Grad
# Hang - sie sitzt je nach Kontinentform auf einem anderen Stueck Land und hat
# andere Nachbarn. Weniger als vier Seeds eichen auf eine Form, nicht auf die
# Regel.

# DREI KULTURNAMEN GEAENDERT AM 2026-08-06 (docs/spezifikation/14_SIEDLUNGEN.md):
#
#   Nevadin           "-"          -> Alemannen    hatte gar keine Kultur und
#                                                    bekam damit keine Siedlungen
#   Nebelrode       Franken      -> Sachsen      war doppelt mit der
#                                                    Estrande belegt
#   Thalassia  Phoenizier   -> Byzantiner   die phoenizischen Stadt-
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
# ist derselbe wie beim Relief: ein Pixel, das als Samarcia gefuehrt wird,
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
# auf - die Morobora-Jahresspanne lief von 29 ueber 41.9 auf 56.2.
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
# normieren macht die Eichung ueberfluessig; die Morobora stand zwischenzeitlich
# auf 79 mm, damit 600 ankamen.
# NIEDERSCHLAG_ZIEL und KLIMA_ZIEL SIND HIER NICHT MEHR ALS LITERAL
# EINGETRAGEN (Ticket #29, 2026-09-19). Sie deckten sich schon vorher
# ZAHLENGLEICH mit REGIONEN[...]["niederschlag_mm"] bzw. den Paaren
# (temp_mittel_m0, temp_spanne) - siehe die Begruendung oben ("Die
# Eingabewerte sind dadurch nicht mehr als 'Klima von Bergen' lesbar") und
# tests/smoke_test_weather_temperature_direktnormierung.py ("Die
# Eingabewerte in REGIONEN sind seither die lesbaren KLIMA_ZIEL-Werte
# selbst."). Ein zweites Mal denselben Wert im Code stehen zu haben war
# genau die Art von Duplikat, die Ticket #29 beheben sollte - deshalb
# werden beide jetzt AUS REGIONEN abgeleitet, direkt nachdem REGIONEN aus
# core/daten/regionen.toml geladen ist (siehe unten, kurz vor REGLER).

# KLIMA JE REGION (2026-08-07). Drei Werte, alle auf MEERESHOEHE:
#
#   temp_mittel_m0    Jahresmittel in Grad, auf 0 m zurueckgerechnet
#   temp_spanne       Jahresspanne (Juli minus Januar) in Kelvin
#   niederschlag_mm   Jahresniederschlag
#
# Abgeleitet aus Bezugsorten, die der Nutzer vorgegeben hat: Cork, Bergen,
# Wologda, La Rochelle, Chur, Bamberg, Madrid, Rom, Iraklio. Die Rueckrechnung
# auf Meereshoehe benutzt 0.6 K je 100 m; Herleitung in docs/spezifikation/13_KLIMA_UND_BIOME.md.
#
# WARUM MEERESHOEHE UND NICHT REGIONSHOEHE. Eine Regionshoehe ist ein
# GEEICHTER Wert - `hoehe_m` wurde in dieser Woche zweimal nachgezogen. Waere
# das Klima darauf bezogen, waere es stillschweigend mitgewandert. Meereshoehe
# ist der einzige Bezug, der nicht mitwandert.
#
# Die Jahresspanne traegt den Unterschied zwischen See- und Kontinentalklima
# von selbst: Clonagh 9.5 K, Morobora 29.0 K. Niemand muss das modellieren.

# FLAECHENEICHUNG, eingeregelt am 2026-08-24 mit tools/flaeche_eichen.py.
#
# `flaeche_soll` steuert, wieviel GRUNDflaeche eine Region bekommt. Sie ist
# noetig, weil Grundflaeche und NUTZWERT nicht linear zusammenhaengen:
# Skerrheim verliert erst rund ein Drittel ans Wasser und dann die Haelfte
# des Rests an zu steile Haenge, die Griechischen Inseln zwei Drittel ans
# Wasser. Ohne Ausgleich haetten sie ein Vielfaches weniger besiedelbaren
# Raum als die Samarcia.
#
# ES IST EIN NULLSUMMENSPIEL - die Werte verteilen den Kontinent um, sie
# vergroessern ihn nicht. Wer waechst, nimmt allen anderen etwas weg.
# Deshalb von Hand kaum einzustellen: jede Aenderung verschiebt alle
# anderen mit. `tools/flaeche_eichen.py` regelt sie in wenigen Runden ein
# (gemessen: groesste Abweichung 0.261 -> 0.102 in drei Runden).
#
# DIE ZIELWERTE selbst stehen NICHT hier, sondern in
# tests/smoke_test_regionen_fairness.py - sie sind eine Entscheidung ueber
# das Zielbild (Nevadin und Skerrheim 0.80, alle anderen 1.00), nicht
# ueber die Rechnung.
# DIE WERTE SELBST STEHEN NICHT MEHR HIER (Ticket #29, 2026-09-19), SONDERN
# IN core/daten/regionen.toml.
#
# Vorher standen REGIONEN und KUESTEN_ARCHETYPEN als Python-Literale in
# dieser Datei: nicht diffbar (jede Aenderung verschwand im Rauschen eines
# Python-Diffs), nicht ohne Import lesbar, nicht ohne Codeaenderung
# verstellbar. `_lade_regionsdaten()` liest jetzt stattdessen die
# TOML-Datei und baut daraus dasselbe 3x3-Gitter - byte- und wertgleich zu
# vorher, siehe tests/smoke_test_regionen_welt.py (unveraendertes Ergebnis)
# und tests/smoke_test_regionsdaten_vollstaendig.py (Vollstaendigkeit aller
# neun Saetze).
#
# KEIN STILLER RUECKFALL: fehlt die Datei, fehlt ein Pflichtfeld, oder
# fehlt eine der neun Gitterpositionen, bricht der Import mit einer klaren
# Fehlermeldung ab - CLAUDE.md nennt genau das die teuerste wiederkehrende
# Fehlerklasse dieses Projekts (stille Rueckfaelle nach Pfad-/Werteumzuegen).
_REGIONENDATEI = Path(__file__).resolve().parent / "daten" / "regionen.toml"

_REGION_PFLICHTFELDER = (
    "name", "farbe", "volk", "bemerkung", "hoehe_m", "relief_m",
    "formgroesse_m", "rauheit", "potenz", "wasser_soll", "flaeche_soll",
    "kuestenform", "temp_mittel_m0", "temp_spanne", "niederschlag_mm",
    "wind_mittel_ms", "hang_trockenheit", "talform")

_ARCHETYP_PFLICHTFELDER = (
    "name", "hoehe_faktor", "winkel_grad", "kantig", "strand_anteil",
    "max_anteil", "reichweite_km")


def _lade_regionsdaten(pfad=None):
    """
    Laedt REGIONEN (3x3-Gitter) und KUESTEN_ARCHETYPEN (je Regionsname drei
    Eintraege) aus core/daten/regionen.toml.

    Prueft dabei genau das, was tests/smoke_test_regionsdaten_vollstaendig.py
    von aussen noch einmal nachprueft: neun Regionen, jede mit allen
    Pflichtfeldern und genau drei Kuestenarchetypen, jede Gitterposition
    (zeile, spalte) genau einmal belegt. Fehlt etwas, gibt es KEINEN
    Vorgabewert, der das Loch still fuellt - der Import bricht ab.
    """
    pfad = Path(pfad) if pfad is not None else _REGIONENDATEI
    if not pfad.exists():
        raise RuntimeError(
            "Regionsparameter-Datei fehlt: %s. Ohne sie kann "
            "core.terrain_weltkarte keine Welt rechnen (Ticket #29)." % pfad)
    with open(pfad, "rb") as datei:
        rohdaten = tomllib.load(datei)

    eintraege = rohdaten.get("region", [])
    if len(eintraege) != 9:
        raise RuntimeError(
            "%s enthaelt %d Regionen, erwartet werden genau neun "
            "(3x3-Gitter)." % (pfad, len(eintraege)))

    gitter = [[None, None, None] for _ in range(3)]
    kuesten_archetypen = {}
    namen = set()
    for eintrag in eintraege:
        name = eintrag.get("name", "<ohne Namen>")
        fehlend = [f for f in _REGION_PFLICHTFELDER if f not in eintrag]
        if fehlend:
            raise RuntimeError(
                "Region %r in %s fehlt Feld(er): %s"
                % (name, pfad, fehlend))
        zeile, spalte = eintrag.get("zeile"), eintrag.get("spalte")
        if zeile not in (0, 1, 2) or spalte not in (0, 1, 2):
            raise RuntimeError(
                "Region %r in %s hat keine gueltige Gitterposition "
                "(zeile=%r, spalte=%r)." % (name, pfad, zeile, spalte))
        if gitter[zeile][spalte] is not None:
            raise RuntimeError(
                "Gitterposition (%d, %d) in %s doppelt belegt (%r und %r)."
                % (zeile, spalte, pfad, gitter[zeile][spalte]["name"], name))

        archetypen = eintrag.get("kuesten_archetypen", [])
        if len(archetypen) != 3:
            raise RuntimeError(
                "Region %r in %s hat %d Kuestenarchetypen, erwartet werden "
                "drei." % (name, pfad, len(archetypen)))
        for archetyp in archetypen:
            fehlend_a = [f for f in _ARCHETYP_PFLICHTFELDER
                        if f not in archetyp]
            if fehlend_a:
                raise RuntimeError(
                    "Ein Kuestenarchetyp der Region %r in %s fehlt "
                    "Feld(er): %s" % (name, pfad, fehlend_a))

        gitter[zeile][spalte] = {k: eintrag[k] for k in _REGION_PFLICHTFELDER}
        kuesten_archetypen[name] = tuple(dict(a) for a in archetypen)
        namen.add(name)

    if len(namen) != 9:
        raise RuntimeError("%s enthaelt doppelte Regionsnamen." % pfad)
    for zeile in gitter:
        for zelle in zeile:
            if zelle is None:
                raise RuntimeError(
                    "%s deckt nicht alle neun Gitterpositionen ab." % pfad)

    return gitter, kuesten_archetypen


REGIONEN, KUESTEN_ARCHETYPEN = _lade_regionsdaten()

# Abgeleitet aus REGIONEN, nicht verdoppelt - siehe die Begruendung weiter
# oben bei der (jetzt entfernten) literalen Fassung dieser beiden Dicts.
NIEDERSCHLAG_ZIEL = {r["name"]: float(r["niederschlag_mm"])
                     for zeile in REGIONEN for r in zeile}
KLIMA_ZIEL = {r["name"]: (float(r["temp_mittel_m0"]), float(r["temp_spanne"]))
             for zeile in REGIONEN for r in zeile}

REGLER = ("hoehe_m", "relief_m", "formgroesse_m", "rauheit",
          "potenz", "kuestenform",
          "temp_mittel_m0", "temp_spanne", "niederschlag_mm", "wind_mittel_ms",
          "hang_trockenheit", "talform")

# DIE TALFORM JE REGION (Nutzervorgabe 2026-08-24):
#
#   *"ja flusstypen sollte es geben, nach region. zB alpen eher V.
#   Skerrheim U und im Atlantik irgendwas dazwischen zB."*
#
# `talform` ist der Exponent der Querschnittskurve in
# `taeler_eingraben()`: `profil = (1 - exp(-abstand/breite)) ** talform`.
#
#   klein (0.8-1.1)  V-Tal  - das Profil steigt sofort, die Sohle ist
#                             schmal, die Flanken stehen steil. Fluvial
#                             eingeschnitten: Nevadin, Macchia.
#   mittel (1.3-1.5) dazwischen: Clonagh, Estrande.
#   gross (2.0-2.6)  U-Tal  - das Profil steigt traege, die Sohle ist
#                             breit und flach. Glazial ausgeschuerft:
#                             Skerrheim, Morobora.
#
# Der bisherige Festwert war 1.3 fuer alle Regionen; er bleibt der
# Rueckfall, wenn das Feld fehlt.

# WIE STARK DER SUEDHANG AUSTROCKNET (Nutzervorgabe 2026-08-24):
#
#   *"Skerrheim, voronoi mit viel suedhang ist etwas trockener als
#   fjordland nordhang, dann Samarcia suedhang total trocken, nordhang etwas
#   feuchter."*
#
# `hang_trockenheit` je Region ist der Anteil, um den ein voller Suedhang
# trockener wird als die Ebene - ein voller Nordhang wird um denselben
# Betrag feuchter. Bei 0.45 (Samarcia) schwankt der Niederschlag also
# zwischen 55 % und 145 % des Regionswerts, bei 0.10 (Skerrheim) nur
# zwischen 90 % und 110 %.
#
# DIE MODULATION IST RELATIV, nicht absolut: ein Suedhang in der Samarcia
# bleibt trockener als ein Suedhang im Skerrheim, weil beide von ihrem
# eigenen Regionswert ausgehen (471 mm gegen 1967 mm).
#
# Angewandt wird sie am Ende von weltfeld(), wo das Gelaende steht - die
# Parameterfelder entstehen vorher und kennen noch keine Haenge.

# Ab welcher Neigung die Hangausrichtung voll wirkt. Darunter waechst sie
# linear an: eine Ebene hat keine Exposition, und ohne diese Kopplung
# bekaeme flaches Land zufaellige Feuchteunterschiede aus dem
# Rundungsrauschen des Gradienten.
HANG_VOLL_GRAD = 12.0


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


def oktavenstapel(size, seed, shader_manager=None, mpp=None, hoehe=None):
    """
    Die einzelnen Rauschoktaven als (OKTAVEN, size, size).

    Getrennt erzeugt, damit sie je Pixel verschieden gewichtet werden koennen -
    das ist die Voraussetzung dafuer, dass Formgroesse und Rauheit sich
    raeumlich aendern duerfen.

    Wellenlaenge der Oktave k: GRUNDFORM_M / 2^k, in METERN. Damit haengt das
    Ergebnis an der Wirklichkeit und nicht an der Pixelzahl (SPEZIFIKATION §10).

    `mpp` ueberschreibt die aus WELT_KM abgeleitete Pixelgroesse. Gebraucht
    von Pruefwerkzeugen, die einen ANDEREN Weltausschnitt betrachten als die
    Karte (tools/inseltest.py zeigt Inseln von 150 m bis 10 km). Ohne diesen
    Weg muesste dort ein zweiter Rauschgenerator gebaut werden - und zwei
    Rauschquellen waeren eine zweite Wahrheit (SPEZIFIKATION §4.5): das
    Testgelaende saehe anders aus als das Programm, und niemand wuesste,
    welches der beiden man gerade beurteilt.
    """
    # RECHTECKIG MOEGLICH (2026-08-26). `hoehe` ueberschreibt die
    # Zeilenzahl; ohne sie bleibt es quadratisch wie bisher. Gebraucht von
    # der Regionsansicht (docs/archiv/2026-08-25_AUFRAEUMPLAN.md 4.10), die eine EINZELNE
    # Region als Rechteck zeigt.
    #
    # `noise2array(x, y)` nimmt seit jeher ZWEI Achsen - hier wurde nur
    # zweimal dieselbe uebergeben.
    hoehe = int(size if hoehe is None else hoehe)
    schluessel = (size, hoehe, seed, shader_manager is not None, mpp)
    if schluessel in _STAPEL_CACHE:
        return _STAPEL_CACHE[schluessel]

    if mpp is None:
        mpp = WELT_KM * 1000.0 / size
    rechteckig = hoehe != size
    if rechteckig and shader_manager is not None and shader_manager.gpu_available:
        # LAUT MELDEN statt still auf CPU auszuweichen (CLAUDE.md): der
        # Rauschshader kennt nur quadratische Felder.
        logging.getLogger(__name__).info(
            "Oktavenstapel %dx%d ist rechteckig - der GPU-Pfad kann nur "
            "quadratisch, es wird auf CPU gerechnet", size, hoehe)
    stapel = np.zeros((OKTAVEN, hoehe, size), dtype=np.float32)
    for k in range(OKTAVEN):
        wellenlaenge = GRUNDFORM_M / (2.0 ** k)
        frequenz = mpp / wellenlaenge
        # Je Oktave ein eigener Seed - sonst sind die Oktaven verschobene
        # Fassungen desselben Musters und die Landschaft bekommt Streifen.
        okt_seed = int(seed) + k * 7919
        if (not rechteckig and shader_manager is not None
                and shader_manager.gpu_available):
            stapel[k] = shader_manager.process_noise_generation(
                size=size, octaves=1, frequency=frequenz, persistence=0.5,
                lacunarity=2.0, seed=okt_seed)
        else:
            from opensimplex import OpenSimplex
            gen = OpenSimplex(seed=okt_seed)
            achse_x = np.arange(size, dtype=np.float64) * frequenz
            achse_y = np.arange(hoehe, dtype=np.float64) * frequenz
            stapel[k] = gen.noise2array(achse_x, achse_y).astype(np.float32)

    _STAPEL_CACHE[schluessel] = stapel
    return stapel


def oktavengewicht(k, formgroesse, rauheit):
    """
    Gewicht der Oktave `k` - DIE EINE FASSUNG DIESER FORMEL.

    `formgroesse` und `rauheit` duerfen Skalare ODER volle Felder sein; die
    Rueckgabe hat entsprechend dieselbe Gestalt.

    WARUM ES DIESE FUNKTION GIBT (2026-08-26): sie stand zweimal im Code -
    einmal in `weltfeld()` (mit ortsabhaengigen Feldern) und einmal in
    `regionsfeld()` (mit einem Zahlensatz). Zwei Fassungen derselben Formel
    sind zwei Wahrheiten (SPEZIFIKATION 4.5): aendert jemand das Tor in
    `weltfeld()` und vergisst die Vorschau, stellt der Nutzer seine Regionen
    an einem Gelaende ein, das es auf der Karte nicht gibt - ohne
    Fehlermeldung, nur "irgendwie anders".

    DAS TOR IST ZWEISEITIG. Oben daempft es Oktaven, die GROESSER sind als
    die Formgroesse. Unten daempft es alles unter `formgroesse /
    FEINHEIT_TEILER` - ohne das trug das Nevadin 250 m Amplitude in
    Formen unter 375 m Breite und zeigte hunderte Nadelspitzen statt zwei
    Dutzend Gipfeln (gemessen 2026-08-25).
    """
    welle = GRUNDFORM_M / (2.0 ** k)
    form = np.asarray(formgroesse, dtype=np.float64)
    tor = 1.0 / (1.0 + np.exp(-(form - welle) / (0.35 * welle)))
    fein_grenze = form / FEINHEIT_TEILER
    tor_fein = 1.0 / (1.0 + np.exp(-(welle - fein_grenze)
                                   / (0.35 * fein_grenze)))
    return np.power(np.asarray(rauheit, dtype=np.float64), k) * tor * tor_fein


def _wellenlaengen():
    return np.array([GRUNDFORM_M / (2.0 ** k) for k in range(OKTAVEN)])


# =============================================================================
# DIE KONTINENTFORM - "PLAETZCHEN"
# =============================================================================

# DER FORMREGLER DES KONTINENTS (Nutzerentwurf 2026-08-26)
#
# *"zB laesst sich hier ein eher runder kontinent erstellen oder aber einer
# mit vielen armen (also die grundformen als slider, links rund, mitte
# laenglich, rechts mit vielen auslaeufern und dazwischen wandelt sich die
# form dann etwas)."*
#
# Der Kontinent entsteht aus einer Zentralscheibe plus Lappen, weich
# vereinigt, minus zwei bis drei abgezogenen Buchten. Der Regler
# interpoliert genau diese Groessen:
#
#   RUND   grosser Kern, Lappen nah und gross, weiche Vereinigung
#          -> alles verschmilzt zu einem Klecks mit gekraeuseltem Rand
#   ARME   kleiner Kern, Lappen fern und klein, harte Vereinigung
#          -> die Lappen bleiben als eigene Arme stehen
#
# LAENGLICH IST KEINE INTERPOLATION DERSELBEN GROESSEN, sondern eine
# zusaetzliche Zutat: die Lappenmitten werden entlang einer Achse gestaucht.
# Das laesst sich nicht zwischen "rund" und "Arme" hineinmischen - wer das
# beim Bauen uebersieht, bekommt in der Reglermitte einfach etwas
# Halbrundes statt einer laenglichen Landmasse.
# DIE ABGEZOGENEN BUCHTEN GEHOEREN MIT IN DIE INTERPOLATION.
#
# Ein erster Entwurf liess sie fest (2-3 Stueck, Radius 0.30-0.55). Gemessen
# war die Stellung "rund" damit die UNRUNDESTE von allen: die Buchten
# schnitten tief in eine Landmasse, die durch die nahen Lappen ohnehin
# kompakter geworden war, und die Flaecheneichung blies sie zusaetzlich auf.
# Eine runde Landmasse hat wenige und flache Buchten, eine zerlappte viele
# und tiefe - das ist Teil derselben Gestalt.
KONTINENT_RUND = dict(kern=0.55, nah=0.30, fern=0.55,
                      r_min=0.45, r_max=0.70, weich=0.130,
                      buchten_min=0, buchten_max=1,
                      bucht_min=0.18, bucht_max=0.30)
KONTINENT_ARME = dict(kern=0.20, nah=0.62, fern=0.95,
                      r_min=0.24, r_max=0.42, weich=0.032,
                      buchten_min=3, buchten_max=5,
                      bucht_min=0.32, bucht_max=0.58)
# Die Werte, mit denen der Kontinent bis zum 2026-08-26 gebaut wurde. Sie
# gelten weiterhin, wenn KEIN Formregler uebergeben wird - jede Eichung
# haengt daran (Regionsflaechen, Wasseranteile, smoke_test_regionen_welt).
KONTINENT_VORGABE = dict(kern=0.34, nah=0.55, fern=0.95,
                         r_min=0.26, r_max=0.52, weich=0.055,
                         buchten_min=2, buchten_max=4,
                         bucht_min=0.30, bucht_max=0.55)
# Wie stark die Querachse in der Reglermitte gestaucht wird.
KONTINENT_STRECKUNG = 0.55


def _kontinent_gestalt(form):
    """Die sechs Formgroessen zu einem Reglerwert 0..1, plus Streckung."""
    if form is None:
        return dict(KONTINENT_VORGABE), 0.0
    t = float(np.clip(form, 0.0, 1.0))
    gestalt = {k: (1.0 - t) * KONTINENT_RUND[k] + t * KONTINENT_ARME[k]
               for k in KONTINENT_RUND}
    # Die Buchtenzahl ist eine ANZAHL, kein Mass - sie muss ganzzahlig
    # werden, und zwar bevor sie in rng.integers() geht.
    gestalt["buchten_min"] = int(round(gestalt["buchten_min"]))
    gestalt["buchten_max"] = max(int(round(gestalt["buchten_max"])),
                                 gestalt["buchten_min"] + 1)
    # Streckung mit Maximum in der Reglermitte: 4t(1-t) ist an beiden Enden
    # 0 und bei 0.5 gleich 1.
    return gestalt, KONTINENT_STRECKUNG * 4.0 * t * (1.0 - t)


def kontinentform(size, seed, shader_manager=None, unruhe_m=700.0, form=None):
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
    gestalt, streckung = _kontinent_gestalt(form)
    felder = []
    for k in range(SCHEIBEN):
        if k == 0:
            mx, my, radius = 0.0, 0.0, gestalt["kern"] * halb
        else:
            winkel = 2.0 * np.pi * (k - 1) / (SCHEIBEN - 1) \
                + rng.uniform(-0.55, 0.55)
            weite = halb * rng.uniform(gestalt["nah"], gestalt["fern"])
            mx, my = weite * np.cos(winkel), weite * np.sin(winkel)
            # LAENGLICH: die Querachse stauchen. Die Zufallszahlen werden
            # dabei in DERSELBEN Reihenfolge gezogen wie vorher - sonst
            # verschoebe der Regler die Seedfolge, und jede Reglerstellung
            # saehe aus wie ein anderer Seed statt wie dieselbe Landmasse in
            # anderer Gestalt.
            my *= (1.0 - streckung)
            radius = halb * rng.uniform(gestalt["r_min"], gestalt["r_max"])
        felder.append(radius - np.hypot(WX - mx, WY - my))

    # Weiche Vereinigung. Klein gewaehlt, damit die Zwickel zwischen den Lappen
    # spitz bleiben - sie SIND die Buchten.
    weich = gestalt["weich"] * halb
    feld = weich * np.log(np.sum(np.exp(np.stack(felder) / weich), axis=0))

    # ABGEZOGENE SCHEIBEN. Die Zwickel allein geben nur flache Einbuchtungen.
    # Zwei bis drei Scheiben, die Material WEGNEHMEN, schneiden tief herein -
    # das sind die langen Buchten der Vorlage. Sie sitzen am Rand und zeigen
    # nach innen.
    for _ in range(rng.integers(gestalt["buchten_min"],
                                gestalt["buchten_max"])):
        winkel = rng.uniform(0.0, 2.0 * np.pi)
        weite = halb * rng.uniform(0.75, 1.15)
        bx, by = weite * np.cos(winkel), weite * np.sin(winkel)
        radius = halb * rng.uniform(gestalt["bucht_min"],
                                    gestalt["bucht_max"])
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
    liegt Skerrheim auch dann im Norden, wenn die Landmasse dort einen Lappen
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
        # Ohne die Umkehr stand die Welt auf dem Kopf: Samarcia und Macchia
        # oben, Skerrheim und Morobora unten. Aufgefallen erst, als die Regionen
        # BESCHRIFTET wurden - an einer namenlosen Hoehenkarte sieht man nicht,
        # wo Norden ist. Genau das ist der Grund, warum die Regionenkarte
        # gebaut wurde: sie macht eine Himmelsrichtung ueberhaupt pruefbar.
        #
        # Die Konvention kommt nicht von hier, sie gilt im ganzen Projekt: die
        # Anzeige zeichnet mit origin='lower' (Zeile 0 unten), und
        # core/weather_generator.py haelt bei Wind und Schattenwurf ausdruecklich
        # "Zeile height-1 = Norden" fest. Ein Regionsgitter, das dem
        # widerspricht, wuerde Skerrheim in die Sonne und die Samarcia in den
        # Schatten legen.
        lage[k, 1] = 1.0 - (punkte[k, 0] - y0) / max(y1 - y0, 1) \
            + verzerrung * float(versatz_v[py, px])          # v, Nord -> Sued

    d2 = np.stack([(lage[:, 0] - mitten[s]) ** 2 + (lage[:, 1] - mitten[z]) ** 2
                   for z in range(3) for s in range(3)], axis=1)

    # FLAECHENEICHUNG.
    #
    # Macchia und Thalassia verlieren 40 bzw. 65 Prozent ihrer
    # Flaeche ans Wasser, die Estrande 45. Sie brauchen deshalb mehr
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
    # Fehler) schwang: Nevadin landete bei 0.049 statt 0.098, Macchia bei
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
    # DIE ZELL-ETIKETTEN MIT ZURUECKGEBEN (2026-08-26).
    #
    # Bis hierher wurden sie hier weggeworfen. Das Gebietssystem
    # (docs/archiv/2026-08-25_AUFRAEUMPLAN.md 4.8) braucht sie: es waechst Hoehengebiete
    # UEBER DEN ZELLGRAPHEN von der Kueste ins Land, so wie
    # `seegliederung()` den Seegrad ueber denselben Graphen nach aussen
    # traegt. Ohne die Etiketten muesste es eine zweite Zellzerlegung
    # bauen - und zwei Zerlegungen waeren zwei Wahrheiten ueber dieselbe
    # Karte (SPEZIFIKATION 4.5).
    return gewichte, etikett



# =============================================================================
# KUESTENGEBIETE IM HINTERLAND (docs/archiv/2026-08-25_AUFRAEUMPLAN.md 4.8)
# =============================================================================

# WIE WEIT DIE GEBIETSHOEHEN AUSEINANDERLIEGEN, als Anteil des Regionsreliefs.
#
# Nutzerentwurf 2026-08-25: *"wir haben die regionen verteilt, warum gehen wir
# dann nicht entlang der kuestenvoronois ... und vergeben dort werte 1 bis 3.
# von der kueste dann werden immer benachbarte voronois der gleichen region
# nach und nach mit zahlen aufgefuellt ... die hoehe der gebiete richtet sich
# an den profilen der kueste."*
#
# ERSTER AUSBAU: NUR DIE MITTLERE HOEHE. Nutzervorgabe 2026-08-26: *"lass uns
# anfangen nur mit hoehenwerten, also wie hoch die mittlere hoehe ist. die
# anderen werte wird von der region vererbt."* `formgroesse_m`, `rauheit`,
# `potenz` und `relief_m` kommen also unveraendert aus dem Regionskatalog.
GEBIET_HUB = 0.45

# Ueber welche Strecke die Gebietsgrenzen ineinander uebergehen.
# Nutzerentwurf: *"die werte gehen aber an der grenze von einem zum anderen
# gebiet (auch unterschiedlicher regionen) ineinander ueber"*, Beispiel
# [1][1][1.33][1.66][2][2].
GEBIET_GLAETTUNG_M = 800.0

# Wo das Gebiet die Kueste abloest. Bis p2 (hoechstens 700 m, siehe P2_MAX_M
# in core/vektor_kueste.py) fuehrt das gemessene Kuestenprofil; das Gebiet
# blendet auf derselben Strecke ein. So entsteht keine zweite Naht - genau
# die sollte das System ja beseitigen.
GEBIET_ANLAUF_M = 700.0

# Flaechenanteil je Gebiet innerhalb seiner Region.
# Nutzerentwurf: *"am ende sollen also min. 15% aber maximal 50% der region
# jeweils zu einem gebiet gehoeren."* Zielwert ist `max_anteil` aus dem
# Archetypkatalog, geklemmt auf dieses Fenster.
GEBIET_ANTEIL_MIN = 0.15
GEBIET_ANTEIL_MAX = 0.50

# WIE WEIT DIE KUESTENSAAT INS ALPENLAND REICHT, in Zellschritten.
#
# Nutzervorgabe 2026-08-26: *"alpen wuerde ich sagen ist ein sonderfall. hier
# gibt es entweder eine saat, dann geht das aber immer nur wenige voronoi von
# der kueste rein. meistens ist es aber einfach nur alpin. ein typ der diese
# optik erzeugt die es jetzt hat."*
#
# Jenseits davon bekommt das Nevadin Delta 0 - also genau die heutige Optik,
# die der Nutzer behalten will. Das loest zugleich das Problem, dass das
# Nevadin auf 62 von 64 Karten ueberhaupt keine Kueste hat.
ALPEN_SAAT_REICHWEITE = 2


def kuestengebiete(H, felder, zell_etikett, seed):
    """
    Hoehengebiete, von der Kueste ueber den Zellgraphen ins Land gewachsen.

    Rueckgabe: (delta_m, gebiet_raster, hinterlandhoehe_m).

    `delta_m` ist das Hoehen-Delta in Metern - auf See 0, an der Wasserlinie
    0 (Anlauf), und je Region mittelwerttreu. `gebiet_raster` ist die
    Gebietsnummer je Pixel (Index in KUESTEN_ARCHETYPEN der Region, -1 auf
    See und im alpinen Sonderfall).

    WARUM ALS DELTA HINTERHER UND NICHT ALS `hoehe_m`-FELD VORHER: die
    Kuestenarchetypen entstehen aus H - die Vektorkueste braucht eine
    Kuestenlinie, um ueberhaupt zu wissen, wo Kueste ist. Sie in `hoehe_m`
    zurueckzuspeisen waere ein Zirkel. Da `hoehe_m` aber ADDITIV in H
    eingeht, ist ein nachtraeglicher Summand exakt dasselbe wie ein anderes
    `hoehe_m` je Gebiet - nur ohne Zirkel.

    WARUM MITTELWERTFREI JE REGION: `hoehe_m`, `relief_m`, `wasser_soll` und
    `flaeche_soll` sind allesamt REGIONSMITTEL, und die gesamte Eichung
    haengt daran (smoke_test_regionen_welt). Ein Gebietsfeld, dessen Mittel
    verrutscht, wirft sie um. Die Gebiete verteilen also innerhalb der
    Region um, sie verschieben sie nicht.
    """
    import logging

    size = H.shape[0]
    mpp = WELT_KM * 1000.0 / size
    land = H > 0
    # NUR GEGEN DAS HAUPTMEER, NICHT GEGEN BINNENSEEN (Bugfix, dieselbe
    # Vorgabe 2026-08-13 wie in _kuesten_umformen(): ein Binnensee ist keine
    # Meereskueste). `_hauptmeer_maske()` liefert nur die mit dem Kartenrand
    # verbundene Wasserflaeche; ein abgeschlossener Binnensee faellt heraus.
    hauptmeer = _hauptmeer_maske(H)
    binnensee = (~land) & (~hauptmeer)
    delta = np.zeros((size, size), dtype=np.float64)
    archetyp = felder.get("kuesten_archetyp")
    region_map = felder.get("regionen")
    if archetyp is None or region_map is None or not land.any():
        logging.getLogger(__name__).info(
            "Kuestengebiete uebersprungen - keine Archetypfelder "
            "(Kuestenformung abgeschaltet?)")
        return (delta, np.full(H.shape, -1, dtype=np.int16),
                np.full(H.shape, np.nan, dtype=np.float32))

    n_zellen = int(zell_etikett.max()) + 1
    regionsnamen = {i: r["name"] for i, (_z, _s, r) in enumerate(alle_regionen())}

    # Landflaeche je Zelle einmal vorrechnen - die Schleifen unten brauchen
    # sie mehrfach, und ein bincount ist billiger als n_zellen Masken.
    zell_flaeche_alle = np.bincount(zell_etikett[land].ravel(),
                                    minlength=n_zellen).astype(np.float64)

    # --- Zelle -> Region (Mehrheit ueber ihre Landpixel) ---------------------
    zell_region = np.full(n_zellen, -1, dtype=np.int32)
    for z in np.flatnonzero(zell_flaeche_alle >= 4):
        pix = land & (zell_etikett == z)
        werte, anzahl = np.unique(region_map[pix], return_counts=True)
        zell_region[z] = int(werte[np.argmax(anzahl)])

    # --- Nachbarschaft der Zellen (wie in seegliederung) ---------------------
    nachbarn = [set() for _ in range(n_zellen)]
    for a, b in ((zell_etikett[:, :-1], zell_etikett[:, 1:]),
                 (zell_etikett[:-1, :], zell_etikett[1:, :])):
        u = a != b
        for p, q in zip(a[u].ravel().tolist(), b[u].ravel().tolist()):
            nachbarn[p].add(q)
            nachbarn[q].add(p)

    # --- Saat: welcher Archetyp beruehrt welche Zelle ------------------------
    #
    # NUR GEGEN DAS HAUPTMEER, NICHT GEGEN BINNENSEEN. Vorher markierte
    # `kueste` jedes Landpixel, das an IRGENDEIN Wasser grenzte - ein
    # Seeufer im Landesinneren zaehlte damit genauso als Kueste wie das
    # Hauptmeer und lieferte Saat-Stimmen fuer einen Kuesten-Archetyp, den
    # es fachlich nicht hat (derselbe Fehler, den _kuesten_umformen() schon
    # gegen `distanz_m` behoben hat - hier die zellbasierte Entsprechung).
    kueste = land & ndimage.binary_dilation(hauptmeer, iterations=1)
    zell_saat = np.full(n_zellen, -1, dtype=np.int32)
    for z in np.flatnonzero(zell_region >= 0):
        pix = kueste & (zell_etikett == z) & (archetyp >= 0)
        if pix.sum() < 3:
            continue
        werte, anzahl = np.unique(archetyp[pix], return_counts=True)
        zell_saat[z] = int(werte[np.argmax(anzahl)])

    # Zellen, deren Land NUR an einen Binnensee grenzt (nicht ans
    # Hauptmeer), bekommen unten keine gewachsene Gebietszuordnung - ohne
    # eigene Saat wuerden sie sonst trotzdem den naechstgelegenen Archetyp
    # per Breitensuche ERBEN, und ein Seeufer waere optisch weiterhin
    # "Kueste", nur ohne eigene Saatstimme. Eine Zelle, die BEIDES beruehrt
    # (z.B. eine schmale Landenge zwischen Meer und See), bleibt regulaer -
    # sie ist echte Kueste.
    kueste_see = land & ndimage.binary_dilation(binnensee, iterations=1)
    zell_hauptmeer_kontakt = np.zeros(n_zellen, dtype=bool)
    if kueste.any():
        zell_hauptmeer_kontakt[np.unique(zell_etikett[kueste])] = True
    zell_binnensee_kontakt = np.zeros(n_zellen, dtype=bool)
    if kueste_see.any():
        zell_binnensee_kontakt[np.unique(zell_etikett[kueste_see])] = True
    zell_nur_binnensee = zell_binnensee_kontakt & ~zell_hauptmeer_kontakt

    # --- Gebiete wachsen lassen, je Region -----------------------------------
    zell_gebiet = np.full(n_zellen, -1, dtype=np.int32)
    zell_tiefe = np.full(n_zellen, np.inf, dtype=np.float64)
    for r_index, r_name in regionsnamen.items():
        typen = KUESTEN_ARCHETYPEN.get(r_name)
        if not typen:
            continue
        in_region = np.flatnonzero(zell_region == r_index)
        if len(in_region) == 0:
            continue
        menge = set(in_region.tolist())

        # Breitensuche je Archetyp - Abstand in ZELLSCHRITTEN, genau wie der
        # Seegrad in seegliederung() nach aussen zaehlt.
        abstand = np.full((len(typen), n_zellen), np.inf)
        for j in range(len(typen)):
            front = [int(z) for z in in_region if zell_saat[z] == j]
            for z in front:
                abstand[j, z] = 0.0
            tiefe = 0
            while front:
                tiefe += 1
                neu = []
                for z in front:
                    for nb in nachbarn[z]:
                        if nb in menge and abstand[j, nb] > tiefe:
                            abstand[j, nb] = tiefe
                            neu.append(nb)
                front = neu
        if not np.isfinite(abstand[:, in_region]).any():
            continue                      # Region ohne jede Kuestensaat

        # FLAECHENQUOTE per Log-Regelung - dasselbe Verfahren wie die
        # Flaecheneichung in voronoi_regionen(), samt der dort gemessenen
        # Erkenntnis, dass eine lineare Regelung schwingt.
        flaeche = zell_flaeche_alle[in_region]
        soll = np.array([min(max(t["max_anteil"], GEBIET_ANTEIL_MIN),
                             GEBIET_ANTEIL_MAX) for t in typen])
        soll = soll / soll.sum()
        d = abstand[:, in_region]
        # UNERREICHBAR IST NICHT UNENDLICH.
        #
        # Ein Archetyp, der auf DIESER Karte keine einzige Saatzelle hat
        # (seine Kuestenlaenge streut je Karte um 8-21 Prozentpunkte, siehe
        # SAAT_BUDGET_KORREKTUR in core/vektor_kueste.py), bekam hier
        # Abstand 1e3. Die Log-Regelung darunter bewegt den Vorteil um
        # hoechstens rund 21 - sie kann 1000 nie einholen, und der Archetyp
        # blieb dauerhaft ausgeschlossen. GEMESSEN (384 px, Seed 20260804)
        # ergab das in der Samarcia 0 % / 97 % / 2 % statt dreier Gebiete.
        #
        # Mit `max + 2` ist ein saatloser Archetyp die LETZTE Wahl, aber
        # erreichbar: die Regelung gibt ihm die Zellen, die von allen
        # anderen Saaten am weitesten weg sind - also das Regionsinnere.
        # Genau dort gehoert ein drittes Gebiet hin, wenn die Kueste es
        # nicht hergibt.
        # Der Ersatz muss VARIIEREN, sonst gewinnt der Typ alles oder nichts.
        #
        # Ein erster Anlauf setzte einen ueberall GLEICHEN Ersatzabstand.
        # Damit ist die Zeile des Typs konstant, und `argmax` kippt bei
        # steigendem Vorteil schlagartig alle Zellen auf einmal um - er
        # steht auf 0 % oder auf 100 %, nie dazwischen. GEMESSEN blieben die
        # Moher-Klippen im Clonagh dadurch bei 0 %.
        #
        # Richtig ist ein Abstand, der IM REGIONSINNEREN klein wird: was von
        # allen vorhandenen Kuestensaaten weit weg liegt, ist nah am
        # saatlosen Typ. Das ist zugleich die sachlich richtige Semantik -
        # gibt die Kueste einen dritten Typ nicht her, gehoert er ins Innere.
        endlich = d[np.isfinite(d)]
        ersatz = (float(endlich.max()) + 2.0) if endlich.size else 1.0
        ohne_saat = ~np.isfinite(d).any(axis=1)
        if ohne_saat.any() and (~ohne_saat).any():
            naechste_saat = np.min(np.where(np.isfinite(d), d, np.inf)[~ohne_saat],
                                   axis=0)
            naechste_saat = np.where(np.isfinite(naechste_saat), naechste_saat, 0.0)
            for j in np.flatnonzero(ohne_saat):
                d[j] = ersatz - naechste_saat
        d = np.where(np.isfinite(d), d, ersatz)
        # DEN BESTEN DURCHGANG BEHALTEN, NICHT DEN LETZTEN.
        #
        # `voronoi_regionen()` benutzt dieselbe Log-Regelung und nimmt dort
        # den letzten Durchgang - das geht gut, weil die Abstaende dort
        # STETIG sind (d^2/0.055). Hier ist `d` eine ganzzahlige
        # Graphdistanz (meist 0..4). Die Zuordnung ist damit eine
        # Treppenfunktion des Vorteils: eine winzige Aenderung kippt ganze
        # Zellgruppen auf einmal, die Regelung schwingt statt zu
        # konvergieren, und welcher Zustand am Ende steht, ist Zufall.
        #
        # GEMESSEN (384 px, Seed 20260804), letzter Durchgang:
        #     Samarcia soll 30/35/35, ist 0/100/0 - bei DREI vorhandenen Saaten.
        #
        # Der beste gesehene Zustand ist immer mindestens so gut wie der
        # letzte und kostet nur eine Kopie je Runde.
        vorteil = np.zeros(len(typen))
        bester, bester_fehler = None, np.inf
        for _runde in range(60):
            fuehrt = np.argmax(-d + vorteil[:, None], axis=0)
            ist = np.array([flaeche[fuehrt == j].sum() for j in range(len(typen))])
            ist = np.maximum(ist / max(ist.sum(), 1.0), 1e-4)
            # Das 15-50-%-Fenster ausdruecklich bestrafen - es ist eine
            # Nutzervorgabe (*"am ende sollen also min. 15% aber maximal
            # 50% der region jeweils zu einem gebiet gehoeren"*), und die
            # blosse Abstandssumme haelt sie nicht ein: das Nebelrode
            # landete bei 14 %, obwohl die Summe gut aussah.
            verstoss = (np.maximum(GEBIET_ANTEIL_MIN - ist, 0.0)
                        + np.maximum(ist - GEBIET_ANTEIL_MAX, 0.0)).sum()
            fehler = float(np.abs(ist - soll).sum() + 3.0 * verstoss)
            if fehler < bester_fehler:
                bester_fehler, bester = fehler, fuehrt.copy()
            vorteil += 0.35 * np.log(soll / ist)
        fuehrt = bester if bester is not None else np.argmax(
            -d + vorteil[:, None], axis=0)
        zell_gebiet[in_region] = fuehrt
        zell_tiefe[in_region] = d[fuehrt, np.arange(len(in_region))]

        # BINNENSEEN SIND KEINE KUESTE: hier erzwungen statt nur ueber die
        # fehlende Saat gehofft - ohne diese Zeile wuerden reine Seeufer-
        # Zellen trotzdem den Archetyp der naechsten echten Kuestenzelle
        # per Breitensuche erben (siehe zell_nur_binnensee oben).
        zell_gebiet[in_region[zell_nur_binnensee[in_region]]] = -1

        # LAUT MELDEN, WAS HERAUSKAM. Eine Quote, die ihr Ziel verfehlt,
        # liefert trotzdem ein plausibles Gelaende - sie waere von Erfolg
        # nicht zu unterscheiden (CLAUDE.md, "jeder stille Rueckfall
        # braucht eine laute Logzeile").
        erreicht = np.array([flaeche[fuehrt == j].sum()
                             for j in range(len(typen))])
        erreicht = erreicht / max(erreicht.sum(), 1.0)
        mit_saat = sum(1 for j in range(len(typen))
                       if np.isfinite(abstand[j, in_region]).any())
        logging.getLogger(__name__).info(
            "Kuestengebiete %s: %d Zellen, %d von %d Archetypen mit Saat, "
            "soll %s, ist %s",
            r_name, len(in_region), mit_saat, len(typen),
            " ".join(f"{v:.0%}" for v in soll),
            " ".join(f"{v:.0%}" for v in erreicht))

        # ALPENLAND: die Saat reicht nur wenige Zellen weit, der Rest ist
        # "alpin" - siehe ALPEN_SAAT_REICHWEITE.
        if r_name == "Nevadin":
            zu_tief = in_region[zell_tiefe[in_region] > ALPEN_SAAT_REICHWEITE]
            zell_gebiet[zu_tief] = -1

    # --- Zellwerte -> Pixelfeld ---------------------------------------------
    # Wert eines Gebiets: der `hoehe_faktor` seines Archetyps, innerhalb der
    # Region auf Mittelwert 0 gezogen und auf seine Spanne normiert.
    wert_je_zelle = np.zeros(n_zellen, dtype=np.float64)
    for r_index, r_name in regionsnamen.items():
        typen = KUESTEN_ARCHETYPEN.get(r_name)
        if not typen:
            continue
        in_region = np.flatnonzero((zell_region == r_index) & (zell_gebiet >= 0))
        if len(in_region) == 0:
            continue
        # DIE GEMESSENE HINTERLANDHOEHE, NICHT DER KATALOGFAKTOR (2026-08-26).
        #
        # `hoehe_faktor` beschreibt das UFER. Fuer die Hoehe des Hinterlands
        # zaehlt das Hinterland - und in vier von neun Regionen dreht sich
        # die Reihenfolge dadurch um (die Begruendung samt Tabelle steht bei
        # HINTERLAND_BAND_M in core/vektor_kueste.py).
        from core.vektor_kueste import GEMESSENE_HINTERLANDHOEHE
        hf = np.array([GEMESSENE_HINTERLANDHOEHE.get(t["name"],
                                                     t["hoehe_faktor"])
                       for t in typen], dtype=np.float64)
        spanne = max(float(hf.max() - hf.min()), 1e-9)
        flaeche = np.maximum(zell_flaeche_alle[in_region], 1e-9)
        roh = hf[zell_gebiet[in_region]]
        mittel = float(np.average(roh, weights=flaeche))
        wert_je_zelle[in_region] = (roh - mittel) / spanne

    feld = wert_je_zelle[zell_etikett]
    feld = ndimage.gaussian_filter(feld, max(GEBIET_GLAETTUNG_M / mpp, 0.5))

    # --- Anwenden: MULTIPLIKATIV, nicht additiv ------------------------------
    #
    # DER ERSTE ENTWURF WAR ADDITIV UND FALSCH. Gemessen (384 px, Seed
    # 20260804): das Delta lag bei bis zu +-70 m, und an der Wasserlinie
    # standen davon noch 68 m. Der Grund ist einfach und war vorher nicht
    # offensichtlich - ein negatives Delta DRUECKT LAND UNTER NULL. Damit
    # wandert die Kuestenlinie, und zwar genau dort, wo ein niedriges Gebiet
    # ans Meer stoesst. Die Regionsmittel stimmten anschliessend auch nicht
    # mehr (Skerrheim +1.06 m), weil sie gegen die alte Landmaske gerechnet
    # waren und die neue eine andere war.
    #
    # Multiplikativ kann das nicht passieren: an der Wasserlinie ist H = 0,
    # und 0 mal irgendetwas bleibt 0. **Die Kuestenlinie ist damit exakt
    # erhalten, nicht nur naeherungsweise.** Sachlich ist es ausserdem das
    # Richtigere - ein Gebiet mit hoeherer mittlerer Hoehe ist eines, dessen
    # Gelaende hoeher AUFRAGT, nicht eines, unter das jemand einen Sockel
    # geschoben hat.
    zur_kueste = ndimage.distance_transform_edt(land) * mpp
    anlauf = np.clip(zur_kueste / max(GEBIET_ANLAUF_M, 1.0), 0.0, 1.0)
    anlauf = anlauf * anlauf * (3.0 - 2.0 * anlauf)      # smoothstep

    hub = GEBIET_HUB * feld * anlauf

    # --- Mittelwerttreu JE REGION - siehe den Kopf dieser Funktion -----------
    #
    # Verlangt ist mittel(H * (1 + hub - c*anlauf)) == mittel(H), also
    #
    #     c = mittel(H * hub) / mittel(H * anlauf)
    #
    # Abgezogen wird wieder ein Vielfaches DES ANLAUFS, nicht eine Konstante:
    # eine Konstante staende auch dort, wo der Anlauf 0 ist, und haette die
    # Kuestenzone verstellt.
    faktor = np.ones((size, size), dtype=np.float64)
    for r_index in regionsnamen:
        m = land & (region_map == r_index)
        if m.sum() < 16:
            continue
        nenner = float((H[m] * anlauf[m]).mean())
        c = (float((H[m] * hub[m]).mean()) / nenner) if abs(nenner) > 1e-9 else 0.0
        faktor[m] = 1.0 + hub[m] - c * anlauf[m]
    # Nie das Vorzeichen drehen - ein Gebiet darf Land nicht zu Meer machen.
    faktor = np.clip(faktor, 0.05, 3.0)
    # Die Gebietszuteilung als Raster mit herausgeben - sie ist die Groesse,
    # die geprueft und spaeter angezeigt wird. Ohne sie liesse sich von
    # aussen nicht feststellen, OB die Gebiete zusammenhaengen; ein erster
    # Pruefversuch griff mangels Raster auf `kuesten_archetyp` zurueck und
    # mass damit das Kuesten-Voronoi statt der Gebiete.
    gebiet_raster = np.where(land, zell_gebiet[zell_etikett], -1).astype(np.int16)
    # PIXELGENAUER NACHSCHLAG (Bugfix, zusaetzlich zur Zellen-Sperre oben):
    # eine Zelle, die BEIDES beruehrt - echtes Hauptmeer irgendwo an ihrem
    # Rand UND einen Binnensee naeher am Seeufer -, behaelt zu Recht ihren
    # Archetyp (sie IST teilweise echte Kueste). Aber die paar Pixel, die
    # UNMITTELBAR am Binnensee liegen, sollen den Archetyp dieser Zelle
    # trotzdem nicht tragen - sonst waere GENAU DAS Seeufer optisch wieder
    # "Kueste", nur weil es zufaellig in derselben grossen Zelle wie ein
    # Stueck echter Kueste liegt. Die Zellen-Granularitaet kann das nicht
    # ausdruecken, ein direkter Pixel-Ueberschreib schon.
    gebiet_raster = np.where(kueste_see, -1, gebiet_raster).astype(np.int16)

    # DIE HOEHENFAKTOR-KARTE (2026-08-26). Nutzerwunsch: *"ich will den
    # hoehenfaktor sehen koennen (3d und 2D)"*. Je Pixel die gemessene
    # Hinterlandhoehe seines Gebiets in METERN - also genau die Groesse, aus
    # der das Delta oben entsteht. Auf See und im alpinen Sonderfall NaN,
    # damit die Anzeige dort nichts einfaerbt.
    hoehen_karte = np.full(H.shape, np.nan, dtype=np.float32)
    for r_index, r_name in regionsnamen.items():
        typen = KUESTEN_ARCHETYPEN.get(r_name)
        if not typen:
            continue
        from core.vektor_kueste import GEMESSENE_HINTERLANDHOEHE
        for j, t in enumerate(typen):
            m = land & (region_map == r_index) & (gebiet_raster == j)
            if m.any():
                hoehen_karte[m] = GEMESSENE_HINTERLANDHOEHE.get(
                    t["name"], float(t["hoehe_faktor"]))
    return (np.where(land, H * (faktor - 1.0), 0.0), gebiet_raster,
            hoehen_karte)


# =============================================================================
# DIE REGIONSANSICHT (docs/archiv/2026-08-25_AUFRAEUMPLAN.md 4.10/4.11)
# =============================================================================

# Wie breit eine Region ist. Die Welt ist ein 3x3-Raster, also ein Drittel.
REGIONSBREITE_M = WELT_KM * 1000.0 / 3.0

# Die Regler, die die Regionsansicht ueberschreiben darf. Alles andere kommt
# unveraendert aus dem Katalog.
#
# Nutzervorgabe 2026-08-25 zum Umfang: die Anpassung gilt **nur fuer die
# aktuelle Karte**, der Katalog bleibt die Vorgabe. Deshalb ein
# Ueberschreibungs-Dict statt eines Schreibzugriffs auf `alle_regionen()` -
# und deshalb messen die Tests weiter gegen den Katalog.
REGIONSREGLER = ("hoehe_m", "relief_m", "formgroesse_m", "rauheit", "potenz")



# DER LAND-SEE-AUFBAU DER REGIONSANSICHT (Nutzertabelle 2026-08-26)
#
# *"bei huegelland haben wir dann 3/4 rechts sind standard noise. und 1/4 ist
# meer. dazwischen cliffs of moher (entlang einer leicht geschwungenen linie
# die einmal durch die map geht) ... griechische Inseln sind tatsaechlich 1/2
# festland 1/2 inseln, getrieben wird das durch clamping der einen haelfte auf
# zahlen ueber 5m ... Skerrheim erhaelt links und rechts geiranger kueste und
# in der mitte einen leicht geschwungenen fjord ... alpenland hat keine
# kueste, nur berge."*
#
# WARUM ES DAS BRAUCHT. Der reine Parametersatz einer Region enthaelt keine
# Kueste - `hoehe_m` ist ein Mittelwert ueber Land UND See. Gemessen liegen
# Estrande (-81 m) und Skerrheim (-52 m) als reiner Satz komplett unter
# dem Meeresspiegel; auf der Karte hebt sie erst die Kontinentmaske heraus.
# Ohne Aufbau zeigte die Vorschau dort nur blaues Wasser.
#
# WIE ES WIRKT, und was es ausdruecklich NICHT tut: der Aufbau verschiebt nur
# die HOEHENLAGE der beiden Seiten. Das Relief, die Formgroesse, die Rauheit
# und die Potenzkurve der Region bleiben unangetastet - die Vorschau zeigt
# denselben Charakter wie vorher, nur mit einer Kueste darin.
#
# DAS IST EIN VORSCHAU-AUFBAU, KEINE KARTENEIGENSCHAFT. Das Clonagh hat
# `wasser_soll = 0`, bekommt hier aber ein Viertel Meer - weil man einen
# Kuestentyp nur an einer Kueste beurteilen kann. Auf der echten Karte
# entscheidet weiterhin die Kontinentform, wo Wasser liegt.
REGIONS_AUFBAU = {
    "Clonagh":         ("kueste", 0.75),
    "Skerrheim":          ("fjord", 0.62),
    "Morobora":              ("kueste", 0.75),
    "Estrande":     ("kueste", 0.75),
    "Nevadin":          ("ohne_kueste", 1.00),
    "Nebelrode":      ("kueste", 0.75),
    "Samarcia":             ("kueste", 0.75),
    "Macchia":         ("kueste", 0.75),
    "Thalassia": ("inseln", 0.50),
}

# Hoehe, auf die das untere Ende der Landseite gehoben wird, und Tiefe, auf
# die das obere Ende der Seeseite gedrueckt wird. Bewusst klein: die Kueste
# selbst formen die Archetypen, hier geht es nur darum, dass Land Land ist.
AUFBAU_LANDSOCKEL_M = 20.0
AUFBAU_SEETIEFE_M = 30.0
# Ueber welche Strecke Land und See ineinander uebergehen.
AUFBAU_UEBERGANG_M = 600.0
# Wie stark die Kuestenlinie schwingt, als Anteil der Kartenbreite.
AUFBAU_SCHWUNG = 0.09


def _aufbau_seeanteil(art, anteil, breite_px, hoehe_px, mpp, seed):
    """
    Wie "seeisch" jedes Pixel ist: 0 = Land, 1 = offene See.

    Die Kuestenlinie ist eine leicht geschwungene Kurve quer durch die Karte
    - eine gerade Trennung saehe nach Schnittkante aus, und der Nutzer hat
    ausdruecklich *"eine leicht geschwungene linie"* verlangt.
    """
    yy, xx = np.mgrid[0:hoehe_px, 0:breite_px]
    u = xx / max(breite_px - 1, 1)
    v = yy / max(hoehe_px - 1, 1)

    rng = np.random.default_rng(int(seed) ^ 0xA0F5)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
    schwung = AUFBAU_SCHWUNG * (
        0.6 * np.sin(2.0 * np.pi * v + phase[0])
        + 0.3 * np.sin(4.0 * np.pi * v + phase[1])
        + 0.1 * np.sin(7.0 * np.pi * v + phase[2]))

    weich = max(AUFBAU_UEBERGANG_M / max(mpp * breite_px, 1e-9), 1e-3)

    if art == "ohne_kueste":
        return np.zeros((hoehe_px, breite_px), dtype=np.float64)

    if art == "inseln":
        # Linke Haelfte Festland, rechte frei - dort bilden sich die Inseln
        # aus dem Rauschen der Region selbst.
        return np.clip((u - anteil + schwung) / weich, 0.0, 1.0)

    if art == "fjord":
        # Land links UND rechts, dazwischen ein geschwungener Fjord. `anteil`
        # ist der Landanteil; der Fjord nimmt den Rest in der Mitte ein.
        halbe = 0.5 * (1.0 - anteil)
        mitte = 0.5 + schwung
        d = np.abs(u - mitte) - halbe
        return np.clip(-d / weich, 0.0, 1.0)

    # "kueste": Land links, Meer rechts, Grenze bei `anteil`.
    return np.clip((u - anteil + schwung) / weich, 0.0, 1.0)


def _aufbau_anwenden(H, art, anteil, mpp, seed, frei_rechts=False):
    """
    Land anheben, See absenken - ohne das Relief anzutasten.

    Die Verschiebung ist je Seite EIN Wert, kein Feld: dadurch bleibt jede
    Steigung, jede Form und jede Rauheit der Region genau so, wie der
    Parametersatz sie erzeugt. Nur die Hoehenlage wechselt.
    """
    hoehe_px, breite_px = H.shape
    see = _aufbau_seeanteil(art, anteil, breite_px, hoehe_px, mpp, seed)
    if not see.any():
        # Ohne Kueste: nur sicherstellen, dass alles ueber Wasser liegt.
        fehlt = AUFBAU_LANDSOCKEL_M - float(np.percentile(H, 2.0))
        return H + max(fehlt, 0.0), see

    # Der Versatz wird aus der VERTEILUNG bestimmt, nicht geraten: das untere
    # Ende des Landes soll knapp ueber Null liegen, das obere Ende der See
    # klar darunter.
    #
    # JEDER VERSATZ WIRKT NUR IN EINE RICHTUNG. Ein erster Entwurf setzte
    # beide unbedingt - und ZOG DAMIT LAND HERUNTER, das laengst ueber Wasser
    # lag: das Clonagh (121..194 m) landete bei -90..81 m, weil sein
    # 5.-Perzentil auf 20 m gedrueckt wurde. Gemessen an allen neun Regionen
    # war das sofort sichtbar; die Landanteile stimmten trotzdem, der Fehler
    # steckte nur in den Hoehen.
    #
    # Mit max/min bleibt jede Region auf ihrer eigenen Hoehe, und der Aufbau
    # greift nur dort ein, wo er muss - bei Estrande und Skerrheim, die
    # als reiner Parametersatz unter Null liegen.
    versatz_land = max(0.0, AUFBAU_LANDSOCKEL_M - float(np.percentile(H, 5.0)))
    versatz_see = min(0.0, -AUFBAU_SEETIEFE_M - float(np.percentile(H, 95.0)))

    if frei_rechts:
        # Thalassia: die Seeseite bekommt KEINEN Versatz. Dort
        # entscheidet das Rauschen der Region selbst, wo Inseln auftauchen -
        # genau das meint *"und rechts ist es frei (dort wo die inseln
        # entstehen)"*.
        versatz_see = 0.0

    versatz = (1.0 - see) * versatz_land + see * versatz_see
    return H + versatz, see


def regionsfeld(regionsname, breite_px, hoehe_px=None, seed=0,
                ueberschreibung=None, km=None, erosion=None,
                shader_manager=None, aufbau=None, kueste=False):
    """
    EINE Region auf Regionsmassstab - die Vorschau des Regionsreiters.

    Rueckgabe: (H in Metern, verwendete Parameter als dict).

    `aufbau` waehlt die Land-See-Aufteilung: None nimmt den Eintrag aus
    REGIONS_AUFBAU, ein (art, anteil)-Paar setzt ihn, und False schaltet ihn
    ganz ab (dann liegt der reine Parametersatz vor, bei den wasserreichen
    Regionen also fast nur Meer).

    `kueste=True` legt zusaetzlich die VEKTORKUESTE mit den drei Archetypen
    der Region auf - dann zeigt die Vorschau Moher-Klippen, Fjordwaende oder
    Straende, und der Erosionsfilter arbeitet darauf. Kostet Zeit (gemessen
    unten), deshalb abschaltbar.

    QUADRATISCH GERECHNET, WENN DIE KUESTE AN IST. `VektorKueste` leitet
    ihren Massstab aus `shape[0]` ab und benutzt `self.size` an 15 Stellen -
    sie setzt ein quadratisches Feld voraus. Eine Region IST quadratisch
    (7.1 x 7.1 km); ein rechteckiger Ausschnitt wird deshalb quadratisch
    gerechnet und danach zugeschnitten, statt 15 Stellen umzubauen.

    WAS HIER NICHT PASSIERT, und das ist der ganze Grund fuer die
    Geschwindigkeit: kein Kontinent, keine Voronoi-Zerlegung, keine
    Regionsmischung, keine Vektorkueste, kein Flussnetz. Nur der
    Oktavenstapel mit EINEM Parametersatz, die Potenzkurve und - wenn
    gewuenscht - der Erosionsfilter. Gemessen 0.08 s bei 128 px, 0.32 s bei
    256 px; damit ist eine Live-Vorschau am Regler moeglich
    (docs/archiv/2026-08-25_AUFRAEUMPLAN.md 4.9).

    MASSSTAB. `km` ist die Kantenlaenge des Ausschnitts; ohne Angabe die
    volle Regionsbreite (7.1 km). Bei 256 px sind das 27.7 m/px - ein
    Drittel groeber als die Kontinentalansicht bei 1024 px (20.8 m/px), aber
    die ganze Region im Bild. Ein Fenster mit GLEICHEM Massstab zeigte bei
    128 px nur 2.66 km, also 37 % der Regionsbreite - im Nevadin weniger
    als eine Bergform (gemessen 2026-08-26).

    WAS DIE VORSCHAU NICHT ZEIGT. Auf der fertigen Karte wird dieselbe
    Region ueber die Voronoi-Gewichte mit ihren Nachbarn verschmolzen, von
    den Kuestenarchetypen umgeformt und vom Gebietssystem in der Hoehe
    verschoben. Die Vorschau ist der CHARAKTER der Region, nicht ihr
    Aussehen auf der Karte.
    """
    r = None
    for _z, _s, kandidat in alle_regionen():
        if kandidat["name"] == regionsname:
            r = dict(kandidat)
            break
    if r is None:
        raise ValueError(f"unbekannte Region: {regionsname!r}")
    for schluessel, wert in (ueberschreibung or {}).items():
        if schluessel in REGIONSREGLER and wert is not None:
            r[schluessel] = float(wert)

    breite_px = int(breite_px)
    hoehe_px = int(breite_px if hoehe_px is None else hoehe_px)
    kante_m = float(REGIONSBREITE_M if km is None else km * 1000.0)
    mpp = kante_m / breite_px
    zuschnitt = None
    if kueste and hoehe_px != breite_px:
        zuschnitt = hoehe_px
        hoehe_px = breite_px

    stapel = oktavenstapel(breite_px, int(seed) ^ 0x5EED, shader_manager,
                           mpp=mpp, hoehe=hoehe_px)
    wellen = _wellenlaengen()

    # DIESELBE Oktavengewichtung wie weltfeld() - buchstaeblich dieselbe
    # Funktion, nicht eine Abschrift (siehe oktavengewicht()).
    form = float(r["formgroesse_m"])
    rauheit = float(np.clip(r["rauheit"], 0.2, 0.9))
    gewicht = np.array([float(oktavengewicht(k, form, rauheit))
                        for k in range(OKTAVEN)])
    relief = np.tensordot(gewicht, stapel, axes=(0, 0)) / max(gewicht.sum(), 1e-9)

    # Potenzkurve UM DEN MEDIAN - wortgleich zu weltfeld(). Ohne die
    # Rueckverschiebung wuerde die Potenz auch die mittlere Hoehe aendern,
    # und `hoehe_m` waere kein Mittelwert mehr (dort ausfuehrlich begruendet).
    t = np.clip(0.5 + SPREIZUNG * relief, 0.0, 1.0)
    potenz = float(np.clip(r["potenz"], 0.2, 4.0))
    t = np.clip(np.power(t, potenz) - np.power(0.5, potenz) + 0.5, 0.0, 1.0)

    H = (float(r["hoehe_m"]) + float(r["relief_m"]) * (t - 0.5)).astype(np.float64)

    # DER LAND-SEE-AUFBAU, siehe REGIONS_AUFBAU weiter oben. Er laeuft VOR
    # dem Erosionsfilter, weil der nur Land sehen soll (`max(H, 0)`) - danach
    # angewandt haette der Filter auf einer Karte ohne Meer gearbeitet und
    # die spaetere Seeseite mitgeformt.
    see_anteil = np.zeros_like(H)
    if aufbau is not False:
        art, anteil = REGIONS_AUFBAU.get(r["name"], ("kueste", 0.75))
        if isinstance(aufbau, tuple):
            art, anteil = aufbau
        H, see_anteil = _aufbau_anwenden(
            H, art, float(anteil), mpp, int(seed),
            frei_rechts=(art == "inseln"))

    if kueste:
        # DIE VEKTORKUESTE MIT DEN ARCHETYPEN DER REGION.
        #
        # `region_map` ist hier ein KONSTANTES Feld mit dem Index dieser
        # Region - dadurch waehlt `VektorKueste` genau ihre drei Archetypen
        # aus KUESTEN_ARCHETYPEN und mischt nichts von Nachbarn dazu. Genau
        # das soll die Vorschau zeigen.
        from core.vektor_kueste import VektorKueste, als_raster
        index = next(i for i, (_z, _s, k) in enumerate(alle_regionen())
                     if k["name"] == r["name"])
        if H.shape[0] != H.shape[1]:
            logging.getLogger(__name__).info(
                "Regionsansicht mit Kueste: %dx%d wird quadratisch gerechnet "
                "und danach zugeschnitten (VektorKueste setzt ein "
                "quadratisches Feld voraus)", H.shape[1], H.shape[0])
        if float(np.max(H)) > 0.0 and float(np.min(H)) < 0.0:
            regionen = np.full(H.shape, index, dtype=np.int16)
            _vk = VektorKueste(H, regionen, int(seed) ^ 0x4B55,
                               welt_km=kante_m / 1000.0)
            H = np.asarray(als_raster(_vk), dtype=np.float64)
        else:
            # Ohne Wasserlinie gibt es keine Kueste zu formen - das ist der
            # Normalfall im Nevadin. Laut melden statt still ueberspringen.
            logging.getLogger(__name__).info(
                "Regionsansicht %s: keine Wasserlinie, Kuestenformung "
                "uebersprungen", r["name"])

    if erosion:
        # UEBER filter_heightmap(), NICHT ueber erosion_filter(). Letztere
        # gibt ein DELTA im Einheitsquadrat zurueck, kein fertiges Gelaende -
        # ein erster Entwurf hier hat es als Ergebnis behandelt. Die Huelle
        # ist ausserdem derselbe Einstieg, den `_weltkarte_erosionsfilter()`
        # in der Pipeline benutzt, die Vorschau zeigt also denselben Filter.
        from core.terrain_erosion_filter import filter_heightmap
        werte = {k: v for k, v in erosion.items() if v is not None}
        # WIE IN DER PIPELINE: der Filter sieht nur Land. Er normiert gegen
        # die Spanne der uebergebenen Karte; mit dem Meeresboden darin ginge
        # die halbe Spanne fuer Wasser drauf (siehe
        # BaseTerrainGenerator._weltkarte_erosionsfilter).
        ergebnis = filter_heightmap(np.maximum(H, 0.0), mpp, werte)
        H = H + np.asarray(ergebnis["height_delta"], dtype=np.float64)

    if zuschnitt is not None:
        anfang = (H.shape[0] - zuschnitt) // 2
        H = H[anfang:anfang + zuschnitt]
        see_anteil = see_anteil[anfang:anfang + zuschnitt]
    r = dict(r)
    r["see_anteil"] = see_anteil
    return H, r

# Zieltiefe je Seegrad, docs/spezifikation/12_WASSER.md Abschnitt 7 - eine TABELLE statt einer
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
# (z.B. Skerrheim/Klippenkueste) wird dadurch nicht angehoben. `kuestenform`
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
    "Skerrheim": {0: -3.0, 1: -90.0, 2: -150.0, 3: -180.0, 4: MEERESBODEN_M},
    # HUEGELLAND: bleibt bis Grad 1 flach ("ab Seegrad 2" - erst dort beginnt
    # die eigentliche Vertiefung), dann schneller auf Meeresbodenniveau.
    "Clonagh": {0: -3.0, 1: -10.0, 2: -60.0, 3: -140.0, 4: MEERESBODEN_M},
}

# Seeeis-Wahrscheinlichkeit je Seegrad vor der Morobora (Nutzer 2026-08-11,
# Nachbesserung der ersten Fassung "nur Grad 1 und 2" - hart an/aus wirkte zu
# kuenstlich): "grade 0 ist 100% eis grade 1 75% chance und grade 2 ist 50%
# chance. und grade 3 ist 25%." Grad 4+ bewusst NICHT gelistet (0 %, .get()
# faellt darauf zurueck) - dort verlaufen Seewege, die frei bleiben sollen.
# NOCH OFFEN, vom Nutzer selbst vertagt: "es wird spaeter im spiel nur im
# winter erscheinen" - diese Karte hier bleibt EIN statischer Schnappschuss
# ohne Jahreszeit (siehe docs/spezifikation/13_KLIMA_UND_BIOME.md Abschnitt 1), die Saisonalitaet gehoert in
# das spaetere Zeitmodell, nicht in diese Funktion.
EISWAHRSCHEINLICHKEIT_JE_SEEGRAD = {0: 1.0, 1: 0.75, 2: 0.5, 3: 0.25}


def seegliederung(maske, gewichte, seed, punktzahl_land=200, punktzahl_see=400,
                  glaettung_m=260.0, shader_manager=None):
    """
    Die See als EIGENE Voronoi-Gliederung, docs/spezifikation/12_WASSER.md Abschnitt 7. Nutzer:
    "wenn wir das inland als voronoi kacheln haben, dann koennen wir ja auch
    das gleiche bei der see machen ... aber auch zB dass die voronois an der
    kueste nicht staerker vertieft werden, aber dass die voronois mit dem
    grad 1 etwas vertiefter sind, grad 2 noch vertiefter etc."

    SEEGRAD per Breitensuche ueber den Zellnachbarschaftsgraphen: Grad 0 sind
    alle Zellen, die Land enthalten, Grad 1 grenzt an eine Grad-0-Zelle, Grad 2
    an Grad 1, und so weiter (docs/spezifikation/12_WASSER.md Abschnitt 7). Ersetzt den alten
    Kuestenschelf `-t*(1-exp(-d/L))`, der rein auf dem EUKLIDISCHEN Abstand zur
    Kueste beruhte (an der Aufloesung der Distanztransformation haengend) durch
    eine Zellstruktur, die spaeter auch fuer Seewege/Seemonster/Fischgruende
    gebraucht wird - "eine Karte, die spaeter etwas bedeutet".

    UFERREGIONEN: je Zelle die bis zu zwei naechstgelegenen LAND-Punkte, ueber
    deren fuehrende Region nachgeschlagen ("bis zu zwei Uferregionen ... damit
    faellt das Ufer vor dem Skerrheim steil ab, das Clonagh ab Seegrad 2,
    und die Morobora bekommt Seeeis" - diese drei konkreten Regeln selbst sind
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
    # Skerrheim faellt steil ab ... das Clonagh ab Seegrad 2 ... die Morobora
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
    # spaetere Zeitmodell (docs/spezifikation/13_KLIMA_UND_BIOME.md Abschnitt 1), nicht in diese Funktion.
    try:
        taiga_index = regionsnamen.index("Morobora")
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
# Gemessen am 2026-08-07: das Nebelrode hat 12.1 K Klimaspanne INNERHALB
# seines eigenen Gebiets, waehrend seine ganze Hoehenspanne nur 3.8 K liefert.
# Die Regionsmischung reicht so tief, dass sie die Hoehenabnahme voellig
# ueberdeckt - der Nutzer sah genau das: "ist die Hoehentemperatur vorhanden,
# augenscheinlich nicht so deutlich zu erkennen".
#
# In einer reinen Stichprobe (Klimasockel konstant gehalten) trifft die Morobora
# ihre -0.60 K/100 m exakt. Die Hoehenabnahme ist also richtig, sie geht nur
# unter.
#
# Das GELAENDE braucht die breite Mischung - ohne sie stuenden Stufen im
# Relief. Das KLIMA nicht: eine Region darf innen klimatisch einheitlich sein,
# solange der Uebergang nach aussen weich bleibt. Genau das leistet ein
# Exponent auf die Gewichte: 2.5 laesst einen Uebergang von rund einem Drittel
# der bisherigen Breite, aber immer noch ohne Kante.
KLIMA_SCHAERFE = 2.5


def parameterfeld(name, gewichte, schaerfe=1.0, ueberschreibung=None):
    """
    Ein Regler als volles Feld - gewichtete Mischung der neun Werte.

    `schaerfe` > 1 zieht die Mischung zur fuehrenden Region hin, ohne eine
    harte Kante zu erzeugen. Siehe KLIMA_SCHAERFE.

    `ueberschreibung` ist {Regionsname: {Reglername: Wert}} und ersetzt
    einzelne Katalogwerte - der EINZIGE Weg, auf dem die Einstellungen des
    Regionsreiters in die Karte kommen (docs/archiv/2026-08-25_AUFRAEUMPLAN.md 4.2).
    Nutzerentscheidung 2026-08-25: sie gelten **nur fuer die aktuelle
    Karte**; der Katalog bleibt unangetastet, und die Tests messen weiter
    gegen ihn.
    """
    if ueberschreibung:
        werte = np.array(
            [float(ueberschreibung.get(r["name"], {}).get(name, r[name]))
             for _z, _s, r in alle_regionen()], dtype=np.float64)
    else:
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
# Die Werte selbst (27 Archetypen, 9 Regionen zu je 3) stehen nicht mehr hier,
# sondern in core/daten/regionen.toml, geladen ueber _lade_regionsdaten() ganz
# oben in dieser Datei (Ticket #29, 2026-09-19) - inklusive der
# Thalassia-Santorini-Kliff-Korrektur vom 2026-08-13, wertgleich uebernommen.
# KUESTEN_ARCHETYPEN ist ab hier bereits gefuellt (siehe oben).

MAX_KLIPPENWINKEL_GRAD = 85.0   # nie eine reine 90-Grad-Wand
KUESTEN_ARCHETYP_PUNKTE_JE_REGION = 24  # Saatpunkte laengs der Kueste je Region

# LOKALE TERRAINHOEHE MISCHT SICH IN DIE ZIELHOEHE EIN (3.11, Nutzer-Vorgabe
# 2026-08-13: "Cliffs of Moher stehen im Rohgelaende auf flachem Land - das
# bekommt auch ein Gradient nicht logisch hin ... die hoehe der klippen soll
# von der lokalen terrainhoehe abhaengen, aber abgeschwaecht"). 0 = reine
# Tabelle wie bisher, 1 = die Klippe waere nur noch das lokale Rohgelaende.
KUESTEN_LOKALER_EINFLUSS = 0.35

# Wie weit die AEUSSERE Blendmaske (kuesten_staerke) mindestens ueber das
# INNERE Anstiegsprofil (skala_m, siehe unten) hinausreicht - 3.5 deckt rund
# 97% des exponentiellen Anstiegs ab (1-exp(-3.5)=0.970). Nutzer-Vorgabe:
# "wenn die klippen sehr hoch sind, dann strahlen diese auch etwas weiter
# rein, damit es realistischer aussieht (nicht so steil ueberall)".
KUESTEN_REICHWEITE_SKALA_FAKTOR = 3.5

# REICHWEITE: frueher EINE Zahl fuer alle (KUESTEN_BAND_KM = 1.2 km, im
# Bandtest sogar mal 1.5 = 1.8 km). Das war der Fehler hinter 3.9 UND 3.10:
# auf einer 21.3-km-Karte mit stark gegliederter Kueste liegt fast jeder
# Landpunkt naeher als 1.8 km am Wasser - gemessen trugen 96 % der Landflaeche
# einen Kuesten-Archetyp. Die "Kuestenregel" war damit faktisch eine
# Inselregel und hat die Regions-Hangeichung mitverschoben.
#
# Jetzt traegt JEDER Archetyp seine eigene `reichweite_km` (Nutzer-Vorgabe
# 2026-08-13: "die strahlwirkung soll je nach kuestentyp auch unterschiedlich
# stark strahlen. manche kuestenformen sind ja tiefer vielleicht") - eine
# Fjordwand wirkt 0.70 km ins Land, eine Ostsee-Flachkueste 0.18 km. Diese
# Konstante ist nur noch die OBERGRENZE fuers Zonen-Suchband.
KUESTEN_BAND_KM = 0.70

# MINDESTTIEFE FUER DEN NEUEN, VEREINHEITLICHTEN SEETIEFE-PROZESS (Redesign
# 2026-08-13, siehe _seetiefe_aus_archetyp() weiter unten). Ersetzt die
# vorherige eigenstaendige Unterwasser-Eintiefung in _kuesten_umformen() plus
# die anschliessende Ring-Ausbreitung (_meerestiefe_monoton, hatte messbare
# Kreuz-vs-Diagonal-Artefakte) durch eine einzige Formel direkt aus dem
# Voronoi-Seegrad-Feld.
SEETIEFE_MINDEST_M = 10.0


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


def _hauptmeer_maske(H):
    """
    Trennt das durchgehende Hauptmeer von abgeschlossenen Binnengewaessern
    (Nutzer-Vorgabe 2026-08-13: "an einem Binnensee soll keine Klippe
    entstehen ... wenn es Kontakt zum Hauptmeer hat dann ist das ok").

    Hauptmeer = die Zusammenhangskomponente von H<=0, die den Kartenrand
    beruehrt. In diesem Weltmodell liegt der Kontinent immer zentral, von
    Ozean bis zum Kartenrand umgeben (kontinentform()) - randberuehrend ist
    hier also ein zuverlaessiges Kriterium, keine Naeherung. Ein Meeresarm,
    der sich weit ins Land zieht, bleibt Teil dieser EINEN Komponente,
    solange er durchgaengig unter 0 m liegt - ein echter Binnensee ist per
    Definition davon getrennt (sonst waere er kein Binnensee).
    """
    see = H <= 0.0
    beschriftet, _ = ndimage.label(see, structure=np.ones((3, 3), dtype=bool))
    rand_ids = set(beschriftet[0, :].tolist()) | set(beschriftet[-1, :].tolist())
    rand_ids |= set(beschriftet[:, 0].tolist()) | set(beschriftet[:, -1].tolist())
    rand_ids.discard(0)
    if not rand_ids:
        return see
    return np.isin(beschriftet, list(rand_ids))


# Ueber welche Strecke die Tiefe von der Wasserlinie auf die Seegrad-Tiefe
# laeuft - siehe _seetiefe_aus_archetyp(). Entspricht UEBERGANG_MEER_M in
# core/vektor_kueste.py; beide beschreiben dieselbe Zone von der Vorgabe
# 2026-08-24, nur von je einer Seite.
UFERUEBERGANG_M = 200.0


def _seetiefe_aus_archetyp(H, felder):
    """
    Meerestiefe in EINER Formel, direkt aus dem Voronoi-Seegrad-Prozess
    (Redesign 2026-08-13, Nutzer-Vorgabe: "ALLE Meerestiefe wird jetzt
    einfach nur noch ueber die Voronoi-Seetiefen-Prozesse erzeugt, also nicht
    im Klippenteil oder so ... die Seetiefe kann die negative Amplitude der
    Klippenhoehen sein (min. -10 m) und einem Seegradfaktor - je weiter weg,
    umso tiefer wird das Meer. Ganz einfach alles in einer Funktion").

    Ersetzt die vorherige eigenstaendige Unterwasser-Eintiefung in
    `_kuesten_umformen()` PLUS die anschliessende Ring-Ausbreitung
    (`_meerestiefe_monoton`, gemessen: Kreuz-vs-Diagonal-Ungleichgewicht bis
    69 m, 7% der Seepixel > 5m Abweichung - die vom Nutzer bemerkten
    diagonalen Linien/Stufen im Wasser). Diese Formel ist NICHT iterativ,
    dadurch auch nicht anfaellig fuer die Gitter-Achsen-Verzerrung einer
    Dilation-basierten Ausbreitung.

        tiefe = -max(SEETIEFE_MINDEST_M, amplitude_des_naechsten_kuestentyps)
                * seegrad_faktor

    `amplitude_des_naechsten_kuestentyps`: die Zielhoehe (`KUESTENHOEHE_M *
    hoehe_faktor`) des naechstgelegenen Kuesten-Archetyps, ueber eine exakte
    euklidische Distanztransformation gefunden (nicht ringweise angenaehert -
    an einer Insel/Bucht bleibt das exakt, ohne Sonderfall-Code).
    `seegrad_faktor`: `felder["seegrad_tiefe"]` (schon vorhandenes, ueber die
    Seegrad-Zellgrenzen geglaettetes Feld aus `seegliederung()`) geteilt durch
    die tiefste Tabellentiefe - 0 knapp an der Kueste, waechst zum offenen
    Meer hin. Kein neues Feld, keine neue Distanztransformation fuer den
    Ferntiefe-Anteil - nur die naechste-Archetyp-Suche ist neu.
    """
    archetyp = felder.get("kuesten_archetyp")
    region_map = felder.get("regionen")
    seegrad_tiefe = felder.get("seegrad_tiefe")
    if archetyp is None or region_map is None or seegrad_tiefe is None:
        return H

    hauptmeer = _hauptmeer_maske(H)
    if not hauptmeer.any():
        return H

    land = H > 0.0
    amplitude_land = np.zeros(H.shape, dtype=np.float64)
    hat_archetyp = np.zeros(H.shape, dtype=bool)
    for i, (_z, _s, r) in enumerate(alle_regionen()):
        archetypen = KUESTEN_ARCHETYPEN.get(r["name"])
        if not archetypen:
            continue
        for lokal_index, typ in enumerate(archetypen):
            treffer = land & (region_map == i) & (archetyp == lokal_index)
            if not treffer.any():
                continue
            amplitude_land[treffer] = KUESTENHOEHE_M * typ["hoehe_faktor"]
            hat_archetyp |= treffer

    if not hat_archetyp.any():
        return H

    # Exakte naechste-Archetyp-Suche (euklidisch, nicht ringweise angenaehert)
    # - liefert fuer jeden Punkt der Karte den naechstgelegenen Land-Archetyp,
    # unabhaengig von der (jetzt kurzen) Reichweite des Archetyp-Bands selbst.
    _abst, index = ndimage.distance_transform_edt(~hat_archetyp, return_indices=True)
    amplitude_je_pixel = amplitude_land[index[0], index[1]]

    # GEGLAETTET - eine reine Naechster-Nachbar-Zuordnung hat selbst harte
    # Kanten, genau dort, wo zwei Archetyp-Zonen sich die Naehe zu einem
    # Seepixel teilen (Voronoi-Grenze zwischen "naechste Klippe A" und "B").
    # Erstmessung ohne Glaettung: Kreuz-vs-Diagonal-Max SCHLECHTER als die
    # vorherige Ring-Loesung (342 m statt 69 m) - die Sprungstelle war nur
    # verschoben, nicht weg. `mpp`-bezogenes Sigma haengt sich an dieselbe
    # Groessenordnung wie die Archetyp-Reichweiten selbst (0.18-0.70 km).
    size = H.shape[0]
    mpp = WELT_KM * 1000.0 / size
    sigma_px = max(2.0, 1200.0 / mpp)
    amplitude_je_pixel = ndimage.gaussian_filter(amplitude_je_pixel, sigma=sigma_px)

    tiefste_tabellentiefe = abs(min(TIEFE_JE_SEEGRAD.values()))
    seegrad_faktor = np.clip(seegrad_tiefe / -tiefste_tabellentiefe, 0.0, 1.5)

    tiefe = -np.maximum(SEETIEFE_MINDEST_M, amplitude_je_pixel) * seegrad_faktor
    tiefe = np.minimum(tiefe, -SEETIEFE_MINDEST_M)

    # DER UFERUEBERGANG (Nutzervorgabe 2026-08-24, woertlich):
    #
    #   *"Wenn x < 0 (Meer): Blende ueber eine Distanz von 200m (x = 0 bis
    #   x = -200) sanft mittels Smoothstep von coast_profile(x) zu
    #   ocean_profile(x) ueber. Fuer x < -200 gilt 100% ocean_profile(x)."*
    #
    # GEMESSEN, was vorher passierte: die Tiefe stand ab dem ersten
    # Seepixel konstant auf -10 m (SEETIEFE_MINDEST_M) und blieb es bis
    # 500 m hinaus, danach -34 m. Es gab also gar keinen Uebergang,
    # sondern eine 10-m-Stufe direkt an der Wasserlinie - im Schnitt eine
    # senkrechte Wand unter Wasser, egal ob dahinter ein Strand oder eine
    # Klippe liegt.
    #
    # `coast_profile` ist auf der Seeseite die Wasserlinie selbst, also 0.
    # Nicht aus Bequemlichkeit: die Vorbildprofile geben unter Wasser
    # nichts her. COP30 ist ein OBERFLAECHENmodell und setzt offenes Meer
    # auf 0 - gemessen liegen alle 27 Archetypen zwischen -0.1 und -1.8 m
    # und aendern sich ueber 250 m nicht. Bathymetrie ist in diesen Daten
    # nicht enthalten; ein gemessenes Seeprofil waere eine Erfindung.
    #
    # Bleibt der Smoothstep von 0 auf die Seegrad-Tiefe ueber
    # UFERUEBERGANG_M. Das ist ein Schelf statt einer Wand, und es ist
    # genau das, was die Vorgabe verlangt.
    if UFERUEBERGANG_M > 0.0:
        mpp = WELT_KM * 1000.0 / float(H.shape[0])
        abstand_m = ndimage.distance_transform_edt(H <= 0.0) * mpp
        t = np.clip(abstand_m / UFERUEBERGANG_M, 0.0, 1.0)
        anteil = t * t * (3.0 - 2.0 * t)
        tiefe = tiefe * anteil

    return np.where(hauptmeer, tiefe, H)


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
    #
    # NUR GEGEN DAS HAUPTMEER, NICHT GEGEN BINNENSEEN (Nutzer-Vorgabe
    # 2026-08-13). Vorher mass `dist_land` den Abstand zum naechsten Gewaesser
    # ueberhaupt - ein Ufer am Binnensee zaehlte damit genauso als "Kueste"
    # wie eines am offenen Meer, und bekam dieselben Klippen-Archetypen. Jetzt
    # zaehlt als Ziel nur `_hauptmeer_maske(H)`: Land neben einem See, das weit
    # vom echten Meer entfernt liegt, faellt automatisch aus dem Kuestenband -
    # dort bleibt das rohe Rauschen unveraendert, wie gewuenscht.
    land = H > 0.0
    hauptmeer = _hauptmeer_maske(H)
    binnensee = (~land) & (~hauptmeer)
    dist_land = ndimage.distance_transform_edt(~hauptmeer) * mpp
    dist_see = ndimage.distance_transform_edt(~land) * mpp
    distanz_m = np.where(land, dist_land, -dist_see)
    # Binnenseen komplett aus dem Kuestenband ausschliessen - ohne diese
    # Zeile koennten Seeuferpixel (kleiner Abstand zu IHREM eigenen Ufer)
    # trotzdem als "Kuestenlinie" durchrutschen.
    distanz_m = np.where(binnensee, -np.inf, distanz_m)

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
    # Reichweite je Pixel - gefuellt aus der `reichweite_km` des dort
    # zugeordneten Archetyps (3.10). Ersetzt die frueher globale Bandbreite.
    reichweite_karte = np.zeros(H.shape, dtype=np.float32)
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
        # Lokale Terrainhoehe je Pixel (3.11) - dieselbe naechster-Saatpunkt-
        # Zuordnung wie fuer den Archetyp, nur mit dem schon vorhandenen
        # `lokale_hoehe`-Wert des Saatpunkts statt seinem Archetyp-Index.
        # Stueckweise konstant je Saatpunkt-Zelle, wird von der Zonengrenzen-
        # Glaettung am Ende der Funktion mitgeglaettet - kein eigener
        # Glaettungsschritt noetig.
        lokale_hoehe_karte = np.zeros(H.shape, dtype=np.float32)
        for pixel_idx, punkt_idx in enumerate(naechster):
            y, x = idx_pixel_y[pixel_idx], idx_pixel_x[pixel_idx]
            archetyp_id_karte[y, x] = name_zu_index[zuordnung[punkt_idx]["name"]]
            lokale_hoehe_karte[y, x] = lokale_hoehe[punkt_idx]
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
            # ZIELHOEHE ALS MISCHUNG AUS TABELLE UND LOKALEM ROHGELAENDE (3.11,
            # Nutzer-Vorgabe 2026-08-13: "Cliffs of Moher stehen auf flachem
            # Terrain, das bekommt auch ein Gradient nicht logisch hin - die
            # klippenhoehe soll von der lokalen terrainhoehe abhaengen, aber
            # abgeschwaecht"). Ein weicherer UEBERGANG loest das Problem nicht,
            # weil nicht die KANTE falsch war, sondern die absolute Zielhoehe
            # selbst, unabhaengig vom Umfeld. `lokale_hoehe_karte` (oben, aus
            # der ohnehin schon berechneten Saatpunkt-Hoehe) liefert die lokale
            # Referenz je Pixel; negative (See-)Werte werden nicht als
            # Referenz zugelassen, sonst koennte tiefes Wasser die Klippe nach
            # unten ziehen.
            ziel_hoehe_tabelle = KUESTENHOEHE_M * archetyp["hoehe_faktor"]
            lokal_ref = np.clip(lokale_hoehe_karte[zone], 0.0, None)
            ziel_hoehe_effektiv = (ziel_hoehe_tabelle * (1.0 - KUESTEN_LOKALER_EINFLUSS)
                                   + lokal_ref * KUESTEN_LOKALER_EINFLUSS)

            winkel = min(archetyp["winkel_grad"], MAX_KLIPPENWINKEL_GRAD)
            # Reichweite aus dem gewuenschten Neigungswinkel abgeleitet, statt
            # aus einer festen, hoehenunabhaengigen Konstante (Nutzerbefund
            # 2026-08-12: gezackte "Mauer" an jeder Kueste in 3D). Fuer
            # `H0*(1-exp(-d/L))` ist die Anfangssteigung bei d=0 genau H0/L -
            # `L = H0/tan(winkel)` setzt also direkt die tatsaechliche Steigung
            # an der Kuestenlinie auf den gewuenschten Winkel. `H0` ist jetzt
            # `ziel_hoehe_effektiv` (Array, je Pixel) statt der festen
            # Tabellenzahl - eine durch die lokale Referenz gedaempfte Klippe
            # bekommt automatisch eine kuerzere, eine durch eine hohe lokale
            # Referenz verstaerkte automatisch eine laengere Anstiegsstrecke.
            skala_m = np.maximum(2.0 * mpp,
                                 ziel_hoehe_effektiv / max(np.tan(np.radians(winkel)), 0.05))
            hoehe_lokal = np.where(strand_feld[zone], ziel_hoehe_effektiv * 0.25, ziel_hoehe_effektiv)
            d = distanz_m[zone]
            d_land = np.clip(d, 0.0, None)
            profil = hoehe_lokal * (1.0 - np.exp(-d_land / skala_m))
            if archetyp["kantig"]:
                facetten_seed = basis_seed + i * 97 + name_zu_index[archetyp["name"]] * 13 + 1
                facetten = _kuesten_rauschen_lokal(zone, facetten_seed, sigma=max(1.0, band_px * 0.15))
                # RAUSCHEN AN DER WASSERLINIE AUSBLENDEN (3.11, Ursache der
                # "zelligen"/blobartigen Kueste nach dem Reichweiten-Fix - siehe
                # Bild vom 2026-08-13). Die Facette wurde bisher mit VOLLER
                # Staerke ueberall in der Zone addiert, auch exakt bei d=0, wo
                # das Basisprofil bewusst gegen 0 geht, damit die Kuestenlinie
                # sauber definiert bleibt. Eine Wackelamplitude von ±zig Metern
                # GENAU an der Stelle, wo ueber Land/See entschieden wird, kippt
                # das Vorzeichen zufaellig hin und her - sichtbar als winzige
                # Inseln/Buchten statt einer glatten Linie. Jetzt faehrt die
                # Staerke von 0 an der Wasserlinie auf voll hoch, sobald der
                # Anstieg im Wesentlichen abgeschlossen ist (d_land >= skala_m)-
                # Textur an der KlippenFLAECHE bleibt, die Kuestenlinie selbst
                # bleibt sauber.
                rausch_gewicht = np.clip(d_land / skala_m, 0.0, 1.0)
                profil = profil + (facetten[zone] - 0.5) * 2.0 * hoehe_lokal * 0.15 * rausch_gewicht

            # NUR NOCH DAS LAND FORMEN (Redesign 2026-08-13, Nutzer-Vorgabe:
            # "ALLE Meerestiefe wird jetzt einfach nur noch ueber die Voronoi-
            # Seetiefen-Prozesse erzeugt. also nicht im Klippenteil oder so").
            # Die Seeseite wurde hier zuvor eigenstaendig eingestochen und
            # anschliessend per Ring-Ausbreitung monoton gebuegelt
            # (`_meerestiefe_monoton()`, 3.13) - GEMESSEN erzeugte diese
            # Ausbreitung ein Kreuz-vs-Diagonal-Ungleichgewicht (median 0.12 m,
            # aber bis 69 m, 7% der Seepixel > 5m Abweichung) - sichtbar als
            # die vom Nutzer bemerkten diagonalen Linien/Stufen im Wasser.
            # `_seetiefe_aus_archetyp()` (weiter unten) ersetzt das durch eine
            # einzige, nicht-iterative Formel direkt aus dem Voronoi-Seegrad-
            # Feld - dieser Block schreibt die Seeseite deshalb gar nicht mehr,
            # `veraendert` bleibt dort False und die Zonenglaettung/-blendung
            # weiter unten laesst diese Pixel folgerichtig unangetastet.
            land_in_zone = d >= 0.0
            ziel_hoehe[zone] = np.where(land_in_zone, profil, ziel_hoehe[zone])
            veraendert[zone] = veraendert[zone] | land_in_zone

            # Reichweite der Blendmaske waechst mit der effektiven Hoehe mit
            # (3.11, Nutzer-Vorgabe "wenn die klippen sehr hoch sind, dann
            # strahlen diese auch etwas weiter rein, damit es realistischer
            # aussieht"): mindestens die Tabellenreichweite, aber nie kuerzer
            # als das, was das Anstiegsprofil selbst braucht, um weitgehend
            # anzukommen (KUESTEN_REICHWEITE_SKALA_FAKTOR * skala_m deckt rund
            # 97% des Anstiegs ab) - sonst wuerde die AEUSSERE Maske vor dem
            # INNEREN Profil abschneiden.
            reichweite_karte[zone] = np.maximum(
                archetyp["reichweite_km"] * 1000.0,
                KUESTEN_REICHWEITE_SKALA_FAKTOR * skala_m)
            veraendert[zone] = True

    if not np.any(veraendert):
        return H

    # Zonengrenzen (Naechster-Punkt-Zuordnung hat harte Kanten) entschaerfen -
    # gleiche Ueberlegung wie bei den geglaetteten Klimafeldern.
    #
    # GETRENNT FUER LAND UND SEE (2026-08-13). Vorher lief EIN gaussian_filter
    # ueber das ganze Feld und damit quer ueber die Kuestenlinie: dort stossen
    # Landwerte von mehreren hundert Metern auf Seewerte von -175 m, das
    # Mittel daraus landet nahe null. Der absichtlich tief eingestochene
    # Klippenfuss wurde so wieder hochgezogen und anschliessend von der
    # Vorzeichen-Klemme unten auf -0.5 m gepinnt - gemessen kam die Seetiefe
    # an der Kueste dadurch auf -1.7 m statt der erwarteten Tiefe, also
    # FLACHER als ganz ohne den Pass (-8.1 m). Maskiertes Glaetten
    # (normalisierte Faltung je Seite) haelt die Kante an der Wasserlinie und
    # glaettet trotzdem die Zonengrenzen INNERHALB jeder Seite.
    # SIGMA KLEIN HALTEN (Fehlerbehebung 2026-08-13). Vorher `band_px * 0.2`,
    # bei 384 px also rund 140 m - in derselben Groessenordnung wie die
    # Anstiegsstrecke einer Klippe (Moher: 172 m). Die Glaettung hat das
    # Profil damit praktisch eingeebnet: gemessen kamen von 82 Grad Sollwinkel
    # nur 26.5 Grad an, ueber alle 23 Zonen im Mittel 29 Grad zu flach.
    # Diese Glaettung soll die SEITLICHEN Zonennaehte entschaerfen (harte
    # Kanten der Naechster-Punkt-Zuordnung), nicht die Klippenfront - dafuer
    # genuegen ein bis zwei Pixel. Nach oben gedeckelt, damit sie bei jeder
    # Bandbreite und Aufloesung klein gegen die Profilstrecke bleibt.
    # Gemessen 2026-08-13: mit sigma = 2 px (bei 384 px sind das 111 m) lag
    # die Glaettung GENAU auf der Anstiegsstrecke der Klippen (Moher 111 m,
    # Bretagne 132 m) und ebnete sie ein - steile Typen kamen mit 34.8 statt
    # 74 Grad an. Mit 0.8 px bleiben die Zonennaehte weich, das Profil aber
    # erhalten (steile Typen danach 46.5 Grad gemessen).
    sigma_zone = float(np.clip(band_px * 0.2, 0.5, 0.8))

    def _glaetten_in_maske(feld, maske):
        m = maske.astype(np.float32)
        gewicht = ndimage.gaussian_filter(m, sigma=sigma_zone)
        summe = ndimage.gaussian_filter(feld * m, sigma=sigma_zone)
        return np.where(gewicht > 1e-6, summe / np.maximum(gewicht, 1e-6), feld)

    ist_land = H > 0.0
    ziel_hoehe_glatt = np.where(
        ist_land,
        _glaetten_in_maske(ziel_hoehe, ist_land),
        _glaetten_in_maske(ziel_hoehe, ~ist_land))

    # BLENDSTAERKE JE ARCHETYP STATT GLOBAL (3.10). Vorher lief hier eine
    # lineare Rampe ueber die EINE globale Bandbreite (1.2 km) - dadurch trugen
    # 96 % der Landflaeche einen Kuestenwert und der Pass verschob die
    # Regions-Hangeichung (3.9). Jetzt zaehlt die `reichweite_km` des jeweils
    # zugeordneten Archetyps, und der Abfall ist quadratisch statt linear:
    # kraeftig direkt an der Wasserlinie, dann zuegig verblassend - genau die
    # Vorgabe "kraeftige farbe an der kueste, dann verblassen ins land hinein".
    with np.errstate(divide="ignore", invalid="ignore"):
        anteil = np.where(reichweite_karte > 0.0,
                          np.abs(distanz_m) / np.maximum(reichweite_karte, 1e-6),
                          np.inf)
    # VERLAUF: innen voll, aussen weich auslaufend (Fehlerbehebung 2026-08-13).
    # Die vorherige quadratische Rampe fiel schon direkt an der Wasserlinie
    # ab - der Median der Staerke im Kuestensaum lag bei 0.56, es kam also nur
    # gut die Haelfte des vorgegebenen Profils an, der Rest blieb Rohgelaende.
    # Genau deshalb trafen die festgelegten `winkel_grad` nicht. Jetzt bleibt
    # die Staerke im inneren Viertel der Reichweite nahezu voll und faellt
    # danach glatt (Smoothstep) auf null - die Klippe bekommt ihren Winkel,
    # der Uebergang ins Landesinnere bleibt trotzdem ohne Kante.
    t = np.clip((anteil - 0.25) / 0.75, 0.0, 1.0)
    staerke = (1.0 - t * t * (3.0 - 2.0 * t)) * 0.95
    staerke = np.where(veraendert, staerke, 0.0)
    felder["kuesten_staerke"] = staerke.astype(np.float32)

    # Archetyp-Zuordnung dort zuruecknehmen, wo der Pass praktisch nicht mehr
    # wirkt - sonst meldet das Feld einen Kuestentyp fuer Land, an dem nichts
    # geformt wurde (genau der Eindruck "die kuestenbereiche erstrecken sich
    # ueber das gesamte land"). Die 2D-Anzeige liest dasselbe Feld.
    felder["kuesten_archetyp"] = np.where(
        staerke > 0.02, felder["kuesten_archetyp"], -1).astype(np.int8)

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

# Gemessene Anteile am 2026-08-22, 1024 px - Grundlage der Ladebalken-
# verteilung in weltfeld(). Beim naechsten groesseren Umbau nachfuehren.
WELTFELD_PLAN = (
    ("kontinentform", 3.0),
    ("voronoi_regionen", 6.0),
    ("parameterfelder", 2.0),
    ("seegliederung", 8.0),
    ("oktavenstapel", 4.0),
    ("oktaven_mischen", 6.0),
    ("potenzkurve", 2.0),
    ("grundhoehe_und_schelf", 3.0),
    ("kuestenform_potenz", 2.0),
    ("vk_linie_bauen", 25.0),
    ("vk_als_raster", 25.0),
    ("vk_archetyp_felder", 10.0),
    ("kuesten_umformen_raster", 60.0),
    ("kuestengebiete", 6.0),
    ("seetiefe", 4.0),
)


def weltfeld(size, seed, punktzahl=200, tiefe_skala_m=1400.0, shader_manager=None,
             schritte=None, kuesten_aktiv=True, kontinentform_regler=None,
             regionen_ueberschreibung=None):
    """
    Die Hoehenkarte der ganzen Welt, in Metern. Unter 0 ist Meer.

    Rueckgabe: (H, felder) - felder enthaelt jeden Regler als volles Feld,
    damit spaetere Stufen (Fluesse, Taeler) ortsabhaengig arbeiten koennen.

    `regionen_ueberschreibung` ist {Regionsname: {Reglername: Wert}} aus dem
    Regionsreiter. Sie ersetzt einzelne Katalogwerte NUR FUER DIESEN LAUF -
    siehe `parameterfeld()`.

    `kontinentform_regler` ist der Formregler des Kontinents (0 rund,
    0.5 laenglich, 1 viele Auslaeufer, siehe `_kontinent_gestalt()`). None
    behaelt die bis zum 2026-08-26 gueltige Gestalt bei - und daran haengt
    jede Eichung, also ist None hier keine Bequemlichkeit, sondern Pflicht,
    solange der Nutzer nichts anderes einstellt.

    `kuesten_aktiv=False` UEBERSPRINGT die Kuestenformung ganz - weder der
    Vektorweg noch `_kuesten_umformen()` laufen, die Kuestenlinie bleibt so,
    wie das Rauschen sie gezogen hat. Gedacht zum Vergleichen: erst damit
    laesst sich sehen, welchen Anteil die Archetypen am Bild haben
    (Nutzerwunsch 2026-08-25: *"checkboxen ... mit denen ich die effekte
    immer auch ausschalten kann"*).

    Die Folgestufe `_seetiefe_aus_archetyp()` vertraegt das: sie liest ihre
    Felder mit `felder.get()` und steigt ohne sie sofort aus - genauso wie
    beim bereits vorhandenen Rasterweg, der `vektor_kueste` ebenfalls nicht
    setzt.

    `schritte` ist ein optionales managers.teilschritte.Teilschritte-Objekt.
    Ohne eines aendert sich nichts - Tools und Smoke-Tests rufen weltfeld()
    unveraendert auf.
    """
    from managers.teilschritte import schritt as _s

    with _s(schritte, "kontinentform"):
        maske, sdf = kontinentform(size, seed, shader_manager,
                                   form=kontinentform_regler)
    with _s(schritte, "voronoi_regionen"):
        gewichte, zell_etikett = voronoi_regionen(
            maske, seed, punktzahl=punktzahl, shader_manager=shader_manager)

    felder = {}
    KLIMAFELDER = ("temp_mittel_m0", "temp_spanne", "niederschlag_mm", "wind_mittel_ms")
    _p_ctx = _s(schritte, "parameterfelder")
    _p_ctx.__enter__()
    for name in REGLER:
        # Die Klimafelder schaerfer mischen als die Gelaendefelder - siehe
        # KLIMA_SCHAERFE. Das Relief braucht die breite Ueberblendung, das
        # Klima verliert dadurch seine Hoehenabhaengigkeit.
        felder[name] = parameterfeld(
            name, gewichte,
            KLIMA_SCHAERFE if name in KLIMAFELDER else 1.0,
            ueberschreibung=regionen_ueberschreibung)

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
    _p_ctx.__exit__(None, None, None)

    # SEEGLIEDERUNG (docs/spezifikation/12_WASSER.md Abschnitt 7) - braucht `maske` (Kontinentform)
    # und `gewichte` (fuer die Uferregionen), beide stehen jetzt. Das Ergebnis
    # (seegrad_tiefe) wird weiter unten anstelle des alten Kuestenschelfs
    # angewandt - siehe "SEEGRAD-SCHELF" dort.
    with _s(schritte, "seegliederung"):
        see = seegliederung(maske, gewichte, seed, punktzahl_land=punktzahl,
                            shader_manager=shader_manager)
    felder["seegrad"] = see["seegrad"]
    felder["seegrad_tiefe"] = see["seegrad_tiefe"]
    felder["ufer_region_a"] = see["ufer_region_a"]
    felder["ufer_region_b"] = see["ufer_region_b"]
    felder["see_eis"] = see["see_eis"]

    with _s(schritte, "oktavenstapel"):
        stapel = oktavenstapel(size, seed, shader_manager)
    wellen = _wellenlaengen()

    # OKTAVENGEWICHTE JE PIXEL.
    #
    # Eine Oktave zaehlt nur, wenn ihre Wellenlaenge in die Formgroesse der
    # Region passt. Der Uebergang ist weich - eine harte Grenze wuerde beim
    # Wandern ueber eine Regionsgrenze schlagartig eine ganze Oktave zu- oder
    # abschalten, und das saehe man als Kante.
    _o_ctx = _s(schritte, "oktaven_mischen")
    _o_ctx.__enter__()
    relief = np.zeros((size, size), dtype=np.float64)
    summe = np.zeros((size, size), dtype=np.float64)
    form = felder["formgroesse_m"]
    rauheit = np.clip(felder["rauheit"], 0.2, 0.9)
    # DAS TOR IST ZWEISEITIG (2026-08-25). Vorher war es das nicht, und das
    # war die Ursache der Alpenspitzen.
    #
    # Nutzermeldung: *"die alpen brauchen hier viiiiiel weniger spitzen. also
    # sowas wie 5 berge auf der regionsflaeche. hier sind hunderte zu sehen.
    # amplitude ist ok, aber wir brauchen fuer den bereich noisewellen die
    # bis zum tal herabfallen und wieder hoch gehen und halt auf der flaeche
    # so 5 stueck."*
    #
    # Das obere Tor daempft Oktaven, die GROESSER sind als die Formgroesse.
    # Nach unten lief es dagegen offen bis zur feinsten Welle (47 m),
    # gebremst nur durch `rauheit^k` - und das ist zu schwach. GEMESSEN,
    # Anteil des Reliefs in Wellenlaengen <= 375 m:
    #
    #     Region              form   relief   Anteil   in Metern   Formen/Region
    #     Nevadin           3800    1050 m   23.9 %      250 m       3.5
    #     Thalassia  1100     404 m   30.2 %      122 m      41.7
    #     Nebelrode       1400     135 m   31.4 %       42 m      25.7
    #     Morobora               3000     118 m    5.5 %        6 m       5.6
    #
    # Die GROSSEN Formen des Alpenlands stimmten bereits (3.5 Massive auf
    # der Regionsflaeche gegen den Wunsch von 5). Der Fehler waren die
    # 250 m Amplitude in Formen unter 375 m Breite - bei 1024 px ist die
    # feinste Welle zwei Pixel breit. Genau das zerhackte auch den
    # Geologie-Querschnitt.
    #
    # WARUM DAS UNTERE TOR AN DER FORMGROESSE HAENGT und nicht an einer
    # festen Meterzahl: es soll das VERHAELTNIS begrenzen. Eine Region mit
    # 3800-m-Massiven darf nicht dieselben 47-m-Zacken tragen wie eine mit
    # 1100-m-Huegeln - im ersten Fall sind das Nadeln auf einem Berg, im
    # zweiten normale Rauheit. Eine feste Grenze wuerde entweder die feinen
    # Regionen glattbuegeln oder die groben nicht erreichen.
    # EINE FASSUNG DER FORMEL, siehe oktavengewicht() weiter oben.
    for k in range(OKTAVEN):
        gewicht = oktavengewicht(k, form, rauheit)
        relief += gewicht * stapel[k]
        summe += gewicht
    relief /= np.maximum(summe, 1e-9)

    # Auf 0..1 mit FESTER Spreizung, dann die Potenzkurve. <1 hebt an
    # (Hochflaeche), >1 drueckt herunter (weite Ebene mit einzelnen Gipfeln).
    _o_ctx.__exit__(None, None, None)
    _q_ctx = _s(schritte, "potenzkurve")
    _q_ctx.__enter__()
    t = np.clip(0.5 + SPREIZUNG * relief, 0.0, 1.0)
    potenz = np.clip(felder["potenz"], 0.2, 4.0)

    # DIE POTENZ UM DEN MEDIAN DREHEN, nicht um die Null.
    #
    # t^p verschiebt den Median von 0.5 auf 0.5^p - die Potenz aenderte damit
    # nicht nur die FORM, sondern auch die mittlere Hoehe. Gemessen: die
    # Estrande stand bei 73 % Wasser statt 45, das Skerrheim bei 0 statt
    # 20, obwohl an hoehe_m nichts falsch war. Zwei Regler, die sich
    # gegenseitig verstellen, sind nicht eichbar.
    #
    # Mit der Rueckverschiebung um 0.5^p - 0.5 bleibt der Median bei 0.5:
    # hoehe_m ist dann wirklich die mittlere Hoehe, und potenz formt nur noch.
    t = np.clip(np.power(t, potenz) - np.power(0.5, potenz) + 0.5, 0.0, 1.0)

    # DIE KUESTENFORM (2026-08-07).
    #
    # Der Nutzer hat je Region beschrieben, WIE das Land ins Meer uebergehen
    # soll: das Nebelrode "keine klippe ins meer sondern in das meer
    # abfallen", die Morobora "sanft abfallend mit kleinen inseln", das Skerrheim
    # "hohe huegel und tiefe graeben die ins meer gehen", die Samarcia "nur
    # teilweise klippen und diese gering", das Clonagh Klippen mit Buchten.
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

    _q_ctx.__exit__(None, None, None)
    _g_ctx = _s(schritte, "grundhoehe_und_schelf")
    _g_ctx.__enter__()
    # hoehe_m ist die MITTE, nicht der Boden - deshalb t - 0.5.
    H = felder["hoehe_m"] + (t - 0.5) * felder["relief_m"]

    # DAS MEER AUSSERHALB DER FORM.
    #
    # Kein Kastengradient mehr: abgesenkt wird nach dem ABSTAND ZUR KUESTE, und
    # die Kueste ist der Rand der Plaetzchenform. Damit folgt der Meeresboden
    # der Landmasse statt dem Kartenrand.
    #
    # Innerhalb der Form bleibt das Regionengelaende unveraendert - die Buchten
    # von Macchia und Griechischen Inseln entstehen weiter aus deren
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
    #     gebraucht wurde - das Nebelrode steigt binnen 300 m auf 163 m,
    #     und ein 63-m-Band erreichte das mit Gewicht 0.001.
    #  2. Ein ABSTANDSband exp(-(d/500 m)^2) griff, machte die flachen
    #     Regionen aber STEILER statt flacher (Morobora 8.9 -> 12.1 Grad): ein
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
    _g_ctx.__exit__(None, None, None)
    _k_ctx = _s(schritte, "kuestenform_potenz")
    _k_ctx.__enter__()
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
    # DIE KUESTENFORMUNG - Raster oder Vektor (docs/spezifikation/11_GELAENDE.md).
    #
    # `_kuesten_umformen()` arbeitet auf Pixelmasken und zieht seine
    # Saatpunkte per Index aus einer Pixelliste; dadurch haengt das Ergebnis
    # an der Aufloesung (gemessen: 34.7 m mittlere Abweichung zwischen 256
    # und 512 px gegen 11.9 m ohne diesen Pass).
    #
    # Der Vektorweg (core/vektor_kueste.py) beschreibt dieselbe Kueste als
    # Polylinie mit Stationen in METERN, bekommt Profilform, Zielhoehe und
    # Reichweite aus den an 19 realen Vorbildkuesten gemessenen Werten, und
    # ist an beliebigen Fliesskommakoordinaten auswertbar - dieselbe Funktion
    # bedient Rasterkarte und Mesh.
    #
    # Umschaltbar, weil das ein Eingriff in die Gelaendeform ist und die
    # Regionseichung daran haengt (docs/OFFENE_PUNKTE.md 3.9).
    _k_ctx.__exit__(None, None, None)
    from gui.config.value_default import VEKTOR_KUESTE_AKTIV
    if not kuesten_aktiv:
        # ABGESCHALTET - siehe den Kopf dieser Funktion. Laut protokollieren,
        # weil eine stumm uebersprungene Stufe von einem Erfolg nicht zu
        # unterscheiden waere (CLAUDE.md, "jeder stille Rueckfall braucht
        # eine laute Logzeile").
        import logging
        logging.getLogger(__name__).info(
            "Kuestenformung UEBERSPRUNGEN (kuesten_aktiv=False) - "
            "keine Archetypen, keine Klippen, keine Seetiefe aus Archetyp")
    elif VEKTOR_KUESTE_AKTIV:
        from core.vektor_kueste import VektorKueste, als_raster
        # DREI Teilschritte statt einem: der Bau der Kuestenlinie (Konturen,
        # Saaten, Segmente), das Abtasten auf das Raster und die
        # Archetypfelder sind voellig verschiedene Arbeiten, und nur die
        # Aufteilung zeigt, welche davon die Zeit frisst.
        with _s(schritte, "vk_linie_bauen"):
            _vk = VektorKueste(H, felder["regionen"], seed, welt_km=WELT_KM)
        felder["vektor_kueste"] = _vk
        with _s(schritte, "vk_als_raster"):
            H = als_raster(_vk)
        # ARCHETYPFELDER NACHTRAGEN - Pflicht, nicht Kosmetik:
        # `_seetiefe_aus_archetyp()` gleich darunter braucht sie, um ueberhaupt
        # zu wirken, und die 2D-Anzeige liest sie fuer den Kuestentypen-Modus
        # (docs/OFFENE_PUNKTE.md 3.8/3.13). Mit leerem Feld blieb die See
        # gemessen auf Meereshoehe stehen.
        with _s(schritte, "vk_archetyp_felder"):
            felder["kuesten_archetyp"], felder["kuesten_staerke"] = \
                _vk.archetyp_felder(H)
    else:
        with _s(schritte, "kuesten_umformen_raster"):
            H = _kuesten_umformen(H, felder, seed, size)

    # DIE KUESTENGEBIETE IM HINTERLAND (docs/archiv/2026-08-25_AUFRAEUMPLAN.md 4.8).
    #
    # HIER und nicht frueher: die Archetypfelder stehen erst seit der Zeile
    # darueber. Vor der Seetiefe, weil das Delta nur auf Land wirkt und die
    # Wasserlinie nicht verschieben darf - siehe den Kopf von
    # `kuestengebiete()`.
    with _s(schritte, "kuestengebiete"):
        gebiets_delta, gebiet_raster, hinterland_m = kuestengebiete(
            H, felder, zell_etikett, seed)
        felder["gebiets_delta_m"] = gebiets_delta
        felder["gebiet"] = gebiet_raster
        felder["hinterlandhoehe_m"] = hinterland_m
        # Die Voronoi-Zellen selbst - fuer die Hoehenfaktor-Ansicht
        # (Nutzerwunsch 2026-08-26). Bis hierher blieben sie in weltfeld().
        felder["voronoi"] = zell_etikett.astype(np.int32)
        H = H + gebiets_delta

    # SEETIEFE - EINE FORMEL FUER DAS GANZE HAUPTMEER (Redesign 2026-08-13,
    # siehe _seetiefe_aus_archetyp() weiter oben fuer die volle Begruendung).
    #
    # Vorgeschichte, zur Einordnung: hier standen bis zum 2026-08-13 drei
    # aufeinander aufbauende Schritte - eine Mindesttiefe aus der reinen
    # Seegrad-TABELLE (docs/OFFENE_PUNKTE.md 3.2), eine Anhebung des rohen
    # Rauschens auf -10 m davor (3.7), und ein weicher Uebergang zur
    # Kuestenform (3.10/3.11) - gefolgt von einer nachtraeglichen Ring-
    # Ausbreitung, die Monotonie erzwang (3.13). Die Ring-Ausbreitung
    # erzeugte dabei ein GEMESSENES Kreuz-vs-Diagonal-Ungleichgewicht (median
    # 0.12 m, bis 69 m, 7% der Seepixel > 5m) - im 3D-Bild sichtbar als
    # diagonale Linien/Stufen im Wasser (Nutzerbefund 2026-08-13). Die
    # Kuestenform selbst wird seither ausschliesslich vom Archetyp bestimmt,
    # die Seegrad-TABELLE (`TIEFE_JE_SEEGRAD`) liefert nur noch den reinen
    # Ferntiefe-Faktor - beides in einer einzigen, nicht-iterativen Formel.
    with _s(schritte, "seetiefe"):
        H = _seetiefe_aus_archetyp(H, felder)

    with _s(schritte, "hangfeuchte"):
        felder["niederschlag_mm"] = _hangfeuchte(H, felder, size)

    return H.astype(np.float64), felder


def _hangfeuchte(H, felder, size):
    """
    Suedhaenge trocknen aus, Nordhaenge bleiben feucht.

    NUTZERVORGABE 2026-08-24: *"Skerrheim, voronoi mit viel suedhang ist
    etwas trockener als fjordland nordhang, dann Samarcia suedhang total
    trocken, nordhang etwas feuchter."*

    WARUM ERST HIER, am Ende von weltfeld(). Die Parameterfelder entstehen
    weiter oben aus den Voronoi-Gewichten, und zu dem Zeitpunkt gibt es
    noch gar kein Gelaende - `H` steht erst ab der Grundhoehenbildung. Eine
    Hangausrichtung braucht aber Haenge.

    NICHT JE VORONOI-ZELLE, sondern PUNKTWEISE. Der Nutzer hatte nach
    Zellen gefragt; der Gradient ist aber ohnehin feiner und
    trifft die Sache genauer: eine Zelle kann Nord- und Suedhaenge
    enthalten.

    DIE STAERKE koppelt an die Hangneigung. Auf einer Ebene gibt es keine
    Exposition, und ohne diese Kopplung bekaeme flaches Land zufaellige
    Feuchteunterschiede aus Rundungsrauschen im Gradienten.

    Y WAECHST NACH SUEDEN in diesem Projekt (Zeile 0 ist Norden, siehe
    regionsname()). Ein Suedhang faellt also mit wachsendem y ab, hat
    demnach dH/dy < 0 - deshalb das Minuszeichen.
    """
    staerke_feld = felder.get("hang_trockenheit")
    nied = np.asarray(felder["niederschlag_mm"], dtype=np.float64)
    if staerke_feld is None:
        return nied

    mpp = WELT_KM * 1000.0 / float(size)
    dy, dx = np.gradient(np.asarray(H, dtype=np.float64), mpp)
    betrag = np.hypot(dx, dy)

    # Suedexposition: +1 voller Suedhang, -1 voller Nordhang, 0 eben.
    sued = np.where(betrag > 1e-9, -dy / np.maximum(betrag, 1e-9), 0.0)

    # Auf flachem Land keine Exposition. HANG_VOLL_GRAD ist die Neigung,
    # ab der die Modulation voll wirkt - darunter waechst sie linear an.
    neigung = np.degrees(np.arctan(betrag))
    wirkung = np.clip(neigung / HANG_VOLL_GRAD, 0.0, 1.0)

    faktor = 1.0 - np.asarray(staerke_feld, dtype=np.float64) * sued * wirkung
    return nied * np.clip(faktor, 0.05, 3.0)

