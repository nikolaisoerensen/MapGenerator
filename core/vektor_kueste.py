"""
Path: core/vektor_kueste.py

DIE KUESTE ALS VEKTOR - eine Hoehenfunktion, zwei Abtaster.

Nutzer-Vorgabe 2026-08-17: *"wir brauchen zwei hoehenmaps die immer
verfuegbar sind. 1 ohne diese veraenderung zur erstellung des meshs und dann
aendern wir immer das mesh entsprechend mit vektoren (damit wir keine
gridbindung mehr haben). und eine zweite heightmap die vektoriell veraenderte
heightmap die gridgebunden darstellt wie die hoehe an jedem punkt ist. somit
sollten mesh und angepasste heightmap sehr aehnlich sein, aber einmal
gridgebunden und einmal freie punkte."*

DIE EINE REGEL, AN DER ALLES HAENGT

Es gibt GENAU EINE Funktion, die sagt, wie hoch das Gelaende an einer Stelle
ist: `VektorKueste.hoehe(x, y, ...)`. Sie nimmt beliebige
FLIESSKOMMAkoordinaten, kein Rasterindex.

    Raster-Abtaster  ->  Pixelmitten            ->  Heightmap B
    Punkt-Abtaster   ->  freie Vertexpositionen ->  Mesh

Beide rufen DIESELBE Funktion. Der Unterschied zwischen Mesh und Heightmap B
ist damit ausschliesslich die Abtastdichte - kein Modellunterschied. Wuerden
Mesh und Raster jeweils eigenen Code fuer "was macht dieser Vektor hier mit
der Hoehe" haben, liefen sie unweigerlich auseinander; genau davor warnt
SPEZIFIKATION §4.5, und genau das ist diesem Projekt schon mehrfach passiert.

WAS SICH GEGENUEBER `_kuesten_umformen()` AENDERT

Die heutige Fassung in `core/terrain_weltkarte.py` rechnet auf ZONENMASKEN:
`zone = (archetyp_id_karte == k) & region_maske`, dann numpy-Operationen auf
`H[zone]`. Das ist schnell, aber an das Raster gebunden - eine Maske hat
keinen Wert "zwischen zwei Pixeln".

Hier ist dieselbe Formel als PUNKTAUSWERTUNG geschrieben:

    * Kuestenlinie aus `kuestenlinien()` (Marching Squares, interpoliert also
      ZWISCHEN den Pixeln - das ist die eigentliche Quelle der Rasterfreiheit)
    * Saatpunkte darauf, je Saatpunkt ein Archetyp und eine lokale
      Referenzhoehe (gleiche Zuteilungsregel wie heute: anspruchsvollster
      Archetyp zuerst, Quote aus `max_anteil`, Seed-Jitter)
    * zwei cKDTrees - einer ueber die dicht abgetastete Linie (fuer den
      ABSTAND), einer ueber die Saatpunkte (fuer den ARCHETYP). Beide
      beantworten Anfragen an beliebigen Float-Koordinaten.

    profil(d) = hoehe_lokal * (1 - exp(-d / skala))
    skala     = max(mindest_skala_m, ziel_hoehe / tan(winkel))

Das ist Zeile fuer Zeile dieselbe Kurve wie heute (siehe `_kuesten_umformen`,
Abschnitt "profil = hoehe_lokal * (1.0 - np.exp(-d_land / skala_m))").

`mindest_skala_m` IST DER EINZIGE UNTERSCHIED ZWISCHEN DEN ABTASTERN, und
das mit Absicht: heute steht dort `2 * mpp`, also eine Pixelgroesse. Eine
Klippe, die ihre volle Hoehe in weniger als zwei Pixeln erreicht, ist im
Raster nicht darstellbar (6.17/6.19) - im MESH dagegen sehr wohl, dort
duerfen zwei Vertices beliebig nah beieinanderstehen. Der Rasterabtaster
uebergibt deshalb `2 * mpp`, der Punktabtaster einen kleinen festen Wert.
Dieselbe Funktion, ein bewusst verschieden gesetzter Parameter - und genau
dieser Parameter ist es, der die Rastertreppe erzeugt.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from core.terrain_weltkarte import (KUESTEN_ARCHETYPEN, KUESTENHOEHE_M,
                                    KUESTEN_LOKALER_EINFLUSS,
                                    MAX_KLIPPENWINKEL_GRAD, WELT_KM,
                                    alle_regionen)


# Abstand der Saatpunkte auf der Kuestenlinie, in METERN. Bestimmt, wie
# kleinteilig die Archetypen laengs der Kueste wechseln. In Metern und nicht
# in Pixeln, damit jede Aufloesung dieselben Zonen bekommt (§10).
SAAT_ABSTAND_M = 420.0

# Abtastabstand der Linie selbst, in METERN - nur fuer die Abstandsmessung.
# Dichter als die Saatpunkte, weil daraus die Entfernung zur Wasserlinie
# kommt und ein grober Abtast die Kurve durch Sehnen ersetzen wuerde.
LINIEN_ABSTAND_M = 60.0

# Mindest-Anstiegsstrecke fuer den PUNKT-Abtaster (Mesh). Klein, aber nicht
# null: eine exakt senkrechte Wand hat eine unendliche Steigung und macht
# jede spaetere Normalenrechnung unbrauchbar.
MESH_MINDEST_SKALA_M = 8.0

# Wieviel Zielhoehe je Meter groesster Kuestenentfernung hoechstens zulaessig
# ist - der Deckel aus docs/KUESTENMODELL.md §3. An echten Inseln gemessen
# liegt das Verhaeltnis zwischen 0.08 (Aran, flache Kalkinsel) und 0.83
# (Capri); 1.0 laesst also alles Reale durch und faengt nur die Faelle ab,
# in denen ein Archetyp mehr Hoehe verlangt, als die Landmasse traegt.
ZIEL_JE_HINTERLAND = 1.0

# Wieviel eine Kueste ihr Hinterland ueberhoeht, je Einheit `hoehe_faktor`
# des Archetyps. 0.25 heisst: die steilste Klippe (hoehe_faktor 1.3) liegt
# rund 33 % ueber dem Plateau dahinter, eine Flachkueste praktisch darauf.
UEBERHOEHUNG_JE_HOEHENFAKTOR = 0.25

# Mindesthoehe der Kuestenformung, damit eine Kueste auch dort sichtbar ist,
# wo das Hinterland selbst fast auf Meereshoehe liegt.
KUESTEN_SOCKEL_M = 12.0

# Wieviele Kuestenabschnitte an einem Ort hoechstens mitmischen. Zwei reichen
# fast ueberall (die zwei Seiten einer Landzunge), drei an Ecken und an
# Dreiplattenpunkten. Mehr aendert am Bild kaum etwas und kostet Zeit.
K_KUESTEN = 3

# Wieviele naechste Konturpunkte durchsucht werden, um daraus K_KUESTEN
# VERSCHIEDENE Abschnitte zu finden. Die naechsten Punkte liegen fast immer
# alle auf demselben Kuestenstueck - ohne diese Ueberzahl faende man nie die
# gegenueberliegende Seite.
K_KANDIDATEN = 24

# Ab welchem Abstand ENTLANG DER KUESTE zwei Punkte als verschiedene
# Abschnitte gelten. Raeumliche Naehe taugt als Kriterium nicht: im Hals
# einer Landzunge liegen die beiden Seiten raeumlich dicht beieinander, sind
# aber entlang der Kueste weit auseinander - und genau die will man trennen.
MIN_BOGEN_TRENNUNG_M = 400.0

# Mindestzahl Stationen je geschlossener Kontur. Der feste Meterabstand allein
# genuegt nicht: eine 500-m-Insel hat 1670 m Kueste und bekaeme damit vier
# Stationen - vier Sektoren, deren Naehte man als Tortenstuecke sieht. Bei
# kurzen Konturen wird der Abstand deshalb so weit verkleinert, bis diese
# Zahl erreicht ist.


# BUDGETKORREKTUR JE ARCHETYP - gemessen ueber 64 Karten (2026-08-25)
#
# Nutzervorgabe: *"es sollte gleichmaessig sein ueber viele maps hinweg.
# eine map kann sich von einer anderen unterscheiden. also neu gewichten."*
#
# Bis hierher war jede Aussage ueber die Archetypverteilung an EINEM Seed
# gemessen. Das ist bei dieser Groesse wertlos: die Streuung des
# Kuestenanteils eines Archetyps von Karte zu Karte betraegt 8 bis 21
# Prozentpunkte. Was auf einer Karte wie ein grober Fehler aussieht, ist im
# Mittel oft genau richtig -
#
#     Kola-Steilkueste: Einzelkarte 5.53x ueber Soll, Mittel ueber 64
#     Karten 29.7 % gegen 30.0 % Soll.
#
# GEMESSEN ueber 64 Seeds a 384 px (Anteil an der Kuestenlaenge INNERHALB
# der eigenen Region, Mittel +- Standardfehler des Mittels):
#
#     mittlere Abweichung vom max_anteil:  2.9 Prozentpunkte
#     22 von 24 auswertbaren Archetypen:   innerhalb von 2 Standardfehlern
#
# Nur zwei sind belegbar daneben - alle anderen Abweichungen sind Rauschen
# und werden ABSICHTLICH NICHT korrigiert (eine Korrektur auf einen Wert
# innerhalb des Standardfehlers kalibriert Rauschen ein und macht die
# Verteilung schlechter, nicht besser):
#
#     Archetyp            soll     ist     Abw    +-SE   Faktor
#     Vendee-Straende      45%    39.9%   -5.1    1.7     3.0     (keiner)
#     Fjordwand            30%    34.9%   +4.9    2.2     2.2      0.86
#
# Das Nevadin fehlt in dieser Rechnung: es hat auf 62 von 64 Karten
# ueberhaupt keine Kueste und ist damit nicht kalibrierbar.
#
# Die Faktoren werden INNERHALB der Region wieder auf die Summe der
# `max_anteil` normiert - ein groesseres Budget fuer den einen darf dem
# anderen nichts wegnehmen, das Gesamtbudget bleibt die Kueste der Region.
#
# WARUM VENDEE-STRAENDE KEINEN FAKTOR BEKOMMT, obwohl es der groesste
# belegbare Ausreisser ist: der Faktor wurde gebaut, mit 1.13 ueber
# dieselben 64 Karten nachgemessen - und bewegte den Anteil von 39.9 % auf
# 39.8 %. Wirkungslos, und fuer die Nachbarn sogar leicht schaedlich
# (Ile-de-Re-Watt +2.5 -> +3.5 Punkte).
#
# Der Grund steht unten in der Zuordnungsschleife: die uebrig gebliebenen
# Stationen gehen an `rest = max(archetypen, key=max_anteil)`, und das IST
# in der Estrande Vendee-Straende (45 % gegen 30 % und 25 %). Ein
# Archetyp, der ohnehin alle Reste einsammelt, ist nicht budgetbegrenzt -
# sein Budget zu erhoehen kann per Konstruktion nichts aendern. Dasselbe
# gilt fuer jeden anderen groessten Typ seiner Region.
#
# Seine fehlenden 5 Punkte kommen aus dem Einschmelzen kurzer Segmente in
# `_segmente_schliessen()` (siehe docs/OFFENE_PUNKTE.md). Dort muesste die
# Korrektur ansetzen, nicht hier.
SAAT_BUDGET_KORREKTUR = {
    "Fjordwand": 0.86,
}

MIN_STATIONEN_JE_KONTUR = 16

# Ueber wieviele Nachbarstationen die gemessene Hinterlandhoehe geglaettet
# wird. Ungeglaettet schwankte sie zwischen benachbarten Stationen um bis zu
# 85 m auf 400 m Kueste (= 200 m/km); an echten Kuesten wurden 20 m/km
# gemessen (19 km Doolin-Moher), also Faktor 10 zu viel. Die Ursache ist das
# Gelaenderauschen im Messring, nicht eine echte Formaenderung.
HINTERLAND_GLAETTUNG = 5

# Breite der Uebergangszone zwischen zwei Kuestentypen, in Metern. Sie ist
# der Regler fuer "schnell, aber nicht zu schnell": innerhalb eines Segments
# ist der Typ REIN, nur hier ueberblenden zwei Typen ineinander.
UEBERGANG_M = 250.0

# Kuerzestes Segment, das einen eigenen Typ fuehren darf. Kuerzere Laeufe
# werden in den Nachbarn eingeschmolzen. Muss deutlich ueber der
# Uebergangsbreite liegen - sonst besteht das Segment nur aus Uebergaengen
# und sein Typ ist nie rein zu sehen.
MIN_SEGMENT_M = 3.0 * UEBERGANG_M

# WIE WEIT DER TYP ENTLANG DER KUESTE ZUSAMMENHAENGT, in Saatstationen.
#
# GEMESSENER ANLASS (2026-08-24): von 27 Archetypen kamen auf der fertigen
# Karte nur 19 ueberhaupt vor - acht waren SPURLOS verschwunden, darunter
# Fjordbucht, Foerdenkueste und Kotor-Steilfjord. Die Fjordwand hatte 13
# Saatstationen und ueberlebte mit einem einzigen Segment von 0.7 km
# (0.4 % der Kueste), waehrend die Algarve-Klippen aus 8 Stationen 27.8 km
# (15.5 %) machten.
#
# URSACHE: die Zuweisung sortierte die Stationen einer Region nach Score
# und schnitt oben ab - ohne jeden Bezug darauf, welche Station neben
# welcher liegt. Der `jitter` war weisses Rauschen je Station. Ein Typ
# landete damit in Laeufen von ein bis zwei Stationen, also 420-840 m.
# MIN_SEGMENT_M verlangt 750 m, und das Einschmelzen gibt einen zu kurzen
# Lauf an den LAENGEREN Nachbarn - wer schon lang war, wurde laenger.
#
# ABHILFE: der Jitter wird entlang der Bogenlaenge geglaettet. Benachbarte
# Stationen bekommen dadurch aehnliche Scores und damit denselben Typ, und
# die Laeufe werden lang genug, um das Einschmelzen zu ueberstehen. Die
# ZIELANTEILE bleiben davon unberuehrt - das Meterbudget (`ziel_meter`,
# bis 2026-08-25 `ziel_anzahl`) und die Auswahlschleife sind unveraendert,
# es aendert sich nur, WELCHE Stationen ein Typ bekommt, nicht wieviel
# Kueste.
# GEMESSEN 2026-08-24 (512 px, Seed 20260804), Ausfaelle ab 8 Stationen:
#   sigma 2.5 -> 2 fehlen (Fjordbucht, Moher-Klippen), Median 505 m
#   sigma 3.5 -> 1 fehlt  (Algarve-Klippen), Median 502 m   <-- gewaehlt
#   sigma 5.0 -> 2 fehlen (Algarve, Fjordbucht), Median 397 m
# Nach oben wird es wieder schlechter: zu lange Laeufe lassen einem
# seltenen Typ keine Luecke mehr, in die er passt.
SAAT_KOHAERENZ_STATIONEN = 3.5

# Bis zu welchem Anteil der Reichweite die Kuestenformung VOLL gilt, bevor
# sie zum Basisgelaende auslaeuft.
#
# GEMESSENER GRUND: vorher fiel die Staerke ab der Wasserlinie quadratisch
# (bei 20 % der Reichweite schon auf 0.64). Das Profil selbst hat Moher-Form
# - 90 % der Hoehe nach u=0.083, gemessen an der echten Kueste 0.128 - aber
# die Blendung gab direkt hinter der Wand ans Basisgelaende ab. Im
# Querschnitt kam dadurch eine Glockenkurve heraus statt "steil, dann
# flach": 90 % der Hoehe erst bei u=0.846, also Faktor 7 zu spaet.
#
# Mit einem Plateau haelt das Profil ueber den groessten Teil seiner
# Reichweite und laeuft erst am Rand aus.
PLATEAU_ANTEIL = 0.6

# Untergrenze der Zielhoehe als Anteil des Katalogwerts.
#
# Nutzer-Vorgabe 2026-08-19: *"die cliffs of moher sollten schon eher bei
# min. 1/3 der echten hoehe liegen und bis zur echten Hoehe gehen wenn es das
# hinterland zulaesst, vielleicht auch mehr"*.
#
# Vorher bestimmte allein das gemessene Hinterland die Zielhoehe. Das war der
# richtige Weg gegen den Ringgraben, hat die Klippe aber auf 7-16 m Amplitude
# zusammenschrumpfen lassen - bei dieser Groesse kann das Profil im
# Gesamtbild gar nicht sichtbar werden, egal wie richtig seine Form ist. Der
# Katalogwert wirkt jetzt als BODEN, das Hinterland kann darueber hinaus
# heben.
MIN_ANTEIL_KATALOG = 1.0 / 3.0

# Wie schnell jedes Groessenband des Rauschgelaendes hinter der Kueste
# einblendet - Exponent auf (1 - Kuestenstaerke), von GROB nach FEIN.
# Gross = spaet, klein = frueh. Alle auf 1.0 gibt die frueher benutzte
# gleichmaessige Ueberblendung.
#   Band 0  ueber 3000 m   Oktaven 1-2, die Grossform
#   Band 1  750 - 3000 m   Oktaven 3-4
#   Band 2  188 - 750 m    Oktaven 5-6
#   Band 3  unter 188 m    Oktaven 7-9, Feinstruktur
BAND_EXPONENTEN = (2.5, 1.6, 0.9, 0.45)

# ------------------------------------------------------------------ #
# WIE STARK DAS HINTERLAND DIE KUESTENHOEHE BESTIMMT
# (Nutzervorgabe 2026-08-24)
#
#   *"ein strandgebiet soll nicht so stark an noise angepasst werden wie
#   klippen ... bei hoehe 0 im profil keine anhebung, und das erst mit
#   wachsender hoehe auf den jetzigen wert steigen. also flache kueste
#   bleibt flach, hohe kueste ist hoehenvariabel"*
#
# DAS PROBLEM, GEMESSEN (smoke_test_kuestenprofiltreue, 2026-08-24):
# die Weissmeer-Flachkueste hat einen Katalogwert von 15 m und kam auf
# 175 m heraus. Die Zielhoehe entsteht als
#
#     ziel = max(hinterland * ueberhoehung + SOCKEL,
#                MIN_ANTEIL_KATALOG * katalog)
#
# und der erste Term gewinnt fast immer: bei 150 m Rohgelaende hinter der
# Kueste sind das 150 * 1.06 + 12 = 171 m - genau der gemessene Wert. Das
# HINTERLAND, also das rohe Rauschgelaende, bestimmt die Kuestenhoehe
# praktisch allein, und der Archetyp hat kaum noch Einfluss. Deshalb
# treffen Klippen ihre Vorlage gut (sie sind ohnehin hoch) und Straende
# schlecht (sie werden zu Klippen hochgezogen).
#
# DER EINGRIFF: die Hinterlandkopplung an den Katalogwert binden. Ein
# Archetyp mit Katalogwert 0 folgt dem Hinterland gar nicht mehr und
# bleibt bei seiner eigenen Hoehe; mit wachsendem Katalogwert steigt die
# Kopplung auf 1, also auf das bisherige Verhalten. Genau die vom Nutzer
# beschriebene Kennlinie.
#
# WAS SICH NICHT AENDERT: der Katalogboden (`MIN_ANTEIL_KATALOG`), die
# Profilform, die Reichweite und die Kuestenlinie. Nur der Betrag, um den
# das Hinterland die Zielhoehe anhebt, haengt jetzt am Archetyp.
#
# WORAN "FLACH" ERKANNT WIRD - nicht am Katalogwert.
#
# Der erste Anlauf koppelte an `katalog_m`, die Klippenhoehe des
# Archetyps. GEMESSEN WAR DAS FALSCH: der Katalogwert ist je REGION
# tabelliert, nicht je Archetyp, und trennt Strand und Klippe deshalb
# nicht. Die Kola-STEILKUESTE steht bei 30 m (die Morobora hat insgesamt
# niedrige Kuesten), der Kykladen-STRAND bei 69 m. Eine Schwelle auf
# diesen Wert daempfte ausgerechnet die niedrigen Klippen mit: die
# schlechteste Klippenform stieg von RMS 0.138 auf 0.214 (Schwelle 80 m)
# bzw. 0.442 (Schwelle 170 m).
#
# `ueberhoehung` trennt dagegen sauber, und zwar aus einem sachlichen
# Grund: sie sagt GENAU, wie stark der Archetyp das Hinterland ueberhoeht
# - also exakt die Groesse, um die es hier geht.
#
#   Straende   1.06 - 1.11  (Weissmeer 1.06, Vendee 1.07, Toskana 1.10,
#                            Luce Bay 1.10, Kykladen 1.11)
#   Klippen    1.23 - 1.45  (Bretagne 1.23, Kola 1.27, Santorini 1.29,
#                            Algarve 1.32, Moher 1.35, Amalfi 1.40,
#                            Fjordwand 1.45)
#
# Unterhalb von UEBER_FLACH folgt die Kueste dem Hinterland gar nicht
# mehr, oberhalb von UEBER_STEIL wie bisher voll.
# GEMESSEN am 2026-08-24 ueber fuenf Spannen (Formtreue-RMS,
# smoke_test_kuestenprofiltreue, 20 Gruppen):
#
#   Spanne        Gesamt   Strand   Klippe   schlechteste Klippe
#   aus            0.121    0.317    0.059    0.138
#   1.05 - 1.30    0.129    0.147    0.067    0.138
#   1.05 - 1.25    0.107    0.114    0.055    0.141   <- gewaehlt
#   1.08 - 1.30    0.107    0.102    0.066    0.179
#   1.05 - 1.45    0.122    0.101    0.077    0.355
#   1.00 - 1.30    0.101    0.157    0.067    0.133
#
# 1.05-1.25 macht die Strandform fast dreimal so treu und laesst die
# Klippen dabei unangetastet - ihr Median wird sogar minimal besser. Die
# weiteren Spannen holen bei den Straenden noch etwas heraus, bezahlen es
# aber mit der schlechtesten Klippe (0.179 bzw. 0.355).
UEBER_FLACH = 1.05
UEBER_STEIL = 1.25

# WO DIE ANGLEICHUNG ANS HINTERLAND STATTFINDET (Nutzervorgabe 2026-08-24,
# Praezisierung):
#
#   *"das ist richtig dass die zielhoehen stimmen muessen, das profil
#   dazwischen muss erstmal stimmen, aber bei flachen straenden soll dann
#   hinten heraus die angleichung passieren und die abweichung vom profil
#   dann stattfinden"*
#
# Also DREI Zusicherungen, nicht zwei: an der Wasserlinie exakt das
# Profil, dazwischen die richtige Zielhoehe UND die richtige Form - und
# die Abweichung vom Profil erst hinten heraus, konzentriert am Ende der
# Reichweite.
#
# `PLATEAU_ANTEIL` legt fest, bis zu welchem Anteil der Reichweite die
# Kueste mit voller Staerke gilt; danach blendet sie ueber den Rest aus.
# Bei 0.6 verteilt sich die Angleichung ueber die letzten 40 % - fuer eine
# Klippe richtig, denn dort geht die Form ohnehin in den Hang ueber. Fuer
# einen Strand ist es zu frueh: sein Profil ist flach, und die
# Ueberblendung mit dem Rauschgelaende ueberlagert genau den Teil, der
# stimmen soll.
#
# `PLATEAU_FLACH` ist der Wert fuer die flachste Kueste. Dazwischen wird
# mit derselben `kopplung` aus der Ueberhoehung gemischt, die auch die
# Hinterlandanhebung steuert - EIN Mass fuer "wie flach ist diese Kueste",
# nicht zwei.
#
# STEHT AUF 0.6, ALSO GLEICH `PLATEAU_ANTEIL`, ALSO WIRKUNGSLOS.
# GEMESSEN am 2026-08-24 (Formtreue-RMS der Straende):
#
#   PLATEAU_FLACH   Strand-RMS   Klippen-RMS   schlechteste Klippe
#   0.60 (aus)         0.114        0.055           0.141
#   0.75               0.117        0.055           0.148
#   0.85               0.145        0.057           0.151
#   0.92               0.168        0.057           0.149
#
# Ein laengeres Plateau macht die Strandform MESSBAR SCHLECHTER, nicht
# besser. VERMUTUNG (nicht geprueft): die Vorbildprofile laufen weich aus,
# und ein hartes Plateau mit anschliessendem steilen Abfall trifft diese
# Form schlechter als der weiche Uebergang ab 60 %. Der Regler bleibt
# stehen, damit die Messung nachvollziehbar ist und eine andere Umsetzung
# derselben Absicht hier ansetzen kann.
PLATEAU_FLACH = 0.6

# GEMESSENE Klippenhoehen je Region, (p10, p90) der Gipfelhoehen entlang der
# jeweiligen Vorbildkueste, in Metern.
#
# Nutzer-Vorgabe 2026-08-19: *"einfach reale Werte nehmen und keinen Katalog
# mehr. wir haben doch genau dafuer die Profile gemacht"*.
#
# Der Katalogwert `KUESTENHOEHE_M * hoehe_faktor` gab fuer die Moher-Klippen
# 630 m an - das Vier- bis Fuenffache des tatsaechlich gemessenen Moher
# (p90 = 170 m). Diese Zahlen stammen aus tools/kuestenlaengsschnitt.py,
# Schnitte auf den Kuestennormalen alle 100 m, ueber die volle Vorbildkueste:
#
#   Clonagh          Doolin-Moher, 19.4 km
#   Skerrheim           Lofoten bei Reine, 61.5 km
#   Morobora               Stockholmer Schaeren, 57.2 km
#   Estrande      Cabo da Roca, 13.6 km
#   Nevadin           Lofoten (Nutzer-Vorgabe: "alpen kann lofoten
#                       entsprechen, das ist gut genug")
#   Nebelrode       Ruegen/Koenigsstuhl, 15.2 km
#   Samarcia              Kap Kaliakra, 20.9 km
#   Macchia          Calanques Marseille, 33.3 km
#   Thalassia  Santorini, 42.9 km
#
# p10/p90 statt Min/Max: die Extremwerte sind einzelne Stationen und setzen
# sonst die ganze Skala.
MESSWERTE_JE_REGION = {
    "Clonagh": (17.0, 170.0),
    "Skerrheim": (42.0, 448.0),
    "Morobora": (15.0, 30.0),
    "Estrande": (50.0, 191.0),
    "Nevadin": (42.0, 448.0),
    "Nebelrode": (80.0, 142.0),
    "Samarcia": (47.0, 83.0),
    "Macchia": (53.0, 297.0),
    "Thalassia": (69.0, 280.0),
}


# GEMESSENE Reichweite je Region, in Metern.
#
# Nutzerfrage 2026-08-19: *"fuer jedes profil sollte die laenge selber
# entschieden werden, ab wann wuerdest du sagen ist das kuestenprofil von der
# tiefe her zuende."*
#
# KRITERIUM: das Profil endet dort, wo seine Steigung auf 10 % der
# Klippensteigung gefallen ist - also wo die Wand aufhoert und gewoehnliches
# Gelaende anfaengt.
#
# Warum nicht "wo es seine Endhoehe erreicht": an echten Kuesten gibt es
# keine Endhoehe. Gemessen liefen 90/95/99 %-Kriterien bei sechs von acht
# Regionen in die 900-m-Messgrenze, weil das Gelaende landeinwaerts einfach
# weitersteigt. Ein Saettigungskriterium ist an realen Daten unbrauchbar.
#
# Die Schwelle 10 % ist geprueft stabil: zwischen 5 % und 20 % aendern sich
# die Werte moderat (Clonagh 185/141/105 m), bei 30 % bricht das
# Verfahren zusammen (Estrande springt auf 3 m - ein Fehltreffer im
# verrauschten Anfang).
MESS_REICHWEITE_M = {
    "Clonagh": 141.0,
    "Skerrheim": 296.0,
    "Morobora": 132.0,
    "Estrande": 185.0,
    "Nevadin": 296.0,          # Lofoten, wie bei den Hoehen
    "Nebelrode": 172.0,
    "Samarcia": 110.0,
    "Macchia": 203.0,
    "Thalassia": 349.0,
}


# GEMESSENE PROFILFORMEN je Region, h(u) an 17 gleichabstaendigen Stellen
# von u = 0 (Wasserlinie) bis u = 1 (Ende der Reichweite), auf 0..1 normiert.
#
# Nutzer-Vorgabe: *"einfach reale Werte nehmen und keinen Katalog mehr"* -
# das gilt nicht nur fuer Hoehe und Reichweite, sondern auch fuer die FORM.
# Bis hierher benutzte das Modell ueberall dieselbe Exponentialkurve
# `ziel * (1 - exp(-d/skala))`, unabhaengig von der Region.
#
# Gewonnen aus den Schnitten auf den Kuestennormalen (alle 100 m ueber die
# volle Vorbildkueste), je Station auf seine eigene Spanne normiert, dann der
# Median ueber alle Stationen. Monoton gemacht (kumuliertes Maximum), damit
# Messrauschen keine Rueckfaelle erzeugt - eine Kueste, die landeinwaerts
# wieder abfaellt, waere ein Tal und gehoert nicht ins Kuestenprofil.
#
# Die Formen unterscheiden sich deutlich: Thalassia steigen am
# steilsten (0.42 bei u=0.25), Skerrheim am flachsten (0.20) - Lofoten ist
# eine Rampe, keine Wand. Genau diese Unterschiede gingen mit der einen
# Formel verloren.
# GEMESSENE MASSE JE ARCHETYP: (Distanz in m, Hoehe in m).
#
# Erzeugt von tools/archetyp_masse_messen.py aus denselben DEM-Kacheln,
# aus denen MESS_FORM_JE_ARCHETYP stammt, mit derselben Aufteilung: die
# Schnitte einer Vorbildstrecke nach Steilheit sortiert und auf die drei
# Archetypen ihrer Region verteilt.
#
# WARUM ES DIESE TABELLE BRAUCHT (Nutzervorgabe 2026-08-24): *"die
# kuestenprofile sind dadurch nicht gestreckt oder gestaucht sondern haben
# das gleiche hoehen zu tiefen verhaeltnis wie in echt"*. Ein Profil
# behaelt sein Verhaeltnis nur, wenn es in METERN ausgewertet wird - und
# dafuer braucht es beide Masse je Archetyp.
#
# Bis hierher gab es nur `MESS_REICHWEITE_M` und `MESSWERTE_JE_REGION`,
# beide je REGION. Das ist die Wurzel der vier bekannten Hoehenausreisser:
# die Morobora-Hoehen stammen von den Stockholmer Schaeren (15-30 m) und
# galten damit auch fuer die Kola-Steilkueste. Jetzt misst die
# Weissmeer-Flachkueste 17 m und die Kola-Steilkueste 19 m - beide
# innerhalb ihres Bandes, aber mit dem Verhaeltnis 0.114 gegen 0.170
# unterschieden.
#
# Das Verhaeltnis Hoehe/Distanz reicht von 0.056 (Labrador-Buchten) bis
# 1.131 (Moher-Klippen) - eine Moher-Klippe steigt also ueber 127 m um
# 144 m, eine Weissmeer-Flachkueste ueber 145 m um 17 m.
# =====================================================================
# DIE KUESTENPROFILE IN METERN - je Archetyp aus SEINER Vorbildkueste
# =====================================================================
#
# h(x) ueber der Uferhoehe, an 19 Stellen alle 50 m von 0 bis 900 m.
# NICHT normiert, NICHT auf eine Reichweite gestreckt.
#
# Nutzervorgabe 2026-08-24: *"die kuestenprofile sind dadurch nicht
# gestreckt oder gestaucht sondern haben das gleiche hoehen zu tiefen
# verhaeltnis wie in echt"* und *"immer pro region 1x flach, 1x
# mittelsteil, 1 steil ... es sollen 3 unterschiedliche profile sein"*.
#
# WAS VORHER FALSCH WAR. Es gab nur 11 Vorbildstrecken fuer 27
# Archetypen. Die Schnitte EINER Region wurden nach h(150 m) sortiert und
# in Drittel geteilt - das ergab drei Kurven mit praktisch identischer
# FORM, die sich nur in der HOEHE unterschieden (normiert 0.28/0.26/0.30
# nach 50 m, 0.58/0.57/0.57 nach 100 m). Dazu ein Zirkelschluss: nach
# h(150 m) sortieren und dann h(150 m) messen trennt zwangslaeufig genau
# in dieser Groesse und in sonst nichts.
#
# JETZT hat jeder Archetyp seine eigene Kueste (tools/archetyp_vorbilder.py,
# 27 DEM-Kacheln in tools/_dem_cache/), und die Unterschiede sind gemessen
# statt konstruiert. Steilheit h(150 m)/150 m je Region, flach -> steil:
#
#   Clonagh         Luce Bay 0.039  West-Cork 0.070   Moher 0.973
#   Skerrheim          Schaeren 0.094  Fjordbucht 0.215  Fjordwand 0.799
#   Morobora              Weissmeer 0.036 Labrador 0.102    Kola 0.146
#   Estrande     Ile-de-Re 0.037 Vendee 0.070      Bretagne 0.331
#   Nevadin          Flussmdg 0.052  Dalmatien 0.118   Kotor 0.257
#   Nebelrode      Foerde 0.012    Ostsee 0.062      Ruegen 0.488
#   Samarcia             Donana 0.106    Algarve 0.298     Costa Brava 0.346
#   Macchia         Toskana 0.026   Amalfi 0.535      Cinque Terre 0.662
#   Griech. Inseln     Kykladen 0.008  Kreta 0.438       Santorini 0.542
#
# Algarve und Costa Brava liegen im Anfangsanstieg nah beieinander,
# unterscheiden sich aber im VERLAUF deutlich: Algarve steht nach 150 m
# bei 45 m und bleibt dort (klassische Klippe mit Plateau), Costa Brava
# steigt weiter bis 121 m (Bergkueste).
#
# Erzeugt von tools/archetyp_profile_messen.py.
#
# NEU GEMESSEN 2026-08-24 NACH EINEM GEOMETRIEFEHLER. Die erste Messung
# nahm fuer BEIDE Rasterrichtungen dieselbe Pixelgroesse
# (`cellsize * 111320`). DEM-Raster haben aber konstante GRADabstaende,
# und ein Grad Laenge ist bei 62 Grad Nord nur halb so lang wie am
# Aequator: bei Geiranger misst ein Pixel 31.1 m in Nord-Sued- und nur
# 14.5 m in Ost-West-Richtung. Jeder Schnitt nach Osten lief damit
# doppelt so weit, wie er sollte.
#
# Betroffen waren ALLE 27 Vorbilder, nach Breitengrad verschieden stark -
# von 1.22x (Kreta, 35 Grad) bis 2.81x (Kola, 69 Grad). Die Auswirkung auf
# die Profile blieb mit rund +-15 % moderat, weil sich Nord-Sued- und
# Ost-West-Schnitte im Median weitgehend ausgleichen; systematisch falsch
# war es trotzdem. `zellgroesse_m()` in tools/kuestenlaengsschnitt.py
# liefert seither beide Werte getrennt.
PROFIL_STELLEN_M = (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0)

MESS_PROFIL_M_JE_ARCHETYP = {
    "Algarve-Klippen": (
        0.0, 23.2, 38.7, 43.2, 43.2, 43.2, 43.2,
        43.2, 43.2, 43.2, 43.2, 43.2, 43.2, 43.2,
        43.2, 43.2, 44.7, 44.7, 44.7),
    "Alpine-Flussmuendung": (
        0.0, 3.4, 8.0, 8.0, 8.3, 9.1, 9.1,
        9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 10.6,
        13.4, 14.7, 20.8, 32.3, 41.1),
    "Amalfi-Steilkueste": (
        0.0, 25.9, 62.1, 84.7, 109.3, 135.2, 155.8,
        170.6, 175.5, 178.9, 197.4, 209.7, 219.3, 224.7,
        225.1, 225.8, 227.4, 227.8, 227.8),
    "Bretagne-Klippen": (
        0.0, 19.0, 41.5, 50.7, 55.8, 57.0, 57.7,
        58.1, 59.1, 59.1, 60.1, 61.1, 61.2, 61.2,
        61.2, 61.2, 61.2, 61.2, 61.2),
    "Cinque-Terre-Buchten": (
        0.0, 34.0, 76.3, 103.9, 118.2, 142.4, 167.0,
        178.0, 205.5, 218.8, 232.6, 258.3, 276.6, 283.9,
        285.6, 285.6, 285.6, 285.6, 286.7),
    "Costa-Brava-Buchten": (
        0.0, 21.4, 42.4, 56.8, 61.8, 68.4, 71.2,
        78.1, 78.1, 80.9, 83.2, 95.6, 98.5, 103.0,
        110.2, 113.3, 126.7, 138.0, 140.2),
    "Dalmatien-Klippen": (
        0.0, 9.6, 15.8, 17.6, 20.8, 24.9, 27.7,
        31.1, 36.0, 42.0, 46.0, 50.8, 55.4, 59.5,
        65.1, 68.7, 72.4, 74.5, 82.3),
    "Fjordbucht": (
        0.0, 12.3, 23.5, 34.9, 52.4, 74.4, 95.0,
        121.0, 144.1, 163.5, 193.1, 202.7, 212.7, 224.3,
        227.6, 227.6, 227.6, 227.6, 227.6),
    "Fjordwand": (
        0.0, 37.5, 85.4, 127.7, 181.1, 224.4, 277.8,
        335.5, 402.0, 476.3, 543.4, 589.8, 613.2, 632.6,
        657.5, 678.7, 706.5, 740.8, 774.9),
    "Foerdenkueste": (
        0.0, 1.1, 1.5, 2.1, 2.5, 2.9, 2.9,
        3.1, 3.1, 3.6, 3.6, 3.6, 3.6, 3.6,
        3.6, 3.6, 3.6, 3.6, 3.6),
    "Ile-de-Re-Watt": (
        0.0, 4.8, 5.2, 5.5, 5.9, 6.1, 6.6,
        7.5, 7.7, 7.8, 7.8, 8.0, 8.1, 8.5,
        8.6, 8.6, 8.6, 8.6, 8.7),
    "Kola-Steilkueste": (
        0.0, 7.1, 15.8, 23.6, 30.2, 37.3, 44.5,
        48.2, 51.5, 53.1, 55.6, 58.7, 60.1, 61.1,
        67.0, 67.0, 71.5, 71.5, 71.5),
    "Kotor-Steilfjord": (
        0.0, 9.6, 26.0, 43.6, 60.7, 74.5, 92.7,
        109.2, 129.2, 153.7, 172.3, 186.5, 208.0, 222.8,
        237.1, 253.9, 271.6, 291.3, 310.8),
    "Kreta-Buchten": (
        0.0, 24.7, 50.0, 72.6, 82.2, 97.3, 109.1,
        132.4, 151.2, 178.4, 191.4, 201.5, 215.4, 230.6,
        242.9, 243.4, 244.1, 261.3, 269.6),
    "Kykladen-Strand": (
        0.0, 0.7, 1.2, 1.5, 2.0, 2.5, 3.4,
        3.4, 3.9, 5.1, 6.3, 7.0, 7.6, 7.8,
        8.7, 8.8, 9.8, 11.0, 12.4),
    "Labrador-Buchten": (
        0.0, 4.1, 9.5, 16.8, 27.1, 37.8, 50.3,
        60.4, 70.9, 80.5, 87.4, 94.3, 96.6, 97.5,
        102.7, 106.4, 106.4, 106.4, 106.4),
    # NEU GEMESSEN 2026-08-25 an Sandhead / Luce Bay, Galloway (vorher
    # Inch Beach / Dingle - siehe die Begruendung in
    # tools/archetyp_vorbilder.py). Die alte Vorlage stieg hinter dem
    # Strand auf 51.9 m, die neue auf 22.8 m, und sie ist ueber die ersten
    # 350 m deutlich flacher (6.2 statt 3.4 m bei 200 m, aber 10.1 statt
    # 6.4 m bei 350 m). Das ist die vom Nutzer verlangte Form: flach an
    # der Wasserlinie, Huegel im Hinterland statt Bergfuss.
    "Luce-Bay-Straende": (
        0.0, 3.9, 4.8, 5.8, 6.2, 6.2, 6.6,
        10.1, 15.7, 16.5, 18.6, 18.6, 19.3, 19.3,
        19.7, 21.2, 21.5, 22.1, 22.8),
    "Moher-Klippen": (
        0.0, 57.3, 129.1, 146.0, 148.2, 148.2, 148.2,
        148.2, 148.2, 148.2, 148.2, 148.2, 148.2, 148.2,
        148.2, 148.2, 148.2, 148.2, 148.2),
    "Ostsee-Flachkueste": (
        0.0, 4.0, 8.0, 10.5, 14.2, 15.8, 17.0,
        17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0,
        17.0, 17.0, 17.2, 17.2, 17.2),
    "Ruegen-Kreidekueste": (
        0.0, 27.3, 62.7, 78.3, 85.3, 85.3, 87.1,
        87.1, 87.1, 90.2, 93.1, 97.4, 97.4, 99.8,
        101.0, 103.2, 104.6, 107.7, 108.9),
    "San-Sebastian-Bucht": (
        0.0, 4.3, 10.7, 16.0, 18.3, 18.3, 19.4,
        19.5, 19.9, 19.9, 19.9, 19.9, 19.9, 19.9,
        20.1, 20.4, 20.5, 20.5, 20.5),
    "Santorini-Kliff": (
        0.0, 24.9, 58.8, 85.8, 103.4, 113.4, 120.3,
        120.3, 120.3, 120.3, 120.3, 120.3, 120.3, 120.3,
        120.3, 120.3, 120.3, 120.3, 120.3),
    "Schaerenkueste": (
        0.0, 7.4, 12.6, 14.7, 15.3, 15.3, 15.3,
        15.3, 15.3, 15.3, 15.3, 15.3, 15.3, 15.3,
        15.3, 15.3, 15.3, 15.3, 15.3),
    "Toskana-Straende": (
        0.0, 2.0, 2.8, 3.6, 4.9, 5.5, 5.5,
        5.5, 5.7, 7.1, 8.1, 8.1, 8.1, 8.1,
        8.1, 8.1, 8.1, 10.7, 19.4),
    "Vendee-Straende": (
        0.0, 9.0, 10.2, 10.2, 11.0, 12.6, 13.7,
        15.3, 15.7, 15.7, 15.7, 15.7, 15.7, 15.7,
        15.7, 15.7, 15.7, 15.7, 15.7),
    "Weissmeer-Flachkueste": (
        0.0, 1.9, 3.2, 5.0, 6.0, 6.3, 6.5,
        6.9, 7.0, 7.3, 7.8, 7.9, 8.5, 9.3,
        9.6, 10.0, 10.9, 11.4, 11.7),
    "West-Cork-Buchten": (
        0.0, 5.1, 8.3, 10.5, 11.6, 13.3, 13.3,
        13.3, 13.3, 14.9, 15.3, 17.8, 17.8, 18.9,
        20.3, 21.5, 21.7, 21.9, 24.8),
}


# HINTERLANDHOEHE JE ARCHETYP - die Zweipunktmethode des Nutzers.
#
# Nutzervorgabe 2026-08-26: *"höhenwert vom küstenprofil sei die mittlere
# höhe von 400 bis 700 m tiefe im hinterland (zweipunktmethode)"*.
#
# WOFUER: das Gebietssystem (`kuestengebiete()` in
# core/terrain_weltkarte.py) leitet daraus ab, wie hoch das Hinterland
# hinter einem Kuestenabschnitt liegt.
#
# WARUM NICHT `hoehe_faktor` AUS DEM KATALOG - der beschreibt das UFER,
# nicht das Hinterland, und in VIER von neun Regionen dreht sich die
# Reihenfolge dadurch um (gemessen 2026-08-26):
#
#     Region              Archetyp             hoehe_faktor   h(400-700)
#     Skerrheim           Fjordbucht               0.30          195 m
#     Skerrheim           Schaerenkueste           0.50           15 m
#     Thalassia  Kreta-Buchten            0.80          202 m
#     Thalassia  Santorini-Kliff          1.15          120 m
#     Macchia          Cinque-Terre-Buchten     1.00          252 m
#     Macchia          Amalfi-Steilkueste       1.60          204 m
#     Morobora               Labrador-Buchten         0.70           90 m
#     Morobora               Kola-Steilkueste         1.10           58 m
#
# Eine Fjordbucht hat ein niedriges Ufer und steile Waende direkt dahinter.
# Der Katalogfaktor sagt "niedrig", das Hinterland ist das Zweithoechste
# seiner Region. Fuer das Gebietssystem zaehlt das Hinterland.
#
# ABGELEITET, NICHT GETIPPT: der Wert wird aus MESS_PROFIL_M_JE_ARCHETYP
# gerechnet. Eine zweite von Hand gepflegte Tabelle waere eine zweite
# Wahrheit ueber dieselbe Messung (SPEZIFIKATION 4.5).
HINTERLAND_BAND_M = (400.0, 700.0)


def _gemessene_hinterlandhoehen():
    """Archetypname -> mittlere Profilhoehe im Band HINTERLAND_BAND_M."""
    stellen = np.asarray(PROFIL_STELLEN_M, dtype=np.float64)
    band = (stellen >= HINTERLAND_BAND_M[0]) & (stellen <= HINTERLAND_BAND_M[1])
    return {name: float(np.asarray(werte, dtype=np.float64)[band].mean())
            for name, werte in MESS_PROFIL_M_JE_ARCHETYP.items()}


MESS_ARCHETYP_MASSE = {
    "Algarve-Klippen": (96.0, 59.0),
    "Alpine-Flussmuendung": (376.0, 19.0),
    "Amalfi-Steilkueste": (198.0, 165.0),
    "Bretagne-Klippen": (172.0, 112.0),
    "Cinque-Terre-Buchten": (287.0, 109.0),
    "Costa-Brava-Buchten": (119.0, 55.0),
    "Dalmatien-Klippen": (392.0, 22.0),
    "Fjordbucht": (349.0, 46.0),
    "Fjordwand": (529.0, 553.0),
    "Foerdenkueste": (216.0, 81.0),
    "Ile-de-Re-Watt": (376.0, 50.0),
    "Kola-Steilkueste": (110.0, 19.0),
    "Kotor-Steilfjord": (405.0, 46.0),
    "Kreta-Buchten": (258.0, 135.0),
    "Kykladen-Strand": (225.0, 58.0),
    "Labrador-Buchten": (256.0, 14.0),
    # Wert unveraendert beim Wechsel der Vorbildkueste 2026-08-25: diese
    # Tabelle kommt NICHT aus der Kachel des Archetyps, sondern aus dem
    # Streckensplit seiner REGION (Schnitte nach Steilheit sortiert, in
    # Drittel geteilt) - siehe tools/archetyp_masse_messen.py.
    "Luce-Bay-Straende": (176.0, 17.0),
    "Moher-Klippen": (127.0, 144.0),
    "Ostsee-Flachkueste": (167.0, 78.0),
    "Ruegen-Kreidekueste": (145.0, 83.0),
    "San-Sebastian-Bucht": (198.0, 49.0),
    "Santorini-Kliff": (323.0, 213.0),
    "Schaerenkueste": (500.0, 204.0),
    "Toskana-Straende": (278.0, 47.0),
    "Vendee-Straende": (323.0, 114.0),
    "Weissmeer-Flachkueste": (145.0, 17.0),
    "West-Cork-Buchten": (163.0, 109.0),
}


MESS_FORM_JE_ARCHETYP = {
    "Algarve-Klippen": (
        0.000, 0.072, 0.214, 0.359, 0.518, 0.640,
        0.732, 0.812, 0.885, 0.943, 0.967, 0.983,
        0.989, 0.994, 0.996, 0.998, 1.000),
    "Alpine-Flussmuendung": (
        0.000, 0.021, 0.051, 0.074, 0.117, 0.165,
        0.199, 0.247, 0.305, 0.363, 0.422, 0.501,
        0.592, 0.687, 0.787, 0.910, 1.000),
    "Amalfi-Steilkueste": (
        0.000, 0.088, 0.231, 0.392, 0.535, 0.675,
        0.790, 0.880, 0.929, 0.947, 0.981, 0.999,
        1.000, 1.000, 1.000, 1.000, 1.000),
    "Bretagne-Klippen": (
        0.000, 0.060, 0.177, 0.300, 0.434, 0.561,
        0.674, 0.745, 0.809, 0.874, 0.900, 0.932,
        0.949, 0.969, 0.982, 0.995, 1.000),
    "Cinque-Terre-Buchten": (
        0.000, 0.037, 0.106, 0.195, 0.299, 0.395,
        0.481, 0.566, 0.629, 0.692, 0.754, 0.809,
        0.851, 0.897, 0.936, 0.974, 1.000),
    "Costa-Brava-Buchten": (
        0.000, 0.027, 0.088, 0.159, 0.244, 0.334,
        0.436, 0.543, 0.645, 0.743, 0.819, 0.876,
        0.923, 0.963, 0.986, 0.995, 1.000),
    "Dalmatien-Klippen": (
        0.000, 0.031, 0.081, 0.127, 0.181, 0.253,
        0.316, 0.381, 0.449, 0.518, 0.587, 0.656,
        0.727, 0.804, 0.880, 0.949, 1.000),
    "Fjordbucht": (
        0.000, 0.021, 0.051, 0.074, 0.117, 0.165,
        0.199, 0.247, 0.305, 0.363, 0.422, 0.501,
        0.592, 0.687, 0.787, 0.910, 1.000),
    "Fjordwand": (
        0.000, 0.087, 0.210, 0.322, 0.422, 0.534,
        0.634, 0.734, 0.792, 0.849, 0.896, 0.939,
        0.955, 0.970, 0.996, 1.000, 1.000),
    "Foerdenkueste": (
        0.000, 0.029, 0.082, 0.161, 0.283, 0.396,
        0.493, 0.566, 0.661, 0.733, 0.782, 0.843,
        0.899, 0.947, 0.972, 0.986, 1.000),
    "Ile-de-Re-Watt": (
        0.000, 0.029, 0.086, 0.165, 0.241, 0.346,
        0.438, 0.532, 0.634, 0.709, 0.771, 0.821,
        0.869, 0.902, 0.947, 0.981, 1.000),
    "Kola-Steilkueste": (
        0.000, 0.093, 0.244, 0.402, 0.537, 0.638,
        0.733, 0.832, 0.872, 0.919, 0.953, 0.984,
        0.995, 1.000, 1.000, 1.000, 1.000),
    "Kotor-Steilfjord": (
        0.000, 0.087, 0.210, 0.322, 0.422, 0.534,
        0.634, 0.734, 0.792, 0.849, 0.896, 0.939,
        0.955, 0.970, 0.996, 1.000, 1.000),
    "Kreta-Buchten": (
        0.000, 0.056, 0.173, 0.299, 0.423, 0.523,
        0.621, 0.700, 0.785, 0.850, 0.899, 0.936,
        0.969, 0.991, 0.997, 1.000, 1.000),
    "Kykladen-Strand": (
        0.000, 0.033, 0.098, 0.175, 0.255, 0.330,
        0.403, 0.468, 0.541, 0.610, 0.681, 0.744,
        0.804, 0.865, 0.920, 0.965, 1.000),
    "Labrador-Buchten": (
        0.000, 0.041, 0.113, 0.195, 0.298, 0.417,
        0.522, 0.627, 0.715, 0.774, 0.832, 0.869,
        0.909, 0.939, 0.970, 0.987, 1.000),
    # Wert unveraendert beim Wechsel der Vorbildkueste 2026-08-25: diese
    # Tabelle kommt NICHT aus der Kachel des Archetyps, sondern aus dem
    # Streckensplit seiner REGION (Schnitte nach Steilheit sortiert, in
    # Drittel geteilt) - siehe tools/archetyp_masse_messen.py.
    "Luce-Bay-Straende": (
        0.000, 0.020, 0.057, 0.093, 0.143, 0.218,
        0.292, 0.366, 0.460, 0.549, 0.636, 0.717,
        0.792, 0.866, 0.921, 0.970, 1.000),
    "Moher-Klippen": (
        0.000, 0.064, 0.188, 0.326, 0.449, 0.552,
        0.660, 0.773, 0.853, 0.883, 0.915, 0.938,
        0.960, 0.976, 0.993, 1.000, 1.000),
    "Ostsee-Flachkueste": (
        0.000, 0.020, 0.057, 0.105, 0.179, 0.245,
        0.331, 0.399, 0.492, 0.580, 0.681, 0.757,
        0.815, 0.871, 0.922, 0.964, 1.000),
    "Ruegen-Kreidekueste": (
        0.000, 0.069, 0.173, 0.293, 0.432, 0.555,
        0.665, 0.735, 0.798, 0.865, 0.905, 0.938,
        0.977, 0.986, 0.989, 1.000, 1.000),
    "San-Sebastian-Bucht": (
        0.000, 0.014, 0.045, 0.083, 0.129, 0.189,
        0.269, 0.355, 0.449, 0.532, 0.619, 0.697,
        0.772, 0.847, 0.913, 0.961, 1.000),
    "Santorini-Kliff": (
        0.000, 0.149, 0.371, 0.570, 0.693, 0.770,
        0.817, 0.872, 0.914, 0.949, 0.962, 0.989,
        0.996, 0.999, 1.000, 1.000, 1.000),
    "Schaerenkueste": (
        0.000, 0.031, 0.081, 0.127, 0.181, 0.253,
        0.316, 0.381, 0.449, 0.518, 0.587, 0.656,
        0.727, 0.804, 0.880, 0.949, 1.000),
    "Toskana-Straende": (
        0.000, 0.016, 0.045, 0.090, 0.143, 0.197,
        0.253, 0.318, 0.385, 0.455, 0.541, 0.633,
        0.714, 0.798, 0.876, 0.939, 1.000),
    "Vendee-Straende": (
        0.000, 0.013, 0.039, 0.083, 0.136, 0.198,
        0.264, 0.342, 0.413, 0.494, 0.581, 0.664,
        0.739, 0.815, 0.897, 0.961, 1.000),
    "Weissmeer-Flachkueste": (
        0.000, 0.020, 0.054, 0.096, 0.150, 0.204,
        0.265, 0.333, 0.417, 0.496, 0.598, 0.691,
        0.758, 0.822, 0.898, 0.953, 1.000),
    "West-Cork-Buchten": (
        0.000, 0.033, 0.095, 0.177, 0.274, 0.379,
        0.498, 0.612, 0.694, 0.769, 0.834, 0.889,
        0.925, 0.954, 0.977, 0.992, 1.000),
}


# =====================================================================
# DIE ZONEN (Nutzervorgabe 2026-08-24, woertlich)
# =====================================================================
#
#   x >= 0 (Land):
#     0 bis PROFIL_VOLL_M          100 % Kuestenprofil
#     bis + UEBERGANG_LAND_M       Smoothstep ins Rauschgelaende
#     darueber                     100 % Rauschgelaende
#   x < 0 (Meer):
#     0 bis -UEBERGANG_MEER_M      Smoothstep ins Ozeanprofil
#     darunter                     100 % Ozeanprofil
#
# JEDE KUESTE STRAHLT GLEICH TIEF, egal ob hoch oder niedrig - das ist
# die Folge davon, dass die Profile nicht mehr gestreckt werden. Ein
# flacher Strand hoert nach 350 m auf, flach zu sein, genau wie eine
# Klippe aufhoert, Klippe zu sein; was sich unterscheidet, ist die HOEHE
# auf dieser Strecke, nicht die Strecke selbst.
#
# VARIIERBAR ANGELEGT (Nutzerpunkt c): *"baue es so auf, dass spaeter die
# strahltiefe auch irgendwie variieren kann bzw. auch entlang der kueste
# zufaellig tief reicht um varianz zu erzeugen"*. Die Tiefe steht deshalb
# je SEGMENT in `voll_m`/`uebergang_m` und wird wie alle anderen
# Segmentwerte ueber die Bogenlaenge interpoliert - ein Feld, keine
# Konstante. Heute tragen alle Segmente denselben Wert; um Varianz zu
# erzeugen, reicht es, ihn beim Segmentbau zu streuen.
# Wie hoch ein Landpixel mindestens bleibt. Nur dazu da, das VORZEICHEN
# zu halten - siehe `untergrenze` in _hoehe_block(). Bewusst winzig: ein
# groesserer Wert waere ein Sockel und wuerde die gemessenen Flachprofile
# verfaelschen (die Weissmeer-Flachkueste steht nach 150 m bei 5 m).
MINDEST_LANDHOEHE_M = 0.01

# DIE ZONENGRENZEN, je nach Steilheit des Archetyps
# (Nutzervorgabe 2026-08-24):
#
#   *"koennen wir machen, dass flache kuestenprofile p1 bei 300 m haben
#   und p2 bei 550 m, so dass es etwas sanfter ist?"*
#
# p1 ist die Stelle, ab der das Rauschgelaende einblendet, p2 die, ab der
# es allein gilt. Eine flache Kueste geht also FRUEHER und ueber eine
# LAENGERE Strecke ins Rauschen ueber - sie hat kein markantes Profil, das
# sich bis zuletzt behaupten muesste. Eine Klippe haelt ihre Form bis
# 350 m und wechselt dann auf 150 m.
#
# Gemischt wird ueber die Steilheit h(150 m)/150 m des Archetyps:
#   Kykladen-Strand 0.010, Foerdenkueste 0.014, Toskana 0.024 (flach)
#   Cinque Terre 0.693, Fjordwand 0.851, Moher 0.973 (steil)
# (Stand 2026-08-25, nach dem Wechsel der Clonagh-Strandvorlage von
# Dingle auf Luce Bay: 0.017 -> 0.039. Die Zonen p1/p2 aendern sich
# dadurch NICHT - beide Werte liegen im flachen Sattel der Mischung.)
# 2026-08-25 um 30 % naeher an die Kueste gezogen (Nutzervorgabe nach der
# Sichtpruefung: *"die kuestenprofile sind teilweise etwas zu agressiv, ich
# wuerde p1 noch naeher an die kueste ziehen (30% naeher ran)"*).
# Vorher 300 / 350 m.
#
# p2 bleibt, wo es war. Die Uebergangszone wird dadurch LAENGER (flach:
# 210 statt 300 bis 550 m, also 340 statt 250 m Ueberblendung) - das ist
# genau die gewuenschte Wirkung: das gemessene Profil setzt sich kuerzer
# durch und geht sanfter ins Rauschgelaende ueber.
# 2026-08-25, ZWEITER ZUG: nochmal 15 % naeher (Nutzervorgabe *"p1 noch
# frueher ist, also nochmal 15% naeher an der kueste"*). Von 210/245 auf
# 178/208. Der erste Zug am selben Tag hatte 300/350 auf 210/245 gebracht.
PROFIL_VOLL_FLACH_M = 178.0     # p1 der flachsten Kueste
PROFIL_VOLL_STEIL_M = 208.0     # p1 der steilsten
PROFIL_ENDE_FLACH_M = 550.0     # p2 der flachsten
PROFIL_ENDE_STEIL_M = 500.0     # p2 der steilsten
STEIL_FLACH = 0.05              # Steilheit, ab der gemischt wird
STEIL_STEIL = 0.50              # Steilheit, ab der voll "steil" gilt

# Rueckfallwerte, wenn ein Archetyp kein gemessenes Profil hat.
PROFIL_VOLL_M = 350.0
UEBERGANG_LAND_M = 150.0
UEBERGANG_MEER_M = 200.0

# VARIANZ DER ZONENGRENZEN ENTLANG DER KUESTE (Nutzervorgabe 2026-08-24):
#
#   *"koennen wir dann noch eine varianz hereinbringen insgesamt, also p1
#   kann bis zu 50 m rein oder 50 raus gehen und das gleiche mit p2. das
#   ganze ist deterministisch aber sanft wechselnd (nicht regelmaessig)
#   entlang der kueste."*
#
# Umgesetzt als Summe dreier Sinuswellen ueber die BOGENLAENGE. Die drei
# Wellenlaengen sind bewusst zueinander inkommensurabel (1700, 2900,
# 4600 m - keine geht in einer anderen auf), damit sich das Muster erst
# nach vielen Kilometern wiederholt und nicht regelmaessig wirkt. Die
# Phasen kommen aus dem Kartenseed: dieselbe Karte ergibt dieselbe
# Varianz, eine andere Karte eine andere.
#
# p1 und p2 bekommen UNTERSCHIEDLICHE Phasen - sonst wanderten beide
# Grenzen im Gleichschritt und die Uebergangsbreite bliebe konstant.
ZONEN_VARIANZ_M = 50.0
ZONEN_WELLEN_M = (1700.0, 2900.0, 4600.0)

# Wie schmal die Uebergangszone hoechstens werden darf. Ohne diese
# Untergrenze koennte die Varianz p2 unter p1 druecken - die Zone waere
# negativ und der Smoothstep spraenge.
MIN_UEBERGANG_M = 60.0

# WIE WEIT p2 REICHT - AUS DER AEHNLICHKEIT ZUM NACHBARABSCHNITT
#
# Nutzervorgabe 2026-08-25: *"p2 abhaengig von der kuestenaehnlichkeit. also
# wenn die hoehenfaktoren sich stark aendern, dann bricht p2 frueher ab.
# damit es sich einfuegt. also kann 1500 m weit reichen, oder aber 400 m
# weit (natuerlich ist das mindeste immer 2x p1 oder so)."*
#
# DER GEDANKE DAHINTER: die Uebergangszone ist die Strecke, auf der das
# gemessene Kuestenprofil ins Rauschgelaende ausblendet. Steht neben einer
# Klippe ein Strand, muessen beide auf kurzer Strecke zueinanderfinden -
# eine lange Ausblendung wuerde sie ueber hunderte Meter ineinanderziehen
# und beide Formen verwischen. Gleicht der Nachbar dagegen dem eigenen
# Abschnitt, darf die Kueste ihre Form weit ins Land tragen.
#
# GEMESSEN wird ueber `hoehe_faktor` aus dem Archetypkatalog, weil das die
# Groesse ist, die der Nutzer nennt. Spannen je Region:
#
#     Skerrheim   1.8 / 0.5 / 0.3   Spanne 1.50   (Wand neben Bucht)
#     Nevadin   1.7 / 1.2 / 0.4   Spanne 1.30
#     Nebelrode 0.85/0.3/0.35   Spanne 0.55   (alle drei aehnlich)
#
# Der Vergleich laeuft gegen die GLOBALE Spanne (0.25 bis 1.8), nicht die
# der Region - sonst haette das Nebelrode, dessen drei Typen ohnehin
# dicht beieinanderliegen, dieselben kurzen Uebergaenge wie das Skerrheim
# mit seinem Wand-neben-Bucht-Sprung, und genau der Unterschied ist gemeint.
# WIE WEIT p2 HOECHSTENS REICHT - GEBAUT, GEMESSEN, VORERST AUS.
#
# Nutzerwunsch 2026-08-25: *"p2 abhaengig von der kuestenaehnlichkeit ...
# also kann 1500 m weit reichen, oder aber 400 m weit"*. Die Mechanik dafuer
# steht unten in `_uebergaenge_aus_aehnlichkeit()` und ist ueber
# P2_AUS_AEHNLICHKEIT einzuschalten. **Vorgabe ist AUS**, und zwar aus
# gemessenem Grund - nicht aus Vergesslichkeit.
#
# smoke_test_regionen_welt (5 Seeds, 384 px):
#
#     p2-Regel                      Befunde   Naht   Atlantik-Hang (soll 10)
#     fest je Archetyp (Vorgabe)       4      1.546        ok
#     Aehnlichkeit, max  620           5      1.521        ok
#     Aehnlichkeit, max  700           5      1.457        ok
#     Aehnlichkeit, max 1000, exp 1.6  7      1.314       7.7
#     Aehnlichkeit, max 1500, exp 2.2  7      1.301       6.0
#     Aehnlichkeit, max 1500, exp 1.0  6      1.300       5.0
#
# smoke_test_kuestenprofiltreue, selbst bei der zahmsten Fassung (max 700):
#
#     fest je Archetyp:  3/3 gruen, Median 1.8 m, 13 von 13 flachen getroffen
#     Aehnlichkeit 700:  2/3,       Median 5.6 m, Kykladen-Strand 12 m daneben
#
# DER ZIELKONFLIKT, offen benannt: die variable Reichweite verbessert die
# Naht (1.546 -> 1.457), kostet aber die Zusicherung *"flache kueste bleibt
# flach"* - und die ist eine aeltere, ausdrueckliche Nutzervorgabe. Ein
# flaches Profil, das weiter ins Land traegt, wird unterwegs von steilen
# Nachbarn hochgezogen; genau das beschreibt schon der Kommentar bei
# MISCH_EXPONENT, hier nur mit umgekehrtem Vorzeichen.
#
# GEGENGEPRUEFT, WOHER DIE KOSTEN KOMMEN: p1 auf 178/208 zu ziehen (der
# andere Teil derselben Nutzervorgabe) kostet auf BEIDEN Tests nichts -
# 4 Befunde und 3/3, jeweils unveraendert. Die Kosten stecken
# ausschliesslich in der Reichweite.
#
# WOHIN DAS GEHOERT: die tiefe Kopplung Kueste -> Hinterland ist der Zweck
# des Gebietssystems (docs/archiv/2026-08-25_AUFRAEUMPLAN.md) - es leitet die mittlere Hoehe
# des Hinterlands aus dem Kuestenarchetyp ab UND erhaelt dabei das
# Regionsmittel, weshalb es die Eichung nicht umwirft. p2 zu strecken
# erreicht dasselbe Ziel auf die grobe Tour und kaempft gegen die Eichung.
# Wenn das Gebietssystem steht, ist die Reichweitenfrage neu zu stellen.
P2_MAX_M = 700.0                # bei voellig gleichem Nachbarn
# Wie scharf die Aehnlichkeit auf die Reichweite durchschlaegt. 1 = linear.
#
# GEMESSEN 2026-08-25 (384 px, Seed 20260804). Linear reicht p2 zu weit:
# benachbarte Abschnitte laufen ueber hunderte Meter ineinander, flache
# Archetypen werden von steilen Nachbarn hochgezogen, und ganze Regionen
# verflachen, weil ein Strandprofil weit ins Land traegt.
#
#     Exp   p2 Median   Profilfehler   flache daneben   Atlantik-Hang
#     1.0     1206 m        9.2 m            1               7.4
#     1.6     1067 m        8.1 m            1               9.2
#     2.2      951 m        7.4 m            0              10.8
#     3.0      825 m        7.1 m            0              12.0
#
# Der Estrande-Hang soll 10.0 sein - 2.2 trifft ihn am besten und
# raeumt zugleich die flachen Ausreisser weg. p2 liegt damit im Median noch
# immer bei 951 m, also fast doppelt so weit wie die festen 500-550 m
# davor. 3.0 waere schaerfer, uebersteuert den Hang aber nach oben.
# Bei P2_MAX_M = 700 ist der Exponent nur noch eine Feinheit (die Spanne
# ist klein). 1.6 aus der Messreihe oben uebernommen.
P2_AEHNLICHKEIT_EXPONENT = 1.6

# False stellt p2 auf den festen Archetypwert zurueck - siehe P2_MAX_M.
P2_AUS_AEHNLICHKEIT = False
P2_MIN_FAKTOR = 2.0             # Untergrenze als Vielfaches von p1
HOEHENFAKTOR_SPANNE = 1.8 - 0.25

# WIE SCHARF DER NAECHSTE KUESTENABSCHNITT DOMINIERT.
#
# Das Mischgewicht faellt mit `(1 - d/zone) ** MISCH_EXPONENT`. Der Wert
# steuert, wie stark ein weiter entfernter Abschnitt noch mitredet:
#
#   1  linear      - entfernte Abschnitte zaehlen fast voll mit
#   2  quadratisch - erster Anlauf, GEMESSEN ZU SCHWACH
#   4  scharf      - der naechste Abschnitt setzt sich durch
#
# GEMESSEN 2026-08-24, warum 2 nicht reicht: das Skerrheim hat 2106
# Schaerenkuesten-Pixel (h(150 m) = 14 m) gegen 1299 Fjordwand-Pixel
# (Soll 120 m). Die Fjordwand wurde von ihren flachen Nachbarn
# heruntergezogen - an ihren eigenen Pixeln stand 41 m, wo das Profil
# 270 m sagt.
#
# ZU SCHARF WAERE AUCH FALSCH: bei sehr hohem Exponenten gewinnt praktisch
# immer der naechste Abschnitt allein, und an jeder Segmentgrenze
# entstuende wieder die harte Naht, deretwegen ueberhaupt gemischt wird.
MISCH_EXPONENT = 4.0


# WIE TIEF DIE KUESTE JE REGION INS LAND GREIFT (Nutzervorgabe
# 2026-08-24: *"kannst du machen, dass die fjordland kueste 50% tiefer
# greift?"*).
#
# Faktor auf p1 UND p2 gleichermassen - die Zone wird als Ganzes
# gestreckt, das Verhaeltnis von vollem Profil zu Uebergang bleibt.
#
# WARUM DAS GERADE DEM FJORDLAND HELFEN KANN: seine Archetypen sind die
# steilsten der Karte (Fjordwand h(500 m) = 466 m), und gemessen erreichen
# sie ihr Profil nicht - die Fjordwand kommt auf 23 m statt 120 m nach
# 150 m. Ein Grund ist, dass das Rauschgelaende dort landeinwaerts SINKT
# (105 m bei 50 m, 59 m bei 150 m Abstand): schmale Fjordgrate. Greift die
# Kueste tiefer, haelt das Profil laenger gegen dieses absinkende
# Hinterland an.
KUESTEN_TIEFE_JE_REGION = {
    "Skerrheim": 1.5,
}


def _region_von_archetyp(name):
    """Zu welcher Region gehoert dieser Archetyp."""
    for _z, _s, r in alle_regionen():
        for t in KUESTEN_ARCHETYPEN.get(r["name"], []):
            if t["name"] == name:
                return r["name"]
    return None


def _tiefe_faktor(name):
    region = _region_von_archetyp(name)
    return KUESTEN_TIEFE_JE_REGION.get(region, 1.0)


def _steilheit(name):
    """h(150 m) / 150 m des Archetyps, 0 wenn kein Profil vorliegt."""
    profil = MESS_PROFIL_M_JE_ARCHETYP.get(name)
    if profil is None:
        return None
    return float(np.interp(150.0, PROFIL_STELLEN_M,
                           np.asarray(profil, dtype=np.float64))) / 150.0


def _zone_mischung(name):
    """0 = flachster Archetyp, 1 = steilster."""
    st = _steilheit(name)
    if st is None:
        return 1.0                      # Rueckfall: wie eine Klippe
    t = (st - STEIL_FLACH) / max(STEIL_STEIL - STEIL_FLACH, 1e-9)
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)      # smoothstep


def _zone_p1(name):
    """Ab hier blendet das Rauschgelaende ein."""
    m = _zone_mischung(name)
    return _tiefe_faktor(name) * (
        PROFIL_VOLL_FLACH_M + m * (PROFIL_VOLL_STEIL_M - PROFIL_VOLL_FLACH_M))


def _zone_p2(name):
    """Ab hier gilt nur noch das Rauschgelaende."""
    m = _zone_mischung(name)
    return _tiefe_faktor(name) * (
        PROFIL_ENDE_FLACH_M + m * (PROFIL_ENDE_STEIL_M - PROFIL_ENDE_FLACH_M))


# PROFILMASSSTAB: die Kuestenprofile auf die Groesse der Welt bringen.
#
# Nutzerbefund 2026-08-26: *"wenn ich sehe dass Skerrheim nur 200m hoehe hat
# (im Profil) und wir aber einen faktor zu den hoehen haben ... dementsprechend
# sollten die kuestenprofile von der hoehe etwas gestaucht sein."*
#
# DER BEFUND. Die Regionsparameter sind als Zielbild gesetzt (Nevadin 1050 m
# Relief auf 3800 m Formgroesse). Die Kuestenprofile sind an echten DEMs in
# ECHTEN Metern gemessen. Beide Skalen wurden nie aufeinander bezogen.
# Verhaeltnis "hoechstes Profil im Band 400-700 m" zu "halbem Regionsrelief":
#
#     Clonagh 3.73   Skerrheim 2.31   Samarcia 1.70   Macchia 1.61
#     Morobora 1.52   Nebelrode 1.41   Griech. Inseln 1.00
#     Estrande 0.82   Nevadin 0.36
#
# In sieben von neun Regionen ragt die Kueste hoeher auf als das Land dahinter.
#
# IN BEIDEN ACHSEN, NICHT NUR IN DER HOEHE. Am 2026-08-24 hat derselbe Nutzer
# verlangt: *"die kuestenprofile sind dadurch nicht gestreckt oder gestaucht
# sondern haben das gleiche hoehen zu tiefen verhaeltnis wie in echt"*. Eine
# reine Hoehenstauchung wuerde das verletzen - die Profile waeren flacher als
# die Wirklichkeit. Ein Faktor auf BEIDE Achsen laesst Form und Neigung
# unveraendert und macht nur ein Modell im Massstab daraus:
#
#     h'(x) = f * h(x / f)
#
# DIE FLACHEN TYPEN BLEIBEN UNVERAENDERT. Nutzervorgabe 2026-08-26:
# *"lass uns erstmal verkleinern, ausser bei den flachen typen pro region ...
# dann haben wir zB auch kuerzere straende und das mag ich nicht so."*
# Kriterium ist die BEREITS VORHANDENE Definition aus
# tests/smoke_test_kuestenprofiltreue.py (h(150 m) <= 25 m) - eine zweite
# Definition daneben waere eine zweite Wahrheit. Sie passt auch sachlich: zu
# hoch ragen die STEILEN Profile, die flachen waren nie das Problem.
#
# NUR VERKLEINERN, NIE STRECKEN. Das Nevadin braeuchte rechnerisch Faktor
# 2.81, also eine Streckung des 900-m-Profils auf 2529 m - weit ueber p2
# (hoechstens 700 m). Der Faktor ist deshalb bei 1.0 gedeckelt. Praktisch
# gegenstandslos: das Nevadin hat auf 62 von 64 Karten keine Kueste.
PROFILMASSSTAB_MIN = 0.35
FLACH_GRENZE_H150_M = 25.0


def _profil_flach(name):
    """Ist der Archetyp flach im Sinne von smoke_test_kuestenprofiltreue?"""
    werte = MESS_PROFIL_M_JE_ARCHETYP.get(name)
    if werte is None:
        return True
    h150 = float(np.interp(150.0, PROFIL_STELLEN_M,
                           np.asarray(werte, dtype=np.float64)))
    return h150 <= FLACH_GRENZE_H150_M


def _profilmasstab():
    """Archetypname -> Faktor fuer BEIDE Achsen. 1.0 heisst unveraendert."""
    stellen = np.asarray(PROFIL_STELLEN_M, dtype=np.float64)
    band = ((stellen >= HINTERLAND_BAND_M[0])
            & (stellen <= HINTERLAND_BAND_M[1]))
    faktoren = {}
    for _z, _s, r in alle_regionen():
        typen = KUESTEN_ARCHETYPEN.get(r["name"]) or []
        steile = [t["name"] for t in typen
                  if t["name"] in MESS_PROFIL_M_JE_ARCHETYP
                  and not _profil_flach(t["name"])]
        for t in typen:
            faktoren[t["name"]] = 1.0
        if not steile:
            continue                      # Region ganz aus flachen Typen
        hoechstes = max(
            float(np.asarray(MESS_PROFIL_M_JE_ARCHETYP[n],
                             dtype=np.float64)[band].mean())
            for n in steile)
        ziel = 0.5 * float(r["relief_m"])
        f = float(np.clip(ziel / max(hoechstes, 1e-9), PROFILMASSSTAB_MIN, 1.0))
        for n in steile:
            faktoren[n] = f
    return faktoren


PROFILMASSSTAB = _profilmasstab()


def _gemessene_profile_m():
    """
    Archetypname -> Hoehenprofil in METERN, auf den Weltmassstab gebracht.

    h'(x) = f * h(x / f) - siehe PROFILMASSSTAB oben. Bei f = 1 kommt die
    Messtabelle unveraendert heraus.
    """
    stellen = np.asarray(PROFIL_STELLEN_M, dtype=np.float64)
    tabelle = {}
    for name, werte in MESS_PROFIL_M_JE_ARCHETYP.items():
        roh = np.asarray(werte, dtype=np.float64)
        f = PROFILMASSSTAB.get(name, 1.0)
        if abs(f - 1.0) < 1e-9:
            tabelle[name] = roh
            continue
        # Ausserhalb der Messung (x/f > 900 m) haelt np.interp den letzten
        # Wert - genau richtig: das Hinterland steigt dort nicht weiter,
        # weil wir es nicht gemessen haben.
        tabelle[name] = f * np.interp(stellen / f, stellen, roh)
    return tabelle


def _gemessene_formen():
    """Archetypname -> gemessene Profilform seiner Region."""
    tabelle = {}
    for _z, _s, r in alle_regionen():
        for t in KUESTEN_ARCHETYPEN.get(r["name"], []):
            form = MESS_FORM_JE_ARCHETYP.get(t["name"])
            if form is not None:
                tabelle[t["name"]] = np.asarray(form, dtype=np.float64)
    return tabelle


def _gemessene_reichweiten():
    """
    Archetypname -> Reichweite in Metern, aus der Messung skaliert.

    Der gemessene Wert ist EIN Wert je Region (Medianprofil ueber alle
    Stationen der Vorbildkueste), die Archetypen einer Region haben aber
    verschiedene Reichweiten. Deshalb wird die Katalogstaffelung erhalten und
    nur ihr MEDIAN auf den gemessenen Wert gezogen - die Abstufung Klippe /
    Bucht / Strand bleibt, die Groessenordnung stimmt.
    """
    tabelle = {}
    for _z, _s, r in alle_regionen():
        archetypen = KUESTEN_ARCHETYPEN.get(r["name"])
        gemessen = MESS_REICHWEITE_M.get(r["name"])
        if not archetypen or not gemessen:
            continue
        katalog = np.array([t["reichweite_km"] * 1000.0 for t in archetypen])
        faktor = gemessen / max(float(np.median(katalog)), 1e-9)
        for t, k in zip(archetypen, katalog):
            tabelle[t["name"]] = float(k * faktor)
    return tabelle


def _gemessene_hoehen():
    """
    Archetypname -> gemessene Zielhoehe in Metern.

    Die Archetypen einer Region behalten ihre RANGFOLGE aus dem Katalog
    (Klippe hoeher als Bucht hoeher als Strand), aber die absolute Skala
    kommt aus der Messung. So bleibt die gestalterische Abstufung erhalten
    und die Groessenordnung stimmt.
    """
    tabelle = {}
    for _z, _s, r in alle_regionen():
        archetypen = KUESTEN_ARCHETYPEN.get(r["name"])
        spanne = MESSWERTE_JE_REGION.get(r["name"])
        if not archetypen or not spanne:
            continue
        faktoren = [t["hoehe_faktor"] for t in archetypen]
        lo, hi = min(faktoren), max(faktoren)
        for t in archetypen:
            anteil = ((t["hoehe_faktor"] - lo) / (hi - lo)) if hi > lo else 0.5
            tabelle[t["name"]] = spanne[0] + anteil * (spanne[1] - spanne[0])
    return tabelle


GEMESSENE_HOEHE = _gemessene_hoehen()
GEMESSENE_REICHWEITE = _gemessene_reichweiten()
GEMESSENE_FORM = _gemessene_formen()
GEMESSENES_PROFIL_M = _gemessene_profile_m()
GEMESSENE_HINTERLANDHOEHE = _gemessene_hinterlandhoehen()


def _bogen_glaetten(werte, kontur, bogen, sigma_stationen):
    """
    Werte entlang der Kuestenlinie glaetten, Kontur fuer Kontur.

    Die Stationen kommen NICHT in Bogenreihenfolge an (sie stammen aus
    einer Regionsmaske) und koennen auf mehreren Konturen liegen. Beides
    muss beruecksichtigt werden, sonst glaettet man ueber einen
    Inselsprung hinweg zwei Kuesten zusammen, die sich nie beruehren.

    Rueckgabe hat dieselbe Reihenfolge wie die Eingabe.
    """
    werte = np.asarray(werte, dtype=np.float64)
    if sigma_stationen <= 0.0 or len(werte) < 3:
        return werte
    aus = werte.copy()
    for nummer in np.unique(kontur):
        wo = np.flatnonzero(kontur == nummer)
        if len(wo) < 3:
            continue
        ordnung = np.argsort(bogen[wo])
        reihe = wo[ordnung]
        # `wrap` waere fuer geschlossene Konturen richtig, `nearest` fuer
        # offene. Da hier nur die Stationen EINER REGION vorliegen, ist der
        # Ausschnitt praktisch immer offen - auch auf einer geschlossenen
        # Insel endet die Region irgendwo. Deshalb `nearest`.
        aus[reihe] = ndimage.gaussian_filter1d(
            werte[reihe], sigma_stationen, mode="nearest")
    return aus


def _bogen_rauschen(rng, kontur, bogen, staerke, sigma_stationen):
    """
    Kohaerentes Rauschen entlang der Kueste - siehe
    SAAT_KOHAERENZ_STATIONEN.

    Das Glaetten senkt die Streuung (Gauss ueber sigma daempft um rund
    1/sqrt(2*sqrt(pi)*sigma)), deshalb wird HINTERHER auf `staerke`
    normiert. Ohne das waere der Jitter faktisch abgeschaltet und die
    Zuweisung rein hoehengetrieben, also ueberall dieselbe.
    """
    roh = rng.normal(0.0, 1.0, size=len(bogen))
    glatt = _bogen_glaetten(roh, kontur, bogen, sigma_stationen)
    streuung = float(np.std(glatt))
    if streuung < 1e-9:
        return np.zeros_like(glatt)
    return glatt * (staerke / streuung)


class VektorKueste:
    """
    Die Kuestenlinie einer Karte als vektorielle Beschreibung.

    EINMAL aus der Basis-Heightmap gebaut, danach beliebig oft an beliebigen
    Koordinaten auswertbar - das ist der ganze Zweck: das Mesh fragt an
    seinen freien Vertexpositionen, das Raster an den Pixelmitten.
    """

    def __init__(self, H_basis, region_map, seed, welt_km=WELT_KM,
                 schliessen=True):
        self.H_basis = np.asarray(H_basis, dtype=np.float64)
        self.size = self.H_basis.shape[0]
        self.mpp = welt_km * 1000.0 / self.size
        self.region_map = np.asarray(region_map)
        self.seed = int(seed)
        self.schliessen = bool(schliessen)

        self._inseln_bauen()
        self._linie_bauen()
        self._linien_inseln_zuordnen()
        self._saaten_bauen()
        self._segmente_bauen()
        self._segment_tabelle_bauen()
        self._tiefenfeld_bauen()
        self._baender_bauen()

    def _hinterland_je_station(self):
        """
        Die Gelaendehoehe, auf die eine Station zulaeuft - gemessen im
        Basisgelaende, nicht aus dem Katalog.

        GEMESSENER GRUND (Inseltest 2026-08-19): mit der Katalog-Zielhoehe
        hebt das Profil das Gelaende bei 300 m Kuestenabstand auf 128 m,
        bei 600 m endet die Reichweite und uebergibt an ein Basisgelaende
        mit 29 m - ein Absturz von 99 m als Ringgraben rings um jede Insel.
        Dieselbe Erscheinung mit und ohne Zackenkueste, es ist also kein
        Formartefakt, sondern ein Bruch zwischen Katalog und Gelaende.

        Die Wirklichkeit sagt dasselbe: Moher ist 147 m hoch, WEIL das
        Plateau dahinter 147 m hoch ist - beides ist dieselbe Zahl, kein
        Zufall. Die Hoehe kommt vom Gelaende, der Archetyp bestimmt nur, wie
        schnell und ueber welche Strecke man sie erreicht.

        Gemessen wird der Median des Basisgelaendes in einem Ring um die
        Station: Kuestenabstand zwischen 0.7 und 1.4 Reichweiten, hoechstens
        1.5 Reichweiten Luftlinie entfernt. Median statt Mittel, damit ein
        einzelner Gipfel im Ausschnitt die Kueste nicht hochzieht.
        """
        n = len(self.saat_xy)
        if n == 0:
            return np.zeros(0)

        land = self.H_basis > 0.0
        if not land.any():
            return np.zeros(n)
        abstand_m = ndimage.distance_transform_edt(land) * self.mpp

        # Landpixel ausgeduennt - fuer einen Median genuegt jedes zweite in
        # jeder Richtung, und es macht die Nachbarschaftssuche bezahlbar.
        schritt = max(1, int(round(60.0 / self.mpp)))
        ys, xs = np.nonzero(land[::schritt, ::schritt])
        ys, xs = ys * schritt, xs * schritt
        if len(ys) < 4:
            return np.zeros(n)
        proben_xy = np.column_stack([xs, ys]).astype(np.float64)
        proben_d = abstand_m[ys, xs]
        proben_h = self.H_basis[ys, xs]
        baum = cKDTree(proben_xy)

        ergebnis = np.zeros(n)
        for k in range(n):
            reichweite_px = self.saat_reichweite_m[k] / self.mpp
            nachbarn = baum.query_ball_point(self.saat_xy[k], 1.5 * reichweite_px)
            if not nachbarn:
                continue
            idx = np.asarray(nachbarn)
            d_dort = proben_d[idx]
            im_ring = ((d_dort >= 0.7 * self.saat_reichweite_m[k])
                       & (d_dort <= 1.4 * self.saat_reichweite_m[k]))
            if im_ring.sum() >= 3:
                ergebnis[k] = float(np.median(proben_h[idx[im_ring]]))
            else:
                # KEIN RING VORHANDEN - die Landmasse ist schmaler als die
                # Reichweite des Archetyps (jede kleine Insel).
                #
                # Vorher stand hier `max(...)`, also der hoechste Punkt in
                # Reichweite. Das ist eine Extremwertstatistik und schwankt
                # entsprechend: auf der 150-m-Insel sprang die Zielhoehe
                # zwischen Nachbarstationen um 64 m. Jetzt der Median des
                # TIEFSTEN VIERTELS (dem Landesinneren am naechsten) - eine
                # robuste Lage statt eines Ausreissers.
                d_dort_alle = proben_d[idx]
                schwelle = np.quantile(d_dort_alle, 0.75)
                tief = d_dort_alle >= schwelle
                ergebnis[k] = float(np.median(proben_h[idx[tief]]))
        return ergebnis

    def _segmente_bauen(self):
        """
        Zusammenhaengende Stationen gleichen Typs zu SEGMENTEN gruppieren.

        Bis hierher wurden die Profilwerte zwischen JE ZWEI Stationen
        interpoliert - damit ist jeder Punkt der Kueste eine Mischung und
        kein Typ je rein zu sehen. Nutzer-Vorgabe war das Gegenteil: *"soll ja
        nicht zu abrupt sein, aber auch nicht zu langsam (definierte kuesten
        nicht eine mischung ueberall)"*.

        Deshalb: im Inneren eines Segments ist der Typ rein, nur an den
        Segmentgrenzen liegt eine Uebergangszone fester Breite.

        Segmente unter MIN_SEGMENT_M werden in den laengeren Nachbarn
        eingeschmolzen - ein Segment, das kuerzer ist als zwei
        Uebergangsbreiten, besteht nur aus Uebergaengen und zeigt seinen Typ
        nie. Auf einer kleinen Insel ist die GANZE Kontur kuerzer als das
        Mindestsegment; sie bekommt dadurch von selbst genau einen Typ.
        """
        self.segmente = []
        if not len(self.saat_xy):
            return

        for nummer in np.unique(self.saat_kontur):
            idx = np.flatnonzero(self.saat_kontur == nummer)
            idx = idx[np.argsort(self.saat_bogen[idx])]
            namen = [self.saat_archetyp[i]["name"] for i in idx]
            bogen_m = self.saat_bogen[idx] * self.mpp
            laenge = self.kontur_laenge.get(int(nummer))
            laenge_m = laenge * self.mpp if laenge is not None else None

            # LAUF-ID JE STATION statt Indexbereichen.
            #
            # Ein erster Anlauf fuehrte Laeufe als [start, ende]-Paare und
            # verschmolz mit min/max. Bei geschlossener Kontur laeuft ein
            # Lauf aber ueber das Ende hinaus, und min/max ueber die
            # umgebrochenen Indizes ergab dann den GANZEN Umfang als ein
            # Segment (gemessen: 0..31711 m, alle anderen darin enthalten).
            # Mit IDs je Station gibt es keine Umlaufarithmetik mehr.
            n_st = len(namen)
            ids = np.zeros(n_st, dtype=np.int32)
            for k in range(1, n_st):
                ids[k] = ids[k - 1] + (0 if namen[k] == namen[k - 1] else 1)
            zyklisch = laenge_m is not None
            if zyklisch and n_st > 1 and namen[0] == namen[-1] and ids[-1] > 0:
                ids[ids == ids[-1]] = 0

            def _laenge_von_id(wert):
                treffer = np.flatnonzero(ids == wert)
                if len(treffer) < 2:
                    return 0.0
                # Bogenlaenge des Laufs, Umlauf mitgerechnet
                schritte = np.diff(np.sort(bogen_m[treffer]))
                if zyklisch and (0 in treffer and n_st - 1 in treffer):
                    return laenge_m - (schritte.max() if len(schritte) else 0.0)
                return float(bogen_m[treffer].max() - bogen_m[treffer].min())

            # Zu kurze Laeufe in den laengeren Nachbarn einschmelzen
            for _runde in range(n_st):
                werte = list(dict.fromkeys(ids.tolist()))
                if len(werte) < 2:
                    break
                kurz = [w for w in werte if _laenge_von_id(w) < MIN_SEGMENT_M]
                if not kurz:
                    break
                w = kurz[0]
                i = werte.index(w)
                vor = werte[(i - 1) % len(werte)]
                nach = werte[(i + 1) % len(werte)]
                ziel = vor if _laenge_von_id(vor) >= _laenge_von_id(nach) else nach
                ids[ids == w] = ziel

            for wert in dict.fromkeys(ids.tolist()):
                treffer = np.flatnonzero(ids == wert)
                station = idx[treffer[0]]
                typ = self.saat_archetyp[station]
                b_sortiert = np.sort(bogen_m[treffer])
                if zyklisch and len(b_sortiert) > 1:
                    # Groesste Luecke finden - dort laeuft das Segment NICHT
                    # entlang, also beginnt es dahinter.
                    luecken = np.diff(np.concatenate(
                        [b_sortiert, [b_sortiert[0] + laenge_m]]))
                    g = int(np.argmax(luecken))
                    a_m = float(b_sortiert[(g + 1) % len(b_sortiert)])
                    b_m = float(b_sortiert[g])
                    if b_m < a_m:
                        b_m += laenge_m
                else:
                    a_m, b_m = float(b_sortiert[0]), float(b_sortiert[-1])
                self.segmente.append({
                    "kontur": int(nummer), "a": float(a_m), "b": float(b_m),
                    "laenge_m": laenge_m, "name": typ["name"],
                    "tanwinkel": max(np.tan(np.radians(
                        min(typ["winkel_grad"], MAX_KLIPPENWINKEL_GRAD))), 0.05),
                    # GEMESSEN statt Katalog - siehe MESS_REICHWEITE_M.
                    "reichweite_m": GEMESSENE_REICHWEITE.get(
                        typ["name"], typ["reichweite_km"] * 1000.0),
                    "ueberhoehung": 1.0 + UEBERHOEHUNG_JE_HOEHENFAKTOR
                                    * typ["hoehe_faktor"],
                    # GEMESSEN statt Katalog - siehe MESSWERTE_JE_REGION.
                    "katalog_m": GEMESSENE_HOEHE.get(
                        typ["name"], KUESTENHOEHE_M * typ["hoehe_faktor"]),
                    "form": GEMESSENE_FORM.get(typ["name"]),
                    # Das gemessene Profil in METERN - siehe
                    # MESS_PROFIL_M_JE_ARCHETYP. Ersetzt die Kombination
                    # aus normierter Form, Zielhoehe und Reichweite.
                    "profil_m": GEMESSENES_PROFIL_M.get(typ["name"]),
                    # Strahltiefe je Segment, aus der Steilheit des
                    # Archetyps - siehe PROFIL_VOLL_FLACH_M.
                    "voll_m": _zone_p1(typ["name"]),
                    "uebergang_m": _zone_p2(typ["name"]) - _zone_p1(typ["name"]),
                })

        self._segmente_schliessen()
        self._uebergaenge_aus_aehnlichkeit()

    def _uebergaenge_aus_aehnlichkeit(self):
        """
        p2 je Segment aus der Aehnlichkeit zu seinen NACHBARN - siehe
        P2_MAX_M weiter oben.

        Laeuft NACH `_segmente_schliessen()`, weil erst dort feststeht,
        welche Segmente es am Ende wirklich gibt: die Funktion schmilzt
        Segmente unter MIN_SEGMENT_M in ihre Nachbarn ein. Vorher gerechnet
        haetten wir die Aehnlichkeit gegen Nachbarn bestimmt, die es
        anschliessend nicht mehr gibt.
        """
        hoehe_faktor = {}
        for _z, _s, r in alle_regionen():
            for t in KUESTEN_ARCHETYPEN.get(r["name"], []):
                hoehe_faktor[t["name"]] = float(t["hoehe_faktor"])

        je_kontur = {}
        for seg in self.segmente:
            je_kontur.setdefault(int(seg["kontur"]), []).append(seg)

        for nummer, segmente in je_kontur.items():
            segmente.sort(key=lambda s: s["a"])
            n = len(segmente)
            # Geschlossene Konturen haben einen zyklischen Nachbarn, offene
            # nicht - bei denen ist der Rand sein eigener Nachbar (also
            # maximal aehnlich, langer Uebergang).
            zyklisch = self.kontur_laenge.get(nummer) is not None
            for k, seg in enumerate(segmente):
                eigen = hoehe_faktor.get(seg["name"])
                if eigen is None:
                    continue
                nachbarn = []
                for versatz in (-1, 1):
                    j = k + versatz
                    if 0 <= j < n:
                        nachbarn.append(segmente[j])
                    elif zyklisch and n > 1:
                        nachbarn.append(segmente[j % n])
                sprung = max(
                    (abs(eigen - hoehe_faktor.get(nb["name"], eigen))
                     for nb in nachbarn), default=0.0)
                aehnlich = 1.0 - min(sprung / HOEHENFAKTOR_SPANNE, 1.0)
                aehnlich = aehnlich ** P2_AEHNLICHKEIT_EXPONENT

                p1 = float(seg["voll_m"])
                unten = P2_MIN_FAKTOR * p1
                p2 = unten + aehnlich * max(P2_MAX_M - unten, 0.0)
                if not P2_AUS_AEHNLICHKEIT:
                    # Der Zustand VOR dem 2026-08-25, absichtlich erreichbar
                    # geblieben: p2 fest je Archetyp. Die Messreihe oben
                    # vergleicht gegen genau diesen Zustand, und ohne den
                    # Schalter muesste sie jemand von Hand nachbauen.
                    p2 = _zone_p2(seg["name"])
                seg["uebergang_m"] = max(p2 - p1, MIN_UEBERGANG_M)
                seg["aehnlichkeit"] = float(aehnlich)

    def _segmente_schliessen(self):
        """
        Segmentgrenzen bis zur Mitte zum Nachbarn ziehen, damit die Kontur
        LUECKENLOS bedeckt ist.

        GEMESSENER GRUND (2026-08-22): ein Segment reichte bis hierher nur
        von seiner ersten bis zu seiner letzten Station. Zwischen dem letzten
        Punkt eines Laufs und dem ersten des naechsten liegt aber ein ganzer
        Stationsabstand (rund 400 m), den keins der beiden Segmente abdeckte.
        Gemessen war die Fenstersumme dort 0, und `_segment_felder()` griff
        auf seine Vorgabewerte zurueck - im Diagramm der Typwerte laengs der
        Kueste als Spitzen auf 450 m Zielhoehe und 300 m Reichweite sichtbar,
        also auf Werte, die kein Archetyp dieser Region ueberhaupt hat.
        9.6 % der Kontur waren betroffen.

        Jetzt treffen sich benachbarte Segmente auf halbem Weg. Zusammen mit
        der halben Uebergangsbreite, um die jedes Fenster ohnehin
        hinausreicht, ist die Kontur damit vollstaendig und einfach bedeckt.
        """
        for nummer in {s["kontur"] for s in self.segmente}:
            teil = sorted([s for s in self.segmente if s["kontur"] == nummer],
                          key=lambda s: s["a"])
            if len(teil) < 2:
                if teil and teil[0]["laenge_m"]:
                    # Einziges Segment einer geschlossenen Kontur: voller Umlauf
                    teil[0]["a"] = 0.0
                    teil[0]["b"] = teil[0]["laenge_m"]
                continue
            laenge = teil[0]["laenge_m"]
            for i in range(len(teil) - 1):
                mitte = 0.5 * (teil[i]["b"] + teil[i + 1]["a"])
                teil[i]["b"] = mitte
                teil[i + 1]["a"] = mitte
            if laenge:
                # Zyklisch: letztes und erstes Segment treffen sich ebenfalls
                lueck = (teil[0]["a"] + laenge) - teil[-1]["b"]
                mitte = teil[-1]["b"] + 0.5 * lueck
                teil[-1]["b"] = mitte
                teil[0]["a"] = mitte - laenge

    @staticmethod
    def _smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _segment_tabelle_bauen(self, schritt_m=10.0):
        """
        Die Segmentwerte EINMAL ueber die Bogenlaenge tabellieren.

        GEMESSENER GRUND: `_segment_felder()` lief fuer JEDEN Abfragepunkt
        ueber alle Segmente - bei 102 Segmenten, drei Kuestenplaetzen und
        147000 Pixeln sind das rund 45 Millionen Fensterauswertungen. Das
        volle Weltfeld brauchte dadurch 10.9 s statt 1.2 s (Faktor 9).

        Die Werte haengen aber nur von (Kontur, Bogenlage) ab, nicht vom
        Abfrageort. Einmal alle 10 m tabelliert und danach linear
        nachgeschlagen ist dasselbe Ergebnis - die Fenster sind ueber
        UEBERGANG_M = 250 m glatt, 10 m Stuetzweite loest das um Faktor 25
        ueber.
        """
        self._tabelle = {}
        formen = [s["form"] for s in self.segmente if s.get("form") is not None]
        n_form = len(formen[0]) if formen else 0
        profile_m = [s["profil_m"] for s in self.segmente
                     if s.get("profil_m") is not None]
        n_profil = len(profile_m[0]) if profile_m else 0
        halb = 0.5 * UEBERGANG_M

        for nummer in {s["kontur"] for s in self.segmente}:
            teil = [s for s in self.segmente if s["kontur"] == nummer]
            laenge = teil[0]["laenge_m"]
            if laenge is None:
                enden = [s["b"] for s in teil] + [s["a"] for s in teil]
                laenge = max(enden) - min(enden) + UEBERGANG_M
            # Winzige Konturen (wenige Pixel) haben Bogenlaenge nahe 0 - ohne
            # diese Untergrenze wird die Tabellen-Schrittweite 0, die
            # Indexrechnung liefert NaN und der Zugriff stuerzt ab.
            laenge = max(float(laenge), UEBERGANG_M)
            n = max(8, int(np.ceil(laenge / schritt_m)) + 1)
            s_gitter = np.linspace(0.0, laenge, n)

            summe = np.zeros(n)
            tanw = np.zeros(n)
            reich = np.zeros(n)
            ueber = np.zeros(n)
            katalog = np.zeros(n)
            form = np.zeros((n, n_form)) if n_form else None
            profil_m = np.zeros((n, n_profil)) if n_profil else None
            voll = np.zeros(n)
            uebergang = np.zeros(n)
            # WELCHES Segment an dieser Bogenstelle fuehrt, und wie stark.
            # Genau dieselbe Frage beantwortete archetyp_felder() bisher mit
            # einer eigenen Schleife ueber ALLE Segmente mal ALLE Landpixel -
            # O(S*N), gemessen 4.2 s bei 1024 px, weil die Segmentzahl mit
            # der Aufloesung waechst (die Kuestenkontur wird laenger).
            # Hier kostet es nichts extra: das Fenster wird ohnehin gerechnet.
            bestes = np.full(n, -1, dtype=np.int32)
            bestes_w = np.zeros(n)
            global_index = [self.segmente.index(t_seg) for t_seg in teil]

            for j_seg, seg in enumerate(teil):
                fenster = np.zeros(n)
                versaetze = [0.0]
                if seg["laenge_m"]:
                    versaetze += [seg["laenge_m"], -seg["laenge_m"]]
                for v in versaetze:
                    w = (self._smoothstep((s_gitter + v - seg["a"] + halb)
                                          / UEBERGANG_M)
                         * self._smoothstep((seg["b"] - (s_gitter + v) + halb)
                                            / UEBERGANG_M))
                    fenster = np.maximum(fenster, w)
                nimm = fenster > bestes_w
                # Der Index im GESAMTEN Segmentfeld, einmal vorab bestimmt.
                # `self.segmente.index(seg)` waere eine lineare Suche je
                # Segment und damit O(S^2) - gemessen stieg _linie_bauen()
                # dadurch von 0.73 s auf 1.58 s.
                bestes = np.where(nimm, global_index[j_seg], bestes)
                bestes_w = np.where(nimm, fenster, bestes_w)
                summe += fenster
                tanw += fenster * seg["tanwinkel"]
                reich += fenster * seg["reichweite_m"]
                ueber += fenster * seg["ueberhoehung"]
                katalog += fenster * seg["katalog_m"]
                voll += fenster * seg.get("voll_m", PROFIL_VOLL_M)
                uebergang += fenster * seg.get("uebergang_m", UEBERGANG_LAND_M)
                if form is not None and seg.get("form") is not None:
                    form += fenster[:, None] * seg["form"][None, :]
                if profil_m is not None and seg.get("profil_m") is not None:
                    profil_m += fenster[:, None] * seg["profil_m"][None, :]

            sicher = np.maximum(summe, 1e-9)
            leer = summe <= 1e-9
            tanw, reich = tanw / sicher, reich / sicher
            ueber, katalog = ueber / sicher, katalog / sicher
            voll, uebergang = voll / sicher, uebergang / sicher

            # VARIANZ ENTLANG DER KUESTE - siehe ZONEN_VARIANZ_M.
            #
            # p1 und p2 wandern unabhaengig voneinander um bis zu
            # ZONEN_VARIANZ_M, als glatte Funktion der BOGENLAENGE. Drei
            # inkommensurable Wellenlaengen ergeben ein Muster, das sich
            # erst nach vielen Kilometern wiederholt - "sanft wechselnd,
            # nicht regelmaessig".
            #
            # DIE PHASEN sind aus Kartenseed UND Konturnummer abgeleitet:
            # dieselbe Karte gibt dieselbe Varianz, aber zwei Konturen
            # schwingen nicht im Gleichtakt.
            #
            # UNTERSCHIEDLICHE PHASEN fuer p1 und p2 (Versatz 1.7 bzw.
            # 3.9 im Bogenmass): wanderten beide im Gleichschritt, bliebe
            # die Uebergangsbreite konstant und die Varianz waere nur eine
            # Verschiebung der ganzen Zone.
            p1 = voll
            p2 = voll + uebergang
            wurf = np.random.default_rng(
                (int(self.seed) << 12) ^ (int(nummer) & 0xFFF))
            phasen = wurf.random(6) * 2.0 * np.pi
            for versatz, ziel in ((0, "p1"), (3, "p2")):
                welle = np.zeros_like(s_gitter)
                for k, wl in enumerate(ZONEN_WELLEN_M):
                    welle += np.sin(2.0 * np.pi * s_gitter / wl
                                    + phasen[versatz + k])
                # Drei Sinuswellen summieren sich auf +-3; auf +-1 bringen,
                # damit ZONEN_VARIANZ_M wirklich die Hoechstauslenkung ist.
                welle = welle / 3.0
                if ziel == "p1":
                    p1 = p1 + ZONEN_VARIANZ_M * welle
                else:
                    p2 = p2 + ZONEN_VARIANZ_M * welle

            # p2 muss hinter p1 bleiben, sonst waere die Uebergangszone
            # negativ und der Smoothstep sprunghaft.
            p2 = np.maximum(p2, p1 + MIN_UEBERGANG_M)
            voll = p1
            uebergang = p2 - p1
            if form is not None:
                form = form / sicher[:, None]
            if profil_m is not None:
                profil_m = profil_m / sicher[:, None]
            if leer.any():
                tanw[leer], reich[leer] = 1.0, 200.0
                ueber[leer], katalog[leer] = 1.0, 100.0
                voll[leer] = PROFIL_VOLL_M
                uebergang[leer] = UEBERGANG_LAND_M
                if form is not None:
                    form[leer] = np.linspace(0.0, 1.0, form.shape[1])
                if profil_m is not None:
                    profil_m[leer] = 0.0
            self._tabelle[int(nummer)] = (s_gitter, tanw, reich, ueber,
                                          katalog, form, laenge, bestes,
                                          profil_m, voll, uebergang)

    def _segment_felder(self, kontur_q, bogen_q):
        """
        Nachschlagen in der Tabelle - siehe _segment_tabelle_bauen().

        Rueckgabe (tanw, reich, ueber, katalog, form, profil_m, voll,
        uebergang). `profil_m` ist das gemessene Hoehenprofil in METERN an
        PROFIL_STELLEN_M, je Abfragepunkt zwischen den Segmenten gemischt;
        `voll`/`uebergang` sind die Zonentiefen dieses Kuestenabschnitts.
        """
        n = len(bogen_q)
        s_q = bogen_q * self.mpp
        tanw = np.ones(n)
        reich = np.full(n, 200.0)
        ueber = np.ones(n)
        katalog = np.full(n, 100.0)
        voll = np.full(n, PROFIL_VOLL_M)
        uebergang = np.full(n, UEBERGANG_LAND_M)
        form = None
        profil_m = None
        for nummer, eintrag in self._tabelle.items():
            treffer = kontur_q == nummer
            if not treffer.any():
                continue
            gitter, t_, r_, u_, k_, f_, laenge, _bestes, p_, v_, ue_ = eintrag
            lage = np.mod(s_q[treffer], laenge) if laenge else s_q[treffer]
            tanw[treffer] = np.interp(lage, gitter, t_)
            reich[treffer] = np.interp(lage, gitter, r_)
            ueber[treffer] = np.interp(lage, gitter, u_)
            katalog[treffer] = np.interp(lage, gitter, k_)
            voll[treffer] = np.interp(lage, gitter, v_)
            uebergang[treffer] = np.interp(lage, gitter, ue_)
            if p_ is not None:
                # Gleichmaessiges Gitter -> Index statt Interpolation je
                # Stuetzstelle, wie schon bei `form`.
                schritt_p = (gitter[1] - gitter[0]) if len(gitter) > 1 else 1.0
                schritt_p = schritt_p if schritt_p > 1e-9 else 1.0
                pos_p = np.clip(lage / schritt_p, 0.0, len(gitter) - 1.001)
                pos_p = np.nan_to_num(pos_p, nan=0.0, posinf=0.0, neginf=0.0)
                unten_p = pos_p.astype(np.int32)
                bruch_p = (pos_p - unten_p)[:, None]
                if profil_m is None:
                    profil_m = np.zeros((n, p_.shape[1]))
                profil_m[treffer] = (p_[unten_p] * (1.0 - bruch_p)
                                     + p_[unten_p + 1] * bruch_p)
            if f_ is not None:
                if form is None:
                    form = np.tile(np.linspace(0.0, 1.0, f_.shape[1]), (n, 1))
                # EIN Zugriff statt einer Interpolation je Stuetzstelle: das
                # Tabellengitter ist gleichmaessig, der Index also direkt
                # ausrechenbar. Mit 17 Stuetzstellen und 84 Konturen waren es
                # sonst rund 1400 einzelne np.interp-Aufrufe.
                schritt = (gitter[1] - gitter[0]) if len(gitter) > 1 else 1.0
                schritt = schritt if schritt > 1e-9 else 1.0
                pos = np.clip(lage / schritt, 0.0, len(gitter) - 1.001)
                pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
                unten = pos.astype(np.int32)
                bruch = (pos - unten)[:, None]
                form[treffer] = (f_[unten] * (1.0 - bruch)
                                 + f_[unten + 1] * bruch)
        return tanw, reich, ueber, katalog, form, profil_m, voll, uebergang

    def _segment_felder_direkt(self, kontur_q, bogen_q):
        """
        Typabhaengige Profilwerte an beliebigen Bogenpositionen.

        Je Segment ein Fenster, das im Inneren 1 ist und ueber
        UEBERGANG_M auf 0 abfaellt - um eine halbe Uebergangsbreite ueber
        die Segmentgrenze hinaus verlaengert, damit sich die Fenster
        benachbarter Segmente genau an der Grenze zur Haelfte ueberlappen.
        Dort ergibt die gewichtete Mischung 50:50, im Segmentinneren 100:0.

        Smoothstep statt linear: eine lineare Rampe hat an beiden Enden der
        Uebergangszone einen Knick in der Ableitung, und der ist als Kante
        laengs der Kueste sichtbar.
        """
        n = len(bogen_q)
        bogen_m = bogen_q * self.mpp
        summe = np.zeros(n)
        tanw = np.zeros(n)
        reich = np.zeros(n)
        ueber = np.zeros(n)
        katalog = np.zeros(n)
        # `or ()` geht hier nicht - ein numpy-Array hat keinen Wahrheitswert.
        formen = [s["form"] for s in self.segmente if s.get("form") is not None]
        n_form = len(formen[0]) if formen else 0
        form = np.zeros((n, n_form)) if n_form else None
        halb = 0.5 * UEBERGANG_M

        for seg in self.segmente:
            auf_kontur = kontur_q == seg["kontur"]
            if not auf_kontur.any():
                continue
            s = bogen_m
            fenster = np.zeros(n)
            # Zyklisch: auch die um +-Laenge verschobenen Lagen pruefen.
            versaetze = [0.0]
            if seg["laenge_m"]:
                versaetze += [seg["laenge_m"], -seg["laenge_m"]]
            for v in versaetze:
                w = (self._smoothstep((s + v - seg["a"] + halb) / UEBERGANG_M)
                     * self._smoothstep((seg["b"] - (s + v) + halb) / UEBERGANG_M))
                fenster = np.maximum(fenster, w)
            fenster = np.where(auf_kontur, fenster, 0.0)
            summe += fenster
            tanw += fenster * seg["tanwinkel"]
            reich += fenster * seg["reichweite_m"]
            ueber += fenster * seg["ueberhoehung"]
            katalog += fenster * seg["katalog_m"]
            if form is not None and seg.get("form") is not None:
                form += fenster[:, None] * seg["form"][None, :]

        sicher = np.maximum(summe, 1e-9)
        leer = summe <= 1e-9
        tanw, reich = tanw / sicher, reich / sicher
        ueber, katalog = ueber / sicher, katalog / sicher
        if form is not None:
            form = form / sicher[:, None]
        # Punkte ohne jedes Segment (Kontur ohne Stationen): Vorgabewerte.
        # Sollte seit _segmente_schliessen() nicht mehr vorkommen - gemessen
        # war das frueher 9.6 % der Kontur, und die Vorgabewerte erschienen
        # dort als Spitzen auf Hoehen, die kein Archetyp hat.
        if leer.any():
            tanw[leer] = 1.0
            reich[leer] = 200.0
            ueber[leer] = 1.0
            katalog[leer] = 100.0
            if form is not None:
                form[leer] = np.linspace(0.0, 1.0, form.shape[1])
        return tanw, reich, ueber, katalog, form

    def _glaetten_laengs(self, werte):
        """
        Gleitendes Mittel ueber Nachbarstationen DERSELBEN Kontur, zyklisch.

        GEMESSENER GRUND: die rohe Hinterlandmessung schwankte zwischen
        benachbarten Stationen um bis zu 85 m auf 400 m Kueste, also
        200 m/km. An echten Kuesten wurden 20 m/km gemessen (19 km
        Doolin-Moher, Median) - Faktor 10 zu viel. Die Ursache ist das
        Gelaenderauschen im Messring, nicht eine echte Formaenderung der
        Kueste; ungeglaettet wandert diese Zufallszahl direkt in die
        Zielhoehe und laesst die Klippe laengs zappeln.
        """
        if len(werte) == 0 or HINTERLAND_GLAETTUNG < 2:
            return werte
        ergebnis = werte.copy()
        for nummer in np.unique(self.saat_kontur):
            idx = np.flatnonzero(self.saat_kontur == nummer)
            if len(idx) < 3:
                continue
            idx = idx[np.argsort(self.saat_bogen[idx])]
            reihe = werte[idx]
            k = min(HINTERLAND_GLAETTUNG, len(reihe))
            geschlossen = self.kontur_laenge.get(int(nummer)) is not None
            if geschlossen:
                # Zyklisch: die Reihe umlaufend verlaengern, dann falten.
                lang = np.concatenate([reihe[-k:], reihe, reihe[:k]])
                gefaltet = np.convolve(lang, np.ones(k) / k, mode="same")
                ergebnis[idx] = gefaltet[k:k + len(reihe)]
            else:
                ergebnis[idx] = np.convolve(
                    np.pad(reihe, k, mode="edge"), np.ones(k) / k,
                    mode="same")[k:k + len(reihe)]
        return ergebnis

    def _baender_bauen(self):
        """
        Das Basisgelaende in GROESSENBAENDER zerlegen.

        Nutzer-Vorgabe 2026-08-19 zu (e): *"dass die oktaven 1 und 2 langsam
        sich hineinblenden. 3 und 4 koennten eventuell auch frueher schon
        herauskommen."*

        Dafuer muss bekannt sein, welcher Teil des Gelaendes grob und welcher
        fein ist. `VektorKueste` bekommt aber ein fertiges Hoehenfeld, keinen
        Oktavenstapel - also wird per Gauss-Pyramide zerlegt. Das ist
        allgemeiner als den Generator anzuzapfen: es funktioniert auch fuer
        Gelaende, das gar nicht aus Oktaven stammt (Testfaelle, importierte
        Karten), und koppelt dieses Modul nicht an die Rauschquelle.

        Die Schnittwellenlaengen folgen den Oktaven des Programms
        (GRUNDFORM_M / 2^k = 12000, 6000, 3000, ... m):

            Band 0   ueber 3000 m     Oktaven 1-2, die Grossform
            Band 1   750 - 3000 m     Oktaven 3-4
            Band 2   188 - 750 m      Oktaven 5-6
            Band 3   unter 188 m      Oktaven 7-9, die Feinstruktur

        Die Baender summieren sich exakt zum Basisgelaende - es geht nichts
        verloren und nichts kommt doppelt vor.
        """
        H = self.H_basis
        schnitte_m = (3000.0, 750.0, 188.0)

        # Jeweils ALLES oberhalb der Schnittwellenlaenge, direkt aus H -
        # nicht kaskadiert. So ist die Summe der Baender exakt H.
        tiefpass = []
        for welle in schnitte_m:
            sigma_px = max(0.6, (welle / 6.0) / self.mpp)
            tiefpass.append(ndimage.gaussian_filter(H, sigma=sigma_px))

        self.baender = [tiefpass[0]]                       # > 3000 m
        for k in range(1, len(tiefpass)):
            self.baender.append(tiefpass[k] - tiefpass[k - 1])
        self.baender.append(H - tiefpass[-1])               # < 188 m

    def archetyp_felder(self, H):
        """
        `kuesten_archetyp` und `kuesten_staerke` wie sie der Rasterweg liefert.

        NICHT KOSMETIK, sondern Voraussetzung fuer die Meerestiefe:
        `_seetiefe_aus_archetyp()` sucht je Seepixel den naechstgelegenen
        Land-Archetyp und leitet daraus die Tiefe ab. Ohne belegtes
        Archetypfeld findet es keinen Treffer und gibt die Karte UNVERAENDERT
        zurueck - gemessen blieben dadurch 21485 Seepixel (14.6 % der Karte)
        auf nahezu Meereshoehe stehen, statt auf mindestens -10 m
        abzufallen. Der Fehler war stumm: keine Ausnahme, nur eine flache See.

        `archetyp` ist der Index INNERHALB der Region (0..2), wie ihn
        KUESTEN_ARCHETYPEN[regionsname] auffuehrt - dieselbe Konvention wie
        im Rasterweg, damit Anzeige und Seetiefe unveraendert weiterlesen.
        """
        archetyp = np.full(H.shape, -1, dtype=np.int8)
        staerke = np.zeros(H.shape, dtype=np.float32)
        if not self.segmente:
            return archetyp, staerke

        # Archetypname -> (Regionsindex, lokaler Index)
        lage = {}
        for i, (_z, _s, r) in enumerate(alle_regionen()):
            for k, t in enumerate(KUESTEN_ARCHETYPEN.get(r["name"], [])):
                lage[t["name"]] = (i, k)

        land = H > 0.0
        if not land.any():
            return archetyp, staerke
        ys, xs = np.nonzero(land)

        # DIE AUSWAHL AUS hoehe() WIEDERVERWENDEN, nicht neu rechnen.
        #
        # Bis 2026-08-23 stand hier ein zweiter, vollstaendiger
        # _kuesten_waehlen()-Aufruf ueber alle Landpixel - dieselbe
        # KD-Baum-Abfrage mit K_KANDIDATEN Nachbarn, dieselbe gierige
        # Auswahl, dasselbe _auf_strecken(). als_raster() hatte die Werte
        # unmittelbar davor schon gerechnet und weggeworfen. Gemessen kostete
        # die Doppelarbeit 7.9 s von 43 s im Knoten terrain.redistribution.
        #
        # Der Cache haengt an der Punktzahl (`n_alle`), damit er nur greift,
        # wenn hoehe() wirklich mit demselben Vollraster gefragt wurde. Ein
        # Aufruf mit anderer Punktmenge - etwa der Mesh-Abtaster an_punkten()
        # - laesst ihn nicht zu und rechnet wie bisher neu.
        #
        # AUSSERHALB DER REICHWEITE bleibt `archetyp` auf -1 und `staerke`
        # auf 0. Das ist keine Luecke: `staerke` war dort schon vorher
        # rechnerisch 0 (clip(1 - d/reichweite)^2 mit d > reichweite), und
        # der Archetyp eines Pixels tief im Landesinneren wird nirgends
        # gelesen - _seetiefe_aus_archetyp() sucht je Seepixel den
        # NAECHSTGELEGENEN Land-Archetyp, und das ist immer ein Kuestenpixel.
        cache = getattr(self, "_wahl_cache", None)
        flach = ys.astype(np.int64) * self.size + xs.astype(np.int64)
        if cache is not None and cache["n_alle"] == H.size:
            pos = np.full(H.size, -1, dtype=np.int64)
            pos[cache["nah"]] = np.arange(len(cache["nah"]))
            zeile = pos[flach]
            trifft = zeile >= 0
            ys, xs, zeile = ys[trifft], xs[trifft], zeile[trifft]
            gew_d = cache["d"][zeile]
            gew_kontur = cache["kontur"][zeile]
            gew_bogen = cache["bogen"][zeile]
            punkte = np.stack([xs.astype(float), ys.astype(float)], axis=1)
        else:
            punkte = np.stack([xs.astype(float), ys.astype(float)], axis=1)
            nah, _g = self._nahe_punkte(punkte)
            ys, xs, punkte = ys[nah], xs[nah], punkte[nah]
            gew_d, gew_kontur, gew_bogen, _anzahl = self._kuesten_waehlen(punkte)
        if len(punkte) == 0:
            return archetyp, staerke

        # WELCHES SEGMENT DECKT DIESE BOGENLAGE - aus der Tabelle, nicht aus
        # einer eigenen Schleife.
        #
        # Hier stand bis 2026-08-23 eine Schleife ueber ALLE Segmente, in der
        # je Segment ein Smoothstep-Fenster ueber ALLE Landpixel gerechnet
        # wurde: O(Segmente * Pixel). Das faellt bei kleinen Karten nicht
        # auf, waechst aber doppelt mit der Aufloesung - mehr Pixel UND eine
        # laengere Kuestenkontur, also mehr Segmente. Gemessen 4.2 s bei
        # 1024 px.
        #
        # `_segment_tabelle_bauen()` rechnet dieselben Fenster ohnehin, um
        # Winkel, Reichweite und Profilform ueber die Bogenlaenge zu mitteln.
        # Seit demselben Tag legt es dabei auch ab, WELCHES Segment an jeder
        # Gitterstelle fuehrt - das Nachschlagen ist damit ein Indexzugriff.
        s_q = gew_bogen[:, 0] * self.mpp
        bestes = np.full(len(punkte), -1, dtype=np.int32)
        for nummer, eintrag in self._tabelle.items():
            treffer = gew_kontur[:, 0] == nummer
            if not treffer.any():
                continue
            gitter, _t, _r, _u, _k, _f, laenge, tab_bestes, _p, _v, _ue = eintrag
            # NICHT `lage` nennen - so heisst weiter oben das Dict
            # Archetypname -> (Regionsindex, lokaler Index), und es wird
            # unten noch gebraucht.
            bogen_lage = np.mod(s_q[treffer], laenge) if laenge else s_q[treffer]
            schritt = (gitter[1] - gitter[0]) if len(gitter) > 1 else 1.0
            schritt = schritt if schritt > 1e-9 else 1.0
            pos = np.clip(bogen_lage / schritt, 0.0, len(gitter) - 1.0)
            pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
            bestes[treffer] = tab_bestes[np.round(pos).astype(np.int32)]

        gut = bestes >= 0
        namen = np.array([s["name"] for s in self.segmente])
        lokal = np.array([lage.get(n, (0, 0))[1] for n in namen], dtype=np.int8)
        archetyp[ys[gut], xs[gut]] = lokal[bestes[gut]]

        # STAERKE MIT DENSELBEN ZONEN WIE hoehe(), nicht mit der alten
        # `reichweite_m`.
        #
        # GEMESSENER FEHLER (2026-08-24): hier stand
        # `clip(1 - d / reichweite_m)^2`. `reichweite_m` ist die Groesse
        # aus dem FRUEHEREN Modell (110-349 m je Region); seit dem
        # Zonenumbau rechnet `_hoehe_block()` aber mit `voll_m`/
        # `uebergang_m` (350/500 m, im Skerrheim 525/750 m). Bei einem
        # Pixel 392 m vor der Kueste wurde damit `1 - 392/296` negativ und
        # auf 0 geklemmt - die Fjordwand-Pixel bekamen `staerke` 0.07, die
        # Schaerenkueste 0.00, obwohl die Kueste dort voll wirkt.
        #
        # Die Hoehenfunktion selbst war davon NICHT betroffen (sie rechnet
        # eigenstaendig), wohl aber alles, was `kuesten_staerke` liest:
        # `_seetiefe_aus_archetyp()` fuer die Meerestiefe und die
        # 2D-Anzeige im Kuestentypen-Modus.
        #
        # Jetzt dieselbe Kennlinie wie in _hoehe_block(): voll bis `voll_m`,
        # dann Smoothstep ueber `uebergang_m`.
        voll = np.array([s.get("voll_m", PROFIL_VOLL_M) for s in self.segmente])
        ueberg = np.array([s.get("uebergang_m", UEBERGANG_LAND_M)
                           for s in self.segmente])
        v_q = np.where(gut, voll[np.maximum(bestes, 0)], PROFIL_VOLL_M)
        u_q = np.where(gut, ueberg[np.maximum(bestes, 0)], UEBERGANG_LAND_M)
        auslauf = self._smoothstep(
            (gew_d[:, 0] - v_q) / np.maximum(u_q, 1e-9))
        staerke[ys, xs] = np.clip(1.0 - auslauf, 0.0, 1.0)
        return archetyp, staerke

    def _tiefenfeld_bauen(self):
        """
        Je Landmasse ihre groesste Kuestenentfernung (`d_max`) - die Zahl, mit
        der zur Inselmitte hin geschlossen wird (docs/KUESTENMODELL.md §4).

        Gemessen am Inseltest: ohne dieses Schliessen waechst eine 500-m-Insel
        von 179 m auf 271 m Hoehe, weil das Kuestenprofil die Mitte hochschiebt
        statt dort auszulaufen. Je kleiner die Insel, desto groesser der
        Fehler.

        d_max je ZUSAMMENHAENGENDER Landmasse, nicht punktgenau: fuer eine
        kleine Insel ist das exakt (eine Insel, ein Wert), fuer das Festland
        grob - dort greift das Schliessen aber ohnehin nicht, weil die
        Reichweite (max 700 m) weit unter d_max (gemessen 2750 m) liegt.
        """
        land = self.H_basis > 0.0
        if not land.any():
            self.d_max_karte = np.full(self.H_basis.shape, 1e9)
            return
        abstand = ndimage.distance_transform_edt(land) * self.mpp
        komponente, anzahl = ndimage.label(land)
        if anzahl == 0:
            self.d_max_karte = np.full(self.H_basis.shape, 1e9)
            return
        je_komponente = ndimage.maximum(abstand, komponente,
                                        index=np.arange(1, anzahl + 1))
        nachschlag = np.concatenate([[1e9], np.maximum(je_komponente, 1e-6)])
        self.d_max_karte = nachschlag[komponente]

        # HOECHSTER PUNKT JE LANDMASSE - die zweite Haelfte des Deckels.
        #
        # GEMESSENER GRUND: der Deckel `ZIEL_JE_HINTERLAND * d_max` misst nur
        # die BREITE der Landmasse. Auf der 150-m-Insel ergibt das 70 m,
        # obwohl das Basisgelaende dort 180 m hoch ist - die Kueste wurde auf
        # 110 m gekappt und die Insel dadurch um 70 m niedriger als ihr
        # eigenes Gelaende. Ein Laplace-Ausgleich behob das, aber nur weil er
        # die Basis direkt als Randbedingung nimmt.
        #
        # Der Deckel soll verhindern, dass die Kueste Hoehe ERFINDET, die die
        # Landmasse nicht traegt. Was das Basisgelaende schon hat, ist per
        # Definition getragen - also gilt der groessere der beiden Werte.
        hoehe_je_komponente = ndimage.maximum(
            self.H_basis, komponente, index=np.arange(1, anzahl + 1))
        nachschlag_h = np.concatenate([[0.0], np.maximum(hoehe_je_komponente, 0.0)])
        self.h_max_karte = nachschlag_h[komponente]

    # ------------------------------------------------------------------ #

    def _inseln_bauen(self):
        """
        Welches Landstueck ist welches - und welche Kontur gehoert dazu.

        NUTZERVORGABE 2026-08-24: *"der einfluss soll nur innerhalb der
        eigenen insel erfolgen (nicht zB zwei inseln und der einfluss reicht
        auf die andere insel oder ins meer rueber)"*. Und zum Weg dorthin:
        *"wir erstellen eine kontur an einer stelle fuer die kueste, dabei
        kann doch der koordinatenbereich einer insel zugewiesen werden
        anschliessend"* - genau so, und billiger als gedacht.

        `ndimage.label` auf der Landmaske gibt jedem zusammenhaengenden
        Landstueck eine Nummer. Ein Abfragepunkt bekommt die Nummer des
        Pixels, auf dem er liegt; eine Kontur die Nummer ihrer LANDSEITE.
        Beim Vergleich zaehlt dann nur, was zur selben Nummer gehoert.

        WARUM DAS NOETIG IST. Der KD-Baum kennt nur Abstaende. Zwei Inseln,
        die 200 m auseinanderliegen, liegen beide innerhalb der 350-m-Zone
        der jeweils anderen - ohne diese Pruefung formt die Steilkueste der
        einen Insel das Ufer der anderen mit.

        SEE BEKOMMT NUMMER 0 und gehoert damit zu keiner Insel. Das ist
        richtig so: die Landformung wirkt ohnehin nur auf Land
        (`ist_land`), und der Ozeanuebergang bekommt seine eigene Regel.
        """
        land = self.H_basis > 0.0
        if not land.any():
            self.insel_karte = np.zeros(self.H_basis.shape, dtype=np.int32)
            self.linien_insel = np.zeros(0, dtype=np.int32)
            return
        marken, _anzahl = ndimage.label(land)
        self.insel_karte = marken.astype(np.int32)

    def _linien_inseln_zuordnen(self):
        """
        Je Konturpunkt die Insel seiner LANDSEITE.

        Die Kontur liegt auf der Wasserlinie, also zwischen Land und See -
        das Pixel unter ihr kann beides sein. Deshalb wird in einem kleinen
        Umkreis das haeufigste Landstueck genommen: robust gegen den
        halben Pixel Versatz, den Marching Squares erzeugt.
        """
        n = len(self.linienpunkte)
        if n == 0 or not hasattr(self, "insel_karte"):
            self.linien_insel = np.zeros(n, dtype=np.int32)
            return
        marken = self.insel_karte
        gefunden = np.zeros(n, dtype=np.int32)
        # Im Ring um den Punkt suchen, bis eine Landmarke auftaucht.
        for radius_px in (0.7, 1.5, 2.5):
            offen = gefunden == 0
            if not offen.any():
                break
            for winkel in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
                xi = np.clip(np.round(self.linienpunkte[offen, 0]
                                      + radius_px * np.cos(winkel)
                                      ).astype(np.int32), 0, self.size - 1)
                yi = np.clip(np.round(self.linienpunkte[offen, 1]
                                      + radius_px * np.sin(winkel)
                                      ).astype(np.int32), 0, self.size - 1)
                marke = marken[yi, xi]
                wo = np.flatnonzero(offen)
                nimm = (gefunden[wo] == 0) & (marke > 0)
                gefunden[wo[nimm]] = marke[nimm]
        self.linien_insel = gefunden

    def _linie_bauen(self):
        """Kuestenlinie abtasten und einen KDTree fuer den Abstand bauen."""
        from gui.widgets.kuesten_mesh import _entlang_abtasten, kuestenlinien

        abstand_px = max(1.0, LINIEN_ABSTAND_M / self.mpp)
        stuecke, kontur_id, bogen_px = [], [], []
        laenge_px = 0.0
        for nummer, linie in enumerate(kuestenlinien(self.H_basis, pegel=0.0)):
            if len(linie) < 3:
                continue
            # Laenge auf der ORIGINALKONTUR summieren, nicht auf der
            # abgetasteten: der Abtastabstand ist bei grober Aufloesung auf
            # 1 px geklemmt, `Punktzahl * Abstand` waere dort also nicht die
            # Kuestenlaenge, sondern eine Funktion der Klemme. Genau diese
            # Verwechslung hat der erste Testlauf gemessen.
            laenge_px += float(np.hypot(*(np.diff(linie, axis=0).T)).sum())
            abgetastet = _entlang_abtasten(linie, abstand_px)
            stuecke.append(abgetastet)
            kontur_id.append(np.full(len(abgetastet), nummer, dtype=np.int32))
            # BOGENLAENGE je Punkt - die Ordnungszahl entlang der Kueste.
            # Sie ist der Schluessel fuer die Parameterinterpolation: ein
            # Abfragepunkt bekommt seine Profilwerte nicht vom naechsten
            # Saatpunkt zugeteilt, sondern ZWISCHEN den beiden benachbarten
            # Stationen interpoliert.
            schritte = np.hypot(*(np.diff(abgetastet, axis=0).T))
            bogen_px.append(np.concatenate([[0.0], np.cumsum(schritte)]))

        if stuecke:
            self.linienpunkte = np.concatenate(stuecke, axis=0)
            self.linien_kontur = np.concatenate(kontur_id)
            self.linien_bogen = np.concatenate(bogen_px)
        else:
            self.linienpunkte = np.zeros((0, 2), dtype=np.float64)
            self.linien_kontur = np.zeros(0, dtype=np.int32)
            self.linien_bogen = np.zeros(0, dtype=np.float64)
        self.kuesten_laenge_m = laenge_px * self.mpp
        self.linien_baum = (cKDTree(self.linienpunkte)
                            if len(self.linienpunkte) else None)

    def _saaten_bauen(self):
        """
        Saatpunkte auf der Linie, je einer mit Archetyp und lokaler
        Referenzhoehe.

        Zuteilungsregel wie in `_kuesten_umformen()`: je Region der
        anspruchsvollste Archetyp zuerst, Quote strikt aus `max_anteil`
        (garantiertes Vorkommen, nicht nur Wahrscheinlichkeit), Auswahl nach
        Passung zum vorhandenen Rohgelaende plus Seed-Jitter.
        """
        self.saat_xy = np.zeros((0, 2), dtype=np.float64)
        self.saat_archetyp = []
        self.saat_lokal_hoehe = np.zeros(0, dtype=np.float64)
        self.saat_baum = None
        if self.linien_baum is None:
            return

        # SCHRITTWEITE AUS DEM TATSAECHLICHEN LINIENABSTAND, nicht aus dem
        # angeforderten.
        #
        # `_linie_bauen()` klemmt den Abtastabstand bei 1 px - bei grober
        # Aufloesung liegen die Linienpunkte also WEITER auseinander als
        # LINIEN_ABSTAND_M verlangt. Rechnete man `schritt` gegen den
        # angeforderten Wert, kaeme bei 256 px ein Saatabstand von 582 m
        # heraus statt der vorgegebenen 420 m (gemessen), bei 512 px dagegen
        # genau 420 - die Archetypzonen waeren also aufloesungsabhaengig
        # gross. Genau dieser Fehlertyp macht `_kuesten_umformen()`
        # aufloesungsabhaengig (dort ueber die Pixelindexliste), und er waere
        # hier fast unbemerkt nachgebaut worden.
        # STATIONSABSTAND JE KONTUR, nicht global.
        #
        # Der feste Meterabstand allein liess einer 500-m-Insel vier
        # Stationen (1670 m Kueste / 420 m) - vier Sektoren, deren Naehte im
        # Bild als Tortenstuecke erscheinen. Kurze Konturen bekommen deshalb
        # einen engeren Abstand, bis MIN_STATIONEN_JE_KONTUR erreicht ist.
        # Lange Konturen bleiben beim Sollabstand, es wird also nichts teurer,
        # wo nichts zu gewinnen ist.
        linien_abstand_echt_px = max(1.0, LINIEN_ABSTAND_M / self.mpp)
        teile_p, teile_k, teile_b, teile_g = [], [], [], []
        for nummer in np.unique(self.linien_kontur):
            maske = self.linien_kontur == nummer
            punkte_k = self.linienpunkte[maske]
            bogen_k = self.linien_bogen[maske]
            if len(punkte_k) < 2:
                continue
            laenge_px = float(bogen_k[-1])
            soll_px = SAAT_ABSTAND_M / self.mpp
            if laenge_px > 0:
                soll_px = min(soll_px, laenge_px / MIN_STATIONEN_JE_KONTUR)
            abstand_px = max(linien_abstand_echt_px, soll_px)
            schritt = max(1, int(round(abstand_px / linien_abstand_echt_px)))
            teile_p.append(punkte_k[::schritt])
            teile_k.append(np.full(len(punkte_k[::schritt]), nummer, dtype=np.int32))
            teile_b.append(bogen_k[::schritt])
            # WIEVIEL KUESTE DIESE STATION VERTRITT, in Metern.
            #
            # Nicht jede Station steht fuer gleich viel Kueste: die Zeile
            # `soll_px = min(soll_px, laenge_px / MIN_STATIONEN_JE_KONTUR)`
            # oben verdichtet kurze Konturen absichtlich, damit eine kleine
            # Insel nicht in vier Tortenstuecke zerfaellt. Eine 1.7-km-Insel
            # bekommt dadurch eine Station je ~105 m, das Festland eine je
            # 420 m. Fuer die Zonenform ist das richtig - fuer die Quote
            # weiter unten waere es ein Zaehlfehler, siehe dort.
            teile_g.append(np.full(len(punkte_k[::schritt]),
                                   schritt * linien_abstand_echt_px * self.mpp,
                                   dtype=np.float64))

        if not teile_p:
            return
        kandidaten = np.concatenate(teile_p)
        kandidaten_kontur = np.concatenate(teile_k)
        kandidaten_bogen = np.concatenate(teile_b)
        kandidaten_gewicht = np.concatenate(teile_g)
        if len(kandidaten) == 0:
            return

        # Konturlaenge merken - fuer die zyklische Interpolation gebraucht.
        self.kontur_laenge = {}
        for nummer in np.unique(self.linien_kontur):
            b = self.linien_bogen[self.linien_kontur == nummer]
            p = self.linienpunkte[self.linien_kontur == nummer]
            zu = float(np.hypot(*(p[0] - p[-1]))) if len(p) > 1 else 1e9
            # Geschlossen, wenn Anfang und Ende praktisch aufeinanderliegen.
            self.kontur_laenge[int(nummer)] = (
                float(b[-1]) if zu < 2.0 else None)

        # Region und lokale Rohgelaendehoehe je Saatpunkt.
        xi = np.clip(np.round(kandidaten[:, 0]).astype(int), 0, self.size - 1)
        yi = np.clip(np.round(kandidaten[:, 1]).astype(int), 0, self.size - 1)
        region_je_saat = self.region_map[yi, xi]
        lokal = np.array([
            float(np.mean(self.H_basis[max(0, y - 2):y + 3, max(0, x - 2):x + 3]))
            for y, x in zip(yi, xi)])

        alle_xy, alle_typ, alle_lokal = [], [], []
        alle_kontur, alle_bogen = [], []
        for i, (_z, _s, r) in enumerate(alle_regionen()):
            archetypen = KUESTEN_ARCHETYPEN.get(r["name"])
            if not archetypen:
                continue
            treffer = np.flatnonzero(region_je_saat == i)
            if len(treffer) < 2:
                continue

            rng = np.random.RandomState((self.seed ^ 0x4B57) + i * 97)
            hoehe_norm = np.clip(lokal[treffer] / KUESTENHOEHE_M, 0.0, 2.0)
            # Fuer das kohaerente Rauschen: wer liegt neben wem.
            kontur_je = kandidaten_kontur[treffer]
            bogen_je = kandidaten_bogen[treffer]
            # Auch die HOEHE glaetten. Sie ist der andere Summand im Score
            # und flackert sonst von Station zu Station - eine einzelne
            # hohe Stelle in einer flachen Bucht riss dort einen
            # Klippentyp auf.
            hoehe_norm = _bogen_glaetten(
                hoehe_norm, kontur_je, bogen_je, SAAT_KOHAERENZ_STATIONEN)
            n = len(treffer)

            # QUOTE NACH KUESTENLAENGE, NICHT NACH STATIONSZAHL (2026-08-25)
            #
            # `max_anteil` ist im Katalog als "Obergrenze am gesamten
            # Kuestenumfang der Region" beschrieben. Gerechnet wurde sie bis
            # hierher gegen die ZAHL der Saatstationen, und das ist NICHT
            # dasselbe, weil MIN_STATIONEN_JE_KONTUR kurze Konturen
            # absichtlich verdichtet:
            #
            #     55.5 % aller Saatstationen liegen auf 14.5 % der Kueste
            #     (384 px, Seed 20260804, 62 Konturen, 155 km).
            #
            # Faktor 3.8 - kleine Inseln kauften einen ueberproportionalen
            # Teil jeder Quote auf. Jede Station bringt jetzt ihr Stueck
            # Kueste als GEWICHT mit, und ein Archetyp bekommt Stationen,
            # bis sein Meterbudget voll ist. Eine Inselstation mit 105 m
            # zaehlt damit ein Viertel einer Festlandstation mit 420 m.
            #
            # WAS DAS BRINGT UND WAS ES KOSTET - beides gemessen:
            #
            #   + smoke_test_kuestenprofiltreue: Luce-Bay-Straende faellt
            #     von 32 m Abweichung auf 1 m. Der Archetyp stand vorher bei
            #     44 m Hoehe nach 150 m, wo die Vorlage 6 m sagt - ein
            #     Huegel, wo ein Strand stehen soll. Er war seit Beginn der
            #     Messreihe rot (als Dingle-Straende 57 m). Alle 11 flachen
            #     Archetypen sind damit gruen, schlechtester 4 m.
            #   + schlechteste Quotenabweichung 65.0 -> 40.1 Prozentpunkte.
            #   - smoke_test_regionen_welt bekommt EINEN Befund dazu (7 -> 8):
            #     Morobora-Hang 10.4 statt 7.5. Die Morobora hat nur ~33 Stationen
            #     in der ganzen Region, da schlaegt jede Umverteilung durch.
            #   o mittlere Quotenabweichung praktisch unveraendert
            #     (12.5 -> 12.8 Prozentpunkte).
            #
            # DIE NAHELIEGENDE VERSCHAERFUNG IST GEMESSEN SCHLECHTER.
            # Eine Variante, die das Budget NICHT ueberzieht (Station
            # ueberspringen statt anhaengen), sieht sauberer aus und ist in
            # jedem Punkt schwaecher:
            #
            #                              ueberziehend   nicht ueberziehend
            #   kuestenprofiltreue            3/3 gruen   2/3 (Luce-Bay 32 m)
            #   archetyp_verteilung           1 Befund    2 Befunde, darunter
            #                                             Fjordwand faellt bei
            #                                             17 Stationen ganz aus
            #   Quotentreue Mittel            12.8 P      14.0 P
            #   Quotentreue schlechteste      40.1 P      65.0 P
            #   regionen_welt Morobora-Hang      10.4        9.5 (beide rot)
            #
            # Grund: wer eine Station ueberspringt, weil sie nicht mehr ins
            # Budget passt, gibt sie an den NAECHSTEN Archetyp weiter - und
            # die Schleife laeuft vom steilsten zum flachsten. Die flachen
            # Typen erben dadurch genau die Stationen, die keiner wollte.
            # Deshalb die ueberziehende Fassung.
            #
            # DER EIGENTLICHE ENGPASS SITZT DANACH und ist NICHT behoben:
            # `_segmente_schliessen()` schmilzt Segmente unter MIN_SEGMENT_M
            # (750 m) in den laengeren Nachbarn ein. Ein Archetyp mit vielen
            # einzeln verstreuten Stationen verliert seine ganze Laenge an
            # den Nachbarn, egal wie die Quote gerechnet wurde -
            # Schaerenkueste und Algarve-Klippen kommen in BEIDEN
            # Zaehlweisen auf 0.0 % Laenge bei 16 bzw. 8 zugeteilten
            # Stationen. Dort muesste eine weitere Korrektur ansetzen.
            #
            # (Die urspruengliche ANNAHME, die flachen Archetypen laegen auf
            # Inseln und koennten ihr Profil dort nicht halten, war falsch.
            # Gemessen liegen sie auf dem Festland; siehe den Kommentar zur
            # Messstrecke in tests/smoke_test_kuestenprofiltreue.py.)
            gewicht_je = kandidaten_gewicht[treffer]
            gesamt_m = float(gewicht_je.sum())
            # Gemessene Korrektur, innerhalb der Region normiert - siehe
            # SAAT_BUDGET_KORREKTUR.
            roh = {a["name"]: a["max_anteil"]
                              * SAAT_BUDGET_KORREKTUR.get(a["name"], 1.0)
                   for a in archetypen}
            skala = (sum(a["max_anteil"] for a in archetypen)
                     / max(sum(roh.values()), 1e-9))
            ziel_meter = {name: wert * skala * gesamt_m
                          for name, wert in roh.items()}

            frei = set(range(n))
            zuordnung = {}
            for archetyp in sorted(archetypen, key=lambda a: -a["hoehe_faktor"]):
                if not frei:
                    break
                # KOHAERENT statt weiss - siehe SAAT_KOHAERENZ_STATIONEN.
                jitter = _bogen_rauschen(
                    rng, kontur_je, bogen_je, 0.15,
                    SAAT_KOHAERENZ_STATIONEN)
                score = -np.abs(hoehe_norm - archetyp["hoehe_faktor"]) + jitter
                kandidaten_idx = sorted(frei, key=lambda k: -score[k])
                # Auffuellen bis das Meterbudget erreicht ist. Mindestens
                # eine Station, damit ein Archetyp nie voellig verschwindet
                # (das war die Rolle des alten `max(1, ...)`).
                # Auffuellen, bis das Meterbudget erreicht ist. Mindestens
                # eine Station, damit ein Archetyp nie voellig verschwindet
                # (das war die Rolle des alten `max(1, ...)`).
                budget = ziel_meter[archetyp["name"]]
                gewaehlt, summe_m = [], 0.0
                for k in kandidaten_idx:
                    if gewaehlt and summe_m >= budget:
                        break
                    gewaehlt.append(k)
                    summe_m += float(gewicht_je[k])
                zuordnung.update({k: archetyp for k in gewaehlt})
                frei -= set(gewaehlt)
            if frei:
                rest = max(archetypen, key=lambda a: a["max_anteil"])
                zuordnung.update({k: rest for k in frei})

            for k in range(n):
                alle_xy.append(kandidaten[treffer[k]])
                alle_typ.append(zuordnung[k])
                alle_lokal.append(lokal[treffer[k]])
                alle_kontur.append(kandidaten_kontur[treffer[k]])
                alle_bogen.append(kandidaten_bogen[treffer[k]])

        if not alle_xy:
            return
        self.saat_xy = np.asarray(alle_xy, dtype=np.float64)
        self.saat_archetyp = alle_typ
        self.saat_lokal_hoehe = np.asarray(alle_lokal, dtype=np.float64)
        self.saat_baum = cKDTree(self.saat_xy)
        self.saat_kontur = np.asarray(alle_kontur, dtype=np.int32)
        self.saat_bogen = np.asarray(alle_bogen, dtype=np.float64)

        # Profilgroessen je Saatpunkt vorrechnen - sie haengen nur vom
        # Saatpunkt ab, nicht vom abgefragten Ort. Spart bei jeder Abfrage
        # eine Schleife ueber die Archetypen.
        self.saat_tanwinkel = np.array([
            max(np.tan(np.radians(min(t["winkel_grad"], MAX_KLIPPENWINKEL_GRAD))), 0.05)
            for t in alle_typ])
        self.saat_reichweite_m = np.array([
            t["reichweite_km"] * 1000.0 for t in alle_typ])

        # DIE ZIELHOEHE KOMMT AUS DEM GELAENDE, NICHT AUS DEM KATALOG.
        #
        # Der Katalogwert (`hoehe_faktor`) sagt nur noch, wieviel eine Kueste
        # ihr Hinterland UEBERHOEHT - er setzt keine absolute Hoehe mehr.
        # Begruendung und Messung siehe _hinterland_je_station(): mit einer
        # absoluten Zielhoehe entsteht rings um jede Insel ein Ringgraben,
        # weil das Profil auf Katalogniveau hebt und am Reichweitenrand an
        # ein viel niedrigeres Basisgelaende uebergibt.
        #
        # `ueberhoehung` ist bewusst klein: eine Klippenkante liegt etwas
        # ueber dem Plateau dahinter, aber nicht um ein Vielfaches. Der
        # Sockel haelt eine flache Kueste sichtbar, wo das Hinterland selbst
        # fast auf Null liegt.
        hinterland = self._glaetten_laengs(self._hinterland_je_station())
        self.saat_hinterland = hinterland
        ueberhoehung = np.array([
            1.0 + UEBERHOEHUNG_JE_HOEHENFAKTOR * t["hoehe_faktor"]
            for t in alle_typ])
        katalog = np.array([KUESTENHOEHE_M * t["hoehe_faktor"] for t in alle_typ])
        self.saat_zielhoehe = np.minimum(
            np.maximum(hinterland, 0.0) * ueberhoehung + KUESTEN_SOCKEL_M,
            katalog)

    # ------------------------------------------------------------------ #

    def basis_hoehe(self, x, y):
        """Bilinear abgetastete Basis-Heightmap an beliebigen Koordinaten."""
        H = self.H_basis
        h, w = H.shape
        x = np.clip(np.asarray(x, dtype=np.float64), 0.0, w - 1.001)
        y = np.clip(np.asarray(y, dtype=np.float64), 0.0, h - 1.001)
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        fx, fy = x - x0, y - y0
        oben = H[y0, x0] * (1 - fx) + H[y0, x1] * fx
        unten = H[y1, x0] * (1 - fx) + H[y1, x1] * fx
        return oben * (1 - fy) + unten * fy

    def _entlang_interpolieren(self, kontur_q, bogen_q, werte_je_station):
        """
        Mischt einen Stationswert linear zwischen den beiden Nachbarstationen
        DERSELBEN Kontur, nach Bogenposition.

        Je Kontur getrennt, weil zwei verschiedene Konturen (Festland und
        eine vorgelagerte Insel) an derselben Bogenposition voellig
        Verschiedenes bedeuten - waere die Interpolation kontur-uebergreifend,
        bekaeme die Insel die Klippe des Festlands.
        """
        ergebnis = np.empty(len(kontur_q), dtype=np.float64)
        for nummer in np.unique(kontur_q):
            treffer = kontur_q == nummer
            stationen = np.flatnonzero(self.saat_kontur == nummer)
            if len(stationen) == 0:
                # Kontur ohne eigene Station (zu kurz, oder ihre Region hat
                # keine Archetypen): naechstgelegene Station ueberhaupt.
                _a, idx = self.saat_baum.query(self.linienpunkte[
                    np.flatnonzero(self.linien_kontur == nummer)[:1]])
                ergebnis[treffer] = werte_je_station[idx[0]]
                continue
            if len(stationen) == 1:
                ergebnis[treffer] = werte_je_station[stationen[0]]
                continue
            ordnung = np.argsort(self.saat_bogen[stationen])
            stationen = stationen[ordnung]
            bogen = self.saat_bogen[stationen]
            werte = werte_je_station[stationen]

            # ZYKLISCH UMLAUFEN BEI GESCHLOSSENER KONTUR.
            #
            # `np.interp` klemmt ausserhalb des Stuetzstellenbereichs auf den
            # Randwert. Auf einer Insel heisst das: hinter der letzten Station
            # bleibt der Wert konstant, und an der willkuerlichen Startstelle
            # der Kontur springt er auf den ersten Wert zurueck. Gemessen an
            # einer 3-km-Insel: 336 m Kueste ohne Interpolation und 23.2 m
            # Sprung an der Naht - bei einer kleinen Insel mit fuenf
            # Stationen betrifft das ein Fuenftel des Umfangs.
            #
            # Behoben, indem die letzte Station VOR den Anfang und die erste
            # HINTER das Ende gespiegelt wird. Danach ist der Umlauf
            # geschlossen und es gibt keine ausgezeichnete Startstelle mehr.
            laenge = self.kontur_laenge.get(int(nummer))
            if laenge is not None and len(bogen) >= 2:
                bogen = np.concatenate([[bogen[-1] - laenge], bogen,
                                        [bogen[0] + laenge]])
                werte = np.concatenate([[werte[-1]], werte, [werte[0]]])

            ergebnis[treffer] = np.interp(bogen_q[treffer], bogen, werte)
        return ergebnis

    def _auf_strecken(self, punkte, idx):
        """
        Exakter Abstand zur KUESTENLINIE statt zu ihren Abtastpunkten.

        GEMESSENER GRUND (2026-08-22): die Linie wird alle LINIEN_ABSTAND_M
        (60 m) abgetastet, und der Abstand ging bis hierher gegen diese
        Einzelpunkte. Der Abstand zu einer Punktwolke ist aber nicht der
        Abstand zur Kurve - jeder Punkt bekommt seine eigene Voronoi-Zelle,
        und die Zellgrenzen erscheinen als PERLENKETTE laengs der Kueste.
        Gemessen gegen die echte Streckendistanz: Median 4.7 m Fehler, bis
        29 m - also rund die halbe Abtastweite, genau wie es die Theorie
        vorhersagt.

        Sichtbar wurde das erst bei 3.3 m/px; auf groeberen Karten lagen
        mehrere Abtastpunkte in einem Pixel und die Zellen fielen unter die
        Aufloesung.

        Hier wird deshalb auf die beiden an den Kandidaten angrenzenden
        STRECKEN projiziert. Das ist exakt, und es kostet nur zwei
        Projektionen je Kandidat - kein dichteres Abtasten, kein groesserer
        Baum.

        Rueckgabe (d_px, kontur, bogen_px) - Bogenlage entlang der Strecke
        interpoliert, damit auch sie stetig ist.
        """
        lp = self.linienpunkte
        lb = self.linien_bogen
        lk = self.linien_kontur
        n_l = len(lp)

        # ZWEI DURCHGAENGE, EINER JE NACHBARSTRECKE - und das ist die
        # schnellere Fassung, nicht die naive.
        #
        # Am 2026-08-23 wurde versucht, beide Versaetze in EINEN Durchgang
        # mit einer zusaetzlichen Achse zu ziehen (Punkt 1.4 der
        # Leistungsliste). GEMESSEN WAR DAS LANGSAMER: 0.826 s statt 0.690 s
        # fuer als_raster() bei 384 px. Grund ist der Speicherverkehr - die
        # Zwischenfelder haetten Form (N, K, 2, 2) statt (N, K, 2), bei
        # N = 145 000 und K = 24 also 111 MB statt zweimal 55 MB
        # nacheinander, und der zweite Durchgang laeuft ueber bereits warmen
        # Cache. Der Vorschlag ist damit widerlegt und bleibt bewusst
        # unumgesetzt.
        bester_d = np.full(idx.shape, np.inf)
        bester_b = np.zeros(idx.shape)
        for versatz in (-1, 1):
            nachbar = np.clip(idx + versatz, 0, n_l - 1)
            # Nur innerhalb derselben Kontur verbinden - sonst entstuende
            # eine Strecke quer ueber die Karte zwischen zwei Inseln.
            gleich = lk[nachbar] == lk[idx]
            a = lp[idx]
            b = np.where(gleich[..., None], lp[nachbar], lp[idx])
            ab = b - a
            l2 = np.einsum("...i,...i->...", ab, ab)
            l2 = np.where(l2 > 1e-12, l2, 1.0)
            ap = punkte[:, None, :] - a
            t = np.clip(np.einsum("...i,...i->...", ap, ab) / l2, 0.0, 1.0)
            fuss = a + t[..., None] * ab
            d = np.hypot(*(punkte[:, None, :] - fuss).transpose(2, 0, 1))
            bogen = lb[idx] + t * (np.where(gleich, lb[nachbar], lb[idx])
                                   - lb[idx])
            nimm = d < bester_d
            bester_d = np.where(nimm, d, bester_d)
            bester_b = np.where(nimm, bogen, bester_b)
        return bester_d, lk[idx], bester_b

    def _kuesten_waehlen(self, punkte):
        """
        Je Abfragepunkt bis zu K_KUESTEN VERSCHIEDENE Kuestenabschnitte.

        Die k naechsten Konturpunkte liegen fast immer alle auf demselben
        Kuestenstueck - naehme man sie unbesehen, mischte man einen Abschnitt
        mit sich selbst und die gegenueberliegende Seite einer Landzunge kaeme
        nie vor. Deshalb wird aus einer Ueberzahl von Kandidaten gierig
        ausgewaehlt, mit einer Trennbedingung ENTLANG DER KUESTE (siehe
        MIN_BOGEN_TRENNUNG_M).

        Rueckgabe (d_m, kontur, bogen, anzahl) - je (N, K_KUESTEN), nicht
        belegte Plaetze haben d_m = inf.
        """
        # K_KANDIDATEN NICHT SENKEN - gemessen am 2026-08-23, 384 px.
        #
        # Punkt 1.3 der Leistungsliste schlug vor, statt 24 zunaechst nur 12
        # Nachbarn zu pruefen ("die Trennbedingung schlaegt selten zu"). Die
        # Messung sagt das Gegenteil: bis drei VERSCHIEDENE Abschnitte
        # beisammen sind, braucht der MEDIAN-Punkt 18 Kandidaten, das
        # 90. Perzentil 23, und bei 6.4 % der Punkte werden die drei Plaetze
        # auch mit allen 24 nie voll.
        #
        # Der Grund steht in MIN_BOGEN_TRENNUNG_M (400 m) und
        # LINIEN_ABSTAND_M (60 m): die ersten rund sieben Nachbarn liegen
        # zwangslaeufig auf demselben Kuestenstueck innerhalb der
        # Trennschwelle und werden alle verworfen. 24 ist damit knapp
        # bemessen, nicht grosszuegig - eine Senkung wuerde Ergebnisse
        # aendern, nicht nur Zeit sparen.
        n = len(punkte)
        k = int(min(K_KANDIDATEN, len(self.linienpunkte)))
        abst, idx = self.linien_baum.query(punkte, k=k)
        if k == 1:
            abst = abst[:, None]
            idx = idx[:, None]

        # NUR DIE EIGENE INSEL (Nutzervorgabe 2026-08-24, Punkt b).
        #
        # Der KD-Baum kennt nur Abstaende. Zwei Inseln 200 m auseinander
        # liegen beide in der 350-m-Zone der jeweils anderen; ohne diese
        # Pruefung formt die Steilkueste der einen das Ufer der anderen.
        #
        # Ein Abfragepunkt auf SEE (Inselnummer 0) wird nicht gefiltert -
        # dort wirkt die Landformung ohnehin nicht (`ist_land` in
        # _hoehe_block), und der Ozeanuebergang hat seine eigene Regel.
        # Ohne diese Ausnahme verloere der Meerteil jede Zuordnung.
        if getattr(self, "linien_insel", None) is not None and len(self.linien_insel):
            xi = np.clip(np.round(punkte[:, 0]).astype(np.int32), 0, self.size - 1)
            yi = np.clip(np.round(punkte[:, 1]).astype(np.int32), 0, self.size - 1)
            eigene = self.insel_karte[yi, xi]
            fremd = ((self.linien_insel[idx] != eigene[:, None])
                     & (eigene[:, None] > 0)
                     & (self.linien_insel[idx] > 0))
        else:
            fremd = None
        # EXAKTER Abstand zur Linie statt zu den Abtastpunkten - siehe
        # _auf_strecken(). Ohne diesen Schritt liegt laengs der Kueste eine
        # Perlenkette aus Voronoi-Zellen der Abtastpunkte.
        d_px, kand_kontur, bogen_px = self._auf_strecken(punkte, idx)
        kand_d = d_px * self.mpp
        kand_bogen = bogen_px * self.mpp

        # Fremde Inseln JETZT ausschliessen, nicht vorher: `abst` aus der
        # KD-Baum-Abfrage wird gar nicht weiterverwendet - massgeblich ist
        # der exakte Streckenabstand aus _auf_strecken(). Ein unendlicher
        # Abstand faellt in der Auswahlschleife unten von selbst heraus.
        if fremd is not None and fremd.any():
            kand_d = np.where(fremd, np.inf, kand_d)

        gew_d = np.full((n, K_KUESTEN), np.inf)
        gew_kontur = np.zeros((n, K_KUESTEN), dtype=np.int32)
        gew_bogen = np.zeros((n, K_KUESTEN), dtype=np.float64)
        anzahl = np.zeros(n, dtype=np.int32)

        for j in range(k):
            offen = anzahl < K_KUESTEN
            if not offen.any():
                break
            nehmbar = offen.copy()
            for platz in range(K_KUESTEN):
                belegt = platz < anzahl
                gleiche = gew_kontur[:, platz] == kand_kontur[:, j]
                nah = (np.abs(gew_bogen[:, platz] - kand_bogen[:, j])
                       < MIN_BOGEN_TRENNUNG_M)
                nehmbar &= ~(belegt & gleiche & nah)
            wo = np.flatnonzero(nehmbar)
            if not len(wo):
                continue
            platz = anzahl[wo]
            gew_d[wo, platz] = kand_d[wo, j]
            gew_kontur[wo, platz] = kand_kontur[wo, j]
            gew_bogen[wo, platz] = kand_bogen[wo, j]
            anzahl[wo] += 1

        return gew_d, gew_kontur, gew_bogen / self.mpp, anzahl

    def _max_reichweite_px(self):
        """
        Ab welchem Abstand zur Kuestenlinie kann die Kueste NICHTS mehr tun.

        Das Blendgewicht in _hoehe_block() ist
            w = 1 - smoothstep((d - p1) / (p2 - p1))
        und smoothstep saettigt bei 1. Fuer d >= p2 ist w EXAKT 0 - nicht
        klein, sondern null. Ein Punkt jenseits der groessten p2 aller
        Segmente faellt damit auf `basis` zurueck.

        Das ist keine Naeherung, sondern eine Rechnung, die man sich sparen
        kann: gemessen liegen bei 1024 px rund 60 % aller Pixel jenseits
        dieser Grenze, und fuer sie lief bisher die volle Auswertung mit
        24 KD-Baum-Nachbarn, gieriger Auswahl und Profilinterpolation - um
        am Ende die Eingabe zurueckzugeben.

        SEIT DEM ZONENUMBAU AUS p2, NICHT AUS `reichweite_m` (2026-08-24).
        Die alte Fassung nahm `seg["reichweite_m"]`, also die Groesse aus
        dem frueheren Modell. Solange die Zonen bei 500 m lagen und die
        Reichweiten bei 110-349 m, schnitt die Maske die Zone AB - der
        aeussere Teil des Uebergangs fiel weg. Mit dem Skerrheim-Faktor
        (p2 = 823 m) waere die Abweichung noch groesser geworden.

        Der Zuschlag LINIEN_ABSTAND_M deckt den Unterschied zwischen dem
        Abstand zum naechsten ABTASTPUNKT (was der KD-Baum liefert) und dem
        Abstand zur STRECKE (was _auf_strecken rechnet, und was hoechstens
        um den halben Punktabstand kleiner sein kann).
        """
        if not self.segmente:
            return 0.0
        p2_max = max(float(seg.get("voll_m", PROFIL_VOLL_M))
                     + float(seg.get("uebergang_m", UEBERGANG_LAND_M))
                     for seg in self.segmente)
        # Die Bogenvarianz kann p2 zusaetzlich nach aussen schieben.
        return (p2_max + ZONEN_VARIANZ_M + LINIEN_ABSTAND_M) / self.mpp

    def _nahe_punkte(self, punkte):
        """
        Index der Punkte, bei denen die Kueste ueberhaupt wirken kann.

        Eine k=1-Abfrage statt der vollen k=K_KANDIDATEN-Abfrage: hier zaehlt
        nur, OB ein Punkt in Reichweite liegt, nicht welche Abschnitte ihn
        formen.
        """
        grenze = self._max_reichweite_px()
        if grenze <= 0.0 or self.linien_baum is None:
            return np.zeros(0, dtype=np.int64), 0.0
        d_punkt, _idx = self.linien_baum.query(punkte, k=1)
        return np.flatnonzero(d_punkt <= grenze), grenze

    # Wieviele Abfragepunkte hoechstens in EINEM Rutsch. Siehe hoehe().
    KACHEL_PUNKTE = 65536

    def hoehe(self, x, y, mindest_skala_m):
        """
        DIE Hoehenfunktion - Kachelschleife um `_hoehe_block()`.

        WARUM GEKACHELT (Punkt 1.5 der Leistungsliste). Die Rechnung legt je
        Abfragepunkt Felder der Form (N, K_KANDIDATEN) an, und das an einem
        guten Dutzend Stellen. Bei 1024 px und N = 1 Mio sind das rund
        400 MB Zwischenfelder in einem Rutsch.

        GEMESSEN am 2026-08-23: `als_raster()` schwankte dadurch zwischen
        5.6 s und 16.4 s bei identischer Eingabe, je nachdem wie der
        Arbeitsspeicher gerade aussah - eine Laufzeit, die vom Zufall
        abhaengt, ist weder messbar noch verlaesslich. Eine feste
        Kachelgroesse haelt die Zwischenfelder unter 25 MB und damit im
        Cache.

        BITGLEICH, weil die Rechnung PUNKTWEISE ist: jeder Abfragepunkt
        bestimmt seine Kuestenabschnitte allein aus dem KD-Baum, kein Punkt
        liest einen anderen. Die Kachelgrenzen sind daher keine Naehte -
        `tests/smoke_test_vektor_kueste.py` prueft es ueber die
        Aufloesungsreihe mit.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = x.size
        self._wahl_teil = None
        if n <= self.KACHEL_PUNKTE:
            ergebnis = self._hoehe_block(x, y, mindest_skala_m)
            self._wahl_cache_setzen(n, [(0, self._wahl_teil)])
            return ergebnis

        form = x.shape
        xf, yf = x.ravel(), y.ravel()
        aus = np.empty(n, dtype=np.float64)
        teile = []
        for a in range(0, n, self.KACHEL_PUNKTE):
            b = min(a + self.KACHEL_PUNKTE, n)
            self._wahl_teil = None
            aus[a:b] = self._hoehe_block(xf[a:b], yf[a:b], mindest_skala_m)
            teile.append((a, self._wahl_teil))
        self._wahl_cache_setzen(n, teile)
        return aus.reshape(form)

    def _wahl_cache_setzen(self, n_alle, teile):
        """
        Die Kuestenauswahl aller Kacheln zu EINEM Cache zusammensetzen.

        archetyp_felder() liest ihn und spart sich damit einen zweiten
        vollstaendigen _kuesten_waehlen()-Durchlauf (Punkt 1.1). Die
        `nah`-Indizes sind je Kachel LOKAL und werden hier um den
        Kachelanfang versetzt, damit sie wieder auf die ganze Punktmenge
        zeigen.
        """
        belegt = [(a, t) for a, t in teile if t is not None and len(t["nah"])]
        if not belegt:
            self._wahl_cache = None
            return
        self._wahl_cache = {
            "n_alle": n_alle,
            "nah": np.concatenate([t["nah"] + a for a, t in belegt]),
            "d": np.concatenate([t["d"] for _a, t in belegt]),
            "kontur": np.concatenate([t["kontur"] for _a, t in belegt]),
            "bogen": np.concatenate([t["bogen"] for _a, t in belegt]),
        }

    def _hoehe_block(self, x, y, mindest_skala_m):
        """
        Ein Block Abfragepunkte. Beliebige Float-Koordinaten in PIXELeinheiten.

        `mindest_skala_m` ist der einzige Parameter, in dem sich Raster- und
        Punktabtaster unterscheiden duerfen - siehe Modulkopf.

        Rueckgabe: Hoehe in Metern, gleiche Form wie `x`.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        basis = self.basis_hoehe(x, y)
        if self.saat_baum is None or self.linien_baum is None:
            return basis

        punkte_alle = np.stack([x.ravel(), y.ravel()], axis=1)
        ist_land = basis > 0.0

        # NUR DIE PUNKTE IN REICHWEITE RECHNEN (Punkt 1.2 der
        # Leistungsliste). Alles Weitere in dieser Methode laeuft auf
        # `punkte`, also auf der Teilmenge; ganz am Ende werden `profil` und
        # `staerke` auf die volle Form zurueckgestreut, wo sie 0 bleiben -
        # und 0 ist genau der Wert, den die Rechnung dort ohnehin geliefert
        # haette (siehe _max_reichweite_px()).
        nah, _grenze = self._nahe_punkte(punkte_alle)
        n_alle = len(punkte_alle)
        if len(nah) == 0:
            return basis                    # _wahl_teil bleibt None
        punkte = punkte_alle[nah]

        # BIS ZU K_KUESTEN VERSCHIEDENE ABSCHNITTE, nicht nur der naechste.
        #
        # Nutzer-Vorgabe 2026-08-19: *"wie kann man machen das jeder ort in
        # der summe nur jeweils den groessten kuesteneinfluss einer kueste
        # abbekommt. also zB aus Kueste A 40%, Kueste B 20%, Kueste C 20%.
        # dann muesste daraus werden Kueste A 20% Kueste B und C jeweils 10
        # insgesamt 40%"*.
        #
        # Das zerfaellt in ZWEI getrennte Normierungen, und dass sie getrennt
        # sind, ist der Kern:
        #
        #   FORM    was die Kueste hier will  -> gewichteter MITTELWERT
        #           h_kueste = Sum(w_i h_i) / Sum(w_i)
        #   STAERKE wieviel sie ueberhaupt zaehlt -> das MAXIMUM
        #           staerke = max(w_i)
        #
        # Das Maximum statt der Summe ist die eigentliche Zusicherung: eine
        # Stelle kann NIE mehr Kuesteneinfluss bekommen, nur weil dort mehr
        # Kuesten in Reichweite liegen. Zwei Kuesten, die sich in einem Hals
        # treffen, ergeben zusammen nicht mehr als eine einzelne im selben
        # Abstand. Aufsummieren ist damit konstruktiv ausgeschlossen - nicht
        # durch nachtraegliches Klemmen, sondern weil die Formel es nicht
        # kann. (Nachgerechnet: das Zahlenbeispiel des Nutzers und diese
        # Formel liefern identisch 0.6*basis + 0.2 h_A + 0.1 h_B + 0.1 h_C.)
        #
        # Vorher gewann der naechste Abschnitt allein - daher die harten
        # Naehte auf jeder Medialachse (gemessen 292 m Hoehensprung quer
        # durch eine Landzunge).
        gew_d, gew_kontur, gew_bogen, anzahl = self._kuesten_waehlen(punkte)

        # ERGEBNIS DER AUSWAHL MERKEN. archetyp_felder() braucht exakt
        # dieselben drei Spalten (Platz 0) und hat sie bisher ein ZWEITES Mal
        # gerechnet - dieselbe KD-Baum-Abfrage, dieselbe gierige Auswahl,
        # noch einmal ueber alle Landpixel (Punkt 1.1 der Leistungsliste,
        # gemessen 7.9 s von 43 s). Der Cache haengt an der Punktmenge, damit
        # er nur greift, wenn wirklich dieselben Punkte gefragt sind.
        self._wahl_teil = {"nah": nah, "d": gew_d, "kontur": gew_kontur,
                           "bogen": gew_bogen}

        if self.schliessen:
            xi = np.clip(np.round(punkte[:, 0]).astype(np.int32), 0, self.size - 1)
            yi = np.clip(np.round(punkte[:, 1]).astype(np.int32), 0, self.size - 1)
            d_max_karte = self.d_max_karte[yi, xi]
            h_max_karte = self.h_max_karte[yi, xi]
        else:
            d_max_karte = None
            h_max_karte = None

        summe_gewicht = np.zeros(len(punkte))
        summe_profil = np.zeros(len(punkte))
        max_gewicht = np.zeros(len(punkte))

        for platz in range(K_KUESTEN):
            belegt = np.isfinite(gew_d[:, platz])
            if not belegt.any():
                continue
            d_i = np.where(belegt, gew_d[:, platz], 0.0)

            # ZWEI QUELLEN, BEWUSST GETRENNT:
            #   was vom TYP kommt (Winkel, Reichweite, Ueberhoehung) ->
            #     stueckweise konstant je Segment, mit Uebergangszonen
            #   was vom GELAENDE kommt (Hinterlandhoehe) ->
            #     ueberall glatt entlang der Kueste interpoliert
            # Wuerde man beides gleich behandeln, waere entweder der Typ
            # ueberall gemischt (Nutzereinwand) oder die Gelaendehoehe
            # spraenge an jeder Segmentgrenze.
            (tanw_i, reich_i, ueber_i, katalog_i, form_i,
             profil_m_i, voll_i, ueberg_i) = self._segment_felder(
                gew_kontur[:, platz], gew_bogen[:, platz])
            tanw_i = np.maximum(tanw_i, 0.05)

            d_land_i = np.maximum(d_i, 0.0)

            # ZUR INSELMITTE SCHLIESSEN - bleibt, trotz "nicht stauchen".
            #
            # d_eff = d_max * (t - t^2/2) mit t = d/d_max. f(0)=0 und
            # f'(0)=1: an der Wasserlinie aendert sich nichts. f'(1)=0: auf
            # der Medialachse verschwindet die Steigung, es entsteht eine
            # Kuppe statt einer Spitze.
            #
            # Das IST eine Stauchung und widerspricht scheinbar der Vorgabe
            # "nicht gestreckt oder gestaucht". Auf dem Festland wirkt sie
            # praktisch nicht (d_max gemessen 2750 m, die Profilzone reicht
            # 500 m, also t < 0.18). Auf einer 150-m-Insel dagegen ist sie
            # die einzige Alternative zu einer Spitze in der Inselmitte -
            # dort treffen sich die Profile aller Uferseiten.
            if d_max_karte is not None:
                t = np.clip(d_land_i / np.maximum(d_max_karte, 1e-6), 0.0, 1.0)
                d_land_i = d_max_karte * (t - 0.5 * t * t)

            # DAS GEMESSENE PROFIL IN METERN AUSWERTEN.
            #
            # `profil_m_i` ist h(x) an PROFIL_STELLEN_M (19 Stellen alle
            # 50 m), je Abfragepunkt schon zwischen den Segmenten gemischt.
            # Der Abstand geht DIREKT hinein - keine Normierung auf eine
            # Reichweite, keine Ruecktransformation. Damit behaelt jeder
            # Archetyp sein gemessenes Hoehen-zu-Tiefen-Verhaeltnis
            # (Nutzervorgabe 2026-08-24).
            #
            # Was damit entfaellt: `ziel_i` aus Hinterland mal Ueberhoehung,
            # der Katalogboden und die Streckung auf `reich_i`. Die Hoehe
            # steht im Profil.
            if profil_m_i is not None:
                schritt_m = (PROFIL_STELLEN_M[1] - PROFIL_STELLEN_M[0]
                             if len(PROFIL_STELLEN_M) > 1 else 50.0)
                lage_p = np.clip(d_land_i / schritt_m, 0.0,
                                 profil_m_i.shape[1] - 1.001)
                u_p = np.floor(lage_p).astype(np.int32)
                br_p = lage_p - u_p
                zeilen = np.arange(len(d_land_i))
                profil_i = (profil_m_i[zeilen, u_p] * (1.0 - br_p)
                            + profil_m_i[zeilen, u_p + 1] * br_p)
            else:
                # RUECKFALL mit lauter Kennung - ein Archetyp ohne
                # gemessenes Profil darf nicht stillschweigend eine
                # Ersatzkurve bekommen (CLAUDE.md).
                if not getattr(self, "_profil_fehlt_gemeldet", False):
                    _LOGGER.warning(
                        "Kuestenabschnitt ohne gemessenes Meterprofil - "
                        "Ersatzkurve aus Zielhoehe und Winkel. Pruefen: "
                        "MESS_PROFIL_M_JE_ARCHETYP in core/vektor_kueste.py")
                    self._profil_fehlt_gemeldet = True
                ziel_i = np.maximum(np.maximum(hinter_i, 0.0) * ueber_i
                                    + KUESTEN_SOCKEL_M,
                                    MIN_ANTEIL_KATALOG * katalog_i)
                skala_i = np.maximum(mindest_skala_m, ziel_i / tanw_i)
                profil_i = ziel_i * (1.0 - np.exp(-d_land_i / skala_i))

            # HOEHENDECKEL AUS DER LANDMASSE. Eine 150-m-Insel kann keine
            # Fjordwand von 466 m tragen - das gemessene Profil kennt die
            # Inselgroesse nicht. Deckel ist die Breite mal
            # ZIEL_JE_HINTERLAND oder die dort vorhandene Hoehe, der
            # groessere gewinnt (siehe _tiefenfeld_bauen).
            if d_max_karte is not None:
                profil_i = np.minimum(profil_i, np.maximum(
                    ZIEL_JE_HINTERLAND * d_max_karte, h_max_karte))

            # BLENDSTAERKE AUS DEN FESTEN ZONEN (Nutzervorgabe woertlich):
            #   0 .. voll_i                    100 % Kuestenprofil
            #   voll_i .. voll_i + ueberg_i    Smoothstep ins Rauschgelaende
            #   darueber                       100 % Rauschgelaende
            #
            # `voll_i` und `ueberg_i` kommen aus der Segmenttabelle, sind
            # also je Kuestenabschnitt verschieden setzbar - die Grundlage
            # dafuer, dass die Strahltiefe spaeter entlang der Kueste
            # streuen kann (Nutzerpunkt c).
            #
            # AUF KLEINEN INSELN GEDECKELT: reicht die Zone weiter als die
            # halbe Insel, laeuft das Gewicht frueher aus, sonst laesst die
            # Kueste das Basisgelaende nie durch (gemessen an der
            # 150-m-Insel: 109 m statt 178 m).
            zone_i = voll_i
            if d_max_karte is not None:
                zone_i = np.minimum(voll_i, 0.9 * d_max_karte)
            auslauf = self._smoothstep(
                (d_i - zone_i) / np.maximum(ueberg_i, 1e-9))
            w_i = np.where(belegt, np.clip(1.0 - auslauf, 0.0, 1.0), 0.0)

            # ZWEI GETRENNTE GEWICHTE, und dass sie getrennt sind, ist der
            # Kern (Nutzervorgabe: *"einfluesse teilen sich gewichtet auf,
            # gesamteinfluss entspricht dem hoechsten der einflusse"*):
            #
            #   w_i        WIEVIEL Kuesteneinfluss ueberhaupt -> MAXIMUM
            #   misch_i    WELCHE Kueste davon                -> MITTELWERT
            #
            # WARUM misch_i NICHT w_i SEIN DARF. Innerhalb der vollen Zone
            # ist w_i fuer JEDEN Abschnitt exakt 1 - das ist der Sinn des
            # Plateaus. Mittelt man damit, zaehlt ein Kuestenabschnitt am
            # anderen Ende der 350-m-Zone genauso stark wie der direkt
            # daneben. GEMESSEN 2026-08-24: die Fjordwand kam dadurch auf
            # 28 m statt 120 m nach 150 m, weil sie mit der flachen
            # Schaerenkueste nebenan gleichberechtigt gemittelt wurde;
            # dieselbe Ursache bei Cinque Terre (30 statt 99 m) und Amalfi
            # (44 statt 80 m). Die flachen Archetypen fielen nicht auf, weil
            # ihre Nachbarn ebenfalls flach sind.
            #
            # Frueher trat das nicht auf, weil die Reichweiten 110-349 m
            # betrugen und das Plateau bei 60 % endete - da fiel das Gewicht
            # ueber den groessten Teil der Zone ohnehin ab.
            #
            # `misch_i` faellt quadratisch mit dem Abstand und ist damit
            # genau das vom Nutzer beschriebene Aufteilen: der naechste
            # Abschnitt bekommt den groessten Anteil, entferntere weniger.
            # DER EXPONENT entscheidet, wie stark der naechste Abschnitt
            # dominiert. Quadratisch war zu schwach: das Skerrheim hat
            # 2106 Schaerenkuesten-Pixel (h(150 m) = 14 m) gegen 1299
            # Fjordwand-Pixel (Soll 120 m), und die Fjordwand wurde von
            # ihren flachen Nachbarn heruntergezogen - gemessen 41 m, wo
            # das Profil 270 m sagt. Siehe MISCH_EXPONENT.
            misch_i = w_i * np.power(
                np.clip(1.0 - d_i / np.maximum(zone_i + ueberg_i, 1e-9),
                        0.0, 1.0), MISCH_EXPONENT)

            summe_gewicht += misch_i
            summe_profil += misch_i * profil_i
            max_gewicht = np.maximum(max_gewicht, w_i)

        profil_nah = np.where(summe_gewicht > 1e-12,
                              summe_profil / np.maximum(summe_gewicht, 1e-12),
                              0.0)
        profil_flach = np.zeros(n_alle)
        staerke_flach = np.zeros(n_alle)
        profil_flach[nah] = profil_nah
        staerke_flach[nah] = max_gewicht
        profil = profil_flach.reshape(x.shape)
        staerke = staerke_flach.reshape(x.shape)

        # BANDWEISE EINBLENDUNG DES RAUSCHGELAENDES (Punkt (e) der
        # Nutzerliste): *"die oktaven 1 und 2 langsam sich hineinblenden.
        # 3 und 4 koennten eventuell auch frueher schon herauskommen."*
        #
        #     h = profil * s  +  Summe_k  band_k * (1 - s)^p_k
        #
        # Bei s = 1 (an der Wasserlinie) ist jeder Bandanteil 0, es zaehlt
        # nur das Profil - die Wasserlinie bleibt also exakt erhalten. Bei
        # s = 0 (jenseits der Reichweite) ist jeder Anteil 1 und die Summe
        # der Baender ist per Konstruktion wieder das Basisgelaende.
        #
        # Der Exponent p_k steuert, WANN ein Band kommt: gross heisst spaet
        # (die Grossform haelt sich zurueck), klein heisst frueh (die
        # Feinstruktur ist gleich an der Klippe da). Mit p_k = 1 fuer alle
        # Baender ergibt sich exakt die frueher benutzte lineare Ueberblendung.
        # DIE KUESTENLINIE DARF SICH NICHT VERSCHIEBEN.
        #
        # Das ist die aelteste Zusicherung dieses Moduls: die Formung
        # aendert die HOEHE, nicht die Lage der Wasserlinie - sonst
        # aendert sich der Wasseranteil jeder Region und die gesamte
        # Regionseichung (smoke_test_regionen_welt) waere hinfaellig.
        #
        # Im alten Pfad hielt `+ KUESTEN_SOCKEL_M` (12 m) jedes Landpixel
        # ueber Null. Die gemessenen Meterprofile beginnen dagegen bei
        # exakt 0 m ueber Uferhoehe - richtig fuer die Form, aber damit
        # kann ein Landpixel dicht an der Linie auf 0 oder darunter
        # gedrueckt werden. GEMESSEN 2026-08-24 nach dem Umbau: 0.24 %
        # (256 px) bis 0.35 % (512 px) der Pixel wechselten die Seite.
        #
        # Statt einen Sockel auf das PROFIL zu legen - der die gemessene
        # Form verfaelschen wuerde, gerade bei den flachen Archetypen, um
        # die es hier geht - wird das Vorzeichen am Ende erzwungen: was
        # vorher Land war, bleibt Land. Die Hoehe darf beliebig nah an
        # Null gehen, nur nicht darueber hinweg.
        untergrenze = np.where(ist_land, MINDEST_LANDHOEHE_M, -np.inf)

        if getattr(self, "baender", None) and BAND_EXPONENTEN:
            xi = np.clip(np.round(x).astype(np.int32), 0, self.size - 1)
            yi = np.clip(np.round(y).astype(np.int32), 0, self.size - 1)
            rest = np.zeros(x.shape, dtype=np.float64)
            gegen = np.clip(1.0 - staerke, 0.0, 1.0)

            for band, p in zip(self.baender, BAND_EXPONENTEN):
                rest += band[yi, xi] * np.power(gegen, p)
            return np.where(ist_land,
                            np.maximum(profil * staerke + rest, untergrenze),
                            basis)

        return np.where(ist_land,
                        np.maximum(basis * (1.0 - staerke)
                                   + profil * staerke, untergrenze),
                        basis)

    def _hoehe_alt_einzelkueste(self, x, y, mindest_skala_m, basis, punkte,
                                ist_land):
        """Die frühere Fassung mit nur der naechsten Kueste - Vergleichsstand."""

        # PROFILWERTE ENTLANG DER KONTUR INTERPOLIERT, nicht vom naechsten
        # Saatpunkt zugeteilt (Nutzer-Vorgabe 2026-08-17: *"das muss doch
        # nicht pro pixel sein, sondern kann entlang der kontur alle x m
        # passieren ... daraus kann doch ein vektorfeld erzeugt werden"*).
        #
        # WARUM DAS BESSER IST ALS DIE NAECHSTER-SAATPUNKT-ZUTEILUNG: die
        # Zuteilung ist eine Voronoi-Aufteilung, und an jeder Zellgrenze
        # springt der Archetyp hart um. Die heutige Rasterfassung
        # (`_kuesten_umformen`) muss diese Naht deshalb mit einem
        # Gauss-Filter verwischen - ein Nachbearbeitungsschritt, der nur
        # existiert, um einen Sprung zu verstecken, den es gar nicht geben
        # muesste. Mit der Interpolation entlang der Bogenlaenge entsteht der
        # Sprung erst gar nicht: zwischen zwei Stationen laufen Zielhoehe,
        # Winkel und Reichweite linear ineinander ueber.
        #
        # Der Weg dahin: naechster LINIENpunkt (nicht Saatpunkt) liefert
        # Kontur-Nummer und Bogenposition; dort wird zwischen den beiden
        # umliegenden Stationen DERSELBEN Kontur gemischt.
        _abst_l, naechster_l = self.linien_baum.query(punkte)
        kontur_q = self.linien_kontur[naechster_l]
        bogen_q = self.linien_bogen[naechster_l]

        ziel = self._entlang_interpolieren(kontur_q, bogen_q, self.saat_zielhoehe)
        tanw = self._entlang_interpolieren(kontur_q, bogen_q, self.saat_tanwinkel)
        reichweite = self._entlang_interpolieren(kontur_q, bogen_q,
                                                 self.saat_reichweite_m)
        ziel = ziel.reshape(x.shape)
        tanw = np.maximum(tanw.reshape(x.shape), 0.05)
        reichweite = reichweite.reshape(x.shape)

        skala = np.maximum(mindest_skala_m, ziel / tanw)

        # ZUR INSELMITTE SCHLIESSEN (docs/KUESTENMODELL.md §4).
        #
        # Nutzer-Vorgabe: *"dann muss die kueste in der tiefe auch nicht
        # gestaucht werden bei kleinen inseln, sondern eine glaettung zur
        # inselmitte hin erzwungen werden"*. Also NICHT die Kurve auf die
        # Inselgroesse stauchen (das gaebe eine Miniaturlandschaft), sondern
        # ihre STEIGUNG zur Medialachse hin auf null zwingen.
        #
        # Dafuer wird der Abstand durch d_eff = d_max * f(d/d_max) ersetzt mit
        #
        #     f(t) = t - t^2/2
        #
        # f(0)=0 und f'(0)=1: an der Wasserlinie aendert sich NICHTS, die
        # Klippe bleibt so steil wie bestellt. f'(1)=0: auf der Medialachse
        # verschwindet die Steigung, es entsteht eine Kuppe statt einer
        # Spitze. Monoton auf [0,1], weil f' = 1-t >= 0.
        #
        # ERSTER ANLAUF WAR FALSCH HERUM und ist am Inseltest aufgefallen:
        # `f(t) = t + t^2 - t^3` erfuellt dieselben Randbedingungen, liegt
        # aber UEBER der Diagonalen. Das Profil wurde dadurch weiter aussen
        # ausgewertet und die Inseln wurden noch hoeher (500-m-Insel 271 ->
        # 282 m) statt flacher. Die sattigende Form (f(1)=0.5, also halber
        # Weg) ist die richtige.
        #
        # Auf dem Festland tut das nichts: dort ist d_max gemessen 2750 m und
        # die Reichweite hoechstens 700 m, also t < 0.25 im ganzen Band.
        d_land = np.maximum(d_m, 0.0)
        if self.schliessen:
            xi = np.clip(np.round(x).astype(np.int32), 0, self.size - 1)
            yi = np.clip(np.round(y).astype(np.int32), 0, self.size - 1)
            d_max = self.d_max_karte[yi, xi]
            t = np.clip(d_land / np.maximum(d_max, 1e-6), 0.0, 1.0)
            d_land = d_max * (t - 0.5 * t * t)

            # ZIELHOEHE AN DAS VERFUEGBARE HINTERLAND KOPPELN
            # (docs/KUESTENMODELL.md §3, "Tiefenklasse").
            #
            # Das Schliessen allein behebt nur die SPITZE, nicht die absurde
            # HOEHE: eine 500-m-Insel bekam eine Moher-Klippe mit 630 m
            # Zielhoehe, obwohl ihr groesster Kuestenabstand 233 m betraegt.
            # Ein Archetyp kann nicht hoeher werden, als sein Hinterland
            # traegt. Faktor 1.0 ist bewusst grosszuegig - an echten Inseln
            # gemessen liegt das Verhaeltnis Gipfel zu Kuestenabstand
            # zwischen 0.08 (Aran) und 0.83 (Capri), der Deckel greift also
            # erst bei klar unmoeglichen Faellen.
            ziel = np.minimum(ziel, ZIEL_JE_HINTERLAND * d_max)
            skala = np.maximum(mindest_skala_m, ziel / tanw)

        profil = ziel * (1.0 - np.exp(-d_land / skala))

        # Blendstaerke: an der Linie voll, zum Bandrand hin auf 0. Quadratisch
        # wie in 3.10 ("kraeftig an der Wasserlinie, dann zuegig verblassend").
        staerke = np.clip(1.0 - d_m / np.maximum(reichweite, 1e-9), 0.0, 1.0) ** 2

        # NUR LAND FORMEN - die Meerestiefe kommt aus dem Seegrad-Prozess
        # (3.13, `_seetiefe_aus_archetyp`), nicht von hier.
        return np.where(ist_land, basis * (1.0 - staerke) + profil * staerke, basis)


# ====================================================================== #
# DIE ZWEI ABTASTER - beide rufen VektorKueste.hoehe()
# ====================================================================== #

def als_raster(vk):
    """
    Abtaster 1: Pixelmitten -> Heightmap B (gitterbunden).

    `mindest_skala_m = 2 * mpp` wie heute in `_kuesten_umformen()`: feiner
    kann ein Raster eine Klippe nicht darstellen.
    """
    gy, gx = np.mgrid[0:vk.size, 0:vk.size]
    return vk.hoehe(gx.astype(np.float64), gy.astype(np.float64),
                    mindest_skala_m=2.0 * vk.mpp)


def an_punkten(vk, x, y, mindest_skala_m=MESH_MINDEST_SKALA_M):
    """
    Abtaster 2: freie Punkte -> Mesh-Vertices (nicht rastergebunden).

    Kleinere Mindestskala als beim Raster: zwei Vertices duerfen beliebig
    nah beieinanderstehen, eine steile Wand ist also darstellbar.
    """
    return vk.hoehe(np.asarray(x, dtype=np.float64),
                    np.asarray(y, dtype=np.float64),
                    mindest_skala_m=mindest_skala_m)
