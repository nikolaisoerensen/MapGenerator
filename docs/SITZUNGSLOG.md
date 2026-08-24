# Sitzungslog

Fortlaufendes Protokoll der Arbeitssitzungen. **Neueste Sitzung oben.**
Zweck: bei einem Wechsel auf einen anderen Account ohne Rueckfragen
weiterarbeiten koennen.

Aufbau je Eintrag: was gemacht wurde, was gemessen wurde, was NICHT
funktioniert hat, was offen bleibt. Vermutungen sind als solche
gekennzeichnet; alles andere ist geprueft.

---

# 2026-08-24 (Teil 2) — Leistung, Kuestenprofile in Metern, Fluesse nach Wasser

**Ausgangslage:** HEAD `950a949`, uncommitted auf `main`. Kein Worktree.

## 1. Ein Wicklungsfehler zerschnitt das 3D-Gelaende in Streifen

Nutzerbefund mit Bild: *"warum habe ich kein normal aussehendes mesh mehr
... von unten ist es uebrigens nicht unterbrochen."*

Der letzte Halbsatz war der ganze Beweis. `gui/widgets/kuesten_schnitt.py`
wickelte die Dreiecke gegen den Uhrzeigersinn (signierte Flaeche +0.5), das
Gitter in `map_display_3d` im Uhrzeigersinn (-0.5). Mit
`glFrontFace(GL_CW)` + `glCullFace(GL_BACK)` wird dann exakt verkehrt herum
gecullt: alles zur Kamera hin verschwindet, sichtbar bleiben nur die
abgewandten Rueckhaenge.

**Warum der Test das nicht fand:** er prueft, dass die Wicklung EINHEITLICH
ist. Die richtige Frage war, ob sie zum Gitter PASST. Neue Gruppe
`wicklung_wie_gitter` in `tests/smoke_test_kuesten_schnitt.py`.

## 2. Leistung — vier Knoten sind 84 % der Ladezeit

Neu: `managers/teilschritte.py` (Teilzeiten INNERHALB eines Knotens, fuer
Log und Ladebalken) und `tools/pipeline_kritischer_pfad.py`.

**Parallelitaet zwischen Knoten bringt 7 %** — der Abhaengigkeitsgraph ist
praktisch eine Kette (25 Ebenen, breiteste 3 Knoten, mittlere Breite 1.5).
Alles muss INNERHALB der vier grossen Knoten passieren.

Umgesetzt, alle A/B im selben Prozess gemessen, alle Ergebnisse bitgleich
bzw. Punkt-fuer-Punkt identisch:

| Massnahme | Faktor |
|---|---:|
| A-Stern mit numba (`core/wegsuche_schnell.py`) | 14-16x |
| `archetyp_felder`: Doppelarbeit + Tabellen-Nachschlag | 23x |
| `taeler_eingraben`: Stuetzstellen vektorisiert | 7.7x |
| Erosionsfilter bandweise (sequenziell) | 3.0x |
| Nahmaske in `hoehe()` | 2.5x |
| `poisson_points`: numpy-Skalarzugriffe raus | 2.4x |

`settlement.pathfinding` 60 s -> 3.8 s (Median je Route 0.85 s -> 0.055 s).

**FUENF Vorschlaege wurden durch Messung WIDERLEGT** und sind im Code an
Ort und Stelle mit ihrem Messwert kommentiert, damit sie nicht wiederholt
werden:

* `K_KANDIDATEN` von 24 auf 12 senken — der MEDIAN-Punkt braucht 18.
* `_auf_strecken` in einen Durchgang ziehen — langsamer (0.826 gegen
  0.690 s), die Zwischenfelder waeren 111 MB statt zweimal 55 MB.
* Die Kandidatenschleife auf 22 Kerne verteilen — durch numba
  gegenstandslos.
* Den 5x5-Nachbarschaftstest in `poisson_points` vektorisieren — 1.7 bis
  3x LANGSAMER, es sind nur ein bis fuenf belegte Nachbarn und die
  Schleife bricht beim ersten Treffer ab.
* **Der parallele Erosionsfilter.** 5x schneller (9.9 -> 1.3 s), aber JEDE
  nachfolgende numpy-Rechnung im selben Prozess dauerhaft **2.4x
  langsamer** (Referenzlast 1.47 -> 3.55 s, ohne Erholung). `weltfluesse`
  stieg dadurch von 8.4 s ueber 17.9 s auf 31.7 s — der Knoten wurde als
  Ganzes langsamer, obwohl sein teuerster Teilschritt schneller war.
  Ursache ist die gleichzeitige Allokation grosser Felder aus 22 Threads.
  Die sequenzielle Fassung bringt 3.0x ohne die Nebenwirkung; der Gewinn
  kam aus der Cache-Lokalitaet, nicht aus der Parallelitaet.
  `tests/smoke_test_erosionsfilter_baender.py::keine_nachwirkung` faengt
  den Fall kuenftig — an der Zeit des Filters allein ist er NICHT zu sehen.

**MESSWARNUNG:** diese Maschine schwankt um Faktor 2-3. Dieselbe feste
Referenzlast (5x `np.fft.fft2` auf 2000x2000) mass zwischen 0.80 s und
1.91 s ohne erkennbare Fremdlast (CPU 18 %, 12.9 GB RAM frei, kein
Swapping). Derselbe unveraenderte Knoten `terrain.redistribution` mass an
einem Nachmittag 43, 48, 58 und 96 s. **Nur A/B im selben Prozess ist
belastbar.**

Details: `docs/PERFORMANCE_2026-08-23.md`.

## 3. Kuestenprofile: von normiert auf METER

Anlass: *"ein strandgebiet soll nicht so stark an noise angepasst werden
wie klippen ... flache kueste bleibt flach, hohe kueste ist
hoehenvariabel."* Die Weissmeer-Flachkueste stand bei **187 m**, ihre
Vorlage sagt 15-30 m.

**Die Ursache:** die Zielhoehe entstand als
`max(hinterland * ueberhoehung + SOCKEL, MIN_ANTEIL_KATALOG * katalog)`,
und der erste Term gewann fast immer. Bei 150 m Rohgelaende hinter der
Kueste sind das 150 * 1.06 + 12 = 171 m — genau der gemessene Wert. Das
rohe Rauschgelaende bestimmte die Kuestenhoehe praktisch allein.

**Zwei Sackgassen zuerst**, beide gemessen, beide verworfen:

1. Die Bandeinblendung daempfen. Wirkungslos, weil im Plateau ohnehin
   `staerke = 1` gilt und `rest` dort 0 ist.
2. Die Hinterlandkopplung an den KATALOGWERT binden. Die Straende wurden
   deutlich besser (Toskana 0.374 -> 0.031), aber die Foerdenkueste fiel
   von RMS 0.040 auf 0.307: der Katalogwert ist je REGION tabelliert und
   trennt Strand und Klippe nicht — Kola-STEILkueste 30 m,
   Kykladen-STRAND 69 m.

**Der Umbau** nach Nutzerspezifikation: Profil in METERN statt gestreckt,
feste Zonen (0-350 m volles Profil, 350-500 m Uebergang ins Rauschen,
0 bis -200 m ins Ozeanprofil), Einfluss nur innerhalb der eigenen Insel
(`ndimage.label`), Strahltiefe je Segment variierbar angelegt.

**Neue Datengrundlage:** 27 Archetypen, jeder aus SEINER eigenen
Vorbildkueste (`tools/archetyp_vorbilder.py`, 22 DEM-Kacheln neu
abgerufen). Vorher gab es 11 Strecken fuer 27 Archetypen, und die Schnitte
einer Region wurden nach Steilheit gedrittelt — das ergab drei Kurven mit
identischer FORM und nur verschiedener HOEHE (normiert 0.28/0.26/0.30 nach
50 m). Dazu ein Zirkelschluss: nach h(150 m) sortieren und dann h(150 m)
messen trennt zwangslaeufig nur in dieser Groesse.

Vier Ausschnitte mussten nachgebessert werden; bei zweien wurden je vier
Kandidaten gemessen, bevor einer eingetragen wurde:
Isonzo-Muendung (0 m ueber 900 m, reines Schwemmdelta) -> Roja bei
Ventimiglia; Rybachi (Kachel zu 100 % an Land) -> Teriberka Ostkap;
Ponta da Piedade (flacher als die Nachbarbucht) -> Cabo de Sao Vicente;
La Concha (Berge im Ruecken) -> Donana.

**Ergebnis:** flache Kuesten treffen ihre Vorlage auf 0-2 m, Median ueber
20 Gruppen 2.6 m. Weissmeer-Flachkueste 187 m -> 8 m (Soll 5 m).

**Drei Fehler beim Umbau, alle gemessen und behoben:**

1. **Die Mittelung war zu breit.** Im Plateau ist das Blendgewicht fuer
   JEDEN Abschnitt exakt 1 — bei 350 m Zone wurde eine Fjordwand
   gleichberechtigt mit der flachen Schaerenkueste nebenan gemittelt.
   Jetzt zwei getrennte Gewichte: `max` fuer WIEVIEL Kuesteneinfluss, ein
   quadratisch fallendes fuer WELCHE Kueste. Median 8.2 -> 3.4 m.
2. **Die Kuestenlinie verschob sich** um 0.24-0.35 % der Pixel — die
   Meterprofile beginnen bei 0, der alte Sockel von 12 m fehlte. Statt
   einen Sockel aufs Profil zu legen (der die flachen Archetypen
   verfaelscht haette) wird das Vorzeichen erzwungen.
3. **Der Test mass das Falsche.** Er verglich normierte Formen — eine
   Kueste mit richtiger Form und voellig falscher Hoehe war gruen. Genau
   so blieb die Weissmeer-Flachkueste bei 187 m unentdeckt. Er misst jetzt
   Meter.

**Nicht loesbar mit diesen Daten:** ein gemessenes Unterwasserprofil.
COP30 ist ein Oberflaechenmodell und setzt offenes Meer auf 0; alle 27
Archetypen liegen zwischen -0.1 und -1.8 m und aendern sich ueber 250 m
nicht. Der Uferuebergang blendet deshalb von 0 auf die Seegrad-Tiefe —
vorher stand dort eine 10-m-Stufe direkt an der Wasserlinie.

**OFFEN UND NACH VIER GEPRUEFTEN HYPOTHESEN UNGEKLAERT:** die drei
steilsten Archetypen (Fjordwand 22 statt 120 m, Cinque Terre 24 statt 99,
Amalfi 42 statt 80).

**Der Kern des Raetsels:** `hoehe()` liefert AN DEN FJORDWAND-SAATPUNKTEN
richtige Werte. Gemessen ueber 14 Stationen und 102 Landrichtungen bei
d = 150 m: **Median 144 m** bei Soll 120 m - also sogar darueber. Der Test
misst an denselben Archetypen 22 m. Dieselbe Funktion, verschiedene
Ergebnisse.

Vier Hypothesen wurden gemessen und WIDERLEGT:

1. **Der Hoehendeckel greift.** Nein. `max(ZIEL_JE_HINTERLAND * d_max,
   h_max)` liegt im Median bei 2740 m, im Minimum bei 83 m.
2. **Die Kueste greift zu kurz.** Auf Nutzerwunsch wurde der
   Fjordland-Griff um 50 % vertieft (`KUESTEN_TIEFE_JE_REGION`, p1/p2 von
   350/500 auf 525/750). Die Fjordwand blieb bei 23 m.
3. **Die Mischung mit flachen Nachbarn zieht herunter.** Plausibel - das
   Fjordland hat 2106 Schaerenkuesten-Pixel (h(150 m) = 14 m) gegen 1299
   Fjordwand-Pixel. Der Mischexponent wurde von 2 auf 4 und 6 erhoeht:
   der MEDIAN ueber alle Gruppen verbesserte sich von 2.2 auf 1.8 m, die
   Fjordwand blieb bei 22 m.
4. **Die Basis sinkt landeinwaerts.** Trifft an einzelnen Stationen zu
   (105 m bei 50 m, 59 m bei 150 m), erklaert aber nicht, warum die
   Hoehenfunktion an denselben Stellen 144 m liefert.

**GEFUNDEN 2026-08-24 (spaeter am Tag): `archetyp_felder()` rechnete die
Staerke mit der ALTEN Reichweite.** Dort stand
`clip(1 - d / reichweite_m)^2`; `reichweite_m` ist die Groesse aus dem
frueheren Modell (110-349 m je Region), waehrend `_hoehe_block()` seit dem
Zonenumbau mit `voll_m`/`uebergang_m` rechnet (350/500 m, im Fjordland
525/750 m). Bei einem Pixel 392 m vor der Kueste wurde
`1 - 392/296` negativ und auf 0 geklemmt.

Gemessen VOR dem Fix: Fjordwand-Pixel `staerke` 0.07, Schaerenkueste 0.00.
NACH dem Fix: beide 1.00, und 52 % der Landpixel haben ueberhaupt eine
Staerke.

**Die Hoehenfunktion war davon NICHT betroffen** - sie rechnet
eigenstaendig. Betroffen war alles, was `kuesten_staerke` LIEST:
`_seetiefe_aus_archetyp()` fuer die Meerestiefe (5 Fundstellen) und die
2D-Anzeige im Kuestentypen-Modus (3 Fundstellen).

**Der Fjordwand-Befund relativiert sich damit.** An den Pixeln, die
tatsaechlich als Fjordwand markiert sind, liefert die Hoehenfunktion 218 m
bei einem Sollwert von 392 m - nicht die 22 m, die
`smoke_test_kuestenprofiltreue` meldet. Der Test misst entlang der
Kuestennormalen und ordnet ueber `kuesten_archetyp` zu; mit der falschen
Staerke griff diese Zuordnung ins Leere.

**Ebenfalls gemessen:** ohne die Vektorkueste trifft das Fjordland seine
Vorbild-Hoehenverteilung fast exakt (h_p25 0.29 gegen 0.30 bei Geiranger),
mit ihr faellt sie auf 0.02. Die Kuestenlinie selbst ist in beiden Faellen
identisch - es ist allein die Hoehenverteilung in Kuestennaehe.

### ES GAB KEINE GROSSEN FLUESSE — 2026-08-24

Beim Pruefen von Block B.2/B.3 (*"die grossen fluesse koennen als overlay
bei biome drin sein"*) gezaehlt: `water_biomes_map` enthielt
**ausschliesslich Creeks und Seen**. Die Stufen `river` (2) und
`grand_river` (3) kamen NIE vor - auf keiner Karte.

```
creek-Schwelle                        1716.77
river braucht  4x                     6867.08
grand  braucht 20x                   34335.41
groesster Abfluss der ganzen Karte     6408.09     <-- verfehlt beide
```

**URSACHE - zwei Masstaebe vermischt.** Die Creek-Schwelle ist ein
PERZENTIL der wasserfuehrenden Zellen und passt sich der Verteilung an.
Die Faktoren fuer River und Grand River waren dagegen ABSOLUTE Vielfache
davon. Zwischen einem hohen Perzentil und dem Maximum liegt in einer
Abflussverteilung aber systematisch weniger als eine Groessenordnung -
hier Faktor 3.73. Ein absoluter Faktor 20 darauf ist nicht streng,
sondern unerfuellbar.

**ABHILFE:** die Faktoren wirken jetzt auf die HAEUFIGKEIT. Ein River ist
der Lauf, der zu den obersten (abundance / 4) gehoert, ein Grand River zu
den obersten (abundance / 20). Das ist die Bedeutung, die die Namen immer
schon nahelegten - "vier mal so selten", nicht "vier mal so viel Wasser".

| | vorher | nachher |
|---|---:|---:|
| creek / river / grand_river | 1495 / **0** / **0** | 1119 / 299 / 77 |
| Anteil mindestens River | 0 % | 25.2 % (Soll 25.0) |
| Anteil Grand River | 0 % | 5.2 % (Soll 5.0) |

**WARUM ES NIEMAND GEMERKT HAT:** die Klassifikation lief fehlerfrei
durch und lieferte eine plausible Karte voller Baeche. Nichts stuerzte
ab, nichts warnte. Dasselbe Muster wie beim Archetyp-Verteilungsbug
weiter unten - ein Ergebnis, das von einem richtigen nicht zu
unterscheiden ist, solange niemand nachzaehlt.
`tests/smoke_test_flussstufen.py` zaehlt jetzt nach.

**NEBENBEI GEKLAERT: die Straende fehlen NICHT.** Es gibt zwei
Biomkarten mit unterschiedlicher Aufgabe:

  * `biome_map` - die grobe. Enthaelt die Wasserstufen, aber KEINE
    Wahrscheinlichkeits-Biome (beach, cliff, lake_edge, river_bank,
    snow/alpine_level). Alle sechs sind dort 0.
  * `biome_map_super` - die feine. Erst das Supersampling setzt die
    Wahrscheinlichkeiten in Pixel um; dort liegen 311 Strandpixel.

Wer Straende sehen will, muss also `biome_map_super` anzeigen. Das ist
Bauart, kein Fehler - und die Vermutung in ANZEIGE_UND_SEEN.md, es sei
ein reines Anzeigeproblem, war fuer die STRAENDE richtig und fuer die
GROSSEN FLUESSE falsch.

### FLAECHENEICHUNG: alle neun Regionen auf ihren Zielwert — 2026-08-24

Nutzerentscheidung nach dem Verteilungsfix: *"lass uns zielwert 0.8 fuer
fjordland festlegen, aber dann muessen wir noch etwas mehr flaeche
bekommen ... insgesamt bekommt halt jede region einen bestimmten
zielwert der erreicht werden soll."*

`ZIELWERT["Fjordland"] = 0.80` (wie das Alpenland - Fjordland verliert
DOPPELT, erst ein Drittel ans Wasser, dann die Haelfte des Rests an zu
steile Haenge).

Der Ausgleich laeuft ueber `flaeche_soll`. Der Parameter gab es schon,
aber nur drei Regionen hatten ihn gesetzt. Von Hand ist er kaum
einzustellen: **es ist ein Nullsummenspiel** - der Index misst gegen den
MEDIAN, also druckt jede wachsende Region alle anderen nach unten, und
jede Runde kostet eine volle Weltberechnung.

Deshalb `tools/flaeche_eichen.py`: ein logarithmischer Regelkreis, aus
demselben Grund logarithmisch wie die Eichung in `voronoi_regionen()` -
ein linearer Schritt schwingt, weil der Vorteil multiplikativ wirkt.

```
Runde 0: groesste Abweichung 0.261, mittlere 0.127
Runde 1: groesste Abweichung 0.204, mittlere 0.075
Runde 2: groesste Abweichung 0.102, mittlere 0.038  -> fertig
```

| | vorher | nachher |
|---|---:|---:|
| `smoke_test_regionen_fairness` | **1/2 Gruppen** | **2/2 Gruppen** |
| Spanne beste/schlechteste Region | 2.10x | **1.25x** |
| Fjordland Index (Ziel 0.80) | 0.64 | **0.80** |
| Fjordland Flaeche / nutzbar | 9652 / 33 % | 10920 / 50 % |

Die eingeregelten Werte stehen jetzt fest in `REGIONEN`; die ZIELWERTE
bleiben bewusst in `tests/smoke_test_regionen_fairness.py`, weil sie eine
Entscheidung ueber das Zielbild sind und keine ueber die Rechnung.

### URSACHE GEFUNDEN UND BEHOBEN — 2026-08-24

Die Fjordwand erreichte 23 m, wo ihr Profil 128 m verlangt. Nach dem
Ausschluss von Hoehendeckel, Reichweite, Mischung, Inselschluss und
`staerke` blieb nur, `_hoehe_block()` an einem einzelnen Punkt
mitzurechnen. Das Ergebnis war eindeutig:

```
Station 117, Punkt 150 m landeinwaerts, Archetyp Fjordwand
   hoehe():  14.6 m
   Platz 0: d=149.2 m   Profil h(150) = 14.7 m      <-- Schaerenkueste!
   Platz 1: d=443.3 m   Profil h(150) = 14.7 m
   Platz 2: d=505.1 m   Profil h(150) = 14.7 m
```

Die Hoehenfunktion war die ganze Zeit RICHTIG - sie bekam nur das falsche
Profil. Die Station gilt als Fjordwand, aber jeder Kuestenabschnitt in
ihrer Naehe traegt das Schaerenkuesten-Profil.

**Die Zaehlung zeigte, wie gross das Problem wirklich war:**

| | vorher | nachher |
|---|---:|---:|
| Archetypen ohne ein einziges Segment | **8 von 27** | 0 von 27 |
| Fjordwand: Saatstationen → Kuestenanteil | 13 → **0.4 %** | 13 → 5.2 % |
| Algarve-Klippen: Saatstationen → Anteil | 8 → **15.5 %** | 8 → 1.6 % |
| schlimmste Verzerrung | **18x** | 3.6x |
| Segmente / Medianlaenge | 115 / 169 m | 148 / 502 m |

Acht Archetypen - Fjordbucht, Foerdenkueste, Kotor-Steilfjord,
Labrador-Buchten, San-Sebastian-Bucht, Costa-Brava-Buchten,
Dalmatien-Klippen, Alpine-Flussmuendung - kamen auf der fertigen Karte
**ueberhaupt nicht vor**. Die Arbeit, alle 27 Profile aus echten
Vorbildern zu vermessen, war fuer knapp ein Drittel davon wirkungslos.

**URSACHE:** Die Zuweisung in `_saat_bauen()` sortierte die Stationen
einer Region nach Score und schnitt oben ab - ohne jeden Bezug darauf,
welche Station neben welcher liegt. `jitter` war weisses Rauschen je
Station. Ein Typ landete damit in Laeufen von ein bis zwei Stationen,
also 420-840 m. `MIN_SEGMENT_M` verlangt 750 m, und das Einschmelzen gibt
einen zu kurzen Lauf an den LAENGEREN Nachbarn weiter. Wer schon lang
war, wurde laenger - ein sich selbst verstaerkender Prozess, an dessen
Ende ein Archetyp mit 0.8 % der Stationen 15.5 % der Kueste hielt.

**ABHILFE** (`SAAT_KOHAERENZ_STATIONEN = 3.5`): Jitter und lokale Hoehe
werden entlang der Bogenlaenge geglaettet, Kontur fuer Kontur
(`_bogen_glaetten`, `_bogen_rauschen`). Benachbarte Stationen bekommen
dadurch aehnliche Scores und denselben Typ. Die ZIELANTEILE sind
unberuehrt - `ziel_anzahl` und die Auswahlschleife sind unveraendert, es
aendert sich nur, WELCHE Stationen ein Typ bekommt, nicht wieviele.

**DER WERT WURDE GEMESSEN, nicht geraten** (512 px, Seed 20260804):

| sigma | Ausfaelle ab 8 Stationen | Segment-Median |
|---:|---|---:|
| 2.5 | 2 (Fjordbucht, Moher-Klippen) | 505 m |
| **3.5** | **1 (Algarve-Klippen, 8 Stationen)** | 502 m |
| 5.0 | 2 (Algarve, Fjordbucht) | 397 m |

Nach oben wird es wieder schlechter: zu lange Laeufe lassen einem
seltenen Typ keine Luecke mehr, in die er passt.

**Ergebnis:** alle vier zuvor auffaelligen Archetypen treffen jetzt ihr
Profil - Fjordwand 129/128 m (vorher 23), Moher-Klippen 134/146 m (vorher
gar nicht vorhanden), Fjordbucht 44/35 m (vorher gar nicht vorhanden),
San-Sebastian-Bucht 15/16 m (vorher 5). `smoke_test_kuestenprofiltreue`
beide Gruppen gruen, Median 3.1 m, flache Kuesten 14 von 14.
`BEKANNTE_ABWEICHUNGEN` ist jetzt LEER - die San-Sebastian-Bucht (5
statt 16 m) hatte dieselbe Ursache; der dort notierte Verdacht
"Landmasse der Steppe" war falsch.

**Was dabei SCHLECHTER wurde, ehrlich notiert:**
  * `smoke_test_regionen_fairness`: Fjordland 0.72 → 0.64. Das ist
    inhaltlich richtig - es gibt jetzt echte Fjordwaende, und die sind
    steil (52 %). Der Kuestenbonus (1.08) gleicht das nicht aus. Eine
    KALIBRIERUNGSfrage, keine Fehlfunktion: der Zielwert 1.00 fuer eine
    Fjordlandschaft ist zu hoch, oder der Kuestenbonus zu schwach.
  * `smoke_test_regionen_welt` Naht 1.474 → 1.540 (Grenze 1.25). Der
    Test war vorher schon rot, mit denselben Befunden. Die Kuestenformen
    praegen jetzt staerker, und diese Metrik mischt Kuestenformen mit
    Regionsnaehten.

**Die Lehre, zum dritten Mal in diesem Projekt:** eine Kette aus lauter
einzeln richtigen Teilen kann als Ganzes falsch sein. Profil, Mischung,
Reichweite, Deckel und Staerke waren jeder fuer sich korrekt - der Fehler
sass in der Zuordnung DAZWISCHEN, und keine der Einzelpruefungen konnte
ihn sehen. Gefunden wurde er erst, als eine Messung Archetyp-Anteile
gegen Kuestenlaengen-Anteile hielt, also zwei Enden der Kette
gegeneinander. `tests/smoke_test_archetyp_verteilung.py` haelt das
jetzt fest.

### DER MESSAUFBAU WAR FALSCH — aufgeloest 2026-08-24

Alle Messungen dieser Runde, die `hoehe()` "richtige" Werte an den
Fjordwand-Stationen bescheinigten (144 m bei Soll 120), waren **an der
falschen Karte**. Sie bauten die `VektorKueste` so:

```python
vd.VEKTOR_KUESTE_AKTIV = False
H0, f = weltfeld(512, SEED)          # <-- laeuft den ALTEN Rasterpfad!
vk = VektorKueste(H0, ...)
```

Mit dem Schalter auf False laeuft `weltfeld()` durch
`_kuesten_umformen()` - den alten Pixelmasken-Pfad. Das Gelaende, auf dem
die Kueste dann gebaut wurde, ist ein anderes als im echten Lauf.

**Mit der `VektorKueste` aus dem ECHTEN `weltfeld()`** (sie liegt dort als
`felder["vektor_kueste"]`) liefert `hoehe()` bei d = 150 m nur **14 m**
statt der Soll-128 m. Der Test hatte die ganze Zeit recht; die
Fjordwand ist wirklich zu flach.

**Was daraus fuer kuenftige Messungen folgt:** immer
`felder["vektor_kueste"]` aus dem echten Lauf nehmen, nie eine selbst
gebaute. Dieselbe Lehre wie in smoke_test_kuesten_schnitt (dort steht sie
seit dem 2026-08-22 im Kopf) - sie hat sich hier unabhaengig wiederholt,
weil ich sie beim Messen nicht angewandt habe.

**Der Widerspruch ist damit aufgeloest, die URSACHE aber weiter offen.**
Bekannt ist jetzt: alle Einzelteile pruefen sich einzeln als richtig -
das Profil liefert bei d = 150 m die erwarteten 128 m, alle drei
Mischplaetze sind Fjordwand (keine Verduennung durch flache Nachbarn),
der Hoehendeckel greift nicht (d_max 2740 m), das Schliessen zur
Inselmitte kuerzt 150 m auf 146 m, und `staerke` ist nach dem Fix 1.00.
Trotzdem kommen 14 m heraus. Der naechste Schritt ist, `_hoehe_block()`
an einem einzelnen Fjordwand-Punkt Zeile fuer Zeile mitzurechnen, statt
weiter Hypothesen zu pruefen.

**Weiterhin offen** ist auch die Zuordnung in `archetyp_felder()`:
die Pixel, die dort als "Fjordwand" markiert sind, liegen im Median 335 m
von der Kueste (dort sagt das Profil rund 270 m) und haben H = 41 m. Die
Fjordbucht - der FLACHSTE Archetyp des Fjordlands - bekommt dagegen Pixel
mit H = 201 m. Entweder stimmt diese Zuordnung nicht mit der ueberein, die
`hoehe()` intern trifft, oder der Test misst ueber sie an den falschen
Stellen. **Das ist zu pruefen, bevor weiter am Modell gedreht wird.**

MISCH_EXPONENT = 4 wurde uebernommen (Median 2.2 -> 1.9 m), der
Fjordland-Tiefenfaktor ebenfalls (Nutzerwunsch, schadet nicht).

## 4. Fluesse folgen jetzt dem Wasser, nicht der Knotenzahl

Nutzerbefund: im Fjordland fehlen grosse Fluesse.

**Die Ursache stand in einer Zeile** (`terrain_weltfluesse.py:377`):
`flaeche = np.ones(n)`. Jeder Knoten trug 1 bei, egal ob dort 471 mm oder
1967 mm fallen. Was das Modell "Einzugsgebiet" nannte, war die FLAECHE in
Knoten, nicht die Wassermenge. Die Korrelation zwischen Wassermenge und
Flussgroesse war NEGATIV: das Fjordland hatte die meiste Wassermenge
(13.0 Mio) und den kleinsten Hauptfluss (123), die Steppe die wenigste
(4.3 Mio) und einen dreimal groesseren.

Umgesetzt: Niederschlag als Knotengewicht (Block 1.1) und die Regionsquote
nach Nutzervorgabe (Block 2, Fjordland 100 %, Taiga und Atlantik je 66 %).
Fjordland 123 -> 215 -> **700**.

**Drei Fehler beim Bau der Quote:** Doppelmultiplikation (2275 statt 700),
Zufluesse oberhalb mitskaliert (dort fliesst kein zusaetzliches Wasser),
multiplikativ statt additiv (blaehte die Atlantikkueste auf 2396 — mehr
Wasser im Oberlauf heisst flussabwaerts eine KONSTANTE Zugabe).

**Die Ursache des Fjordland-Rueckstands bleibt offen.** Drei Hypothesen
wurden gemessen und widerlegt — zerstueckelte Landmasse (Fjordland liegt
zu 100 % in einem Stueck), kurze Fliesswege (es hat mit 832 m den
ZWEITLAENGSTEN mittleren Kuestenabstand), zu wenig Buendelung (die Metrik
ist nicht schluessig). Sie stehen in `docs/FLUESSE_UND_WASSER.md`, damit
sie niemand erneut prueft.

**Block 3 (Seen als Sammler) ist fuer das Fjordland wirkungslos:** die
Karte hat 11 Binnenseen ueber 4 Pixel, und keiner liegt dort.

Ordnung und weitere Bloecke: `docs/FLUESSE_UND_WASSER.md`.

## Offen am Ende dieser Sitzung

* **VISUELL NICHT BESTAETIGT.** Die gesamte Kuesten- und Flussarbeit ist
  headless gemessen. Niemand hat sie im laufenden Programm gesehen. Das
  ist der Blocker.
* `smoke_test_regionen_welt`: Naht-Kennzahl 1.474 (Grenze 1.25).
  **Nachgemessen 2026-08-24, A/B im selben Prozess:**

  | | Naht | Inneres | Verhaeltnis | Befunde |
  |---|---:|---:|---|---:|
  | `VEKTOR_KUESTE_AKTIV=False` | 1.247 | 1.175 | 1.06 ok | 10 |
  | `VEKTOR_KUESTE_AKTIV=True` | 1.474 | **0.621** | 2.37 FEHLER | **7** |

  Bemerkenswert: MIT Vektorkueste hat der Test INSGESAMT WENIGER Befunde
  (7 gegen 10) - die Regionseichung ist als Ganzes besser geworden. Nur
  die Nahtpruefung kippt.

  **Die Ursache ist eingegrenzt, aber nicht abschliessend geklaert.** Die
  Naht selbst aendert sich kaum (1.247 -> 1.474); das INNERE faellt von
  1.175 auf 0.621. Gemessen wirkt die Vektorkueste exakt in ihrer Zone
  und nirgends sonst:

  | Abstand von der Kueste | ohne VK | mit VK | Faktor |
  |---|---:|---:|---:|
  | 0-100 m | 0.581 | 0.176 | 0.30 |
  | 100-250 m | 0.324 | 0.092 | 0.28 |
  | 250-500 m | 0.322 | 0.285 | 0.89 |
  | **ab 500 m** | 0.202 | 0.202 | **1.00 (bitgleich)** |

  Der Hang in der Kuestenzone faellt, weil die gemessenen Profile die
  Wahrheit sind: die meisten Archetypen SIND flach (Kykladen 1 m,
  Toskana 4 m, Dingle 2 m nach 150 m), nur wenige sind Klippen. Das
  "Innere" des Tests ist als `max(gewichte) > 0.85` definiert und
  schliesst Kuestennaehe nicht aus - der Test wurde gebaut, als die
  Kuestenzone 110-349 m breit war, jetzt sind es 500 m.

  **NACHGEMESSEN mit der ECHTEN Testmetrik** (p99.5 der Steigung auf dem
  Gauss-geglaetteten Feld, dieselben Masken wie im Test):

  | | Naht p99.5 | Innen p99.5 | Verhaeltnis |
  |---|---:|---:|---:|
  | ohne Vektorkueste | 1.273 | 1.020 | **1.25** |
  | mit Vektorkueste | 1.500 | 0.713 | 2.11 |

  **DER TEST STAND SCHON VORHER EXAKT AUF DER GRENZE.** 1.25 bei einer
  Grenze von 1.25 - null Reserve. Er war nicht "gruen", sondern "gerade
  eben noch gruen", und jede Geländeaenderung kippt ihn.

  Beide Seiten bewegen sich: die Naht steigt um 18 %, das Innere faellt
  um 30 %. Erklaerbar ist beides:

  * **Das Innere faellt**, weil die extremsten Landhaenge oft
    KUESTENhaenge sind. `innen_band` schliesst nur das Nahtband aus,
    nicht die Kueste. Die gemessenen Profile sind ueberwiegend flach
    (Kykladen 1 m, Toskana 4 m nach 150 m), also sinkt das p99.5.
  * **Die Naht steigt**, weil die Vektorkueste manche Klippen STEILER
    macht als das alte Modell (Moher: 145 m ueber 150 m). Laeuft eine
    Regionsgrenze durch eine Klippenregion, wird sie steiler gemessen.

  Ein Gegencheck nur ueber 500 m von der Kueste entfernt zeigt denselben
  Anstieg der Naht (1.060 -> 1.304). Das liegt an der Gauss-Glaettung
  (sigma 120 m, Einfluss bis rund 360 m) - sie zieht Werte aus der
  Kuestenzone in die Auswertung hinein.

  **NICHT BEHOBEN, und die Grenze wurde NICHT angehoben.** Eine
  Testgrenze zu lockern, weil das Ergebnis nicht passt, macht den Test
  wertlos. Der eigentliche Befund ist, dass die Metrik Kuestenformen und
  Regionsnaehte vermischt: sie misst den steilsten Hang im
  "Regionsinneren" und trifft damit oft eine Klippe. Wer das aufloest,
  sollte die Kuestenzone aus BEIDEN Masken nehmen und die Glaettung
  entsprechend kuerzen - dann misst der Test wirklich Naehte.
* `smoke_test_settlement_roads`: rot, aber VORBESTEHEND — `bau_kostenfeld`
  wurde auf eine exponentielle Hangformel umgestellt, der Test prueft noch
  die alte quadratische Erwartung.
* `docs/KUESTENMODELL.md` beschreibt noch das alte Modell (Reichweite,
  normierte Formen) und ist nachzufuehren.

---

# 2026-08-24 — Wegoptik, dann Abarbeitung der 20 einfachsten offenen Punkte

**Ausgangslage:** HEAD `950a949` (2026-08-12), Arbeit laeuft uncommitted im
Hauptcheckout auf `main`. Kein Worktree.

## Teil 1: Wege besser auf dem Mesh darstellen

Nutzerauftrag: zehn Methoden vorschlagen, davon die sinnvollen umsetzen.
Umgesetzt wurden **Methoden 1-3** plus ein Nebenbefund.

### 1. `glPolygonOffset` statt Weltversatz
`gui/widgets/map_display_3d.py` `_render_wegbaender()`. Vorher hielt allein
`SCHWEBE_ANTEIL` das Band ueber dem Gelaende - ein Versatz in der WELT, bei
flachem Blickwinkel sieht man darunter. Jetzt `glPolygonOffset(-2.0, -4.0)`,
das nur im Tiefenpuffer wirkt. `SCHWEBE_ANTEIL` 0.0006 -> 0.0001 als
Sicherheitsnetz.

### 2. Weicher Rand statt Polygonkante
Neues Vertexattribut `deckung` (Vertexformat 7 statt 6 float), 1 auf der
Fahrbahn, 0 an den Kanten. `wegband.frag` multipliziert es in die Alpha
(mit `sqrt`, sonst wirkt der deckende Teil zu schmal) und verwirft Fragmente
nahe 0. Die lineare Interpolation zwischen Schulter und Kante erzeugt den
Verlauf von allein.

### 3. Querprofil statt flachem Rechteck
`gui/widgets/wege_geometrie.py`: fuenf Bahnen je Wegpunkt
(Kante/Schulter/Scheitel/Schulter/Kante), Woelbung `h(t) = w * (1 - t^2)`.
Geometrische Woelbung klein (`WOELBUNG_ANTEIL = 0.02`, gemessen 2.1 m), die
NORMALEN aber um `NORMALEN_WOELBUNG = 0.40` verkippt - absichtlich viel
staerker, damit die Schattierung die Woelbung zeigt und der Umriss sie nur
andeutet.

### Nebenbefund: GL-Puffer wurden JEDEN FRAME neu angelegt
`_render_wegbaender()` legte VAO/VBO/EBO pro Frame an, lud die kompletten
Baender hoch (gemessen 1.1 MB bei 40 Wegen) und loeschte alles wieder. Kein
Leck, aber der Python-Cache darueber sparte nur das Rechnen, nicht die
Uebertragung. Jetzt haengen die Puffer am Cache;
`_wegband_puffer_freigeben()` raeumt beim Geometriewechsel und in
`_cleanup_mesh_buffers()`.

### Zwei Messbefunde beim Umbau
* **Hoehe stuetzte sich auf drei Querstellen** (Mitte, linker, rechter Rand) -
  ein Grat DAZWISCHEN wurde uebersehen. Jetzt `QUER_STUETZSTELLEN = 9`.
* **Die Laengsglaettung schneidet an Kuppen ein**, gemessen 0.7 m bei 256 px.
  Kein Fehler - das tut eine echte Trasse auch, und mit dem Tiefenversatz ist
  es unsichtbar.

### GELOCKERTE ZUSICHERUNG - bitte nachvollziehen
`tests/smoke_test_wege_geometrie.py`: die Bedingung "kein Vertex unter dem
Gelaende" wurde zu "hoechstens `EINSCHNITT_MAX_M = 2.0` m Einschnitt, im
Median darueber" **gelockert**. Begruendung steht im Test. Wer das anders
sieht, muss die Laengsglaettung abschalten oder den Weltversatz
zurueckholen.

### Auch behoben: stiller Rueckfall
`_render_wegbaender()` stieg ohne `wegband_shader_program` kommentarlos aus -
"Shader liess sich nicht uebersetzen" war von "es gibt keine Wege" nicht zu
unterscheiden. Meldet jetzt einmal laut.

**Tests:** `smoke_test_wege_geometrie.py` 6/6 gruen (drei Zusicherungen an das
neue Profil angepasst), `smoke_test_shader_paths.py` um eine vierte
Zusicherung erweitert (Varyings vert<->frag, Attributplaetze lueckenlos).
Deren erste Fassung war selbst loechrig: `dict()` auf (typ, name)-Paare macht
den TYP zum Schluessel, `FragPos` verschwand hinter `Normal`.

**NICHT bestaetigt:** wie es aussieht. OpenGL laesst sich headless nicht
pruefen.

### Nachbesserung 2026-08-24 nach Sichtprüfung (Nutzerbefund)

Nutzer: *"sieht besser aus. aber verschwindet noch immer bei vielen
kamera-bewegungs-aktionen. dann kommt es auf distanz zb auch mal vor, dass
der weg nicht komplett dargestellt wird sondern so gestueckelt sein kann ...
wege sind sehr windy ... nur 60% von der dicke ... nicht markierbar."*

**Verschwinden und Stueckeln war NICHT die Wegdarstellung, sondern die
Projektion.** In `_update_projection_matrix()` standen fest `near=0.1,
far=2000.0` mit dem Kommentar, das Verhaeltnis sei "fuer einen 24-Bit-
Tiefenpuffer unkritisch". Das ist falsch - die Genauigkeit haengt fast allein
an der NEAR-Plane. Nachgerechnet (Welt 10 Einheiten breit, 24 Bit):

| Kameraabstand | Tiefenaufloesung |
|---:|---:|
| 17 (Vorgabe) | 0.000172 Welteinheiten |
| 30 | 0.000536 |
| 60 | 0.002146 |

Das Wegband schwebt 0.001 Einheiten. **Ab Kameraabstand ~45 ist die
Tiefenaufloesung groesser als der Abstand** - genau das Symptom. Jetzt wandern
Near und Far mit (`near = Abstand/20`, `far = Abstand + 200`), das ist 8.6x
bis 30.8x genauer. Polygon-Offset zusaetzlich auf (-3, -6).

**Wiggle:** das A*-Routing laeuft auf einer 8er-Nachbarschaft, eine schraege
Strecke wird dort zur Treppe mit 45-Grad-Wechseln. Bisher wurde nur die HOEHE
geglaettet, nicht der Verlauf. Neu `GLAETTUNG_XY_FENSTER = 7` mit zwei
Durchgaengen (zwei kleine statt eines grossen Fensters - gleiche Glaette bei
halbem Versatz), Endpunkte fest. Gemessen: Gesamtdrehung **4410 -> 83 Grad
(53x ruhiger)**, groesster Versatz 1.0 px, Endpunkte exakt getroffen.

**Breite auf 60 %:** 35/26 m -> 21/16 m, `MINDEST_BREITE_PX` 2.5 -> 1.5. Bei
512 px sind das 62 m statt 104 m.

**Markierbar:** die Auswahl funktionierte bereits (Klick -> Treffer -> Text im
Seitenfeld), aber **im Bild passierte nichts** - der Shader kann die
Einfaerbung seit dem 2026-08-16, sie wurde nur nie eingeschaltet. Jetzt gibt
`treffer_suchen()` den `index` mit zurueck, das Display merkt sich den
gewaehlten Weg und zeichnet dessen Indexbereich ein zweites Mal mit
`ausgewaehlt = 1`. Zweiter Draw-Call statt Vertexfarbe: die Auswahl aendert
sich pro Klick, die Geometrie nicht.

**Woelbung:** der Nutzer sieht sie nicht und haelt sie bei dieser
Darstellungsart fuer verzichtbar. Bleibt drin, kostet nichts.

---

## Teil 1b: Kuestenprofile — gibt es die, und prueft sie jemand?

Nutzerfrage: *"sag mal ob du zugriff auf die ganzen kuestenprofile hast die
es haben soll und ob es einen test gibt ob die kuesten so aussehen wie die
profile?"*

**Die Profile gibt es**, in `core/vektor_kueste.py`: `MESS_FORM_JE_ARCHETYP`
(27 Archetypen, je 17 Stuetzstellen), `MESSWERTE_JE_REGION` (Klippenhoehe
p10/p90) und `MESS_REICHWEITE_M` - alle aus echten DEMs gewonnen.

**Geprueft hat sie niemand.** `smoke_test_vektor_kueste.py` prueft sechs
Dinge, und alle sechs sind STRUKTURELL (eine Funktion, Rasterfreiheit,
Aufloesung, Determinismus, Lage der 0-Linie, Randfaelle). Keines vergleicht
die entstandene FORM mit der Vorlage.

**Neu: `tests/smoke_test_kuestenprofiltreue.py` (2/2 gruen).** Misst mit
denselben Funktionen, mit denen die Tabellen entstanden sind
(`tools/kuestenlaengsschnitt.py`), gruppiert nach (Region, Archetyp).

Ergebnis ueber 20 Gruppen:

* **Form: Median-RMS 0.121** auf der 0..1-Skala. Klippenarchetypen treffen gut
  (Kola 0.028, Kotor 0.031, Foerdenkueste 0.040, Santorini 0.063, Amalfi
  0.063). **Straende und Flachkuesten treffen schlecht** (Toskana-Straende
  0.374, Weissmeer-Flachkueste 0.320, Dingle 0.318, Kykladen-Strand 0.317) -
  eine markante Klippe setzt sich gegen das Rauschgelaende durch, ein Strand
  geht darin unter.
* **Hoehe: vier Ausreisser**, namentlich in `BEKANNTE_HOEHENABWEICHUNGEN`
  gefuehrt. **Ursache ist ein Modellfehler in der TABELLE, kein Rechenfehler:**
  `MESSWERTE_JE_REGION` ist je REGION tabelliert und stammt aus je EINER
  Vorbildkueste, waehrend jede Region DREI Archetypen unterschiedlichen
  Charakters hat. Die Taiga-Hoehen kommen von den Stockholmer Schaeren
  (15-30 m), ihre Archetypen heissen aber "Kola-Steilkueste" und
  "Weissmeer-Flachkueste" - eine Steilkueste kann das Schaerenband gar nicht
  einhalten. **Der saubere Weg waere, die Hoehen je ARCHETYP zu messen statt
  je Region.**

Eigener Messfehler dabei gefunden und behoben: die erste Fassung nahm die
Klippenhoehe als Maximum ueber die GANZE Landseite - damit misst man das
Hinterland mit, und alle Regionen sahen zu hoch aus. Jetzt nur innerhalb der
Profilreichweite, gegen die Uferhoehe gerechnet.

---

## Teil 2: Die 20 einfachsten offenen Punkte

### [x] 12.3 + 12.4 — Doppelte und unerreichbare Parameter
`erosion_strength` steht in class EROSION (0.0-2.0, Vorgabe 0.5) UND class
WATER (0.1-5.0, Vorgabe 2.5). **Nicht geloescht** - `core/water_generator.py`
liest die Schluessel weiter ueber `parameters.get(...)`, ein Loeschen wuerde
nur die dokumentierte Spanne entfernen und den stillen Rueckfall auf die
Literalwerte im Generator hinterlassen.

Stattdessen: neues Register `DOPPELTE_SCHLUESSEL` in
`gui/config/value_default.py`, und die stillgelegten Droplet-Regler
(`erosion_passes`, `sediment_capacity_factor`, `settling_velocity`,
`thermal_erosion_strength`, `evaporation_base_rate`, `diffusion_radius`)
tragen jetzt eine Begruendung in `stillgelegte_regler()`.

**Neuer Test `tests/smoke_test_parameter_eindeutig.py` (4/4 gruen)** - er hat
sofort einen ZWEITEN Doppelschluessel gefunden, von dem in 12.3 nichts stand:
`octaves`. Nachgeprueft: harmlos, nur der Attributname ist gleich
(`TERRAIN.OCTAVES` laeuft als `octaves`, `EROSION_FILTER.OCTAVES` als
`erosion_filter_octaves`). Steht mit dieser Erklaerung im Register.

**Eigener Messfehler dabei:** die erste Testfassung verglich Attributnamen
statt Parameterschluessel und meldete `octaves` faelschlich als Kollision.
Der Test prueft jetzt beides getrennt.

### [x] 12.2 — Erosionsreiter kennzeichnen
`gui/tabs/erosion_tab.py`: `_create_stilllegungs_hinweis()` zeigt einen
gelben Hinweisstreifen, solange `EROSION_AKTIV` False ist. **An den Schalter
gekoppelt, nicht fest verdrahtet** - wird er True, verschwindet der Streifen
von selbst.

### [x] 6.3 — WAR SCHON ERLEDIGT
Der Wetter-Reiter hat bereits zwei getrennte Zeilen (Messgroesse /
Atmosphaere-Schicht), der Docstring von `create_visualization_controls()`
nennt 6.3 ausdruecklich. Nur der Haken in der Liste fehlte.

### [x] 6.26 — Umschalten von Settlements/Roads
**Gemessen:** `update_display(heightmap)` bei 512 px = 150.3 ms, davon
`canvas.draw()` allein 90.6 ms. Beim Umschalten einer Checkbox wird ZWEIMAL
gezeichnet (Basiskarte, dann Overlays).

**Fix:** alle 17 `self.canvas.draw()` in `gui/widgets/map_display_2d.py` auf
`draw_idle()` umgestellt. Qt fasst mehrere Anfragen zu einem Durchgang
zusammen. **150.3 ms -> 42.2 ms**, alle 30 Darstellungen
(`smoke_test_display_2d.py`) weiter gruen.

WICHTIG: die Modulfunktionen `rasterize_*` zeichnen auf EIGENE
Offscreen-Canvases und lesen direkt danach `buffer_rgba()` - die brauchen
`draw()` synchron und wurden NICHT umgestellt.

### [x] 6.22 — Monats-Wetterkarten
`_on_month_cycle_tick()` prueft jetzt `viewport_widget.isVisible()` und
zeichnet nur, wenn der Reiter vorn ist. Der Monatsindex laeuft dabei bewusst
nicht weiter, damit die Anzeige beim Zurueckwechseln nicht springt. Gleiche
Ueberlegung wie 6.14.

### [x] 3b.9 — Rechenzeit der Vektor-Kueste, jetzt gemessen

| px | Vektor | Raster | Faktor |
|---|---:|---:|---:|
| 256 | 1.41 s | 0.65 s | 2.18x |
| 384 | 2.13 s | 1.02 s | 2.09x |
| 512 | 3.19 s | 1.61 s | 1.98x |
| **1024** | **10.67 s** | **5.39 s** | **1.98x** |

**Antwort: der Faktor bleibt konstant bei ~2x**, er explodiert nicht mit der
Aufloesung. (Die Doku nannte fuer 384 px 3.26/1.25 s - andere Maschinenlast
oder aelterer Stand; das Verhaeltnis stimmt ueberein.)

### [x] 13.1 — Export auf feste Weltgroesse
`gui/utils/map_export.py`: `EXPORT_KANTENLAENGE_PX = 2048`, jeder Layer wird
beim Export darauf gebracht. **Kategorien und Farbbilder per naechstem
Nachbarn**, skalare Felder bilinear - zwischen Biom 3 und Biom 7 liegt kein
Biom 5. Manifest fuehrt jetzt `export_kantenlaenge_px`, `welt_km`,
`meter_pro_pixel` und je Layer einen `groesse`-Vermerk ("nativ" /
"bilinear 512->2048" / "naechster Nachbar ...").

### [x] 13.2 — Daempfungsmaske fuers Engine-Rauschen
`daempfungsmaske()` in `map_export.py`, exportiert als
`noise_damping_mask.png`. 0 = nicht rauschen, 1 = volle Wildnis. Quellen:
`city_mask`, `street_mask`, `house_parcel_map`, `roads`/`sea_roads` und ein
Uferstreifen um die Wasserlinie. Radien in METERN
(`WEG_SCHUTZ_M = 45`, `UFER_SCHUTZ_M = 35`, `BAU_SCHUTZ_M = 25`,
`UEBERGANG_M = 60`), nicht in Pixeln - sonst haengt die Korridorbreite an der
Exportaufloesung.

**Neuer Test `tests/smoke_test_export_2048.py` (3/3 gruen).**

### [ ] 3b.8 — BEWUSST NICHT UMGESETZT
Auftrag war, `_kuesten_umformen()` (~200 Zeilen) als toten Code zu loeschen.
**Ist kein toter Code:** es ist der `else`-Zweig von `VEKTOR_KUESTE_AKTIV`,
und der Schalter existiert laut Kommentar in `core/terrain_weltkarte.py`
genau deshalb, weil die Regionseichung (3.9, noch offen) daran haengt. Den
Vergleichsmassstab zu loeschen, solange 3.9 ungeklaert ist, waere falsch.
**Erst 3.9 klaeren, dann loeschen.**

### [x] 13.5 — Vektordaten exportieren
`vektordaten()` in `map_export.py`, geschrieben als `vektor.json`. Enthaelt
Wege, Seewege, Grundstuecksgrenzen und Ortslagen (mit Typ, Rang, Kultur,
Haeuserzahl, Radius). **Koordinaten in METERN, nicht in Pixeln** - der
Pixelwert haengt an der Kartengroesse, Meter haengen an der Welt.

**Fluesse fehlen, mit Begruendung im Code und im JSON-Feld `fehlt`:**
`core/terrain_weltfluesse.flussnetz()` baut sehr wohl einen Knotengraphen,
aber `core/terrain_generator.py:1707` behaelt daraus nur die Raster
`river_mask`/`river_order` und wirft den Graphen weg. Aus dem Raster wieder
Linienzuege zu machen waere Arbeit mit eigenen Fehlerquellen fuer etwas, das
vorher schon vorlag. **Der richtige naechste Schritt ist, den Graphen in den
Ausgaben zu behalten** - ein Eingriff in den Terrain-Generator, nicht in den
Exporteur.

### [x] 6.19 — Grenze der Heightmap dort dokumentiert, wo man sucht
Der Punkt war ausfuehrlich in `OFFENE_PUNKTE.md` beschrieben, aber nicht im
Code. Jetzt steht im Kopf von `gui/widgets/adaptive_terrain_mesh.py`, warum
dieses Modul die 90-Grad-Kueste PRINZIPIELL nicht beheben kann (eine
Heightmap speichert je (x,y) genau einen Wert) und wohin man stattdessen
schaut: `terrain_remesh.py` (hilft ueber die Flaeche, nicht an der
Kuestenlinie) bzw. Zwangskante/Vektorweg.

### [ ] 7.4 — NICHT UMSETZBAR, Angaben fehlen
"Vier Schwellwertkipper aus float32 als Toleranz fuehren". **Welche vier,
steht nirgends** - weder im Eintrag (eine Zeile ohne Details) noch im Code
(`grep` nach "Kipper"/"kipp" findet ausserhalb dieser Zeile nichts, und keiner
der Paritaetstests fuehrt eine solche Liste). Ohne die Angabe waere jede
Toleranz geraten. **Braucht die Messung, aus der die vier stammen.**

---

## Testlage nach dieser Sitzung

Alle acht beruehrten Testdateien gruen:
`smoke_test_wege_geometrie`, `smoke_test_shader_paths`,
`smoke_test_parameter_eindeutig` (neu), `smoke_test_export_2048` (neu),
`smoke_test_display_2d`, `smoke_test_display_methoden_existieren`,
`smoke_test_terrain_remesh`, `smoke_test_adaptive_mesh_vectorized`.

---

## Noch offen aus den 20

**Nicht angefasst, weil deutlich groesser als "einfach":**

* **8.1** Logzeilen/Fortschrittstexte von `lod` befreien (~373 Stellen) -
  mechanisch, aber jede Stelle einzeln zu pruefen.
* **7.9** Geology ohne GPU-Anbindung - ein ganzer Generator-Port, kein
  Kleinkram.
* **9.2** Neun Niederschlagskappungen gegen Modellgrenzen messen - braucht
  Messlaeufe je Regler.
* **12.5** Orchestrator-Rundenbetrieb testen - der Eintrag sagt selbst, dass
  er mit 8.2 wegfaellt.
* **6.4/6.13** 3D-Ansicht testen - OpenGL ist headless nicht pruefbar, ein
  Test kann nur Struktur pruefen, nicht das Bild.
* **3b.3** Mesh-Schnitt auf Vektorhoehen - haengt an 3b.1 (visuelle
  Bestaetigung), sonst optimiert man ins Blaue.
* **1.8** Klimatabelle gegenpruefen - Recherche an externen Quellen.
* **7.4** siehe oben, Angaben fehlen.
* **3b.8** siehe oben, waere falsch solange 3.9 offen ist.

## Was der Nutzer pruefen muss

Steht in `docs/PRUEFLISTE_LIVE.md`.
