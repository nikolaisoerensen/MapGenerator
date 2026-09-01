# Fluesse und Wasserverteilung — Plan

Stand 2026-08-24. Angelegt auf Nutzerwunsch: *"ich sag viele sachen,
versuche es fuer mich etwas zu ordnen und die ordnung im programm fuer mich
zu halten."*

**Diese Datei ist die Ordnung.** Was hier nicht steht, ist nicht
beschlossen. Erledigtes wird abgehakt, nicht geloescht.

---

## Der Befund, der alles ausgeloest hat

Gemessen am 2026-08-24, 512 px, Seed 20260804:

| Region | Wassermenge* | groesster Fluss | Laeufe > 300 | Niederschlag |
|---|---:|---:|---:|---:|
| **Skerrheim** | **13.0 Mio** | **123** | **0** | 1967 mm |
| Clonagh | 8.1 Mio | 167 | 0 | 1148 mm |
| Morobora | 8.1 Mio | 391 | 9 | 865 mm |
| Estrande | 7.4 Mio | 654 | 16 | 777 mm |
| Nevadin | 7.3 Mio | 587 | 33 | 828 mm |
| Macchia | 6.6 Mio | 320 | 8 | 785 mm |
| Nebelrode | 6.2 Mio | 103 | 0 | 634 mm |
| Samarcia | 4.3 Mio | 374 | 28 | 471 mm |
| Thalassia | 3.6 Mio | 167 | 0 | 480 mm |

\* Niederschlag mal Landflaeche

**Die Korrelation ist negativ.** Das Skerrheim hat die meiste Wassermenge
und den kleinsten Hauptfluss; die Samarcia hat die wenigste und einen
dreimal groesseren.

**Die Ursache steht in einer Zeile.** `core/terrain_weltfluesse.py:377`:

```python
flaeche = np.ones(n)
```

Jeder Knoten traegt 1 bei, unabhaengig davon, ob dort 471 mm oder 1967 mm
fallen. Was das Modell "Einzugsgebiet" nennt, ist die FLAECHE des
Einzugsgebiets in Knoten - nicht die Wassermenge. Der Niederschlag steht
als volles Feld bereit (`felder["niederschlag_mm"]`) und wird nie gelesen.

---

## Was da ist, und was nicht

Geprueft am 2026-08-24, damit niemand danach sucht:

| Frage | Antwort |
|---|---|
| Kennt die Flusserzeugung die Regionen? | **Ja.** `_weltfluesse(heightmap, felder, size, seed)` bekommt `felder` komplett: `niederschlag_mm`, `regionen`, `seegrad`, `wind_mittel_ms`, `temp_mittel_m0`. Sie werden nur nicht an `flussnetz()` weitergereicht. |
| Wie entsteht Regen? | **Nicht simuliert.** `niederschlag_mm` ist ein Regionsparameter aus der Tabelle in `core/terrain_weltkarte.py` (Skerrheim 2250, Samarcia 600), ueber die Voronoi-Gewichte weich interpoliert. |
| Gibt es einzelne Voronoi-Zellen? | **Nein.** `voronoi_regionen()` verrechnet sie zu Gewichten; uebrig bleibt `regionen` (0-8, die fuehrende Region je Pixel). Fuer Hangausrichtung braucht es sie nicht - die ist punktweise aus dem Gradienten zu haben, feiner als jede Zelle. |
| Gibt es Hangausrichtung schon? | **Ja, aber woanders.** `core/biome_generator.py:560` rechnet `south_facing`. Das Muster ist da, im Terrain-/Flussteil fehlt es. |
| Reihenfolge Kuesten/Fluesse? | **Kuesten zuerst.** kontinentform -> voronoi_regionen -> seegliederung -> oktavenstapel -> **Vektorkueste** -> seetiefe -> erosionsfilter -> **weltfluesse**. Die Kueste weiss nichts von den Fluessen. |

---

## Block 1 — Wasser richtig verteilen

Grundlage fuer alles andere. Ohne diesen Block sind die spaeteren Bloecke
Kosmetik auf einer falschen Verteilung.

### [x] 1.1 Niederschlag als Knotengewicht — UMGESETZT 2026-08-24
`flaeche = np.ones(n)` in `baue_stufe()` ersetzt durch das
Niederschlagsgewicht des Knotens, bezogen auf den Mittelwert ueber LAND
(nicht ueber die ganze Karte - ueber See faellt zwar Regen, aber er
speist keinen Fluss). `flussnetz()` und `_weltfluesse()` reichen
`felder["niederschlag_mm"]` durch. Ohne Feld bleibt es bei `ones`, damit
Werkzeuge und Tests ohne Regionsdaten unveraendert laufen.

GEMESSEN (512 px, Seed 20260804), groesster Fluss je Region:

| Region | vorher | nachher | Niederschlag |
|---|---:|---:|---:|
| Skerrheim | 123 | **215** (+75 %) | 1967 mm |
| Clonagh | 167 | 247 (+48 %) | 1148 mm |
| Morobora | 391 | 533 (+36 %) | 865 mm |
| Estrande | 654 | 737 (+13 %) | 777 mm |
| Nevadin | 587 | 676 (+15 %) | 828 mm |
| Nebelrode | 103 | 123 (+19 %) | 634 mm |
| Samarcia | 374 | **226** (-40 %) | 471 mm |
| Macchia | 320 | 197 (-38 %) | 785 mm |
| Thalassia | 167 | 119 (-29 %) | 480 mm |

Die Richtung stimmt durchgehend: nasse Regionen wachsen, trockene
schrumpfen. `smoke_test_river_reaches_sea` bleibt gruen.

**ABER DAS FJORDLAND FUEHRT IMMER NOCH NICHT.** 215 gegen 737 an der
Estrande. Der Niederschlag allein reicht nicht.

### Warum nicht — drei widerlegte Hypothesen

Damit niemand sie erneut prueft:

1. **"Skerrheim ist in viele Landstuecke zerteilt."** FALSCH. Gemessen
   liegt es zu 100 % in EINEM zusammenhaengenden Stueck, genau wie die
   Estrande. (Die Griechischen Inseln haben mit 53 % tatsaechlich
   das Problem, das Skerrheim nicht.)
2. **"Die Fjorde schneiden ein, die Fliesswege sind kurz."** FALSCH, und
   zwar deutlich: das Skerrheim hat mit 832 m den ZWEITLAENGSTEN mittleren
   Abstand zur Kueste, die Estrande nur 395 m. Niederschlag mal
   Fliessweg ist im Skerrheim mit Abstand am hoechsten (1637 gegen 307).
3. **"Die Laeufe buendeln sich zu wenig."** NICHT SCHLUESSIG. Die
   Kennzahl Knoten je Muendung ist an der Estrande am niedrigsten
   (0.8) und dort ist der Fluss am groessten - die Metrik misst
   offenbar nicht, was sie soll (Unterwasserknoten zaehlen als
   Muendung mit).

**Die Ursache ist damit offen.** Naechster Verdacht waere die
Kantenkostenstruktur von Dijkstra im steilen Gelaende (Vorschlag 8 der
urspruenglichen Zehnerliste), aber das ist ungeprueft.

### [x] 1.2 Hangausrichtung — UMGESETZT 2026-08-24

`_hangfeuchte()` in `core/terrain_weltkarte.py`, aufgerufen am ENDE von
`weltfeld()`. Neuer Regionsparameter `hang_trockenheit` (Anteil, um den
ein voller Suedhang trockener wird; ein Nordhang wird um denselben Betrag
feuchter).

**Warum erst am Ende:** die Parameterfelder entstehen aus den
Voronoi-Gewichten, und zu dem Zeitpunkt gibt es noch gar kein Gelaende -
`H` steht erst ab der Grundhoehenbildung. Eine Hangausrichtung braucht
aber Haenge.

**Punktweise statt je Voronoi-Zelle.** Der Nutzer hatte nach Zellen
gefragt; die werden nicht behalten. Der Gradient ist ohnehin feiner und
trifft die Sache genauer - eine Zelle kann Nord- UND Suedhaenge enthalten.

**An die Neigung gekoppelt** (`HANG_VOLL_GRAD = 12`): auf einer Ebene gibt
es keine Exposition, und ohne diese Kopplung bekaeme flaches Land
zufaellige Feuchteunterschiede aus dem Rundungsrauschen des Gradienten.

GEMESSEN (384 px, nur Haenge ueber 8 Grad):

| Region | Suedhang | Nordhang | Verhaeltnis | `hang_trockenheit` |
|---|---:|---:|---:|---:|
| Samarcia | 362 mm | 648 mm | 0.56 | 0.45 |
| Macchia | 538 mm | 1019 mm | 0.53 | 0.35 |
| Nevadin | 645 mm | 1013 mm | 0.64 | 0.30 |
| Morobora | 980 mm | 1278 mm | 0.77 | 0.20 |
| Skerrheim | 1698 mm | 2153 mm | 0.79 | 0.10 |
| Clonagh | 1138 mm | 1315 mm | 0.87 | 0.15 |

**Ein Nebeneffekt, den man kennen muss:** das Verhaeltnis folgt nicht
allein `hang_trockenheit`, sondern auch der HANGNEIGUNG der Region. Das
Skerrheim (Median 14.8 Grad) kommt trotz des niedrigsten Parameters auf
0.79, das flachere Clonagh (4.97 Grad) auf 0.87. Inhaltlich richtig -
ein steiler Suedhang trocknet staerker aus als ein flacher -, aber wer
die Regionen gegeneinander einstellen will, muss es wissen.

**`niederschlag_mm` ist damit nicht mehr nur ein Regionswert.** Das Feld
geht auch in Biome und Wetter; die sehen die Hangfeuchte jetzt mit. Das
ist gewollt (eine Wahrheit statt zweier), aber es ist eine Aenderung an
einem Feld, das andere Systeme lesen.

### Wirkung auf die Fluesse — das Skerrheim fuehrt jetzt

Groesster Fluss je Region, 512 px, mit 1.1 + 1.2 + Block 2:

| Region | vorher (Start) | jetzt |
|---|---:|---:|
| **Skerrheim** | 123 | **700** |
| Estrande | 654 | 571 |
| Morobora | 391 | 542 |
| Nevadin | 587 | 514 |

Das Skerrheim hat damit den groessten Fluss der Karte - die
Nutzervorgabe *"Skerrheim soll es groesser sein als jetzt der groesste
fluss"* ist erfuellt. Die Estrande faellt von 1222 (Stand nach
Block 2 ohne Hangfeuchte) auf 571, weil ihr Wassergewicht jetzt
differenzierter verteilt ist.

`smoke_test_river_reaches_sea` und `smoke_test_vektor_kueste` bleiben
gruen.

### ~~[ ] 1.2 Hangausrichtung als Feuchtemodulation~~ (urspruengliche Fassung)
Nutzeridee: *"Skerrheim, voronoi mit viel suedhang ist etwas trockener als
fjordland nordhang, dann Samarcia suedhang total trocken, nordhang etwas
feuchter."*

Punktweise aus dem Gradienten, nicht je Voronoi-Zelle. Der Effekt ist eine
MODULATION um den Regionswert, kein eigener Wert: ein Suedhang in der
Samarcia bleibt trockener als ein Suedhang im Skerrheim.

Staerke je Region regelbar - in der Samarcia soll der Unterschied gross
sein ("total trocken" gegen "etwas feuchter"), im Skerrheim klein.

### [ ] 1.3 Verdunstung abziehen
Abfluss ist Niederschlag minus Verdunstung. Die Samarcia verliert bei
471 mm und hoher Temperatur fast alles, das Skerrheim bei 1967 mm und
niedriger Temperatur kaum etwas. Verstaerkt 1.1 von Faktor 2.2 auf
geschaetzt 3-4.
*Nachrangig gegenueber 1.1 und 1.2 - erst messen, ob noetig.*

### [x] ~~1.4 Normierung je Region~~ — ZURUECKGEZOGEN
War Vorschlag 4 der Zehnerliste. Haette `gebiet = fl / fl.max()` je Region
statt global normiert.

**Zurueckgezogen nach Nutzerpraezisierung** (*"ansonsten finde ich es
derzeit sehr gut von der groesse her, nur halt von der verteilung etwas
mehr nach wasserbeckenmenge richten"*): der Vorschlag haette JEDEM
Regionshauptfluss ein volles Tal gegeben, unabhaengig von seiner
Wassermenge - also genau den Unterschied verwischt, um den es geht.
Sobald 1.1 wirkt, waechst der Skerrheim-Fluss von selbst, und globale
Normierung ist dann richtig: ein grosser Fluss ist ein grosser Fluss,
egal in welcher Region.

---

## Block 2 — Grosse Fluesse garantieren

Nutzervorgabe woertlich: *"dann werden grosse fluesse zumindest in
Skerrheim (100% chance), Morobora (66%) und Atlantik (66% chance) generiert.
der rest generiert weiterhin wie jetzt. nur ein kleiner nudge erstmal."*

### [x] 2.1 + 2.2 Regionsquote — UMGESETZT 2026-08-24

`_hauptstrom_erzwingen()` in `core/terrain_weltfluesse.py`. Je Region mit
Quote wird der groesste Lauf auf sein Sollmass angehoben, falls er es
nicht ohnehin erreicht. Der Wuerfel kommt aus dem Seed, damit dieselbe
Karte dasselbe Ergebnis gibt.

GEMESSEN (512 px, Seed 20260804), groesster Fluss je Region:

| Region | Start | nach 1.1 | nach Quote | Quote |
|---|---:|---:|---:|---|
| **Skerrheim** | 123 | 215 | **700** | 100 % |
| Morobora | 391 | 533 | 533 | 66 %, Ziel 500 schon erreicht |
| Estrande | 654 | 737 | 1222 | 66 %, Ziel schon erreicht |
| Nevadin | 587 | 676 | 1161 | keine |
| alle uebrigen | | | unveraendert | keine |

Estrande und Nevadin wachsen mit, weil der Skerrheim-Hauptstrom
dort durchlaeuft und muendet - ein grosser Fluss bleibt gross. Ihre
eigenen Laeufe sind unberuehrt.

`smoke_test_river_reaches_sea` bleibt gruen.

### Drei Fehler beim Bau, alle gemessen

1. **Doppelmultiplikation.** Kette und Zufluesse als getrennte Masken
   multiplizierten den Hauptknoten zweimal: 215 * 3.26 * 3.26 = 2275
   statt der angepeilten 700.
2. **Zufluesse oberhalb mitskaliert.** Falsch - dort fliesst kein
   zusaetzliches Wasser. Ein grosser Strom hat normale Nebenfluesse.
3. **Multiplikativ statt additiv.** Ein Faktor vergroesserte jeden Knoten
   flussabwaerts im selben VERHAELTNIS, auch die ohnehin grossen: die
   Estrande sprang auf 2396, das Nevadin auf 2197. Mehr Wasser im
   Oberlauf heisst flussabwaerts eine KONSTANTE Zugabe.

### ~~[ ] 2.1 Regionsquote fuer grosse Fluesse~~ (urspruengliche Fassung)
Als Regionsparameter neben `formgroesse_m` und `rauheit`:

| Region | Wahrscheinlichkeit |
|---|---:|
| Skerrheim | 100 % |
| Morobora | 66 % |
| Estrande | 66 % |
| alle uebrigen | wie bisher, keine Quote |

Aus dem Seed abgeleitet, damit dieselbe Karte dasselbe Ergebnis gibt.

### [ ] 2.2 Skerrheim groesser als der bisherige Groesste
Nutzervorgabe: *"Skerrheim soll es groesser sein als jetzt der groesste
fluss"* - also ueber 654 Knoten (Estrande, Messung oben).

**Reihenfolge beachten:** erst 1.1 umsetzen und messen. Wenn das Skerrheim
danach von selbst dorthin kommt, ist 2.1/2.2 nur noch eine Absicherung
gegen unguenstige Seeds. Eine Garantie, die eine falsche Physik
ueberdeckt, versteckt den Fehler statt ihn zu beheben.

---

## Block 3 — Seen als Sammler

Nutzer: *"klingt spannend! wenn du das umsetzen kannst."*

Ein See buendelt alle Zufluesse und gibt EINEN Ausfluss ab - der
Mechanismus, der in Norwegen die grossen Fluesse macht. Heute laufen die
Ketten an Seen vorbei, statt in ihnen zusammenzukommen.

`felder["seegrad"]` und die Seeflaechen liegen vor. Zu klaeren: ob der
Seeausfluss als eigener Knoten in den Baum kommt oder ob die
Kantenkosten innerhalb eines Sees auf ~0 gesetzt werden (dann findet
Dijkstra die Buendelung von selbst).

**FUER DAS FJORDLAND WIRKUNGSLOS — gemessen 2026-08-24.** Die Karte hat
11 Binnenseen ueber 4 Pixel, und KEINER liegt im Skerrheim: 252 Pixel
(Macchia), 2x 123 (Estrande), 71 (Thalassia), 35 und
28 (Clonagh), Rest unter 15 Pixel. Block 3 wuerde anderswo etwas
bringen, aber nicht dort, wo die grossen Fluesse gewuenscht sind.
Deshalb wurde stattdessen Block 2 gebaut.

---

## [x] Block 4 — Talformen je Region — UMGESETZT 2026-08-24

Nutzer: *"ja flusstypen sollte es geben, nach region. zB alpen eher V.
Skerrheim U und im Atlantik irgendwas dazwischen zB."*

Neuer Regionsparameter `talform`. Er ist der Exponent der
Querschnittskurve in `taeler_eingraben()`:

```
profil = (1 - exp(-abstand/breite)) ** talform
```

Klein heisst, das Profil steigt sofort - schmale Sohle, steile Flanken,
fluvial eingeschnittenes V-Tal. Gross heisst, es steigt traege - breite
flache Sohle, glazial ausgeschuerftes U-Tal.

`form` war bis dahin ein FESTWERT (1.3) fuer die ganze Karte; er bleibt
der Rueckfall, wenn das Feld fehlt. Als Regionsparameter laeuft `talform`
durch dieselbe Voronoi-Ueberblendung wie alle anderen - an einer
Regionsgrenze geht die Talform allmaehlich ueber, statt zu springen.

GEMESSEN im fertigen Gelaende (512 px), Hoehe ueber der Talsohle in 80 m
Abstand - das misst, wie breit die Sohle ist:

| Region | `talform` | Hoehe bei 80 m | bei 300 m |
|---|---:|---:|---:|
| Nevadin | 1.0 | **11.9 m** | 142.3 m |
| Macchia | 0.9 | 5.6 m | 23.0 m |
| Samarcia | 1.1 | 3.4 m | 83.2 m |
| Nebelrode | 1.4 | 1.0 m | 26.0 m |
| Clonagh | 1.4 | 0.5 m | 6.4 m |
| Estrande | 1.4 | 0.5 m | 17.3 m |
| Morobora | 2.2 | **0.1 m** | 40.4 m |
| Skerrheim | 2.3 | **0.2 m** | 45.1 m |

Die Ordnung ist durchgehend: V-Taeler stehen nach 80 m schon deutlich
ueber der Sohle, U-Taeler praktisch gar nicht. Bild:
`docs/inseltest/talformen.png`.

**Aendert das Aussehen, nicht die Wassermenge** - unabhaengig von
Block 1-3.

---

## Block 6 — Seen in der Morobora

Nutzerfrage 2026-08-24: *"ich finde taiga sollte seenlandschaften haben.
warum sind dort keine? kann man dort ein paar laengliche seen erzeugen?
also so wie in lappland oder sowas."*

### Warum es dort keine gibt

**Das Terrain kennt ueberhaupt keine Seen ueber Meeresniveau.** Was das
Modell "Binnensee" nennt, sind Flaechen mit H <= 0 ohne Verbindung zum
Kartenrand - also zufaellige Senken, die unter den Meeresspiegel fallen.
Ein Lappland-See liegt aber auf 200 m Hoehe. Es gibt keinen Mechanismus,
der eine Senke bis zum Ueberlauf FUELLT.

Gemessen: 11 solcher Zufallssenken auf der ganzen Karte, keine in der
Morobora (dort liegt das Land im Median auf 141 m).

### Das Gelaende waere geeignet

Senken, die sich fuellen liessen (Fuellhoehe ueber 1 m, 9-px-Fenster):

| Region | Senkenpixel | Anteil des Landes | tiefste |
|---|---:|---:|---:|
| Nevadin | 5994 | 68.3 % | 160 m |
| Nebelrode | 4671 | 47.5 % | 122 m |
| **Morobora** | **3333** | **35.6 %** | **77 m** |
| Samarcia | 3526 | 38.2 % | 75 m |

Die Morobora hat also reichlich Senken - sie werden nur nie zu Seen.

### Was zu klaeren ist, bevor gebaut wird

1. **Gibt es schon Seen im Wassersystem?** `water.lake_detection` liefert
   eine `lake_map` und laeuft nach dem Terrain. Ob dort Morobora-Seen
   entstehen und ob sie im Bild ankommen, ist UNGEPRUEFT. Falls ja, ist
   es ein Anzeige- und kein Erzeugungsproblem.
2. **Sollen die Seen ins Terrain oder in die Wasserebene?** Ins Terrain
   heisst: die Heightmap bekommt eine Seeflaeche auf Ueberlaufhoehe, alle
   nachgelagerten Stufen sehen sie. In die Wasserebene heisst: das
   Gelaende bleibt, nur die Anzeige und das Wassersystem kennen sie.
3. **Laenglich wie in Lappland** ist glaziale Form - Rinnen entlang einer
   Eisbewegungsrichtung. Das waere derselbe Mechanismus wie
   `taeler_eingraben`, nur ohne Gefaelle: eine Rinne, die sich fuellt.
   Passt zu Block 4 (Talformen je Region, U-Form fuer glaziale
   Landschaften).

**AUF SPAETER GELEGT** (Nutzerentscheidung 2026-08-24). Der erste Schritt bleibt Punkt 1 - pruefen, ob `water.lake_detection` schon Morobora-Seen liefert. Erst wenn nicht, lohnt ein eigener Mechanismus.

---

## Block 5 — Offen, noch nicht beschlossen

### [ ] 5.1 Fjordarme als Vorfluter
War Vorschlag 6. Nutzer: *"da bin ich nicht sicher ob du es gut
hinbekommst. muesste ich sehen."*
**Nicht anfangen, bevor Block 1-3 gemessen sind** - moeglicherweise
erledigt sich der Bedarf.

### [ ] 5.2 Fjord bekommt Kuestentyp Geiranger
Nutzeridee: *"das waere ja sinnvoll wenn die fjorde dann den kuestentyp
geiranger bekommen wuerden."*

**Braucht eine Reihenfolgeaenderung.** Heute laufen die Kuesten VOR den
Fluessen; die Kueste kann also nicht wissen, wo ein Fluss muendet. Zwei
Wege:
  * Fluesse vorziehen - grosser Eingriff, das Flussnetz braucht das
    fertige Gelaende.
  * Kuestenzuweisung nachtraeglich korrigieren: wo ein grosser Fluss
    muendet, den Archetyp auf den Fjord-/Muendungstyp der Region setzen
    und die Umgebung neu mischen. Kleiner, aber ein zweiter Durchgang
    ueber die Kuestenzone.

Zu entscheiden, wenn Block 1-4 stehen.

### [ ] 5.3 Saisonale Schneeschmelze
War Vorschlag 9. Nutzer: *"saisonal kommt spaeter denke ich."*

### [x] ~~5.4 Knotendichte je Region~~ — VERWORFEN
War Vorschlag 7. Nutzer: *"nein eher nicht."*

---

## Reihenfolge der Umsetzung

1. [x] **1.1** (Niederschlag als Gewicht) — Skerrheim 123 -> 215
2. [x] **1.2** (Hangausrichtung) — Suedhang/Nordhang je Region
3. [x] Gepruefte Zwischenfrage: das Skerrheim fuehrt NICHT von selbst
4. [x] **2.1/2.2** (Regionsquote) — Skerrheim auf 700, groesster der Karte
5. [x] **3** (Seen als Sammler) — fuer das Skerrheim WIRKUNGSLOS, kein
       einziger Binnensee liegt dort
6. [x] **4** (Talformen) — V/U je Region
7. [ ] **1.3** (Verdunstung) und Block 5/6 nach Bedarf

**Stand nach diesen Schritten**, groesster Fluss je Region (512 px):

| Region | Start | jetzt |
|---|---:|---:|
| **Skerrheim** | 123 | **700** |
| Estrande | 654 | 571 |
| Morobora | 391 | 542 |
| Nevadin | 587 | 514 |
| Clonagh | 167 | 216 |
| Samarcia | 374 | 217 |
| Macchia | 320 | 188 |
| Thalassia | 167 | 140 |
| Nebelrode | 103 | 128 |

Das Skerrheim hat den groessten Fluss der Karte. Die trockenen Regionen
sind kleiner geworden, die nassen groesser - die Verteilung folgt jetzt
der Wassermenge.

Nach jedem Schritt messen, nicht am Ende. Die Messung ist
`tools/`-seitig aufzubauen, damit sie wiederholbar bleibt.
