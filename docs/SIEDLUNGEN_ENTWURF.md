# Siedlungen und Wege — Entwurf

Stand 2026-08-06. Abgestimmt mit dem Nutzer am 2026-08-06.
Auswahl der Ortsnamen: `docs/KULTUREN_UND_ORTE.md`.


## 1. Drei Groessen, alle klein

Alle Orte der Welt sind aehnlich klein — 15 bis 50 Haeuser. Die Unterscheidung
ist keine von Stadt gegen Metropole, sondern eine von **Rang innerhalb einer
sehr kleinen Spanne**:

| Rang | Haeuser | rund |
|---|---|---|
| Dorf | 15 – 25 | zwei Dutzend Hoefe, eine Gasse |
| Siedlung | 25 – 35 | mehrere Gassen, ein Markt |
| Stadt | 35 – 50 | Markt, Wehr, mehrere Wege laufen zusammen |

Damit ist auch die "Stadt" noch ein Ort, den man in einer Viertelstunde
durchquert. Das entspricht 932 der Wirklichkeit fast ueberall noerdlich der
Alpen.


## 2. Wo die Orte liegen — und warum

Ein Eignungswert je Pixel, aus fuenf Faktoren. Jeder ist begruendet; keiner ist
nur "sieht gut aus".

| Faktor | Wirkung | Begruendung |
|---|---|---|
| **Wasser am Ort** — grosser Fluss, Muendung, geschuetzte Bucht | staerkster Einzelfaktor | Trinkwasser, Muehlen, Fischerei und vor allem der Transportweg. 932 ist Wasser der einzige billige Weg fuer Masse. Deshalb liegen die groessten Orte dort. |
| **Ebener Grund** | stark | Bauplatz und Pflugland. Ein Hang ueber ca. 15 % traegt weder das eine noch das andere. |
| **Ackerland im Umkreis** — Anteil flacher, tiefer Flaechen im Radius | stark | Bestimmt, **wieviele** Menschen der Ort ernaehren kann. Das ist der Faktor, der die GROESSE traegt, waehrend Wasser die LAGE traegt. |
| **Hoehenlage** | daempfend | Je hoeher, desto kuerzer die Wachstumszeit und desto weniger Ertrag. In den Bergen werden Orte deshalb kleiner, nicht seltener. |
| **Erreichbarkeit** — wie teuer der Weg nach draussen ist | daempfend | Ein Ort, den niemand billig erreicht, waechst nicht. Er verschwindet aber auch nicht — er bleibt Dorf. |

**Varianz ist Pflicht.** Der Rang folgt nicht starr aus dem Wert. Gezogen wird
mit Rauschen um den Wert herum, sodass gelegentlich ein Dorf an bester Lage
sitzt und eine Stadt an mittelmaessiger. Ohne das wirkt die Karte gerechnet.


## 3. Wieviele Orte je Kultur

**Nicht fest drei.** Zwischen **2 und 5**, abgeleitet aus dem, was die Region
hergibt: die Summe der Eignung ueber der Region, verglichen mit allen neun.

Eine Welt, deren Skerrheim zufaellig weite bewohnbare Hochflaechen bekam, traegt
dort fuenf Orte; eine, in der es fast nur Steilwand ist, zwei. Das macht den
Seed spuerbar, statt ihn zu uebermalen.

Zusaetzlich immer: **mindestens ein Ort je Kultur ist Stadt** — sonst haette
eine Kultur keinen Mittelpunkt und keine garantierte Anbindung (Abschnitt 4).


## 4. Das Wegenetz

Reihenfolge — bewusst so und nicht anders:

### 4.1 Kostenfeld zuerst

Je Pixel ein Preis, bevor irgendein Weg gesucht wird:

| Gelaende | Preis | Begruendung |
|---|---|---|
| eben, Land | 1.0 | Bezugswert |
| Hang | + steigend mit dem Quadrat der Neigung | Ein doppelt so steiler Hang kostet deutlich mehr als das Doppelte — genau deshalb suchen sich Wege Saettel. |
| Flachwasser 0 bis -5 m | ca. 8x | **Furt oder kurze Bruecke.** Teuer, aber machbar. |
| Wasser -5 bis -10 m | ca. 25x | Nur noch als Bruecke, lohnt selten. |
| Wasser tiefer als -10 m | gesperrt fuer Landwege | |
| vorhandener Weg | ca. 0.4x | **Der wichtigste Trick.** Wege buendeln sich zu Hauptstrecken, statt parallel zu laufen. |

Der hohe Wasserpreis **pro Meter** ist der Grund, weshalb der billigste Weg von
selbst die schmalste Stelle sucht — Furten entstehen, ohne dass man sie setzt.

### 4.2 Welche Paare ueberhaupt in Frage kommen

**Gabriel-Graph** ueber alle Orte: zwei Orte sind Kandidaten, wenn im Kreis
ueber ihrer Verbindungsstrecke kein dritter liegt. Das ergibt ein
zusammenhaengendes Netz mit typisch 2–3 Nachbarn je Ort — kein Stern, keine
Vollverknuepfung.

### 4.3 Ob der Weg auch gebaut wird

Jeder Kandidat wird geprueft: **Bereitschaft gegen Wegkosten.**

```
Bereitschaft  =  Rang(A) * Rang(B)  *  (gleiche Kultur ? 1.0 : 0.45)
```

Gebaut wird, wenn Bereitschaft > Wegkosten. Damit gilt von selbst:

- zwei Staedte derselben Kultur verbinden sich fast immer
- zwei Doerfer verschiedener Kultur mit einem Gebirge dazwischen **nicht** —
  genau der Fall, den der Nutzer wollte
- ein abgelegener Ort bleibt an 1–2 Wegen haengen, weil alle weiteren Kandidaten
  an den Kosten scheitern

**Eine Ausnahme, die immer greift:** die Orte einer Kultur sind untereinander
**garantiert** verbunden. Nach dem Bereitschaftstest wird geprueft, ob jede
Kultur einen zusammenhaengenden Teilgraphen hat; fehlende Kanten werden
nachgetragen, egal was sie kosten.

### 4.4 Seewege

Fuer Paare, die keinen Landweg bekommen haben (Inseln, getrennte Kuesten), wird
ein **Seeweg** gesucht: derselbe Wegealgorithmus auf einem eigenen Kostenfeld,
in dem Land gesperrt ist und Flachwasser teuer.

Auflage: der Weg muss den **groessten Teil seiner Laenge in Wasser ab 10 m
Tiefe** liegen. Ein Seeweg, der sich an der Kueste entlangtastet, waere kein
Seeweg, sondern ein schlechter Landweg.

Seewege werden **anders gezeichnet** — gestrichelt, in einem eigenen Blau.

### 4.5 Kreuzungen

Erst NACH dem Routing: alle Wegepixel, an denen sich zwei Strecken treffen und
die **nicht** auf einem Ort liegen, sind Kreuzungen. Sie entstehen von selbst
durch den Wegerabatt aus 4.1 — dort, wo zwei Strecken ein Stueck gemeinsam
gehen und sich wieder trennen.

### 4.6 Roadsites zuletzt

Sie werden gesetzt, wenn das Netz steht:

- bevorzugt **an Kreuzungen** (ein Gasthof lebt vom Verkehr)
- an Furten und Passhoehen
- auf langen Zwischenstuecken ohne Ort, damit eine Tagesreise einen Halt hat

### 4.7 Landmarks

Unabhaengig vom Netz, nach eigenen Kriterien: Gipfel, Kliffs, Quellen,
abgelegene Stellen. Sie duerfen ausdruecklich weitab jedes Weges liegen.


## 5. Reihenfolge im Ueberblick

```
1  Eignungsfeld              (Wasser, Ebene, Ackerland, Hoehe)
2  Orte setzen               je Kultur 2-5, Rang mit Varianz
3  Kostenfeld                Hang, Wasser, Bruecken
4  Gabriel-Graph             welche Paare in Frage kommen
5  Wege routen               billigster Weg, mit Wegerabatt
6  Bereitschaftstest         manche Verbindungen scheitern
7  Kulturzusammenhang        fehlende Kanten nachtragen
8  Seewege                   fuer was uebrig bleibt
9  Kreuzungen finden
10 Roadsites setzen          bevorzugt an Kreuzungen
11 Landmarks setzen          unabhaengig
```

Erreichbarkeit (Faktor 5 in Abschnitt 2) haengt am Netz, das Netz an den Orten.
Aufgeloest wird das in **einem** Rueckschritt: nach Schritt 7 werden die Raenge
einmal nachkorrigiert, ohne die Orte zu verschieben. Kein zweiter Durchlauf.

## 6. Die Naht zum Spiel — Feldliste

**Entschieden in #19 (Fragen 19.2 und 19.3), festgeschrieben in Ticket #35.**

**Naht** heisst hier: die eine Stelle, an der das Spiel Siedlungsdaten
abholt — genau die Felder unten, nicht mehr und nicht weniger. Alles
andere ist Innenleben des Editors (Parzellen-Simulation, Wegephysik,
Plot-Knoten) und darf sich jederzeit aendern, ohne das Spiel zu brechen.

### 6.1 Die sechs Felder

| Feld | Datentyp | Einheit | Herkunft heute | Status |
|---|---|---|---|---|
| Kultur | `str`, einer von 9 Kulturnamen (Issue #19.5: ausdruecklich vorlaeufige Platzhalter) | Aufzaehlung | `Location.culture` | **geliefert** |
| Rang | `str`, einer von `'dorf'` / `'siedlung'` / `'stadt'` | Aufzaehlung (`RANG_HAEUSER`, Zeile 257) | `Location.rank` | **geliefert** |
| Stadtgroesse | `int`, 15-50 (Haeuserzahl je Rang, siehe `RANG_HAEUSER`) | Anzahl Haeuser | `Location.house_count` | **geliefert** |
| Stadttyp | `str`, einer von `'bergdorf'` / `'marktstadt'` / `'agrarstadt'` / `'sonstige'` | Aufzaehlung (`STADTTYPEN`, Zeile 1181) | `Location.settlement_type` | **geliefert** |
| Kontur der Stadtgrenze | Polygon: Liste von (x, y)-Punkten je Siedlung | Karten-Pixel der aktuellen Aufloesung | `_build_city_boundary_polygons()` (Zeile 2781) — nur **intern**, speist ausschliesslich die Plot-Knoten-Verteilung; der Calculator-Knoten `settlement.city_boundary` gibt nur Raster (`city_mask`, `city_cost_map`) aus, kein Polygon | **fehlt** |
| Anschlusspunkte der Wege | `dict {settlement_id: [(x, y), ...]}` — wo ein Weg die Stadtgrenze schneidet | Karten-Pixel der aktuellen Aufloesung | Calculator-Knoten `settlement.pathfinding`, Output-Key `road_entry_points` (`_calc_pathfinding()`, siehe `_stadtgrenzen_polygone_aus_maske()`/`_wege_anschlusspunkte()` in `core/settlement_generator.py`, Zeile ~2304) — schneidet die geroutete `roads`-Liste (Abschnitt 4) gegen eine aus `city_mask` (Knoten `settlement.city_boundary`) gebaute Stadtgrenzen-Kontur | **geliefert** (Ticket #73) |

Zum Feld `Stadtgroesse`: `Location` fuehrt zusaetzlich `radius` (`float`,
Karten-Pixel, `(3 + Haeuseranteil*2) * scale_factor` — siehe
`_get_prepared_settlement_inputs()`, Zeile 5029, wo `scale_factor =
heightmap.shape[0] / 128.0` die 128px-Referenzgroesse auf die tatsaechliche
Kartenaufloesung skaliert). `radius` ist eine interne Rundungshilfe fuer die
Plot-Node-Verteilung und deckt sich nicht mit der Kontur aus Feld 5 — sobald
die Kontur exportiert wird (Ticket TBD), ersetzt sie `radius` als
Groessenangabe fuer das Spiel; bis dahin ist `house_count` das massgebliche
Naht-Feld fuer "Stadtgroesse", `radius` bleibt Editor-intern.

### 6.2 Was ausdruecklich NICHT zur Naht gehoert

- **Parzellen** (`SettlementData.plots`, `plot_map`)
- **Innenstraessen der Stadt** (Strassennetz *innerhalb* der Stadtgrenze —
  zu unterscheiden vom ueberregionalen Wegenetz aus Abschnitt 4, das SEHR
  wohl zur Naht gehoert, aber nur bis zu den Anschlusspunkten)
- **Haeuserformen**
- **Gewerbe(-symbole)**

Das entsteht laut Issue #19 (Zitat des Nutzers, 19.3) **im Spiel** aus den
sechs Feldern oben, nicht im Editor: *"wir uebernehmen nur die kontur der
stadtgrenze und wo wege in die stadt fuehren und dann wird mit weiteren
informationen, wie stadtgroesse, stadttyp usw. die parzelle gezeichnet."*

### 6.3 Fehlende Felder — eigene Tickets

Eines der sechs Felder fehlt noch als externe Ausgabe:

- Kontur der Stadtgrenze als Polygon exportieren (baut auf dem bereits
  vorhandenen internen `_build_city_boundary_polygons()` auf) — Ticket #72

Anschlusspunkte der Wege an der Stadtgrenze sind seit Ticket #73 geliefert
(siehe 6.1) — der zweite Punkt dieser Liste ist damit erledigt.

### 6.4 Test

`tests/smoke_test_siedlungsnaht_felder.py` prueft, dass jede Siedlung im
`settlement_list`-Ausgabefeld die vier `Location`-Naht-Felder (Kultur, Rang,
Stadtgroesse, Stadttyp) traegt und mit den dokumentierten Typen/
Wertebereichen uebereinstimmt, UND dass `settlement.pathfinding`s
`road_entry_points`-Ausgabe je Siedlung eine Liste von (x, y)-Zahlenpaaren
innerhalb des Kartenrasters liefert. Er dokumentiert weiterhin ausdruecklich
(nicht als Fehlschlag, sondern als benannte Erwartung), dass die Kontur der
Stadtgrenze heute fehlt — sobald Ticket #72 sie liefert, muss dieser Test
darum erweitert werden.
