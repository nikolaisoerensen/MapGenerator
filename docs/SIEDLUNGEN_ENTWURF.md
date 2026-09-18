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


## 6. Die Siedlungsnaht — was das Spiel abholt

**Ticket #35, Stand 2026-09-18.** Die Siedlungsschicht ist die aufwendigste
des Editors (Eignungsfeld, Wegenetz, Grundstuecke, Strassenverkehr —
Abschnitte 1–5). Bisher war nirgends festgeschrieben, was das SPIEL davon
tatsaechlich braucht. Das ist jetzt entschieden: es gibt eine feste
Feldliste, und alles, was nicht darin steht, ist Innenleben des Editors und
darf sich aendern, ohne das Spiel zu brechen.

> *Naht: die eine Stelle, an der das Spiel Siedlungsdaten abholt — mit genau
> diesen Feldern und nicht mehr. Dieselbe Idee wie in
> `docs/NACHTBETRIEB.md` §8 fuer `sperre.pruefe()`.*

### 6.1 Die sechs Felder

| # | Feld | Typ / Einheit | Traeger heute | Status |
|---|---|---|---|---|
| 1 | Kontur der Stadtgrenze | Polygon je Siedlung: Liste von (x,y)-Punkten in Pixelkoordinaten des aktuellen LOD-Rasters | `settlement.city_boundary` liefert bisher nur `city_mask` (Rastermaske: (H,W) int, Siedlungs-ID pro Pixel, -1 = ausserhalb jeder Stadt) | **fehlt als Naht-Feld** — siehe 6.2 |
| 2 | Anschlusspunkte der Wege | Liste von (x,y)-Punkten je Siedlung: dort, wo eine Strecke aus `settlement.pathfinding` (`roads`/`sea_roads`) die Stadtgrenze schneidet | nirgends berechnet | **fehlt** — siehe 6.2 |
| 3 | Stadtgroesse | int, Einheit "Anzahl Haeuser" (15–50, siehe §1) | `settlement.settlements` → `settlement_list[i].house_count` | geliefert |
| 4 | Stadttyp | str, einer aus `"bergdorf" \| "marktstadt" \| "agrarstadt" \| "sonstige"` | `settlement.settlements` → `settlement_list[i].settlement_type` | geliefert |
| 5 | Rang | str, einer aus `"dorf" \| "siedlung" \| "stadt"` | `settlement.settlements` → `settlement_list[i].rank` | geliefert |
| 6 | Kultur | str, Name aus `core.terrain_weltkarte` REGIONEN (`volk`-Feld, z.B. "Kelten") | `settlement.settlements` → `settlement_list[i].culture` | geliefert |

Schluesselfeld fuer alle sechs (nicht Teil der urspruenglichen Liste aus dem
Ticket, aber noetig, um Kontur/Anschlusspunkte spaeter eindeutig einer
Siedlung zuzuordnen): `settlement_list[i].location_id` (int, kartenweit
eindeutig) — existiert schon und wird von 6.2 nur wiederverwendet, nicht neu
erfunden.

### 6.2 Was fehlt, als eigene Tickets

**#72 — Kontur als Polygon.** `settlement.city_boundary` muss zusaetzlich zu
`city_mask` ein `city_boundary_polygons: Dict[int, List[(x,y)]]` liefern (ein
Polygon je Siedlung, Schluessel = `location_id`). Der Baustein dafuer
existiert bereits, nur an der falschen Stelle:
`PlotPhysicsSystem._build_city_boundary_polygons()`
(core/settlement_generator.py) macht per Marching-Squares genau das — Kontur
von `city_mask == settlement_id` als Shapely-Polygon —, aber nur intern fuer
das Grundstuecks-System, nicht als Ausgabe des Calculator-Knotens. #72 muss
diese Extraktion (oder eine leichtgewichtigere eigene Kopie davon) in
`_calc_city_boundary()` einhaengen.

**#73 — Anschlusspunkte der Wege.** Sobald #72 die Polygon-Kontur liefert,
kann jede Strecke aus `roads`/`sea_roads` gegen das Polygon der jeweiligen
Zielsiedlung geschnitten werden (der Punkt, an dem die Strecke die Kontur
durchstoesst). Ohne #72 gibt es keine Kontur zum Schneiden — #73 ist deshalb
"Blocked by #72".

### 6.3 Was ausdruecklich NICHT zur Naht gehoert

Das Spiel erzeugt aus den sechs Feldern oben sein eigenes Innenleben. Der
Editor liefert dafuer die Zutaten, nicht das fertige Ergebnis:

- **Parzellen** (`plot_map`, `house_parcel_map`, `plots`) — die
  Grundstuecksteilung entsteht im Spiel aus Stadtgroesse und Kontur.
- **Strassennetz INNERHALB der Stadt** (`street_mask`, `plot_nodes`,
  `plot_edges`) — nur das Netz ZWISCHEN Staedten (Abschnitt 4) gehoert zur
  Naht, nicht die inneren Gassen.
- **Haeuserformen** — kommt im heutigen Code gar nicht vor; ausdruecklich
  Aufgabe des Spiels.
- **Gewerbe** — ebenfalls nicht Teil des heutigen Codes; das Spiel leitet es
  aus Stadttyp/Rang/Kultur ab.

Aenderungen an diesen vier Punkten (z.B. ein neues Parzellen-Layout, ein
anderer Strassenverkehrs-Algorithmus) duerfen sich frei aendern, ohne dass
sich irgendetwas auf der Spiel-Seite anpassen muss — genau das ist der Zweck
einer Naht.

### 6.4 Wer das prueft

`tests/smoke_test_siedlungsnaht_felder.py` faehrt die echte Pipeline bis
`settlement.city_boundary`/`settlement.pathfinding` und zaehlt die sechs
Felder nach: die vier gelieferten pruefen GRUEN, die zwei fehlenden schlagen
ABSICHTLICH und NAMENTLICH fehl (kein Crash, sondern "dieses Feld fehlt noch,
Ticket #72/#73") — solange, bis diese beiden Tickets geschlossen sind. Danach
werden die zwei Platzhalter-Fehlschlaege durch echte Pruefungen ersetzt.
