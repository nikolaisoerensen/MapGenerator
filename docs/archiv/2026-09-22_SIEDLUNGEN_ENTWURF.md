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


## 6. Die Siedlungsnaht — was das Spiel vom Editor abholt

Stand 2026-09-21 (Ticket #35). **Naht** heisst hier: die eine Stelle, an der
das Spiel Siedlungsdaten vom Editor abholt — mit genau den Feldern, die unten
stehen, und nicht mehr. Alles andere, was der Editor intern rechnet (Parzellen,
Strassenraster, Hausformen, Gewerbe), ist Innenleben des Editors und darf sich
frei aendern, ohne dass das Spiel etwas davon merkt oder bricht. Der Editor
liefert nur die **Aussenform** einer Stadt, das Spiel baut das Innere selbst.

Diese Entscheidung ist bewusst eng: **vier Dinge**, keins davon ein
Strassennetz oder eine Hausliste.

1. die Kontur der Stadtgrenze
2. die Anschlusspunkte der Wege — wo Wege die Stadtgrenze schneiden
3. Stadtgroesse und Stadttyp
4. Rang und Kultur

Dazu kommt zwingend eine **Kennung samt Ort** je Stadt (`city_id`,
`city_center`) — ohne sie waere keines der vier Felder oben einer bestimmten
Stadt zuzuordnen. Das ist keine fuenfte Kategorie, sondern die Vorbedingung
dafuer, dass die anderen vier ueberhaupt lesbar sind.

### 6.1 Die Feldliste

Koordinaten (`city_center`, alle Punkte in `city_boundary_polygons` und
`road_entry_points`) sind **Pixelkoordinaten im generierten Raster**
(`map_size` × `map_size`, z.B. 512×512). Die Umrechnung in Meter laeuft ueber
`map_distance_km / map_size` (Standard 15 km, siehe `docs/SPEZIFIKATION.md`);
das ist bewusst nicht Teil der Naht selbst — der Editor liefert Pixel, das
Spiel rechnet um, wenn es seine eigene Kartenaufloesung kennt.

| Feld | Typ | Einheit | Geliefert? | Fundstelle |
|---|---|---|---|---|
| `city_id` | int | — (eindeutige Kennung) | **JA** | `core/settlement_generator.py:441` (`Location.location_id`), gesetzt in `calculate_settlements()` Zeile 5639-5643 |
| `city_center` | (float, float) | Pixel im `map_size`-Raster | **JA** | `Location.x`/`Location.y` (Zeile 442-443), gesetzt Zeile 5640 |
| `city_boundary_polygons` | Liste von Polygonen, je Polygon eine Liste von (x,y)-Punkten (Pixel) | Pixel | **JA** (seit Ticket #72) | `core/settlement_generator.py:2443` (`_polygonize_settlement_mask()`, Marching-Squares ueber `city_mask`), ausgegeben von `_calc_city_boundary()` Zeile 5438-5446 als Ausgabe `city_boundary_polygons` des Knotens `settlement.city_boundary`. Dieselbe Funktion bedient auch `PlotPhysicsSystem._build_city_boundary_polygons()` (Zeile 3075), damit Innenleben und Naht garantiert dieselbe Kontur sehen. |
| `road_entry_points` | Liste von (x,y)-Punkten je Stadt (Pixel), an denen eine Wegverbindung `city_boundary_polygons` schneidet | Pixel | **JA** (seit Ticket #73) | `core/settlement_generator.py:2595` (`_wege_anschlusspunkte()`), gerufen in `_calc_pathfinding()` Zeile 5495-5501, ausgegeben Zeile 5505 als Ausgabe `road_entry_points` des Knotens `settlement.pathfinding`. |
| `city_size` (als `house_count`) | int | Haeuser, Bereich 15-50 | **JA** | `Location.house_count` (Zeile 450), gesetzt Zeile 5696 nach §1-Rangspanne |
| `city_size` (als `radius`, ergaenzend) | float | Pixel | **JA** | `Location.radius` (Zeile 445), gesetzt Zeile 5700 aus `house_count` abgeleitet |
| `city_type` | str, einer von `bergdorf`/`marktstadt`/`agrarstadt`/`sonstige` | — | **JA** | `Location.settlement_type` (Zeile 456), zugewiesen Zeile 5485-5500 (siehe auch `docs/OFFENE_PUNKTE.md` 5.16) |
| `rank` | str, einer von `dorf`/`siedlung`/`stadt` | — | **JA** | `Location.rank` (Zeile 449), gesetzt Zeile 5695 |
| `culture` | str, Name aus `core/terrain_weltkarte.py` REGIONEN-`volk`-Feld | — | **JA** | `Location.culture` (Zeile 448), gesetzt Zeile 5642 |

**Befund in einem Satz:** seit den Tickets #72 und #73 liefert der Editor
alle acht Felder. Sechs (`city_id`, `city_center`, `city_size` in beiden
Auspraegungen, `city_type`, `rank`, `culture`) kommen direkt aus der
`Location`-Dataclass; `city_boundary_polygons` liefert der Knoten
`settlement.city_boundary`, `road_entry_points` der Knoten
`settlement.pathfinding`. Dieses Ticket (#35) hat nur festgelegt, WIE die
Felder heissen und WAS sie tragen — gebaut wurden die beiden letzten in
#72 und #73.

### 6.2 Was NICHT zur Naht gehoert

Diese Ausgaben existieren im Editor, sind aber ausdruecklich **Innenleben**
und Teil der Naht — sie duerfen sich jederzeit aendern, ohne das Spiel zu
brechen, weil das Spiel sie gar nicht lesen soll:

| Ausgabe | Warum sie draussen bleibt |
|---|---|
| `plot_map`, `house_parcel_map`, `street_mask` | Parzellen- und Strassenraster INNERHALB einer Stadt — laut Ticketvorgabe entsteht das im Spiel aus `city_boundary_polygons` + `city_size` + `city_type`, nicht im Editor |
| `plots`, `plot_nodes`, `plot_cores`, `plot_edges` | Das Grundstücks-/Wege-Innenmodell des Editors (Voronoi-Zellen, Federphysik) — reine Rechenhilfe fuer die Editor-eigene Darstellung |
| `potential_field` | Kraftfeld der Plot-Physik-Simulation, rein intern |
| `voronoi_cell_map` | Landschafts-Plot-Zell-ID pro Pixel ausserhalb von Staedten — Wildnis-Binnenstruktur, nicht Stadtform |
| `wilderness_polygons` | Kontur der Wildnisflaechen (das Gegenteil von `city_boundary_polygons`) — nicht Teil dieses Tickets |

**Ausdruecklich offen gelassen, nicht Teil DIESES Tickets:** `landmark_list`,
`roadsite_list` und die vollen Weg-Polylinien (`roads`, `sea_roads`,
`landmark_roads`) sind eigene Ausgaben desselben Generators, aber die
Ticketvorgabe fuer #35 nennt nur die Siedlungsflaeche selbst (Kontur,
Wege-Anschluss, Groesse/Typ/Rang/Kultur). Ob und wie Landmarks/Roadsites/das
volle Wegenetz Teil einer (ggf. eigenen) Naht werden, ist hier nicht
entschieden.
