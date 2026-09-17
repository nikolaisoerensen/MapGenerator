# Plan: Regionenwelt im Labor

Stand 2026-08-04, zweite Fassung. Die erste ging von einer 240-km-Welt mit
Zoomfenstern aus; beides ist nach den Festlegungen des Nutzers hinfällig.
Ergänzt `docs/archiv/2026-07-29_SPEZIFIKATION.md`, ersetzt sie nicht.

> **UMGESETZT — dieses Dokument ist ab 2026-08-12 ein Planungsarchiv, keine
> Vorgabe mehr.** Der hier geplante Umbau ist gebaut und steht in
> `core/terrain_weltkarte.py`. Wer wissen will, wie die Welt HEUTE aussieht,
> liest den Code oder `docs/OFFENE_PUNKTE.md` Abschnitt 3 — nicht mehr diese
> Datei. **Die Zahlen unten stimmen nicht mehr** (siehe Kasten in §1); sie
> bleiben stehen, weil die BEGRÜNDUNGEN daneben weiterhin gelten und erklären,
> warum die Welt so gebaut ist, wie sie ist.

---

## 1. Die Welt in Zahlen

> **VERALTET seit der Umsetzung.** Die tatsächlichen Werte stehen in
> `core/terrain_weltkarte.py` und lauten:
>
>     WELT_KM       = 21.3        (geplant war 15)
>     KONTINENT_KM  = 13.11       (nur als Flächenmaß, geplant war 12)
>
> Die 21.3 km entstanden, als die Regionsflächen (`flaeche_soll`) und der
> Wasseranteil je Region ausgeeicht wurden — die Welt musste wachsen, damit
> neun Regionen plus Meer hineinpassen. Das Verhältnis Region : Kontinent :
> Welt ist dabei erhalten geblieben, nur der Maßstab ist ein anderer.

    Region        4 x 4 km        VERALTET
    Kontinent     12 x 12 km      VERALTET (3 x 3 Regionen)
    Welt          15 x 15 km      VERALTET (1.5 km Wasser ringsum)

**Das ist ein Prototypmaßstab, keine echte Geografie.** Der Nutzer dazu: *"das
sind dann natürlich keine echten relationen, deshalb muss das fjordland zB nur
ein hauptfjord sein"*. Eine Region zeigt also **ein Exemplar** ihrer Sorte —
ein Fjord, ein Mittelgebirgstal, eine Bucht — nicht eine ganze Landschaft.

### Auflösung

| px | m/px | Region in px | Speicher je Schicht |
|---|---|---|---|
| 1024 | 14.6 | 273 | 4 MB |
| 2048 | 7.3 | 546 | 17 MB |

**1024 zum Arbeiten, 2048 für die Abnahme.** Bei 1024 kommen auf ein Tal rund
40 px — genau die Grenze, unter der laut Smoke-Test das Querprofil zerfällt.
2048 gibt Luft.

Zum Vergleich: eine ganze Welt in dieser Auflösung ist **kleiner als die
heutige Spielkarte an Daten**. Rauschen auf der GPU 0.06 s, Distanz­transformation
0.44 s bei 2048. Rechenzeit ist hier kein Thema — das war sie erst bei der
240-km-Fassung.

### Was das für die Pyramide bedeutet

**Die Zoomfenster entfallen.** Es gibt eine Karte. Damit fallen weg:
`erbe_bilden`, Ein-/Auslauf zwischen Stufen, Randsaum, die Kreuzungsmessung
zwischen Fenstern — rund die Hälfte der gestrigen Werkstatt.

**Was bleibt, ist der Kern:** Makro → Meso → Mikro als **Rechenstufen auf
derselben Karte**. Erst die großen Ströme, dann die Nebenflüsse, dann die
Bäche; jede Stufe erbt die vorige, geerbte Ketten werden erzwungen. Genau das
hat gestern die abgerissenen Flüsse behoben und wird hier gebraucht.

> Alles bleibt in **Metern** und mit Welt­größe als Parameter geschrieben, damit
> ein späterer Wechsel auf 60 oder 240 km kein Umbau ist.

---

## 2. Die neun Regionen

Anordnung: Spalte = West → Ost, Zeile = Nord → Süd.

|  | West | Mitte | Ost |
|---|---|---|---|
| **Nord** | Skerrheim *(Wikinger)* | Hügelland *(Kelten)* | Morobora *(Slawen)* |
| **Mitte** | Atlantikküste *(Franken)* | Nebelrode *(Franken)* | Nevadin |
| **Süd** | Samarcia/Trockenland *(Andalus)* | Macchia *(Italien)* | Thalassia *(Phönizier)* |

### Höhen sind auf 4 km umgerechnet, nicht abgeschrieben

Echte Alpen haben 2500 m Relief auf 10 km. Dieselbe Zahl auf 4 km wäre eine
Wand mit 60° Durchschnittshang. Übernommen wird deshalb das **Verhältnis von
Relief zu Breite**, nicht der Meterwert:

| # | Region | Basis | Gipfel | Relief | Charakter |
|---|---|---|---|---|---|
| 1 | Skerrheim | −30 | 700 | 730 | **ein** Hauptfjord, Hochfläche, steile Wände |
| 2 | Hügelland | 20 | 220 | 200 | sanfte Wellen, breite Sohlen, dichtes Bachnetz |
| 3 | Morobora | 60 | 260 | 200 | flach, weite Mulden, Seen, träge Mäander |
| 4 | Atlantikküste | −40 | 120 | 160 | Küstenebene mit Ästuar, Kliff im Norden |
| 5 | Nebelrode | 150 | 550 | 400 | dichte dendritische Zertalung |
| 6 | Nevadin | 400 | 1400 | 1000 | Trogtäler, scharfe Grate |
| 7 | Samarcia/Trockenland | 80 | 380 | 300 | Trockentäler, weite Flächen, wenig Netz |
| 8 | Macchia | −50 | 350 | 400 | Küstengebirge direkt am Meer, kurze steile Läufe |
| 9 | Thalassia | −80 | 250 | 330 | Archipel, viel Wasser, kleine steile Inseln |

Kegelkarst und Tafelland sind gestrichen — *"macht für mich keinen sinn in
europa"*.

---

## 3. Wie die Regionen ineinander übergehen

**Kein Regionen-Mosaik, sondern ein Parameterfeld.** Jeder Regler wird als
3×3-Gitter angegeben und auf volle Auflösung **kubisch hochgerechnet**
(`ndimage.zoom`, Ordnung 3). Damit entsteht die Überblendung von selbst und
ohne Nahtlogik.

    basis_m       3x3  ->  1024x1024   weich
    gipfel_m      3x3  ->  1024x1024   weich
    talabstand_m  3x3  ->  1024x1024   weich
    ... je Regler dasselbe

Zwei Zutaten dazu:

* **Verzerrter Übergang.** Die Gitterkoordinate wird vor dem Hochrechnen mit
  Rauschen verzogen, sonst verlaufen die Grenzen als saubere Kreuze. Stärke als
  Regler.
* **Übergangsbreite** als Regler: von hart (Grenze in 200 m) bis sehr weich
  (über eine halbe Region).

### Das Meer entsteht aus demselben Feld

Kein eigener Inselgradient mehr. Die Küste ist dort, wo die überblendete
**Basishöhe plus Relief unter 0** fällt — und weil Atlantikküste, Macchia
und Thalassia negative Basishöhen haben, entstehen Buchten und
Archipel an genau den richtigen Stellen. Ringsum zieht ein Randabfall die
äußeren 1.5 km auf −200 m.

Das ist sachlich richtiger als der Kastengradient: die Küstenform folgt den
Regionen statt einer aufgesetzten Form.

---

## 4. Flüsse

### Bis −50 m, dann abgeschnitten

Nach Vorgabe des Nutzers laufen die Läufe **bis zur −50-m-Tiefenlinie** weiter
und werden erst beim Zeichnen und Eingraben an der 0-Linie gekappt. Damit
mündet ein Fluss sichtbar ins Meer statt an der Küste zu enden, und die
Mündungsrichtung stimmt von selbst.

Der Auslauf-Behelf von gestern (gerade Strecke zum nächsten Meerpixel) entfällt
damit ersatzlos.

### Drei Rechenstufen auf einer Karte

| Stufe | Knotenabstand | erzeugt |
|---|---|---|
| Makro | ~1200 m | die Hauptströme zur Küste |
| Meso | ~400 m | Nebenflüsse, münden in die Ströme |
| Mikro | ~130 m | Bäche |

Jede Stufe übernimmt die Knoten der vorigen an derselben Stelle, ihre
Verbindungen werden durch den feineren Graphen geroutet und auf 12 % der Kosten
gesetzt, und ein Knoten bekommt **höchstens einen Kettenelternknoten** — trifft
eine Kette auf eine bestehende, mündet sie dort. Alles drei ist gestern
gemessen und behoben worden.

### Offen für später

Ob die so erzeugten Flüsse **bleiben** oder am Ende durch eine Erosionsrechnung
mit echten Wassermengen ersetzt werden (dann mit den wahren Flussgrößen), ist
bewusst offen. Die Regionen brauchen die Entscheidung nicht.

---

## 5. Speichern und Reproduzierbarkeit

Vorgabe: *"alles muss aus dem map seed und den slidern reproduzierbar sein und
am ende möglichst komprimiert aber abrufbar gespeichert werden."*

Daraus folgt eine Trennung, die von Anfang an eingehalten wird:

| Was | Woraus | Gespeichert |
|---|---|---|
| Gelände, Regionenfeld | Seed + Regler, rein deterministisch | **nichts** — jederzeit neu rechenbar |
| Flussnetz (Knoten, Kanten, Generation, Einzugsgebiet) | Seed + Regler | **das Netz**, wenige tausend Zahlen |
| Heightmap, Biome, Wasser | daraus abgeleitet | nichts |

Das Netz ist der einzige Teil, dessen Neuberechnung teuer ist *und* dessen
Ergebnis nicht in einer Formel steht. Es wird als Knotenliste abgelegt, nicht
als Bild — bei 1024 px sind das rund 3000 Knoten gegen 4 MB Rasterbild.

**Prüfbar gemacht wird das durch eine Zusicherung, nicht durch Zusage:** zweimal
rechnen mit gleichem Seed und gleichen Reglern muss **bitgleich** sein.

---

## 6. Der Bauplan

Jede Stufe ist einzeln lauffähig und hat ein Abnahmemaß.

### Stufe A — Regionenfeld *(Grundlage)*

`tools/regionen_lab.py` wird zum Fenster mit Reglern, wie die
Flussnetz-Werkstatt. Neun Parametersätze, kubische Überblendung, verzerrte
Grenzen, Randabfall ins Meer. **Noch ohne Flüsse.**

*Abnahme:* Landanteil, Küstenlänge und Basishöhe je Region messbar; Bild der
neun Regionen mit eingezeichnetem Gitter; keine sichtbare Naht.

### Stufe B — Flussnetz in drei Stufen

Der Kern der gestrigen Werkstatt, entkernt um die Fensterlogik. Auslässe an der
−50-m-Linie.

*Abnahme:* null Generationsbrüche, jeder Lauf endet unter 0 m, Einzugsgebiet
wächst flussabwärts monoton.

### Stufe C — Täler eingraben

`taeler_eingraben`, aber Breite und Tiefe **je Region** aus dem Parameterfeld —
im Nevadin tiefe Tröge, im Hügelland breite flache Sohlen.

*Abnahme:* Überhöhung über dem lokalen Tiefpunkt sinkt; größter Nachbarsprung
und p99.9 protokolliert.

### Stufe D — Die neun eichen

Erst hier wird an den Zahlen gedreht, Region für Region, gegen den
Fingerabdruck aus Abschnitt 7.

*Abnahme:* alle neun im Sollbereich **und** vom Nutzer am Bild abgenommen.

### Stufe E — Übergänge

Übergangsbreite und Grenzverzerrung eichen.

*Abnahme:* Nahtprüfung (Abschnitt 7) grün.

---

## 7. Wie geprüft wird

### Fingerabdruck je Region

„Sieht gut aus" ist nicht messbar, der Charakter einer Landschaft schon. Fünf
Zahlen, je Region mit Sollbereich:

| Maß | trennt |
|---|---|
| Relief (m) | Gebirge von Flachland |
| Talabstand (m) | dicht zergliedert von weiträumig |
| Entwässerungsanteil (%) | Netzdichte |
| Median-Hangneigung (°) | Grundcharakter |
| Anteil über 30° (%) | schroff von sanft |

Eine Region ist fertig, wenn ihr Fingerabdruck im Bereich liegt **und** das Bild
abgenommen ist. Damit ist eine spätere Verschlechterung messbar statt nur
gefühlt.

### Vier Zusicherungen für alle

1. **Reproduzierbarkeit** — zweimal gerechnet, bitgleich
2. **Flusskontinuität** — null Generationsbrüche, jeder Lauf mündet unter 0 m
3. **GPU/CPU-Parität** — r = 1.000000, seit heute erreicht, muss bleiben
4. **Nahtprüfung** — an einer Regionsgrenze darf der Höhengradient keinen
   Sprung zeigen, der größer ist als der stärkste Gradient **innerhalb** der
   beiden angrenzenden Regionen

Zusicherung 4 ist die einzige neue und die wichtigste für Stufe E: sie sagt
genau das, was der Nutzer verlangt hat — *"so dass die regionen smooth
ineinanderlaufen"* — ohne auf den Augenschein angewiesen zu sein.

---

## 8. Danach, nicht jetzt

Nach Abnahme der neun Regionen folgt laut Nutzer die **Reintegration ins
Hauptprogramm als Erweiterung „Worldmap"**, und zwar in dieser Reihenfolge:

1. Biome, Wasser und Wetter laufen auf der **ganzen Welt** statt je Karte —
   sonst sehen die Biome an den Rändern falsch aus
2. Orte setzen
3. Straßen und Netze

Das ist der Punkt, an dem die 24 Knoten hinter Terrain angefasst werden müssen,
und er bekommt einen eigenen Plan. Er beginnt nicht, bevor die Regionen stehen.

**Negative Höhen** (Meer auf der Karte) gehören zu diesem Schritt: das Programm
muss dafür bereinigt werden, damit keine Fehler entstehen — *"aber negative
höhen sind dann ganz einfach nur unter wasser"*.

---

# Teil II: Portierung ins Hauptprogramm

Stand 2026-08-05, nach Abnahme der Regionenwelt im Labor.

## Der schnellste Weg zu 3D

Ziel des Nutzers: *"den rest will ich erstmal in 3D sehen"*. Dafür genügt
**eine** Sache — die Heightmap der Regionenwelt muss dort ankommen, wo der
3D-Aufbau sie abholt. Alles andere kann warten.

Geprüft: `gui/widgets/map_display_3d.py:1065` rechnet
`pos_y = heightmap * terrain_height_scale`, also **linear** um die Mitte.
Negative Höhen rendern damit unterhalb der Wasserlinie statt abzustürzen — das
größte vermutete Risiko besteht nicht.

### Stufe P1 — Gelände sichtbar *(ein halber Tag)*

1. `tools/regionen_welt.py` → `core/terrain_weltkarte.py`, unverändert bis auf
   die Importe.
2. Schalter `WELTKARTE_AKTIV` in `value_default.py`. Steht er, liefert
   `_calc_redistribution` die Weltkarte statt Noise + Filter + Flussnetz.
3. `map_distance_km` wird auf `WELT_KM` gesetzt, `map_size` ist die Auflösung.

*Abnahme:* die Insel erscheint in der 2D-Anzeige und im 3D-Mesh. Der Nutzer
sagt, ob ihm die Landschaft gefällt. **Vor dieser Abnahme beginnt nichts
weiter.**

### Stufe P2 — Flüsse und Täler *(ein Tag)*

`tools/regionen_fluesse.py` → `core/terrain_weltfluesse.py`, angeschlossen in
derselben Weiche. `river_mask`, `river_order` und neu `river_generation` als
Outputs.

*Abnahme:* null Generationsbrüche, jeder Lauf mündet unter 0 m.

## Was danach kommt, und in welcher Reihenfolge

Der Nutzer hat die Reihenfolge vorgegeben: erst Biome/Wasser/Wetter **auf der
ganzen Welt**, dann Orte, dann Straßen.

### Die offenen Schnittstellenfragen

**S1: Negative Höhen in 24 Knoten.** Geology, Weather, Water, Biome und
Settlement haben nie eine Höhe unter 0 gesehen. Der Nutzer: *"hier müssen wir
dann einmal das programm bereinigen so das es keine fehler dadurch gibt. aber
negative höhen sind dann ganz einfach nur unter wasser."*

> Vorgehen: `smoke_test_pipeline_outputs.py` mit der Weltkarte als Eingang
> fahren und die Befunde einzeln abarbeiten. Der Test existiert und meldet
> heute 7 Befunde — jede neue Zeile ist ein Schaden aus den negativen Höhen.

**S2: Auflösung.** 21.3 km auf 1024 px sind 20.8 m/px. Die Pipeline rechnet
heute mit 256 px. Bei 1024 px sind es 16× mehr Pixel je Knoten, bei 73 Outputs
also rund 300 MB.

> Vorgehen: mit 512 px anfangen (41.6 m/px, 75 MB) und messen, bevor
> hochgegangen wird. Kein Kachelbetrieb, solange es nicht nötig ist.

**S3: Was wird gespeichert.** Gelände und Regionenfeld sind reine Funktionen
von Seed und Reglern — nichts speichern. Nur das **Flussnetz** als Knotenliste
(rund 4600 Knoten), weil seine Neuberechnung teuer ist und nicht in einer
Formel steht.

> Zusicherung statt Zusage: zweimal rechnen muss bitgleich sein. Der Test dafür
> steht schon (`smoke_test_regionen_welt.py`, Zusicherung 1).

**S4: Der Erosionsfilter.** Er läuft heute in `_calc_redistribution` und ist
gegenüber der Weltkarte neutral (rein lokal je Pixel).

> Vorgehen: bleibt, wo er ist, wird aber in der Weiche NACH dem Weltfeld und
> VOR dem Flussnetz angewandt — dieselbe Reihenfolge wie heute.

**S5: Die alten Terrain-Regler.** `feature_size_m`, `amplitude`,
`redistribute_power` verlieren bei aktiver Weltkarte ihre Bedeutung — die
Regionen bringen ihre eigenen mit.

> Vorgehen: bei aktivem Schalter ausgrauen, nicht löschen. Der alte Pfad muss
> lauffähig bleiben, bis die Weltkarte abgenommen ist.

## Reihenfolge

    P1  Gelände in 2D und 3D sichtbar          <- zuerst, Abnahme durch Nutzer
    P2  Flüsse und Täler
    P3  Negative Höhen bereinigen (S1)
    P4  Biome / Wasser / Wetter auf der Welt
    P5  Orte
    P6  Straßen und Netze

P3 ist der Punkt, an dem die 24 Knoten hinter Terrain angefasst werden. Er
beginnt nicht, bevor P1 abgenommen ist — sonst wird an einer Landschaft
repariert, die vielleicht noch geändert wird.

---

## P3 abgeschlossen: negative Höhen richten keinen Schaden an

Ergebnis vom 2026-08-05. **Es gab nichts zu reparieren** — und das ist gemessen,
nicht vermutet.

### Wie das festgestellt wurde

Dieselbe Welt zweimal durch alle 38 Knoten: einmal wie sie ist, einmal mit
`heightmap = max(heightmap, 0)`. Was sich zwischen beiden Läufen unterscheidet,
liegt an den negativen Höhen; alles andere ist Altlast.

**Unterschied: genau einer** — `settlement.roadsites/roadsite_list` war mit Meer
leer. Nachgegangen: bei 64 px deckt ein Pixel 333 m ab, die Siedlungen liegen
dann so dicht, dass jede Straße kürzer als drei Punkte ist, und
`calculate_roadsites()` überspringt solche Straßen. Bei 128 px und darüber
erscheint die Liste. **Ein Auflösungsartefakt des Tests, kein Schaden aus dem
Meer.** Der Pipeline-Test läuft deshalb jetzt mit 128 px statt 64.

### Was dabei doch gefunden wurde

Ein echter Fehler, aber ein anderer: **die Siedlungsknoten waren nicht
reproduzierbar.** `random.seed(map_seed)` steht einmal im Konstruktor, aber
`random` ist Modulzustand — bis ein später Knoten an die Reihe kommt, hängt er
davon ab, wieviele Zufallszahlen die Knoten davor gezogen haben, und das
unterscheidet sich zwischen GPU- und CPU-Pfad.

Behoben durch `_knoten_zufall(name)`: ein eigener Generator je Knoten, abgeleitet
aus `map_seed` und dem Knotennamen. Er hängt nicht mehr an der
Ausführungsgeschichte. Betroffen waren Siedlungsplatzierung, Roadsites und
Landmarks.

> Beim Einbau fast ein zweiter Fehler: der Generator stand zuerst INNERHALB der
> Siedlungsschleife. Jede Siedlung hätte dieselbe Zufallszahl bekommen und alle
> lägen auf demselben Punkt.

### Was bleibt und warum es nicht P3 ist

Neun Befunde im Pipeline-Test, alle älter als die Weltkarte:

| Befund | Art |
|---|---|
| `erosion.*` (7 Outputs) | **nicht mehr erwartet** — galt, solange `EROSION_AKTIV = False` war; seit 27.08.2026 steht der Schalter auf `True` (`gui/config/value_default.py:1088`), Nullkarten sind hier also ein offener Befund |
| `settlement.city_boundary/city_cost_map` | `np.inf` als Marke für "unerreichbar", kein NaN |
| `geology.intrusions/height_delta`, `settlement.plot_nodes/plots` | leer, Altlast |
| `lake_map`, `climate_classification`, `evaporation_map`, `roadsite_list` | **Empfindlichkeit**, siehe unten |

Die vier "Pfade verschieden" sind keine echten Paritätsverletzungen mehr. GPU-
und CPU-Rauschen stimmen seit dem Shader-Fix auf **7.65e-05** überein — das ist
float32-Rundung, nicht Uneinigkeit. Eine so kleine Differenz kippt aber eine
diskrete Entscheidung: die Siedlung liegt ein Pixel weiter, die Straße hat zwei
statt drei Punkte, die Liste ist leer.

> **Daraus folgt eine Aufgabe, aber keine für P3:** Knoten, deren Ausgabe an
> einer Schwelle hängt, sind gegen Rundung nicht robust. Das gehört behandelt,
> wenn diese Knoten selbst drankommen (P4/P5) - nicht vorher, und nicht durch
> Nachjustieren von Schwellwerten, bis es zufällig passt.
