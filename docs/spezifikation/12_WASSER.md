# 12 — Flüsse, Erosion und Seen

Normativ für die Wasserverteilung auf der Weltkarte: wie ein Knoten des Flussnetzes sein Wasser
bekommt, wie die Hangausrichtung den Niederschlag moduliert, welche Regionen einen garantierten
Hauptstrom haben, welche Talform sie eingraben, und wie die See in Seegrade, Zieltiefen und Seeeis
gegliedert ist. **Nicht hier:** Klimavorgaben und Biome (→ `13_KLIMA_UND_BIOME.md`), Küstenlinie und
Küsten-Archetypen (→ `11_GELAENDE.md`).

## 1. Wasserverteilung — Niederschlag ist das Knotengewicht

Ein Knoten des Flussnetzes trägt **seine Wassermenge** bei, nicht seine Existenz. Das Gewicht ist
der Niederschlag am Knoten, bezogen auf den **Mittelwert über Land** — nicht über die ganze Karte,
denn über See fällt zwar Regen, der aber keinen Fluss speist.

| Größe | Festlegung | Ort |
|---|---|---|
| Gewichtsfeld | `niederschlag_mm / mittel(niederschlag_mm über H > 0)` | `core/terrain_weltfluesse.py:463-469` |
| Untergrenze je Knoten | `WASSER_MINDEST = 0.15` | `core/terrain_weltfluesse.py:53` |
| Akkumulation | flussabwärts entlang `eltern`, additiv | `core/terrain_weltfluesse.py:411-419` |
| Durchreichung | `flussnetz(..., niederschlag_mm=felder["niederschlag_mm"])` | `core/terrain_generator.py:2062-2077` |

Die Untergrenze verhindert, dass eine sehr trockene Region Knoten mit Gewicht nahe null bekommt und
ihre Läufe ganz verschwinden. **Ohne Feld gilt `flaeche = np.ones(n)`** — kein stiller Rückfall,
sondern die Zusicherung, dass Werkzeuge und Tests ohne Regionsdaten unverändert weiterlaufen
(`core/terrain_weltfluesse.py:414-415`). **Normierung ist global, nicht je Region.** Ein großer Fluss
ist ein großer Fluss, unabhängig von der Region; eine regionsweise Normierung
(`gebiet = fl / fl.max()`) ist ausdrücklich zurückgezogen, sie hätte jedem Regionshauptfluss ein
volles Tal gegeben und damit genau den Mengenunterschied verwischt, um den es geht. **Verdunstung
wird nicht abgezogen** — Abfluss ist im Modell der Niederschlag selbst. **Umgesetzt**
(`core/terrain_weltfluesse.py:405-419`, `core/terrain_generator.py:2074`). Verdunstung (1.3)
**offen** — im Flussgewicht kommt keine vor; `core/water_generator.py:2600ff.` rechnet Verdunstung
nur im getrennten Wassersystem.

*Herkunft: docs/archiv/2026-08-24_FLUESSE_UND_WASSER.md (Stand 2026-08-24), Block 1 „Wasser richtig verteilen".*

## 2. Hangausrichtung — Südhänge trocknen aus

`niederschlag_mm` ist **kein reiner Regionswert mehr**, sondern ein Feld: am Ende von `weltfeld()`
moduliert `_hangfeuchte()` es nach Exposition. Das Feld geht auch in Wetter und Biome — eine
Wahrheit statt zweier.

**Punktweise aus dem Gradienten**, nicht je Voronoi-Zelle: eine Zelle kann Nord- und Südhänge
zugleich enthalten, der Gradient ist feiner. **Relativ, nicht absolut** — ein Südhang in der
Samarcia bleibt trockener als einer im Skerrheim. **An die Neigung gekoppelt**: die Modulation wächst
linear bis `HANG_VOLL_GRAD = 12.0` Grad und wirkt darüber voll, denn auf einer Ebene gibt es keine
Exposition; ohne diese Kopplung bekäme flaches Land zufällige Feuchteunterschiede aus dem
Rundungsrauschen des Gradienten. **Erst nach der Grundhöhenbildung**, weil die Parameterfelder aus den
Voronoi-Gewichten entstehen und zu dem Zeitpunkt noch kein Gelände existiert. **y wächst nach Süden**
(Zeile 0 ist Norden): ein Südhang hat `dH/dy < 0`.

Formel: `faktor = 1 - hang_trockenheit * suedexposition * wirkung`, geklemmt auf 0.05 bis 3.0
(`core/terrain_weltkarte.py:2920-2938`). `hang_trockenheit` ist der Anteil, um den ein voller Südhang
trockener wird; ein voller Nordhang wird um denselben Betrag feuchter.

| Region | Wert | Region | Wert | Region | Wert |
|---|---:|---|---:|---|---:|
| Skerrheim | 0.10 | Morobora | 0.20 | Thalassia | 0.30 |
| Estrande | 0.12 | Nebelrode | 0.20 | Macchia | 0.35 |
| Clonagh | 0.15 | Nevadin | 0.30 | Samarcia | 0.45 |

Zu wissen: das gemessene Süd-/Nordhang-Verhältnis folgt **nicht allein** diesem Parameter, sondern
auch der Hangneigung der Region. Wer Regionen gegeneinander einstellt, muss das mitrechnen.
**Umgesetzt** — `_hangfeuchte()` in `core/terrain_weltkarte.py:2894`, aufgerufen als letzter Schritt
von `weltfeld()` (`core/terrain_weltkarte.py:2888-2889`); Werte in
`core/daten/regionen.toml:83, 133, 183, 237, 287, 337, 391, 441, 491`.

*Herkunft: docs/archiv/2026-08-24_FLUESSE_UND_WASSER.md (Stand 2026-08-24), Block 1 „Wasser richtig verteilen".*

## 3. Garantierte Hauptströme je Region

Drei Regionen haben eine **Quote**: mit der angegebenen Wahrscheinlichkeit wird ihr größter Lauf auf
ein Sollmaß angehoben, falls er es nicht ohnehin erreicht. Skerrheim 100 % auf 700, Morobora 66 % auf
500, Estrande 66 % auf 500; alle übrigen Regionen haben keine Quote. Das Sollmaß 700 liegt über dem
größten Fluss, den die Karte vor der Niederschlagsgewichtung überhaupt hatte. Die Regel ist eine
**bewusste Setzung, kein Modell**: sie behebt nicht die ungeklärte Ursache dafür, dass das Skerrheim
trotz höchster Wassermenge nicht von selbst führt, sondern überstimmt sie an genau drei Stellen.

Bindend: **aus dem Seed gewürfelt**, nicht aus `random` — dieselbe Karte gibt dasselbe Ergebnis,
unabhängig davon, wie viele Zufallszahlen vorher gezogen wurden. **Der Baum wird nicht umgehängt**:
nur das Gewicht entlang der bestehenden Hauptkette steigt, Geometrie und Kettenlogik bleiben
unberührt, Talbreite und -tiefe folgen. **Nur abwärts, nur die Hauptkette** — Zuflüsse oberhalb
werden nicht mitskaliert, ein großer Strom hat normale Nebenflüsse. **Additiv, nicht
multiplikativ**: mehr Wasser im Oberlauf heißt flussabwärts eine konstante Zugabe, ein Faktor hätte
auch die ohnehin großen Knoten mitvergrößert. Kette und Zuflüsse dürfen **nicht als getrennte
Masken** angewandt werden, sonst wird der Hauptknoten zweimal multipliziert. Nachgelagerte Regionen
wachsen mit, wenn der erzwungene Hauptstrom durch sie hindurch mündet — ein großer Fluss bleibt
groß; ihre eigenen Läufe bleiben unberührt. **Umgesetzt** — `HAUPTSTROM_QUOTE` in
`core/terrain_weltfluesse.py:533-537`, `_hauptstrom_erzwingen()` in
`core/terrain_weltfluesse.py:539`, aufgerufen aus `flussnetz()`
(`core/terrain_weltfluesse.py:506-507`) mit `region_map=felder["regionen"]`
(`core/terrain_generator.py:2077`).

*Herkunft: docs/archiv/2026-08-24_FLUESSE_UND_WASSER.md (Stand 2026-08-24), Block 2 „Grosse Fluesse garantieren".*

## 4. Talformen je Region

`talform` ist der Exponent der Querschnittskurve in `taeler_eingraben()`:
`profil = (1 - exp(-abstand/breite)) ** talform`. Klein heißt, das Profil steigt sofort — schmale
Sohle, steile Flanken, fluvial eingeschnittenes **V-Tal**. Groß heißt, es steigt träge — breite
flache Sohle, glazial ausgeschürftes **U-Tal**.

| Region | Wert | Region | Wert | Region | Wert |
|---|---:|---|---:|---|---:|
| Nevadin | 0.8 | Samarcia | 1.1 | Estrande | 1.5 |
| Macchia | 0.9 | Clonagh | 1.3 | Morobora | 2.0 |
| Thalassia | 0.9 | Nebelrode | 1.3 | Skerrheim | 2.6 |

**Rückfall ist 1.3** für die ganze Karte, wenn das Feld fehlt. **Voronoi-überblendet** wie jeder
Regionsparameter: an einer Regionsgrenze geht die Talform allmählich über, statt zu springen. **Der
GUI-Regler skaliert, er ersetzt nicht** — `river_valley_form` wirkt als Faktor gegen
`TALFORM_REGLER_NEUTRAL = 1.3`, ist bei seinem Vorgabewert neutral und lässt die
Regionsunterschiede vollständig bestehen; Ergebnis geklemmt auf 0.3 bis 6.0. Die Talform **ändert
das Aussehen, nicht die Wassermenge** — unabhängig von Abschnitt 1 bis 3. **Umgesetzt** —
`TALFORM_REGLER_NEUTRAL` in `core/terrain_weltfluesse.py:883`, Auswertung in
`core/terrain_weltfluesse.py:1081-1087`, Werte in
`core/daten/regionen.toml:84, 134, 184, 238, 288, 338, 392, 442, 492`.

*Herkunft: docs/archiv/2026-08-24_FLUESSE_UND_WASSER.md (Stand 2026-08-24), Block 4 „Talformen je Region".*

## 5. Seen als Sammler — nicht gebaut

Zielbild: ein See bündelt alle Zuflüsse und gibt **einen** Ausfluss ab — der Mechanismus, der in
Norwegen die großen Flüsse macht. Heute laufen die Ketten an Seen vorbei. Offen bleibt die Bauform:
Seeausfluss als eigener Knoten im Baum, oder Kantenkosten innerhalb eines Sees auf ~0, sodass
Dijkstra die Bündelung selbst findet. **Für das Fjordland wirkungslos, gemessen:** von elf
Binnenseen über vier Pixel liegt keiner im Skerrheim — deshalb wurde stattdessen die
Hauptstromquote (Abschnitt 3) gebaut. **Offen** — in `core/terrain_weltfluesse.py` gibt es keine
Seebündelung; `flussnetz()` kennt weder Seeflächen noch `seegrad` (Signatur
`core/terrain_weltfluesse.py:440-441`).

*Herkunft: docs/archiv/2026-08-24_FLUESSE_UND_WASSER.md (Stand 2026-08-24), Block 3 „Seen als Sammler".*

## 6. Seen über Meeresniveau — nicht gebaut

**Das Terrain kennt keine Seen über Meeresniveau.** Was das Modell „Binnensee" nennt, sind Flächen
mit H ≤ 0 ohne Verbindung zum Kartenrand, also Zufallssenken unter dem Meeresspiegel. Es gibt
keinen Mechanismus, der eine Senke bis zum Überlauf **füllt**. Vor einem Bau ist zu klären: (1) ob
`water.lake_detection` bereits Seen in der Taiga liefert und ob sie im Bild ankommen — dann wäre es
ein Anzeige-, kein Erzeugungsproblem; (2) ob die Seen ins **Terrain** gehören (Heightmap bekommt
eine Seefläche auf Überlaufhöhe, alle nachgelagerten Stufen sehen sie) oder in die **Wasserebene**
(Gelände bleibt, nur Anzeige und Wassersystem kennen sie); (3) längliche Seen sind glaziale Rinnen —
derselbe Mechanismus wie `taeler_eingraben()`, nur ohne Gefälle. **Offen**, ausdrücklich auf später
gelegt: `water.lake_detection` existiert (`core/water_generator.py:407-409`), ein Senkenfüllen im
Weltterrain nicht.

*Herkunft: docs/archiv/2026-08-24_FLUESSE_UND_WASSER.md (Stand 2026-08-24), Block 6 „Seen in der Morobora".*

## 7. Seegliederung — Seegrad, Zieltiefe, Seeeis

Die See ist eine **eigene Voronoi-Gliederung**: `voronoi_regionen()` berechnet die Zellzuordnung
ohnehin für die ganze Karte, die Seezellen werden benutzt statt verworfen. **Der Seegrad** ist eine
Breitensuche über den Zellnachbarschaftsgraphen — Grad 0 sind alle Zellen, die Land enthalten, Grad
1 grenzt an Grad 0, Grad 2 an Grad 1, und so weiter. **Die Zieltiefe ist eine Tabelle, keine
Formel** — die Leitlinie „Festlegung statt Regelkreis": das Ergebnis hängt nur an Seed und Reglern,
nicht an Schrittzahl oder Auflösung. Sie ersetzt den Küstenschelf `-t * (1 - exp(-d/L))`, dessen
Ergebnis an der Auflösung der Abstandstransformation hing.

| Seegrad | Standard | Skerrheim (Fjordland) | Clonagh (Hügelland) |
|---|---:|---:|---:|
| 0 | −3 m | −3 m | −3 m |
| 1 | −40 m | −90 m | −10 m |
| 2 | −90 m | −150 m | −60 m |
| 3 | −150 m | −180 m | −140 m |
| 4+ | −200 m (`MEERESBODEN_M`) | −200 m | −200 m |

**Grad 0 ist ein Minimum, kein Festwert** (`np.minimum`): eine bereits tiefere Küstenzelle wird nicht
angehoben, die Land-Seite formt weiterhin `kuestenform` und der Küsten-Archetyp. **Nur die genannten
Regionen weichen ab** — das Fjordland fällt schon ab Grad 1 steil weg (ein Fjord hat keinen flachen
Schelf), das Hügelland bleibt bis Grad 1 flach und vertieft sich erst ab Grad 2. **Der Seetyp folgt
dem nächsten Ufer**: je Seezelle werden die bis zu zwei nächsten Landpunkte bestimmt, maßgeblich ist
die führende Region des ersten (`ufer_region_a`). **Der Übergang zwischen zwei Graden wird
geglättet** (dieselbe Gaußglättung wie bei den Regionsgewichten), sonst stünden dort Stufen. **Die
See wird feiner gerastert als das Land**: 400 Punkte gegen 200, damit die Kachelung nicht sichtbar
wird.

**Seeeis** entsteht nur vor der Taiga (Morobora) und als **Wahrscheinlichkeit je Seegrad**, nicht als
harter Schnitt — hart an/aus wirkte künstlich: Grad 0 100 %, Grad 1 75 %, Grad 2 50 %, Grad 3 25 %,
Grad 4+ 0 %. Der Würfel fällt **je Zelle**, nicht je Pixel, sonst wäre Eis ein
Salz-und-Pfeffer-Rauschen statt zusammenhängender Schollen; deterministisch aus dem Kartenseed. Grad
4+ bleibt bewusst eisfrei, dort verlaufen Seewege. Die Karte ist ein **statischer Schnappschuss ohne
Jahreszeit**; die Saisonalität des Eises gehört ins spätere Zeitmodell.

**Seewege** gelten ab **Seegrad 1** als tief — das ersetzt die frühere Höhenschwelle „ab 10 m
Tiefe". Der Seegrad ist damit eine Karte, die mehrfach gebraucht wird: Seewege, Fischgründe, später
Seemonster. **Umgesetzt** — `seegliederung()` in `core/terrain_weltkarte.py:1599` (Vorgaben
`punktzahl_land=200`, `punktzahl_see=400`), `TIEFE_JE_SEEGRAD` in `core/terrain_weltkarte.py:1571`,
`SEETYP_TIEFENTABELLE` in `core/terrain_weltkarte.py:1578-1586`,
`EISWAHRSCHEINLICHKEIT_JE_SEEGRAD` in `core/terrain_weltkarte.py:1596`, Glättung in
`core/terrain_weltkarte.py:1758-1759`, Eiswurf in `core/terrain_weltkarte.py:1776-1786`, Seewege in
`core/settlement_generator.py:1225` und `core/settlement_generator.py:1278`.

*Herkunft: docs/archiv/2026-09-01_KLIMA_UND_SEE.md (Stand 2026-08-07, Vermerk 2026-08-12), §0 „Die Leitlinie: Festlegung statt Regelkreis" und §2 „Die See als Voronoi-Gliederung".*

## 8. Reihenfolge im Gelände

Bindend ist: **Küsten vor Flüssen.**

```
kontinentform -> voronoi_regionen -> parameterfelder -> seegliederung -> oktavenstapel
-> potenzkurve -> grundhoehe -> kuestenform -> Vektorkueste -> seetiefe -> hangfeuchte
   (= weltfeld)   danach:  weltfluesse
```

Die Küste weiß deshalb **nichts** von den Flüssen. Eine Küstenzuweisung nach Flussmündung (etwa ein
Mündungs-Archetyp am großen Fluss) braucht entweder ein Vorziehen der Flüsse — großer Eingriff, das
Flussnetz braucht das fertige Gelände — oder einen zweiten Durchgang über die Küstenzone, der den
Archetyp nachträglich korrigiert und die Umgebung neu mischt. **Umgesetzt und unverändert** —
Schrittfolge in `core/terrain_weltkarte.py:2575-2889`, Flüsse danach in
`core/terrain_generator.py:1883`. Die nachträgliche Küstenkorrektur (5.2) ist **offen**, ebenso
Fjordarme als Vorfluter (5.1) und saisonale Schneeschmelze (5.3); Knotendichte je Region (5.4) ist
verworfen.

*Herkunft: docs/archiv/2026-08-24_FLUESSE_UND_WASSER.md (Stand 2026-08-24), Abschnitt „Was da ist, und was nicht" und Block 5 „Offen, noch nicht beschlossen".*

## Offene Fragen

1. **Seegrad 0: 0 m oder −3 m?** `docs/archiv/2026-09-01_KLIMA_UND_SEE.md:159` nennt „0 | 0 m (unveraendert)"; der
   Code setzt seit 2026-08-11 −3 m (`core/terrain_weltkarte.py:1571`), begründet mit einem
   Nutzerbefund am laufenden Programm. Oben steht der jüngere Wert.
2. **Talform-Werte weichen von der Messtabelle ab.**
   `docs/archiv/2026-08-24_FLUESSE_UND_WASSER.md:322-332` misst Skerrheim 2.3, Morobora 2.2, Nevadin
   1.0, Clonagh/Estrande/Nebelrode je 1.4; `core/daten/regionen.toml` führt heute
   2.6 / 2.0 / 0.8 / 1.3 / 1.5 / 1.3. Oben stehen die TOML-Werte; wann und warum sie nachgezogen
   wurden, ist aus beiden Quellen nicht belegbar.
3. **Regionsniederschlag gegen gemessenes Feld.** `docs/archiv/2026-09-01_KLIMA_UND_SEE.md:51-59` gibt Skerrheim
   2250 mm, Samarcia 430 mm, Morobora 600 mm vor;
   `docs/archiv/2026-08-24_FLUESSE_UND_WASSER.md:18-26` misst im Feld 1967 / 471 / 865 mm. Ob das
   allein die Voronoi-Überblendung und die Hangfeuchte erklärt, ist nicht belegt.
4. **Block 3 gilt in der Quelle als abgehakt.**
   `docs/archiv/2026-08-24_FLUESSE_UND_WASSER.md:431-432` setzt das Häkchen bei „3 (Seen als
   Sammler)", der Blocktext selbst (Zeile 277-296) beschreibt nur die Messung „wirkungslos", keinen
   Bau. Im Code ist keine Seebündelung auffindbar; oben als **offen** geführt.
5. **Ursache des Fjordland-Defizits weiter offen.**
   `docs/archiv/2026-08-24_FLUESSE_UND_WASSER.md:112-114` nennt als nächsten Verdacht die
   Kantenkostenstruktur von Dijkstra im steilen Gelände, ausdrücklich ungeprüft. Solange das so
   bleibt, ist Abschnitt 3 eine Setzung über einem unverstandenen Verhalten.
6. **Fjordland-/Hügelland-Sondertabellen und Seeeis stehen in keiner der beiden Quellen als
   Tabelle.** `docs/archiv/2026-09-01_KLIMA_UND_SEE.md:8-9` erwähnt sie nur im Kopfvermerk und verweist auf
   `docs/OFFENE_PUNKTE.md` 3.6. Die Zahlen in Abschnitt 7 stammen deshalb aus dem Code
   (`core/terrain_weltkarte.py:1578-1596`) — **nicht** aus einer Spezifikationsquelle.
7. **Vorbehalt Kachelgröße ungeprüft.** `docs/archiv/2026-09-01_KLIMA_UND_SEE.md:182-188` verlangt eine Sichtprobe, ob
   die Seezellen als Kachelung sichtbar werden. Der Code steht auf `punktzahl_see=400`
   (`core/terrain_weltkarte.py:1599`), also bereits auf dem erhöhten Wert; ob die Sichtprobe
   stattgefunden hat, ist **nicht verifiziert**.
