# 14 — Siedlungen, Wege und Kulturen

Normativer Teil zu Ortsgrößen, Standortwahl, Rang- und Kulturzuordnung, Wegenetz, Landmarken,
Roadsites und zur Siedlungsnaht zum Spiel. Zeitschnitt ist **932 n. Chr.** Nicht enthalten:
Sitzungsberichte, Messprotokolle, Entwicklungsgeschichte. Katalognamen (Abschnitt 9) stehen
wörtlich in der ae/oe/ue-Schreibung des Codes — sie sind Daten.

## 1. Drei Größenklassen, alle klein

Alle Orte sind ähnlich klein — 15 bis 50 Häuser. Unterschieden wird nicht Stadt gegen Metropole,
sondern **Rang innerhalb einer sehr kleinen Spanne**:

| Rang | Häuser | Gestalt |
|---|---|---|
| Dorf | 15 – 25 | zwei Dutzend Höfe, eine Gasse |
| Siedlung | 25 – 35 | mehrere Gassen, ein Markt |
| Stadt | 35 – 50 | Markt, Wehr, mehrere Wege laufen zusammen |

Auch die „Stadt" ist damit in einer Viertelstunde durchquert — 932 fast überall nördlich der
Alpen die Wirklichkeit. Im Code: `RANG_HAEUSER`, `RANG_REIHENFOLGE`.

*Herkunft: docs/archiv/2026-09-22_SIEDLUNGEN_ENTWURF.md (Stand 2026-08-06), Abschnitt „1. Drei Groessen, alle klein".*

## 2. Wo die Orte liegen — das Eignungsfeld

Ein Eignungswert je Pixel. Jeder Faktor ist begründet; keiner ist nur „sieht gut aus".

| Faktor | Wirkung | Begründung |
|---|---|---|
| **Wasser am Ort** (großer Fluss, Mündung, geschützte Bucht) | stärkster Einzelfaktor | Trinkwasser, Mühlen, Fischerei und vor allem Transport; 932 ist Wasser der einzige billige Weg für Masse |
| **Ebener Grund** | stark | Bauplatz und Pflugland; ein Hang über ca. 15 % trägt weder das eine noch das andere |
| **Ackerland im Umkreis** | stark | bestimmt, **wieviele** Menschen der Ort ernähren kann — trägt die GRÖSSE, während Wasser die LAGE trägt |
| **Höhenlage** | dämpfend | kürzere Wachstumszeit, weniger Ertrag; in den Bergen werden Orte kleiner, nicht seltener |
| **Erreichbarkeit** | dämpfend | ein Ort, den niemand billig erreicht, wächst nicht — verschwindet aber auch nicht, er bleibt Dorf |
| **Biom** | dämpfend | seit Ticket #34 (2026-09-21) sechster Faktor; vorher entstanden Siedlungen unabhängig von Wüste, Sumpf oder Wiese |

`create_combined_suitability()`: Wasser, Ebene und Ackerland gehen **additiv gewichtet** ein
(`wasser 0.45`, `flach 0.30 * terrain_factor`, `acker 0.25 * terrain_factor`, auf ihre Summe
normiert); Höhe, Biom und Erreichbarkeit wirken **multiplikativ dämpfend**, nicht additiv.
Wassertyp gestaffelt nach `WASSERTYP_GEWICHT = {4: 1.0, 3: 1.0, 2: 0.8, 1: 0.5}` (See,
Großfluss, Fluss, Bach); unbekannte Biom-IDs laufen mit Faktor 1.0 und melden eine WARNING.
**Varianz ist Pflicht**: der Rang wird mit Rauschen um den Wert gezogen.

*Herkunft: docs/archiv/2026-09-22_SIEDLUNGEN_ENTWURF.md (Stand 2026-08-06), Abschnitt „2. Wo die Orte liegen — und warum".*

## 3. Wieviele Orte je Kultur, und welchen Rang sie tragen

**Nicht fest drei, sondern 2 bis 5** — abgeleitet aus der Summe der Eignung über der Region im
Vergleich mit allen neun: wo Skerrheim weite bewohnbare Hochflächen bekam, stehen fünf Orte, wo
fast nur Steilwand ist, zwei. Das macht den Seed spürbar. Zusätzlich immer **mindestens ein Ort
je Kultur als Stadt** — sonst fehlte der Kultur ihr Mittelpunkt und ihre garantierte Anbindung.

Rangvergabe (`_rang_zuweisen()`): der beste verrauschte Wert wird immer `stadt`, vom Rest (r =
n−1) die obere Hälfte `siedlung`, der Rest `dorf` — bei 2 Orten [stadt, dorf], bei 3 [stadt,
siedlung, dorf], bei 4 [stadt, 1 siedlung, 2 dorf], bei 5 [stadt, 2 sied., 2 dorf].
**Mindestabstand je Kultur, nicht global**: Wurzel der eigenen Landfläche / (Zielzahl + 1),
mindestens 4 Pixel; wird das Ziel verfehlt, halbiert er sich bis zu zweimal, erst danach werden
weniger Orte gesetzt.

*Herkunft: docs/archiv/2026-09-22_SIEDLUNGEN_ENTWURF.md (Stand 2026-08-06), Abschnitt „3. Wieviele Orte je Kultur".*

## 4. Stadttypen

Zusätzlich zum Rang trägt jede Siedlung einen Stadttyp. Er folgt aus der **Lage**, wird **nach**
der Platzierung bestimmt (die bleibt unberührt) und regelt die mögliche Größe (`rang_erlaubt`)
und das Handelsinteresse (`handelsgewicht()`).

| Typ | Lage | erlaubter Rang |
|---|---|---|
| `bergdorf` | in den Bergen (hoch, steil) | dorf, siedlung |
| `marktstadt` | am Wasser oder viele Orte gut erreichbar; **je Region höchstens einmal** | siedlung, stadt |
| `agrarstadt` | viel flaches, fruchtbares Umland | siedlung, stadt |
| `fischersiedlung` | kleine Hafenlage, Inselort ohne Hinterland | dorf, siedlung |
| `sonstige` | Auffangtyp | dorf, siedlung, stadt |

`TYP_GRUNDGUETE = 0.42` ist die Grundgüte des Auffangtyps — Sondertyp wird ein Ort nur, wenn
seine Lage dort besser ist. `FISCHER_DECKEL = 0.62` hält die Fischersiedlung unter der
Marktstadt, damit an guter Hafenlage **mit** Hinterland die Marktstadt gewinnt. Die
Typ-Eignungen bauen auf denselben Teilfaktoren wie Abschnitt 2 auf.

*Herkunft: verifiziert an `core/settlement_generator.py` (`STADTTYPEN`, `stadttyp_eignungen()`, `_typen_zuweisen()`); Vorgabe laut Codekommentar zu docs/OFFENE_PUNKTE.md 5.16.*

## 5. Das Wegenetz

### 5.1 Kostenfeld zuerst

Je Pixel ein Preis, **einmal** gebaut (`bau_kostenfeld()`), bevor ein Weg gesucht wird:

| Gelände | Preis | Begründung |
|---|---|---|
| eben, Land | 1.0 | Bezugswert |
| Hang | **exponentiell** im Neigungswinkel, `STEIGUNG_SKALA_GRAD = 8.0` Grad | je 8 Grad vervielfachen sich die Zusatzkosten um e |
| Hang ab `MAX_WEG_STEIGUNG_GRAD = 30.0` | `WEGEBAU_UNMOEGLICH = 500.0` | bewusst **endlich**: eine harte Sperre schnitte einen Ort hinter einem Steilwall ganz vom Netz ab |
| Flachwasser 0 bis −5 m | `WASSERKOSTEN_FLACH = 8.0` | Furt oder kurze Brücke, teuer aber machbar |
| Wasser −5 m bis `WASSER_SPERRE_M = −10.0` | `WASSERKOSTEN_TIEF = 25.0`, tiefer gesperrt | nur noch als Brücke, lohnt selten; darunter kein Landweg mehr |
| vorhandener Weg | `WEGERABATT = 0.4` | **der wichtigste Trick**: Wege bündeln sich zu Hauptstrecken statt parallel zu laufen |
| Uferstreifen am Fluss | `UFERWEG_RABATT = 0.7`, Radius `UFERWEG_RADIUS_PX = 2` | längs dem Ufer ist flach und wassernah |
| Flussquerung | `FURT_KOSTEN_JE_TYP = {1: 5.0, 2: 10.0, 3: 22.0, 4: 25.0}` | Bach mit nassen Füßen, Großfluss praktisch nur mit Brücke |
| Brücke an ihrer Stelle | `BRUECKE_KOSTEN = 1.5` | billiger als jede Furt, bleibt aber ein Nadelöhr |

Die Wasserstufen greifen nur bei h ≤ 0. Flüsse schneiden sich **nicht** in die Heightmap ein;
sie kommen über `water_map` aus `water.manning_flow` (0 = kein Wasser, 1 = Bach, 2 = Fluss, 3 =
Großfluss, 4 = See). Weil Wasser **pro Meter** kostet, sucht der billigste Weg die schmalste
Stelle von selbst — Furten entstehen, ohne dass man sie setzt. Brücken sind kein Geländewert:
`platziere_bruecken()` läuft nach dem Bereitschaftstest dort, wo sich Furten bündeln; danach
wird das Kostenfeld neu gebaut.

### 5.2 Welche Paare in Frage kommen, und ob der Weg gebaut wird

**Gabriel-Graph** über alle Orte (`_gabriel_kandidaten()`): zwei Orte sind Kandidaten, wenn im
Kreis über ihrer Verbindungsstrecke kein dritter liegt — ein zusammenhängendes Netz mit typisch
2–3 Nachbarn je Ort. Gebaut wird davon nur, was sich lohnt:

```
Bereitschaft = Rang(A) * Rang(B) * (gleiche Kultur ? 1.0 : BEREITSCHAFT_FREMDKULTUR)
Wegkosten    = Pfadkosten-Summe / Luftlinienabstand
```

mit `RANG_ZAHL = {"dorf": 1, "siedlung": 2, "stadt": 3}` und `BEREITSCHAFT_FREMDKULTUR = 0.45`.
`Wegkosten` ist dimensionslos und damit vergleichbar (flaches Land nahe 1.0, Gebirge und Wasser
weit darüber). **Gebaut wird, wenn Bereitschaft > Wegkosten**, abgearbeitet absteigend nach
Bereitschaft, damit die eifrigsten Verbindungen zuerst Haupttrasse werden und sich schwächere
per Wegerabatt aufbündeln. Zwei Städte derselben Kultur (9.0) verbinden sich fast immer, zwei
Dörfer verschiedener Kultur (0.45) mit Gebirge dazwischen nicht.

**Eine Ausnahme, die immer greift:** die Orte einer Kultur sind untereinander **garantiert**
verbunden. Nach dem Bereitschaftstest wird geprüft, ob jede Kultur einen zusammenhängenden
Teilgraphen bildet; die günstigsten fehlenden Kanten werden nachgetragen, egal was sie kosten,
notfalls als Seeweg. **Bedarfsausbau darüber hinaus:** eine zusätzliche Strecke nur bei
`NETZAUSBAU_MINDESTNUTZEN = 3.0` fachem Gesamtnutzen, höchstens `NETZAUSBAU_MAX_KANTEN = 3` je
Kultur — sonst würde das Netz zum Vollgraphen.

### 5.3 Seewege

Für Paare ohne Landweg (Inseln, getrennte Küsten) derselbe Wegealgorithmus auf einem eigenen
Kostenfeld (`bau_seekostenfeld()`), dem **Spiegelbild** des Landfelds: Land gesperrt,
Flachwasser `SEEWEG_KOSTEN_FLACH = 3.0`, tiefes Wasser 1.0, keine Hangkosten. Auflage
(`_seeweg_anteil_tief()`): der Weg muss den **größten Teil seiner Länge in tiefem Wasser**
liegen, sonst wäre er ein schlechter Landweg; maßgeblich ist mit vorhandenem `seegrad` die
See-Voronoi-Gliederung (Grad 0 teuer, ab Grad 1 billig), sonst `SEEWEG_TIEFE_ZIEL_M = −10.0 m`.
Da eine Siedlung selten auf der Wasserlinie steht, zieht `_naechster_kuestenpunkt()` den
Ausgangspunkt aufs nächste Wasserpixel; das Stück Siedlung → Küste wird gerade angehängt.

**Umsteigekosten** `HAFEN_UMSTEIGEKOSTEN_KM = 40.0` je Hafen (zweimal je Seeweg), über
`hafenkosten(meter_pro_pixel)` umgerechnet — in Kilometern, damit derselbe Weg bei 384 px und
1024 px gleich stark gebremst wird. Zielgröße `SEEHANDEL_ZIEL = 0.35`, also rund 35 % Seehandel
**im Mittel über Karten**. Seewege werden getrennt als `sea_roads` geliefert (eigene Zeichnung:
gestrichelt, eigenes Blau).

### 5.4 Kreuzungen und Roadsites

Kreuzungen erst NACH dem Routing (`kreuzungen_finden()`): alle Wegepixel, an denen sich zwei
Strecken treffen und die **nicht** auf einem Ort liegen — entstanden durch den Wegerabatt aus
5.1. `kreuzungsgrade()` zählt nach; ab `KREUZUNG_MIN_WEGE = 3` gilt eine echte Wegscheide.

Roadsites zuletzt, wenn das Netz steht (`calculate_roadsites()`), in der Kategorienreihenfolge
**echte Wegscheide (≥ 3 Wege) → Furt → Passhöhe → Strecke**. Ein Passpunkt zählt nur bei
wirklichem Anstieg (Höhenspanne > 30 m), sonst wäre „höchster Punkt" auf flachem Land nur
Rauschen; auf langen Zwischenstücken ohne Ort steht eine Roadsite als Halt einer Tagesreise.
Zielzahl **je Region**; je Kategorie kommen randferne Kandidaten zuerst (`_fern_zuerst()`,
Schwelle 200 m zur Kastengrenze) — Präferenz, kein Verbot.

### 5.5 Landmarks

Unabhängig vom Netz (`calculate_landmarks()`); sie dürfen ausdrücklich weitab jedes Weges
liegen. Grundbedingung: `civ_map < landmark_wilderness` (Standard 0.3) und Hangbetrag < 0.5.
Vier Kategorien, jeweils **nur auf Land**: `gipfel` (normierte Höhe > 0.6, echtes lokales
Maximum), `kueste` (Küstenabstand < Kartenkante/20), `quelle` (Wasserabstand unter derselben
Schwelle), `abgelegen` (normierte Höhe < 0.7, Auffangtyp, Deckel `ABGELEGEN_DECKEL = 0.55`). Der
Landfilter ist zwingend — ohne ihn erfüllt fast jedes Meerespixel „abgelegen".

Ausgewählt wird **nach kontinuierlicher Eignung** (`landmark_eignungen()`), nicht per
Zufallsziehung aus einer Binärmaske. Nach jeder Wahl wird die Kategorie in der Region mit
`KATEGORIE_WIEDERHOLUNG = 0.45` abgewertet, damit die flächenmäßig stärkste nicht alles belegt.
Mindestabstand Kartenkante/20, mindestens 3 px. `calculate_landmark_roads()` verbindet jedes
Landmark deterministisch per A* mit dem nächsten Punkt des Hauptnetzes.

*Herkunft: docs/archiv/2026-09-22_SIEDLUNGEN_ENTWURF.md (Stand 2026-08-06), Abschnitte „4.1 Kostenfeld zuerst" bis „4.7 Landmarks"; Zahlenwerte verifiziert an `core/settlement_generator.py`.*

## 6. Reihenfolge im Überblick

```
1  Eignungsfeld        (Wasser, Ebene, Ackerland, Höhe, Biom)
2  Orte setzen         je Kultur 2-5, Rang mit Varianz
3  Kostenfeld          Hang, Wasser, Furten, Uferwege
4  Gabriel-Graph       welche Paare in Frage kommen
5  Wege routen         billigster Weg, mit Wegerabatt
6  Bereitschaftstest   manche Verbindungen scheitern
                       -> Brücken platzieren, Kostenfeld neu bauen
7  Kulturzusammenhang  fehlende Kanten nachtragen
8  Seewege             für was übrig bleibt
9  Kreuzungen finden / 10 Roadsites setzen / 11 Landmarks setzen
```

Erreichbarkeit (Abschnitt 2) hängt am Netz, das Netz an den Orten. Aufgelöst in **einem**
Rückschritt: die erste Platzierung läuft ohne Erreichbarkeit (Faktor neutral 1.0), nach Schritt
7 werden die Ränge einmal nachkorrigiert, ohne die Orte zu verschieben. **Kein zweiter
Durchlauf.** Deshalb nimmt `create_combined_suitability()` ein optionales `reachability_map`
entgegen, statt es selbst zu berechnen.

*Herkunft: docs/archiv/2026-09-22_SIEDLUNGEN_ENTWURF.md (Stand 2026-08-06), Abschnitt „5. Reihenfolge im Ueberblick".*

## 7. Die Siedlungsnaht — was das Spiel vom Editor abholt

**Naht** heißt: die eine Stelle, an der das Spiel Siedlungsdaten vom Editor abholt — mit genau
den Feldern unten und nicht mehr. Der Editor liefert die **Außenform** einer Stadt, das Spiel
baut das Innere selbst. Bewusst eng gehalten auf **vier Dinge** (Kontur der Stadtgrenze,
Anschlusspunkte der Wege, Größe und Typ, Rang und Kultur) plus **Kennung und Ort** je Stadt.
Koordinaten sind **Pixel im Raster** (`map_size` × `map_size`); die Umrechnung in Meter
(`map_distance_km / map_size`, Standard 15 km) gehört nicht dazu.

| Feld | Typ | Fundstelle |
|---|---|---|
| `city_id` | int | `Location.location_id`, gesetzt in `calculate_settlements()` |
| `city_center` | (float, float), Pixel | `Location.x` / `Location.y` |
| `city_boundary_polygons` | Liste von Polygonen aus (x,y)-Pixeln | `_polygonize_settlement_mask()` (Marching-Squares über `city_mask`), ausgegeben von `_calc_city_boundary()` als Ausgabe des Knotens `settlement.city_boundary`; dieselbe Funktion bedient `PlotPhysicsSystem._build_city_boundary_polygons()`, damit Innenleben und Naht dieselbe Kontur sehen |
| `road_entry_points` | Liste von (x,y)-Pixeln je Stadt | `_wege_anschlusspunkte()`, gerufen in `_calc_pathfinding()`, Ausgabe des Knotens `settlement.pathfinding` |
| `city_size` als `house_count` | int, 15–50 Häuser | `Location.house_count`, nach der Rangspanne aus Abschnitt 1 |
| `city_size` als `radius` | float, Pixel | `Location.radius`, aus `house_count` abgeleitet |
| `city_type` | str | `Location.settlement_type` (Abschnitt 4) |
| `rank` | str: `dorf`/`siedlung`/`stadt` | `Location.rank` |
| `culture` | str, Name aus dem `volk`-Feld der Region | `Location.culture` |

**Ausdrücklich NICHT Teil der Naht** — Innenleben, das sich jederzeit ändern darf: `plot_map`,
`house_parcel_map`, `street_mask` (entsteht im Spiel aus `city_boundary_polygons` + `city_size`
+ `city_type`), `plots`, `plot_nodes`, `plot_cores`, `plot_edges`, `potential_field`,
`voronoi_cell_map`, `wilderness_polygons`. Offen gelassen, nicht entschieden: `landmark_list`,
`roadsite_list`, die Weg-Polylinien (`roads`, `sea_roads`, `landmark_roads`).

*Herkunft: docs/archiv/2026-09-22_SIEDLUNGEN_ENTWURF.md (Stand 2026-08-06), Abschnitt „6. Die Siedlungsnaht — was das Spiel vom Editor abholt".*

## 8. Die neun Kulturen

`volk` ist ein Schlüssel, kein Schmuck: der Siedlungsgenerator liest ihn, und zwei Regionen mit
demselben `volk` wären eine Kultur mit doppelter Fläche. Die Zuordnung steht in
`core/daten/regionen.toml` und wird von `core/terrain_weltkarte.py` als `REGIONEN` geladen.

| Lage | Region | Kultur | Charakter 932 |
|---|---|---|---|
| NW | Clonagh | Kelten | christlich-monastisch, Ringwallgehöfte, keine Städte im römischen Sinn |
| N | Skerrheim | Wikinger | vorchristlich, Hof und Thing, alles hängt am Wasserweg |
| NO | Morobora | Slawen | heidnisch, Holzburgen (Gorod), Pelz- und Flusshandel |
| W | Estrande | Franken | Westfranken nach dem Zerfall, Wikingerzüge, Motten und Abteien |
| M | Nevadin | Alemannen | Passverkehr, Saumhandel, Bergklöster |
| O | Nebelrode | Sachsen | jung christianisiert, Burgwarde Heinrichs I., Landwehren |
| SW | Samarcia | Andalusier | Kalifat von Córdoba (seit 929), Bewässerung, Grenzwehr |
| S | Macchia | Italiener | Königreich Italien, Incastellamento gegen Sarazenenzüge |
| SO | Thalassia | Byzantiner | Themenverwaltung, Inselkastra, arabische Seeräuberei |

Die drei Korrekturen der Entwurfsfassung sind **im Code umgesetzt**: Thalassia trägt Byzantiner
statt Phönizier (deren Stadtstaaten enden rund 1500 Jahre früher), Nevadin Alemannen statt „-"
(hatte keine Kultur und damit keine Städte), Nebelrode Sachsen statt Franken (war doppelt mit
der Estrande belegt). Alle neun Regionen sind unterschieden.

*Herkunft: docs/archiv/2026-09-01_KULTUREN_UND_ORTE.md (Stand 2026-08-06), Abschnitte „Drei Korrekturen an der Kulturzuordnung" und „Die neun Kulturen"; verifiziert in `core/daten/regionen.toml`.*

## 9. Kataloge — 45 Landmarks, 45 Roadsites (umgesetzt)

Je Kultur fünf Landmarks und fünf Roadsites, jede Art mit ihrer Platzierungskategorie. Die
Kategorie folgt aus dem Namen selbst: „Steinkreis auf der Kuppe" will auf einen Gipfel, „Osteria
an der Kreuzung" an eine Wegscheide. Landmark-Kategorien sind `gipfel`, `kueste`, `quelle`,
`abgelegen` (Auffangtyp, siehe 5.5); Roadsite-Kategorien `furt` (Furt, Brücke, Fähre), `pass`
(Passhöhe, Kamm, Sattel), `kreuzung` (Wegscheide, Markt, Grenze), `strecke` (Rest).

| Kultur | Landmarks | Roadsites |
|---|---|---|
| Kelten | Steinkreis auf der Kuppe (gipfel) · Heilige Quelle mit Opfergaben (quelle) · Ogham-Stein als Grenzmal (abgelegen) · Ganggrab (abgelegen) · Bienenkorbzellen am Kliff (kueste) | Furtstein an der Flussquerung (furt) · Rastkreuz an der Wegscheide (kreuzung) · Bardenlager (strecke) · Pilgerherberge (strecke) · Zollringwall (kreuzung) |
| Wikinger | Runenstein (abgelegen) · Langhaus des Jarls (abgelegen) · Hoergr — Steinaltar auf der Hoehe (gipfel) · Thingplatz (abgelegen) · Gestrandetes Langschiff (kueste) | Faehrstelle ueber den Fjord (furt) · Sennhuette (pass) · Handelsplatz am Strand (strecke) · Kohlenmeiler (strecke) · Salzsiederei (strecke) |
| Slawen | Gorod — Ringwallburg (abgelegen) · Heiliger Hain (abgelegen) · Wehrturm aus Blockholz (abgelegen) · Verlassene Brandrodung (abgelegen) · Baerenhoehle mit Opferstelle (abgelegen) | Bohlenweg durchs Sumpfland (strecke) · Pelzhaendlerlager (strecke) · Blockhausherberge (strecke) · Grenzverhau aus Staemmen (kreuzung) · Faehre am Strom (furt) |
| Franken | Steinerne Abtei (abgelegen) · Rest einer Koenigspfalz (abgelegen) · Salzgarten am Aestuar (kueste) · Kliffkapelle (kueste) · Aquaeduktstueck der Roemer (abgelegen) | Zollbruecke (furt) · Wechselstall fuer Pferde (strecke) · Fischerweiler (strecke) · Weinschenke (strecke) · Muehlenwehr (furt) |
| Alemannen | Bergkloster auf dem Sattel (gipfel) · Trutzburg auf dem Felskopf (gipfel) · Gletscherzunge (gipfel) · Eisenerzgrube (abgelegen) · Wildheu-Alm (gipfel) | Passhospiz (pass) · Wechselstall fuer Saumtiere (pass) · Klause mit Wegzoll (kreuzung) · Holzriese (strecke) · Kaesespeicher (strecke) |
| Sachsen | Stumpf der gefaellten Irminsul (abgelegen) · Missionskirche aus Bruchstein (abgelegen) · Silbergrube (abgelegen) · Alte Landwehr (abgelegen) · Opferstein im Buchenwald (abgelegen) | Warte auf dem Kamm (pass) · Gerichtslinde (kreuzung) · Wuestung (strecke) · Kruggasthof (strecke) · Kalkofen (strecke) |
| Andalusier | Hisn — Felsenburg (gipfel) · Alcazaba-Ruine (gipfel) · Noria — Schoepfrad am Fluss (quelle) · Atalaya — Signalturm (gipfel) · Nekropole am Wadi (abgelegen) | Funduq — Karawanserei (strecke) · Aljibe — Zisterne am Weg (strecke) · Canada — Herdenweg (strecke) · Zoco — Marktflecken (kreuzung) · Oelmuehle (strecke) |
| Italiener | Roemische Bruecke (quelle) · Bergdorfkastell (gipfel) · Basilika mit Campanile (abgelegen) · Terrassierte Olivenhaenge (abgelegen) · Schwefelquelle (quelle) | Via-Rest mit Meilenstein (strecke) · Fischtrockenplatz (strecke) · Weinpresse (strecke) · Rastplatz auf dem Bergsattel (pass) · Osteria an der Kreuzung (kreuzung) |
| Byzantiner | Kastro — Inselfestung (kueste) · Klippenkloster (kueste) · Antiker Tempel als Steinbruch (abgelegen) · Antikes Amphitheater (abgelegen) · Schiffswrackriff (kueste) | Skala — Anlegebucht (furt) · Zisternenhof (strecke) · Schwammtaucherlager (strecke) · Eselspfad mit Stuetzmauern (pass) · Xenodocheion (strecke) |

Der Typ einer Roadsite oder eines Landmarks kommt aus dem Katalog der **nächstgelegenen
Siedlung** (`_naechste_kultur()`); beide tragen keine eigene Kulturzuordnung, sondern führen sie
in `properties['culture']`.

*Herkunft: docs/archiv/2026-09-01_KULTUREN_UND_ORTE.md (Stand 2026-08-06), Abschnitt „AUSWAHL DES NUTZERS — 2026-08-06"; wörtlich verifiziert gegen `LANDMARK_KATALOG` und `ROADSITE_KATALOG` in `core/settlement_generator.py`.*

## Reserve (nicht umgesetzt)

Aus jeder Zehnerliste wurden fünf Arten gewählt; die übrigen fünf stehen als Reserve und kommen
**im Programm nicht vor** — verifiziert: beide Kataloge enthalten je genau 45 Einträge, und die
Namen unten tauchen nirgends im Code auf.

| Kultur | Landmarks (Reserve) | Roadsites (Reserve) |
|---|---|---|
| Kelten | Hochkreuz aus Stein · Ringwall (rath) · Pfahlbauinsel in der Bucht (crannog) · Rundturm des Klosters · Feenhuegel | Viehtriebpferch · Grenzhecke mit Torstein · Schmiede am Bachlauf · Torfstich · Meilenstein aus Rohstein |
| Wikinger | Grabhuegelfeld · Schiffssetzung aus Steinen · Bootshaus am Fjordufer (naust) · Wachtfeuerkuppe (viti) · Wasserfallheiligtum | Naust als Raststelle · Schiffszug (Umtragestelle) · Steinmann als Wegmarke (varde) · Bohlenweg durchs Moor · Passhuette unter dem Grat |
| Slawen | Goetzenpfahl mit vier Gesichtern · Kurgan · Bienenbaumwald der Waldimker · Teerschwelerei · Quellheiligtum mit Baendern | Wolok · Koehlerplatz · Waage der Salzstrasse · Wegpfahl mit Zeichen · Winterlager mit Schlittenspur |
| Franken | Motte · Roemischer Leuchtturm (Ruine) · Brandruine eines Wikingerzuges · Reliquienschrein · Marschendeich mit Warft | Ladestelle mit Tretkran · Roemerstrassenrest mit Meilenstein · Faehrhaus am Aestuar · Marktkreuz · Wachturm gegen Nordmaenner |
| Alemannen | Wasserfall ueber die Trogwand · Bergsturz-Blockfeld · Roemischer Passaltar · Klause (Einsiedelei in der Wand) · Steinbogenbruecke ueber die Klamm | Almhuette · Lawinenunterstand · Schmelzhuette · Wegkreuz auf der Passhoehe · Furt ueber den Gletscherbach |
| Sachsen | Burgward Heinrichs I. · Felsenkamm mit ausgehauenen Nischen · Koehlerwald · Huegelgraeberfeld · Wallburgruine der Alten | Hohlwegbuendel · Landwehrdurchlass · Hammerschmiede am Bach · Furt mit gelegtem Steinbett · Rastplatz der Salzstrasse |
| Andalusier | Ribat · Qanat-Schacht · Bewaesserte Huerta · Roemisches Theater als Steinbruch · Salzpfanne | Wachturm der Signalkette · Wadi-Furt · Toepferofen · Wehrspeicher · Grenzstein der Mark |
| Italiener | Kuestenturm gegen Sarazenen · Verlassene Villa Rustica · Marmorbruch · Katakombe · Aquaedukt-Bogenreihe | Pilgerhospiz · Zollturm am Talausgang · Faehre unter der zerstoerten Bruecke · Ziegelei · Saumpfad zum Bergwerk |
| Byzantiner | Kuppelkirche mit Fresken · Leuchtfeuerkette (phryktoria) · Versunkene Marmormole · Windmuehlenkuppe · Salzgaerten der Bucht | Wachturm ueber der Meerenge · Fischsalzerei · Zollstation des Themas · Kapelle am Kap · Umschlagplatz fuer Oel und Wein |

*Herkunft: docs/archiv/2026-09-01_KULTUREN_UND_ORTE.md (Stand 2026-08-06), Abschnitte „Landmarks" und „Roadsites" (die nicht gewählten Einträge).*

## Offene Fragen

1. **`city_type` hat fünf Werte, nicht vier.** §6.1 des Entwurfs und der Docstring von `Location.settlement_type`
   nennen vier Typen, `STADTTYPEN` zusätzlich `fischersiedlung`. Der Text folgt dem Code.
2. **Hangkosten quadratisch oder exponentiell?** Abschnitt 5.1 sagt „Quadrat der Neigung", der Code
   rechnet seit 2026-08-13 exponentiell im Neigungswinkel. Der Text folgt dem Code.
3. **Seeweg-Auflage „ab 10 m Tiefe" oder „ab Seegrad 1"?** Abschnitt 5.3 nennt die Tiefenschwelle,
   `bau_seekostenfeld()`/`_seeweg_anteil_tief()` benutzen mit vorhandenem `seegrad` die
   See-Voronoi-Gliederung. Welche Fassung verbindlich ist, steht nicht fest.
4. **Sechs Faktoren statt fünf.** Der Biomfaktor kam mit Ticket #34 (2026-09-21) dazu und steht
   im Entwurf nicht; ob gleichrangiger Faktor oder Korrektur, ist offen.
5. **Zeilennummern in §6.1 des Entwurfs stimmen nicht mehr** (`:441`, `:2443`, `:2595`). Alle
   Funktions- und Feldnamen stimmen; dieser Text verweist deshalb nur über Namen.
6. **Keine feste Spanne für Roadsites je Region.** „1–4 jeweils" steht nur als Nutzerzitat im
   Codekommentar zu `calculate_roadsites()`; die Zielzahl hängt am Regler `roadsites`.
7. **Reglerabhängige Werte sind nicht festgeschrieben** (`roadsites`, `landmarks`,
   `landmark_wilderness` Standard 0.3, `terrain_factor_villages` aus
   `gui/config/value_default.py`). Genannt sind Standardwerte, keine Sollbereiche.
