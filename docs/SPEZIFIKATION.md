# Spezifikation MapGenerator

**Dieses Dokument ist die Referenz, die vor jeder Änderung gelesen und nach
jeder Änderung geprüft wird.** Es beantwortet drei Fragen: Was wollen wir
erreichen (§1–§3), was muss dabei immer gelten (§4), und wie wird gearbeitet
(§5–§6).

Angelegt 2026-07-29, nachdem die Arbeit über mehrere Sitzungen reaktiv geworden
war — jeweils dem letzten Befund nachlaufend, ohne Zielbild pro Komponente und
ohne Prüfliste. Die Folge waren drei Messungen am falschen Codepfad an einem
Tag (§5.2) und Änderungen, die an anderer Stelle etwas kaputt machten, ohne
dass es auffiel.

---

## 1. Oberziel

**Reale Landschaften der Erde nachbilden**, an verschiedenen Orten, ohne Meer.
Ein Nutzer stellt über wenige verständliche Regler ein, *welche* Landschaft er
will, und bekommt ein Ergebnis, das der echten Vorlage ähnlich sieht — in Form,
Klima, Gewässern und Bewuchs.

Zwei Bedingungen, die das Oberziel mittragen:

**Kein Reglerstand darf die Welt zerstören.** Wenn eine Einstellung ein
unbrauchbares Ergebnis erzeugen kann, ist entweder ihr Bereich falsch oder sie
ist mit einer anderen Größe nicht verknüpft. Der Nutzer soll nicht wissen
müssen, wo Probleme aufschwingen.

**Die Regler sind Landschaftsbeschreibungen, keine Implementierungsgrößen.**
„Höhenunterschied" und „Geländecharakter" statt `redistribute_power` und
`capacity_kc`.

---

## 2. Zielkatalog: 20 Landschaften

Die Abnahme erfolgt gegen diese Liste. Jede Landschaft braucht Referenzbilder
und ausgefüllte Zielwerte. **Noch offen — gemeinsam mit dem Nutzer zu füllen.**

| # | Landschaft | Breite | Basishöhe | Relief | Talform | Untergrund |
|---|---|---|---|---|---|---|
| 1 | Sahara Sandwüste (Erg) | 25° | 300 m | gering | – | Sand |
| 2 | Sahara Felswüste (Hamada) | 27° | 500 m | mittel | V, kurz | Fels |
| 3 | Atacama | 24° S | 2400 m | mittel | V | Fels/Schotter |
| 4 | Alpen Hochtal (Wallis) | 46° | 700 m | 3000 m | U (glazial) | Fels/Moräne |
| 5 | Dolomiten Kalkgebirge | 46° | 1000 m | 2000 m | V steil | Kalk |
| 6 | Karst Guilin/Vietnam | 23° | 100 m | 300 m | Türme | Kalk |
| 7 | Norwegen Fjordland (Inland) | 62° | 200 m | 1500 m | U tief | Fels |
| 8 | Schottisches Hochland | 57° | 200 m | 1200 m | U breit | Torf/Fels |
| 9 | Island Vulkanhochland | 65° | 600 m | 1000 m | V jung | Basalt/Asche |
| 10 | Amazonas Tiefland | 3° S | 100 m | gering | flach | Lehm |
| 11 | Ostafrikanischer Graben | 2° | 1500 m | 2000 m | Bruchstufen | Basalt |
| 12 | Serengeti Savanne | 3° S | 1500 m | gering | flach | Lehm |
| 13 | Tibet Hochplateau | 32° | 4500 m | gering | flach | Schotter |
| 14 | Anden Altiplano | 18° S | 3800 m | mittel | U weit | Asche |
| 15 | Colorado Plateau/Canyon | 36° | 1800 m | 1500 m | Schluchten | Sandstein |
| 16 | Badlands South Dakota | 44° | 800 m | 200 m | V dicht | Ton |
| 17 | Westsibirische Moorebene | 60° | 100 m | gering | flach | Torf |
| 18 | Kanadischer Schild | 50° | 300 m | gering | Seenwannen | Fels |
| 19 | Toskanisches Hügelland | 43° | 300 m | 400 m | U sanft | Lehm/Mergel |
| 20 | Neuseeland Südalpen | 44° S | 400 m | 2500 m | U | Fels |

### 2.1 Zielwerte je Landschaft — Schema

Für jede Landschaft auszufüllen. Absolute Zahlen sind zweitrangig, die
**Verhältnisse zwischen den Landschaften** sind das Abnahmekriterium.

| Größe | Einheit | Anmerkung |
|---|---|---|
| Temperatur Sommer / Winter | °C auf Basishöhe | Monatsmittel, nicht Extremwerte |
| Jahresschwankung | K | maritim klein, kontinental groß |
| Windstärke | m/s | Jahresmittel und Böenanteil |
| Niederschlag | relativ, Sahara = 1 | absolute mm folgen aus einem Faktor |
| Saisonalität des Regens | Anteil in der feuchtesten Jahreshälfte | Monsun gegen gleichmäßig |
| Anzahl Flüsse | pro 15×15 km | wieviele erreichen den Kartenrand |
| Flussbreite | m, größter Lauf | folgt aus dem Durchfluss |
| Talform | V / U / Schlucht / flach | hängt am Untergrund |
| Biome | Flächenanteile in % | 3–5 dominante |

---

## 3. Komponenten-Ziele

Jede Komponente hat ein eigenes Zielbild, eine Messgröße und einen Stand.
Ohne Messgröße gibt es kein Ziel, sondern nur eine Meinung.

### 3.1 Terrain

**Ziel:** stetiges Gefälle wie in realen Höhenprofilen. Keine Nadelgrate.
Talböden nicht künstlich flach, sondern von der Erosion geformt.

| Messgröße | Ziel | Stand 2026-07-29 |
|---|---|---|
| Nadeln je Querschnitt (Sprung > 3 % Relief) | ≈ rohes Gelände | **6325 gegen 3962 roh** — die Erosion erzeugt sie |
| mittlere Steigung | passend zu Relief/Ausdehnung | 118 % gegen 8.3 % im Referenzprofil |
| Höhenspanne | genau `BASE_ELEVATION_M` … `AMPLITUDE` | erfüllt |

**Erkenntnis:** `redistribute_power` ist **nicht** die Ursache der Nadeln und
flachen Böden. Querschnitte über 3.5 / 2.0 / 1.4 / 1.0 zeigen identische Form,
nur vertikal gestreckt. Vorher behauptet, per Bild widerlegt.

### 3.2 Erosion

**Ziel:** dichte verästelte Entwässerung, breite helle Talsohlen, Kämme
erhalten. Vorbild: die Luftbilder verflochtener Gebirgsflüsse.

| Messgröße | Ziel | Stand |
|---|---|---|
| `beta` (Hangneigung über Einzugsgebiet) | −0.4 … −0.7 | −0.638 ✓ |
| Netzgröße (größte Kanalkomponente) | möglichst groß | 2055 px |
| Zusammenflüsse | viele | 770 |
| Randabfluss (Kanalzellen am Rand) | > 0 | 369 ✓ |
| Relieferhalt | Ausgang ≈ Eingang | 400→400, 1400→1395 ✓ |
| Massenbilanz | < 1 % | 0.18 % ✓ |

**Offener Konflikt:** Böschungserosion beseitigt die Nadeln vollständig (6325 →
**0**), verdreifacht das Netz und verbessert `beta` auf −0.713 — sieht dem
Nutzer aber nicht zu (»als ob Säure drübergeschüttet wurde«), auch beim
kleinsten Wert. Ein Ersatz muss **nur den neuen Abtrag** räumlich verteilen
(Rinnenbreite), nicht das Gelände.

### 3.3 Weather — Temperatur

**Ziel:** die abgestimmte Klimatologie treffen, über alle Breiten und Monate.

| Messgröße | Ziel | Stand |
|---|---|---|
| Abweichung vom Sollwert 20–50° | < 2 K | 0.0 … 2.0 K ✓ |
| Abweichung Tropen / Pole | < 2 K | +1.4 / −3.3 |
| Jahresschwankung 40° | 23 K | 22.6 K ✓ |
| Höhengradient | 6 K/km | erfüllt |

### 3.4 Weather — Niederschlag

**Ziel:** Doppelstruktur über die Breite — feuchter Äquator, **trockener
Subtropengürtel bei 20–33°**, wieder feuchtere Westwindzone.

| Messgröße | Ziel (Verhältnis zum Äquator) | Stand |
|---|---|---|
| 20° | 0.18 | 0.35 |
| 30° | 0.20 | 0.31 |
| 40° | 0.32 | 0.36 |
| Absolutwert Äquator | 2200 mm | 63 mm (Faktor fehlt) |

### 3.5 Weather — Wind

**Ziel:** noch **nicht definiert.** Zu klären: Jahresmittel je Breite,
Luv-/Lee-Kontrast am Gebirge, jahreszeitliche Drehung, Böigkeit. Ohne Zielwerte
ist Wind nicht abnehmbar.

### 3.6 Water — Gewässer

**Ziel:** 2–3 Bäche vereinigen sich zu einem Fluss, dieser trifft weiter unten
auf einen weiteren. Breite wächst mit dem Durchfluss. Mäander in der Talsohle.
Gelegentlich ein See. Flüsse verlassen die Karte.

| Messgröße | Ziel | Stand |
|---|---|---|
| Seenfläche | > 0 | **0.0 %** |
| Mäander (Sinuosität der Hauptläufe) | > 1.2 | nicht gemessen |
| Breitenvariation | Faktor > 3 vom Bach zum Fluss | nicht gemessen |
| Wasserbilanz | < 1 % | **+10.7 %** ungeklärt |

### 3.7 Biome

**Ziel:** zusammenhängende Zonen, keine gestreuten Einzelpixel. Zwei Stufen —
Klimazone aus Breite und Höhe (großflächig), Biom daraus aus Feuchte, Neigung
und Lage im Tal.

| Messgröße | Ziel | Stand |
|---|---|---|
| Zusammenhang (mittlere Flächengröße je Biom) | groß | nicht gemessen |
| Höhengürtel wandern mit der Breite | Baumgrenze 3600 m am Äquator, 900 m bei 60° | umgesetzt |
| Sumpf breitengradabhängig | in hohen Breiten häufiger | umgesetzt |

---

## 4. Invarianten — bei JEDER Änderung zu prüfen

Diese Liste ist die Prüfliste. Sie ist nach Ursachen sortiert, die im Projekt
schon zugeschlagen haben.

### 4.1 CPU und GPU liefern dasselbe

- Jeder Rechenweg mit GPU-Pfad hat einen Paritätstest gegen die CPU.
- **Eine Änderung an einem Pfad ist erst fertig, wenn der andere mitgezogen
  ist.** Am 2026-07-29 wurde die Rinnenbreite nur im CPU-Pfad eingebaut und
  über die GPU gemessen: vier Varianten kamen bitgleich heraus, die Änderung
  lief nie.
- Konstanten werden vom CPU-Code an den Dispatcher **durchgereicht**, nicht
  doppelt gepflegt (`smoke_test_erosion_gpu_contract`).

### 4.2 Der geänderte Code wird im Messlauf ausgeführt

Der teuerste Fehlertyp dieses Projekts. Drei Fälle an einem Tag:

| Fall | Was schiefging |
|---|---|
| Breitengrad im Pre-Biome | in `set_active_parameters` gesetzt, gelesen wird `data_lod_manager.get_map_latitude()` |
| Talsohlen-Term | auf flachem Testgelände gemessen, dort ist der Term identisch null |
| Rinnenbreite | CPU-Code geschrieben, GPU-Pfad gemessen |

**Regel:** vor der Auswertung belegen, dass der neue Zweig betreten wurde —
Zähler, Log, oder ein Ergebnis, das ohne die Änderung unmöglich wäre.

### 4.3 Masse und Bilanz

- Jeder Pass, der Material oder Wasser bewegt, **verteilt um** statt zu
  erzeugen. Der Glättungspass der Erosion schrieb den geglätteten Wert zurück
  und war dadurch der einzige nicht erhaltende Pass; auf Umverteilen
  umgestellt lag die Drift bei 0.000e+00.
- Einheiten und **Zeitbasen** vor dem Vergleich angleichen. Weather rechnet in
  Jahren, Water in 1800 simulierten Sekunden — eine Bilanz aus zwei Uhren
  ergab scheinbar +3517 % Leck.

### 4.4 Absolute Konstanten sind verdächtig

Viermal derselbe Fehler: eine absolute Größe, wo eine relative hingehört.

| Fall | falsch | richtig |
|---|---|---|
| Regen | pro Schritt | pro Sekunde |
| Konvergenz | pro Schritt | pro Sekunde |
| Glättungsschwelle | absolut | relativ zur Nachbardifferenz |
| Sedimentkapazität | 1 m fest | Bruchteil des Reliefs |

**Regel:** jede neue Konstante mit Einheit muss beantworten, gegen *was* sie
bemessen ist.

### 4.5 Reihenfolge und Abhängigkeiten

- Der Knotengraph (`CALCULATOR_GRAPH`) ist die einzige Wahrheit über
  Reihenfolge. Jede Liste von Generatoren wird daraus **abgeleitet**, nie von
  Hand geführt. Fünf handgepflegte Listen haben je einen Deadlock oder eine
  fehlende Invalidierung verursacht.
- Der Generator-Baum ist eine Vergröberung und **darf Kreise haben**, auch wenn
  der Knotengraph keine hat. Jede Rekursion darüber braucht Zyklusschutz.
- Wer eine Größe liest, deklariert sie als Kante. Settlement las `biome_map`
  ohne Kante und benutzte je nach Thread-Timing eine Ersatzkarte.

### 4.6 Anzeige und Bedienung

Nach jeder Änderung an Daten oder Skalen zu prüfen — **nur in der laufenden App
sichtbar**:

- Ändert sich die Karte, ohne dass die Farbskala mitgeht? Passen die Bereiche
  in `gui_default.py layer_ranges` noch zu den tatsächlichen Werten? (Bei der
  Erosion lag einmal die ganze Karte in der hellsten Stufe: Daten da, Skala
  blind.)
- Sind **2D und 3D** dieselbe Skala und derselbe Layer?
  (`smoke_test_layer_2d_3d_parity`)
- Funktioniert **jeder** Tabwechsel, und laden die Grafiken dabei neu?
- Aktualisieren sich Tabs, wenn ein **fremder** Generator geländeformend
  gerechnet hat? (`_TERRAIN_FORMING_GENERATORS`)
- Ist die 3D-Darstellung maßstabstreu? (15 km × 15 km × 4 km → 10 : 10 : 2.67
  Render-Einheiten)
- Sind neue Ausgaben in **allen** Registern eingetragen — Anzeigemodi,
  Layer-Namen, Farbbereiche, Export?

### 4.7 Regler

- Jeder Regler hat eine **sichtbare Wirkung**. Drei tote Water-Slider gab es
  monatelang.
- Kein Reglerstand erzeugt ein unbrauchbares Ergebnis. Wo das droht, muss die
  Größe an eine andere gekoppelt werden.
- Der Name sagt, was passiert. `SMOOTHING` steuerte eine *Schwelle*: kleine
  Werte bedeuteten aggressives Glätten, das Gegenteil der Erwartung.
- Abhängige Defaults ziehen mit. Ändert sich das Relief, müssen die
  Erosions-Vorgaben dazu passen, ohne dass der Nutzer nachstellt.

---

## 5. Arbeitsregeln

### 5.1 Reihenfolge

1. **Messen, bevor geändert wird.** Ohne Ausgangswert ist keine Verbesserung
   belegbar.
2. **Eine Sache auf einmal.** Amplitude und Glättung gleichzeitig zu ändern
   kostete eine halbe Stunde Suche, bis eine Gegenprobe zeigte, welche es war.
3. **Ansehen, nicht nur nachrechnen.** Vier Kennzahlen haben die
   45°-Pyramiden nicht gefunden; ein Bild zeigte sie sofort. Die
   Drainage-Dichte stufte die beste Variante als schlechteste ein.
4. **Gegenprobe.** Eine Zusicherung, die auch ohne die Änderung hält, prüft
   nichts.
5. **Nach der Änderung die volle Prüfliste** aus §4 — nicht nur den Test, der
   zur Änderung gehört.

### 5.2 Messfallen, die schon zugeschlagen haben

- Werkzeug liefert stillschweigend etwas anderes als angefragt.
  `build_terrain(512)` gab ein 256er Array zurück; alle „512 px"-Messungen
  waren falsch beschriftet. Seither eine `assert`-Zeile.
- Messung am halb umgebauten Stand. Die Konvergenz der Rückkopplung schien
  sauber einzuschwingen — gemessen wurde vor dem Umstellen der Lesestellen,
  also Wiederholungen ohne Rückkopplung.
- Maschinenrauschen als Ergebnis. Identischer Code, dreimal: 12.9 / 12.6 /
  19.2 s. Zeitmessungen brauchen den Median mehrerer Läufe.
- Eigener Erklärkommentar wird von der Textsuche gefunden, die den Code prüfen
  soll. Für Codeprüfungen den Syntaxbaum benutzen.
- Falsche Formel im Messgerät. Die Senkenmessung hatte Rand und Innenbereich
  vertauscht und meldete 627 m tiefe Löcher, die es nicht gab.

### 5.3 Was immer mitzudenken ist

Bei jeder Änderung diese vier Fragen beantworten:

1. **Wie ginge es besser?** Ist das die Ursache oder ein Symptom?
2. **Rechnen CPU und GPU noch dasselbe?**
3. **Welcher Regler gehört dazu, und mit welchem hängt er zusammen?**
4. **Was in §4 könnte ich damit gerade kaputt machen?**

---

## 6. Werkzeuge

| Werkzeug | Zweck |
|---|---|
| `tools/erosion_lab.py` | Erosionsvarianten auf der GPU, Formkennzahlen, Kontaktabzug |
| `tools/weather_lab.py` | Wasserbilanz, Rückkopplungs-Konvergenz, Klima je Breite |
| `smoke_test_*.py` (18 Dateien) | Regressionen; manuell über das venv |

**Kontaktabzug und Querschnitt sind Teil jeder Messung**, nicht Zierde.

### 6.1 Was fehlt

- Ein Prüfstand, der **alle** Komponenten gleichzeitig bewertet, statt eine.
  Solange das fehlt, bleibt „ich reiße mit dem Hinterteil etwas ein"
  unbemerkt.
- Zielwerte für Wind (§3.5), Mäander und Flussbreite (§3.6), Biom-Zusammenhang
  (§3.7).
- Die Referenzbilder und Zielwerte der 20 Landschaften (§2).

---

## §7 Erosion: Entwaesserung — gemessener Stand 2026-07-30

Werkzeug: `tools/drainage_lab.py`. Zielgelaende: Alpental (Region 04) und
Mittelgebirge/Bamberg (Region 21). Neue Kennzahl **Entwaesserungsanteil** —
welcher Anteil der Karte findet per D8 einen Weg BIS ZUM RAND. Sie fehlte
bisher; das Erosion-Labor misst nur die GROESSE des Netzes, nicht ob es
irgendwo hinfuehrt.

### Befund 1 — die Becken laufen nie bis zum Ueberlauf voll

| | Wasser steht | noetig zum Ueberlaufen | Faktor |
|---|---|---|---|
| Alpental | 0.29 m | 13.0 m (max 311 m) | 45x zu wenig |
| Mittelgebirge | 0.30 m | 2.36 m | 8x zu wenig |

Ohne Ueberlauf kein Durchfluss, ohne Durchfluss keine Rinne am Ueberlaufpunkt.
Nur 10-20% der Karte entwaessert zum Rand, 1160 geschlossene Senken im
Alpental (vorher 187) — die Erosion ERZEUGT Becken.

Verdunstung auf 0 fuellt sie: im Mittelgebirge stehen dann 11 m gegen 2.4 m
Bedarf. Es entsteht trotzdem keine Rinne (Becken 18.9%, Abfluss 10.1%), und im
Alpental kippt die fluviale Signatur (beta -0.78 -> -0.24). Wassermenge allein
ist NICHT die Ursache.

### Befund 2 — laengeres Rechnen macht es schlechter

Gleiche Staerke, 8000 statt 6500 Schritte: Abfluss 20.5% -> 15.4%, Senken
1160 -> 1256. Dem Modell fehlt jeder Mechanismus, der ein geschlossenes Becken
aufloest. `fill_depressions` wurde in Plan 3 entfernt mit der Begruendung "ein
Becken ist im Feldmodell einfach ein See" — die Begruendung war falsch.

### Befund 3 — "mehr Iterationen, schwaechere Tropfen" ist widerlegt

Bei konstantem Produkt Ks x Schritte, beide Zielgelaende:

| Alpental | Abfluss | Senken | Nadeln | Netz |
|---|---|---|---|---|
| Ks 0.50 x 8000 | 15.4% | 1256 | 5650 | 419 |
| Ks 0.25 x 16000 | 15.8% | 1484 | 6329 | 564 |
| Ks 0.125 x 20000 | 14.8% | 1559 | 6969 | 590 |

Mittelgebirge deutlicher: Abfluss 22.5% -> 13.6%, Netz 315 -> 177.

### Befund 4 — der Kapazitaetsfaktor ist bei 1.0 geklemmt

`discharge_factor = min(1.0, ...)`. Ein Bach mit 50 Zellen Einzug und ein
Hauptfluss mit 5000 bekommen dieselbe Kapazitaet. Eine Gerinne-HIERARCHIE ist
in dieser Form nicht ausdrueckbar.

### Versuch, der NICHT funktioniert hat

`routed_drainage_area()` (Priority-Flood mit Epsilon, Barnes et al.) als
Kapazitaetssignal statt des lokalen Netto-Flusses. Steht im Code,
`DRAINAGE_ROUTING_INTERVAL = 0` also AUS.

* ohne Budget-Angleichung: Relief 3636 -> 8927 m (Deckel 6.0 grub 6x tiefer)
* mit Mittelwert-Angleichung: Relief 3636 -> 23 480 bzw. 46 682 m und
  numerischer Abbruch. Der Mittelwert wird von den vielen Hangzellen mit
  Faktor 0 dominiert, also bekommt die einzelne Rinnenzelle das Hundertfache.

Was fehlt, bevor es bewertbar ist: eine Obergrenze fuer den Abtrag PRO SCHRITT
UND ZELLE (das Droplet-Modell hatte sie als `cap_per_step`, das Feldmodell hat
sie fuer diese Groessenordnung nicht).

### Was funktioniert hat: die Grabungsklemme

`MAX_DIG_TO_NEIGHBOUR_FRACTION = 1.0` (Default AN, CPU **und** Shader).
Keine Zelle traegt in einem Schritt mehr ab, als sie vom tiefsten Nachbarn
entfernt ist. Dieselbe Klemme, die im Droplet-Modell der entscheidende Fix war
(Plan 2) und im Feldmodell ganz fehlte.

Bewusst geometrisch statt "x Meter pro Schritt": eine Obergrenze pro Schritt
haengt an der Schrittzahl, und genau dieser Fehlertyp wurde in diesem Projekt
schon zweimal behoben (Regen pro Schritt, Konvergenz pro Schritt). Die
geometrische Form ist skalenfrei und braucht keine Kalibrierung.

Gemessen auf der GPU, 192 px, Alpental:

| | vorher | mit Klemme |
|---|---|---|
| Relief | 5227 m | **3447 m** |
| Senken | 1160 | 824 |
| groesstes Netz | 893 | 998 |
| Beckenanteil | 12.2% | 10.6% |
| Entwaesserungsanteil | 20.5% | 21.7% |

Der Hauptgewinn ist das Relief: die Erosion hob aus 3700 m Rohrelief vorher
5227 m aus, also ~1500 m zusaetzlich. Mittelgebirge bitidentisch - dort greift
die Klemme nicht, weil der Abtrag ohnehin unter dem Nachbarabstand liegt.

Nadelzahl 5925 -> 8051. NICHT als Verschlechterung belegt: die Nadelschwelle
ist relief-relativ (3% des Reliefs), und mit dem Relief fiel sie von 157 auf
103 m. Absolut nachzumessen.

Gegenprobe gefahren: mit Klemme AUS sind `smoke_test_erosion_quality`s drei
Fehlschlaege ziffernidentisch (45 px, 7.2%, 24.5->33.0%, Faktor 4.25). Sie sind
vorbestehend und von dieser Aenderung nicht beruehrt.

### Weiter offen

* `smoke_test_erosion_gpu_parity`: Export 38.5 (GPU) gegen 0.1 m (CPU),
  unverstanden seit der relief-relativen Kapazitaet.
* Die Becken laufen weiterhin nicht bis zum Ueberlauf voll (Befund 1). Die
  Klemme verhindert, dass neue entstehen, sie loest die vorhandenen nicht auf.
* Routing-Variante bricht im Mittelgebirge nach 25 Schritten ab - Ursache
  ungeklaert, Variante ist aus.


---

## §8 Struktur vor Noise: Skelett-Ansatz, Stand 2026-07-30

Werkzeug: `tools/skeleton_lab.py`. Richtungswechsel nach der
Nutzer-Entscheidung vom 2026-07-30: das Flussnetz wird ZUERST als Graph gebaut
(Auslass am Rand, aufwaerts wachsend, Strahler-Ordnung, monoton steigende
Hoehe), das Gelaende danach aus dem Abstand zum Fluss geformt. Noise ist
Modulation, nicht Basis. Die Erosion soll damit von "muss die Entwaesserung
erfinden" zu "Feinschliff" werden.

Festgelegter Rahmen: **1 Karte = 1 Region**, Netz pro Karte, nur Randauslaesse
(endorheische Becken spaeter), keine Weltkoordinaten und kein Voronoi, solange
die einzelnen Regionen nicht fuer sich gut aussehen.

### Erste Messung, 192 px, beide Seiten OHNE Erosion

| | Abfluss | Senken | Netz | Nadeln | beta | Becken |
|---|---|---|---|---|---|---|
| Noise 04 Alpen | 10.1% | 187 | 157 | 5507 | -0.362 | 19.1% |
| Skelett 04 Alpen | 32.4% | 22560 | 1482 | 3402 | -0.214 | **0.1%** |
| Noise 21 Bamberg | 10.1% | 187 | 178 | 5197 | -0.445 | 19.1% |
| Skelett 21 Bamberg | 25.0% | 27231 | 1181 | 3396 | -0.030 | **0.0%** |

Abfluss 3x besser, Netz 8x groesser, Beckenanteil von 19.1% auf 0 - und zwar
ohne eine einzige Erosionsiteration. Der Beckenanteil ist der eigentliche
Gewinn: §7 Befund 1 (die Becken laufen nie bis zum Ueberlauf voll) existiert in
dieser Form nicht mehr, weil es keine Becken gibt.

### Drei Fehler auf dem Weg, alle vom bekannten Typ

1. **Der Baum wuchs nie.** Der Auslass sitzt auf y=0, die Bereichspruefung des
   Laeufers verlangte >= 1.0, der erste Schritt von 0.9 px kam nicht durch. Das
   Netz bestand aus EINEM Knoten, `d_norm` war auf 99.1% der Flaeche geklemmt,
   und die gemessenen 3% Abfluss waren keine Aussage ueber das Verfahren.
   Wieder §4.2: erst belegen, dass der neue Zweig laeuft.
2. **`tal_breite_m` war absolut**, wo eine relative Groesse hingehoert (§4.4,
   fuenfter Fall dieses Typs). Die Talbreite ist kein freier Meterwert, sie ist
   ein Anteil des Abstands zum Nachbarfluss - die Wasserscheide liegt in der
   Mitte. Umgestellt: Abfluss 21.7% -> 32.4%.
3. **`max_knoten` war zu klein**, aus der Netzdichte gerechnet statt aus der zu
   fuellenden Flaeche: ~780 Knoten, wo 192 px rund 3000 brauchen.

### Was als Naechstes zu tun ist

Die verbleibenden 22517 "Senken" sind KEINE Becken (Fuellbedarf 0.0%), sondern
Flaechen ohne streng tieferen Nachbarn. Gemessen: 100% davon liegen bei
`d_norm = 1.0`, und der maximale Flussabstand betraegt 139 px auf einer
192-px-Karte. **Das Netz fuellt die Karte nicht.** Es waechst vom Auslass aus
und laeuft aus; alles jenseits der Talbreite wird auf `profil = 1` geklemmt und
ist ein exakt gleich hohes Plateau - 62% aller Zellen haben Hoehendifferenz 0
zum tiefsten Nachbarn.

Zwei Baustellen daraus, noch nicht angefasst:

* Das Wachstum muss die Karte FLAECHENDECKEND erreichen (Spitzen dort neu
  ansetzen, wo die Sperrzone noch Luecken hat - nicht nur verzweigen).
* Die Klemme bei `d_norm = 1` darf kein Plateau erzeugen. Entweder reicht die
  Talbreite immer bis zur Wasserscheide, oder oberhalb der Talbreite uebernimmt
  eine andere Form.

Erst danach ist der Entwaesserungsanteil eine belastbare Zahl. Die Talform
"Tuerme" (Region 06) fehlt bewusst: sie ist eine Eigenschaft der Draufsicht,
kein Querprofil, und braucht einen eigenen Mechanismus.


---

## §9 ATEF-Erosionsfilter: CPU-Referenz, Stand 2026-07-30

Quelle: Rune Skovbo Johansens Advanced Terrain Erosion Filter, vom Nutzer als
`shaders/terrain/ATEF_*.comp` abgelegt (Shadertoy-Multipass). Portierung nach
numpy in `core/terrain_erosion_filter.py`, Werkzeug
`tools/erosion_filter_lab.py`. Lizenz MPL 2.0, datei-bezogen - der portierte
Code bleibt mit Hinweis in seiner eigenen Datei.

### Was der Filter ist und was nicht

Ein Filter PRO PIXEL, keine Simulation: kein Nachbarzugriff, keine Ping-Pong-
Buffer, keine Zeitschritte. Fuenf Oktaven mal eine 4x4-Zellenschleife, ein
Durchgang, 11 us/px auf der CPU. Gegen die 6500-8000 Iterationen aus §7 ist das
praktisch kostenlos.

Er bewegt aber KEINE MASSE und kennt kein Routing. §4.3 gilt fuer ihn nicht,
und die Entwaesserungs-Kennzahlen aus §3.2 kann er nicht erfuellen. Er macht
das AUSSEHEN von Erosion; die Entwaesserung kommt aus dem Skelett (§8). Die
beiden sind die zwei Haelften, nicht zwei Alternativen.

Nuetzliche Nebenausgabe: eine `ridge_map`, -1 in Kerben und +1 auf Kaemmen, vom
Autor ausdruecklich als Entwaesserungs-Eingang genannt. Noch nicht benutzt.

### Portierung belegt

Demonstrationsgelaende nachgebaut (`lauf_demo`): aus einem nahezu
strukturlosen fBm entsteht ein vollstaendig zergliedertes Gebirge, das Delta
zeigt das dokumentierte Muster veraesteter Grate, und die `ridge_map` liegt bei
-0.966 .. +0.997 - der dokumentierte Bereich. Die Ridge-Map laeuft durch eine
lange Kette von Masken; ein Tippfehler reisst diesen Bereich.

### Zwei Befunde, die die Anwendung entscheiden

**1. Der Untergrund muss GLATT sein.** Gemessen bei sonst gleichen Werten,
Alpental 256 px: mit 5 Oktaven (heutige Vorgabe) ist das Delta feinkoerniges
Gekrissel und aendert kaum etwas; mit 1 Oktave entsteht ein zusammenhaengendes
veraesteltes Netz. Der Shader sagt es selbst bei `EROSION_ROUNDING.z` ("if the
height function has noise of 5 times lower frequency than the largest
gullies") - die Demo rechnet mit einem rund zehnfach groeberen Untergrund.
Gegenprobe mit 3x groesseren Rinnen auf detailreichem Untergrund: schlechter,
nicht besser (Delta +-700 m, chaotisch).

Folge fuer die Integration: die Oktavenzahl des Terrain-Noise muss von 5 auf
1-2 herunter, der Filter liefert das gesamte Detail. Das ist zugleich
guenstiger.

**2. `terrain_height_offset[1]` ist eine Falle.** Der Shader beschreibt den
Wert als Relieferhalt ("largely preserving the minima and maxima"), und fuer
die Extremwerte stimmt das. Er setzt aber den `fadeTarget` der LETZTEN Oktave
ein, also eine hochfrequente Groesse - das Bild wird dadurch Gekrissel statt
zusammenhaengender Grate. Bei 0.0 ist der Versatz dagegen eine KONSTANTE
(`magnitude` ist ein Skalar) und aendert die Form gar nicht.

Fuer unsere Karten steht er deshalb auf (0.0, 0.0): der konstante Versatz
entfaellt, weil `_apply_redistribution` die Hoehenspanne ohnehin neu festlegt.

Der erste Sweep variierte Oktaven UND Offset gleichzeitig und war dadurch nicht
auswertbar - der Offset-Effekt ueberdeckte den Oktaveneffekt vollstaendig.
§5.1.2, eine Sache auf einmal, zum zweiten Mal an einem Tag.

### Kennzahlen, 192 px, mit Gegenprobe

Erst auf dem 5-Oktaven-Untergrund gemessen, also auf der bereits verworfenen
Konfiguration - das ist §4.2 und wurde nachgeholt.

| Alpental | Abfluss | Senken | Netz | Nadeln | beta |
|---|---|---|---|---|---|
| 5 Okt ohne Filter | 10.1% | 187 | 157 | 5507 | -0.362 |
| 5 Okt mit Filter | 5.8% | 530 | 73 | 6155 | -0.237 |
| 1 Okt ohne Filter | 12.5% | 9 | 1801 | 2889 | -0.718 |
| 1 Okt mit Filter | 12.9% | 79 | 430 | 2175 | -0.330 |

Auf dem Untergrund, der tatsaechlich benutzt werden soll, ist der Filter also
abflussNEUTRAL (12.5% -> 12.9%, Mittelgebirge 12.5% -> 12.2%) und senkt die
Nadelzahl um rund ein Fuenftel (2889 -> 2175). Auf dem 5-Oktaven-Untergrund
haette er die Entwaesserung halbiert - der Unterschied zwischen den beiden
Zeilenpaaren ist der ganze Befund 1.

`beta` wandert von -0.718 auf -0.330, im Mittelgebirge von -0.946 auf -0.416.
Das Zielband ist -0.4 .. -0.7: der Filter zieht beide von "zu steil" her in
Richtung Band, das Mittelgebirge hinein, das Alpental darueber hinaus.

Nebenbefund, nicht erwartet: der 1-Oktaven-Untergrund hat SCHON OHNE Filter
deutlich bessere Kennzahlen als der heutige (9 statt 187 Senken, Netz 1801
statt 157). Das gehoert unabhaengig vom Filter nachgesehen.

Gegenprobe in jedem Lauf: `erosion_strength = 0` gibt exakt 0 m Delta.

### Eingebaut 2026-07-30

Der Filter laeuft in `BaseTerrainGenerator._calc_redistribution()`, direkt nach
der Power-Redistribution. Bewusst KEIN eigener Calculator-Knoten: er liefert die
endgueltige Gelaendeform, und 20+ Lesestellen in core/ und gui/ holen die
Heightmap ueber ("terrain.redistribution", "heightmap"). Sie alle umzuhaengen ist
das Risiko aus §4.5. So sehen Slope, Schatten, Geology, Weather, Water, Biome,
2D-Anzeige, 3D-Ansicht und Export den Filter ohne weitere Aenderung.

**Die Hoehenspanne wird nach dem Filter wiederhergestellt** (`_apply_redistribution`
mit Potenz 1.0, also reine lineare Abbildung). Zwei Gruende: §3.1 fuehrt
"Hoehenspanne genau BASE_ELEVATION_M .. AMPLITUDE" als erfuellt, und das Delta
liess das Relief um 6-8% wachsen; und dadurch kann kein Reglerstand die Karte aus
ihrem Hoehenbereich schieben (§1, §4.7).

Hauptschalter `EROSION_FILTER_AKTIV` (Vorgabe True), sieben Regler in
`class EROSION_FILTER`, im Terrain-Tab als Gruppe "Erosion Filter".
`TERRAIN.OCTAVES` von 4 auf 2 gesenkt (§4.7, abhaengige Defaults ziehen mit).
`ridge_map` als zweiter Output desselben Knotens gespeichert und im
CALCULATOR_GRAPH deklariert - noch von niemandem gelesen und NICHT als
Anzeige-Layer registriert.

`tests/smoke_test_terrain_erosion_filter.py`, alle fuenf gruen: Filter laeuft
(ridge_map -0.998..0.997), Spanne exakt 100..4000 m, Staerke 0 gleich
abgeschaltet (0 m Abweichung), Vorgabewerte wirken (192 m), und **jeder der
sieben Regler hat eine messbare Wirkung** (38 bis 777 m) - §4.7, drei tote
Water-Slider gab es hier monatelang.

Laufzeit auf der CPU, Median aus drei Laeufen: 128 px 0.04 s, 256 px 0.29 s,
512 px 6.38 s. Der Sprung ist ueberlinear, weil mit der Auflaesung auch die
zulaessige Oktavenzahl steigt (3/4/5) und die Zwischenbilder aus dem Cache
fallen. Bis 256 px unauffaellig, ab 512 px sichtbar - das ist das Argument fuer
den GPU-Pfad.

### Offen

* Compute-Shader als GPU-Pfad plus Vertrags- und Paritaetstest (§4.1).
* `ridge_map` als Eingang fuer die Entwaesserung pruefen - Beruehrungspunkt
  zum Skelett aus §8.
* Die Optik ist noch nicht abgenommen; Kennzahlen koennen sie nicht ersetzen
  (§5.1.3).


---

## §10 Skalenverknuepfung, Stand 2026-07-30

Werkzeug: `tests/smoke_test_terrain_scale_coupling.py`. Frage: bleibt eine Landschaft
DIESELBE, wenn man nur die Aufloesung erhoeht oder nur den Ausschnitt
vergroessert?

Messgroesse: charakteristische Wellenlaenge des Erosions-Deltas IN METERN, aus
dem radial aufsummierten Leistungsspektrum, gemessen mit EINER Rinnen-Oktave.

### Befund: zwei Groessen hingen am Bildausschnitt statt an der Wirklichkeit

| | 5 km | 15 km | 50 km | Faktor |
|---|---|---|---|---|
| Rinnengroesse vorher | 631 m | 1893 m | 6310 m | **10.0** |

Exakt proportional zur Kartenbreite: beim Herauszoomen wurden die Rinnen
groesser statt zahlreicher. Dieselbe Landschaft war bei jedem Ausschnitt eine
andere.

Der Fehler steckte an ZWEI Stellen, und die zweite fiel erst durch die erste
auf:

* `EROSION_FILTER.SCALE` war ein Anteil der Kartenbreite.
* `TERRAIN.FREQUENCY` ebenso: `frequency * (64 / size)` in `_calc_noise` haengt
  nur an der PIXELZAHL. Die Karte zeigte bei jedem `map_distance_km` dieselben
  4.74 Zyklen - Grundformen von 1064 m bei 5 km und 10638 m bei 50 km.

Aufgefallen ist der zweite, weil die gemessene Rinnengroesse systematisch zur
Wellenzahl 4.7 gezogen wurde. 4.7 ist `frequency * 64`.

### Behoben

Beide Regler stehen jetzt in METERN und werden in
`_calc_noise()` bzw. `_erosion_filter_parameters()` in den kartenrelativen Wert
umgerechnet, den der Filter braucht:

| alt | neu | Vorgabe |
|---|---|---|
| `TERRAIN.FREQUENCY` (Zyklen) | `TERRAIN.FEATURE_SIZE_M` | 3150 m |
| `EROSION_FILTER.SCALE` (Anteil) | `EROSION_FILTER.GULLY_SIZE_M` | 2250 m |

Beide Vorgaben entsprechen dem bisherigen Verhalten bei den vorgegebenen 15 km,
damit sich die Standardkarte durch die Umstellung nicht aendert.

Untergrenze in `_erosion_filter_parameters()`: eine Rinne unter drei Pixeln
wird angehoben. Bewusst geometrisch (Pixel je Rinne) statt als Meterwert -
dieselbe Ueberlegung wie bei der Grabungsklemme in §7.

### Ergebnis

| Achse | Streuung | |
|---|---|---|
| Messgeraet (doppelte Rinne = doppelte Messung) | 1.94 | ✓ |
| Aufloesung 128/256/512 px bei 15 km | 1.00 | ✓ |
| Ausdehnung 10/20/40 km bei 256 px | 1.06 | ✓ |

Die Grobform bleibt ueber die Aufloesungen identisch (Abweichung 0.008) -
hoehere Aufloesung ergaenzt nur Detail.

### Zwei Messfallen auf dem Weg

1. Zuerst als Differenz der FERTIGEN Karten gemessen. Beide werden auf
   100..4000 m zurueckgebildet, und diese Rueckbildung ist eine
   grossflaechige Verschiebung, die das Spektrum beherrscht - die Gegenprobe
   kam auf Faktor 0.50 statt 2.0. Jetzt direkt aus `filter_heightmap()`.
2. Dann mit fuenf Oktaven gemessen. `_max_safe_octaves` laesst je nach Skala
   vier oder fuenf zu, das Maximum springt auf eine andere Oktave - Faktor
   1.38 statt 2.0. Jetzt mit einer Oktave und Schwerpunkt statt Maximum.

Beide Male haette die Messung ohne die Gegenprobe aus §5.1.4 eine falsche
Aussage geliefert.

---

## §11 Fuenf Referenzlandschaften auf 25 x 25 km - erster Versuch

Werkzeug: `tools/regionen_lab.py`. Fuenf Hoehenkarten des Nutzers (Alpen,
Mittelgebirge, norddeutsches Flachland, Fjordland, Vietnam), jeweils rund
200 km Bildbreite. Ziel: Ausschnitte von 25 x 25 km im jeweiligen Stil.

### Ergebnis: unterscheidbar, aber nicht aehnlich

Die fuenf Parametersaetze ergeben fuenf klar VERSCHIEDENE Landschaften - die
Regler tun also, was sie sollen. Keine sieht ihrer Vorlage aehnlich.

Im Bild ist zu sehen, woran es liegt: auf 25 km liegen 2-5 runde Grundformen,
auf denen der Filter kurze radiale Rinnen anbringt. Die Vorlagen zeigen
stattdessen DURCHGEHENDE Entwaesserungsnetze, die die ganze Flaeche ordnen -
Taeler, die ueber zehn und mehr Kilometer laufen und sich vereinigen.

### Gegenprobe: die Luecke laesst sich nicht mit dem Filter schliessen

Zwischen Grundform (9000 m) und Rinne (2200 m) fuellt nichts. Naheliegende
Hypothese: die Rinnen-Grundskala an die Grundform heranfuehren. Gemessen mit
2200 / 4500 / 6500 / 9000 m, Alpen:

Widerlegt. Groessere Werte schliessen die Luecke nicht, sondern erzeugen
SCHLAUFEN - geschlossene Ringe und Locken, die keiner Neigung folgen. Der
Shader warnt selbst davor (`EROSION_CELL_SCALE`: "too large values produce
chaotic curved gullies that are not aligned with the slopes"). 2200 m war von
den vieren das beste.

### Schlussfolgerung

Was fehlt, ist kein Reglerwert, sondern der Mechanismus aus §8: das
Flussnetz-Skelett. Weder das fBm (das runde Formen liefert) noch der
ATEF-Filter (der Textur unterhalb weniger Kilometer liefert) erzeugt
zusammenhaengende Taeler im Bereich 2-20 km. Genau diese Luecke sollte das
Skelett fuellen.

Damit ist die Reihenfolge aus §8 bestaetigt, und zwar an Bildern statt an einer
Ueberlegung: Skelett zuerst, Filter als Feinschliff darauf.

### Zahlen der fuenf Saetze

Als Ausgangspunkt fuer die spaetere Eichung, ALLE ZU BESTAETIGEN - sie stammen
aus dem Augenschein, nicht aus Hoehenmodellen.

| Region | Basis | Gipfel | Grundform | Rinne | Staerke |
|---|---|---|---|---|---|
| Alpen | 1000 m | 3500 m | 9000 m | 2200 m | 0.30 |
| Mittelgebirge | 200 m | 550 m | 7000 m | 900 m | 0.26 |
| Flachland | 100 m | 165 m | 12000 m | 2500 m | 0.10 |
| Fjordland | 150 m | 1500 m | 8000 m | 2600 m | 0.34 |
| Vietnam | 120 m | 1100 m | 5000 m | 700 m | 0.30 |

Das Verhaeltnis Grundform zu Rinne traegt den Charakter: Hochgebirge ~4:1,
dicht zergliedertes Mittelgebirge ~8:1, Kalkbergland ~7:1.

BEKANNTE EINSCHRAENKUNG: jede Region beginnt bei 100 m, auch das Flachland.
`TERRAIN.BASE_ELEVATION_M` liegt fest, `amplitude` ist die GIPFELHOEHE, und
Werte unter 100 m drehen das Gelaende um (offener Fehler). Die RELIEFS
stimmen, die absoluten Basishoehen nicht.


---

## §12 Flussnetz mit Hochebene - erste Umsetzung 2026-07-30

Werkzeug: `tools/flussnetz_lab.py`. Loest den gewachsenen Baum aus §8 ab.

### Der Entwurf

    z = P - (P - z_Fluss) * (1 - Profil(d~))

P ist das Gelaende ohne Fluesse (fBm + ATEF-Filter). Am Fluss ergibt das
z_Fluss, an der Wasserscheide P. Die alte Form `z_Fluss + Profil * H` konnte
keine Hochebene erzeugen, weil sie jenseits der Talbreite nur noch am NAECHSTEN
Fluss hing und dessen Hoehe an jeder Wasserscheide springt - Terrassen mit
Nahtstellen. In der neuen Form ist P einwertig, die Naht faellt weg.

Aufbau: Poisson-Disk-Punktsatz (Abstand in Metern) -> Delaunay -> Dijkstra vom
Auslass mit hoehenabhaengigen Kantenkosten -> Strahler-Ordnung rueckwaerts
abgelesen. Kein LOD.

### Ergebnis: die Entwaesserung ist geloest

| | Abfluss | Senken | Relief |
|---|---|---|---|
| Alpen | 79.9% | 145 | 4742 m |
| Mittelgebirge | 79.2% | 141 | 513 m |
| Flachland | 93.1% | 24 | 91 m |
| Fjordland | **97.4%** | 16 | 1495 m |
| Vietnam | 60.9% | 582 | 1475 m |

Zum Vergleich: Noise-Pfad 10.1%, Feld-Erosion 10-22% (§7), gewachsener Baum
32.4% (§8). Und das ohne eine einzige Erosionsiteration.

### Drei Fehler, und der dritte war der entscheidende

1. **Steigung als 3% pro Meter** statt relativ. Bei 3000 m langen Kanten sind
   das 90 m Anstieg PRO KANTE - das Flachland kam auf 707 m Relief statt 90 m.
   §4.4, wieder.
2. **P nicht senkenfrei.** Auf der Hochflaeche gilt z = P, also erbt sie jede
   Delle des Rauschens. Aufgefuellt plus eine winzige Neigung zum Netz hin,
   damit die aufgefuellten Flaechen nicht eben bleiben.
3. **Der Auslass lag NICHT auf dem Kartenrand.** Er ist ein Poisson-Punkt in
   Randnaehe, und ohne Anschluss ans Kartenende war er der tiefste Punkt der
   Karte: das gesamte Netz endete in einer Grube wenige Pixel vor dem Ziel.

Befund 3 ist der lehrreiche. Gemessen waren nur **0.1% Senken** bei **1.0%
Abfluss** - fast alles floss zusammen, nur nicht von der Karte herunter. Die
Senkenzahl allein haette das nie gezeigt. Erst der Entwaesserungsanteil macht
den Unterschied zwischen "es gibt keine Loecher" und "es findet hinaus"
sichtbar. Nach dem Anschluss: 1.0% -> 79.9%.

Dazu ein vierter, kleiner: die Treppenfunktion des Talformers erzeugt exakt
EBENE Absaetze, und eben ist fuer D8 dasselbe wie eine Senke. Vietnam fiel
dadurch auf 7.3% Abfluss bei 2601 Senken. Mit leicht geneigten Absaetzen
(`stufen_neigung`): 60.9% bei 582. Echte Schichtstufen sind ohnehin nie
waagerecht.

### Die Form stimmt noch NICHT

Die Bilder zeigen keine Taeler, sondern **Schlaeuche**: gleich breite Roehren
mit runden Kappen an den Enden. Vier benennbare Ursachen:

* **Gleichfoermige Breite.** Die Talbreite haengt nur an der Strahler-Ordnung,
  und die reicht bei 24-38 Knoten nur bis 3. Fast alle Aeste sind Ordnung 1-2,
  also alle Taeler gleich breit. Echte Taeler weiten sich stetig flussabwaerts;
  die Breite muss am Einzugsgebiet haengen, nicht an einer Ordnungsstufe.
* **Runde Kappen an den Quellen.** Der euklidische Abstand zu einem
  Liniensegment ergibt am Segmentende einen Halbkreis. Ein Quellbach laeuft
  aber aus, er endet nicht in einer Kuppel.
* **Querstreifen.** Die Linienrasterung setzt z je ganzzahligem Pixel; entlang
  einer schraegen Linie kommen Nachbarpixel aus verschiedenen Stuetzstellen,
  das ergibt Stufen quer zum Tal.
* **Negative Hoehen** (Alpen -1260 m, Vietnam -392 m): der Einschnitt drueckt
  die Talsohle unter 0.

### Was funktioniert

Die Hochebene funktioniert, und man sieht sie am besten im Fjordland: zwischen
den Troegen liegt eine durchgehende, nahtlose Flaeche. Das war die Frage, die
den Umbau ausgeloest hat, und die Antwort ist ja - `p_relief_anteil` von 28%
(Norwegen) bis 100% (Alpen) schaltet zwischen Hochebene und Bergland um.


---

## §13 Flussnetz im Hauptprogramm, Stand 2026-07-30

`core/terrain_river_network.py`, eingebunden in
`BaseTerrainGenerator._apply_river_network()`, neun Regler in
`class RIVER_NETWORK`, Gruppe "River Network" im Terrain-Tab.
`tests/smoke_test_terrain_river_network.py`, sechs Zusicherungen.

Entwaesserungsanteil 6.4% -> 90.2% (256 px, 15 km). Spanne immer exakt
0..AMPLITUDE. Punktsatz in METERKOORDINATEN, deshalb bei 256 und 512 px
dasselbe Netz (r = +0.991).

### Nutzer-Kritik am ersten Bild und was daraus folgte

Zwei Punkte, beide zutreffend: alles kantig und facettiert, und die Laeufe
schnurgerade. Beides hatte dieselbe Wurzel.

**Kantig.** Das Abstandsfeld zu einem Liniennetz hat eine Knickkante entlang
der Mittelachse zwischen zwei Laeufen - dieselbe Struktur, die ein reines
Voronoi-Gelaende kennzeichnet. Dazu lief das Talprofil mit einer Steigung
ungleich null gegen 1, also entstand an der Wasserscheide zusaetzlich ein
scharfer Grat. Behoben durch smoothstep auf den Mischfaktor (Ableitung an
beiden Enden null), regelbar ueber DIVIDE_BLEND - scharfe Grate bleiben damit
einstellbar.

**Gerade Laeufe.** Jede Kante ueberspannte den vollen Knotenabstand. Behoben
durch Zwischenpunkte mit seitlicher Auslenkung (MEANDER, als Anteil der
Kantenlaenge) und Chaikin-Glaettung. Die Zwischenpunkte nehmen ihre Hoehe aus
P, der Lauf folgt also dem Gelaende statt geradlinig durchzuschneiden.

### Drei Messfehler auf dem Weg, alle bei DERSELBEN Frage

Die Frage war jedesmal "wie tief schneidet das Tal ein". Drei Anlaeufe:

| Messgroesse | Ergebnis | warum falsch |
|---|---|---|
| "Tiefe 0 == Netz aus" | riss | der Lauf schneidet auch ohne Tiefe durch, das SOLL er |
| 90. Perzentil minus Flusshoehe | 1277 m, nicht monoton | misst das Grossrelief, nicht die Eintiefung; zusaetzlich von der Spannen-Rueckbildung verzerrt |
| P_gefuellt minus z an Flusspixeln, direkt am Modul | 417 / 617 / 1075 m, monoton | richtig |

Erst die dritte Fassung zeigte, dass der Regler sauber wirkt. Die zweite hatte
ausserdem eine falsche Ursache nahegelegt (das Mindestgefaelle), was zu einer
Aenderung fuehrte, die die Entwaesserung von 90% auf 15% brach - siehe unten.
§5.2 fuehrt vier Faelle falscher Messgeraete; das sind Fall fuenf bis sieben.

### Das Mindestgefaelle, zweimal falsch bemessen

Erst 3% PRO METER: bei 3000 m langen Kanten 90 m pro Kante, das Flachland kam
auf 707 m Relief statt 90 m. Dann als fester Wert PRO SEGMENT - aber die
Verdichtung machte aus jeder Kante rund siebzig Segmente, also siebzigfacher
Zwangsabtrag. Jetzt PRO METER LAUFLAENGE, bezogen auf die Kartenbreite:
`min_rise_fraction * peak_m` ist der gesamte erzwungene Abfall, unabhaengig
von der Aufloesung des Laufs. Wert 0.06 gemessen als Kompromiss (192 px: 76%
Entwaesserung statt 41%, dafuer 417 statt 330 m Eintiefung bei Regler 0).

### Neue Grenze: Pixel je Tal

Bei 2500 m Talabstand liefern 128 px auf 15 km nur 21 Pixel je Tal, und die
Entwaesserung bricht auf 17% ein; 256 px (43 px je Tal) liefern 87%. Ursache
ist dieselbe wie bei den ebenen Treppenabsaetzen: zu wenige Stufen im
Querprofil ergeben ebene Flaechen, und eben ist fuer D8 dasselbe wie eine
Senke. `carve_river_network()` warnt jetzt unterhalb von 30 Pixeln je Tal.

Praktische Folge fuer die Bedienung: Map Size 256 oder groesser, oder
groesserer Talabstand.

### Was weiterhin nicht stimmt

Die Taeler wirken wie AUFGELEGTE SCHLAEUCHE statt wie eingeschnittene Taeler,
und bei der Vorgabe-Talbreite liegen nur 6% der Karte naeher als 10 m an P -
das Netz ueberschreibt also fast die ganze Flaeche. Gemessen ueber die
Talbreite:

| valley_width_fraction | P sichtbar | Entwaesserung |
|---|---|---|
| 0.55 (Vorgabe) | 6% | 79% |
| 0.35 | 26% | 74% |
| 0.22 | 45% | 68% |

Der Kompromiss ist damit einstellbar, aber nicht geloest. Die Ursache ist, dass
die Eintiefung als feste Tiefe entlang eines gleich breiten Schlauchs wirkt.
Was fehlt, ist die Kopplung von Talbreite UND Tiefe an das EINZUGSGEBIET je
Punkt statt an die Strahler-Ordnung - dann waechst das Tal stetig von der
Quelle zum Auslass, statt ueberall gleich breit zu sein, und die runden Kappen
an den Quellen verschwinden von selbst.
