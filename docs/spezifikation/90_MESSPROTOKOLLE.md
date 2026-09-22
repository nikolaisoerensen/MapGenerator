# Anhang: Messprotokolle 2026-07-30 bis 2026-08-04

**Was das hier ist.** Fünfzehn datierte Messprotokolle aus der Arbeit am
Erosions- und Flussnetz, wörtlich übernommen aus
`docs/archiv/2026-07-29_SPEZIFIKATION.md` §7 bis §21 (Zeilen 474–1626).

**Wie das hier zu lesen ist.** Nicht von vorn nach hinten. Dies ist ein
**Anhang**: man springt hinein, um nachzuschlagen, **woher eine Zahl kam**
und **was damals tatsächlich gemessen wurde**. Jeder Abschnitt trägt sein
Datum im Titel.

**Was hier NICHT gilt.** Nichts davon ist normativ. Wo ein Protokoll etwas
fordert oder festlegt, ist diese Festlegung — sofern sie noch gilt — in die
Themendateien `11_GELAENDE.md` bis `15_ANZEIGE.md` übernommen worden. Steht
sie dort nicht, ist sie überholt. Ein Protokoll beschreibt einen Stand, kein
Ziel.

**Warum es trotzdem bleibt.** Die Protokolle enthalten Messreihen,
Fehlschläge und Begründungen, die nirgends sonst stehen — etwa warum
Festlegung dem Regelkreis vorgezogen wurde (§14) und wie der Randabfluss
gelöst wurde (§21). Sie zu löschen hieße, dieselben Sackgassen erneut zu
betreten.

---

## §7 Erosion: Entwaesserung — gemessener Stand 2026-07-30

Werkzeug: `tools/drainage_lab.py`. Zielgelaende: Alpental (Region 04) und
Nebelrode/Bamberg (Region 21). Neue Kennzahl **Entwaesserungsanteil** —
welcher Anteil der Karte findet per D8 einen Weg BIS ZUM RAND. Sie fehlte
bisher; das Erosion-Labor misst nur die GROESSE des Netzes, nicht ob es
irgendwo hinfuehrt.

### Befund 1 — die Becken laufen nie bis zum Ueberlauf voll

| | Wasser steht | noetig zum Ueberlaufen | Faktor |
|---|---|---|---|
| Alpental | 0.29 m | 13.0 m (max 311 m) | 45x zu wenig |
| Nebelrode | 0.30 m | 2.36 m | 8x zu wenig |

Ohne Ueberlauf kein Durchfluss, ohne Durchfluss keine Rinne am Ueberlaufpunkt.
Nur 10-20% der Karte entwaessert zum Rand, 1160 geschlossene Senken im
Alpental (vorher 187) — die Erosion ERZEUGT Becken.

Verdunstung auf 0 fuellt sie: im Nebelrode stehen dann 11 m gegen 2.4 m
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

Nebelrode deutlicher: Abfluss 22.5% -> 13.6%, Netz 315 -> 177.

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
5227 m aus, also ~1500 m zusaetzlich. Nebelrode bitidentisch - dort greift
die Klemme nicht, weil der Abtrag ohnehin unter dem Nachbarabstand liegt.

Nadelzahl 5925 -> 8051. NICHT als Verschlechterung belegt: die Nadelschwelle
ist relief-relativ (3% des Reliefs), und mit dem Relief fiel sie von 157 auf
103 m. Absolut nachzumessen.

Gegenprobe gefahren: mit Klemme AUS sind `smoke_test_erosion_quality`s drei
Fehlschlaege ziffernidentisch (45 px, 7.2%, 24.5->33.0%, Faktor 4.25). Sie sind
vorbestehend und von dieser Aenderung nicht beruehrt.

### Weiter offen

* `smoke_test_erosion_gpu_parity`: Export 38.5 (GPU) gegen 0.1 m (CPU),
  unverstanden seit der relief-relativen Kapazitaet. **Nachtrag 16.09.2026:**
  war kein Rechenfehler, sondern ein zu grobes GPU-Meldeintervall; behoben,
  seither gruen. Aktueller Stand in `docs/TESTBERICHT.md` Abschnitt 3.
* Die Becken laufen weiterhin nicht bis zum Ueberlauf voll (Befund 1). Die
  Klemme verhindert, dass neue entstehen, sie loest die vorhandenen nicht auf.
* Routing-Variante bricht im Nebelrode nach 25 Schritten ab - Ursache
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
   Wieder 02_INVARIANTEN.md 2: erst belegen, dass der neue Zweig laeuft.
2. **`tal_breite_m` war absolut**, wo eine relative Groesse hingehoert
   (02_INVARIANTEN.md 4, fuenfter Fall dieses Typs). Die Talbreite ist kein
   freier Meterwert, sie ist ein Anteil des Abstands zum Nachbarfluss - die
   Wasserscheide liegt in der Mitte. Umgestellt: Abfluss 21.7% -> 32.4%.
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

Er bewegt aber KEINE MASSE und kennt kein Routing. 02_INVARIANTEN.md 3 gilt
fuer ihn nicht, und die Entwaesserungs-Kennzahlen aus §3.2 kann er nicht
erfuellen. Er macht das AUSSEHEN von Erosion; die Entwaesserung kommt aus dem
Skelett (§8). Die beiden sind die zwei Haelften, nicht zwei Alternativen.

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
Konfiguration - das ist 02_INVARIANTEN.md 2 und wurde nachgeholt.

| Alpental | Abfluss | Senken | Netz | Nadeln | beta |
|---|---|---|---|---|---|
| 5 Okt ohne Filter | 10.1% | 187 | 157 | 5507 | -0.362 |
| 5 Okt mit Filter | 5.8% | 530 | 73 | 6155 | -0.237 |
| 1 Okt ohne Filter | 12.5% | 9 | 1801 | 2889 | -0.718 |
| 1 Okt mit Filter | 12.9% | 79 | 430 | 2175 | -0.330 |

Auf dem Untergrund, der tatsaechlich benutzt werden soll, ist der Filter also
abflussNEUTRAL (12.5% -> 12.9%, Nebelrode 12.5% -> 12.2%) und senkt die
Nadelzahl um rund ein Fuenftel (2889 -> 2175). Auf dem 5-Oktaven-Untergrund
haette er die Entwaesserung halbiert - der Unterschied zwischen den beiden
Zeilenpaaren ist der ganze Befund 1.

`beta` wandert von -0.718 auf -0.330, im Nebelrode von -0.946 auf -0.416.
Das Zielband ist -0.4 .. -0.7: der Filter zieht beide von "zu steil" her in
Richtung Band, das Nebelrode hinein, das Alpental darueber hinaus.

Nebenbefund, nicht erwartet: der 1-Oktaven-Untergrund hat SCHON OHNE Filter
deutlich bessere Kennzahlen als der heutige (9 statt 187 Senken, Netz 1801
statt 157). Das gehoert unabhaengig vom Filter nachgesehen.

Gegenprobe in jedem Lauf: `erosion_strength = 0` gibt exakt 0 m Delta.

### Eingebaut 2026-07-30

Der Filter laeuft in `BaseTerrainGenerator._calc_redistribution()`, direkt nach
der Power-Redistribution. Bewusst KEIN eigener Calculator-Knoten: er liefert die
endgueltige Gelaendeform, und 20+ Lesestellen in core/ und gui/ holen die
Heightmap ueber ("terrain.redistribution", "heightmap"). Sie alle umzuhaengen ist
das Risiko aus 02_INVARIANTEN.md 5. So sehen Slope, Schatten, Geology, Weather,
Water, Biome, 2D-Anzeige, 3D-Ansicht und Export den Filter ohne weitere
Aenderung.

**Die Hoehenspanne wird nach dem Filter wiederhergestellt** (`_apply_redistribution`
mit Potenz 1.0, also reine lineare Abbildung). Zwei Gruende: §3.1 fuehrt
"Hoehenspanne genau BASE_ELEVATION_M .. AMPLITUDE" als erfuellt, und das Delta
liess das Relief um 6-8% wachsen; und dadurch kann kein Reglerstand die Karte aus
ihrem Hoehenbereich schieben (§1, 02_INVARIANTEN.md 7).

Hauptschalter `EROSION_FILTER_AKTIV` (Vorgabe True), sieben Regler in
`class EROSION_FILTER`, im Terrain-Tab als Gruppe "Erosion Filter".
`TERRAIN.OCTAVES` von 4 auf 2 gesenkt (02_INVARIANTEN.md 7, abhaengige Defaults
ziehen mit). `ridge_map` als zweiter Output desselben Knotens gespeichert und
im CALCULATOR_GRAPH deklariert - noch von niemandem gelesen und NICHT als
Anzeige-Layer registriert.

`tests/smoke_test_terrain_erosion_filter.py`, alle fuenf gruen: Filter laeuft
(ridge_map -0.998..0.997), Spanne exakt 100..4000 m, Staerke 0 gleich
abgeschaltet (0 m Abweichung), Vorgabewerte wirken (192 m), und **jeder der
sieben Regler hat eine messbare Wirkung** (38 bis 777 m) - 02_INVARIANTEN.md 7,
drei tote Water-Slider gab es hier monatelang.

Laufzeit auf der CPU, Median aus drei Laeufen: 128 px 0.04 s, 256 px 0.29 s,
512 px 6.38 s. Der Sprung ist ueberlinear, weil mit der Auflaesung auch die
zulaessige Oktavenzahl steigt (3/4/5) und die Zwischenbilder aus dem Cache
fallen. Bis 256 px unauffaellig, ab 512 px sichtbar - das ist das Argument fuer
den GPU-Pfad.

### Offen

* Compute-Shader als GPU-Pfad plus Vertrags- und Paritaetstest
  (02_INVARIANTEN.md 1).
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
Nebelrode, norddeutsches Flachland, Skerrheim, Vietnam), jeweils rund
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
| Nebelrode | 200 m | 550 m | 7000 m | 900 m | 0.26 |
| Flachland | 100 m | 165 m | 12000 m | 2500 m | 0.10 |
| Skerrheim | 150 m | 1500 m | 8000 m | 2600 m | 0.34 |
| Vietnam | 120 m | 1100 m | 5000 m | 700 m | 0.30 |

Das Verhaeltnis Grundform zu Rinne traegt den Charakter: Hochgebirge ~4:1,
dicht zergliedertes Nebelrode ~8:1, Kalkbergland ~7:1.

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
| Nebelrode | 79.2% | 141 | 513 m |
| Flachland | 93.1% | 24 | 91 m |
| Skerrheim | **97.4%** | 16 | 1495 m |
| Vietnam | 60.9% | 582 | 1475 m |

Zum Vergleich: Noise-Pfad 10.1%, Feld-Erosion 10-22% (§7), gewachsener Baum
32.4% (§8). Und das ohne eine einzige Erosionsiteration.

### Drei Fehler, und der dritte war der entscheidende

1. **Steigung als 3% pro Meter** statt relativ. Bei 3000 m langen Kanten sind
   das 90 m Anstieg PRO KANTE - das Flachland kam auf 707 m Relief statt 90 m.
   02_INVARIANTEN.md 4, wieder.
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

Die Hochebene funktioniert, und man sieht sie am besten im Skerrheim: zwischen
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


---

## §14 Fahrplan und Festlegungen, 2026-07-30

Entscheidungen des Nutzers nach der Durchsicht von §12/§13. Sie gelten, bis
sie ausdruecklich geaendert werden.

### Drei Schritte, in dieser Reihenfolge

    1. Flussnetz         <- HIER STEHEN WIR
    2. Erosionsfilter    (der bisherige "ATEF-Filter", so heisst er ab jetzt)
    3. Echte Erosion     erst wenn 1 und 2 vollstaendig stimmen

Schritt 3 wird NICHT angefasst, solange 1 und 2 nicht sitzen.

> **Ueberholt, berichtigt 16.09.2026 (Ticket #28).** Hier stand: *"`EROSION_AKTIV`
> bleibt False."* Das gilt nicht mehr - der Schalter steht seit dem 27.08.2026
> auf `True` (`gui/config/value_default.py:1088`). Schritt 3 ist also bereits
> eingeschaltet worden, ohne dass diese Reihenfolge abgearbeitet war, und vier
> Erosionstests sind seither rot mit ungeklaerter Ursache (siehe
> `docs/TESTBERICHT.md`, Abschnitt 3). Ob das zurueckgedreht oder die
> Reihenfolge aufgegeben wird, ist eine offene Entscheidung.

Vor dem Wiedereinschalten sollte ein **Auswahlfeld mit
Regionsvorgaben** (Alpen, Skerrheim, ...) da sein, das die Regler setzt, mit
einem Klick auf- und zuklappbar fuer das Feintuning. Auch das ist nicht
passiert.

### Festgelegt

**Talsohle gehoert dem Fluss, Berge gehoeren dem Noise-Gelaende.** Das
Einzugsgebiet bestimmt die STRAHLWEITE dieses Einflusses. Kein Tal darf wie
ausgeschnitten wirken - der Uebergang ist die eigentliche Anforderung, nicht
die Talform.

**Hochebenen zurueckgestellt.** `PLATEAU_FLATTEN` Vorgabe 0.0 (voelles
Relief). Der Regler hiess vorher PLATEAU_RELIEF und war umgekehrt gepolt.

**Map Size Vorgabe 256.** Unter 30 Pixeln je Tal bricht die Entwaesserung ein
(§13).

**Auslass aus dem Seed.** Ein Winkel aus dem Map Seed legt die Himmelsrichtung
fest, nicht mehr der tiefste Randpunkt. Vorher wanderte der Auslass bei jedem
Regler, der das Gelaende beeinflusst. Geprueft ueber sieben Reglerstaende:
Auslass identisch.

**Fluesse werden vorerst NICHT zu Wasser.** Die Annahme ist, dass Wasser und
Erosion die Mulden selbst finden, wenn die Taeler erst da sind. Offen bleibt
der maeandrierende Fluss in flacher Landschaft, der nur ein kleiner Einschnitt
in einem breiten Einflussgebiet ist.

### OFFEN, aber festgehalten: Berechnung mit Rand

Fuer Randeffekte (Weather, Auslasspunkte, spaeter Nachbarkacheln) soll die
BERECHNETE Karte groesser sein als die dargestellte:

    map_size 256  ->  gerechnet wird 256 + 2 x 10 % = rund 308 px

Der Auslass ist genau so ein Randeffekt und gehoert dann AUSSERHALB des
sichtbaren Bereichs. Betrifft alle Generatoren und ist deshalb ein eigener
Umbau, noch nicht begonnen.

### OFFEN: kleine Aufloesungen

Statt der Warnung "unter 30 Pixeln je Tal" soll das Netz intern auf
mindestens 256 px gerechnet und danach heruntergerechnet werden, damit kleine
Kartengroessen dasselbe Bild liefern. Noch nicht umgesetzt.

### Umgesetzt am 2026-07-30

**Einzugsgebiet statt Strahler-Ordnung.** Ein Rueckwaertslauf summiert die
Punkte flussaufwaerts; daraus Strahlweite (Exponent 0.4, hydraulische
Geometrie) und Eintiefung (0.3). Die Strahler-Ordnung war eine STUFE und
reichte nur bis 3-5 - alle Taeler gleich breit, Quellen mit runden Kappen.

**Verzerrtes Abstandsfeld.** Der Abstand wird mit der lokalen Rauheit von P
MULTIPLIZIERT (nicht addiert): wo das Gelaende lokal tiefer liegt, greift das
Tal weiter aus. Der Talrand wandert dadurch mit dem Gelaende, statt eine
Parallelkurve zum Lauf zu sein. Am Fluss selbst bleibt der Einfluss exakt
100 %, weil eine Streckung bei d = 0 nichts aendert.

Gemessen, 256 px, 15 km, gegen die Flaeche P:

| | Terrain sichtbar | Abfluss |
|---|---|---|
| gleiche Breite ueberall (alt) | 5% | 92% |
| Einzugsgebiet | 58% | 74% |
| Einzugsgebiet + Verzerrung | 55% | 73% |

Der Sprung von 5 % auf 58 % ist die Forderung "Berge gehoeren dem
Noise-Gelaende". Der Abfluss zahlt dafuer 92 % -> 74 %, liegt aber weiter weit
ueber den 7 % ohne Netz.

### Noch nicht gut

Die Talkoerper wirken weiterhin AUFGELEGT: breite dunkle Baender mit harter
Kante. Dazu Querstreifen aus der Linienrasterung - z wird je ganzzahligem
Pixel gesetzt, auf schraegen Linien kommen Nachbarpixel aus verschiedenen
Stuetzstellen.

Noch offen aus dem Nutzer-Vorschlag: der Abnahme-Verlauf vom Fluss weg soll
zusaetzlich durch die vorhandene Noisemap moduliert werden - nicht additiv,
sondern so, dass sich der Noise-Einfluss sanft mit der Entfernung aendert.
Die Abstandsverzerrung ist ein erster Schritt in diese Richtung, aber sie
wirkt auf die Talgrenze, noch nicht auf den Verlauf dazwischen.


---

## §15 Fluesse durch Berge: Diagnose 2026-07-30

Nutzer-Befund an einer Hoehenlinienkarte: ein Nebenarm frisst sich quer durch
einen 3100-m-Gipfel, statt daran vorbeizulaufen. Dazu Kanten an den
Zusammenfluessen.

### Gemessen

Groesster Abtrag 3732 m - am HOECHSTEN PUNKT der Karte (3334 m, 100.
Hoehenperzentil), wo z auf -397 m gedrueckt wird. 16 % aller Flusspixel haben
ueber 1000 m Abtrag, 3.5 % ueber 2000 m. Das ist systematisch.

### Die Ursache ist NICHT die Wegwahl

Zwei Aenderungen wurden umgesetzt und einzeln geprueft:

**Gerichtete Anstiegskosten.** Vorher hingen die Kantenkosten an der HOEHE und
waren symmetrisch - ein Lauf auf einer Hochflaeche wurde so bestraft wie einer,
der eine Wand hochklettert. Jetzt: Dijkstra laeuft vom Auslass nach aussen,
jede Kantenrichtung ist flussaufwaerts, also laesst sich der Anstieg gerichtet
und quadratisch bestrafen.

**Mehrere Auslaesse.** Mit einem einzigen muss JEDER Punkt dorthin
entwaessern, das Netz also jeden Ruecken ueberqueren. Jetzt drei, Richtungen
gleichmaessig verteilt, Startwinkel aus dem Seed (damit bei jedem Lauf an
derselben Stelle).

Auf KNOTENEBENE gemessen wirkt beides deutlich: groesster erzwungener Anstieg
1426 m -> 644 m, Summe 8829 m -> 2494 m.

Im fertigen Gelaende kommt davon NICHTS an: max. Abtrag bleibt bei ~3700 m.

### Die wirkliche Ursache

Der KNOTEN-Baum meidet die Ruecken. Die VERDICHTETE LINIE zwischen zwei Knoten
tut es nicht - sie laeuft geradeaus plus Maeander und schneidet dabei ueber
alles, was dazwischen liegt. Bei 2500 m Knotenabstand ist das ein ganzer Berg.

Jeder Stuetzpunkt der Linie nimmt sein z aus P an seiner eigenen Stelle;
liegt er auf einem Gipfel, ist z dort hoch, und das Eintiefen drueckt ihn
anschliessend bis unter das Niveau der Gegenseite.

Die Glaettung an den Zusammenfluessen ist aus demselben Grund fast wirkungslos
(Kante p99.5: 523 m ohne, 498 m mit sechs Durchlaeufen): was als "Kante an der
Muendung" gemessen wird, ist ueberwiegend der Durchschnitt durch die Ruecken,
nicht der Sprung des Einzugsgebiets.

### Was zu tun ist

Die verdichtete Linie zwischen zwei Knoten muss dem GELAENDE folgen statt
geradeaus zu laufen - ein Weg geringster Kosten ueber das Hoehenfeld zwischen
den beiden Knotenpositionen. Der Maeander kaeme dann aus dem Gelaende selbst,
was zugleich der Nutzer-Vorgabe entspricht ("der fliesst dran vorbei oder
hinab").

Bis dahin bleiben Anstiegskosten und Mehrfach-Auslaesse drin: sie sind
nachweislich richtig, nur nicht hinreichend.

### Messfalle, wieder derselbe Typ

Der Vorabtest mass den erzwungenen Anstieg auf KNOTENEBENE und zeigte eine
klare Verbesserung. Das fertige Gelaende zeigte keine. Zwei verschiedene
Groessen, und die erste ist fuer die Frage nicht die richtige - §5.2, Fall
acht.


---

## §16 Gelaendefolgende Wegfuehrung, 2026-07-30

Umsetzung der Diagnose aus §15: die Verbindung zwischen zwei Knoten laeuft
nicht mehr geradeaus, sondern als WEG GERINGSTER KOSTEN ueber das Hoehenfeld
(`skimage.graph.route_through_array`). Der Maeander entsteht dabei von selbst -
der Lauf geht um den Berg herum statt hindurch.

### Die richtige Messgroesse: UEBERHOEHUNG

Zwei Messgroessen davor waren unbrauchbar und haben je eine falsche
Schlussfolgerung getragen:

* `max(P - z)` an Flusspixeln - in einem Gebirge liegt die Talsohle zurecht
  tausende Meter unter den Gipfeln, der Wert sagt also nichts ueber
  "durchgefressen".
* der erzwungene Anstieg auf KNOTENEBENE - zeigte eine klare Verbesserung, die
  im fertigen Gelaende nicht ankam (§15).

Brauchbar ist: **wie hoch liegt der Lauf ueber dem tiefsten Punkt seiner
Umgebung** (Minimumfilter mit der Breite eines Tals). Null hiesse "genau in der
Talsohle".

| Wegfuehrung | Ueberhoehung | p95 | Abfluss |
|---|---|---|---|
| geradeaus | 476 m | 1539 m | 67% |
| gelaendefolgend, Kosten 2.5 | 352 m | 1118 m | 76% |
| gelaendefolgend, Kosten 8 | 300 m | 884 m | 80% |

Der hoechste Punkt des Netzes faellt von 3334 m (= Gipfelhoehe) auf 2611 m -
das Netz laeuft nicht mehr ueber die Spitze. `cost_strength` Vorgabe deshalb
von 2.5 auf 6.0 angehoben; sie wirkt jetzt an ZWEI Stellen, auf die
Kantenkosten des Baumes und auf die Wegsuche dazwischen.

### Keine Ueberschneidungen - im unterstuetzten Bereich

Bereits gezeichnete Laeufe sind fuer spaetere Wege gesperrt
(`separation_px`), ausser in der Umgebung des eigenen Elternknotens - dort
muss der Nebenarm muenden. Die Kanten werden vom Auslass nach aussen
abgearbeitet, der Hauptlauf hat also Vorrang. Dazu ein KORRIDOR um die gerade
Verbindung (`corridor_fraction`): der Delaunay-Baum ist planar, gerade Kanten
koennen sich nicht kreuzen, und der Korridor haelt den gesuchten Weg in diesem
Korsett.

| Aufloesung | Talabstand | Pixel je Tal | Ueberschneidungen |
|---|---|---|---|
| 256 px | 900 m | 15 | 5 |
| 256 px | 1200 m | 20 | 5 |
| 256 px | 2500 m | 43 | **0** |
| 256 px | 3500 m | 60 | **0** |
| 512 px | 900 m | 31 | 4 |
| 512 px | 2500 m | 85 | **0** |

Null ab etwa 40 Pixeln je Tal. Darunter bleiben einzelne Beruehrungen - das
ist derselbe Bereich, vor dem `carve_river_network()` ohnehin warnt, aber die
Grenze liegt hoeher als die dort genannten 30 Pixel.

**NICHT GELOEST** fuer den dichten Bereich. Drei Versuche, alle gemessen und
alle erfolglos:

* groesseres Suchfenster (Fenster wuchs bis zur ganzen Karte): keine Wirkung
* Korridor um die gerade Verbindung: 3-4 -> 5, also leicht schlechter
* "nur fremde Aeste sperren, Muendung in den eigenen Hauptlauf erlauben" -
  begrifflich richtiger, gemessen deutlich SCHLECHTER (5 -> 12), weil die Wege
  dann an den Hauptlaeufen entlang abkuerzen und dabei fremde Aeste treffen.
  Zurueckgenommen.

Die Zahl steht als `crossings` im Rueckgabewert und wird bei > 0 als WARNING
geloggt - sie ist damit sichtbar statt still.

### Laufzeit

256 px 0.3 s, 512 px rund 1 s bei dichtem Netz. Die Wegsuche laeuft je Kante
auf einem kleinen Fenster, nicht auf der ganzen Karte.


---

## §17 Kanten an den Zusammenfluessen, geloest 2026-07-30

Zweiter Nutzer-Befund aus §15. Nach der gelaendefolgenden Wegfuehrung (§16)
war er endlich isoliert messbar - vorher war er von den Ruecken-Durchschnitten
ueberdeckt.

### Gemessen statt vermutet

Messgroesse: Hoehensprung zwischen BENACHBARTEN Flusspixeln. Ein Fluss faellt
stetig, jeder Sprung ist ein Fehler. Median 1.2 m, aber p99 176 m und
groesster 574 m.

Die Glaettung entlang des Baumes half kaum (p99 176 -> 149 ueber 15
Durchlaeufe). Also nachgesehen, WAS dort aneinanderstoesst: der groesste
Sprung lag zwischen zwei DIREKT AUFEINANDERFOLGENDEN Punkten desselben Laufs,
nicht zwischen zwei Aesten.

### Ursache

Das Eintiefen zieht einen Muendungsknoten auf die Hoehe seines TIEFSTEN
Zuflusses. Der andere Zufluss steht unmittelbar daneben noch auf seiner
eigenen Hoehe - dazwischen ein Pixel. In der Natur hat sich ein Nebenarm auf
dem Weg zur Muendung laengst eingeschliffen; hier fehlte jede Obergrenze fuer
die Steigung.

### Behoben: zweiter Durchlauf von der Muendung aufwaerts

Nach dem Eintiefen (Mindestgefaelle, von den Quellen abwaerts) laeuft jetzt
ein zweiter Durchlauf in die Gegenrichtung und begrenzt die STEIGUNG:

    z[j] = min(z[j], z[eltern] + max_gradient * strecke)

Der Sprung wird dadurch ueber eine Strecke verteilt statt auf einem Pixel zu
stehen. Bricht die Monotonie nicht - die Obergrenze senkt nur, und sie liegt
ueber dem Mindestgefaelle.

| max_gradient | Sprung p99 | groesster | Ueberhoehung | Abfluss |
|---|---|---|---|---|
| aus | 181 m | 538 m | 321 m | 78% |
| 0.30 | 25 m | 73 m | 321 m | 78% |
| **0.12** | **10 m** | **39 m** | 321 m | 78% |
| 0.06 | 6 m | 24 m | 321 m | 79% |

Vorgabe 0.12 (12 %, ein steiler aber vorkommender Gebirgsbach). Ueberhoehung
und Entwaesserung bleiben unberuehrt - die Aenderung kostet nichts.

Als siebte Zusicherung in `tests/smoke_test_terrain_river_network.py`
festgehalten, Grenze 3 % der Hoehenspanne statt eines festen Meterwerts
(02_INVARIANTEN.md 4).

### Nebenbefund

Die Warnschwelle "Pixel je Tal" steht jetzt bei 40 statt 30: unterhalb davon
brechen nicht nur die Entwaesserung, sondern auch die Kreuzungsfreiheit ein
(§16).


---

## §18 Talrand: Auslaufen statt Abschneiden, 2026-07-30

Der Nutzer-Wunsch, der sich durch mehrere Nachrichten zieht: der Effekt soll
am Fluss 100 % sein und mit der Entfernung ABNEHMEN. Kein Tal, das wie
ausgeschnitten wirkt.

### Ursache

`clip(distance / width, 0, 1)` gab dem Flusseinfluss einen EXAKTEN Radius:
innerhalb wirkt er, ausserhalb gar nicht. Diese Grenze ist im Querschnitt als
senkrechte Stufe zu sehen - an einer Stelle faellt das Gelaende von 1280 m auf
-180 m innerhalb eines Pixels. Unabhaengig davon, wie weich das Querprofil
selbst geformt ist.

### Behoben

Der Abstand wird ohne Obergrenze abgebildet:

    u = 1 - exp(-t^a)        mit t = Abstand / Talbreite

u erreicht die 1 nie, der Einfluss klingt also aus statt zu enden - und zwar
ueberall beliebig oft differenzierbar, es gibt nirgends einen Knick.

**Erst mit t^a/(1+t^a) versucht und VERWORFEN.** Die Form klingt zu langsam
aus (bei dreifacher Talbreite noch 10 % Flusseinfluss); der sichtbare Anteil
des Noise-Gelaendes fiel dadurch von 43 % auf 18 % - genau gegen die Vorgabe
"die Berge gehoeren dem Noise-Gelaende". 1 - exp(-t^a) faellt bei doppelter
Talbreite schon unter 2 %.

| edge_softness | Terrain sichtbar | Kruemmung p99.5 | Abfluss |
|---|---|---|---|
| harter Schnitt | 48% | 2435 | 13% |
| 6.0 | 44% | 1053 | 13% |
| 3.0 | 40% | 742 | 72% |
| **2.0 (Vorgabe)** | **35%** | **660** | **83%** |
| 1.3 | 28% | 573 | 82% |

Kruemmung = 99.5. Perzentil des Laplace-Operators, also ein Mass fuer den
KNICK in der Oberflaeche. Ein Talrand mit harter Grenze ist ein Knick.

Nebenbefund: der harte Schnitt liefert nur 13 % Entwaesserung, weil jenseits
der Grenze reines P steht - mit allen seinen Senken.

### Weiterhin offen

Der Querschnitt zeigt: die Talsohlen sind immer noch BREITE FLACHE Boeden weit
unter dem Umland, die Uebergaenge sind nur nicht mehr senkrecht. Der
Band-Charakter kommt damit nicht mehr vom Rand, sondern von der flachen Sohle
(valley_form 1.6 plus divide_blend 0.75 glaetten sie zusaetzlich). Das ist der
naechste Punkt.


---

## §19 Flache Talsohlen, behoben 2026-07-30

Der Querschnitt aus §18 zeigte breite EBENE Talboeden weit unter dem Umland -
der Band-Charakter kam nicht mehr vom Rand, sondern von der Sohle. Zwei
Ursachen, beide von mir selbst eingebaut.

### 1. Die Glaettung wirkte auf BEIDE Enden

`divide_blend` setzte smoothstep, w*w*(3-2w), auf den Mischfaktor. Die
Ableitung ist dort an beiden Enden null - also auch AM FLUSS (w = 1), und
damit wurde die Sohle eben. Eingebaut war es, um die Wasserscheide zu
glaetten.

Ersetzt durch `w^(1 + 2*divide_blend)`: Ableitung null bei w = 0
(Wasserscheide, weich), aber ungleich null bei w = 1 (Sohle, echte Neigung).
Seit §18 laeuft der Talrand ohnehin exponentiell aus, aussen wird also nichts
mehr abgeschnitten.

### 2. `edge_softness` formte die Sohle mit

`1 - exp(-t^a)` verhaelt sich nahe am Fluss wie **t^a**. Bei a = 2 ist die
Kurve dort FLACH: auf 30 % der Talbreite standen noch 84 % Flusseinfluss.

| t (Abstand/Talbreite) | a = 2 | a = 1 |
|---|---|---|
| 0.1 | 0.98 | 0.82 |
| 0.3 | 0.84 | 0.53 |
| 0.5 | 0.59 | 0.33 |

Vorgabe auf a = 1 gesenkt. Damit sind die beiden Rollen entkoppelt: die FORM
des Querschnitts macht allein `valley_form`, das knickfreie Auslaufen nach
aussen die Exponentialfunktion. Vorher formte ein Regler beides.

### 3. Talform-Vorgabe von 1.6 auf 1.1

1.6 ist ein ausgepraegtes U-Tal und hat per Definition eine flache Sohle. Das
U-Tal bleibt einstellbar, es ist nur nicht mehr der Normalfall - es gehoert zu
glazial ueberformten Landschaften (Wallis, Skerrheim).

### Ergebnis

| | Sohlenneigung | Terrain sichtbar | Kruemmung | Abfluss |
|---|---|---|---|---|
| harter Schnitt (Ausgangslage) | 0.600 | 48% | 2386 | 13% |
| U-Tal 1.6, a = 2 | 0.481 | 42% | 513 | 81% |
| **Talform 1.1, a = 1** | **1.094** | 39% | 519 | 77% |

Sohlenneigung mehr als verdoppelt, Kruemmung (der Knick am Talrand) weiterhin
bei einem Fuenftel der Ausgangslage.

Im Querschnitt: die Taeler laufen jetzt spitz zu, statt einen 2 km breiten
ebenen Boden zu haben, und die Flanken folgen dem Noise-Gelaende deutlich
enger.


---

## §20 Pipeline-Test ueber alle Outputs, 2026-07-30

`tests/smoke_test_pipeline_outputs.py`. Faehrt alle 38 Knoten des
CALCULATOR_GRAPH in topologischer Reihenfolge (aus dem Graphen abgeleitet,
02_INVARIANTEN.md 5) und prueft jeden der 73 deklarierten Outputs auf FEHLT /
NUR NULL / KONSTANT / NICHT-ENDLICH / OK. Zwei Durchgaenge: mit ShaderManager
(GPU) und ohne (CPU).

Angelegt, nachdem der Nutzer meldete, dass viele Anzeige-Schalter nichts mehr
zeigen. Kein vorhandener Test haette das gefunden - sie pruefen je einen
Generator.

### Erster Lauf: 60 von 73 in Ordnung

| Befund | Outputs |
|---|---|
| erosion.hydraulic/* alle NUR NULL | 7 - damals ERWARTET, EROSION_AKTIV stand auf False (§8); seit 27.08.2026 `True`, siehe Nachtrag unten |
| water.lake_detection/lake_map | KONSTANT auf GPU, OK auf CPU - **Paritaet verletzt** |
| biome.climate_classification | KONSTANT auf GPU, OK auf CPU - **Paritaet verletzt** |
| settlement.roadsites/roadsite_list | OK auf GPU, NUR NULL auf CPU - **Paritaet verletzt** |
| settlement.city_boundary/city_cost_map | NICHT-ENDLICH (inf/nan) |
| geology.intrusions/height_delta | NUR NULL auf beiden |
| water.evaporation/evaporation_map | NUR NULL auf beiden |
| settlement.plot_nodes/plots | NUR NULL auf beiden |

Die drei Paritaets-Verletzungen sind der wertvollste Teil: sie waren nur
sichtbar, weil beide Pfade im selben Lauf verglichen werden.

### Die leeren Anzeigen haben ZWEI verschiedene Ursachen

Getrennt nachgemessen:

* **Erosion-Schalter**: leer, weil EROSION_AKTIV auf False steht. Kein Fehler,
  sondern die Entscheidung aus §8. Faellt weg, sobald Schritt 3 des Fahrplans
  (§14) dran ist.

  > **Nachtrag 16.09.2026 (Ticket #28).** Das war der Stand des damaligen
  > Laufs. `EROSION_AKTIV` steht seit dem 27.08.2026 auf `True`. Leere
  > Erosionsausgaben sind heute also **kein** erwarteter Befund mehr, sondern
  > ein offener. Der Messwert oben bleibt stehen, weil er ein Protokoll ist;
  > seine Deutung gilt nicht mehr.
* **Slope und die Geology-Schalter**: die DATEN sind da. Nachgeprueft bis in
  den Domain-Speicher, aus dem die Anzeige liest: heightmap (128,128),
  slopemap (128,128,2), shadowmap (128,128) - alle vorhanden, assemble und
  _save_to_data_manager laufen fehlerfrei. Der Fehler liegt also in der
  DARSTELLUNG, nicht in der Berechnung. Die slopemap ist dreidimensional
  (zwei Komponenten je Pixel); ein Zeichenweg, der ein 2D-Feld erwartet, kann
  damit nichts anfangen.

Das ist noch nicht behoben und braucht die laufende App (02_INVARIANTEN.md 6:
Anzeige und Skalen sind nur dort pruefbar).


## §21 Ring am Kartenrand: Randabfluss, gelöst 2026-08-04

**Der Befund kam aus einem Bild des Nutzers.** Im Flussnetz lief ein durch­
gehender Lauf am linken, unteren und rechten Kartenrand entlang bis zum einen
Auslass – und stieg dafür sogar über die Rücken. Der Nutzer: *"wir wollen ja
schon Täler in eine Richtung erzwingen aber nicht in dieser kreisrunden Form"*.

**Ursache.** Mit einer festen, kleinen Zahl von Auslassen muss JEDER Knoten
dorthin. Der Kartenrand ist im Delaunay-Graph eine durchgehende Kette – für
einen Knoten in der gegenüberliegenden Ecke ist der Weg darauf entlang oft
billiger als quer durchs Gebirge. Kein Regler konnte das abstellen: höheres
`cost_strength` verteuert das Steigen, aber der Ring steigt ja gerade deshalb,
weil er sonst noch teurer wäre.

**Lösung: ein virtueller Wurzelknoten.** `spanning_tree()` hängt an ihn die
Hauptauslasse gratis und JEDEN Randknoten zu einem festen Preis
(`border_outflow`). Dijkstra entscheidet dann selbst: ein Randknoten verlässt
die Karte an Ort und Stelle, sobald der Umweg zum Hauptauslass teurer ist als
der Preis. Der Preis ist relativ angegeben (02_INVARIANTEN.md 4) – 1.0 heißt
"so teuer wie ein Lauf über die halbe Karte".

Das ist auch die physikalisch richtigere Aussage: aus einem 15-km-Fenster
fließt Wasser an vielen Randstellen hinaus, nicht an einer.

**Maß.** Ein Gesamtanteil "Flusslange im Randsaum" verschleiert den Befund –
beklagt wurde EIN langer Lauf. Gemessen wird deshalb die längste
zusammenhängende Kette flussabwärts, die den Randsaum nie verlässt, in Prozent
einer Kantenlänge (256 px, 625 Punkte, Seed 20260730):

| `border_outflow` | 6.0 | 3.0 | 2.0 | 1.5 | 1.0 | 0.5 |
|---|---|---|---|---|---|---|
| längster Randlauf | 212 % | 158 % | 107 % | 93 % | 61 % | 30 % |
| Auslasse | 1 | 21 | 49 | 54 | 66 | 73 |

6.0 entspricht dem Verhalten davor. Vorgabe ist **1.0**.

### Zwei Fallen, beide teuer

**Eine 0 in der Matrix ist für `scipy.sparse.csgraph` KEINE Kante.** Die
Hauptauslasse waren zuerst mit Gewicht 0 an der Wurzel angehängt – also
überhaupt nicht. Ergebnis: jeder Randknoten wurde Auslass, und der Regler war
über seinen ganzen Bereich wirkungslos (81 Auslasse bei jedem Wert). Sie
brauchen einen winzigen positiven Wert. Ebenso mussten die Hauptauslasse aus
der Randliste entfernt werden, weil `csr_matrix` doppelte Einträge SUMMIERT.

**Der Randsaum muss an `size` hängen, nicht an `size - 1`.** Die Punkte kommen
in Metern aus `poisson_points` und laufen nach der Teilung durch mpp bis
`size`. Mit der −1 lag der Streifen bei 256 und 512 px verschieden weit innen,
andere Knoten fielen hinein – das Netz sprang zwischen den Auflösungen
(r = +0.768 statt +0.990, Zusicherung 6 des Smoke-Tests). Nach der Korrektur
r = +0.999, also besser als vor der ganzen Änderung.

### Nebenwirkung: die Kante am Rand ist mit weg

Zusicherung 7 (`Keine Kanten am Lauf`) war seit dem 2026-08-03 rot: größter
Sprung 84 m gegen 54 m Grenze, lokalisiert auf (143,4)→(144,4) am linken
Kartenrand. Dort lief der Auslauf zum Rand direkt neben einem anderen Ast
vorbei, der noch auf 129 m stand, während der Auslauf schon bei 45 m war –
zwei Läufe, die sich am Rand berühren. Mit dem Randabfluss verlässt dieser Ast
die Karte an Ort und Stelle, statt am Rand entlangzulaufen. Größter Sprung
jetzt **8 m**. Die Entwässerung stieg dabei von 51.1 % auf 57.1 %.
