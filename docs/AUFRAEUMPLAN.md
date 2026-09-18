# Aufräumplan — Aufbau des Programms und die nächsten Ziele

Angelegt 2026-08-25 auf Nutzerwunsch: *"lass uns allgemein über das programm
reden. wie es aufgebaut ist und wie wir es jetzt einmal aufräumen werden, was
die nächsten ziele sind"*.

Diese Datei ist der **Plan**. Was davon erledigt ist, steht in
`docs/SITZUNGSLOG.md`; was offen ist, in `docs/OFFENE_PUNKTE.md`.

---

## 1. Wie das Programm aufgebaut ist

Drei Schichten, und der Kern der jetzigen Unzufriedenheit liegt zwischen
Schicht 1 und 2.

**Schicht 1 — die neun Regionen (`core/terrain_weltkarte.py`).**
Die Welt ist ein 3×3-Raster aus Regionen. Jede hat ihren eigenen
Parametersatz als Python-Dict:

    hoehe_m, relief_m, formgroesse_m, rauheit, potenz,
    wasser_soll, flaeche_soll, kuestenform, talform, ...

Daraus wird je Pixel ein **Parameterfeld** gemischt (`parameterfeld()`), und
mit diesen Feldern werden neun Rauschoktaven gewichtet. **Hier entsteht das
Gelände.**

**Schicht 2 — die GUI-Regler (`gui/config/value_default.py`).**
Der Terrain-Reiter bietet globale Rauschregler: `FREQUENCY`, `OCTAVES`,
`PERSISTENCE`, `LACUNARITY`, `AMPLITUDE`, `REDISTRIBUTE_POWER`,
`FEATURE_SIZE_M`.

**DAS IST DIE LÜCKE.** Die Regler sind global, das Gelände ist regional. Wer
das Nevadin ändern will, findet in der GUI keinen Griff dafür — er müsste
`core/terrain_weltkarte.py` editieren. Genau das ist der Grund für den
Wunsch nach einem Regions-Dropdown, und der Wunsch ist architektonisch
richtig: er schließt die Lücke an der Stelle, an der sie ist.

<!-- PRUEFBAR: ausdruck=len(managers.calculator_graph.CALCULATOR_GRAPH) erwartet=38 -->
**Schicht 3 — die Pipeline.** Ein Rechengraph aus 38 Knoten
(`managers/calculator_graph.py`) treibt Terrain → Geologie → Wetter →
Wasser → Biome → Siedlungen. Jeder Reiter hängt an einem Knoten.

---

## 2. Die Reglerlage, gemessen

| Gruppe | Regler | tote Regler |
|---|---:|---|
| TERRAIN | 10 | keine |
| RIVER_NETWORK | 11 | keine |
| EROSION_FILTER | 7 | keine |
| EROSION | 13 | laeuft (`EROSION_AKTIV = True`, `gui/config/value_default.py:1088`) — drei rote Tests dazu sind ungeklaert, siehe `docs/TESTBERICHT.md` Abschnitt 3 |
| GEOLOGY | 15 | keine |

**Alle Regler sind verdrahtet.** Ein erster Grep behauptete das Gegenteil;
er war falsch, weil die Parameter mit Präfix ankommen
(`erosion_filter_strength` statt `strength`). Das Problem ist also nicht
Totholz, sondern **Menge und Benennung**.

---

## 3. Der Nevadin-Befund (gemessen 2026-08-25)

Nutzermeldung: *"die alpen brauchen hier viiiiiel weniger spitzen. also
sowas wie 5 berge auf der regionsfläche. hier sind hunderte zu sehen."*

Anteil des Reliefs, der in Wellenlängen ≤ 375 m steckt:

| Region | Formgröße | Relief | Anteil ≤375 m | in Metern | große Formen/Region |
|---|---:|---:|---:|---:|---:|
| **Nevadin** | 3800 m | **1050 m** | **23.9 %** | **250 m** | **3.5** |
| Thalassia | 1100 m | 404 m | 30.2 % | 122 m | 41.7 |
| Macchia | 1800 m | 313 m | 25.1 % | 79 m | 15.6 |
| Nebelrode | 1400 m | 135 m | 31.4 % | 42 m | 25.7 |
| Morobora | 3000 m | 118 m | 5.5 % | 6 m | 5.6 |

**Die großen Formen stimmen bereits** — 3.5 Massive auf der Regionsfläche
gegen den Wunsch von 5. Der Fehler sitzt in den feinen Oktaven:

```python
tor = 1.0 / (1.0 + np.exp(-(form - wellen[k]) / (0.35 * wellen[k])))
gewicht = np.power(rauheit, k) * tor
```

Das Tor ist **einseitig**: es dämpft Oktaven, die GRÖSSER als die Formgröße
sind, aber nach unten läuft es offen bis zur feinsten Welle (47 m). Gebremst
wird nur durch `rauheit^k`, und bei `rauheit = 0.68` ist das schwach. Weil
das Nevadin zusätzlich das größte Relief aller Regionen hat (1050 m),
werden aus 23.9 % ganze **250 m Amplitude in Formen unter 375 m Breite** —
bei 1024 px ist eine 47-m-Welle zwei Pixel breit. Dieselbe Ursache zerhackt
den Geologie-Querschnitt.

**BEHOBEN 2026-08-25** durch ein zweiseitiges Tor (`FEINHEIT_TEILER = 8.0`):
122 → 24 Gipfel im Nevadin, Relief unter 375 m von 250 auf 67 m, und
`smoke_test_regionen_welt` von 8 auf 4 Befunde. Die Talsohlen-Redistribution
ist NICHT erledigt — `potenz` ist medianerhaltend und kann sie nicht leisten
(über den ganzen Bereich 0.6–2.5 gemessen). Sie braucht eine echte
hypsometrische Kurve als neuen Mechanismus.

**Ursprünglich geplante Abhilfe:**
1. Ein **oberes Tor**: Wellenlängen deutlich unter der Formgröße zusätzlich
   dämpfen, statt sie nur mit `rauheit^k` zu multiplizieren.
2. **Height Redistribution zugunsten der Talsohle** (Nutzerwunsch), also
   eine Kurve, die mehr Fläche auf tiefe Lagen legt.
3. Danach `smoke_test_regionen_welt.py` gegenprüfen — der Nevadin-Hang
   steht dort ohnehin bei 36.5 statt 27.0 und sollte sich mit bessern.

---

## 4. Die Ziele, nach Nutzervorgabe geordnet

### 4.1 Abschalt-Häkchen (Flussnetz, Erosionsfilter, Küstentypen) — **ERLEDIGT 2026-08-25**

*"kannst du mir einmal für flussnetzwerk und erosionfilter und küstentypen
jeweils checkboxen einfügen, mit denen ich die effekte immer auch ausschalten
kann?"*

Billig und mit großem Nutzen: erst damit lässt sich sehen, welche Stufe
welchen Anteil am Bild hat. **Gilt die stehende Regel: 2D und 3D in
derselben Änderung** (siehe CLAUDE.md).

### 4.2 Terrain-Reiter umbauen

* `AMPLITUDE` verschwindet aus *Shape*.
* Neu: **Regions-Dropdown** über *Noise Detail*. Die Regler springen auf die
  Werte der gewählten Region und lassen sich dort nachstellen.
* Die vier Detail-Regler bleiben, **plus Amplitude** (höchster Berg).

**ENTSCHIEDEN 2026-08-25 (Nutzer): nur für die aktuelle Karte.** Die neun
Katalogwerte in `core/terrain_weltkarte.py` bleiben die Vorgabe; eine
Anpassung wird als Überschreibung durchgereicht und gilt bis zum
Zurücksetzen. **Die Tests messen weiter gegen den Katalog** — sonst prüften
sie nur noch, was zuletzt eingestellt war.

**Das ist der größte Eingriff.** Die neun
Parametersätze sind heute Modulkonstanten in `core/terrain_weltkarte.py`,
und die gesamte Eichung samt `smoke_test_regionen_welt.py` hängt an ihren
Werten. Sie zur Laufzeit editierbar zu machen heißt: ein Überschreibungs-
Dict von der GUI in `weltfeld()` durchreichen, mit den Katalogwerten als
Vorgabe. Die Tests müssen weiter gegen die Katalogwerte messen, sonst prüfen
sie nur noch, was der Nutzer zuletzt eingestellt hat.

### 4.3 Flussnetz-Reiter entrümpeln — **ERLEDIGT 2026-08-26**

Von 11 Reglern die fünf wichtigsten auf die Seite. Nach Wirkung auf das
Bild sind das voraussichtlich `SPACING_M` (Dichte des Netzes),
`VALLEY_WIDTH`, `VALLEY_FORM`, `INCISION_SHARE` (Tiefe) und `MEANDER` —
**zu belegen, nicht zu raten**: je Regler eine Messreihe über den
Wertebereich gegen eine Bildkennzahl.

### 4.4 Eine Wasserkarte statt rot/grün — **ERLEDIGT 2026-08-26**

*"ich verstehe noch immer nicht die mehrteilung mit roten und grünen flüssen
... können wir nur eine karte haben die darstellt wie viel wasser für die
flüsse berechnet wurde?"*

Es gibt bereits `flow_map` (akkumulierter Abfluss, seit heute mit
Log-Skala). Der Vorschlag: **`flow_map` wird die eine Flusskarte**, die
Generationen-Einfärbung entfällt oder wird ein Häkchen. Gleichzeitig zu
klären: *"Baeche (Mikro) gibt es ja auch gar nicht"* — die Stufe existiert
im Menü, aber nicht im Ergebnis.

### 4.5 Erosionsfilter verständlich benennen — **ERLEDIGT 2026-09-18 (Ticket #61)**

**Die Messung dafür liegt seit 2026-08-26 vor** (mittlere Höhenänderung über
den vollen Reglerweg): GULLY_SIZE_M 47.04 m, GULLY_WEIGHT 4.27 m,
STRENGTH 3.44 m, OCTAVES 1.17 m, RIDGE_ROUNDING 0.45 m,
CREASE_ROUNDING 0.42 m, **DETAIL 0.18 m**. `DETAIL` ist 260-mal schwächer
als der stärkste Regler und im Bild über seinen ganzen Bereich nicht
unterscheidbar — erster Streichkandidat. Die beiden Rundungen sind
zusammenfassbar.

Sieben Regler, alle funktional, aber die Namen erklären sich nicht
(`CREASE_ROUNDING`, `RIDGE_ROUNDING`, `GULLY_WEIGHT`). Ziel: **fünf Regler
mit Namen, die sagen, was man sieht.** Zusammenfassen, wo zwei Regler
dasselbe Bild in zwei Richtungen drehen.

**Umgesetzt wurde die Umbenennung, NICHT das Zusammenlegen** (das Streichen
von `DETAIL`/Zusammenfassen der Rundungen bleibt offen, ist eine eigene
Entscheidung mit sichtbarer Wirkung auf die GUI, kein reines Umbenennen).
Ticket #61 hat die drei irreführenden `EROSION_FILTER`-Klassenattribute in
`gui/config/value_default.py` nach Wirkung umbenannt:

| Alt | Neu |
|---|---|
| `DETAIL` | `GULLY_REACH` |
| `GULLY_WEIGHT` | `GULLY_VS_SHARPNESS` |
| `CREASE_ROUNDING` | `VALLEY_ROUNDING` |

`STRENGTH`, `GULLY_SIZE_M`, `RIDGE_ROUNDING`, `OCTAVES` blieben unveraendert
(bereits wirkungsbeschreibend). Die Parameterschluessel (`erosion_filter_detail`
usw.) und die ATEF-Quellennamen im Shader (`EROSION_DETAIL` usw. in
`shaders/terrain/ATEF_Buffer_A.comp`) blieben unangetastet - siehe Abschnitt
"Der Anspruch: ATEF von Rune Johansen" oben, das war bereits vorher
beschlossen. Reines Python-Refactoring, Erosionsergebnis bit-exakt gleich
(gemessen per `numpy.array_equal`).

### 4.6 Geologie-Querschnitt gröber rechnen

*"vielleicht kann man hier die gesamte auflösung und rechengenauigkeit etwas
herunterfahren. also es könnte 8 mal weniger genau sein."*

Achtung: die Zacken im Querschnitt sind **zum Teil dieselbe Ursache wie die
Alpenspitzen** (4.3). Erst 3. reparieren, dann messen, wie viel Gröbe
überhaupt noch nötig ist — sonst wird eine Auflösung gesenkt, um einen
Geländefehler zu verstecken.

**BEHOBEN 2026-09-18 (Ticket #62).** Die Vorbedingung aus 4.3 war laengst
erfuellt (`FEINHEIT_TEILER`, 2026-08-25). Gemessen (headless, Agg-Backend,
reale `GeologySystemGenerator`-Daten, 30 Wiederholungen je Groesse):
Zeichnen+Rasterisieren einer Querschnittzeile kostete bei 1024 px 188 ms, bei
512 px 115-165 ms, ganz ueberwiegend in matplotlib `PolyCollection.draw` -
das skaliert linear mit der Punktzahl je Schicht-Polygon, die Slice-Extraktion
selbst ist mit 0.002 ms vernachlaessigbar. Das bestaetigt: es wird tatsaechlich
feiner GEZEICHNET als sichtbar ist - nicht feiner gerechnet, `layer_boundaries`
und `terrain_height` bleiben normale Datenprodukte, an denen `layer_id_map`/
`rock_map` haengen, und wurden NICHT angefasst.

Umgesetzt in `gui/widgets/map_display_2d.py::_render_geology_cross_section()`:
die fuer genau diesen Plot entnommene Stichprobe (Koordinate, Schichtgrenzen,
Terrainlinie, Intrusionsdistanz) wird auf `QUERSCHNITT_MAX_PUNKTE = 512`
Punkte ausgeduennt, per exaktem Rasterindex (`np.linspace(...).round()`,
kein Interpolieren) - dieselben Werte an denselben Stichprobenpunkten wie
vorher, nur seltener. Bei 1024 px sinkt die Zeichenzeit dadurch von 188 ms auf
123 ms (-35 %); darunter (256/128 Punkte: 93/81 ms) sinkt vor allem noch der
matplotlib-Fixkostenanteil, waehrend das Risiko waechst, schmale Verwerfungen
zu verlieren - 512 statt der vorgeschlagenen 128 (Faktor 8) ist der
Kompromiss. Bei map_size ≤ 512 ist die Aenderung ein reiner No-Op (keine
Regression fuer die haeufigeren kleineren Kartengroessen).

Geprueft: `tests/smoke_test_display_2d.py` (30/30 Darstellungen zeichnen
weiterhin etwas, darunter der Geologie-Querschnitt) sowie eine gesonderte
Bitgenauigkeits-Pruefung der Ausduennungslogik (512 von 1024 Punkten erhalten,
Rand nicht abgeschnitten, alle erhaltenen Werte bitgleich zum Original).

### 4.7 ~~Erosion reaktivieren~~ — gestrichen (Ticket #63, veraltet)

Dieser Punkt ist veraltet: die Erosion läuft längst, `EROSION_AKTIV = True`
in `gui/config/value_default.py:1088`. Der Punkt wurde faelschlich als
Beleg benutzt, um die roten Erosionstests jahrelang wegzuerklaeren ("die
Erosionskette ist bewusst abgeschaltet") — die zentrale Korrektur dazu
steht in `docs/TESTBERICHT.md` Abschnitt 3. Der dort ebenfalls genannte
GPU/CPU-Befund `erosion_gpu_parity` (frueher als Faktor 385 gefuehrt) ist
kein offener Befund mehr - er war ein zu grobes GPU-Meldeintervall, laengst
behoben und am 16.09.2026 erneut gruen gemessen; die Korrektur dazu steht
ebenfalls in `docs/TESTBERICHT.md` Abschnitt 3.

---

## 4.8 Küstengebiete im Hinterland — **ERSTER AUSBAU ERLEDIGT 2026-08-26**

**Das Problem, beziffert.** Ein Küstenarchetyp bestimmt heute die ersten
178–208 m voll (p1) und blendet bis 500–550 m aus (p2). Die Region ist rund
**7100 m** breit. Der Archetyp regiert also **7 % des Wegs** von der Küste
zur Regionsmitte, dahinter übernimmt ein Rauschfeld, das von ihm nichts
weiß. Das ist die sichtbare Naht zwischen den farbigen Küstenformen und dem
grünen Hinterland.

**Der Entwurf.** Entlang der Küste tragen die Voronoi-Zellen den Wert ihres
Küstenarchetyps. Von dort füllt eine Breitensuche über den
Zellnachbarschaftsgraphen die Region nach innen auf. Ziel: drei
größtenteils zusammenhängende Gebiete je Region, jedes 15–50 % der Fläche.
Die Werte gehen an den Gebietsgrenzen ineinander über, auch über
Regionsgrenzen hinweg: `[1][1][1.33][1.66][2][2]`.

**Beide Hälften existieren bereits in diesem Programm:**

* `seegliederung()` macht genau diese Breitensuche über den Zellgraphen —
  Seegrad 0…4 von der Küste nach außen, Tabelle Grad→Tiefe, Glättung.
* `voronoi_regionen()` löst die Flächenquote: eine Regelschleife
  `vorteil += 0.35 * log(soll / ist)` über 60 Runden regelt die
  Zellzuteilung auf `flaeche_soll` ein. Der Kommentar dort hält fest, dass
  die lineare Fassung schwang und die logarithmische stabil ist.

**Entscheidungen des Nutzers (2026-08-25/26):**

* **Erster Ausbau nur mittlere HÖHE.** *"lass uns anfangen nur mit
  höhenwerten, also wie hoch die mittlere höhe ist. die anderen werte wird
  von der region vererbt und später kann das bei bedarf angepasst werden."*
  `formgroesse_m`, `rauheit`, `potenz`, `relief_m` erben also unverändert
  von der Region.
* **Nevadin ist ein Sonderfall.** *"hier gibt es entweder eine saat, dann
  geht das aber immer nur wenige voronoi von der küste rein. meistens ist es
  aber einfach nur alpin."* Also: Küstensaat greift nur wenige Zellen weit,
  der Rest bekommt einen eigenen Typ „alpin", der die heutige Optik erzeugt.
* **Statt der Ordinalzahl 1/2/3 der echte `hoehe_faktor`** (Hügelland
  1.4 / 0.8 / 0.4). Die Ordinalzahl verlöre, dass Moher das 3,5-fache von
  Luce-Bay ist.
* **Noch offen, vom Nutzer als Überlegung eingebracht:** ob die Höhenfaktoren
  im Alpinen mit dem Abstand zur Küste weiter steigen dürfen — *"so das am
  rand 800 m berge sind und in der mitte ein 1400 m berg steht"*.

**Die harte Nebenbedingung.** Das flächengewichtete Mittel der Gebietswerte
muss den Katalogwert der Region treffen. `hoehe_m`, `relief_m`,
`wasser_soll`, `flaeche_soll` sind alles *Regionsmittel*, und die gesamte
Eichung hängt daran.

**Was sonst noch zu regeln ist:**

* Binnenregionen ohne Küste (Nevadin auf 62 von 64 Karten) — gelöst über
  den Sondertyp „alpin".
* Die Küstentyp-Anteile streuen 8–21 Punkte je Karte, eine Region kann zu
  80 % einen Typ tragen. Ohne die Regelschleife gäbe das ein Gebiet statt
  dreier.
* Inseln säen aus ihrer **eigenen** Küste, nicht über Seezellen — sonst
  hinge eine Insel an einem Gebiet auf dem fernen Festland.
* `voronoi_regionen()` berechnet `etikett` (die Zell-Etiketten), gibt sie
  aber nicht zurück. Muss raus.

**Vorarbeit, erledigt 2026-08-25/26:** p1 nochmal 15 % näher (210/245 →
178/208), kostet auf beiden Tests nichts. Die variable p2-Reichweite aus der
Küstenähnlichkeit ist gebaut und gemessen, steht aber auf AUS — sie kostet
die Zusicherung „flache Küste bleibt flach". Begründung samt Messreihe bei
`P2_MAX_M` in `core/vektor_kueste.py`. **Die Reichweitenfrage ist neu zu
stellen, wenn das Gebietssystem steht.**

---

## 4.9 Live-Regler und Rechnen je Reiter (Nutzerfrage 2026-08-26)

*"wenn jemand die slider verändert im terrain bereich, dann wird live eine
veränderung dargestellt und alles danach invalidiert. wir teilen das ganze
auf in gelbe slider (wenn diese verändert werden, dauert es länger bis eine
änderung stattfindet) und normale slider (live-änderungen werden direkt
aufgebaut). ist das realistisch?"*

**Antwort: ja — aber nur auf einer Vorschau-Auflösung.** Gemessen
2026-08-26:

| px | m/px | `weltfeld` | Erosionsfilter | Flussnetz + Täler | Vektorküste |
|---:|---:|---:|---:|---:|---:|
| 192 | 111 | 1.07 s | 0.12 s | — | — |
| 256 | 83 | 2.09 s | 0.31 s | — | — |
| 384 | 55 | 2.34 s | 0.67 s | 1.93 s | 1.30 s |
| 512 | 42 | 3.96 s | 1.04 s | 3.37 s | 2.28 s |
| **1024** | 21 | **19.38 s** | **13.28 s** | 5.94 s | 13.47 s |

Der Oktavenstapel ist bereits zwischengespeichert (0.00 s aus dem Cache,
0.03–0.08 s frisch) und spielt keine Rolle.

**Bei 1024 px ist Live ausgeschlossen** — 19 s je Reglerbewegung. **Bei
192–256 px ist es bequem möglich.** Der LOD-Manager, der so eine
Vorschaustufe liefern kann, ist bereits da.

**Der eigentliche Hebel ist aber nicht die Auflösung, sondern das
Zwischenspeichern JE STUFE.** Die Kette ist fast linear:

    Grundrauschen -> Küste -> Erosionsfilter -> Flussnetz -> Täler

Wer nur an den Erosionsreglern dreht, braucht das Grundgelände nicht neu —
es liegt schon da. Dann kostet eine Reglerbewegung **0.31 s bei 256 px**,
nicht 2.09 s. Genau das ist das vom Nutzer angesprochene *"tab-weise
rechnen"*, und es ist derselbe Umbau.

**Vorschlag für die Reglerklassen, aus der Messung statt geraten** (Vorschau
bei 256 px, Zwischenspeicher je Stufe vorausgesetzt):

| Klasse | Regler | Kosten |
|---|---|---|
| **normal (live)** | die sieben Erosionsfilter-Regler | 0.31 s |
| **normal (live)** | Talbreite, Taltiefe, Talform (nur `taeler_eingraben`) | ~0.1 s |
| **gelb** | Talabstand, Flüsse folgen dem Tiefland (Netz neu) | ~1.2 s |
| **gelb** | Rausch- und Regionsregler, Küstentypen (`weltfeld` neu) | ~2.1 s |
| **gelb** | Kartengröße, Seed (alles neu) | ~2.1 s |

**Nebenbefund, ungeklärt:** der Erosionsfilter skaliert schlecht — von 512
auf 1024 px vervierfacht sich die Pixelzahl, die Zeit steigt aber um das
12.8-fache (1.04 → 13.28 s). Bei 1024 px ist er mit 13.3 s der teuerste
Einzelposten der ganzen Terrainstufe. Ursache nicht untersucht.

### Erosionsfilter je Region

*"dabei haben unterschiedliche regionen unterschiedliche filter. wir können
also terrain noise und erosion filter über das drop down auswählen."*

Machbar, und zwar nach demselben Muster wie das Rauschen schon heute:
`parameterfeld()` mischt jeden Regionsparameter über die Voronoi-Gewichte zu
einem vollen Feld. **Die einzige Hürde:** `erosion_filter()` in
`core/terrain_erosion_filter.py` liest jeden Parameter durch `float(...)`,
nimmt also nur Skalare. Der Rest der Funktion ist numpy-vektorisiert über
`px`/`py` — Feldparameter würden von selbst broadcasten.

Zu tun wäre: die `float()`-Umwandlungen für die stetigen Parameter
(`strength`, `gully_weight`, `rounding`, `detail`, `scale`) fallen lassen,
`octaves` skalar behalten (eine Schleifenlänge kann kein Feld sein).

**Nicht in Frage kommt, den Filter neunmal zu rechnen und zu überblenden** —
das wären bei 1024 px 9 × 13.3 s = zwei Minuten.

### Der Anspruch: ATEF von Rune Johansen

Der Nutzer hat die Quelle verlinkt
(blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html). **Das
ist bereits unser Filter** — `core/terrain_erosion_filter.py` ist eine
zeilenweise Portierung von `ATEF_Buffer_A.comp`, mit Quellenangabe und
Copyright-Vermerk.

Das ändert Punkt 4.5: die Reglernamen (`gully_weight`, `ridge_rounding`,
`crease_rounding`) sind **die Namen der Quelle**, nicht willkürlich gewählt.
Sie umzubenennen würde den Bezug zur Referenz kappen. Besser: die deutschen
Beschriftungen erklären, was man sieht, und der Parameterschlüssel behält
den ATEF-Namen.

**Und eine Korrektur an meiner eigenen Empfehlung vom selben Tag:** ich
hatte `strength 0.60` und `gully_weight 0.90` vorgeschlagen. Die Referenz
erzeugt ihre eigenen Bilder mit `erosion_strength 0.22` und
`gully_weight 0.5` — das sind auch unsere `ATEF_DEFAULTS`. Ich lag also weit
über dem, was die Quelle für denselben Look benutzt. Warum unser Gelände
mehr braucht, ist eine offene Frage (Verdacht: die Referenzbilder zeigen
wenige Kilometer bei hoher Auflösung, unsere Karte 21 km bei 21–55 m/px).

---

## 4.10 ZIELBILD DER WOCHE: drei Ansichten nacheinander (Nutzer 2026-08-26)

*"als ziel würde ich vorschlagen, dass wir am anfang eine regionen-ansicht
haben ... dann gehts in kontinent sicht mit map size, form des kontinents ...
dann kommt flussnetzwerke und auch hier sollte eine live sicht möglich sein.
ab dann kann dann der rest generiert werden."*

Die Reihenfolge hat einen sachlichen Vorteil, der nicht offensichtlich ist:
**sie ordnet die Arbeitsschritte nach ihren Kosten.** Die teuren Stufen
(Vektorküste, Erosionsfilter auf voller Auflösung, Biome, Siedlungen) kommen
erst ganz zum Schluss, wenn nichts mehr eingestellt wird.

### Gemessen 2026-08-26 — alle drei sind live-fähig

| Ansicht | 128 px | 192 px | 256 px |
|---|---:|---:|---:|
| Region (EIN Parametersatz, kein Kontinent, keine Küste) | 0.084 s | 0.204 s | 0.324 s |
| Kontinentform allein | 0.037 s | 0.067 s | 0.108 s |
| Flussnetz + Täler | 0.21 s | 0.48 s | 0.72 s |

Alles unter einer Sekunde. **Für die Regionsansicht braucht es nicht einmal
einen Zwischenspeicher je Stufe** — sie rechnet fast nichts: Oktavenstapel
(0.01–0.02 s) plus Erosionsfilter. Der Filter ist dort der ganze Aufwand.

### Was je Ansicht zu bauen ist

**1. Regionsansicht** (128–256 px, rechteckig, Dropdown je Region)

* Eine Funktion "ein Regionsfeld auf Regionsmaßstab" — Oktavenstapel mit
  `mpp` für ~7.1 km Kantenlänge, EIN Parametersatz statt der neun gemischten
  Felder, dann der Erosionsfilter. Der `mpp`-Weg dafür ist bereits da
  (`oktavenstapel(..., mpp=...)`, von `tools/inseltest.py` benutzt).
* **Rechteckig**: `oktavenstapel()` erzeugt heute (size, size) aus EINER
  Achse. `noise2array()` nimmt zwei Achsen — die Änderung ist klein, muss
  aber gemacht werden.
* Die Regler schreiben in das Überschreibungs-Dict aus 4.2 (Entscheidung
  vom 2026-08-25: gilt nur für die aktuelle Karte).

**EIN VORBEHALT, der klar sein muss:** die Regionsansicht zeigt eine Region
FÜR SICH. Auf der fertigen Karte wird sie über die Voronoi-Gewichte mit
ihren Nachbarn verschmolzen, von den Küstenarchetypen umgeformt und vom
Gebietssystem in der Höhe verschoben. Die Vorschau ist also der CHARAKTER
der Region, nicht ihr Aussehen auf der Karte. Wer das verwechselt, dreht
später an den falschen Reglern.

**2. Kontinentansicht** (Kartengröße, Kontinentform)

* Ein Formregler links rund / Mitte länglich / rechts viele Ausläufer.
  `kontinentform()` baut die Form heute aus einer Zentralscheibe
  (Radius 0.34·halb) plus Lappen (Abstand 0.55–0.95·halb, Radius
  0.26–0.52·halb), weich vereinigt (`weich = 0.055·halb`), minus 2–3
  abgezogene Buchten. **Der Regler ist eine Interpolation genau dieser
  Konstanten:**
  - *rund*: Kern groß, Lappen nah und groß, Vereinigung weich
  - *viele Ausläufer*: Kern klein, Lappen fern und klein, Vereinigung hart
  - *länglich* braucht eine ZUSÄTZLICHE Zutat: die Lappenmitten entlang
    einer Achse stauchen. Das ist keine Interpolation derselben Größen,
    sondern Anisotropie — beim Bauen nicht übersehen.
* Die Flächeneichung (`ziel = KONTINENT_KM²` per Intervallhalbierung) bleibt
  unangetastet: der Kontinent behält seine Fläche, egal welche Form.

**3. Flussnetzansicht** — die fünf Regler stehen seit 2026-08-26 auf dem
Reiter. Es fehlt nur der Vorschau-Pfad auf 128–256 px.

**4. Danach der Rest** wie heute, auf voller Auflösung.

### Empfohlene Reihenfolge innerhalb des Ziels

1. **Regionsansicht** — sie ist in sich abgeschlossen, braucht weder
   Kontinent noch Zwischenspeicher, und dort wird die Optik entschieden.
2. **Kontinentansicht** — der Formregler ist der einzige neue Mechanismus.
3. **Flussnetz live** — nur noch der Vorschau-Pfad.
4. Erst danach der Zwischenspeicher je Stufe (4.9) für die Vollkarte.

---

## 4.11 Die Regionsansicht im Einzelnen (Nutzerplanung 2026-08-26)

### Maßstab — die Rechnung stimmt, aber das Fenster ist zu klein

*"128 px (aber pixel zu abstand ist der gleiche wie in der
kontinentalansicht. ergo 21000m/1024px*128px? stimmt das?)"*

Die Rechnung stimmt: 21300/1024 × 128 = **2662 m**. Nur zeigt dieses Fenster
**37 % der Regionsbreite** (eine Region ist 7100 m). Und gemessen an der
größten Geländeform der Region:

| Region | `formgroesse_m` | passt in 2.66 km |
|---|---:|---:|
| Nevadin | 3800 m | **0.7 mal** |
| Morobora | 3000 m | 0.9 mal |
| Samarcia | 2600 m | 1.0 mal |
| Thalassia | 1100 m | 2.4 mal |

**Im Nevadin sähe man weniger als eine Bergform.** Darin lässt sich weder
der Charakter beurteilen noch eine Küste mit Hinterland zeigen.

| Auflösung | ganze Region | Kosten (gemessen) |
|---|---|---|
| 128 px | 55.5 m/px | 0.084 s |
| **256 px** | **27.7 m/px** | **0.324 s** |
| 384 px | 18.5 m/px | ~1.3 s |

**Empfehlung: 256 px auf 27.7 m/px** — ganze Region, live, und nur ein
Drittel gröber als die Kontinentalansicht (20.8 m/px). Der Vorbehalt dazu:
die feinsten Rinnen sehen in der Vorschau etwas gröber aus als später auf
der Karte.

### Aufbau je Region (Nutzervorgabe)

| Region | Vorschau zeigt |
|---|---|
| Hügelland | 3/4 Land (Standardrauschen), 1/4 Meer, dazwischen Moher-Klippen entlang einer leicht geschwungenen Linie quer durch die Karte |
| Thalassia | 1/2 Festland, 1/2 Inseln. Linke Hälfte auf > 5 m geklemmt (dort entsteht kein Meer), rechte frei |
| Skerrheim | links und rechts Geiranger-Küste, in der Mitte ein leicht geschwungener Fjord |
| Morobora | mittlere Küste (nicht so steil) mit etwas Seeeis, sonst 3/4 zu 1/4 |
| Samarcia, Atlantikküste, Nebelrode, Macchia | wie Hügelland, 3/4 zu 1/4 |
| Nevadin | keine Küste, nur Berge |

Das Meergebiet bekommt die Voronoi-Vertiefung, die es schon gibt
(`seegliederung()`).

**Zur Frage "sieht man den Erosionsfilter an den Klippen, falls wir das
vektorisiert haben?" — ja, und zwar ohne Zutun.** Die Küste ist als Vektor
beschrieben (`VektorKueste`, Polylinie mit Stationen in Metern) und wird von
`als_raster()` aufs Raster gelegt. Der Erosionsfilter läuft DANACH
(`weltfeld` → `_weltkarte_erosionsfilter` → Flussnetz). Er arbeitet also auf
dem fertig geformten Kliff.

### Zweite Ansicht "Höhenwerte" — und ein Fehler, den sie aufdeckt

*"höhenwert vom küstenprofil sei die mittlere höhe von 400 bis 700 m tiefe
im hinterland (zweipunktmethode). kann man hier pro profilschnitt einen wert
mit dranhängen?"*

**Die Methode ist besser als das, was heute im Gebietssystem steht — und
deckt einen Fehler von mir auf (2026-08-25).** `kuestengebiete()` rangiert
die Hinterlandhöhe nach `hoehe_faktor` aus dem Archetypkatalog. Der
beschreibt aber das UFER. Gemessen im Band 400–700 m kippt die Reihenfolge
in **vier von neun Regionen**:

| Region | Archetyp | `hoehe_faktor` | h(400–700 m) |
|---|---|---:|---:|
| Skerrheim | Fjordbucht | 0.30 | **195 m** |
| Skerrheim | Schärenküste | 0.50 | 15 m |
| Thalassia | Kreta-Buchten | 0.80 | **202 m** |
| Thalassia | Santorini-Kliff | 1.15 | 120 m |
| Macchia | Cinque-Terre | 1.00 | **252 m** |
| Macchia | Amalfi-Steilküste | 1.60 | 204 m |
| Morobora | Labrador-Buchten | 0.70 | **90 m** |
| Morobora | Kola-Steilküste | 1.10 | 58 m |

Eine Fjordbucht hat ein niedriges Ufer und steile Wände direkt dahinter.
**Das Gebietssystem soll das Hinterland setzen, nicht das Ufer** — also
gehört der gemessene Bandwert hinein, nicht der Katalogfaktor.

**Wie der Wert anzuhängen ist — drei Wege, Empfehlung a):**

**UMGESETZT 2026-08-26** als `GEMESSENE_HINTERLANDHOEHE` in
`core/vektor_kueste.py`, abgeleitet aus `MESS_PROFIL_M_JE_ARCHETYP`. Das
Gebietssystem benutzt sie statt `hoehe_faktor`; die Ansicht „Höhenfaktor"
im Terrain-Reiter zeigt sie in 2D und 3D.

**Zum Maßstab entschieden (Nutzer 2026-08-26):** verkleinern in beiden
Achsen, **außer bei den flachen Typen je Region** — *"dann haben wir zB
auch kürzere strände und das mag ich nicht so"*. Noch nicht umgesetzt.

* **a) je SEGMENT**, aus dem Profil seines Archetyps im Band 400–700 m.
  Deterministisch, kostet nichts, braucht keine neuen Daten, und es ist
  genau die Körnung, die das Gebietssystem säht (es liest Segmente an
  Zellen). **Empfohlen.**
* b) je STATION mit örtlicher Streuung. Die Profile sind je ARCHETYP
  definiert — alle Stationen eines Segments bekämen denselben Wert, außer
  man liest ihn aus dem bereits überblendeten Rasterfeld (`MISCH_EXPONENT`
  mischt Nachbarsegmente ohnehin). Machbar, aber Mehraufwand ohne
  erkennbaren Gewinn.
* c) aus dem FERTIGEN Gelände an jeder Station messen. Am ehrlichsten, aber
  **zirkulär**: das Gebietssystem verändert genau dieses Gelände.

### Küstenform: deterministische Regeln (heute fehlend)

*"die Küstenform allgemein hat derzeit noch keine richtigen regeln."*

Verlangt sind Regeln, die (a) Regionsgröße und Seeanteil einhalten
(Abweichung erlaubt) und (b) 2–3 charakteristische Formen der
Vergleichsländer SIMULIEREN, nicht kopieren:

| Region | Charakter |
|---|---|
| Hügelland | zackig wie West Cork / Schottland, oder sanfter wie Moher |
| Samarcia | lange, leicht geschwungene Küsten (Algarve, Costa Brava, San Sebastián) |
| Macchia | ähnlich: lange Küsten, einzelne Inseln |
| Thalassia | gesprenkelt, mehr Buchten und Fjorde |
| Skerrheim | EIN großer tiefer Fjord mit 1–3 kleinen Armen, darin Geiranger-Typ |
| Morobora | große Formen |
| Nebelrode | Abfallen auf große Küsten, wie Atlantikküste (dort mit z. B. 1 Flussdelta) |

Heute entsteht die Küstenlinie als Nulldurchgang des Rauschens; die
Archetypen formen nur das PROFIL quer dazu, nicht den Grundriss. Diese
Regeln greifen also an einer Stelle an, an der es bisher gar keinen
Mechanismus gibt — das ist ein eigener Bauabschnitt, nicht eine Einstellung.

### Umbenennungen (Nutzerwunsch, noch offen)

* "Macchia" → **"Macchia"** (vom Nutzer bereits so benutzt)
* "Thalassia" → mythisch klingender Name, Vorschlag des Nutzers
  "Athenia oder so"

**Kosten:** Regionsnamen sind Schlüssel in `KUESTEN_ARCHETYPEN`,
`MESSWERTE_JE_REGION`, `MESS_REICHWEITE_M`, `KUESTEN_TIEFE_JE_REGION` und in
mehreren Tests. Mechanisch, aber an vielen Stellen — in EINEM Durchgang
machen, nicht nebenbei.

---

## 4.12 Maßstab: Küstenprofile gegen Regionshöhen — **ERLEDIGT 2026-08-26**

*"wenn ich sehe dass Skerrheim nur 200m höhe hat (im Profil) und wir aber
einen faktor zu den höhen haben ... dementsprechend sollten die
küstenprofile von der höhe etwas gestaucht sein, was denkst du?"*

### Der Befund: zwei Maßstäbe, die nie miteinander verrechnet wurden

Die Regionsparameter sind vom Nutzer als Zielbild gesetzt (Nevadin
1050 m Relief auf 3800 m Formgröße). Die Küstenprofile sind an echten DEMs
in ECHTEN Metern gemessen (`MESS_PROFIL_M_JE_ARCHETYP`, 0–900 m). **Die
beiden Skalen wurden nie aufeinander bezogen.**

Verhältnis "höchstes Profil im Band 400–700 m" zu "halbem Regionsrelief":

| Region | Relief/2 | Profil | Verhältnis |
|---|---:|---:|---:|
| Hügelland | 40 m | 148 m | **3.73** |
| Skerrheim | 242 m | 559 m | **2.31** |
| Samarcia | 55 m | 93 m | 1.70 |
| Macchia | 156 m | 252 m | 1.61 |
| Morobora | 59 m | 90 m | 1.52 |
| Nebelrode | 67 m | 95 m | 1.41 |
| Thalassia | 202 m | 202 m | 1.00 |
| Atlantikküste | 74 m | 60 m | 0.82 |
| Nevadin | 525 m | 187 m | **0.36** |

**Streuung 10-fach.** In sieben von neun Regionen ragt die Küste höher auf
als das Land dahinter — das ist genau, was der Nutzer gesehen hat. Im
Nevadin ist es umgekehrt.

**Ein globaler Faktor löst das nicht.** Der Median liegt bei 1.52; ein
Faktor 0.66 würde das Hügelland noch bei 2.5 lassen und das Nevadin auf
0.24 drücken.

### Das Gelände selbst ist NICHT gestaucht — nur kleiner

| | Formbreite | Höhe | Verhältnis |
|---|---:|---:|---:|
| unser Nevadin | 3800 m | 1050 m | 0.28 |
| echte Alpen (Massiv) | ~15000 m | ~3500 m | 0.23 |
| unser Skerrheim | 1400 m | 485 m | 0.35 |
| echter Sognefjord | ~10000 m | ~1500 m | 0.15 |

Waagerecht UND senkrecht ist um einen ähnlichen Faktor verkleinert - die
NEIGUNG bleibt also etwa erhalten, im Skerrheim ist sie sogar mehr als
doppelt so steil wie in echt. Die Welt ist kleiner, nicht flacher.

### Der Konflikt zweier Nutzervorgaben, und wie er sich auflöst

* **2026-08-24:** *"die kuestenprofile sind dadurch nicht gestreckt oder
  gestaucht sondern haben das gleiche hoehen zu tiefen verhaeltnis wie in
  echt."* — Deshalb ist `profil_m` in echten Metern und die frühere
  Skalierung auf das Regions-Höhenband (`MESSWERTE_JE_REGION`) wurde
  ABGELÖST.
* **2026-08-26:** die Profile sollen zur verkleinerten Welt passen.

**Beides zugleich geht — durch Skalierung in BEIDEN Achsen.** Ein Profil,
das 559 m auf 700 m steigt, wird bei Faktor 0.43 zu 240 m auf 300 m:
dieselbe Form, dieselbe Neigung, nur kleiner. Ein Modell im Maßstab. Eine
Stauchung NUR in der Höhe würde die Vorgabe vom 24.08. verletzen — die
Profile wären flacher als in der Wirklichkeit.

**Die Grenze dieses Wegs, offen benannt:** das Nevadin bräuchte Faktor
2.81, also eine STRECKUNG auf 2529 m — weit über p2 (höchstens 700 m). Das
geht nicht. Praktisch ist es fast gegenstandslos, weil das Nevadin auf
62 von 64 Karten überhaupt keine Küste hat; die Regel muss den Fall
trotzdem abfangen (Faktor nach oben klemmen).

**Nach einer solchen Änderung neu zu prüfen:** p1/p2 (die Profile werden
kürzer), `smoke_test_kuestenprofiltreue` (es misst gegen die
UNskalierten Tabellenwerte und müsste den Faktor mitrechnen) und
`smoke_test_regionen_welt`.

---

## 4.13 Regionsnamen (Nutzerwunsch 2026-08-26)

Register: klassische und mythische Toponyme, wie die beiden schon
beschlossenen.

| heute | Vorschlag | Alternativen | warum |
|---|---|---|---|
| Thalassia | **Thalassia** | — | vom Nutzer gewählt |
| Macchia | **Macchia** | — | vom Nutzer benutzt |
| Atlantikküste | **Armorica** | Hesperia | der antike Name der Bretagne, heißt "am Meer" — die Vorbilder sind Bretagne und Vendée |
| Nebelrode | **Hercynia** | Sylvania | *Silva Hercynia*, der römische Name des deutschen Mittelgebirgswalds |
| Morobora | **Hyperborea** | Borea, Nivalia | das mythische Land hinter dem Nordwind |
| Hügelland | **Hibernia** | Cambria, Ardania | antiker Name Irlands; Vorbilder sind Moher, West Cork, Luce Bay |
| Skerrheim | **Skerrheim** | — | **vom Nutzer gewählt** 2026-08-26. Kompositwort aus *skerry* und *-heim* |
| Samarcia | **Samarcia** | — | **vom Nutzer gewählt** 2026-08-26, ausdrücklich mit c |
| Nevadin | **Rhipaia** | Kaukasia, Montaris | die mythischen Nordgebirge der Antike |

**Kosten der Umbenennung:** Regionsnamen sind Schlüssel in
`KUESTEN_ARCHETYPEN`, `MESSWERTE_JE_REGION`, `MESS_REICHWEITE_M`,
`KUESTEN_TIEFE_JE_REGION`, `TALFORM`-Tabellen und in mehreren Tests. Rein
mechanisch, aber an vielen Stellen — **in einem Durchgang machen**, sonst
bleibt die Hälfte stehen und niemand merkt es.

**Stand 2026-08-26:** Thalassia, Macchia, Skerrheim und Samarcia sind
gewählt. Die übrigen fünf hat der Nutzer als *"ok, aber geht besser"*
beurteilt, mit dem Hinweis: **es muss nicht alles Latein oder Griechisch
sein, Kompositwörter wie Skerrheim sind ausdrücklich erwünscht.**

Zweiter Anlauf für die offenen fünf, mit mehr Komposita:

| heute | Vorschläge |
|---|---|
| Hügelland | **Moherland**, **Kliffmark**, Hibernia, **Buchtland** |
| Morobora | **Frostmark**, **Eismark**, Hyperborea, **Nordwald** |
| Atlantikküste | **Wattmark**, **Gezeitland**, Armorica, **Flutmark** |
| Nebelrode | **Hercynia**, **Waldmark**, **Kreidemark** |
| Nevadin | **Firnheim**, **Gratland**, **Hochmark**, Rhipaia |

*-mark* (Grenzland), *-heim* und *-land* sind dieselbe Bildungsweise wie
Skerrheim und tragen die Landschaft im Namen.

---

## 5. Empfohlene Reihenfolge

Umgestellt am 2026-08-26 auf Nutzerwunsch: *"gehe die folge durch, aber ich
möchte das du mit dem zuletzt beschriebenen anfängst."*

| # | Schritt | Aufwand | Stand / warum hier |
|---|---|---|---|
| — | Häkchen (4.1) | klein | **ERLEDIGT 2026-08-25** |
| — | Nevadin-Spitzen (3.) | klein | **ERLEDIGT 2026-08-25** (122 → 24 Gipfel) |
| — | p1 15 % näher, p2-Mechanik (Vorarbeit zu 4.8) | klein | **ERLEDIGT 2026-08-26**, p2 gemessen und vorerst aus |
| — | Küstengebiete im Hinterland (4.8) | groß | **ERLEDIGT 2026-08-26** — nur mittlere Höhe, Rest erbt von der Region |
| **1** | **Eine Wasserkarte (4.4)** | klein | Reine Anzeige, kein Rechenrisiko |
| — | Flussregler messen und kürzen (4.3) | mittel | **ERLEDIGT 2026-08-26** — fünf auf dem Reiter, drei stille Fehler gefunden |
| — | Regionsansicht (4.10) | mittel | **ERLEDIGT 2026-08-26** — Rechenkern, Land-See-Aufbau, Vektorküste und Reiter |
| — | Kontinentansicht mit Formregler (4.10) | mittel | **ERLEDIGT 2026-08-26** — Mechanismus, Reiter und Durchreichung |
| **1** | **Flussnetz live (4.10)** | klein | Regler stehen, nur der Vorschau-Pfad fehlt |
| 4 | Erosionsfilter-Beschriftungen, `detail` streichen (4.5) | klein | Messung liegt vor; Schlüssel NICHT umbenennen (ATEF-Quelle) |
| 5 | Terrain-Reiter mit Regionsauswahl (4.2) | **groß** | Entscheidung liegt vor: Anpassung gilt nur für die aktuelle Karte, Katalog bleibt Vorgabe |
| — | Geologie-Querschnitt (4.6) | mittel | **ERLEDIGT 2026-09-18** (Ticket #62) — Zeichnen auf 512 Punkte ausgeduennt, 1024 px: 188 ms → 123 ms (-35 %) |
| — | ~~Erosion reaktivieren (4.7)~~ | — | **ENTFAELLT** — 4.7 gestrichen (veraltet, Erosion laeuft laengst); der GPU/CPU-Befund `erosion_gpu_parity` ist ebenfalls kein offener Testbefund mehr (16.09.2026 gruen gemessen, siehe `docs/TESTBERICHT.md` Abschnitt 3) |
