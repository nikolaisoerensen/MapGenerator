# Testbericht — Stand 2026-08-27

Erzeugt mit `tools/testlauf.py`, jede Datei als eigener Prozess.

**MESSWARNUNG:** diese Maschine schwankt um Faktor 2–3. Dieselbe feste
Referenzlast maß zwischen 0,80 s und 1,91 s ohne erkennbare Fremdlast.
Laufzeiten sind Größenordnungen, keine Messwerte.

Diese Datei zeigt den **jetzigen** Stand und wurde dafür neu geschrieben.
Die ausführlichere Fassung vom 24.08. mit der Herleitung der Längenquote
steht in der Versionsverwaltung (`git show HEAD:docs/TESTBERICHT.md`); die
fachliche Begründung je Befund steht in `docs/SITZUNGSLOG.md`.

## Das Wichtigste in fünf Zeilen

* **60 von 70 Testdateien bestehen**, Gesamtlaufzeit **1153 s (19,2 Min)**.
  Nach den beiden Korrekturen unten sind es **62 von 70**.
* **Ein Fehlschlag war neu und selbst verursacht** — `layer_2d_3d_parity`
  fand, dass die vier neuen Terrain-Karten in 3D **ohne Farbtafel** liefen.
  Behoben. Das ist der wichtigste Befund dieses Laufs.
* **Ein Fehlschlag war ein FEHLER IM TEST, nicht im Code** —
  `archetyp_verteilung` maß die Auflösung seiner eigenen Testkarte.
  Geklärt durch zwei Enden gegeneinander, behoben.
* **Sieben Fehlschläge sind bekannt und standen schon im Bericht vom
  24.08.** — die drei Erosionsbefunde, `regionen_welt`, `pipeline_outputs`,
  `settlement_placement`, `weather_temperature_direktnormierung`.
* **Kein einziger Absturz.** Alle Fehlschläge sind Zusicherungen, die eine
  Kennzahl gegen einen Zielwert halten.

---

## 1. Der neue Befund: vier Karten ohne Farbtafel in 3D (behoben)

```
[28/70] smoke_test_layer_2d_3d_parity.py                        FAIL(1)
          [FAIL] terrain/hinterland_height: Range-Mapping vorhanden
          [FAIL] terrain/river_order: Range-Mapping vorhanden
          [FAIL] terrain/river_water: Range-Mapping vorhanden
```

**Was passiert war.** Die vier neuen Karten (`river_water`, `river_order`,
`hinterland_height`, `voronoi_map`) bekamen am 26.08. ihre Farbskalen in
`CanvasSettings.CANVAS_2D["layer_ranges"]`. Damit sah die Sache erledigt
aus. Die 3D-Ansicht liest diese Tabelle aber **nicht direkt** — sie geht
über `_LAYER_RANGE_KEY_MAP` in `gui/widgets/map_display_3d.py`, und dort
fehlten alle vier.

**Warum es unsichtbar war.** `_colorize_layer()` findet keinen `range_key`,
fällt auf Auto-Skalierung ohne Farbtafel zurück und zeichnet **trotzdem
ein plausibles Bild** — nur mit anderen Farben und anderer Skala als 2D.
Kein Absturz, keine Warnung. Das ist derselbe stille Rückfall, den
CLAUDE.md an drei anderen Stellen beschreibt, und er hat sich hier zum
vierten Mal wiederholt.

**Gefunden hat es kein Mensch und kein Blick auf das Bild**, sondern der
Test, der genau diese Doppelregistrierung bewacht. Ohne ihn wäre es erst
beim Vergleich zweier Screenshots aufgefallen — oder gar nicht.

Behoben; `smoke_test_layer_2d_3d_parity.py` ist wieder 4/4 grün.

## 2. Geklärt: Vendée war ein Artefakt des Tests (behoben)

```
[ 4/70] smoke_test_archetyp_verteilung.py                       FAIL(1)
          [FAIL] jeder Archetyp im Mittel innerhalb 16 Punkten
                 - Vendee-Straende -16.7P
```

**Gemessen wurden zwei Enden der Kette bei zwei Auflösungen, je 8 Karten:**

| Vendée-Strände | Saatanteil | Längenanteil |
|---|---:|---:|
| 384 px (55 m/px) | −5,0 | **−20,1** |
| 768 px (28 m/px) | −4,9 | **−0,2** |

Das Saatende ist bei beiden Auflösungen gleich — **die Zuordnung stimmt.**
Der Verlust entsteht danach und verschwindet bei feiner Karte vollständig.
Die Gegenprobe liefert Bretagne-Klippen, Vendées Regionsnachbar: +11,4 bei
384 px, −5,8 bei 768 px. Die beiden tauschen genau das, was Vendée fehlt.

**Ursache:** `MIN_SEGMENT_M` (750 m) ist eine *absolute* Länge. Vendée hat
mit 0,18 km die kürzeste Reichweite der drei Atlantik-Typen; bei 55 m/px
fallen seine Zonen darunter und werden in `_segmente_schliessen()` in den
längeren Nachbarn eingeschmolzen. Der Test hat die Auflösung **seiner
eigenen Testkarte** gemessen, nicht die Verteilung.

**Was daraus wurde.** Der Test prüft jetzt beide Enden mit zwei Grenzen:
den Saatanteil scharf (10 Punkte; schlechtester gemessener Wert 5,2 über 24
Archetypen) und den Längenanteil weich (22 Punkte). Der Fehler, für den es
diese Datei gibt — ein Archetyp, der systematisch ausfällt — liegt 25 bis 50
Punkte daneben und fällt durch **beide** Grenzen; die weiche Längengrenze
verliert also nichts.

Auf der echten Karte (1024 px, 21 m/px) ist das Einschmelzen noch geringer
als bei 768 px. Der Effekt ist damit **kein Problem der erzeugten Karte**,
sondern eine Eigenschaft grober Testauflösungen. Notiert in
`docs/OFFENE_PUNKTE.md`.

## 3. Bekannt und unverändert

| Test | Befund | Stand |
|---|---|---|
| `erosion_field` | Farbskala `[0.5, 300]`, typischer Wert 0,22 | Farbskalenproblem, kein Rechenproblem — seit 24.08. |
| `erosion_quality` | Kanalnetz 45 px statt > 60, Ebenen 7,2 % statt 15–55 % | **ungeklärt** — frühere Erklärung ("Erosionskette abgeschaltet") war falsch, siehe unten |
| `regionen_welt` | 4 Befunde, u. a. Macchia Hang 18,6 statt 14,5 | die Küsten-Archetypen verstimmen die Regionseichung — in CLAUDE.md beschrieben |
| `pipeline_outputs` | 5 Befunde | seit 24.08. |
| `settlement_placement` | 6 Befunde, u. a. 2 Städte statt 1 je Kultur | seit 24.08. |
| `weather_temperature_direktnormierung` | 3 Befunde, Skerrheim 7,86 statt 8,60 K | seit 24.08. |

**`erosion_gpu_parity` stand bis zum 16.09.2026 ebenfalls in dieser
Tabelle**, mit "Export 38,5 gegen 0,1 m auf der CPU, echter Paritätsbruch,
Faktor 385". Das war schon lange falsch: `docs/SITZUNGSLOG.md`, Eintrag
"Erosion wieder eingeschaltet" vom 27.08.2026 (demselben Tag wie dieser
Bericht, aber offenbar danach), beschreibt den echten Fund — kein
Rechenfehler, sondern ein zu
grobes Fortschritts-Meldeintervall auf der GPU-Seite
(`PROGRESS_REPORT_INTERVAL = 500` statt des `CONVERGENCE_CHECK_INTERVAL = 25`
der CPU-Seite), das die GPU zwanzigmal zu lange weiterlaufen liess, bevor sie
ihre Konvergenz prüfte. Behoben, seither grün. Am 16.09.2026 erneut
gegengemessen (`tests/smoke_test_erosion_gpu_parity.py`, echte GPU, kein
Fallback): ein Schritt GPU gegen CPU Abweichung 0, Langlauf beide Seiten
0,1 m Export. Der Bericht behauptete drei Wochen lang das Gegenteil eines
bereits gelösten Befunds, weil er nach der Behebung nicht nachgezogen wurde.

**Die zwei verbleibenden Erosionsbefunde sind offen und nicht erklärt.**
Hier stand bis zum 16.09.2026 das Gegenteil: sie seien kein Zufall, weil die
Erosionskette bewusst abgeschaltet sei. Das war **doppelt falsch** (Ticket
#28 und #63 haben das unabhaengig voneinander gefunden und behoben): erstens
laeuft die Erosion seit dem 27.08.2026 im Betrieb mit -
`EROSION_AKTIV = True` steht in
[`gui/config/value_default.py:1088`](../gui/config/value_default.py#L1088)
und ist die einzig gueltige Aussage zum Schalter. Zweitens stimmte schon die
Zahl nicht: die Tabelle listete zu jenem Zeitpunkt drei Erosionsbefunde
(`erosion_field`, `erosion_gpu_parity`, `erosion_quality`), keine vier - und
`erosion_gpu_parity` war, wie oben beschrieben, ohnehin bereits behoben.

**Nachtrag 17.09.2026 (Ticket #30, im Anschluss an die obige Klärung):** die
Parität war zwar behoben, aber die eigentliche Frage — lohnt sich Parität
überhaupt, wenn der CPU-Pfad im Betrieb nie läuft — war noch offen. Gemessen
bei 1024 px, 21 m/px, headless über `GPUWorker` (echte GPU, kein
`QT_QPA_PLATFORM=offscreen`):

| Pfad | Ergebnis |
|---|---|
| GPU, voller Lauf bis Konvergenz | 14,1 s, 1250 Schritte |
| CPU, hochgerechnet auf dieselben 1250 Schritte (25 echte Schritte gemessen, 878 ms/Schritt) | 1097 s (18,3 min) |
| CPU, hochgerechnet auf `max_steps = 8000` | 7023 s (1,95 h) |

Faktor 78 bis 500 langsamer, je nachdem gegen welche Schrittzahl man
vergleicht — in jedem Fall Größenordnungen, nicht Prozente. Der CPU-Pfad ist
für reale Kartengrößen (512 px aufwärts) damit nicht praxistauglich; genau
deshalb existiert `MAX_CPU_RESOLUTION = 256` in
[`core/erosion_generator.py`](../core/erosion_generator.py) bereits seit
Längerem als harte Grenze.

Diese Grenze hatte aber eine Lücke: sie prüfte nur, ob zu Laufbeginn gar kein
GPU-Dispatch registriert war. Scheiterte die GPU stattdessen erst WÄHREND
eines laufenden Abschnitts (Treiberfehler, 30-s-Timeout in
`GPUWorker.submit`), fiel der Code bisher ungeprüft in die volle CPU-Schleife
— bei 1024 px also bis zu knapp zwei Stunden, ein stiller Hänger statt eines
Fehlers. Behoben in Ticket #30: derselbe laute `ValueError` wie beim
Start-Guard greift jetzt auch nach einem gescheiterten GPU-Abschnitt, wenn
die Auflösung über `MAX_CPU_RESOLUTION` liegt. Unterhalb der Grenze (z. B.
die 64-px-Testkarten in `smoke_test_erosion_gpu_parity.py`) bleibt der
CPU-Rückfall unverändert erlaubt — dort ist er schnell genug, um eine
legitime Notlösung zu sein, nicht ein stiller Fehler.

`erosion_field` hat eine eigene, vom Schalter unabhaengige Erklaerung; der
tatsaechliche Grund fuer `erosion_quality` bleibt **ungeklaert** - das ist
eine offene Aufgabe, kein erledigter oder bewusster Zustand.

Eine falsche Erklärung ist schlimmer als keine: keine Erklärung lädt zum
Nachsehen ein, eine falsche schliesst die Frage. Das gilt fuer die
Erosionskette ebenso wie fuer eine laengst behobene GPU-Messung, die
wochenlang als offen weitergefuehrt wurde.

**Nachtrag 17.09.2026 (Ticket #32, Seenfläche neu gemessen):** die letzte
belastbare Zahl stammte aus `docs/FLUESSE_UND_WASSER.md`, Block 3
(24.08.2026, 384 px): "11 Binnenseen über 4 Pixel, und KEINER liegt im
Skerrheim" — und Block 6 hielt für Morobora ausdrücklich fest: "Gibt es
schon Seen im Wassersystem? ... ist UNGEPRÜFT." Das war eine reine Messung,
keine Korrektur (`bereich:wasser, ready-for-agent, test`), also wurde nichts
am Code geändert.

Gemessen über die echte Pipeline (`tools/weather_lab.py:run_pipeline()`,
echter Dispatcher, keine nachgebaute Reihenfolge), bei 256/512/1024 px, je 3
Seeds (424242, 13, 20260917), **je Region getrennt** — die neun Regionen
(Clonagh, Skerrheim, Morobora, Estrande, Nevadin, Nebelrode, Samarcia,
Macchia, Thalassia) lassen sich direkt aus `region_map`
(Ausgabe des Knotens `terrain.redistribution`) auslesen, weil
`WELTKARTE_AKTIV = True` (Standarad seit der Neun-Regionen-Weltkarte) bei
jedem Lauf ohnehin den vollen Kontinent aus `core/terrain_weltkarte.py`
erzeugt und dabei die Regionszugehörigkeit je Pixel mitliefert:

| Region | 256 px | 512 px | 1024 px | Mittel |
|---|---:|---:|---:|---:|
| Nevadin | 3,553 % | 1,188 % | 0,396 % | 1,712 % |
| Morobora | 2,776 % | 0,600 % | 0,339 % | 1,238 % |
| Nebelrode | 2,508 % | 0,805 % | 0,208 % | 1,173 % |
| Estrande | 2,251 % | 0,689 % | 0,511 % | 1,150 % |
| Skerrheim | 2,102 % | 0,779 % | 0,316 % | 1,066 % |
| Macchia | 1,692 % | 0,914 % | 0,494 % | 1,033 % |
| Clonagh | 1,999 % | 0,532 % | 0,237 % | 0,923 % |
| Thalassia | 1,952 % | 0,428 % | 0,138 % | 0,839 % |
| Samarcia | 1,167 % | 0,516 % | 0,308 % | 0,664 % |

(Seenanteil = Seepixel / Landpixel je Region, gemittelt über die 3 Seeds.)

**Der alte "0,0 %"-Befund gilt nicht mehr.** Skerrheim (historisch: null
Seen) zeigt jetzt in allen 9 Läufen Seen (Mittel 1,066 %). Morobora
(historisch: geeignetes Gelände, aber unbeobachtet, ob Seen ankommen) zeigt
ebenfalls in allen 9 Läufen Seen, mit dem zweithöchsten Mittelwert aller
Regionen (1,238 %) — die frühere offene Frage aus
`docs/FLUESSE_UND_WASSER.md` Block 6 ("ob dort Morobora-Seen entstehen und
ob sie im Bild ankommen, ist ungeprüft") ist damit beantwortet: ja, und es
ist kein Anzeigeproblem, sondern in der Berechnung selbst so.

Nur 2 von 81 gemessenen Region/Größe/Seed-Kombinationen zeigten überhaupt
keinen See (Thalassia bei 1024 px/Seed 13, Samarcia bei 1024 px/Seed
20260917) — beide Regionen haben an anderen Auflösungen/Seeds Seen, es ist
also kein systematischer Ausfall.

**Der gemessene Seenanteil fällt deutlich mit steigender Auflösung**
(Mittel über alle Regionen: 256 px 2,222 %, 512 px 0,717 %, 1024 px
0,327 %), obwohl die reale Kartenbreite bei allen drei Größen dieselbe
21,3 km ist (`WELT_KM` in `core/terrain_weltkarte.py:49`) — nur die
Pixelauflösung ändert sich. Naheliegendste Erklärung, nicht am Code
verifiziert: bei niedriger Auflösung entstehen mehr winzige,
ein-bis-wenige-Pixel-große geschlossene Senken, die Rauschartefakte sind und
bei feinerer Auflösung verschwinden. Die 1024-px-Zahlen dürften die
belastbarsten sein.

Größte Einzelseen (aus dem separaten Gesamtlauf, korrekt mit `WELT_KM =
21,3 km` statt der ursprünglich fälschlich angenommenen 15,0 km
umgerechnet — Prozent- und Stückzahlwerte sind reine Pixelverhältnisse und
von diesem Umrechnungsfehler nicht betroffen): größter gemessener See rund
1,02 km², Median rund 0,01 km², bei 256 px am größten (mehr, aber kleinere
Seen), bei 1024 px am kleinsten.

Welche Änderung seit der alten Messung dafür infrage kommt: laut Ticket
selbst "Seegliederung, Seegrad-Voronoi und Seewege" (siehe
`docs/project_see_voronoi_seegliederung.md`), dazu vermutlich die
Neun-Regionen-Weltkarte (`weltfeld()` in `core/terrain_weltkarte.py`) mit
regionsspezifischen Küsten-Archetypen — beide seit der alten 384-px-Messung
vom 24.08.2026 hinzugekommen. Welche der beiden Änderungen den Ausschlag
gibt, lässt sich mit einer reinen Messung nicht trennen; das würde eine
Vergleichsmessung auf dem alten Codestand erfordern, die außerhalb des
30-Minuten-Rahmens dieses Tickets liegt.

`core/water_generator.py:3103` (`HydrologySystemGenerator._meters_per_pixel()`)
liest den Kartenmaßstab live über
`self.data_lod_manager.get_map_distance_km()` — also denselben, von
`WELTKARTE_AKTIV` bereits korrigierten Wert, den auch die Terrain-Erzeugung
verwendet. Die App selbst ist damit maßstabskonsistent; der
Umrechnungsfehler oben betraf ausschließlich das Wegwerf-Messskript dieses
Tickets, keinen App-Code.

## 4. Was neu grün ist

Acht Testdateien sind seit dem 24.08. dazugekommen, alle grün:

```
smoke_test_anzeige_register_3d.py     2.2s
smoke_test_fluss_vorschau.py          neu, 8/8
smoke_test_kontinentform.py           6.4s
smoke_test_kuestengebiete.py         13.5s
smoke_test_region_tab.py             11.3s
smoke_test_regionsfeld.py             1.8s
smoke_test_regionsregler_wirken.py    7.1s
smoke_test_reiter_vertrag.py          5.0s
smoke_test_stufen_schalter.py         (in erosion_hauptschalter aufgegangen)
```

`smoke_test_reiter_vertrag.py` ist der Wächter für den Fehler vom 26.08.,
bei dem ein abstürzender Reiter die ganze Leiste verschob — er zählt die
fertige Reiterleiste durch, nicht nur die Importe.

## 5. Die größte Lücke bleibt dieselbe

**Die 3D-Ansicht ist von keinem Test erfasst, der wirklich zeichnet.**
`layer_2d_3d_parity` prüft die Registrierung und die Farbtafel — also
genau die Buchhaltung, die diesmal den Fehler fand. Ob das Bild danach
richtig aussieht, prüft nach wie vor niemand außer dir.
