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

Der Satz hat real etwas angerichtet: solange er dort stand, sah jeder Leser
gewollte Fehlschläge statt ungeklärter, und niemand fasste sie an.
`erosion_field` hat eine eigene, vom Schalter unabhaengige Erklaerung; der
tatsaechliche Grund fuer `erosion_quality` bleibt **ungeklaert** - das ist
eine offene Aufgabe, kein erledigter oder bewusster Zustand.

Eine falsche Erklärung ist schlimmer als keine: keine Erklärung lädt zum
Nachsehen ein, eine falsche schliesst die Frage. Das gilt fuer die
Erosionskette ebenso wie fuer eine laengst behobene GPU-Messung, die
wochenlang als offen weitergefuehrt wurde.

**Nachtrag 2026-09-18 (Ticket #30):** die oben beschriebene Paritaet wurde
ein drittes Mal bestaetigt (`smoke_test_erosion_gpu_parity.py`, echte GPU,
PASS, Einzelschritt-Abweichung 0). Zusaetzlich gemessen: bei 1024 px braucht
ein vollstaendiger CPU-Lauf 8-10 Minuten gegen 8-16 Sekunden auf der GPU
(Faktor 37-65x, Maschinenrauschen siehe Messwarnung oben). Der CPU-Pfad wird
im Normalbetrieb nicht genommen - `MAX_CPU_RESOLUTION=256` in
`core/erosion_generator.py` verweigert oberhalb dieser Groesse laut, sowohl
wenn keine GPU vorhanden ist als auch im Normalfall mit GPU. Eine schmale
Ausnahme (GPU vorhanden, aber ein Abschnitt schlaegt mitten im Lauf fehl)
ist NICHT durch dieselbe Sperre abgedeckt; als offener Punkt
`docs/OFFENE_PUNKTE.md` 10.7 vermerkt, absichtlich nicht in dieser Sitzung
behoben. Volle Herleitung in `docs/SITZUNGSLOG.md`, Eintrag 2026-09-18.

**Nachtrag 18.09.2026 (Ticket #32, Seenfläche neu gemessen):** die letzte
belastbare Zahl stammte aus `docs/FLUESSE_UND_WASSER.md`, Block 3
(24.08.2026, 384 px): "11 Binnenseen über 4 Pixel, und KEINER liegt im
Skerrheim" — Block 6 hielt für Morobora ausdrücklich fest, ob dort Seen
ankommen sei "UNGEPRÜFT". Das ist eine reine Messung, keine Korrektur, also
wurde nichts am Verhalten des Generators geändert.

*Hinweis zur Vorarbeit:* dasselbe Ticket wurde bereits am 17.09.2026 in
einer Nachtschicht auf einem anderen, bisher nicht in `main` gemergten
Branch bearbeitet (Commit `333d026`, Ergebnis auch als Kommentar auf
Issue #32 gepostet). Diese Sitzung kannte davon zunächst nichts, hat die
Messung unabhängig neu aufgesetzt und kam auf praktisch identische Zahlen
(siehe Tabelle unten) — das bestätigt die Methode gegenseitig, macht die
Vorarbeit aber nicht überflüssig: jene Fassung hat kein Messskript
hinterlassen (nur den Text in `docs/TESTBERICHT.md` geändert), diese Fassung
legt `tests/smoke_test_seenflaeche_messung.py` an, damit die Messung ohne
manuelles Nachbauen wiederholbar ist.

**Zwei falsche Fährten, bevor die Methode stand** (beide im Skript-Docstring
festgehalten, damit sie nicht nochmal jemand begeht): das Feld `seegrad`
(`terrain.redistribution`) ist eine Abstand-von-Land-Ringstufe über ALLES
Wasser einschließlich Ozean, keine binnensee-spezifische Markierung -
`(seegrad > 0) & (heightmap > 0)` ist rechnerisch immer leer und liefert
IMMER 0,0 %, unabhängig vom Kartenzustand. Die Kontinent-Silhouette vor der
Küstenverformung gegen die fertige Höhenkarte zu halten zählt auch
zurückgewichene Küste (Buchten, Fjorde) mit und ergab unplausible 30-48 %
"Seefläche". Die tatsächlich zuständige Instanz ist der eigene
Calculator-Knoten `water.lake_detection`
(`core/water_generator.py:LakeDetectionSystem.detect_lakes()`,
Prioritäts-Flutung bis zum Überlaufpunkt, nach Volumen gefiltert), dessen
`lake_map` (-1/keine, ≥0/See-ID) dieselbe Definition benutzt, die auch
`water._classify_water_bodies()` im echten Betrieb verwendet
(`is_lake = lake_map >= 0`).

Gemessen über die echte Pipeline (`tools/weather_lab.py:run_pipeline()`,
echter CalculatorDispatcher), bei 256/512/1024 px, je 3 Seeds (424242, 13,
20260917 - dieselben wie in der Vorarbeit, zur Vergleichbarkeit), **je
Region getrennt** über `region_map` (`terrain.redistribution`,
`WELTKARTE_AKTIV=True` liefert bei jedem Lauf die volle
Neun-Regionen-Weltkarte). Seenanteil = Seepixel (`lake_map >= 0`) je
Landpixel (`heightmap > 0`), gemittelt über die 3 Seeds:

| Region | 256 px | 512 px | 1024 px | Mittel |
|---|---:|---:|---:|---:|
| Nevadin | 3,553 % | 1,191 % | 0,396 % | 1,713 % |
| Estrande | 2,277 % | 2,121 % | 0,517 % | 1,638 % |
| Morobora | 2,822 % | 0,600 % | 0,339 % | 1,253 % |
| Nebelrode | 2,508 % | 0,805 % | 0,208 % | 1,173 % |
| Skerrheim | 2,094 % | 1,018 % | 0,317 % | 1,143 % |
| Macchia | 1,748 % | 0,907 % | 0,509 % | 1,055 % |
| Clonagh | 2,069 % | 0,551 % | 0,237 % | 0,952 % |
| Thalassia | 1,953 % | 0,571 % | 0,138 % | 0,887 % |
| Samarcia | 1,167 % | 0,512 % | 0,308 % | 0,662 % |

Mittel über alle Regionen je Größe: 256 px 2,243 %, 512 px 0,919 %, 1024 px
0,330 %.

**Der alte "0,0 %"-Befund gilt nicht mehr, in keiner der 81 gemessenen
Region/Größe/Seed-Kombinationen fällt er zusammen.** Skerrheim (historisch:
null Seen) zeigt jetzt durchgehend Seen (Mittel 1,143 %). Morobora
(historisch unbeobachtet) ebenfalls durchgehend, mit überdurchschnittlichem
Mittel (1,253 %) - die offene Frage aus `docs/FLUESSE_UND_WASSER.md` Block 6
ist damit beantwortet: ja, Morobora bekommt Seen, und es ist kein
Anzeigeproblem, sondern in der Berechnung selbst so.

Nur 2 von 81 Kombinationen zeigten gar keinen See (Thalassia bei 1024 px/Seed
13, Samarcia bei 1024 px/Seed 20260917) - beide Regionen haben an anderen
Auflösungen/Seeds Seen, kein systematischer Ausfall für eine Region.

**Der gemessene Seenanteil fällt deutlich mit steigender Auflösung** (256 px
→ 512 px → 1024 px: 2,243 % → 0,919 % → 0,330 %), obwohl die reale
Kartenbreite bei allen drei Größen dieselbe 21,3 km bleibt (`WELT_KM` in
`core/terrain_weltkarte.py:47`). Naheliegendste Erklärung, nicht am Code
verifiziert: bei niedriger Auflösung bilden sich mehr winzige, ein- bis
wenige-Pixel-große geschlossene Senken, die eher Rauschartefakte als echte
Landschaftsformen sind und bei feinerer Auflösung verschwinden - die
1024-px-Zahlen dürften die belastbarsten sein. Größter gemessener Einzelsee
rund 1,02 km² (256 px, Seed 424242), Median über alle gefundenen Seen rund
0,01 km² bei 256 px sinkend auf rund 0,003-0,004 km² bei 1024 px (mehr, aber
kleinere Seen bei niedriger Auflösung).

Welche Änderung seit der alten 384-px-Messung vom 24.08.2026 dafür infrage
kommt: laut Ticket selbst "Seegliederung, Seegrad-Voronoi und Seewege"
(`docs/project_see_voronoi_seegliederung.md`), dazu die
Neun-Regionen-Weltkarte mit regionsspezifischen Küsten-Archetypen
(`weltfeld()` in `core/terrain_weltkarte.py`) - beide seither hinzugekommen.
Welche der beiden den Ausschlag gibt, lässt sich mit einer reinen Messung
nicht trennen; das bräuchte eine Vergleichsmessung auf dem alten Codestand,
was außerhalb des Rahmens dieses Tickets liegt.

Reproduzierbarkeitshinweis: diese Zahlen wurden unabhängig von der
Vorarbeit (333d026, 17.09.) neu erhoben und weichen von deren Tabelle nur im
Nachkommastellenbereich ab (z. B. Nevadin 256 px hier 3,553 % dort ebenfalls
3,553 %, Morobora 256 px hier 2,822 % dort 2,776 %) - die beiden
unabhängigen Läufe bestätigen sich gegenseitig methodisch.

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
