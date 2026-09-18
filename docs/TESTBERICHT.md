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
* **Sieben Testdateien sind weiterhin rot**, mit insgesamt zwölf
  einzeln verfolgten Befunden, seit Ticket #49 je mit eigenem Ticket
  und Frist (#74-#85, siehe Abschnitt 3) statt einer Rohbefund-Liste.
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

**Diese Tabelle führt keine Rohbefunde mehr, sondern nur noch
Ticket-Nummern.** Bis 16.09.2026 stand hier je Testdatei eine Zeile mit
Ist-Werten direkt im Fließtext — das wurde mit Ticket #49 aufgegeben,
weil eine Rohbefund-Tabelle ohne Frist nur wächst statt sich zu
schließen. Jeder Befund hat jetzt ein eigenes Ticket mit Ist-Wert,
Soll-Wert, erstem Auftreten und Frist; `tests/smoke_test_testbericht_keine_rohbefunde.py`
verhindert, dass hier wieder eine Zahl statt eines Verweises landet.

| Test | Befund | Ticket |
|---|---|---|
| `erosion_field` | Farbskala zeigt den typischen Wert nicht | #74 |
| `erosion_quality` | Kanalnetz/Ebenen/Schwelle/Schrittzahl verfehlen Zielwerte (ungeklärt) | #75 |
| `regionen_welt` | Macchia/Thalassia-Eichung durch Küstenpass verstimmt | #76 |
| `regionen_welt` | Regionsgrenzen deutlich steiler als das Innere | #77 |
| `regionen_welt` | Landschaft nicht auflösungsinvariant (512 vs. 1024 px) | #78 |
| `pipeline_outputs` | `hinterland_height` NaN-Sentinel (Testartefakt) | #71 |
| `pipeline_outputs` | `height_delta` fälschlich als Fehler gemeldet (ist absichtlich immer 0) | #79 |
| `pipeline_outputs` | `sea_roads` liefert keine Daten | #80 |
| `pipeline_outputs` | `landmark_list` liefert keine Daten | #81 |
| `pipeline_outputs` | `plots` liefert keine Daten | #82 |
| `pipeline_outputs` | `landmark_roads` liefert keine Daten | #83 |
| `settlement_placement` | Marktstadt-Randkorrektur erzeugt 2 „stadt“ statt 1 je Kultur | #84 |
| `weather_temperature_direktnormierung` | Jahresmittel einzelner Region/Seed-Paare verfehlt Toleranz | #85 |

`erosion_gpu_parity` steht nicht mehr in dieser Tabelle — kein offenes
Ticket, siehe die Richtigstellung direkt darunter: der Befund war schon
vor dieser Aufschlüsselung gelöst, der Bericht hatte es nur nicht
nachgezogen.

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

## 6. Seenfläche neu erhoben (Ticket #32)

Die frühere Behauptung **"Seenfläche 0,0 %"** trifft nicht mehr zu.
Neu gemessen mit `tests/smoke_test_seenflaeche_messung.py` bei 256/512/1024 px,
je drei Seeds (20260804, 12345, 4242), alle neun Regionen einzeln:

| px | Mittel See-Anteil an Landfläche |
|---:|---:|
| 256 | 4,15 % |
| 512 | 2,61 % |
| 1024 | 1,37 % |

Spanne über alle 9 Messungen: 1,30 % bis 4,42 %, Mittel 2,71 %. Jede der
neun Regionen hat in jeder Messung mindestens einen See — keine Region
bleibt bei 0,0 %. Auffällig, aber nicht Teil der Abnahmekriterien: der
Seenanteil sinkt mit steigender Auflösung (4,15 % → 1,37 %), vermutlich
weil kleine Seen bei feinerem Raster in mehr, aber im Summenanteil
kleinere Einzelbecken zerfallen — nicht weiter untersucht.

Welche Änderung seither verantwortlich ist: der alte 0,0-%-Wert stammt aus
der Zeit vor Seegliederung, Seegrad-Voronoi und Seewegen (siehe
`docs/SITZUNGSLOG.md`, Block "See-Voronoi/Seegliederung") — die Messung war
überholt, kein Fehler wurde behoben. Ob und wie nachgesteuert wird, ist laut
Ticket #32 ein Folgeticket, hier nur die Ist-Erhebung.

## 7. CPU- gegen GPU-Laufzeit der Erosion gemessen und abgesichert (Ticket #30)

**Der Auftrag: "Faktor 385, CPU 0,1 m gegen GPU 38,5 m Export" messen und dann
entweder Parität herstellen oder den CPU-Pfad laut stilllegen.** Die Prämisse
war beim Start dieses Tickets bereits überholt — Abschnitt 3 oben beschreibt,
dass genau dieser Faktor 385 schon am 27.08.2026 gefunden und behoben wurde
(zu grobes GPU-Meldeintervall, nicht Physik) und `smoke_test_erosion_gpu_parity.py`
seit dem 16.09.2026 erneut grün gegengemessen ist. Erneut bestätigt in diesem
Ticket: Einzelschritt-Abweichung 0,00e+00, Massenbilanz beider Pfade < 0,02 %.
**Parität besteht bereits — Konsequenz A aus dem Ticket ist bereits erfüllt.**

**Laufzeit gemessen** (Maschine dieser Sitzung, Streufaktor der Testumgebung
beachten, siehe Kopf dieser Datei):

| Pfad | Größe | Schritte | Zeit | ms/Schritt |
|---|---:|---:|---:|---:|
| CPU | 128 px | 200 | 1,07 s | 5,3 |
| CPU | 256 px | 200 | 3,97 s | 19,9 |
| GPU | 1024 px | 500 | 6,40 s | 12,8 |

256 px ist die reale Obergrenze des CPU-Pfads (`MAX_CPU_RESOLUTION`) — er kann
1024 px **strukturell nicht** real erreichen, das wird unten begründet.
Skalierung 128→256 (×4 Zellen): ×3,7 Zeit, also ungefähr zellzahlproportional.
Hochgerechnet auf 1024 px (×16 Zellen ggü. 256 px) mit dem produktiven
Default `max_steps=8000`: **CPU ≈ 42 Minuten (geschätzt), GPU ≈ 102 Sekunden
(real gemessen)** — Faktor rund 25, also dieselbe Größenordnung wie das
Beispiel im Ticket ("6 Minuten gegen 25 Sekunden"), nur mit dem heutigen
`max_steps`-Default noch deutlicher.

**Wird der CPU-Pfad im Betrieb je genommen? Ja, aber nicht bei 1024 px.**
`generation_orchestrator.py` injiziert immer einen echten `ShaderManager`;
auf jeder Maschine mit OpenGL-4.3-fähiger GPU läuft die Erosion also real auf
der GPU. Der CPU-Pfad greift als **beabsichtigter Rückfall für Maschinen ohne
nutzbare GPU** — und genau dafür ist er bereits doppelt abgesichert:

1. `ErosionSystemGenerator._resolve_simulation_size()` deckelt die
   Simulationsauflösung ohne GPU-Pfad laut auf 256 px (WARNING-Logzeile,
   `smoke_test_erosion_field.py::resolution_cap_follows_gpu_registration`
   deckt das ab).
2. `HydraulicFieldSimulator.simulate()` verweigert oberhalb 256 px mit einem
   klaren `ValueError` statt stundenlang zu rechnen (Test
   `cpu_limit_fails_loudly`).

**Eine dritte, bisher ungeprüfte Lücke in genau diesem Sicherungsnetz wurde
in diesem Ticket gefunden und geschlossen:** War ein GPU-Pfad registriert
(`has_gpu_path()` True, Simulationsauflösung deshalb NICHT auf 256
gedeckelt — z. B. 512 px), fiel `simulate()` bei einem GPU-Ausfall MITTEN im
Lauf (Treiberfehler, Timeout — `_simulate_gpu()` fängt das ab und gibt `None`
zurück) bisher UNGEPRÜFT in die volle CPU-Hauptschleife durch, ohne die
Größe erneut gegen `MAX_CPU_RESOLUTION` zu prüfen. Ergebnis wäre ein
stundenlanger, für den Nutzer nicht als "das dauert jetzt ewig" erkennbarer
Rückfall gewesen — nur eine WARNING-Zeile im Log, kein Fehler. Behoben in
`core/erosion_generator.py::HydraulicFieldSimulator.simulate()`: dieselbe
`ValueError`-Meldung wie beim Start-Deckel greift jetzt auch nach einem
gescheiterten GPU-Versuch. Neuer Test
`smoke_test_erosion_field.py::gpu_midrun_failure_above_cpu_limit_fails_loudly`
(Stub, dessen `request_shader_operation` sofort eine Exception wirft) deckt
das ab — vor der Änderung rot (lief still auf der CPU weiter, `steps_taken`
kam ohne Fehler zurück), nach der Änderung grün.

**Kein stiller Rückfall bleibt übrig:** alle drei Wege in den CPU-Pfad
(Start ohne GPU, Start mit zu großer Anfrage, GPU-Ausfall mitten im Lauf)
enden entweder in einem tatsächlich brauchbar schnellen Lauf (≤ 256 px) oder
in einem klaren Fehler statt in einer stillen Wartezeit.

Betroffene/neue Tests: `smoke_test_erosion_gpu_parity.py` (unverändert grün),
`smoke_test_erosion_field.py` (ein Test ergänzt, Rest unverändert grün außer
dem bereits bekannten, hier nicht behandelten `colour_ranges_fit_the_data`,
siehe Abschnitt 3).
## 8. Sinuosität erstmals gemessen (Ticket #33)

Die Kennzahl aus `docs/SPEZIFIKATION.md` §3.6 ("Mäander (Sinuosität der
Hauptläufe) | > 1,2 | nicht gemessen") ist jetzt gemessen. Neue Messfunktion
in `core/fluss_sinuositaet.py` (`sinuositaet_pfad`, `fluss_segmente`,
`sinuositaet_je_fluss`), TDD-getestet in
`tests/smoke_test_sinuositaet.py` (6/6 grün: gerader Lauf = 1,0, Sägezahn =
√2 = 1,414214, Handbaum-Segmentierung, echte 128px-Karte ohne Absturz).

Definition: Lauflänge des Spannbaum-Astes geteilt durch die Luftlinie
zwischen seinen Enden, gemessen an den Knoten des Flussnetzes aus
`core/terrain_weltfluesse.flussnetz()` (der aktiven, regionsbewussten
Weltkarten-Pipeline — nicht am lokalen, standardmäßig abgeschalteten Netz in
`core/terrain_river_network.py`). Ein "Fluss" ist eine maximale Kette
gleicher Strahler-Ordnung im Baum.

Ist-Erhebung: 512 px und 1024 px, je drei Seeds (20260804, 314159,
20260921), 9054 Fluss-Segmente insgesamt.

**Nach Flussordnung** (Bäche mäandern anders als große Ströme — bestätigt):

| Ordnung | n | Median | Mittel | Max |
|---:|---:|---:|---:|---:|
| 1 (Quellbach) | 8778 | 1,000 | 1,076 | 6,321 |
| 2 | 1454 | 1,062 | 1,241 | 4,991 |
| 3 | 333 | 1,199 | 1,407 | 6,888 |
| 4 | 70 | 1,458 | 1,818 | 6,951 |
| 5 | 19 | 1,817 | 1,987 | 3,904 |
| 6 | 2 | 2,076 | 2,076 | 2,163 |

Die Sinuosität steigt klar und durchgehend mit der Flussordnung — kleine
Bäche laufen im Spannbaum fast gerade (Median 1,0), große Ströme (Ordnung
4-6) liegen im Mittel klar über der Mäander-Schwelle 1,2 aus der
Spezifikation (1,82 bis 2,08).

**Nach Region — Flachland gegen Gebirge:** hier gilt die Vorwarnung aus
`tests/smoke_test_sinuositaet.py`-Auswertung selbst: 80 % aller Flüsse sind
Ordnung 1 und fast schnurgerade, darum ist der Median je Region bei allen
neun Regionen 1,000 und sagt nichts aus. Aussagekräftig ist das Mittel ab
Ordnung 2, gegen den tatsächlich am erzeugten Gelände gemessenen Median-Hang
(nicht die statischen `relief_m`-Reglerwerte — echtes Gelände nach allen
Formungspässen):

| Region | Hang (Grad, gemessen) | Mittel-Sinuosität ab Ordnung 2 |
|---|---:|---:|
| Estrande | 3,82 | 1,345 |
| Morobora | 4,10 | 1,231 |
| Clonagh | 4,73 | 1,321 |
| Thalassia | 4,87 | 1,265 |
| Samarcia | 5,88 | 1,284 |
| Nebelrode | 6,22 | 1,329 |
| Macchia | 8,48 | 1,304 |
| Skerrheim | 9,04 | 1,302 |
| Nevadin | 23,78 | 1,319 |

Flachere Hälfte (Estrande/Morobora/Clonagh/Thalassia) 1,291 gegen steilere
Hälfte (Nebelrode/Macchia/Skerrheim/Nevadin) 1,314 — Differenz −0,023.
Korrelation Hang↔Sinuosität über alle 9 Regionen: r = +0,233 (schwach,
positiv, nicht das erwartete Vorzeichen).

**Antwort auf die Ticketfrage:** Nein, mit diesen Zahlen mäandert das
Flachland in diesem Programm NICHT stärker als das Gebirge — der
Unterschied ist mit vier Regionen je Gruppe nicht von Null zu unterscheiden,
und die schwache Korrelation zeigt eher in die Gegenrichtung. Das ist
plausibel: der Mäander entsteht hier ausschließlich aus dem
Kosten-Spannbaum, der dem Gelände ausweicht (`docs/SPEZIFIKATION.md`
§15/§16) — dieser Mechanismus reagiert auf lokale Hangwechsel im
Wegverlauf, nicht auf den Regions-Median-Hang. Nevadin (steil, viele lokale
Hindernisse zum Umlaufen) mäandert deswegen ähnlich stark wie die flachen
Regionen. Ob das ein gewünschtes Verhalten ist oder ein eigenes Ticket
braucht (z. B. ein Flachland-Bonus auf die Ausweich-Kosten), ist eine
Entscheidung am Tag — hier nur die Ist-Erhebung, wie im Ticket verlangt.

Erhebungsskript nicht Teil des Repos (reine Einmalmessung, kein
Regressionswächter mit Bandgrenzen) — Zahlen und Methode stehen vollständig
hier und im Commit-Text.
