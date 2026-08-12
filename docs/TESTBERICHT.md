# Testbericht — Stand 2026-08-12

Vollstaendiger Lauf der Testsuite, gemessen am 2026-08-12 auf dem
Entwicklungsrechner (Windows 11, GPU verfuegbar, `.venv`).
**Alle Zahlen sind gemessen, keine geschaetzt.**

Dieser Bericht ersetzt die Fassung vom 2026-08-06 vollstaendig. Jene beschrieb
einen Zustand, in dem 49 von 109 Reglern wirkungslos waren, `weather.temperature`
171 von 199 Sekunden frass und drei GPU-Ausgaben kaputt waren — **alle diese
Befunde sind seither erledigt** (siehe `docs/OFFENE_PUNKTE.md` Abschnitt 11,
1.x, 7.1–7.3). Sie stehen hier nicht mehr, weil ein Testbericht den heutigen
Zustand beschreiben soll, nicht die Geschichte.


## Das Wichtigste in fuenf Zeilen

* **34 von 40 Testdateien bestehen**, Gesamtlaufzeit 1027 s (~17 Min).
* **Keiner der sechs Fehlschlaege ist ein Absturz** — alle sind
  Zusicherungen, die eine Kennzahl gegen einen Zielwert pruefen und knapp
  bis deutlich danebenliegen.
* **Ein Fehlschlag ist ein echter, neuer Befund:** die Kuestenformung hat die
  Regions-Hangeichung verschoben (`smoke_test_regionen_welt.py`, siehe unten
  und OFFENE_PUNKTE 3.9).
* **Drei Fehlschlaege sind Folge einer bewussten Abschaltung**
  (`EROSION_AKTIV = False`) und kein Fehler im eigentlichen Sinn.
* **Die 3D-Ansicht ist weiterhin von keinem Test erfasst** — unveraendert die
  groesste Luecke (OFFENE_PUNKTE 6.4/6.13).


## 0. Wie gemessen wurde, und was das NICHT abdeckt

Jede Datei unter `tests/smoke_test_*.py` wurde **einzeln als eigener Prozess**
gestartet (`.venv/Scripts/python.exe <datei>`, Arbeitsverzeichnis = Projektwurzel),
Rueckgabewert und Laufzeit protokolliert. Einzeln und nicht gesammelt, damit ein
Absturz die uebrigen nicht mitreisst und die Laufzeiten vergleichbar bleiben.

| Bereich | Verfahren | Deckt ab |
|---|---|---|
| Rechnungen | Knoten des `CALCULATOR_GRAPH` (38 Knoten), GPU- und CPU-Pfad | Laufzeit, Zustand jedes Outputs, Paritaet |
| 2D-Anzeige | `smoke_test_display_2d.py` ruft alle 30 Darstellungen mit echten Daten auf (Agg-Backend) | dass wirklich gezeichnet wird |
| Gelaende | Flussnetz, Skalenkopplung, Erosionsfilter, Seegliederung, Regionen | Form, Massstabstreue, Determinismus |
| Siedlungen | Platzierung, Klumpung, Wege, Seewege, Roadsites/Landmarks, Regionsgitter | Anzahl, Verteilung, Erreichbarkeit |
| Wetter | Klimatologie, Temperatur-Direktnormierung, Windregionen | Zielwerte je Region |
| 3D-Mesh | `smoke_test_adaptive_terrain_mesh.py` | Risslosigkeit, Determinismus, Dreieckszahl |

**Was hier NICHT geprueft ist:**

* **Das gerenderte 3D-Bild.** Ein OpenGL-Widget laesst sich nicht kopflos
  zeichnen (CLAUDE.md). Der Mesh-Test prueft die *Geometriedaten*, nicht das
  Bild. Compute-Shader dagegen **sind** headless pruefbar und werden geprueft.
* **Der Rundenbetrieb des Orchestrators** (OFFENE_PUNKTE 12.5).
* **Das Zusammenspiel der Reiter im laufenden Programm** — Reiterwechsel,
  Reglerbedienung, Fortschrittsanzeige.


## 1. Ergebnis je Datei

| Datei | | Zeit |
|---|---|---:|
| smoke_test_adaptive_terrain_mesh.py | OK | 3.8 s |
| smoke_test_biome_preseed.py | OK | 2.3 s |
| smoke_test_biome_supersampling.py | OK | 1.7 s |
| smoke_test_camera_controls.py | OK | 3.4 s |
| smoke_test_display_2d.py | OK | 16.3 s |
| **smoke_test_erosion_field.py** | **FEHL** | 30.4 s |
| smoke_test_erosion_gpu_contract.py | OK | 2.1 s |
| **smoke_test_erosion_gpu_parity.py** | **FEHL** | 4.6 s |
| smoke_test_erosion_hauptschalter.py | OK | 6.7 s |
| **smoke_test_erosion_quality.py** | **FEHL** | 71.3 s |
| smoke_test_geology_3dstack.py | OK | 7.9 s |
| smoke_test_geology_speed.py | OK | 18.4 s |
| smoke_test_layer_2d_3d_parity.py | OK | 2.2 s |
| smoke_test_noise_offset_gpu.py | OK | 3.3 s |
| **smoke_test_pipeline_outputs.py** | **FEHL** | 42.2 s |
| **smoke_test_regionen_welt.py** | **FEHL** | 111.9 s |
| smoke_test_river_reaches_sea.py | OK | 63.6 s |
| smoke_test_seegliederung.py | OK | 23.7 s |
| smoke_test_settlement_clustering.py | OK | 22.6 s |
| smoke_test_settlement_placement.py | OK | 22.6 s |
| smoke_test_settlement_region_grid.py | OK | 35.7 s |
| smoke_test_settlement_roads.py | OK | 2.0 s |
| smoke_test_settlement_sites.py | OK | 2.2 s |
| smoke_test_settlement_valley_routing.py | OK | 4.0 s |
| smoke_test_shader_paths.py | OK | 1.6 s |
| smoke_test_slope_compass_color.py | OK | 2.3 s |
| smoke_test_terrain_erosion_filter.py | OK | 9.5 s |
| smoke_test_terrain_river_network.py | OK | 37.6 s |
| smoke_test_terrain_scale_coupling.py | OK | 30.2 s |
| smoke_test_water_drainage_erosion.py | OK | 6.2 s |
| smoke_test_water_edge_sediment.py | OK | 39.0 s |
| smoke_test_water_erosion_quality.py | OK | 45.3 s |
| smoke_test_water_lake_detection_gpu.py | OK | 4.5 s |
| smoke_test_water_pipe_flow.py | OK | 5.2 s |
| smoke_test_water_pipeline_full.py | OK | 2.5 s |
| smoke_test_water_pipeline_order.py | OK | 8.9 s |
| smoke_test_water_thermal_erosion.py | OK | 2.9 s |
| smoke_test_weather_climatology.py | OK | 36.9 s |
| **smoke_test_weather_temperature_direktnormierung.py** | **FEHL** | 132.1 s |
| smoke_test_weather_wind_regions.py | OK | 157.8 s |

**34 / 40 bestanden.**


## 2. Die sechs Fehlschlaege, einzeln

### Befund 1 — `smoke_test_regionen_welt.py`: die Kuestenformung hat die Hangeichung verschoben

**Der einzige neue, inhaltliche Befund dieses Laufs.** Alle neun Regionen
messen einen zu steilen Hang, und die geforderte Reihenfolge flach → steil
verrutscht bei Atlantikkueste und Steppe um mehr als zwei Plaetze.

Der Anteil der Kuesten-Archetypen (OFFENE_PUNKTE 3.8) daran ist **gemessen**,
durch Ab- und Zuschalten des Passes bei sonst identischem Lauf
(384 px, Seed 20260804, roher `weltfeld`-Median-Hang in Grad):

| Region | mit Kuestenpass | ohne | Ziel |
|---|---:|---:|---:|
| Steppe | 10.6 | **6.6** | 6.5 |
| Taiga | 9.1 | **5.9** | 7.5 |
| Fjordland | 21.7 | 19.5 | 16.0 |
| Mittelgebirge | 14.2 | 16.2 | 12.5 |
| Griechische Inseln | 15.0 | 18.0 | 11.5 |
| Atlantikkueste | 15.0 | 15.4 | 8.5 |
| Alpenland | 23.7 | 24.6 | 19.5 |
| Mittelmeer | 17.0 | 17.4 | 14.5 |

Zwei Dinge stehen damit fest, und sie sind zu trennen:

1. **Bei Steppe und Taiga stammt die gesamte Abweichung aus dem Kuestenpass.**
   Ohne ihn treffen beide ihr Ziel nahezu exakt.
2. **Bei Atlantikkueste, Griechischen Inseln und Alpenland liegt eine
   aeltere, unabhaengige Verstimmung vor** — sie verfehlen ihr Ziel auch
   ohne den Pass deutlich.

Ob das ein Fehler ist, ist eine **Entscheidung, keine Messung**: die
Kuestenformung baut absichtlich Klippen, ein hoeherer Kuestenhang ist
teilweise gewollt. Dann muessen die Zielwerte nachgezogen werden statt der
Formung. Naeheres in OFFENE_PUNKTE 3.9.

> **Verfahrensfehler, der das ermoeglicht hat:** 3.8 wurde gegen vier
> Terrain-Tests geprueft — `smoke_test_regionen_welt.py` war nicht darunter.
> Die Regionseichung ist aber genau das, was eine Gelaendeformung als erstes
> verstimmt. Dieser Test gehoert ab sofort in die Pruefliste jeder Aenderung
> an `weltfeld()`.

### Befund 2–4 — die drei Erosionstests: Folge einer bewussten Abschaltung

`EROSION_AKTIV = False` (OFFENE_PUNKTE 10.6, Nutzerentscheidung "vorerst nein")
legt die Erosionskette still. Die drei Tests pruefen sie trotzdem:

* **`smoke_test_erosion_field.py`** — zwei Farbskalen passen nicht zu den
  Daten: `erosion_map` typischer Wert 0.22 bei Skala [0.5, 300.0],
  `sedimentation_map` 0.32 bei derselben Skala. Alle uebrigen 17 Zusicherungen
  derselben Datei bestehen. Reine Anzeige-Eichung.
* **`smoke_test_erosion_quality.py`** — Qualitaetsmasse der abgeschalteten Kette.
* **`smoke_test_erosion_gpu_parity.py`** — **hier ist der Test selbst schief:**
  er vergleicht den Export ueber den Kartenrand zwischen GPU (500 Schritte) und
  CPU (25 Schritte) und erwartet dieselbe Groessenordnung (38.5 gegen 0.1 m).
  Bei zwanzigfacher Schrittzahl ist das keine sinnvolle Erwartung. Die
  eigentliche Paritaetspruefung — Einzelschritt, alle vier Karten — **besteht**
  (Abweichung 1.1e-07 bis 0.0), ebenso die Massenbilanz (0.002 %).

Keiner der drei sollte gruen erzwungen werden, solange die Kette aus ist. Sie
gehoeren aber gekennzeichnet, damit sie nicht als echte Fehler gelesen werden —
zusammen mit dem Hinweisstreifen im Erosionsreiter (OFFENE_PUNKTE 12.2).

### Befund 5 — `smoke_test_pipeline_outputs.py`: zehn leere Ausgaben, alle bekannt

66 von 76 Outputs in Ordnung. Die zehn leeren:

| Output | Erklaerung |
|---|---|
| `erosion.hydraulic/*` (7 Stueck) | `EROSION_AKTIV = False`, gewollt |
| `geology.intrusions / height_delta` | bekannter offener Punkt 10.1 |
| `settlement.pathfinding / sea_roads` | bei 128 px gibt es kein Kulturpaar ohne Landweg — plausibel, kein Fehler |
| `settlement.plot_nodes / plots` | `plots` ist leer, `plot_nodes` traegt die Daten |

Dazu zwei Pfadunterschiede (`landmark_list`, `landmark_roads`: GPU liefert
Daten, CPU nicht). **Das ist ein echter, ungeklaerter Punkt** — er stand
bisher nirgends und ist mit diesem Bericht neu aufgenommen.

### Befund 6 — `smoke_test_weather_temperature_direktnormierung.py`: eine Region, ein Seed

Ein einziger Wert reisst die Zusicherung: **Seed 4242, Steppe, Jahresmittel
19.30 statt 20.00** — 0.70 K Abweichung bei 0.7 K Toleranz, also exakt auf der
Grenze. Alle uebrigen Regionen und Seeds liegen innerhalb. Das ist kein
Modellfehler, sondern eine zu eng gesetzte Toleranz an einem Randfall;
OFFENE_PUNKTE 1.11 hatte "bis 1.7 K im schlechtesten Fall" bereits als
erwartbar beschrieben.


## 3. Was gut ist

Nicht nur die Fehlschlaege gehoeren in einen Testbericht.

* **Die Gelaendekette ist stabil.** Flussnetz, Skalenkopplung, Erosionsfilter,
  Seegliederung, Meeranschluss der Fluesse — alle gruen, teils mit engen
  Zusicherungen (Flussnetz: gleiche Landschaft bei 256 und 512 px, r = +0.990).
* **Die Siedlungskette ist vollstaendig gruen** — acht Testdateien, von der
  Platzierung ueber Klumpung und Wegenetz bis zu Roadsites und Regionsgitter.
* **Die 2D-Anzeige ist abgedeckt**: alle 30 Darstellungen werden wirklich
  gezeichnet. Diese Pruefung hat im August drei stille Anzeigefehler gefunden,
  die vorher niemand bemerkt hatte (OFFENE_PUNKTE 6.9–6.12).
* **Der GPU-Pfad traegt**: Shader-Pfade, Noise-Offset, Seenerkennung,
  Erosions-Vertrag und die Einzelschritt-Paritaet bestehen alle.
* **Das neue adaptive 3D-Mesh ist risslos** und reduziert bei 1024 px auf
  **19.7 %** der Dreiecke (412 855 statt 2 093 058) — im Programm bestaetigt.


## 4. Die groesste Luecke, unveraendert

**Zwischen den geprueften Daten und dem, was der Nutzer im 3D-Fenster sieht,
prueft nach wie vor nichts.** Das ist keine Nachlaessigkeit, sondern eine
technische Grenze (kein kopfloses OpenGL-Rendering), aber die Folgen sind real
und in diesem Projekt mehrfach belegt:

* Das adaptive Mesh lief wochenlang gar nicht, weil eine Groessenbedingung
  falsch war — **zehn gruene Tests deckten es nicht auf**, weil sie mit
  Groessen arbeiteten, die im Programm nicht vorkommen (OFFENE_PUNKTE 6.16).
* Die gezackten Kuestenklippen (6.17) waren in den Daten von Anfang an
  vorhanden und messbar, aber gefunden hat sie erst ein Blick auf den
  Bildschirm.

Praktische Gegenmassnahme bis auf Weiteres: **laute Logzeilen an jeder Stelle,
die still auf einen Ersatzpfad zurueckfallen kann.** Genau die haben 6.16
sichtbar gemacht.
