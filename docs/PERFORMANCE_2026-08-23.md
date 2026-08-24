# Ladezeit: Messung und 40 Verbesserungsmoeglichkeiten

Stand 2026-08-23. Grundlage: der Konsolenlauf des Nutzers vom 2026-08-22
(1024 px, LOD 6, 203 s gesamt), die Kritischer-Pfad-Rechnung
(`tools/pipeline_kritischer_pfad.py`) und die neue Teilschritt-Messung
(`managers/teilschritte.py`).

**Ziel des Nutzers: hoechstens 2 Minuten Ladezeit.** Heute 3.4 Minuten.

---

## 0. Warum Parallelitaet zwischen Knoten NICHT der Hebel ist

| | |
|---|---:|
| seriell (heute) | 183.7 s |
| kritischer Pfad (unendlich viele Kerne) | 171.5 s |
| Ersparnis durch Parallelitaet **zwischen** Knoten | **12.2 s = 7 %** |

Der Abhaengigkeitsgraph ist praktisch eine Kette: 25 Ebenen, breiteste
Ebene 3 Knoten, mittlere Breite 1.5. Der Orchestrator erlaubt bereits 20
parallele Laeufe - es gibt nichts zu parallelisieren.

Vier Knoten machen **84 %** der Rechenzeit aus, und alle vier liegen auf
dem kritischen Pfad. Parallelitaet muss deshalb **innerhalb** dieser vier
Knoten stattfinden, nicht zwischen ihnen.

---

## 1. terrain.redistribution - 61 s

Gemessen mit der neuen Teilschritt-Aufteilung, 1024 px, GPU aktiv
(eigener Lauf 43.2 s; der Unterschied zu den 61 s des Nutzerlaufs ist
Erstlauf-Aufwaermung von GPU und JIT):

| Teilschritt | Dauer | Anteil |
|---|---:|---:|
| `vk_als_raster` | 12.70 s | 29.9 % |
| `erosionsfilter` | 9.15 s | 21.6 % |
| `vk_archetyp_felder` | 7.94 s | 18.7 % |
| `weltfluesse` | 7.86 s | 18.5 % |
| `kontinentform` | 1.22 s | 2.9 % |
| `voronoi_regionen` | 1.01 s | 2.4 % |
| `vk_linie_bauen` | 0.66 s | 1.6 % |
| `seegliederung` | 0.65 s | 1.5 % |
| uebrige 8 Schritte | 1.31 s | 3.1 % |

**Die Vektorkueste ist 20.6 s = 48.6 %** - also das, was in dieser Woche
neu dazugekommen ist. Vorher hatte dieser Knoten diese Kosten nicht.

### Die zehn Vorschlaege

1. **`archetyp_felder()` rechnet `_kuesten_waehlen()` ein ZWEITES Mal.**
   `als_raster()` hat fuer jeden Pixel bereits Abstand, Kontur und
   Bogenlage bestimmt und dann weggeworfen; `archetyp_felder()` baut
   dieselbe KD-Baum-Abfrage samt gieriger Auswahl fuer alle Landpixel neu
   auf. Ergebnis einmal berechnen und weiterreichen.
   *Erwartet: -6 bis -7 s. Der groesste Einzelposten, und reine
   Doppelarbeit.*

2. **`als_raster()` fragt alle 1 048 576 Pixel, auch die tiefe See.**
   Die Kuestenformung wirkt nur bis `reichweite` (110-349 m, also
   5-17 px). Ausserhalb ist das Ergebnis bitgleich mit der Eingabe.
   Eine Maske "Abstand zur Kontur < max(reichweite) + Rand" waehlt
   typisch 10-20 % der Pixel aus.
   *Erwartet: -8 bis -10 s. Zusammen mit (1) faellt die Vektorkueste von
   20.6 s auf etwa 4 s.*

3. **`K_KANDIDATEN = 24` Nachbarn je Pixel, davon werden 3 behalten.**
   Die gierige Auswahlschleife laeuft ueber alle 24, mit einer
   Innenschleife ueber `K_KUESTEN`. Bei 12 Kandidaten pruefen und nur bei
   Bedarf auf 24 erweitern (die Trennbedingung schlaegt selten zu).
   *Erwartet: -2 bis -3 s.*

4. **`_auf_strecken()` laeuft zweimal ueber alle Kandidaten** (Versatz -1
   und +1). Die beiden Nachbarsegmente lassen sich in einem
   `np.stack`-Durchgang statt in zwei Schleifendurchlaeufen behandeln.
   *Erwartet: -1 s.*

5. **`hoehe()` in Kacheln statt in einem Rutsch.** Bei 1 MPixel mal 24
   Kandidaten mal 2 Koordinaten entstehen Zwischenfelder von mehreren
   hundert MB. Kachelweise (z. B. 128x128) bleibt alles im Cache.
   *Erwartet: -1 bis -2 s, und deutlich weniger Speicherdruck.*

6. **Der Erosionsfilter (9.1 s) laeuft auf der CPU.** Er ist ein
   Nachbarschaftsfilter ueber ein regelmaessiges Gitter - genau die Form,
   fuer die es in diesem Projekt bereits Compute-Shader gibt
   (`shaders/`, `ShaderManager.request_shader_operation`).
   *Erwartet: -7 bis -8 s. Groesster Einzelposten nach der Vektorkueste.*

7. **`weltfluesse` (7.9 s) enthaelt einen Flussakkumulations-Durchlauf**,
   der demselben Muster folgt wie `water.flow_network` - und der laeuft
   bereits auf der GPU (2.26 s fuer dieselbe Kartengroesse). Denselben
   Shader auch hier verwenden statt einer zweiten CPU-Implementierung.
   *Erwartet: -4 bis -5 s.*

8. **Der KD-Baum wird je Aufruf neu gebaut.** `linien_baum` entsteht in
   `VektorKueste.__init__` und wird von `als_raster` und
   `archetyp_felder` getrennt benutzt; mit Vorschlag (1) faellt der
   zweite Aufbau ohnehin weg. Zusaetzlich `balanced_tree=False` messen -
   bei ~5000 Punkten ist der Aufbau oft teurer als der Gewinn.
   *Erwartet: -0.3 s.*

9. **`float64` durchgaengig, obwohl `float32` reicht.** Hoehen stehen in
   Metern mit Zielgenauigkeit von Zentimetern; `float32` hat 7 Stellen.
   Halber Speicherverkehr im gesamten Vektorkuesten-Pfad.
   *Erwartet: -1 bis -2 s. VORSICHT: erst gegen
   `smoke_test_vektor_kueste.py` (Determinismus, Rasterfreiheit) pruefen.*

10. **Die vier grossen Teilschritte sind gegenseitig unabhaengig.**
    `kontinentform`, `voronoi_regionen` und `oktavenstapel` haengen nur
    am Seed, nicht voneinander. Bei 22 Kernen liessen sie sich
    ueberlappen. *Erwartet: -1.5 s. Bewusst als Letztes gelistet - der
    kleinste Gewinn bei der hoechsten Komplexitaet, siehe Abschnitt 0.*

**Summe realistisch: 43 s -> 12-15 s.**

---

## 2. settlement.pathfinding - 60 s

Eigene Messung, 1024 px, echte Karte:

| | |
|---|---:|
| A-Stern-Lauf, Median | **0.85 s** |
| Spanne ueber 6 Laeufe | 0.59-1.65 s |
| `bau_kostenfeld` | 0.04 s |
| `PathfindingSystem(feld)` bauen | **0 ms** |
| `np.where` ueber 1 MPixel | 4 ms |
| Kerne verfuegbar / genutzt | **22 / 1** |

Die Zeit steckt **vollstaendig im A-Stern-Lauf selbst**. Kostenfeldaufbau
und Objekterzeugung sind vernachlaessigbar.

`_a_stern()` ist reines Python: `heapq`, ein `dict` fuer `g_score`, ein
`set` fuer `closed`, und eine doppelte `for dx/dy`-Schleife ueber die 8
Nachbarn. Bei `max_search_nodes = min(600 000, 1.3 * map_size^2)` sind das
bis zu 600 000 Expansionen mit je ~8 Dict-Zugriffen im Interpreter.

### Die zehn Vorschlaege

1. **Die Kandidatenschleife parallelisieren.** In
   `calculate_road_network` (Zeile ~5894) laufen die A-Stern-Aufrufe in
   einer Doppelschleife ueber alle Ortspaare; jeder Aufruf schreibt an
   eine eigene Stelle von `C`, `W` und `pfade`, **keine Iteration haengt
   von einer anderen ab** (das Verbilligen bestehender Wege passiert erst
   in der Schleife danach). 22 Kerne, ein `ProcessPoolExecutor`.
   *Erwartet: 60 s -> 5-8 s. Der groesste Einzelgewinn der ganzen
   Pipeline.*

2. **A-Stern mit `numba` uebersetzen.** `numba` ist bereits Abhaengigkeit
   (siehe CLAUDE.md) und der Innenpfad ist reine Arithmetik auf einem
   `float32`-Array. Erfordert `heapq` durch einen Array-Heap zu
   ersetzen. *Erwartet: 10-30x auf den Einzellauf, also 0.85 s -> unter
   0.1 s. Kombinierbar mit (1), aber (1) allein reicht fuer das Ziel.*

3. **Zweistufiges Budget greift zu selten.** `_SCHNELLES_BUDGET` ist
   5000 Knoten, `max_search_nodes` bis 600 000 - Faktor 120. Bei 1024 px
   ist ein 434-px-Pfad mit 540 Punkten realistisch nie in 5000 Knoten
   loesbar, also zahlt fast jeder Aufruf **beide** Stufen. Das schnelle
   Budget an die Kartengroesse koppeln (z. B. `8 * Luftlinie`).
   *Erwartet: -20 bis -30 % auf jeden Lauf, der heute doppelt sucht.*

4. **Gewichtete Heuristik (Weighted A-Stern).** `f = g + w*h` mit
   w = 1.2-1.5 schneidet den Suchbaum drastisch. Der Pfad ist dann nicht
   mehr garantiert optimal, aber hoechstens w-mal teurer - fuer eine
   Handelsstrasse voellig ausreichend.
   *Erwartet: 2-5x weniger Expansionen.*

5. **Pfade auf grobem Gitter vorsuchen.** Erst auf 256 px routen, dann
   nur in einem Korridor um das Ergebnis auf 1024 px verfeinern. Das
   Kostenfeld ist bereits als LOD-Pyramide vorhanden.
   *Erwartet: 5-10x bei langen Strecken.*

6. **`calculate_movement_cost` je Pfadpunkt erneut aufrufen** (in
   `route()`, fuer die Kostensumme) - das ist ein zweiter Durchlauf ueber
   den fertigen Pfad, obwohl A-Stern die Summe als `g_score[ziel]`
   bereits kennt. *Erwartet: -5 %, und es entfaellt eine Fehlerquelle.*

7. **Der Bereitschaftstest routet Paare, die nie gebaut werden.** Erst
   `Bereitschaft` pruefen (billig, reine Rangarithmetik), dann nur bei
   plausibler Bereitschaft routen. Eine Luftlinien-Untergrenze fuer die
   Wegkosten schliesst hoffnungslose Paare vorab aus.
   *Erwartet: -30 bis -50 % der Laeufe.*

8. **`_gabriel_kandidaten` staerker beschneiden.** Bei 36 Siedlungen
   bleiben nach Gabriel typisch 70-90 Paare. Eine zusaetzliche
   Entfernungsschranke (kein Landweg ueber 8 km bei 21 km Weltbreite)
   waere fachlich begruendet.
   *Erwartet: -20 % der Laeufe.*

9. **Ergebnisse zwischen LOD-Stufen wiederverwenden.** Das Wegenetz wird
   nur beim finalen LOD gerechnet (`_is_final_lod`), aber die groberen
   Stufen koennten den Korridor fuer (5) liefern, statt verworfen zu
   werden. *Erwartet: passt zu (5), kein eigener Gewinn.*

10. **`scipy.sparse.csgraph.dijkstra` statt eigenem A-Stern.** Ein
    Gitter-Graph mit 1 M Knoten und 8 M Kanten ist in C implementiert
    und liefert von EINEM Startpunkt aus alle Ziele gleichzeitig - bei
    36 Siedlungen also 36 Laeufe statt 630.
    *Erwartet: sehr gross, aber Umbau des Kostenmodells noetig. Als
    Alternative zu (1)+(2) zu bewerten, nicht zusaetzlich.*

**Summe realistisch mit (1) allein: 60 s -> 6 s.**

---

## 3. settlement.plot_nodes - 20 s

Aus dem Log des Nutzers:

```
PlotPhysicsSystem: MAX_PHYSICS_ITERATIONS (100) erreicht,
eingefroren ohne volle Konvergenz.
```

Die Schleife laeuft **immer** die vollen 100 Iterationen, weil sie nie
konvergiert. Ausserdem stehen dort sieben Warnungen `Stadtkern
settlement_id=N hat keine eigenen Voronoi-Nachbarn` - sieben von 36
Staedten sind in der Nachbarschaftsstruktur nicht angebunden.

Die neue Instrumentierung teilt die Schleife jetzt in `physics_step`,
`sync`, `traffic`, `fortschritt` und `abschluss` auf; die Zahlen liefert
der naechste Programmlauf.

### Die zehn Vorschlaege

1. **Herausfinden, WARUM nicht konvergiert wird.** 100 Iterationen ohne
   Konvergenz heisst entweder, dass die Schrittweite zu gross ist (das
   System schwingt), oder dass `CONVERGENCE_MAX_DISPLACEMENT` zu streng
   ist. Die maximale Verschiebung je Iteration protokollieren und den
   Verlauf ansehen - faellt sie und stagniert, ist es die Schranke;
   springt sie, ist es die Schrittweite.
   *Ohne diese Antwort sind die naechsten Punkte geraten.*

2. **Die sieben Staedte ohne Voronoi-Nachbarn reparieren.** Isolierte
   Knoten koennen ihre Kraft nirgends abgeben und wandern moeglicherweise
   dauerhaft - ein plausibler Grund fuer die ausbleibende Konvergenz.
   *Moeglicherweise loest allein das den Punkt (1).*

3. **Daempfung erhoehen.** Ein Federsystem, das nach 100 Schritten noch
   zappelt, ist unterdaempft. Ein hoeherer Reibungsterm kostet nichts und
   halbiert die Iterationszahl typisch.
   *Erwartet: -50 %, also -10 s.*

4. **Abbruch nach Stagnation statt nur nach Konvergenz.** Wenn die
   maximale Verschiebung ueber 10 Iterationen um weniger als 1 % faellt,
   bringen weitere 90 nichts mehr.
   *Erwartet: -30 bis -60 %.*

5. **`_simulate_traffic()` laeuft alle `TRAFFIC_RECOMPUTE_INTERVAL`
   Iterationen.** Solange die Knoten noch weit wandern, ist der
   Verkehrsfluss ohnehin Makulatur - erst ab der zweiten Haelfte rechnen.
   *Erwartet: -20 bis -40 % je nach gemessenem Anteil.*

6. **`_report_live_state()` kopiert den Zustand fuer die GUI.** Bei
   470 Knoten und 627 Kanten ist das je Meldung eine vollstaendige
   Serialisierung. Seltener melden oder nur Deltas schicken.
   *Erwartet: klein, aber messbar - die neue Instrumentierung zeigt es.*

7. **Kraftberechnung vektorisieren.** 470 Knoten in Python-Schleifen sind
   ~220 000 Paarungen je Iteration; als `numpy`-Matrixoperation ist das
   ein Rutsch. *Erwartet: 5-20x auf `physics_step`.*

8. **Nachbarschaft ueber einen KD-Baum statt aller Paare.** Kraefte
   wirken nur lokal; ein `cKDTree.query_pairs(radius)` je 10 Iterationen
   neu aufgebaut reicht.
   *Erwartet: von O(n^2) auf O(n log n).*

9. **Auf grober Aufloesung vorkonvergieren.** Die Physik ist
   aufloesungsunabhaengig - erst mit 1/4 der Knoten konvergieren, dann
   verfeinern. *Erwartet: -40 %.*

10. **Die Iterationszahl an die LOD-Stufe koppeln.** Bei
    Zwischenstufen reichen 20 Iterationen; die vollen 100 nur beim
    finalen LOD. *Erwartet: hilft nur, wenn Zwischenstufen ueberhaupt
    gerechnet werden - derzeit nicht (`_is_final_lod`), also zuletzt.*

---

## 4. weather.temperature - 13.5 s

Instrumentiert in sechs Teilschritte (`terrain_inputs`, `rauheit_solar`,
`schattenwurf`, `monatsparameter`, `atmosphaere_schleife`,
`nachbereitung`); Zahlen kommen mit dem naechsten Programmlauf.

**Der auffaelligste Befund steht schon im Quelltext.** Der Kommentar bei
`_run_coupled_atmosphere_simulation` sagt:

> Die Simulation laeuft weiter, denn Wind und Feuchte kommen noch aus
> ihr. Nur ihr Temperaturergebnis wird verworfen.

Die gekoppelte 3-Schicht-Atmosphaere laeuft also **sechsmal** (einmal je
Monat) ueber 25-50 Zeitschritte, und ihr Temperaturergebnis wird
weggeworfen, weil die Temperatur seit 2026-08-07 eine geschlossene
Festlegung ist (`temperaturfeld_festgelegt`).

### Die zehn Vorschlaege

1. **Pruefen, ob Wind und Feuchte die volle Simulation wirklich
   brauchen.** Beide sind seither ebenfalls weitgehend festgelegt
   (`_wind_regional_faktor`, `niederschlagsfeld_festgelegt`). Falls ja,
   entfaellt der teuerste Posten des Knotens vollstaendig.
   *Erwartet: bis zu -8 s. Zuerst pruefen, dann alles Weitere.*

2. **Zeitschrittzahl senken.** `_get_atmosphere_loop_steps` liefert
   25-50 Schritte je LOD. Da die Temperatur ohnehin verworfen wird, ist
   der Konvergenzanspruch an Wind/Feuchte geringer.
   *Erwartet: linear - halbe Schritte, halbe Zeit.*

3. **Die sechs Monate parallel rechnen.** Sie sind vollstaendig
   unabhaengig (jeder bekommt seine eigenen `month_params`), bis auf die
   optionale LOD-Vererbung. Sechs Kerne von 22.
   *Erwartet: -70 % auf die Schleife.*

4. **Der Schattenwurf laeuft bereits nur EINMAL statt sechsmal**
   (Aenderung vom 2026-08-07, Faktor 6 gespart). Wenn die Messung ihn
   trotzdem als groessten Posten zeigt, ist `ShadowCalculator` selbst der
   Kandidat - er hat einen GPU-Pfad, der zu pruefen waere.

5. **`terrain.shadow` wird bereits berechnet** (2.9 s, eigener Knoten)
   und hier trotzdem neu gerechnet, weil das Wettergitter eine andere
   Aufloesung hat. Bei gleicher Aufloesung liesse sich der vorhandene
   Output wiederverwenden.
   *Erwartet: -2 bis -3 s, falls die Gitter zusammenfallen.*

6. **`monthly_shadowmaps` haelt sechsmal dieselbe Referenz** - das ist
   korrekt und billig, aber `np.mean(np.stack(...))` in der
   Nachbereitung materialisiert sechs Kopien.
   *Erwartet: klein, aber Speicher.*

7. **`float32` statt `float64` in der Atmosphaerenschleife.** Bei
   3 Schichten mal 1 MPixel mal 6 Monaten ist der Speicherverkehr der
   Flaschenhals, nicht die Arithmetik.
   *Erwartet: bis -40 % auf die Schleife.*

8. **Die Schleife auf die GPU.** Ein 3-Schicht-Advektions-/
   Diffusionsschritt ist das Musterbeispiel eines Compute-Shaders, und
   die Infrastruktur steht (`ShaderManager`, offscreen pruefbar).
   *Erwartet: gross, aber Aufwand hoch - erst nach (1) bewerten.*

9. **Auf grober Aufloesung rechnen und hochskalieren.** Wind und Feuchte
   sind grossraeumige Felder; 256 px reichen fuer beides, und die
   Bikubik-Interpolation steht bereits bereit
   (`_interpolate_2d_bicubic`).
   *Erwartet: -80 % bei 256 statt 1024.*

10. **`_get_roughness_damping` und `_get_solar_absorption_factor` sind
    bereits aus der Schleife herausgezogen** (Kommentar im Quelltext).
    Pruefen, ob `_generate_seasonal_parameters` das ebenfalls verdient -
    es laeuft sechsmal fuer Werte, die nur an Monat und Reglern haengen.
    *Erwartet: klein.*

---

## ERGEBNIS DER UMSETZUNG (2026-08-23)

### Zur Messmethode - bitte lesen, bevor Zahlen verglichen werden

Absolute Laufzeiten sind auf dieser Maschine **nicht zwischen Programm-
laeufen vergleichbar**. Gemessen an derselben festen Referenzlast
(5x `np.fft.fft2` auf 2000x2000) schwankte sie im Lauf dieses Nachmittags
zwischen 0.80 s und 1.91 s - Faktor 2.4, ohne erkennbare Fremdlast
(CPU 18 %, 12.9 GB RAM frei, kein Swapping). Derselbe unveraenderte
Knoten `terrain.redistribution` mass an einem Nachmittag 43 s, 48 s, 58 s
und 96 s.

**Alle Faktoren unten stammen deshalb aus A/B-Messungen im SELBEN
Prozess**, alte und neue Fassung unmittelbar nacheinander. Nur die sind
belastbar.

### Umgesetzt und verifiziert

| Punkt | Massnahme | Faktor | Ergebnis gleich? |
|---|---|---:|---|
| 2.2 | A\* mit numba (`core/wegsuche_schnell.py`) | **14-16x** | Punkt fuer Punkt identisch |
| 1.1 | `archetyp_felder`: Doppelarbeit + Tabellen-Nachschlag | **23x** | identisch |
| 1.6 | Erosionsfilter bandweise (sequenziell) | **3.0x** | bitgleich |
| 1.2 | Nahmaske in `hoehe()` | **2.5x** | bitgleich |
| 1.5 | Kachelung in `hoehe()` | Speicher | bitgleich |
| 2.4 | Weighted A\* als Regler (`WEGSUCHE_H_GEWICHT`) | - | steht auf 1.0 |
| 3.1/3.2/3.4 | plot_nodes: Stagnationsabbruch, Verlaufslog, Sammelwarnung | - | frueherer Abbruch |
| 1.7 | `taeler_eingraben`: Stuetzstellen vektorisiert | **7.7x** | bitgleich |
| 1.7 | `poisson_points`: numpy-Skalarzugriffe raus | **2.4x** | bitgleich, 5 Seeds |

**1.7 brauchte keinen GPU-Port.** Der Vorschlag lautete, die
Flussakkumulation auf die GPU zu bringen. Ein Profil zeigte, dass die Zeit
ganz woanders steckte:

* `taeler_eingraben` (12.6 s): **215 761 einzelne `np.clip`-Aufrufe auf
  Skalaren**, allein 6.5 s, dazu 215 759 `round()`-Aufrufe mit 1.1 s. Die
  innere Schleife ueber die Stuetzstellen einer Kante liess sich in einem
  Zug rechnen - 12.6 s -> 1.6 s.
* `poisson_points` (19.7 s von 24.3 s in `flussnetz`): 465 662 Versuche zu
  43.8 Mikrosekunden, davon der groesste Teil in **`grid[yy, xx]` auf einem
  numpy-Array** - rund 1.5 Mikrosekunden je Zugriff, weil jedes Mal ein
  numpy-Skalarobjekt entsteht. Bei 5.2 Millionen Zellpruefungen sind das
  etwa 8 Sekunden fuer Indexzugriffe, die als Python-Listenzugriff je 0.05
  Mikrosekunden kosten. Auf der teuersten Stufe 13.6 s -> 4.9 s.

**Die Lehre**: numpy ist schnell auf ganzen Feldern und langsam auf
Einzelwerten. Beide Fundstellen greifen ausschliesslich einzeln zu - dort
ist eine Python-Liste die schnellere Datenstruktur, nicht die langsamere.
Eine GPU haette an beidem nichts geaendert.

Auf eine echte 1024-px-Karte umgerechnet:
**`settlement.pathfinding` 60 s -> 3.8 s** (Median je Route 0.85 s -> 0.055 s).

### Durch Messung WIDERLEGT - bewusst nicht umgesetzt

Vier der urspruenglichen Vorschlaege waren falsch. Alle vier sind im
Quelltext an Ort und Stelle mit ihrem Messwert kommentiert, damit sie
nicht ein zweites Mal versucht werden.

* **1.3 `K_KANDIDATEN` von 24 auf 12 senken.** Gemessen braucht der
  MEDIAN-Punkt 18 der 24 Kandidaten, bis drei verschiedene Abschnitte
  beisammen sind; bei 6.4 % werden sie nie voll. 24 ist knapp, nicht
  grosszuegig - eine Senkung wuerde das Gelaende aendern.
* **Den 5x5-Nachbarschaftstest in `poisson_points` vektorisieren.** Ein
  Slice statt 25 Einzelzugriffen klingt zwingend, war aber GEMESSEN 1.7-
  bis 3-mal langsamer: es sind typisch nur ein bis fuenf belegte Nachbarn,
  die Schleife bricht beim ersten Treffer ab, und die Vektorfassung legt
  dafuer mehrere Zwischenarrays an. Erst der Wechsel auf Python-Listen
  brachte den Gewinn.
* **1.4 `_auf_strecken` in einen Durchgang ziehen.** Gemessen LANGSAMER
  (0.826 s statt 0.690 s): die Zwischenfelder waeren (N, K, 2, 2) statt
  (N, K, 2), also 111 MB statt zweimal 55 MB nacheinander.
* **2.1 Die Kandidatenschleife auf 22 Kerne verteilen.** Durch 2.2
  gegenstandslos - bei 3.8 s Gesamtdauer braucht es keine Prozesse, und
  die Prozessvariante haette das Kostenfeld je Worker durch pickle
  schicken muessen.
* **1.6 in der PARALLELEN Fassung.** Der Filter wurde damit 5x schneller
  (9.9 s -> 1.3 s), aber JEDE nachfolgende numpy-Rechnung im selben
  Prozess dauerhaft **2.4x langsamer** (Referenzlast 1.47 s -> 3.55 s,
  ohne Erholung). `weltfluesse` stieg dadurch von 8.4 s auf 31.7 s - der
  Knoten wurde als Ganzes langsamer, obwohl sein teuerster Teilschritt
  schneller geworden war. Ursache ist die gleichzeitige Allokation
  grosser Felder aus 22 Threads. Die sequenzielle Fassung bringt 3.0x
  ohne diese Nebenwirkung; der Gewinn kam ohnehin aus der
  Cache-Lokalitaet, nicht aus der Parallelitaet.
  `tests/smoke_test_erosionsfilter_baender.py::keine_nachwirkung` faengt
  den Fall kuenftig ab - an der Zeit des Filters allein ist er NICHT zu
  sehen.

### Noch offen

| Punkt | Was | Warum nicht gemacht |
|---|---|---|
| 1.7 | GPU-Port | Nicht mehr noetig, siehe oben - die Zeit steckte in Skalarzugriffen, nicht in der Rechnung |
| 1.8, 1.9 | KD-Baum, float32 | Klein; 1.9 braucht eine Determinismus-Runde |
| 2.3, 2.5-2.9 | Budget, Grobgitter, Vorfilter | Bei 3.8 s ohne Gewinn, und mehrere wuerden die Karte aendern |
| 4.1-4.10 | weather.temperature | Nur instrumentiert. Befund: Temperatur UND Niederschlag sind bereits geschlossene Festlegungen, aber Wind und Feuchte brauchen die Simulation wirklich - abschalten geht nicht, nur verbilligen (4.2, 4.3, 4.9) |
| 3.5-3.10 | plot_nodes tiefer | Nutzer-Vorgabe: das Plot-System wird ersetzt, hier nur rudimentaer fixen |

### Neue Tests

* `tests/smoke_test_wegsuche_schnell.py` - 5 Gruppen; die erste vergleicht
  numba gegen Python Punkt fuer Punkt. Dafuer hat `_a_stern` einen
  Parameter `schnell=False` bekommen: ohne ihn verglich der Test nach dem
  Einbau numba gegen numba und war gruen, ohne noch etwas zuzusichern.
* `tests/smoke_test_erosionsfilter_baender.py` - 5 Gruppen, darunter
  `keine_nachwirkung` (siehe oben) und `bandzahl_egal`.
* `tests/smoke_test_kuesten_schnitt.py::wicklung_wie_gitter` - neu nach
  dem Wicklungsfehler, der das 3D-Gelaende in Streifen zerschnitt.

---

## Zusammenfassung: der Weg zu 2 Minuten

| Massnahme | heute | danach |
|---|---:|---:|
| pathfinding parallelisieren (2.1) | 60 s | 6 s |
| Vektorkueste: Doppelarbeit + Maske (1.1, 1.2) | 20.6 s | 4 s |
| Erosionsfilter auf GPU (1.6) | 9.1 s | 2 s |
| plot_nodes: Konvergenz reparieren (3.1-3.4) | 20 s | 8 s |
| Atmosphaerenschleife pruefen (4.1) | 13.5 s | 5 s |
| **Rechenzeit gesamt** | **184 s** | **~75 s** |
| Main-Thread (nicht parallelisierbar) | 19 s | 19 s |
| **Ladezeit** | **3.4 min** | **~1.6 min** |

Die ersten drei Zeilen allein bringen den Lauf unter zwei Minuten. Sie
sind auch die risikoaermsten: (2.1) aendert die Reihenfolge nicht (die
Ergebnisse sind unabhaengig), (1.1) ist reine Doppelarbeit, (1.2) ist
bitgleich, weil ausserhalb der Reichweite nichts geschieht.

**Nicht messbar ohne Programmlauf:** die Teilzeiten von `plot_nodes` und
`weather.temperature`. Beide sind jetzt instrumentiert und erscheinen im
naechsten Konsolenlauf.
