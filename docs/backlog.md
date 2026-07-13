# Backlog

Offene Punkte aus dem Kanban-Board (Stand 2026-07-08, zuletzt aktualisiert
2026-07-08 nach GPU-Session). Diese Datei ist die Quelle der Wahrheit für
Claude, da das Kanban-Plugin selbst nicht auslesbar ist. Bitte beim
Abschließen eines Punktes hier durchstreichen/entfernen statt nur im
Kanban-Board zu verschieben.

## To Do

2. **Flüsse versiegen statt aus der Map zu fließen** — Flüsse, die entstehen,
   sammeln sich aktuell in einem einzelnen Pixel und versiegen dort. Ziel:
   Über die LOD-Stufen hinweg soll der Fluss sich immer einen Weg aus der Map
   heraus suchen und dabei Täler graben. Lösungsansatz z.B. über
   Lakefill-Algorithmen. *(Priority: high)*
   - Update 2026-07-08: beim GPU-Umbau von `water.lake_detection` einen
     direkt verwandten Bug gefunden+gefixt — die Lake-Volumen-Berechnung
     verglich Pixelhöhen gegen die Höhe am Seed (= per Definition der
     tiefste Punkt), wodurch das berechnete Volumen strukturell IMMER ~0
     war und der Volume-Threshold nie griff. Jetzt korrekt (Wasserspiegel =
     höchster Punkt im geflohnenen Becken). Das eigentliche
     "Fluss-gräbt-sich-keinen-Weg-raus"-Verhalten (Lakefill über LOD-Stufen)
     ist davon unberührt und weiterhin offen.
   - Update 2026-07-09 (separate Session, `core/water_generator.py` CPU-Seite):
     der "Wasserspiegel = Becken-Maximum"-Fix von 2026-07-08 hatte selbst noch
     einen Bug — da jeder Punkt im Becken per Definition <= dem Becken-Maximum
     liegt, zählte dadurch das GESAMTE Becken (inkl. Berggipfel) als
     überflutet ("Lake überall", vom User gemeldet). Auf CPU UND GPU
     (`_classify_lake_basins_vectorized` in `shader_manager.py`) jetzt auf
     echten Spill-Point (niedrigster Beckenrand-Übergang) korrigiert, nur
     Pixel darunter zählen als See. Zusätzlich: die Becken-ZUORDNUNG selbst
     (`_apply_jump_flooding()` CPU-seitig, `jumpFloodLakes.comp` GPU-seitig)
     prüft weiterhin nur "current_height >= seed_height" statt echter
     Erreichbarkeit über einen monotonen Abwärtspfad - de facto ein reines
     Luftlinien-Voronoi-Diagramm ohne Rücksicht auf Bergkämme. CPU-Seite auf
     echten Priority-Flood-Watershed (Multi-Source-Dijkstra über Höhe als
     Kosten) umgestellt und verifiziert (See-Abdeckung 100%→4.4% auf
     realistischer Testkarte). GPU-Seite (`jumpFloodLakes.comp`) hat diesen
     Watershed-Fix NICHT bekommen - ein echter paralleler Watershed-Transform
     ist algorithmisch deutlich aufwändiger als simples Jump-Flooding (siehe
     unten, neues Ticket). Der Spill-Point-Filter dämmt die GPU-seitige
     Über-Zuordnung stark ein (die meisten falsch zugeordneten Hochpunkte
     liegen über der Spill-Höhe und werden ausgeschlossen), ist aber keine
     vollständige Lösung.
   - Das eigentliche "Fluss-gräbt-sich-keinen-Weg-raus über LOD-Stufen"-
     Verhalten (Lakefill-Algorithmus) ist weiterhin komplett offen, unabhängig
     von den obigen Bugfixes.

3. **Zu viele Fluss-Biom-Felder** — Es entstehen viel zu viele
   Fluss-Biom-Felder.
   a) Riverbank evtl. auf 2px begrenzen.
   b) Erkennung "was zählt als Fluss" grundsätzlich hinterfragen.
   c) Aktuell entstehen praktisch überall an Hängen Flüsse. *(Priority: high)*
   - Update 2026-07-09: Hauptursache gefunden+gefixt - `RAIN_THRESHOLD`
     (`gui/config/value_default.py`) war ~1000x zu klein für `precip_map`s
     reale Größenordnung (0-2.8 bei Default-Parametern statt der
     angenommenen ~0-500), wodurch praktisch jedes Pixel als Regenquelle
     zählte und selbst einzelne Pixel sofort die "Grand River"-Schwelle
     sprengten. Zusätzlich hatte `core/terrain_generator.py`s
     `SlopeCalculator` (und Kopien davon in `weather_generator.py` und
     `biome_generator.py`) nie die reale Meter-pro-Pixel-Umrechnung benutzt
     (implizit 1 Pixel = 1m statt der realen ~50-300m je nach Kartengröße),
     wodurch Steigungen um Faktor 10-15 überhöht waren (mean 85.6°→18.9°
     nach Fix) - das trieb über die orographische Niederschlags-Komponente
     zusätzlich Flüsse an praktisch jedem Hang. Beide behoben +
     `RAIN_THRESHOLD` neu kalibriert (Default 0.2 statt 0.02/5.0,
     empirisch gegen echte precip_map-Verteilung geprüft: absteigende
     Creek>River>Grand-River-Verteilung statt Grand-River-Dominanz).
     Riverbank-Breitenbegrenzung (a) nicht angefasst.

4. **Settlements überarbeiten** — Es sollen echte Städte und Voronoi-Plots
   für die gesamte Karte entstehen, wie im Deskriptor/der Doku beschrieben.
   Vor Umsetzung mit dem User diskutieren. *(Priority: high)*
   - Update 2026-07-08: User hat eigene Änderungsvorschläge für Settlement,
     die zuerst besprochen werden müssen — deshalb auch die GPU-Anbindung
     für Settlement (siehe Ticket #40) bewusst zurückgestellt, um nicht
     an einem Algorithmus zu bauen, der ohnehin überarbeitet wird.
   - Update 2026-07-08 (nach Design-Gespräch): Architektur steht — zwei
     Plot-Maßstäbe (grobe Feld-Voronoi-Zellen außerhalb, feines
     Straßenraster+Hausparzellen innerhalb der Stadtgrenze), Stadtgrenze
     über terrain-cost-gewichtete Distanz vom Kern, Voronoi-Relaxation
     height/slope-moduliert mit Warm-Start über LOD-Stufen ("Gummiband"),
     Straßen sollen später Voronoi-Kanten folgen, Landmarks deterministisch
     ans Straßennetz angebunden, dekoratives Zufallswegenetz auf Phase 2
     verschoben. Details siehe docs/generation_pipeline_dependencies.md
     (#35/#36).
   - Umgesetzt (CPU-Referenz, jeweils per Smoke-Test verifiziert), Stand
     2026-07-09: #35 Stadtgrenze (`CityBoundaryAnalyzer`) + #37 Landschafts-
     Voronoi (`LandscapeVoronoiSystem`) über einen gemeinsamen terrain-cost-
     gewichteten Multi-Source-Flood (`_terrain_cost_voronoi()` in
     `core/settlement_generator.py`) — CPU-Pendant zum späteren GPU-JFA-Shader
     (Muster wie `shaders/water/jumpFloodLakes.comp`). Warm-Start über
     LOD-Stufen funktioniert (Match-Ratio 1.0 zwischen LOD64→LOD128). #30
     (Straßennetz) bevorzugt jetzt per Edge-Bias-Kostenterm den Verlauf entlang
     der Voronoi-Zellgrenzen (verifiziert: Straßen liegen im Schnitt deutlich
     näher an Zellgrenzen als der Kartendurchschnitt). #36 innerstädtisches
     Block-System (`CityBlockSystem`): Minimum-Spanning-Tree-Straßenskelett
     zwischen Haus-Ankerpunkten + darauf ausgerichtete Hausparzellen, strikt auf
     die Stadtgrenze beschränkt, Parzellenanzahl skaliert automatisch mit der
     verfügbaren Stadt-Pixelfläche pro LOD. #38 Außenverbindungen (2-3 Straßen
     zur Kartengrenze an Suitability-gewichteten, wasserfreien Randpunkten) +
     #39 Landmark-Anbindung (deterministisches Dijkstra zum nächsten
     Straßennetz-Punkt) sind ebenfalls fertig.
   - GPU-Anbindung (Ticket #40) umgesetzt, Stand 2026-07-09:
     `shaders/settlement/terrainCostFlood.comp` (JFA-Approximation von
     `_terrain_cost_voronoi()`) + `_dispatch_terrain_cost_flood()` in
     `gui/OldManagers/shader_manager.py`, genutzt von sowohl
     `CityBoundaryAnalyzer` als auch `LandscapeVoronoiSystem` über den
     gemeinsamen GPU→CPU-Fallback-Helper `_terrain_cost_voronoi_gpu_or_cpu()`.
     `SettlementGenerator` akzeptiert jetzt `shader_manager` (analog zu
     Terrain/Weather/Water/Biome), `GenerationOrchestrator.get_generator_instance()`
     injiziert ihn. Auf echter GPU getestet: 98.8% Pixel-Übereinstimmung mit der
     CPU-Dijkstra-Referenz (kleine Abweichungen durch JFA-Approximation,
     erwartet), 5×-170× Speedup je nach Kartengröße für den Flood selbst.
     **Aber:** Gesamt-Pipeline-Laufzeit ändert sich dadurch kaum, weil der Flood
     gar nicht der Flaschenhals ist — siehe Performance-Hinweis unten. civ_map-
     Decay und die Seed-Relaxation bleiben bewusst CPU (bereits <1s, GPU würde
     hier kaum etwas bringen).
   - **Perf-Bug gefunden+gefixt 2026-07-09** (User-Report: "hängt bei LOD1",
     lief nach sehr langer Zeit doch durch): `LandscapeVoronoiSystem.relax()`
     rief pro Nachbar-Paar `np.clip()`/`np.hypot()` als Python-Scalar-Aufrufe
     auf statt vektorisiert - bei vielen Tausend Paaren pro Iteration (steiles
     Terrain schrumpft `_local_min_spacing()` teils auf einen Bruchteil von
     `base_spacing`, wodurch weit mehr Seeds als bei flachem Terrain passen)
     dominierte der numpy-Funktionsaufruf-Overhead die Laufzeit. Komplett
     vektorisiert (ein `np.add.at()`-Scatter statt Python-for-Schleife über
     Paare). Zusätzlich `generate_seeds()` und `CityBlockSystem._sample_anchors()`
     von linearem O(n)-Scan auf Spatial-Hash-Grid umgestellt (gleiches
     Risikomuster). Gemessen mit echten Default-Parametern (`plotnodes=1000`
     u.a.): map_size 128 vorher ~11s, danach ~2s; map_size 32 vorher ~6.3s,
     danach ~1.7s. Bei map_size 333 dominiert jetzt wieder korrekt die alte
     `calculate_water_proximity()` (~52%) statt des neuen Codes (~16%) -
     bestätigt, dass der Fix wirkt und die neue Voronoi-Pipeline nicht mehr
     der Flaschenhals ist.
   - Noch offen: das alte `PlotNodeSystem` (#34) ist noch nicht durch/mit dem
     neuen Landschafts-Voronoi ersetzt bzw. eingeschränkt, das dekorative
     Zufalls-Wegenetz (Phase 2) fehlt noch.
   - **UI-Feedback-Runde 2026-07-09 (nach erstem GUI-Test durch User), alles
     umgesetzt+verifiziert:**
     - Settlement-Tab: "LOD Level & Validity" + "Generation Steps"-Checkliste
       (eigenes Status-Widget zwischen Pipeline-Status-Spalte und Parameter-
       Slidern) komplett entfernt, analog zu Ticket #6 unten (Vorbild
       GeologyTab, das dafür ebenfalls kein eigenes Widget hat).
     - Settlements/Landmarks/Roadsites/Roads/City Boundary sind jetzt
       Overlay-Checkboxen statt exklusiver Radio-Modi - kombinierbar mit
       jedem Basis-Layer (Suitability/Civ Map/Plot Boundaries/Landschafts-
       Voronoi/City Blocks), damit Plots+Straßen+Settlements gleichzeitig
       sichtbar sind. City Boundary zeichnet jetzt eine deutliche gelbe
       Kontur ("Stadtmauer") statt einer gefüllten Fläche
       (`overlay_city_boundary_contour()` in `map_display_2d.py`).
     - **civ_map-Reichweite substanziell vergrößert:** neuer Parameter
       `CIV_INFLUENCE_RANGE` (Bruchteil der Kartendiagonale, Default 0.30)
       ersetzt die alte, map_size-unabhängige `settlement.radius` (4-6px!)
       als Decay-Längenskala in `CivilizationInfluenceMapper.apply_settlement_influence()` -
       skaliert jetzt korrekt mit jeder LOD-Stufe. Mit Default-Settlements
       jetzt ~50% Wilderness-Anteil (vorher <5%, siehe Nutzer-Vorgabe "ca.
       die Hälfte der Karte"). Nebeneffekt: altes `PlotNodeSystem` (#34)
       erzeugt jetzt tatsächlich sichtbar viele Plots, weil es vorher kaum
       Nicht-Wilderness-Fläche zum Spawnen hatte. Dabei gleich vektorisiert
       (vorher Pixel-für-Pixel-Python-Schleife über die gesamte Karte pro
       Settlement - mit der neuen, oft kartenweiten effective_radius wäre das
       sehr langsam geworden).
     - **Perf-Bug #2 gefunden+gefixt** (User: "hängt bei höchster LOD,
       ungleich map_size"): `calculate_water_proximity()` war O(H×W×Wasserpixel)
       (Pixel-für-Pixel-Python-Schleife, jeder Pixel berechnete seine Distanz
       zu JEDEM Wasserpixel neu) - jetzt mit `scipy.ndimage.distance_transform_edt()`
       auf O(H×W) vektorisiert, numerisch exakt verifiziert identisch zum alten
       Ergebnis. Gleiches Problem in `calculate_landmarks()`
       (`_check_elevation_suitability()` rief `np.min()`/`np.max()` auf die
       komplette heightmap PRO PIXEL auf) behoben, ebenfalls vektorisiert und
       auf Ergebnis-Identität geprüft. Kompletter Pipeline-Lauf bei
       Default-Parametern: map_size 333 vorher ~17.5s, danach ~3.6s;
       map_size 288 (nicht-Zweierpotenz) danach 1.7s (kein Hängen mehr).
       War der eigentliche, größte Anteil an der vom User beobachteten
       "hängt"-Wahrnehmung, nicht ein spezifischer Nicht-Zweierpotenz-Bug.
   - **PlotNodeSystem (#34) komplett überarbeitet 2026-07-09 (User-Feedback-
     Runde 3), umgesetzt+verifiziert:**
     - `generate_plot_nodes()`: civ-wert-abhängige Abstoßung ersetzt die alte
       civ-blinde Gleichverteilung + wirkungslose feste 3px-Abstoßung
       (`optimize_node_positions()`, jetzt komplett entfernt) - hoher Civ-Wert
       (nah an der Stadt) → kleinerer Mindestabstand → dichtere Nodes →
       kleinere Plots, niedriger Civ-Wert → größerer Abstand → größere Plots
       (Nutzer-Vorgabe). Verifiziert: 4.2x mehr Nodes im Hoch-Civ- als im
       Niedrig-Civ-Bereich bei sonst identischer Fläche.
     - Neue `PlotEdge`-Datenklasse: jede Delaunay-Kante ist jetzt adressierbar
       (`edge_id`, `node_a`, `node_b` - Nutzer-Vorgabe "Plotnode 234 und 260
       teilen sich Kante 839"), mit Länge und **Wegintegral-Höhenkosten**
       (`_cumulative_height_cost()` summiert |Höhenänderung| entlang der
       Linie statt nur der Endpunkt-Differenz - eine Linie über Hügel-und-Tal
       hat sonst netto ~0 Differenz, obwohl tatsächlich Auf-/Abstieg
       zurückgelegt wird). Bewegungskosten = `length * (1 + height_cost_factor
       * mittlere_Pfadsteigung)` - dieselbe Formel-Familie wie überall sonst
       im Rework (City-Boundary, Voronoi-Flood), nicht das wörtliche
       "Länge × Höhe" aus der Nutzer-Beschreibung (das würde bei flachen
       Pfaden Kosten = 0 ergeben, ein Dijkstra-Degenerationsfall).
     - Neue `PlotNodeSystem.simulate_plot_traffic()`: jede PlotNode = eine
       Familie, die einmal den günstigsten Weg zur nächstgelegenen Siedlung
       nimmt (Multi-Source-Dijkstra über den Plot-Graphen, NICHT das
       Pixel-Grid) und dabei jede durchquerte Kante um 1 Traffic-Punkt erhöht.
       Zusätzlich tauschen die durch das bestehende Straßennetz verbundenen
       Siedlungspaare `plot_intercity_traffic` (Default 30) Personen pro Weg
       aus. Klassifikation ab `plot_path_traffic_threshold` (Default 25) zu
       "path", ab `plot_road_traffic_threshold` (Default 75) zu "road".
     - **Performance verifiziert wie vom User erfragt:** Plot-Graph hat nur so
       viele Knoten wie PLOTNODES (nicht Pixel-Grid-Größe) - 1800 Nodes/5373
       Kanten in 0.3s, kompletter Settlement-Durchlauf bei map_size 333 mit
       vollem Traffic-Sim nur ~4.4s (vs. ~3.6s Baseline ohne). Bei Default-
       Parametern: 110-120 klassifizierte Pfade/Straßen aus ~1000 Plotnodes.
     - Neue Parameter: `PLOT_BASE_SPACING`, `PLOT_CIV_SPACING_FACTOR`,
       `PLOT_HEIGHT_COST_FACTOR`, `PLOT_PATH_TRAFFIC_THRESHOLD`,
       `PLOT_ROAD_TRAFFIC_THRESHOLD`, `PLOT_INTERCITY_TRAFFIC`.
     - **Bewusst nicht umgesetzt (Scope-Cut):** keine Visualisierung der neuen
       Pfade/Straßen im Settlement-Tab - `plot_edges` ist im
       Calculator-Output/`SettlementData` verfügbar, aber noch kein Overlay
       in `map_display_2d.py`/`settlement_tab.py` dafür verdrahtet.
     - **Nachtrag (User-Rückfrage, selber Tag):** Traffic wird pro Aufruf
       komplett neu berechnet (kein Akkumulieren über LOD-Stufen hinweg,
       verifiziert: identischer Input → identische Traffic-Werte). Was
       vorher fehlte: die Node-*Positionen* wurden nicht über LOD-Stufen
       warmgestartet (anders als beim Landschafts-Voronoi), und Traffic hatte
       keine physische Wirkung auf die Geometrie. Beides jetzt ergänzt:
       - `generate_plot_nodes(previous_nodes=...)`: Warm-Start der
         Node-Positionen über LOD-Stufen (relativ 0..1 gespeichert unter dem
         neuen Output-Key `plot_node_positions`, analog zu
         `_calc_landscape_voronoi()`).
       - Neue `PlotNodeSystem.apply_traffic_attraction()`: stark genutzte
         Kanten ziehen ihre beiden Nodes über die LOD-Iterationen hinweg
         schrittweise zusammen ("Gummiband"), Gegenkraft ist derselbe
         civ-gewichtete Mindestabstand wie bei der Node-Generierung (Terrain/
         Civ-Dichte/Nähe zu anderen Plots) - eine Kante kollabiert nie unter
         diesen Mindestabstand.
       - `calculate_plots()` läuft jetzt zweistufig: Pass 1 (Graph+Traffic auf
         der warmgestarteten Ausgangsgeometrie) ermittelt stark genutzte
         Kanten, Attraction verschiebt die Nodes, Pass 2 (Graph+Traffic auf der
         angepassten Geometrie) ist der tatsächlich zurückgegebene/angezeigte
         Zustand.
       - Neuer Parameter `PLOT_TRAFFIC_ATTRACTION` (Default 0.05, bewusst nur
         grob kalibriert - Feintuning des Kräftegleichgewichts ist laut
         Nutzer-Vorgabe ein späterer Schritt).
       - Verifiziert: Traffic bleibt reproduzierbar bei wiederholten Aufrufen
         (kein Akkumulieren), keine Mindestabstands-Verletzungen über 6
         Iterationen, Settlement-zu-Settlement-Graph-Distanz nimmt über die
         Iterationen tatsächlich ab (~243 → ~228, Gummiband-Effekt), 6
         Iterationen in 0.42s.
   - **Runde 4 (User-Test des `tools/plot_physics_lab.py`-Tools), umgesetzt+verifiziert:**
     - **Marktplatz-Node je Settlement:** `PlotNode` hat jetzt ein
       `settlement_id`-Feld (-1 = normaler Node, sonst Settlement-ID). Jede
       Siedlung bekommt in `generate_plot_nodes()` einen echten Marktplatz-
       Node an ihrer exakten Position (bypasst `valid_mask`, da Stadt-Kerne
       sonst ausgeschlossen wären), der ganz normal an Delaunay-Triangulation
       und Traffic-Graph teilnimmt statt eines separaten virtuellen Anker-
       Knotens (deutliche Vereinfachung von `simulate_plot_traffic()`).
       Marktplatz-Nodes bleiben in `apply_traffic_attraction()` fixiert
       (nie verschoben - repräsentieren die reale Settlement-Position).
     - **Chancendegressions-Formel für die Stadtwahl:** neue
       `PlotNodeSystem._rank_distance_weights(n)` - jede PlotNode verteilt
       ihre "Familien-Masse" jetzt auf ALLE erreichbaren Marktplätze statt nur
       auf den nächsten: P(i)=0.5^i für die i-nächste Stadt, die entfernteste
       bekommt denselben Wert wie die zweit-entfernteste (Summe = 1.0) -
       exakt die vom User vorgegebene Formel/Tabelle (2 Städte: 50/50, 3:
       50/25/25, 4: 50/25/12.5/12.5, ...), per Unit-Test gegen die Tabelle
       verifiziert. `PlotEdge.traffic` ist dadurch jetzt fraktional
       (`float` statt `int`) - Schwellwerte (`PLOT_PATH_TRAFFIC_THRESHOLD`
       etc.) bleiben unverändert, könnten aber neu kalibriert werden wollen,
       da eine einzelne PlotNode ihren Beitrag jetzt auf mehrere Pfade
       aufteilt statt ihn komplett auf einen zu konzentrieren.
     - Alle bestehenden Smoke-Tests laufen unverändert durch (Traffic
       akkumuliert weiterhin nicht über Aufrufe, Gummiband-Effekt weiterhin
       nachweisbar, Performance unverändert schnell).
   - **Neues Test-Tool:** `tools/plot_physics_lab.py` - interaktives PyQt5+
     matplotlib-Programm zum Live-Ausprobieren der PlotNode-Physik (einfache
     Terrain-Heightmap nach Default-Parametern, 3 Settlements, Landschafts-
     Voronoi als echte Vektorgrafik statt Bitmap, Live-Loop mit Slidern für
     alle Plot-Parameter). Start: `.venv/Scripts/python.exe tools/plot_physics_lab.py`.
   - **Offene Frage an User (noch nicht entschieden):** Landschafts-Voronoi
     (`LandscapeVoronoiSystem.generate_seeds()`) schließt aktuell nur
     Stadt-Kerne aus, NICHT Wilderness (civ<0.2) - dadurch ziehen sich die
     Landschafts-Voronoi-Linien auch durch Wildnis-Gebiete. User sagt, das
     gefällt ihm ("Verbindungen durch die Wildnis"), daher bewusst
     unverändert gelassen.
   - **Runde 5 (User probiert `tools/plot_physics_lab.py` weiter aus), 4 neue
     Explorations-Features - bewusst NUR im Lab-Tool, nicht in
     `core/settlement_generator.py` (User: "das ist ja ein Testprogramm in dem
     ich das ergründen will"):**
     - **Wege-Darstellungs-Umschalter:** Checkbox "Wege folgen
       Grundstücksgrenzen (statt durch Grundstücke)". Aus = bisheriges
       Verhalten (Delaunay-Kanten zwischen Plot-Node-Zentren, laufen quer
       durchs Grundstück). An = neue `_simulate_ridge_traffic()`: baut ein
       eigenes `scipy.spatial.Voronoi`-Kantennetz aus den aktuellen Plot-Node-
       Positionen (Grundstücksgrenzen als Graph), simuliert dieselbe Rang-
       Distanz-gewichtete Verkehrsverteilung darauf - nur zur Visualisierung,
       die eigentliche Node-Physik (Attraction/Warm-Start) bleibt immer auf
       dem Delaunay-Graphen. Optisch deutlich unterschiedliches Wegenetz
       verifiziert (Screenshot-Vergleich).
     - **Keine Plots im Stadt-Inneren + Grenz-Nodes:** neue
       `_generate_city_boundary_nodes()` sampelt Nodes entlang der
       city_mask-Außenkontur (gelbe Dreiecke im Tool) statt sie im
       Stadt-Inneren zuzulassen (Trick: city_mask-Fläche in der für
       `generate_plot_nodes()` übergebenen civ_map künstlich auf 1.0 gesetzt,
       nutzt den bestehenden civ<1.0-Ausschluss wieder). Jeder Grenz-Node
       bekommt in `_tick()` eine direkt injizierte Kosten-0-`PlotEdge` zum
       Marktplatz seiner Stadt (verhindert, dass ein einzelner Grenz-Node wie
       eine eigene kleine Stadt behandelt wird) und bleibt bei der Attraction
       fixiert (wie die Marktplätze).
     - **Civ Influence Decay jetzt einstellbar:** neuer Slider im Hintergrund-
       Bereich, macht das Gefälle "Plot-Abstand vs. Distanz zur Stadt"
       direkt tunbar statt nur indirekt über civ_influence_range.
     - **Spline-Wiggle für Wege:** neue `_wiggly_path()` - Slider 0% (immer
       gerade) bis 70% (stark verschlungen), tatsächliche Welligkeit pro Kante
       skaliert zusätzlich mit der mittleren Steigung entlang der Strecke
       (flach = fast gerade auch bei hohem Slider-Wert, steil = deutlich
       wellig) - wie reale Serpentinen-Wegführung. Gilt für beide
       Wege-Darstellungs-Modi.
     - Alle vier Features per Offscreen-Screenshot-Test verifiziert (keine
       Abstürze, sichtbar korrektes Verhalten).
   - **Bekannt, noch nicht adressiert:** alle neuen Settlement-Parameter aus
     dieser und der vorigen Runde (`city_reach_factor`, `voronoi_base_spacing`,
     `voronoi_relax_iterations`, `road_voronoi_edge_bias`, `house_spacing`,
     `civ_influence_range`, alle `plot_*`-Parameter) haben noch keine Slider
     im Parameter-Panel - nutzbar nur über ihre (jetzt abgestimmten) Defaults,
     nicht per UI einstellbar.
   - **Performance-Hinweis (wichtig für zukünftige Optimierung):** Profiling
     zeigt, dass der eigentliche Flaschenhals der Settlement-Pipeline NICHT der
     neue Voronoi/Stadtgrenzen-Code ist (der braucht <1s), sondern die
     unveränderte, unvektorisierte `TerrainSuitabilityAnalyzer.calculate_water_proximity()`
     (~52% der Gesamtzeit bei map_size 256 — reine Python-Pixel-Loop mit O(H×W×Wasserpixel)-
     Distanzberechnung) sowie `calculate_landmarks()`s `_check_elevation_suitability()`
     (~20%, ruft `np.min()`/`np.max()` pro Pixel einzeln auf statt einmal
     vorab). Beides ist alter Code außerhalb des Settlement-Rework-Scopes,
     aber die mit Abstand größte Stellschraube für echte Performance-Gewinne
     bei Settlement — einfache NumPy-Vektorisierung (kein GPU nötig) würde
     hier vermutlich mehr bringen als alles Bisherige zusammen.
   - **Runde 6, echter Core-Fix (nicht nur Lab-Tool):** User-Beobachtung
     "Civ Influence verbreitet sich über flache Gebiete besser, als Berge
     hoch und runter" führte auf einen echten Sättigungsbug in
     `CivilizationInfluenceMapper.apply_settlement_influence()`: der
     Steigungs-Modifier `min(3.0, 1+slope_magnitude*2)` nutzte
     `slope_magnitude` direkt in rohen Höhenmetern-pro-Pixel (bei üblicher
     Terrain-Amplitude oft 10-90+), wodurch er auf 96.8% der Pixel sofort
     am Deckel saturierte — flache und gebirgige Gegenden wurden praktisch
     nicht unterschieden. Fix: `slope_magnitude` wird jetzt gegen das
     75.-Perzentil der jeweiligen Heightmap normalisiert, bevor der
     Multiplikator angewendet wird (Deckel auf 5.0 angehoben). Verifiziert:
     Sättigungsanteil 96.8% → 4.6%, fast-flach-Anteil 0.3% → 26.8%; alle
     bestehenden Smoke-Tests laufen weiter durch; Wildnis-Anteil bei
     Default-`civ_influence_range` verschob sich moderat (~51%→~40%, laut
     User "grob die Hälfte" weiterhin akzeptabel, nicht weiter
     nachkalibriert).
   - **Runde 6, alles Weitere bewusst nur im Lab-Tool** (`tools/plot_physics_lab.py`,
     `core/settlement_generator.py` unangetastet außer obigem Fix):
     - **Begriffs-Klarstellung Plotkern vs. Plotnode:** der bisherige
       "PlotNode" (Zentrum/Seed-Punkt) heißt jetzt "Plotkern" und wird rot
       dargestellt; "Plotnode" bezeichnet ab jetzt die Voronoi-Vertex-
       Kreuzungspunkte eines Plots (3-5 je nach Nachbarzahl), dargestellt in
       Blau, immer sichtbar (auch außerhalb des Grundstücksgrenzen-Modus,
       über neues `_compute_plot_vertices()`).
     - **Zufälliger Traffic-Spawn-Punkt:** `_simulate_ridge_traffic()` wählt
       den Einstiegspunkt jetzt zufällig unter den eigenen Voronoi-Vertices
       des Plots (`vor.point_region`/`vor.regions`) statt immer den
       nächstgelegenen zu nehmen; Fallback auf nächstgelegenen Vertex nur
       bei Zellen ohne endliche Vertices (Rand-/Unbounded-Zellen).
     - **Traffic Attraction feiner justierbar:** Slider-Range auf 0-3.0
       (vorher deutlich kleiner) für feinere Kontrolle bei hohen Werten.
     - **Mindestabstände:** Stadt-zu-Rand ≥25px (Suitability-Map wird vor
       `calculate_settlements()` in einem 25px-Rahmen genullt),
       Stadt-zu-Stadt ≥35px (bereits durch bestehende
       `calculate_settlements()`-Formel erfüllt, kein Code-Änderung nötig,
       nur per Assertion verifiziert).
     - **Map-Size auf 256×256** erhöht (vorher 160×160).
     - **"Stadtgröße"-Slider ersetzt City-Reach-Slider:** ein Regler leitet
       jetzt `city_reach_factor` (2.0 + size*6.0), `civ_influence_range`
       (0.15 + size*0.30) und `plot_intercity_traffic` (20 + size*40, also
       20-60 wie gefordert) gemeinsam ab, statt sie einzeln einzustellen.
     - Verifiziert per Offscreen-Screenshot + Assertions (Rand-/Stadt-
       Abstände eingehalten, abgeleitete Werte bei city_size=0.5 und 0.9
       korrekt berechnet, Rendering rot/blau visuell bestätigt).
   - **Runde 7, komplette Physik-Neufassung im Lab-Tool (9 vom User
     vorgeschlagene Regeln, ausgelöst durch die Beobachtung, dass die alte
     Traffic-Anziehung Plots vom Kartenrand wegzieht und Plotnodes in
     entlegenen Gegenden auf eine Linie kollabieren), weiterhin bewusst nur
     `tools/plot_physics_lab.py`:**
     1. Plotkern-Platzierung von hartem Reject-Sampling auf Best-Candidate-
        Sampling (Mitchell, k=15) umgestellt - deutlich gleichmäßigere
        Verteilung, verifiziert per Nearest-Neighbor-Distanz-Statistik
        (p50=7.2px vs. p95=12px bei 500 Nodes, keine Verklumpung).
     2. Civ-gewichtete Abstoßung unverändert (Formel gab es schon), jetzt
        über neuen `plot_repulsion_strength`-Slider skalierbar, wirkt auf
        frisch pro Tick berechnete Delaunay-Nachbarpaare.
     3. **Traffic-Anziehung komplett entfernt** (war die Ursache des
        Linien-Kollapses) und durch rang-distanz-gewichtete "Gravitation" zu
        den Städten ersetzt (dieselbe 0.5^Rang-Formel wie beim Traffic,
        über `plot_gravity_strength`-Slider, mit Distanz-Dämpfung gegen
        Instabilität nahe der Stadt). Gewichtungsmodell (rang-distanz vs.
        nur nächste Stadt vs. echte N-Body-Summe) per Nutzer-Entscheidung
        auf rang-distanz-gewichtet festgelegt.
     4. Grundstücksgrenzen (Voronoi-Kanten der Plotkerne) werden nur noch
        als gerade Linien gezeichnet - Splines gibt es jetzt ausschließlich
        beim separaten Wege-Netz.
     5. Grenz-Wegknoten-Anzahl pro Stadt skaliert jetzt 3-7 mit der
        Stadtgröße (vorher Abstands-basiert, beliebige Anzahl) über
        Winkel-Sortierung + gleichmäßiges Stride-Sampling je Settlement.
        Kein virtueller Marktplatz-Node mehr nötig - das Erreichen
        irgendeines Grenzknotens einer Stadt zählt direkt als Erreichen des
        Marktplatzes (einfacher als der vorherige Kosten-0-Kanten-Trick).
     6. (siehe Punkt 3)
     7. Jeder Plotkern bekommt einen festen `traffic_weight` (zufällig 3-5,
        einmalig bei Erzeugung gesetzt, bleibt über Ticks konstant - dynamisch
        angehängtes Attribut, kein neues Dataclass-Feld nötig).
     8. Wege sind jetzt ein eigenes Netz auf dem Plotnode-Graphen (der alte
        Delaunay-Zentren-Modus und der `boundary_mode`-Umschalter entfallen
        komplett - nur noch ein Wege-System). Dreistufige Klassifizierung
        (Pfad≥20, Weg≥40, Straße≥60) × ein `plot_tier_factor`-Slider
        (0.25-5). Traffic ist jetzt über Ticks persistent mit 50%
        Abklingen + frischem Beitrag pro Tick (`ridge_traffic_history`,
        geschlüsselt über gerundete Kanten-Mittelpunkt-Position, da
        Voronoi-Vertex-Indizes zwischen Ticks nicht stabil sind) - bewusste
        Abweichung von der früheren "kein Akkumulieren"-Regel aus Runde 3b,
        hier lab-lokal als EMA-Glättung fürs Live-Bild gedacht, nicht ohne
        erneute Entscheidung in core/settlement_generator.py zu übernehmen.
        Zwei getrennte Spline-Slider: `spline_wiggle_pct` (Kurvigkeit) und
        `spline_detail` (Wellenfrequenz).
     9. Wegintegral-Höhenkosten waren schon vorher korrekt (Linspace-Sampling
        entlang der Geraden, bestraft einen Hochpunkt zwischen zwei gleich
        hohen Endpunkten bereits richtig) - per Codeprüfung bestätigt, keine
        Änderung nötig, gilt jetzt für Plotnode- statt Delaunay-Kanten.
     - Bei Gelegenheit auch geklärt: Spline-Wiggle ist rein kosmetisch, wird
       nirgends in `PlotEdge.length`/movement_cost zurückgespeist - das reale
       Kostenmodell rechnet immer mit der geraden Verbindung, was konsistent
       ist, da die tatsächliche Graph-Kante eine Gerade ist.
     - Verifiziert per Offscreen-Test: 80-Tick-Stabilitätslauf (Positionen
       bleiben endlich/im Kartenbereich, Grenzknoten bleiben fixiert, Traffic
       schwankt aber divergiert nicht), Best-Candidate-Verteilungscheck (s.o.),
       Screenshot bestätigt visuell sinnvolle Zellformen und Straßenführung.

5. **Ladebalken kalibrieren** — Jeder Rechenschritt von Anfang bis Ende muss
   mit Kosten festgelegt sein. Das Ende ist NICHT von Mapsize Max abhängig,
   sondern vom "Final LOD" der gewählten Map Size. Beispiel: Map Size 333 →
   üblicherweise 32→64→128→256→333, also LOD5. Kann im Einzelfall abweichen.
   *(Priority: medium)*

6. **Settlement-Tab-UI aufräumen** — Zwischen System Status und den
   Parameter-Slidern steht aktuell zu viel, was da nicht hingehört. Vorbild:
   Geology-Tab. (Bei Terrain wirkt der System-Status-Bereich zu hoch, als ob
   die Höhe des Felds variabel wäre und noch Platz übrig gewesen wäre — das
   gehört mit repariert.) *(Priority: medium)*

7. **Breite ändert sich beim Klicken auf "Berechnen"** — Beim Klicken ändert
   sich die Breite der Slider (evtl. auch anderer Elemente). Möglicherweise
   ändert sich auch die Panel-Größe. Ursache unklar, muss untersucht werden.
   *(Priority: low)*

41. **GPU-Anbindung für Geology** — Wie Settlement bisher komplett ohne
    GPU-Anbindung, außerhalb des Water/Weather/Biome-Fokus der GPU-Session
    vom 2026-07-08. *(Priority: medium)*

42. **Echter paralleler Watershed-Transform für `jumpFloodLakes.comp`** —
    GPU-seitige Becken-Zuordnung nutzt weiterhin nur "current_height >=
    seed_height" + Luftlinien-Distanz statt echter Erreichbarkeit über einen
    monotonen Abwärtspfad (siehe Ticket #2, Update 2026-07-09). Die CPU-Seite
    (`_apply_jump_flooding` in `core/water_generator.py`) wurde auf einen
    echten Priority-Flood/Multi-Source-Dijkstra umgestellt, was sich nicht
    trivial auf GPU parallelisieren lässt (Dijkstra ist inhärent sequenziell
    in Verarbeitungsreihenfolge; echte parallele Watershed-Transforms auf GPU
    sind ein eigenes, nicht-triviales Forschungsthema). Spill-Point-Filter in
    `_classify_lake_basins_vectorized` dämmt die Auswirkung stark ein, ersetzt
    aber keine korrekte Zuordnung. *(Priority: medium, bewusst zurückgestellt
    am 2026-07-09)*

43. **Sedimentations-Menge zu gering trotz Floodplain-Verteilung** —
    `_distribute_sediment_floodplain()` (neu 2026-07-09) verteilt Ablagerungen
    jetzt korrekt über eine lokale Nachbarschaft statt auf einem Pixel zu
    stapeln, aber die absolute Gesamtmenge bleibt gering: `settling_velocity`
    (Default 0.01) lässt nur ~1% des überschüssigen Sediments pro Iteration
    absetzen, bei nur ~7 Iterationen auf LOD3 setzen sich dadurch nur ~7% des
    transportierten Sediments in einem LOD-Durchlauf tatsächlich ab. Braucht
    eine eigene Kalibrierungsrunde (`settling_velocity`- und/oder
    Iterationsbudget-Anpassung). *(Priority: medium)*

44. **Feuchte-Skalierung in `weather_generator.py` fragil kalibriert** —
    `_calculate_atmospheric_moisture_cpu()`s Skalierungsfaktor (140.0,
    2026-07-09 empirisch gegen ein einzelnes Seed/Parameter-Sample gesetzt,
    vorher 10.0 mit dem Effekt "Kondensations-Niederschlag praktisch immer
    exakt 0") hat einen sehr schmalen Übergangsbereich zwischen "nirgends"
    und "überall" übersättigt (~10% Skalierungs-Spannbreite) - eine Folge
    davon, dass Transport+Diffusion das Feuchte-Feld stark glättet und wenig
    räumliche Varianz übrig lässt. Kann bei anderen Wetter-Parameter-
    Kombinationen wieder in einen der beiden Extremzustände kippen. Eine
    robustere Lösung bräuchte entweder mehr räumliche Varianz im
    Feuchte-Feld selbst oder eine adaptive statt einer festen Skalierung.
    *(Priority: medium)*

45. **GPU-Dispatch-Funktionen auf denselben Bug-Typ wie Lake-Detection
    prüfen** — Am 2026-07-09 wurde in `_classify_lake_basins_vectorized`
    (`shader_manager.py`) derselbe Bug wie im CPU-Pfad gefunden (Referenzhöhe
    falsch gewählt, siehe Ticket #2). Nur Lake-Detection wurde geprüft+
    gefixt (bewusst so vom User priorisiert) - Erosion/Sediment-Transport,
    Weather- und Biome-GPU-Pfade in `shader_manager.py` wurden NICHT auf
    dieselbe Fehlerklasse (falsche Referenzwerte, fehlende Grenzfall-Logik)
    untersucht. *(Priority: medium)*

46. **3D-View: `makeCurrent()`/`doneCurrent()`-Fix verifizieren** — User
    meldete durchgängig "Stacking" in der 3D-Ansicht (mehrfach gezeichnete/
    überlappende Geometrie) und persistentes Blau trotz mehrerer Fix-Runden.
    Gefunden+gefixt: `map_display_3d.py` rief nirgends `makeCurrent()`/
    `doneCurrent()` auf, obwohl `update_heightmap()` außerhalb von `paintGL()`
    direkt OpenGL-Buffer erstellt/löscht - potenziell im falschen Kontext
    (pro Tab existiert ein eigenes `MapDisplay3DWidget`). Zusätzlich einen
    dauerhaft hängenden `useShadows`-Uniform gefixt (nie zurückgesetzt,
    obwohl nie eine echte Shadow-Textur gebunden wird) und Blau aus der
    3D-Terrain-Farbskala entfernt (eigene, von der 2D-Colormap unabhängige
    Farbtabelle in `shaders/3d_display/terrain.frag`). **Nicht visuell
    gegen die laufende App verifiziert** (kein Werkzeug-Zugriff auf native
    Qt-Fenster von dieser Umgebung aus) - User-Bestätigung nach Neustart der
    App noch ausständig. *(Priority: high, braucht User-Test)*

## Erledigt (2026-07-08 — siehe docs/tickets.xlsx für Details)

1. ~~**GPU-Support prüfen**~~ — Vollständig untersucht und größtenteils
   behoben. Kernfund: `GenerationOrchestrator` hat NIE einen `ShaderManager`
   in irgendeinen Generator injiziert — GPU lief bei KEINEM der 6 Generatoren,
   unabhängig vom Shader-Code selbst. Nach dem Fix (dedizierter
   GPUWorker-Thread mit eigenem Offscreen-GL-Kontext): Water (alle 7
   Calculator-Knoten) und Weather (alle 4 Knoten) vollständig GPU-beschleunigt
   und in der echten Pipeline verifiziert (nicht nur isoliert getestet).
   Biome: 3 von 5 Knoten (die anderen 2 waren schon vektorisiert bzw. auf CPU
   nachweislich schneller). Terrain: Shadow-Raycast GPU-beschleunigt, Noise
   war noch Platzhalter (siehe Ticket #39, seit 2026-07-09 erledigt).
   Settlement seit 2026-07-09 ebenfalls erledigt (Ticket #40); Geology bleibt
   offen (Ticket #41).

Daneben mehrere kritische, unabhängig davon gefundene Bugs behoben (LOD-
Ceiling-Desync beim Auto-Start, Endlosschleife bei Regenerierung nach
Auto-Start, `impact_matrix`-Tippfehler "size"→"map_size", Pipeline-Status-
Panel jetzt granular über alle 34 Calculator-Knoten) — vollständige Liste
mit Root-Cause-Erklärungen in docs/tickets.xlsx, Nr. 27-38.

## Erledigt (2026-07-09)

39. ~~**Echte GLSL-Simplex-Noise für Terrain-GPU-Pfad schreiben**~~ — Kurt-
    Spencer-OpenSimplex-2D-Algorithmus in `shaders/terrain/noiseGeneration.comp`
    portiert (Hash-basierte Gradientenauswahl statt Permutationstabelle, da
    das generische GPU-Dispatch-Protokoll keine Puffer-Uploads kennt). Ersetzt
    die alte `sin(x)*cos(y)`-Platzhalter-Formel. In echtem Offscreen-GL-4.3-
    Kontext kompiliert, dispatcht und verifiziert (reale Varianz über mehrere
    Größen/Seeds, unabhängige Seeds unkorreliert). Das zuvor deaktivierende
    `if False and`-Guard in `core/terrain_generator.py` entfernt.

40. ~~**GPU-Anbindung für Settlement**~~ — War in diesem Backlog-Eintrag als
    offen gelistet, aber laut Ticket #4 (Settlements überarbeiten) bereits am
    2026-07-09 umgesetzt (`shaders/settlement/terrainCostFlood.comp` +
    `_dispatch_terrain_cost_flood()`, 98.8% Pixel-Übereinstimmung mit der
    CPU-Referenz) - Backlog-Status war veraltet, jetzt korrigiert.

- ~~**"Lake überall" auf Terrain/Water/Biome-Tabs**~~ (User-Report) und
  ~~**Terrain-Farbskala zeigt Blau (2D+3D)**~~ (User-Report) — siehe Ticket
  #2 für die Lake-Fixes (Spill-Point CPU+GPU, echter Watershed CPU-seitig)
  und Ticket #46 für die 3D-Fixes (Farbskala, `makeCurrent()`, `useShadows`).
  Zusätzlich, unabhängig von Lake-Detection, gefunden+gefixt: Biome-
  Klassifikation verglich `precip_map`/`soil_moist_map` (reale Skalen ~0-3
  bzw. 0-100) gegen `biome_definitions`-Bereiche in klassischen Whittaker-
  Jahres-Einheiten (0-4000 bzw. 0-1500) - jedes reale Werte fiel damit
  strukturell in die "outside range"-Straf-Zone für praktisch jedes
  feuchtere Biom. Bereiche in `core/biome_generator.py` `BaseBiomeClassifier`
  neu skaliert (Faktor 50/4000 bzw. 100/1500), Base-Biome-Verteilung danach
  spürbar vielfältiger (kein Einzel-Biom-Dominanz mehr in Testläufen).

- ~~**Sedimentation nur an einem Pixel statt Flood-Plane**~~ (User-Report) —
  `_distribute_sediment_floodplain()` neu in `core/water_generator.py`:
  verteilt jede Ablagerung über eine lokale Nachbarschaft, gewichtet invers
  zur lokalen Höhe (tiefere Punkte bekommen mehr). Absolute Sedimentations-
  Menge bleibt gering, siehe neues Ticket #43.

- ~~**Soil Moisture bimodal (nur ganz nass oder ganz trocken)**~~
  (User-Report) — größtenteils Kaskadeneffekt der Rain-Threshold- und
  Slope-Fixes (Ticket #3); zusätzlich einen eigenständigen Bug in der
  Kondensations-Feuchte-Berechnung gefixt (`weather_generator.py`,
  `_calculate_atmospheric_moisture_cpu()` skalierte `humid_map` so klein,
  dass die Übersättigungs-Schwelle nie erreicht wurde - Kondensations-
  Niederschlag war dadurch strukturell immer exakt 0). Kalibrierung fragil,
  siehe neues Ticket #44.
