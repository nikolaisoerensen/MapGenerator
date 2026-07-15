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
   - Update 2026-07-14: Wasserscheiden-Umleitung implementiert
     (`_redirect_basin_flow_to_spill()` in `core/water_generator.py`) - pro
     geschlossenem Becken wird die `flow_direction` der Senken-Pixel zum
     konkreten Spill-Nachbar-Pixel umgeleitet, per Union-Find (Kruskal-Stil)
     zyklenfrei über alle Becken hinweg gelöst (naive "immer zum eigenen
     niedrigsten Nachbarn"-Umleitung erzeugte zunächst Zyklen zwischen
     gegenseitig niedrigsten Nachbar-Becken, siehe Root-Cause-Analyse in der
     Session). Verifiziert: 78→16 residuale unaufgelöste Senken (79%
     Reduktion) auf Testkarte, keine Zyklen mehr, 76.2% der Kartenrand-Pixel
     haben jetzt nichtnull `flow_accumulation`. CPU-only (siehe Ticket #42
     für GPU-Parität). Läuft sowohl in `water.flow_network` (#16, eigene
     Akkumulation) als auch in `water.steepest_descent` (#17, für Erosion/
     Sedimentation) konsistent. Verbleibende ~16 Restfälle (z.B. echte
     Plateaus ohne jeden tieferen Nachbarn) bleiben offen - kein vollständiger
     Depression-Filling-Algorithmus (ArcGIS-Style Fill+FlowDirection wäre die
     vollständige Lösung, aber deutlich aufwändiger).

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
   - Update 2026-07-14 (a+b jetzt erledigt): neue, von `rain_threshold`
     komplett entkoppelte `STREAM_THRESHOLD`-Konstante (`gui/config/value_default.py`)
     - vorher war Creek = 1x `rain_threshold`, identisch mit der
     Regen-Quellen-Schwelle selbst, wodurch jedes einzelne Regen-Pixel sofort
     als Fluss galt, ganz ohne echten Zufluss von Nachbar-Zellen. Zusätzlich
     echte Flussbreite aus dem Volumen: `ManningFlowCalculator._optimize_channel_geometry()`
     berechnete die Breite über die Kontinuitätsgleichung bereits, verwarf sie
     aber bisher (nur `area`/`hydraulic_radius` wurden weiterverwendet). Neue
     Methoden `calculate_channel_width()` (Breite = sqrt(ratio*area) aus dem
     bereits vorhandenen Querschnitt, Tal-Breite-zu-Tiefe-Ratio wie in
     `_optimize_channel_geometry`) + `paint_channel_width()` (dilatiert jedes
     Zentrallinien-Pixel per `distance_transform_edt` um seine Breite/2,
     übernimmt die Klassifikations-Stufe vom nächsten Zentrallinien-Pixel,
     Lake-Pixel bleiben unberührt) in `core/water_generator.py`. Läuft in
     `_calc_manning_flow()` (#18) und überschreibt gezielt nur den
     `water_biomes_map`-Key von `water.flow_network` (#16) im
     Calculator-Storage (`set_calculator_output()` ist ein Key-Level-Upsert,
     kein Dict-Replace - `flow_accumulation` bleibt unberührt) - dadurch
     sehen alle bestehenden Konsumenten (`biome_generator.py`,
     `settlement_generator.py`, `assemble_water_data`) die gemalte Breite
     automatisch, ohne eigenen neuen Calculator-Knoten und ohne
     Konsumenten-Umhängung. Mit Unit- und Integrationstest verifiziert
     (Formel-Korrektheit, symmetrische Dilatation, Lake-Schutz,
     Massenerhalt/Klassifikations-Erhalt; volle Breite nur bei ausreichend
     feiner Auflösung sichtbar - ein 2-6m breiter Bach bleibt bei sehr grober
     Auflösung/großem `WORLD_SIZE_KM` physikalisch unter Pixelgröße, das ist
     korrektes Verhalten, kein Bug). GPU-Pfad für Manning-Flow liefert
     weiterhin keine Breite (bekannte Lücke, siehe Ticket #45).

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

~~7. **Breite ändert sich beim Klicken auf "Berechnen"**~~ — siehe Erledigt
   2026-07-14.

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

~~43. **Sedimentations-Menge zu gering trotz Floodplain-Verteilung**~~ — siehe
    Erledigt 2026-07-14.

~~44. **Feuchte-Skalierung in `weather_generator.py` fragil kalibriert**~~ —
    siehe Erledigt 2026-07-14.

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

## Erledigt (2026-07-14)

- ~~**map_seed ändert die Heightmap nicht**~~ (User-Report) — Root Cause:
  `SimplexNoiseGenerator` erstellte den `OpenSimplex`-Generator einmalig bei
  der Konstruktion und wurde nie neu geseedet; `_calc_noise()` (der bei jedem
  Generate-Klick laufende Knoten) las `map_seed` außerdem nie aus den
  Parametern. `GenerationOrchestrator.get_generator_instance()` cached
  `BaseTerrainGenerator` zusätzlich dauerhaft und konstruierte es ganz ohne
  `map_seed`-Argument (Klassen-Default 42 griff permanent, unabhängig vom
  UI-Slider). Fix: `SimplexNoiseGenerator.set_seed()` ergänzt (reseeded den
  `OpenSimplex`-Generator bei abweichendem Seed), `_calc_noise()` liest
  `map_seed` jetzt pro Aufruf. Verifiziert über Korrelation zweier
  Heightmaps mit unterschiedlichem Seed (niedrig, nicht ~1.0).

- ~~**#7 Breite ändert sich beim Klicken auf "Berechnen"**~~ — Root Cause:
  `StatusIndicator.status_text` (`gui/widgets/widgets.py`) war ein `QLabel`
  ohne feste Breite; unterschiedlich lange Status-Texte ("Generating LOD
  1/6 (25%)" vs. "Generation completed (LOD 6)") änderten die natürliche
  Widget-Breite und quetschten Geschwister-Widgets (Parameter-Slider) im
  selben Layout. Fix: feste Breite (`setFixedWidth`) + `QFontMetrics.elidedText()`
  statt wachsender Breite. Zentraler Fix an der Widget-Klasse, wirkt
  automatisch auf alle Tabs.

- ~~**Ticket #2 (Teil): Flüsse versiegen in geschlossenen Becken**~~ und
  ~~**Ticket #3 (a+b): Riverbank-Breite/Fluss-Erkennung**~~ — siehe
  ausführliche Updates 2026-07-14 unter den jeweiligen Tickets oben
  (Wasserscheiden-Umleitung per Union-Find, `STREAM_THRESHOLD`-Entkopplung,
  Flussbreite aus dem Volumen via `calculate_channel_width()`/
  `paint_channel_width()`). Ticket #2 bleibt für die verbleibenden ~16
  Restfälle (echte Plateaus) und für GPU-Watershed-Parität (Ticket #42)
  offen; Ticket #3 gilt für a+b als erledigt.

- ~~**Ticket #43: Sedimentations-Menge zu gering**~~ — Zwei unabhängige
  Fixes: (1) GPU-Pfad in `ErosionSedimentationSystem.simulate_erosion_sedimentation()`
  lief bisher NICHT durch `_distribute_sediment_floodplain()` (nur der
  CPU-Fallback tat das) - GPU- und CPU-Ergebnisse liefen dadurch auseinander;
  jetzt auf beiden Pfaden konsistent. (2) `WATER.SETTLING_VELOCITY`-Default
  von 0.01 auf 0.08 angehoben (`gui/config/value_default.py`, innerhalb des
  bestehenden Slider-Bereichs 0.001-0.1) - bei 0.01 setzten sich empirisch
  nur ~0.8% des transportierten Sediments über das gesamte
  LOD-Iterationsbudget ab, der Rest verließ die Karte praktisch unverändert;
  0.08 ergibt empirisch ~7x mehr abgesetztes Sediment bei identischem
  Iterationsbudget. Kalibrierung über direkten Erosion/Sedimentation-Summen-
  Vergleich über mehrere `settling_velocity`-Werte verifiziert.

- ~~**Ticket #44: Feuchte-Skalierung fragil (bimodal trocken/nass)**~~ —
  Drei Teil-Fixes in `core/weather_generator.py`: (1) `geology.hardness`
  für Weather erreichbar gemacht (`_calc_humidity()` holt `hardness_map`
  jetzt via `get_calculator_output()`s bestehendem Best-verfügbares-LOD-
  Fallback statt einer harten `CALCULATOR_GRAPH`-Dependency - Geology/Weather
  sind Geschwister-Knoten ohne feste Reihenfolge, eine echte depends_on-Kante
  hätte den `CalculatorDispatcher` gezwungen, Weather für jedes LOD auf
  Geology warten zu lassen). (2) `_apply_humidity_diffusion()` nutzt jetzt
  echtes `scipy.ndimage.gaussian_filter` statt eines handgerollten
  4-Nachbar-Box-Blurs, geblendet mit dem Original gewichtet nach normierter
  `hardness_map` (weiches Gestein diffundiert stärker, hartes Gestein hält
  Feuchte lokaler - "sedimentation verteilt das besser als harter Stein").
  Ohne verfügbare `hardness_map` (z.B. allererster Lauf vor Geology) fällt
  das auf vollständige Diffusion zurück, kein Crash. (3) Mehrgenerationen-
  Puffer: `_calc_humidity()` blendet mit dem eigenen Vor-LOD-Ergebnis
  (gewichteter Mittelwert 0.4 alt / 0.6 neu, NICHT additiv wie bei Erosion,
  da Feuchte keine kumulative Größe ist) - dämpft das "kippt bei jedem Lauf
  komplett in trocken oder nass"-Verhalten. Verifiziert: Histogramm über
  mehrere LOD-Stufen zeigt keine bimodale 0%/100%-Verteilung mehr, humid_map-
  Mittelwert bleibt über LOD-Übergänge stabil statt zu springen. GPU-Pfad
  (`humidityDiffusion.comp`) bekommt weiterhin nur den einfachen Box-Blur
  ohne Hardness-Gewichtung (bekannte Lücke, analog zur GPU-Watershed-Parität
  in Ticket #42 - eigener Folge-Task).

- ~~**Saisonale (6-Monats-) Wettersimulation**~~ (User-Request nach
  Biome-Screenshot: Karte fast vollständig von Fluss-Biomen dominiert +
  Badlands-Übersättigung) — Root Cause: Windrichtung war in der CFD-
  Simulation hartcodiert (immer West-Ost-Druckgefälle), wodurch
  orographischer Niederschlag strukturell immer an denselben Hang-Seiten
  auftraf. Große Umstellung in `core/weather_generator.py`: alle 4
  Calculator-Knoten (`_calc_temperature/_calc_wind/_calc_humidity/
  _calc_precipitation`) laufen jetzt intern über 6 saisonale
  Zwei-Monats-Perioden (Jan/Feb..Nov/Dez), Water/Biome konsumieren
  weiterhin nur den saisonalen Mittelwert (Key-Level-Upsert, `_monthly`-
  Listen liegen als eigene Output-Keys daneben, kein Water/Biome-Code
  geändert). Kernstücke: (1) neuer `prevailing_wind_direction`-Parameter
  (Default 225°/Südwest) ersetzt das hartcodierte Gefälle, CPU
  (`_build_directional_pressure_field`) UND GPU
  (`shader_manager.py _dispatch_wind_field_cfd`) synchron gefixt,
  regressionsgetestet gegen die alte Formel bei 0°. (2) Echte
  astronomische Sonnenstandsberechnung (Deklination/Zeitgleichung/
  Stundenwinkel, Formel von geoastro.de, neu in `core/terrain_generator.py`
  `calculate_solar_position()`/`generate_seasonal_sun_angles()`) liefert pro
  Periode 7 reale Sonnenwinkel für `ShadowCalculator.calculate_shadows(
  ..., sun_angles_override=...)` - Weather bekommt dadurch pro Monat eine
  eigene, physikalisch plausible Shadowmap (Winter flacher, Sommer steiler),
  der geteilte `terrain.shadow`-Knoten bleibt unverändert. (3) Klimazonen-
  Profile (`WEATHER.CLIMATE_ZONE_SEASONAL_OFFSETS`, vorerst nur
  "temperate") geben Temperatur/Sonnenkraft/Windstärke/Feuchte eine
  plausible Jahresform um den User-Slider-Wert als Zentrum, statt beliebiger
  Zufallswerte - deterministisch aus `map_seed`. Neue Parameter
  `air_humidity_entry`, `map_latitude`/`map_longitude` (treiben die
  Sonnenstandsformel). (4) Wind-Tab zeigt jetzt ein echtes Richtungs-
  Pfeilfeld (`map_display_2d.py _render_wind_map()`, matplotlib quiver,
  fest 32x32 Pfeile unabhängig von map_size) statt nur einer
  Stärke-Heatmap. (5) Weather-Tab animiert die gerade ausgewählte Karte
  (Temperature/Precipitation/Humidity/Wind) jede Sekunde durch die 6
  Monats-Schnappschüsse (`QTimer`, Start/Stop-Gating analog
  `map_display_3d.py`). Performance: GPU-Pfad für alle 4 Weather-Knoten
  bereits vorhanden und genutzt (ein veralteter Kommentar in
  `weather_tab.py` hatte das fälschlich verneint); CPU-only-Messung bei
  LOD3 (128x128) lag bei ~24s (~8% des 300s-Timeout-Budgets pro
  LOD-Anfrage). Umfassend smoke-getestet (Sonnenstandsformel gegen
  Referenzwerte, CPU/GPU-Formel-Parität, saisonale Kohärenz Sommer wärmer
  als Winter, Cross-LOD-Feuchte-Puffer pro Monat, vollständige
  Speicherkette bis `weather_tab.py`s echtem Abrufpfad, Wind-Rendering
  end-to-end über eine echte `MapDisplay2D`-Instanz).

- ~~**3D-Daten-Overlays, feste Farbskalen, Wind-Realismus, Feuchte-Fix**~~
  (User-Feedback nach der saisonalen Wettersimulation, 4 Punkte) —
  (1) **3D-Overlays für alle ~15 Layer**: die Textur-Upload-Infrastruktur
  und der Fragment-Shader (`useOverlay`/`overlayTexture`/`renderMode`)
  existierten bereits als fertiges Grundgerüst, `_render_overlay()`
  (`gui/widgets/map_display_3d.py`) war aber nur ein leerer TODO-Stub -
  jetzt echter `glGenTextures`/`glTexImage2D`-Upload + zweiter Draw-Call
  (mit temporär `GL_LEQUAL` gegen den identischen Tiefenwert des
  Basis-Passes) + neue `_colorize_layer()`-Funktion (Skalar/Kategorie zu
  RGB, nutzt dieselbe Range-Tabelle wie 2D). `getTerrainColor()`
  (renderMode 0, Terrain-Tab) unterstützte als einzige der 6 Farb-
  Funktionen KEIN Overlay-Blending im Shader (`terrain.frag`) - neue
  `getTerrainColorBlended()` ergänzt, für den neuen Slope-Layer gebraucht.
  Zentraler Wiring-Fix in `gui/tabs/base_tab.py`
  `_push_data_to_current_display()`: pusht Overlay-Daten jetzt IMMER an
  `update_overlay_data()`, unabhängig vom aktuell sichtbaren View (2D/3D-
  Widget existiert immer pro Tab) - wirkt automatisch für alle Tabs über
  eine neue Namens-Mapping-Tabelle (2D-`layer_type`-Strings wie `temp_map`
  vs. 3D-interne Namen wie `temperature`). Neue Overlay-Slots für
  `slope`/`flow_map`/`super_biome_mask` ergänzt (inkl. Checkboxen).
  Nebenbefund beim `_LAYER_NAME_MAP_3D`-Aufbau: `biome_tab.py` sendet für
  alle 3 Anzeige-Modi denselben `layer_type`-String - 3D zeigt daher nur
  einen gemeinsamen Biome-Slot statt 3 getrennter, bewusste Vereinfachung.
  (2) **Feste Farbskalen**: neue `CanvasSettings.CANVAS_2D["layer_ranges"]`-
  Tabelle (`gui/config/gui_default.py`), von 2D UND 3D genutzt - vorher
  hatten nur Heightmap und Biome-Map feste Skalen, alle anderen Layer
  skalierten pro Frame automatisch (Ursache des gemeldeten "Flackerns" bei
  der Monats-Animation). Dabei gefunden+gefixt: `temp_map`/`precip_map`
  waren durch einen String-Mismatch (`"temperature_map"`/`"precipitation_
  map"` vs. tatsächlich `"temp_map"`/`"precip_map"`) in `map_display_2d.py`s
  Dispatch NIE erreichbar - liefen immer über den generischen Viridis-
  Fallback statt der eigentlich vorgesehenen RdBu_r-Skala (Blau=kalt,
  Rot=warm) für Temperatur. (3) **Wind-Realismus**: `_generate_pressure_
  noise()` nutzte denselben Noise-Seed/dieselben Koordinaten JEDEN Monat
  (kein `month_index`-Bezug) - jetzt Koordinaten-Offset pro Monat (CPU
  UND GPU `pressureNoise.comp`-Dispatch, dort bereits ein Koordinaten-
  Offset-Uniform vorhanden, kein Shader-Change nötig). Zusätzlich neue
  Druck-Terrain-Kopplung PRO Iteration (Terrain-Höhe senkt lokal den
  effektiven Druck, VOR der Gradientenberechnung, nicht nur additiv aufs
  Windfeld NACH der Berechnung wie der bestehende Term) - räumliche
  Windrichtungs-Varianz auf einem synthetischen Tal/Grat-Höhenmodell stieg
  dadurch messbar von 2.65 auf 3.48 (+31%). (4) **Bodenfeuchte**: `Soil
  MoistureCalculator` setzte bedingungslos JEDEN `water_biomes_map>0`-Pixel
  auf 100% - die Breiten-Malen-Änderung (Stage 3, diese Session) hatte
  die Wasser-Pixel-Fläche vergrößert und damit die 100%-Quellfläche direkt
  mitvergrößert ("fast 100% fast überall"-Meldung). Fix: neuer
  `water_biomes_map_centerline`-Output (die Klassifikation VOR dem Breiten-
  Malen, in `_calc_manning_flow` gesichert bevor sie überschrieben wird)
  als 100%-Quellfläche statt der gemalten Karte (CPU UND GPU-Dispatch
  `soilMoistureGaussian`) - auf synthetischem Test (9px breite vs. 1px
  breite Flussmalerei) sank der Anteil an >90%-feuchten Pixeln von 13.6%
  auf 1.6% (~8.7x). Alle vier Punkte umfassend smoke-getestet (feste
  Farbskalen über Frame-Wechsel stabil, Overlay-Rendering-Pfad bis zum
  echten `glTexImage2D`-Aufruf reproduzierbar, Noise-Varianz pro Monat,
  Wind-Richtungs-Varianz vorher/nachher per `git stash`-Vergleich,
  Feuchte-Flächenanteil vorher/nachher) - volle visuelle Bestätigung
  (3D-Overlay-Optik, Farbskalen in der laufenden Animation) braucht wie
  bei allen 3D-Änderungen dieser Session den Nutzer-Test in der echten App.

## Erledigt (2026-07-15)

- ~~**3-Schicht-Wind/Atmosphäre-Kopplung**~~ (User-Report: "wind funktioniert
  noch nicht") — Root Cause: die bisherige CFD war EINE horizontale 2D-
  Schicht ohne echte Advektion (Wind trug weder Temperatur noch Feuchtigkeit
  von Nachbarzellen mit), keine Latentwärme, keine adiabatische Abkühlung.
  Kompletter Umbau in `core/weather_generator.py`: neue `AtmosphereLayers`-
  Klasse (Boden 0-150m / Mittel 150-1200m / Hoch >1200m AGL, terrain-
  folgend - absolute Höhe jeder Schicht = heightmap(x,y) + Referenzhöhe),
  neue `_run_coupled_atmosphere_simulation()` mit echtem Pro-Zeitschritt-Loop
  (Semi-Lagrange-Rückwärts-Advektion via `scipy.ndimage.map_coordinates` für
  u/v/potentielle-Temperatur/Feuchte - inspiriert von niels747/2D-Weather-
  Sandbox, aber an unsere Top-Down-3-Schicht-Geometrie angepasst statt deren
  2D-Vertikalschnitt; Druckgradient+Terrain-Kopplung nur auf Boden/gedämpft
  Mittel; thermischer vertikaler Austausch zwischen Schichten als
  massenerhaltende symmetrische Relaxation; Latentwärme aus Kondensation/
  Verdunstung mit der BESTEHENDEN Magnus-Skala, kein zweites Feuchte-
  Einheitensystem; 6-Richtungs-Massenbilanz durch Erweiterung von
  `_apply_continuity_correction` um einen vertikalen Fluss-Term). DAG-
  Integration: der gekoppelte Loop läuft in `_calc_temperature` (nicht in
  `_calc_wind`), weil `weather.temperature` im Calculator-Graph keine
  Abhängigkeit zu `weather.wind`/`weather.humidity` hat und dadurch
  topologisch garantiert zuerst fertig wird - ein Loop in `weather.wind`
  hätte ein Race-Window geöffnet, da `biome.super_override` nur von
  `weather.temperature` abhängt. Ergebnisse werden explizit unter allen 4
  Calculator-IDs gespeichert; `_calc_wind`/`_calc_humidity`/
  `_calc_precipitation` sind jetzt dünne Pass-Throughs mit der alten
  Einzelschicht-Logik als Fallback, falls der gekoppelte Loop fehlschlägt.
  GROUND-Schicht bleibt als `temp_map`/`wind_map`/`humid_map`/`precip_map`
  rückwärtskompatibel für 2D/3D-Anzeige, `water.evaporation` und alle
  Biome-Knoten - volle 3-Schicht-Daten liegen zusätzlich unter neuen
  `*_layers`/`*_layers_monthly`-Keys. Vorstufe: `_apply_wind_diffusion`/
  `_apply_continuity_correction`/`_transport_moisture_simple` von Python-
  Doppelschleifen auf `scipy.ndimage`/vektorisiertes NumPy umgestellt
  (Pflicht, da der neue Loop diese Funktionen jetzt pro Schicht UND pro
  Zeitschritt statt nur einmal pro Monat aufruft). GPU-Port bewusst NICHT
  Teil dieser Runde (CPU zuerst, wie beim OpenSimplex-Port). Smoke-getestet:
  Vektorisierungs-Parität (alt vs. neu), 3-Schicht-Plausibilität auf
  synthetischem Berg (Höhen-Abkühlung, Luv/Lee-Niederschlags-Asymmetrie, kein
  NaN/Inf, HIGH kälter als GROUND), DAG-Vertrag (alle 4 Calculator-IDs liefern
  korrekte Daten, Pass-Through ist echtes No-Op), voller Fallback-Ketten-Test
  (erzwungener Fehler im Loop → alle 4 Knoten fallen sauber auf Einzelschicht
  zurück). Performance (CPU, LOD1-3): 0.9s/2.9s/6.1s - höhere LODs (256+)
  werden spürbar langsamer, akzeptiert als Phase-1-Grenze vor dem GPU-Port.

- ~~**Wasser/Flut-Mechanik: Nutzer-Verständnisfrage + Kalibrierung**~~ —
  `flow_accumulation` war bereits die vom Nutzer gewünschte "übertriebene
  Flut" (treibt Erosion/Sedimentation ungefiltert), `water_biomes_map`
  bereits das "Zurückdrehen auf Hauptflüsse" (Schwellen-Klassifikation) -
  per Diagramm erklärt, keine neue Mechanik nötig. Drei Kalibrierungs-Fixes:
  (1) `LAKE_VOLUME_THRESHOLD` (`gui/config/value_default.py`) - `total_volume`
  in `_classify_lake_basins` ist "Meter-Pixel", kein m³; ein realer Smoke-
  Log zeigte bei altem `default=0.1` GAR KEINE Seen (`water_biomes_map` max
  Klasse 3). `min`/`default`/`max` gesenkt (0.001/0.02/0.3). (2) See-Pixel
  visuell von Fluss-Pixeln abgesetzt: neue `set_water_biomes_reference()`-
  Hook in `map_display_2d.py` (analog `set_contour_reference_heightmap`),
  `_render_water_map` overlayt Klasse-4-Pixel in abgesetzter Farbe;
  `water_tab.py` hinterlegt die Referenz vor jedem `water_map`-Display-
  Update. (3) Flussgeschwindigkeit fließt jetzt in die Breiten-Malung ein
  (`paint_channel_width`, neuer optionaler `flow_speed`-Parameter) - schnell
  fließende Reaches malen sich schmaler (`effective_width_factor`, geklemmt
  0.3-1.0 relativ zum Median über alle Fluss-Pixel dieser Karte), OHNE die
  Klassifikations-Schwellen selbst zu ändern (bleibt akkumulationsbasiert).
  **Zweite, ungeplante Neukalibrierung nötig**: der neue gekoppelte 3-Schicht-
  Atmosphäre-Loop (siehe oben) verschob `precip_map`s Größenordnung massiv
  (Mittel vorher ~0.08, jetzt ~13 gH2O/m²) - `RAIN_THRESHOLD`/
  `STREAM_THRESHOLD` lagen dadurch komplett unter jedem real vorkommenden
  Wert (100% der Karte zählte als Regen-Quelle, ~94% als River/Grand River,
  0% "kein Wasser"). Beide neu kalibriert (`RAIN_THRESHOLD` 0.5/3.0/20.0,
  `STREAM_THRESHOLD` 5.0/35.0/150.0 min/default/max) und per End-to-End-
  Smoke-Test (echter Weather-Output → Water-Pipeline) auf eine plausible,
  absteigende Verteilung verifiziert (58.7% kein Wasser, 27.1% Creek, 14.2%
  River). Volle visuelle Bestätigung (See-/Fluss-Farbunterscheidung, Slider-
  Gefühl in der laufenden App) braucht wie immer den Nutzer-Test.

- ~~**3-Schicht-Wetter: drei Live-Test-Befunde (Niederschlag/Feuchte/Wind)**~~
  (User-Report nach dem ersten Live-Test des 3-Schicht-Umbaus) — alle drei
  per Diagnose-Skript gegen den echten `WeatherSystemGenerator`-Output
  empirisch bestätigt (nicht nur vermutet), dann behoben:
  (1) **"Niederschlag konstant bei 5mm"** — kein Physik-Bug, sondern eine
  liegengebliebene Farbskalen-Kalibrierung: `layer_ranges["precip_map"]`
  stand noch auf `vmax=5.0` aus der Zeit vor dem 3-Schicht-Umbau; der neue
  gekoppelte Loop akkumuliert Kondensation über den ganzen Zeitschritt-Loop
  statt aus einem Einzelschuss-Snapshot, precip_map liegt jetzt empirisch bei
  min~5-7/max~24-34/mean~13-16 - 100% der Pixel clippten auf dieselbe
  Voll-Sättigungsfarbe. `vmax` auf 35.0 angehoben. (2) **Feuchte-Muster über
  die Monate fast identisch** — paarweise räumliche Korrelation lag bei
  0.72-0.96 (Form blieb stabil, nur das Niveau verschob sich). Ursache:
  `_generate_atmospheric_noise()` (treibt die initiale Feuchte-Quelle über
  die Bodentemperatur) nahm keinen `month_index` entgegen, anders als
  `_generate_pressure_noise()` (letzte Runde bereits gefixt) - jetzt
  identisches Koordinaten-Offset-Muster ergänzt, plus eigene, höhere
  Schrittzahl für den 3-Schicht-Loop (`_get_atmosphere_loop_steps()`, 8-60
  statt der für die alte Einzelschicht-CFD kalibrierten 3-25, EIGENSTÄNDIG
  von `_get_cfd_iterations()`, das für die Fallback-Pfade unverändert
  bleibt) - Korrelation sank auf 0.61-0.90, spürbare aber nur teilweise
  Verbesserung (Terrain-dominiertes Gleichgewicht bleibt ein starker
  Einfluss, ggf. Folge-Runde nötig). (3) **Wind zu glatt, keine lokale
  Turbulenz** — mittlere Richtungsänderung zwischen Nachbarpixeln lag nur bei
  ~7.8°, `_apply_wind_diffusion()`+`_apply_continuity_correction()` glätten
  aktiv genau die kleinräumige Rotationsstruktur weg, die Turbulenz ausmacht.
  Neue `_apply_vorticity_confinement()` (Fedkiw/Stam-Standardtechnik, Formel
  1:1 aus dem vom Nutzer verlinkten `niels747/2D-Weather-Sandbox`
  (`curlShader.frag`+`vorticityShader.frag`) übernommen) - läuft NACH
  Diffusion, VOR Kontinuitätskorrektur, pro Schicht gedämpft (GROUND am
  stärksten) und mit Rand-Verstärkung (bis 2.5x am Kartenrand, Nutzer-Wunsch)
  über neuen `WEATHER.TURBULENCE_STRENGTH`-Slider (0.0 = exakt altes
  Verhalten, per Spy-Test verifiziert als echtes No-Op). Ergebnis: mittlere
  Richtungsänderung stieg auf ~35-80° je nach Test. **Nutzer-Rückfrage zu
  Lattice-Boltzmann als Alternative diskutiert und verworfen** — würde die
  gerade erst verifizierte 3-Schicht-Engine komplett neu aufbauen, passt
  nicht sauber auf die 3-gestapelte-2D-Schichten-Geometrie, lohnt sich
  höchstens später zusammen mit dem GPU-Port. **Während der Verifikation
  selbst gefundene und behobene Instabilität**: die ungekappte Vorticity-
  Confinement-Kraft eskalierte über viele Zeitschritte (Rückkopplung: mehr
  Curl → mehr Kraft → mehr Curl) - bei LOD3/18 Schritten liefen
  Windgeschwindigkeiten bis 247 m/s auf ("Unrealistic wind speeds
  detected"-Warnung erstmals wirklich ausgelöst, nicht nur theoretisch
  möglich wie bei der Diagnose ganz am Anfang dieser Session). Kraft-Betrag
  pro Aufruf auf denselben Größenordnungsbereich wie die übrigen
  Pro-Iterations-Terme gekappt (`FORCE_CAP=1.0`) - Turbulenz baut sich
  seitdem graduell auf statt zu explodieren, Warnung verschwunden, lokale
  Richtungsvarianz blieb dabei erhalten. Performance-Kosten der höheren
  Schrittzahl vom Nutzer explizit akzeptiert (LOD1/2/3 jetzt ~1.7s/13s/28s
  CPU statt vorher ~0.9s/2.9s/6.1s) - GPU-Port bleibt der Weg, das später zu
  lösen. Volle Smoke-Test-Regression (alle Tests der vorherigen Runde plus
  neue Diagnose-Vergleiche) bestanden.

- ~~**Soil Moisture überall 100%, Sedimentation überall 0**~~ (User-Report:
  "so ist alles lake/river biome") — beide Male dieselbe Grundursache:
  Formeln, die für die ALTE (viel kleinere) precip_map/flow_speed-
  Größenordnung kalibriert waren, brachen bei den seit dem 3-Schicht-
  Atmosphäre-Umbau deutlich größeren Werten strukturell zusammen, nicht nur
  graduell. Beide per Diagnose-Skript gegen den echten Weather→Water-
  Pipeline-Output empirisch verifiziert (nicht nur vermutet).
  **(1) Soil Moisture**: `SoilMoistureCalculator`s "kapillare" Gaussian-
  Diffusion (core/water_generator.py) hatte einen hart kodierten
  `sigma=2.0` - bei einem nach der precip_map-Neukalibrierung dichteren
  Fluss-Netzwerk (Creek+River-Klassifikation deckt empirisch ~41% der Karte
  ab) überlappte dieser Radius zwischen benachbarten Wasser-Quellen so
  stark, dass praktisch die GESAMTE Karte auf hohe Feuchte kam (empirisch:
  mean=63.8%, nur 1.7% der Pixel unter 10%) - der einstellbare
  `DIFFUSION_RADIUS`-Slider (steuert nur die separate "Grundwasser"-
  Komponente) hatte dabei nachweislich GAR KEINEN messbaren Effekt (die
  kapillare Komponente dominierte via `max()` durchgängig). Fix: neuer
  `capillary_sigma`-Parameter (Default 0.5 statt hart 2.0) plus
  `DIFFUSION_RADIUS`-Default von 5.0 auf 2.0 gesenkt (GPU-Dispatch-Parität
  in `shader_manager.py` mitgezogen) - mean sinkt auf 52.3%, 6.9% der Pixel
  bleiben unter 10%, nur noch die tatsächlichen Wasser-Pixel selbst (41%)
  liegen über 90%. **(2) Sedimentation**: `transport_capacity =
  sediment_capacity_factor * flow_speed^2.5` (`ErosionSedimentationSystem.
  _transport_sediment_optimized`) - die `^2.5`-Potenz ließ die Kapazität bei
  den jetzt üblichen, deutlich höheren Manning-Fließgeschwindigkeiten
  (Median ~3 m/s, Ausreißer bis ~30 m/s) praktisch überall jede realistische
  Sediment-Fracht bei weitem übersteigen - Material floss dadurch fast
  ungebremst bis zum Kartenrand oder einer echten Senke durch, statt
  unterwegs abgelagert zu werden (bei den tatsächlichen LOD-Iterationszahlen,
  3-10, reichte das erst recht nicht, siehe `_get_lod_iterations`). Ein
  eigener Diagnose-Umweg über eine zunächst fehlerhafte Test-Fixture (Platz-
  halter-`flow_directions` statt der echten `_calculate_steepest_descent()`-
  Ausgabe) hätte fast zu einer falschen Diagnose geführt - mit den ECHTEN
  Flow-Directions zeigte sich das strukturelle Problem erst klar. Fix:
  `SEDIMENT_CAPACITY_FACTOR`-Default von 0.1 auf 0.001 gesenkt (Bereich
  entsprechend verschoben), verifiziert gegen einen echten Testlauf: nach
  `_distribute_sediment_floodplain` steigt der Anteil ungleich-Null-Pixel
  von ~0.1-4% (praktisch unsichtbar) auf ~34-44%, Maximalwerte bleiben klein
  und plausibel (0.02-0.11). Nebenbei zwei veraltete Fallback-Literale in
  `HydrologySystemGenerator._update_parameters` korrigiert
  (`settling_velocity`-Fallback war noch 0.01 statt des seit einer früheren
  Session-Runde gültigen Defaults 0.08 - ein unabhängiger, vorbestehender
  Bug, beiläufig mitgefunden). Volle Smoke-Test-Regression (alle Tests der
  vorherigen Runden) weiterhin bestanden. Precipitation zeigte in denselben
  Diagnose-Läufen keine neuen Auffälligkeiten über die bereits letzte Runde
  behobene Farbskala hinaus - falls dort noch ein konkretes Problem sichtbar
  ist, braucht es eine genauere Beschreibung des Symptoms für die nächste
  Runde.
