# Generation Pipeline — reale Datenabhängigkeiten

Stand: 2026-07-08. Dieses Dokument beschreibt die **tatsächlichen** Datenabhängigkeiten
zwischen den einzelnen Rechenschritten ("Calculators") der Welt-Generierung, wie sie
im Code (`core/*_generator.py`) tatsächlich implementiert sind — nicht die im Code
verstreuten, teils veralteten/widersprüchlichen Dependency-Listen (siehe
["Bekannte Inkonsistenzen"](#bekannte-inkonsistenzen-im-code) unten).

Es gibt **6 Top-Level-Generatoren** (Terrain, Geology, Weather, Water, Biome, Settlement),
die zusammen **29 einzelne Calculator-Schritte** enthalten. Der aktuelle Orchestrator
(`gui/OldManagers/generation_orchestrator.py`) kennt nur die 6-Knoten-Granularität und
lässt jeden Generator unabhängig durch alle seine LOD-Stufen laufen (siehe
["Bekannte Probleme"](#bekannte-probleme-nicht-nur-dokumentation)) — dieses Dokument ist
die Grundlage für einen künftigen Umbau auf echte Calculator-/LOD-Synchronisation.

## Calculator-Graph

| # | Calculator | Datei:Klasse/Methode | Consumes | Produces | Generator |
|---|---|---|---|---|---|
| 1 | Noise-Generierung | `terrain_generator.py:SimplexNoiseGenerator.generate_noise_grid` | parameters (seed/freq/octaves/...) | roh. Noise-Grid | Terrain |
| 2 | Redistribution | `terrain_generator.py:BaseTerrainGenerator._apply_redistribution` | Noise-Grid, amplitude/redistribute_power | `heightmap` | Terrain |
| 3 | Slope-Berechnung | `terrain_generator.py:SlopeCalculator.calculate_slopes` | `heightmap` | `slopemap` | Terrain |
| 4 | Shadow-Berechnung | `terrain_generator.py:ShadowCalculator.calculate_shadows` | `heightmap` | `shadowmap` | Terrain |
| 5 | Rock-Klassifikation (Basis) | `geology_generator.py:RockTypeClassifier.classify_by_elevation` | `heightmap_combined` | roh. `rock_map` | Geology |
| 6 | Slope-Hardening | `geology_generator.py:RockTypeClassifier.apply_slope_hardening` | `rock_map`(5), `slopemap`(3) | `rock_map` | Geology |
| 7 | Zonen-Blending | `geology_generator.py:RockTypeClassifier.blend_geological_zones` | `rock_map`(6) | `rock_map` | Geology |
| 8 | Tektonische Deformation | `geology_generator.py:TectonicDeformationProcessor` | `rock_map`(7) | `rock_map` | Geology |
| 9 | Mass-Conservation | `geology_generator.py:MassConservationManager.normalize_rock_masses` | `rock_map`(8) | `rock_map` (normiert, R+G+B=255) | Geology |
| 10 | Hardness-Berechnung | `geology_generator.py:HardnessCalculator.calculate_hardness_map` | `rock_map`(9) | `hardness_map` | Geology |
| 11 | Temperatur | `weather_generator.py:TemperatureCalculator` | `heightmap_combined`, `shadowmap`(4) | `temp_map` | Weather |
| 12 | Windfeld | `weather_generator.py:WindFieldSimulator` | `heightmap`, `temp_map`(11), `shadowmap`(4) | `wind_map` | Weather |
| 13 | Luftfeuchtigkeit | `weather_generator.py:AtmosphericMoistureManager` | `heightmap`, `temp_map`(11), `wind_map`(12) | `humid_map` | Weather |
| 14 | Niederschlag | `weather_generator.py:PrecipitationSystem` | `humid_map`(13), `temp_map`(11), `wind_map`(12), `heightmap` | `precip_map` | Weather |
| 15 | Seen-Erkennung | `water_generator.py:LakeDetectionSystem` | `heightmap` | `lake_map` | Water |
| 16 | Flow-Network | `water_generator.py:FlowNetworkBuilder.build_flow_network` | `heightmap`, `precip_map`(14), `lake_map`(15) | `flow_accumulation`, `water_biomes_map` | Water |
| 17 | Steepest-Descent | `water_generator.py:FlowNetworkBuilder._calculate_steepest_descent` | `heightmap` | `flow_directions` | Water |
| 18 | Manning-Flow | `water_generator.py:ManningFlowCalculator` | `flow_accumulation`(16), `slopemap`(3), `heightmap` | `flow_speed`, `cross_section` | Water |
| 19 | Erosion/Sedimentation | `water_generator.py:ErosionSedimentationSystem` | `flow_accumulation`(16), `flow_speed`(18), `flow_directions`(17), `hardness_map`(10) | `erosion_map`, `sedimentation_map` | Water |
| 20 | Bodenfeuchte | `water_generator.py:SoilMoistureCalculator` | `water_biomes_map`(16), `flow_accumulation`(16) | `soil_moist_map` | Water |
| 21 | Verdunstung | `water_generator.py:EvaporationCalculator` | `temp_map`(11), `wind_map`(12), `humid_map`(13), `water_biomes_map`(16) | `evaporation_map` | Water |
| 22 | Terrain-Rückkopplung | `water_generator.py:BiDirectionalTerrainIntegrator.transfer_erosion_data` | `erosion_map`(19), `sedimentation_map`(19) | **nichts — kaputt, siehe unten** | Water→Terrain |
| 23 | Basis-Biome-Klassifikation | `biome_generator.py:BaseBiomeClassifier` | `heightmap`, `temp_map`(11), `precip_map`(14), `soil_moist_map`(20) | `base_biome_map` | Biome |
| 24 | Super-Biome-Override | `biome_generator.py:SuperBiomeOverrideSystem` | `heightmap`, `temp_map`(11), `water_biomes_map`(16), `soil_moist_map`(20) | `super_biome_mask` | Biome |
| 25 | Layer-Integration | `biome_generator.py:_integrate_biome_layers` | `base_biome_map`(23), `super_biome_mask`(24) | `final_biome_map`/`biome_map` | Biome |
| 26 | Supersampling | `biome_generator.py:SupersamplingManager` | `final_biome_map`(25) | `biome_map_super` | Biome |
| 27 | Klima-Klassifikation | `biome_generator.py:_create_climate_classification` | `temp_map`(11), `precip_map`(14) | `climate_classification` | Biome |
| 28 | Terrain-Suitability | `settlement_generator.py:TerrainSuitabilityAnalyzer` | `heightmap`, `slopemap`(3), `water_map`/`water_biomes_map`(16) | `combined_suitability_map` | Settlement |
| 29 | Settlement-Platzierung | `settlement_generator.py:calculate_settlements` | `suitability_map`(28), `heightmap` | `settlement_list` | Settlement |
| 30 | Straßennetz | `settlement_generator.py:PathfindingSystem` | `settlement_list`(29), `slopemap`(3) | `roads` | Settlement |
| 31 | Roadsites | `settlement_generator.py:calculate_roadsites` | `roads`(30) | `roadsite_list` | Settlement |
| 32 | Zivilisations-Einfluss | `settlement_generator.py:CivilizationInfluenceMapper` | `heightmap`, `slopemap`(3), `settlement_list`(29), `roads`(30), `roadsite_list`(31) | `civ_map` | Settlement |
| 33 | Landmarks | `settlement_generator.py:calculate_landmarks` | `civ_map`(32), `heightmap`, `slopemap`(3) | `landmark_list` | Settlement |
| 34 | Plots | `settlement_generator.py:PlotNodeSystem` | `civ_map`(32), `settlement_list`(29), `roads`(30), `heightmap`, `biome_map`(25) | `plot_nodes`, `plots`, `plot_map`, `plot_edges` | Settlement |
| 35 | Stadtgrenze | `settlement_generator.py:CityBoundaryAnalyzer` | `settlement_list`(29), `heightmap`, `slopemap`(3) | `city_mask`, `city_cost_map` | Settlement |
| 36 | Innerstädtische Blöcke | `settlement_generator.py:CityBlockSystem` | `city_mask`(35), `settlement_list`(29), `heightmap` | `street_mask`, `house_parcel_map` | Settlement |
| 37 | Landschafts-Voronoi | `settlement_generator.py:LandscapeVoronoiSystem` | `city_mask`(35), `heightmap`, `slopemap`(3) | `voronoi_seed_positions`, `voronoi_cell_map` | Settlement |
| 38 | Außenverbindungen | `settlement_generator.py:calculate_outer_connections` | `settlement_list`(29), `suitability_map`(28), `slopemap`(3) | `outer_roads` | Settlement |
| 39 | Landmark-Anbindung | `settlement_generator.py:calculate_landmark_roads` | `landmark_list`(33), `roads`(30), `slopemap`(3) | `landmark_roads` | Settlement |

(29 tatsächliche Calculator + der kaputte Rückkopplungsschritt #22 = 30 Zeilen in der Tabelle,
plus 5 neue Settlement-Knoten #35-#39 aus dem Settlement-Rework = 39 aktive Calculators in
`gui/OldManagers/calculator_graph.py`.)

**Settlement-Rework (Ticket #4 in docs/backlog.md), Stand 2026-07-09:** #35 (Stadtgrenze),
#36 (innerstädtische Blöcke) und #37 (Landschafts-Voronoi) hängen nur von `settlement_list`(29)
ab, laufen also parallel zu #30-#33. #37 nutzt einen terrain-cost-gewichteten Multi-Source-Flood
(`_terrain_cost_voronoi()`, CPU-Referenz fürs spätere GPU-JFA - siehe
`shaders/water/jumpFloodLakes.comp` für dasselbe Muster bei Lake-Detection) und einen Warm-Start
über LOD-Stufen hinweg (Seed-Positionen relativ 0..1 gespeichert, siehe
`_calc_landscape_voronoi()`). #36 nutzt denselben Flood (maskiert auf den Stadt-Footprint) für die
Hausparzellen-Zuweisung, aber ein eigenes Minimum-Spanning-Tree-Straßenskelett zwischen
Haus-Ankerpunkten statt Voronoi-Kanten (Häuser richten sich an echten Wegen aus). #30
(Straßennetz) bevorzugt jetzt den Verlauf entlang der Voronoi-Zellgrenzen von #37 (Edge-Bias-
Kostenterm in `PathfindingSystem`, siehe `_voronoi_edge_distance_map()`). #38/#39 sind
deterministische Zusatzverbindungen (Außenrand bzw. Landmarks ans Straßennetz) ohne
Zufallsmechanismus - das dekorative Zufalls-Wegenetz ist bewusst auf Phase 2 verschoben.
Alle 5 neuen Knoten sind CPU-Referenzimplementierungen und per Smoke-Test verifiziert.

**#34 (Plots) überarbeitet, Stand 2026-07-09 (User-Feedback-Runde 3):** Node-Abstoßung ist
jetzt civ-wert-abhängig (hoher Civ-Wert -> kleinerer Mindestabstand -> kleinere Plots), jede
Delaunay-Kante ist über eine neue `PlotEdge`-Struktur adressierbar (`edge_id`, `node_a`,
`node_b`, Länge, Wegintegral-Höhenkosten statt Endpunkt-Differenz). Neuer Schritt
`simulate_plot_traffic()`: jede Plotnode simuliert eine Familie, die einmal zur nächsten
Siedlung läuft (Multi-Source-Dijkstra über den kleinen Plot-Graphen, nicht das Pixel-Grid),
plus Inter-City-Traffic zwischen durch #30 verbundenen Siedlungspaaren - klassifiziert Kanten
ab Traffic-Schwellwert zu "path"/"road". Deshalb hängt #34 jetzt zusätzlich von #30 (`roads`)
ab. Noch offen: das bisherige #34-Ergebnis wird noch nicht mit dem neuen Landschafts-Voronoi
(#37) verschmolzen/eingeschränkt, GPU-Anbindung (Ticket #40) für #35-#37 noch
nicht umgesetzt (CPU-Referenz ist aber bereits so geschrieben, dass sie 1:1 auf den GPU-JFA-
Shader übertragbar ist), und die UI (`settlement_tab.py`) zeigt die neuen Datenstrukturen noch
nicht an.

## Parallelisierbare Geschwister-Paare

Diese Schritte hängen NICHT voneinander ab und könnten parallel laufen, statt (wie aktuell)
strikt sequenziell innerhalb ihres Generators:

- **#3 + #4** (Slope/Shadow, Terrain) — beide brauchen nur `heightmap`
- **#17** parallel zu #16/#18 (Steepest-Descent braucht nur `heightmap`)
- **#20 + #21** (Soil Moisture / Evaporation, Water) — beide brauchen nur Ausgaben von #16/#11-13
- **#23 + #24** (Basis-Klassifikation / Super-Biome-Override, Biome)
- **#27** kann jederzeit parallel zu allem anderen in Biome laufen (braucht nur `temp_map`+`precip_map`)
- **Settlement-Phasen #28-#33** (6 von 7!) brauchen KEIN `biome_map` — nur **#34** (Plots) braucht Biome.
  Das grobe 6-Knoten-Modell blockiert aktuell den GESAMTEN Settlement-Generator bis Biome fertig ist,
  obwohl 6/7 Phasen das gar nicht bräuchten.

## Bekannte Probleme (nicht nur Dokumentation)

1. **`GenerationThread.run()` hatte keinen Dispatch-Fall für Biome/Settlement** — beide
   generierten über den Orchestrator nie etwas (`else`-Branch, `result = None`).
   **Behoben 2026-07-08** in `gui/OldManagers/generation_orchestrator.py`.
2. **`heightmap_combined` ist ein reiner Passthrough zur rohen `heightmap`** —
   `DataLODManager.get_terrain_data_combined()` inkorporiert nie `erosion_map`/`sedimentation_map`,
   trotz des Namens und der Dokumentation in `descriptor.py`. Die Rückkopplungsmethode
   `BiDirectionalTerrainIntegrator.transfer_erosion_data()` ruft `data_lod_manager`-Methoden auf,
   die es gar nicht gibt (`set_terrain_modification_data`, `update_composite_heightmap`,
   `emit_terrain_modification_signal`) — scheitert bei jedem Water-Lauf still (breiter
   `except Exception`). **Noch offen** — geplant als eigener Schritt (echte Erosion/Sedimentation-
   Rückkopplung implementieren).
3. **Das 6-Knoten-`dependency_tree`/`DependencyQueue.dependencies` prüft nur "irgendein LOD
   abgeschlossen", nicht "dasselbe LOD wie der aktuelle Schritt".** Ein Generator, der einmal
   startet, läuft über `execute_next_lod_level()` durch ALLE seine Ziel-LODs, ohne pro Stufe zu
   prüfen, ob abhängige Generatoren dieselbe Stufe schon erreicht haben — nutzt dadurch ggf.
   veraltete (niedrigere) Upstream-Daten. **Noch offen** — geplanter LOD-Lockstep-Umbau.
4. **`base_tab.check_input_dependencies()` übergab die Zahl `1` statt der Parameter-Liste**,
   was zu einem `TypeError` führte, der vom umschließenden `except` verschluckt wurde und immer
   `True` zurückgab — Dependency-Checks liefen faktisch nie. **Behoben 2026-07-08.**
5. **`check_dependencies()` gibt ein `(bool, list)`-Tuple zurück** — ein Tuple ist in Python immer
   truthy, auch `(False, [...])`. `check_input_dependencies()` gab dieses Tuple direkt zurück statt
   nur den bool-Teil, wodurch `if not check_input_dependencies():` nie ausgelöst hätte, selbst nach
   Fix von Punkt 4. **Behoben 2026-07-08** (zusammen mit Punkt 4).
6. **`BiomeTab`/`SettlementTab` nutzten `self.generation_in_progress`/`self.target_lod`, ohne sie
   je zu initialisieren** — `AttributeError` beim ersten `generate()`-Aufruf. **Behoben 2026-07-08.**
7. **`SettlementTab.on_settlement_generation_completed()` las `result_data["settlement_data"]`**,
   ein Key, der im tatsächlich emittierten `result_data`-Dict nie existiert (dort heißt es `"data"`)
   — Statistics/Display-Refresh nach Generation liefen nie. **Behoben 2026-07-08.**
8. **`TerrainSuitabilityAnalyzer`, `PathfindingSystem`, `PlotNodeSystem` (alle in
   `settlement_generator.py`) indizierten LOD-abhängige Settings-Dicts mit alten String-Keys
   (`"LOD64"`/`"LOD128"`/`"LOD256"`/`"FINAL"`), wurden aber vom modernen Orchestrator mit
   numerischen LOD-Levels (`1`, `2`, ...) konstruiert/aufgerufen — `KeyError` bei jedem echten
   Settlement-Lauf. **Behoben 2026-07-08**: alle drei Klassen bucketen jetzt nach der
   tatsächlichen Array-Pixel-Größe (`heightmap.shape[0]`/`slopemap.shape[0]`) statt nach einem
   LOD-Label; funktioniert dadurch sowohl mit numerischem als auch (für die Legacy-Wrapper-Methoden
   in `SettlementGenerator`, die weiterhin `"LOD64"`/`"LOD256"` verwenden) mit altem String-LOD.
   Drei weitere Stellen (`calculate_settlements`/`calculate_roadsites`/`calculate_landmarks`,
   Zeilen ~863/991/1076) haben denselben String-Key-Bug, crashen aber nicht (`dict.get(lod, 1.0)`
   fällt still auf Faktor 1.0 zurück) — **noch offen**, geplant im Rahmen von Punkt 3
   (LOD-Lockstep-Umbau), da eine saubere Lösung sowieso eine echte LOD→Size-Zuordnung braucht.
9. **`DataLODManager.set_settlement_data_complete_lod()` zerlegte `SettlementData` nur in
   `plot_map`/`civ_map`/`settlement_list`/`landmark_list`/`roadsite_list`** — `roads`, `plots`,
   `plot_nodes` und `combined_suitability_map` wurden nie unter ihrem Data-Key gespeichert, obwohl
   `settlement_tab.py` genau diese Keys liest (u.a. für den *Default*-Anzeigemodus "Terrain
   Suitability"). Folge: Settlement-Tab zeigte nach erfolgreicher Generierung trotzdem eine leere
   Karte. **Behoben 2026-07-08** — Keys ergänzt, orientiert an der bereits vorhandenen (aber toten)
   Legacy-Methode `SettlementGenerator._save_to_data_manager()`, die dieselben Keys nutzt.
10. **`calculate_max_lod_for_size()` (`data_lod_manager.py`) rundete via `int(math.log2(...))` immer
    ab** — für jede `map_size`, die keine exakte Zweierpotenz von `MAPSIZEMIN` ist (26 der 32 laut
    UI-Step=32 erlaubten Werte, z.B. 96, 160, 288 — sogar der Beispielwert 288 aus dem eigenen
    Docstring von `LODConfig`), wurde die letzte, ohnehin auf `target_map_size` geklemmte LOD-Stufe
    weggerundet. Die Karte wurde dadurch nie in der tatsächlich eingestellten Auflösung generiert,
    sondern blieb bei der nächstniedrigeren Zweierpotenz stehen (z.B. 96 → nur 64 erreicht). Diese
    Funktion bestimmt den `target_lod` für jeden `request_generation()`-Aufruf ohne explizites
    `target_lod` (`generation_orchestrator.py`, zwei Stellen) — **live im Pfad, kein totes Dokuformat**.
    **Behoben 2026-07-08**: zählt Verdopplungsschritte jetzt direkt per Schleife statt log2/int().
11. **`WeatherGenerator._get_lod_size()` klemmte nicht gegen die tatsächliche Input-Größe** — bei
    `lod_level <= 6` wurde stur `32 * 2^(lod_level-1)` zurückgegeben, ohne gegen `original_size` zu
    minimieren (im Gegensatz zu `water_generator.py`, das das schon richtig macht). Bei
    Nicht-Zweierpotenz-Zielgrößen (z.B. 96) versuchte Weather dadurch, auf 128 hochzuinterpolieren,
    während Terrain korrekt bei 96 geklemmt hatte — Shape-Mismatch zwischen `heightmap` und
    `temp_map`/`wind_map` in allen nachgelagerten Schritten (Biome-Broadcast-Fehler etc.).
    **Behoben 2026-07-08**: `min(32 * 2^(lod_level-1), original_size)`, analog zu `water_generator.py`.
12. **`BaseTerrainGenerator._validate_parameters()` verlangte für `map_size` explizit eine
    Zweierpotenz** (`(x & (x-1)) == 0`), obwohl `gui/config/value_default.py:TERRAIN.MAPSIZE` per
    `step: 32` ausdrücklich jedes Vielfache von 32 als gültige UI-Eingabe vorsieht (z.B. 96, 160,
    192, 288 — 26 der 32 zwischen 32 und 1024 wählbaren Werte). Jeder nicht-Zweierpotenz-`map_size`
    ließ die Terrain-Generierung bei **jedem** LOD-Schritt sofort mit `ValueError` abbrechen; alle
    nachgelagerten Generatoren liefen danach mit veralteten/Fallback-Daten falscher Größe weiter
    (Symptom: Shape-Mismatches wie „(64,64) vs (96,96)" in Biome/Geology). Es gibt keinen
    algorithmischen Grund (kein FFT, kein GPU-Textur-Zwang – Noise ist reines Per-Pixel-OpenSimplex)
    für die Zweierpotenz-Pflicht; sie war schlicht nicht mitgezogen worden, als das LOD-System auf
    beliebige Zielgrößen verallgemeinert wurde. **Nutzerentscheidung 2026-07-08: beliebige
    32er-Schritte sollen erlaubt sein** (nicht die UI auf Zweierpotenzen einschränken). Behoben:
    Validierung auf `x % 32 == 0` gelockert (Instanzmethode + die ungenutzte, aber ebenfalls
    betroffene Modulfunktion `validate_terrain_parameters()`). End-to-End mit `map_size=96` über
    alle 6 Generatoren verifiziert (siehe unten).

## Bekannte Inkonsistenzen im Code

Es gibt **drei verschiedene, sich teils widersprechende** Dependency-Listen, von denen die meisten
gar nicht für die Ausführungs-Steuerung genutzt werden:

- `gui/config/value_default.py:VALIDATION_RULES.DEPENDENCIES` — nur von `biome_tab.py`/
  `settlement_tab.py` gelesen; `geology`/`weather`/`water`/`terrain`-Tabs hardcoden ihre eigenen
  (teils abweichenden) Listen direkt im Tab. `weather`-Eintrag nennt `shademap` statt `shadowmap`
  (Tippfehler/Drift, wird nirgends real referenziert).
- `descriptor.py` — reine Prosa-Dokumentation (keine ausführbaren Imports), beschreibt die
  Pipeline auf 6-Generator-Ebene, plus die (nicht implementierte) Erosion-Feedback-Absicht.
- `gui/OldManagers/data_lod_manager.py:DATA_DEPENDENCY_MATRIX` — 28 von 29 Daten-Keys haben als
  einzige deklarierte Abhängigkeit `["heightmap_combined"]` (bis auf `hardness_map → rock_map`,
  der einzige korrekte Eintrag). Wird nur für Cache-Timestamp/Status-Bookkeeping genutzt, NICHT
  für Ausführungs-Gating.

Die einzige Quelle, die tatsächlich (über LOD-Verfügbarkeit) steuert, ob ein Generator starten darf,
ist das 6-Knoten-`dependency_tree` in `GenerationOrchestrator.__init__()` bzw. das identische
`DependencyQueue.dependencies` — beide auf Generator-, nicht Calculator-Ebene.
