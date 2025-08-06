"""
    Descriptor file:
    Descriptions for all scripts as methods (to find them faster)
"""


def main():
    """
    Path: main.py

    Funktionsweise: Programm-Einstiegspunkt und Application-Setup
    - Initialisiert QApplication mit optimalen Einstellungen
    - Lädt globale Konfiguration und prüft Dependencies
    - Startet MainMenuWindow als zentraler Entry-Point
    - Error-Handling für kritische Startup-Fehler
    - Später: Splash-Screen für lange Initialisierung

    Kommunikationskanäle:
    - Config: gui_default.py für Application-Settings
    - Direkte Instanziierung von MainMenuWindow und informiere navigation_manager.py
    - Dependencies: Prüfung aller Core-Module und GUI-Dependencies
    """


def terrain_generator():
    """
    Path: core/terrain_generator.py

    Funktionsweise: Simplex-Noise basierte Terrain-Generierung mit BaseGenerator-Integration und Validity-System
    - BaseTerrainGenerator erbt von BaseGenerator für einheitliche Generator-Architektur
    - Multi-Scale Noise-Layering für realistische Landschaften
    - Progressive Heightmap-Generierung durch Interpolation und Detail-Noise
    - TerrainData-Container für strukturierte Datenorganisation mit Validity-System und Cache-Invalidation
    - Höhen-Redistribution für natürliche Höhenverteilung
    - Deformation mittels ridge warping
    - LOD-System: LOD64 (64x64, 1 Sonnenwinkel) → LOD128 (128x128, 3 Sonnenwinkel) → LOD256 (256x256, 5 Sonnenwinkel) → FINAL (512x512, 7 Sonnenwinkel)
    - Shadowmap konstant 64x64 für alle LODs, Sonnenwinkel-Anzahl steigt progressiv
    - GPU-Shader-Integration über ShaderManager für Noise-Generation und Shadow-Berechnung
    - Threading-Support für Hintergrund-Berechnung ohne GUI-Blocking mit CPU-Yields

    Parameter Input:
    - map_size, amplitude, octaves, frequency, persistence, lacunarity, redistribute_power, map_seed

    Output:
    - TerrainData-Objekt mit heightmap, slopemap, shadowmap, validity_state und LOD-Metadaten
    - Legacy-Kompatibilität: heightmap array, slopemap 2D array (dz/dx and dz/dy), shadowmap array

    LOD-Definitionen:
    - LOD64: heightmap64x64, slopemap64x64, shadowmap64x64 (1 Sonnenwinkel - Mittag)
    - LOD128: heightmap128x128, slopemap128x128, shadowmap64x64 (3 Sonnenwinkel - Vormittag/Mittag/Nachmittag)
    - LOD256: heightmap256x256, slopemap256x256, shadowmap64x64 (5 Sonnenwinkel + Morgen/Abend)
    - FINAL: heightmapFullSize, slopemapFullSize, shadowmap64x64 (7 Sonnenwinkel + Dämmerung)

    Validity-System:
    - Parameter-Change-Detection: Erkennt signifikante Parameter-Änderungen für Cache-Invalidation
    - LOD-Consistency-Checks: Stellt sicher dass alle LOD-Level mit gleichen Parametern berechnet sind
    - Dependency-Invalidation: Invalidiert nachgelagerte Generatoren bei kritischen Parameter-Änderungen
    - State-Tracking: validity_flags in TerrainData für jeden LOD-Level und Output-Type

    Klassen:
    TerrainData
        Funktionsweise: Container für alle Terrain-Daten mit Validity-System und Cache-Management
        Aufgabe: Speichert Heightmap, Slopemap, Shadowmap mit LOD-Level, Validity-State und Parameter-Hash
        Attribute: heightmap, slopemap, shadowmap, lod_level, actual_size, validity_state, parameter_hash, calculated_sun_angles, parameters
        Validity-Methods: is_valid(), invalidate(), validate_against_parameters(), get_validity_summary()

    BaseTerrainGenerator
        Funktionsweise: Hauptklasse für Simplex-Noise basierte Terrain-Generierung mit einheitlicher Generator-Architektur
        Aufgabe: Koordiniert alle Terrain-Generierungsschritte, verwaltet Parameter, Threading und Validity-Tracking
        Methoden: generate() [Main-Funktion], calculate_heightmap(), calculate_slopes(), calculate_shadows()
        BaseGenerator-Integration: _load_default_parameters(), _get_dependencies(), _execute_generation(), _save_to_data_manager()
        Threading: Hintergrund-Berechnung mit CPU-Yields und Generation-Interrupt-Support
        GPU-Integration: Nutzt ShaderManager für performance-kritische Noise- und Shadow-Operationen

    SimplexNoiseGenerator
        Funktionsweise: Erzeugt OpenSimplex-Noise mit GPU-Shader-Beschleunigung und CPU-Fallback
        Aufgabe: Basis-Noise-Funktionen für alle anderen Module mit LOD-optimierter Batch-Verarbeitung
        Methoden: noise_2d(), multi_octave_noise(), ridge_noise()
        LOD-Optimiert: generate_noise_grid() für Batch-Verarbeitung, interpolate_existing_grid() für LOD-Upgrades, add_detail_noise() für Progressive Verfeinerung
        GPU-Integration: Nutzt ShaderManager.process_noise_generation() für parallele Multi-Octave-Berechnung

    ShadowCalculator
        Funktionsweise: Berechnet Verschattung mit Raycasts für LOD-spezifische Sonnenwinkel mit GPU-Shader-Support
        Aufgabe: Erstellt shadowmap (konstant 64x64) für Weather-System und visuelle Darstellung
        Methoden: calculate_shadows_multi_angle(), calculate_shadows_with_lod(), calculate_shadows_progressive(), raycast_shadow(), combine_shadow_angles()
        LOD-System: get_sun_angles_for_lod() - 1,3,5,7 Sonnenwinkel je nach LOD-Level
        GPU-Integration: Nutzt ShaderManager für GPU-beschleunigte Raycast-Shadow-Berechnung mit CPU-Fallback
        Progressive-Enhancement: Berechnet nur neue Sonnenwinkel bei LOD-Upgrades, kombiniert mit bestehenden Shadows
    """


def geology_generator():
    """
    Path: core/geology_generator.py
    date changed: 06.08.2025

    Funktionsweise: Geologische Schichten und Gesteinstypen mit BaseGenerator-Integration und TerrainData-ähnlicher Struktur
    - GeologyGenerator erbt von BaseGenerator für einheitliche Generator-Architektur
    - Hauptfunktion: generate_geology() koordiniert alle Berechnungsschritte
    - Progressive Berechnung: calculate_elevation_classification() → calculate_slope_hardening() → calculate_geological_zones() → calculate_deformation_effects() → calculate_hardness_distribution()
    - GeologyData-Container für strukturierte Datenorganisation mit Status-System und Cache-Invalidation
    - Mass-Conservation-System: R+G+B=255 für spätere Erosion/Sedimentation
    - LOD-System: LOD64 → LOD128 → LOD256 → FINAL mit progressiver Verfeinerung
    - GPU-Shader-Integration über ShaderManager für geologische Zonen-Berechnung
    - Threading-Support für Hintergrund-Berechnung mit CPU-Yields

    Parameter Input:
    - sedimentary_hardness, igneous_hardness, metamorphic_hardness
    - ridge_warping, bevel_warping, metamorph_foliation, metamorph_folding, igneous_flowing

    data_manager Input:
    - heightmap, slopemap (von terrain_generator)

    Output:
    - rock_map RGB array (R=Sedimentary, G=Igneous, B=Metamorphic mit R+G+B=255)
    - hardness_map array (gewichtete Härte-Verteilung)

    Berechnungsschritte mit Status-Tracking:
    calculate_elevation_classification():
        Status: elevation_classification_valid, elevation_base_distribution
        Aufgabe: Basis-Gesteinsverteilung basierend auf verzerrten Höhen-Noise-Funktionen
        Output: base_rock_distribution (interne 3-Kanal-Map)

    calculate_slope_hardening():
        Status: slope_hardening_valid, hardening_applied
        Aufgabe: Härtere Gesteine in steile Hänge einmischen
        Input: base_rock_distribution, slopemap
        Output: slope_hardened_distribution (modifizierte rock_distribution)

    calculate_geological_zones():
        Status: geological_zones_valid, zone_count, zone_boundaries
        Aufgabe: Geologische Zonen durch Simplex-Funktionen (Sedimentär >0.2, Metamorph >0.6, Igneous >0.8)
        Input: slope_hardened_distribution
        Output: zoned_rock_distribution (mit geologischen Zonen)

    calculate_deformation_effects():
        Status: deformation_valid, ridge_strength, bevel_strength, foliation_intensity
        Aufgabe: Ridge/Bevel Warping, Metamorph Foliation/Folding, Igneous Flowing
        Input: zoned_rock_distribution, deformation_parameters
        Output: deformed_rock_distribution (finale rock_distribution)

    calculate_hardness_distribution():
        Status: hardness_valid, hardness_range, average_hardness
        Aufgabe: Gewichtete Härte-Map aus Gesteinstyp-Verhältnissen und Härte-Parametern
        Input: final_rock_map, hardness_parameters
        Output: hardness_map (finale Härte-Verteilung)

    GeologyData-Container:
        Funktionsweise: Container für alle Geology-Daten mit Status-System und LOD-Management
        Aufgabe: Speichert alle internen und externen Maps mit Validity-State und Parameter-Hash
        Attribute: rock_map, hardness_map, base_rock_distribution, geological_zones, deformation_effects
        Status-Attribute: elevation_classification_valid, slope_hardening_valid, geological_zones_valid, deformation_valid, hardness_valid
        LOD-Tracking: lod_level, actual_size, validity_state, parameter_hash
        Methods: is_step_valid(step), invalidate_step(step), get_step_status(step)

    BaseGenerator-Integration:
    - _load_default_parameters(): Lädt GEOLOGY-Parameter aus value_default.py
    - _get_dependencies(): Holt heightmap/slopemap aus DataManager mit Validation
    - _execute_generation(): Führt generate_geology() mit Progress-Updates für jeden calculate-Schritt aus
    - _save_to_data_manager(): Speichert GeologyData-Objekt automatisch mit Status-Tracking
    """

def settlement_generator():
    """
    Path: core/settlement_generator.py

    Funktionsweise: Intelligente Settlement-Platzierung mit BaseGenerator-Integration und LOD-System
    - SettlementGenerator erbt von BaseGenerator für einheitliche Generator-Architektur
    - Hauptfunktion: generate() koordiniert alle Settlement-Berechnungsschritte
    - Progressive Berechnung: calculate_terrain_suitability() → calculate_settlements() → calculate_road_network() → calculate_roadsites() → calculate_civilization_mapping() → calculate_landmarks() → calculate_plots()
    - SettlementData-Container für strukturierte Datenorganisation mit Status-System und Cache-Invalidation
    - BaseGenerator-Integration mit einheitlicher API und LOD-System
    - Terrain-Suitability Analysis (Steigung, Höhe, Wasser-Nähe)
    - Locations:
        Settlements: Städte oder Dörfer die an bestimmten Orten vorkommen können (Täler, flache Hügel). Settlements verringern die Terrainverformung in der Nähe etwas. Je nach Radius (Siedlungsgröße) ist der Einfluss auf die Umgebung größer/kleiner. Die Form der Stadt soll zB Linsenförmig sein und über die Slopemap erzeugt werden. Zwischen Settlements gibt es einen Minimalabstand je nach map_size und Anzahl von Settlements. Innerhalb der Stadtgrenzen ist civ_map = 1, außerhalb nimmt der Einfluss ab.
        Roads: Nachdem Settlements entstanden sind werden die ersten Wege zwischen den Ortschaften geplottet. Dazu soll der Weg des geringsten Widerstands gefunden werden (Pathfinding via slopemap-cost). Danach werden die Straßen etwas gebogen über sanfte Splineinterpolation zwischen zB jedem 3.Waypoint. Erzeugen sehr geringen Einfluss entlang der Wege (z.B. 0.3).
        Roadsites: z.B. Taverne, Handelsposten, Wegschrein, Zollhaus, Galgenplatz, Markt, besondere Industrie. Entstehen in einem Bereich von 30%-70% Weglänge zwischen Settlements entlang von Roads. Der civ_map-Einfluss ist wesentlich geringer als der von Städten.
        Landmarks: z.B. Burgen, Kloster, mystische Stätte etc. entstehen in Regionen mit einem civ_map value < thresholds (landmark_wilderness). Erzeugen einen ähnlich geringen Einfluss wie Roadsites. Außerdem werden beide nur in niedrigeren Höhen und Slopes generiert.
        Wilderness: Bereiche unterhalb eines civ_map-Werts unterhalb von 0.2 werden genullt und als Wilderness deklariert. Hier spawnen keine Plotnodes. Hier sollen in der  späteren Spielentwicklung Questevents stattfinden.
        civ_map-Logik: civ_map wird mit 0.0 initialisiert. Jeder Quellpunkt trägt akkumulativ zum civ-Wert bei. Einflussverteilung um Quellpunkt über radialen Decay-Kernel (z.B. Gauß, linear fallend oder benutzerdefinierte Kurve). Decay ist stärker an Hanglagen, so dass Zivilisation nicht auf Berge reicht. Decayradius und Initialwert abhängig von Location-Typ: Stadt-Grenzpunkte starten bei 0.8 (innerhalb der Stadt ist 1.0), Roadwaypoints addieren 0.2 bis max. 0.5, Roadsite/Landmarks 0.4. Optional bei sehr hohen Berechnungzeiten kann die Einflussverteilung mit GPU-Shadermasken erfolgen.
        Plotnodes: Es wird eine feste Anzahl an Plotnodes generiert (plotnodes-parameter). Gleichmäßige Verteilung auf alle Bereiche außerhalb von Städten und Wilderness. Die Plotnodes verbinden sich mit mit Nachbarnodes über Delaunay-Triangulation. Dann verbinden sich die Delaunay-Dreiecke mit benachbarten Dreiecken zu Grundstücken. Die Plotnode-Civwerte werden zusammengerechnet und wenn sie einen Wert (plot-size-parameter) überschreiten ist die Größe erreicht. So werden Grundstücke in Region mit hohem Civ-wert kleiner. Über Abstoßungslogik können die Nodes "physisch" umarrangiert werden. Kanten mit geringem Winkel sollen sich glätten und die Zwischenpunkte können verschwinden. Sehr spitze Winkel lockern sich ebenso. Plotnode-Eigenschaften:
            node_id, node_location, connector_id (list of nodes), connector_distance (x,y entfernung), connector_elevation (akkumulierter höhenunterschied zu connector), connector_movecost (movecost abhängig von biomes)
        Plots: Plots bestehend aus Plotnodes haben folgende Eigenschaften:
            biome_amount: akkumulierte Menge eines jeden Bioms in den Grenzen des Plots
            resource_amount: später im Spiel sich verändernde Menge an natürlichen Rohstoffen.
            plot_area: Größe des Plots
            plot_distance: Anzahl der Nodepunkte*Distance Entfernung zu

    Parameter Input (aus value_default.py SETTLEMENT):
    - settlements, landmarks, roadsites, plotnodes: number of each type
    - civ_influence_decay: Influence around Locationtypes decays of distance
    - terrain_factor_villages: terrain influence on settlement suitability
    - road_slope_to_distance_ratio: rather short roads or steep roads
    - landmark_wilderness: wilderness area size by changing cutoff-threshold
    - plotsize: how much accumulated civ-value to form plot

    data_manager Input:
    - map_seed (Globaler Karten-Seed für reproduzierbare Settlement-Platzierung)
    - heightmap (2D-Array in meter Altitude) - REQUIRED
    - slopemap (2D-Array in m/m mit dz/dx, dz/dy) - REQUIRED
    - water_map (2D-Array mit Wasser-Klassifikation) - REQUIRED
    - biome_map (2D-Array mit Biom-Indices) - OPTIONAL (Fallback: Höhen-basiert)

    Output:
    - settlement_list (List[Location] - Alle Settlements)
    - landmark_list (List[Location] - Alle Landmarks)
    - roadsite_list (List[Location] - Alle Roadsites)
    - plot_map (2D-Array mit Plot-IDs)
    - civ_map (2D-Array mit Zivilisations-Einfluss)

    Berechnungsschritte mit Status-Tracking:
    calculate_terrain_suitability():
        Status: terrain_suitability_valid, analysis_complete, slope_analyzed, water_proximity_calculated, elevation_fitness_evaluated
        Aufgabe: Terrain-Suitability Analysis für optimale Settlement-Platzierung basierend auf Steigung, Höhe, Wasser-Nähe
        Input: heightmap, slopemap, water_map, terrain_factor_villages
        Output: combined_suitability_map (finale Settlement-Eignung-Map)

    calculate_settlements():
        Status: settlements_valid, placement_complete, settlement_count, suitability_threshold_applied
        Aufgabe: Platziert Settlements basierend auf Terrain-Suitability mit Mindestabstand und Größen-Variation
        Input: combined_suitability_map, settlements_parameter
        Output: settlement_list (List[Location] mit allen Settlements)

    calculate_road_network():
        Status: road_network_valid, pathfinding_complete, spline_smoothing_applied, total_road_length
        Aufgabe: Erstellt Minimum Spanning Tree zwischen Settlements mit A*-Pathfinding und Spline-Glättung
        Input: settlement_list, slopemap, road_slope_to_distance_ratio
        Output: road_network (List[List[Tuple]] - alle Straßenverläufe)

    calculate_roadsites():
        Status: roadsites_valid, placement_complete, roadsite_count, distribution_along_roads
        Aufgabe: Platziert Roadsites (Tavern/Trading Post/etc.) entlang von Roads zwischen 30-70% Weglänge
        Input: road_network, roadsites_parameter
        Output: roadsite_list (List[Location] mit allen Roadsites)

    calculate_civilization_mapping():
        Status: civilization_mapping_valid, influence_applied, decay_calculated, wilderness_defined
        Aufgabe: Erstellt civ_map durch radialen Decay von Settlement/Road/Roadsite-Punkten mit Slope-Modifikation
        Input: settlement_list, road_network, roadsite_list, slopemap, civ_influence_decay
        Output: civ_map (2D-Array mit Zivilisations-Einfluss, Wilderness < 0.2 → 0.0)

    calculate_landmarks():
        Status: landmarks_valid, wilderness_placement_complete, landmark_count, elevation_constraints_applied
        Aufgabe: Platziert Landmarks (Castle/Monastery/etc.) in Wilderness-Bereichen mit niedrigem civ_map-Wert
        Input: civ_map, heightmap, slopemap, landmarks_parameter, landmark_wilderness
        Output: landmark_list (List[Location] mit allen Landmarks)

    calculate_plots():
        Status: plots_valid, plotnodes_generated, delaunay_triangulation_complete, plot_merging_complete, node_optimization_applied
        Aufgabe: Generiert PlotNode-System mit Delaunay-Triangulation und fusioniert zu Plots basierend auf akkumuliertem Civ-Wert
        Input: civ_map, settlement_list, heightmap, biome_map, plotnodes_parameter, plotsize
        Output: plot_nodes (List[PlotNode]), plots (List[Plot]), plot_map (2D-Array mit Plot-IDs)

    SettlementData-Container:
        Funktionsweise: Container für alle Settlement-Daten mit Status-System und LOD-Management
        Aufgabe: Speichert alle internen und externen Maps mit Validity-State und Parameter-Hash
        Attribute: settlement_list, landmark_list, roadsite_list, plot_map, civ_map, plot_nodes, plots, roads, combined_suitability_map
        Status-Attribute: terrain_suitability_valid, settlements_valid, road_network_valid, roadsites_valid, civilization_mapping_valid, landmarks_valid, plots_valid
        LOD-Tracking: lod_level, actual_size, validity_state, parameter_hash
        Methods: is_step_valid(step), invalidate_step(step), get_step_status(step)

    Klassen:
    SettlementGenerator (BaseGenerator)
        Funktionsweise: Hauptklasse für intelligente Settlement-Platzierung mit BaseGenerator-API und LOD-System
        Aufgabe: Koordiniert alle Settlement-Aspekte und erstellt civ_map mit Progress-Updates
        Methoden: generate(), _execute_generation(), _load_default_parameters(), _get_dependencies()
        BaseGenerator-Integration: _load_default_parameters(), _get_dependencies(), _execute_generation(), _save_to_data_manager()

    TerrainSuitabilityAnalyzer
        Funktionsweise: Analysiert Terrain-Eignung für Settlements basierend auf Steigung, Höhe, Wasser-Nähe
        Aufgabe: Erstellt Suitability-Map für optimale Settlement-Platzierung
        Methoden: analyze_slope_suitability(), calculate_water_proximity(), evaluate_elevation_fitness()

    PathfindingSystem
        Funktionsweise: Findet Wege geringsten Widerstands zwischen Settlements für Straßen
        Aufgabe: Erstellt realistische Straßenverbindungen mit Spline-Interpolation und LOD-Optimierung
        Methoden: find_least_resistance_path(), apply_spline_smoothing(), calculate_movement_cost()

    CivilizationInfluenceMapper
        Funktionsweise: Berechnet civ_map durch radialen Decay von Settlement/Road/Landmark-Punkten
        Aufgabe: Erstellt realistische Zivilisations-Verteilung mit Decay-Kernels
        Methoden: apply_settlement_influence(), calculate_road_influence(), apply_decay_kernel()

    PlotNodeSystem
        Funktionsweise: Generiert Plotnodes mit Delaunay-Triangulation und Grundstücks-Bildung
        Aufgabe: Erstellt Grundstücks-System für späteres Gameplay mit LOD-abhängiger Dichte
        Methoden: generate_plot_nodes(), create_delaunay_triangulation(), merge_to_plots(), optimize_node_positions()

    BaseGenerator-Integration:
    - _load_default_parameters(): Lädt SETTLEMENT-Parameter aus value_default.py
    - _get_dependencies(): Holt heightmap/slopemap/water_map aus DataManager mit intelligenten Fallback-Werten
    - _execute_generation(): Führt generate() mit Progress-Updates für jeden calculate-Schritt aus
    - _save_to_data_manager(): Speichert SettlementData-Objekt automatisch mit Status-Tracking
    """

def weather_generator():
    """
    Path: core/weather_generator.py
    date changed: 06.08.2025

    Funktionsweise: Dynamisches Wetter- und Feuchtigkeitssystem mit BaseGenerator-Integration und TerrainData-ähnlicher Struktur
    - WeatherSystemGenerator erbt von BaseGenerator für einheitliche Generator-Architektur
    - Hauptfunktion: generate_weather() koordiniert alle atmosphärischen Berechnungsschritte
    - Progressive Berechnung: calculate_temperature() → calculate_pressure_gradients() → calculate_wind_field() → calculate_moisture_transport() → calculate_precipitation()
    - WeatherData-Container für strukturierte Datenorganisation mit Status-System und Cache-Invalidation
    - Berg-Wind-Simulation mit 8 GPU-Shadern für realistische orographische Effekte
    - LOD-System mit Array-Interpolation für konsistente Größen-Synchronisation
    - Threading-Support für Hintergrund-Berechnung mit CPU-Yields

    Parameter Input:
    - air_temp_entry, solar_power, altitude_cooling
    - thermic_effect, wind_speed_factor, terrain_factor

    data_manager Input:
    - heightmap, shadowmap (terrain), soil_moist_map (optional, water)

    Output:
    - wind_map (Windvektoren in m/s), temp_map (Lufttemperatur in °C)
    - precip_map (Niederschlag in gH2O/m²), humid_map (Luftfeuchtigkeit in gH2O/m³)

    Berechnungsschritte mit Status-Tracking:
    calculate_temperature():
        Status: temperature_valid, temp_range, altitude_effect_applied, solar_effect_applied
        Aufgabe: Lufttemperatur aus Altitude-Cooling, Solar-Heating, Latitude-Gradient + Noise-Variation
        Input: heightmap, shadowmap, noise_variation
        Output: temp_map (Basis-Temperaturfeld)

    calculate_pressure_gradients():
        Status: pressure_valid, gradient_strength, boundary_noise_applied
        Aufgabe: Druckgradienten von West nach Ost mit Noise-Modulation für Turbulenz
        Input: map_shape, noise_generator
        Output: pressure_field (Basis für Wind-Berechnung)

    calculate_wind_field():
        Status: wind_field_valid, terrain_deflection_applied, thermal_effects_applied
        Aufgabe: Windfeld durch Druckgradienten + Terrain-Deflection + Thermal-Konvektion
        Input: pressure_field, slopemap, heightmap, temp_map, shadowmap
        Output: wind_map (finale Windvektoren mit allen Effekten)

    calculate_moisture_transport():
        Status: moisture_valid, evaporation_calculated, transport_applied, diffusion_applied
        Aufgabe: Luftfeuchtigkeit durch Evaporation + Advektion + Diffusion
        Input: soil_moist_map, temp_map, wind_map
        Output: humid_map (Luftfeuchtigkeit-Verteilung)

    calculate_precipitation():
        Status: precipitation_valid, orographic_calculated, condensation_calculated
        Aufgabe: Niederschlag durch orographische Effekte + Kondensation bei Übersättigung
        Input: humid_map, temp_map, wind_map, heightmap, slopemap
        Output: precip_map (finale Niederschlags-Verteilung)

    WeatherData-Container:
        Funktionsweise: Container für alle Weather-Daten mit Status-System und LOD-Management
        Aufgabe: Speichert alle internen und externen Maps mit Validity-State und Parameter-Hash
        Attribute: wind_map, temp_map, precip_map, humid_map, pressure_field, evaporation_data
        Status-Attribute: temperature_valid, pressure_valid, wind_field_valid, moisture_valid, precipitation_valid
        LOD-Tracking: lod_level, actual_size, validity_state, parameter_hash
        Methods: is_step_valid(step), invalidate_step(step), get_step_status(step)

    BaseGenerator-Integration:
    - _load_default_parameters(): Lädt WEATHER-Parameter aus value_default.py
    - _get_dependencies(): Holt heightmap/shadowmap, erstellt Fallback für soil_moist_map
    - _execute_generation(): Führt generate_weather() mit Progress-Updates für jeden calculate-Schritt aus
    - _save_to_data_manager(): Speichert WeatherData-Objekt automatisch mit Status-Tracking
    """

def water_generator():
    """
    Path: core/water_generator.py
    date changed: 06.08.2025

    Funktionsweise: Dynamisches Hydrologiesystem mit BaseGenerator-Integration und TerrainData-ähnlicher Struktur
    - HydrologySystemGenerator erbt von BaseGenerator für einheitliche Generator-Architektur
    - Hauptfunktion: generate_hydrology() koordiniert alle hydrologischen Berechnungsschritte
    - Progressive Berechnung: calculate_lake_detection() → calculate_flow_network() → calculate_manning_flow() → calculate_erosion_sedimentation() → calculate_soil_moisture() → calculate_evaporation()
    - WaterData-Container für strukturierte Datenorganisation mit Status-System und Cache-Invalidation
    - Komplexeste Dependency-Resolution: 8 Input-Maps von terrain/geology/weather
    - LOD-System mit performance-optimierten Algorithmen (reduzierte Iterationen bei niedrigen LODs)
    - GPU-Shader-Integration für alle 8 hydrologischen Shader
    - Threading-Support für Hintergrund-Berechnung mit CPU-Yields

    Parameter Input:
    - lake_volume_threshold, rain_threshold, manning_coefficient
    - erosion_strength, sediment_capacity_factor, evaporation_base_rate
    - diffusion_radius, settling_velocity

    data_manager Input:
    - heightmap, slopemap, hardness_map, rock_map (terrain/geology)
    - precip_map, temp_map, wind_map, humid_map (weather)

    Output:
    - water_map, flow_map, flow_speed, cross_section, soil_moist_map
    - erosion_map, sedimentation_map, rock_map_updated, evaporation_map
    - ocean_outflow, water_biomes_map

    Berechnungsschritte mit Status-Tracking:
    calculate_lake_detection():
        Status: lake_detection_valid, lakes_found, lake_volume_total, jump_flooding_iterations
        Aufgabe: Jump Flooding Algorithm für parallele Senken-Identifikation und Lake-Basin-Classification
        Input: heightmap, lake_volume_threshold
        Output: lake_map, valid_lakes (See-Standorte und Einzugsgebiete)

    calculate_flow_network():
        Status: flow_network_valid, flow_convergence_achieved, upstream_accumulation_stable
        Aufgabe: Steepest Descent Flow-Directions + Upstream-Akkumulation für Flussnetzwerk-Aufbau
        Input: heightmap, precip_map, lake_map, rain_threshold
        Output: flow_directions, flow_accumulation, water_biomes_map (Creek/River/Grand River/Lake)

    calculate_manning_flow():
        Status: manning_flow_valid, channel_geometry_optimized, hydraulic_radius_calculated
        Aufgabe: Manning-Gleichung mit adaptiven Querschnitten für realistische Fließgeschwindigkeiten
        Input: flow_accumulation, slopemap, heightmap, manning_coefficient
        Output: flow_speed, cross_section (optimierte Kanal-Geometrie)

    calculate_erosion_sedimentation():
        Status: erosion_valid, sedimentation_valid, mass_conservation_maintained, rock_transport_stable
        Aufgabe: Stream Power Erosion + Hjulström-Sediment-Transport + Massenerhaltung
        Input: flow_accumulation, flow_speed, slopemap, hardness_map, rock_map
        Output: erosion_map, sedimentation_map, rock_map_updated (mit R+G+B=255 Erhaltung)

    calculate_soil_moisture():
        Status: soil_moisture_valid, gaussian_diffusion_applied, groundwater_effects_calculated
        Aufgabe: Gaussian-Diffusion von Gewässern + Groundwater-Effects für Biome-System
        Input: water_biomes_map, flow_accumulation, heightmap, diffusion_radius
        Output: soil_moist_map (Bodenfeuchtigkeit für Weather-System)

    calculate_evaporation():
        Status: evaporation_valid, atmospheric_effects_applied, wind_enhancement_calculated
        Aufgabe: Atmosphärische Evaporation basierend auf temp/humid/wind-Maps mit Verfügbarkeits-Begrenzung
        Input: temp_map, humid_map, wind_map, water_biomes_map, flow_accumulation
        Output: evaporation_map, ocean_outflow (finale Verdunstung und Meer-Abfluss)

    WaterData-Container:
        Funktionsweise: Container für alle Water-Daten mit Status-System und LOD-Management
        Aufgabe: Speichert alle internen und externen Maps mit Validity-State und Parameter-Hash
        Attribute: water_map, flow_map, flow_speed, cross_section, soil_moist_map, erosion_map, sedimentation_map, rock_map_updated, evaporation_map, water_biomes_map, ocean_outflow
        Interne Attribute: lake_data, flow_network_data, manning_data, erosion_transport_data
        Status-Attribute: lake_detection_valid, flow_network_valid, manning_flow_valid, erosion_valid, soil_moisture_valid, evaporation_valid
        LOD-Tracking: lod_level, actual_size, validity_state, parameter_hash
        Methods: is_step_valid(step), invalidate_step(step), get_step_status(step)

    BaseGenerator-Integration:
    - _load_default_parameters(): Lädt WATER-Parameter aus value_default.py
    - _get_dependencies(): Komplexeste Dependency-Resolution aller Generatoren (8 Dependencies) mit Validation
    - _execute_generation(): Führt generate_hydrology() mit Progress-Updates für jeden calculate-Schritt aus
    - _save_to_data_manager(): Speichert WaterData-Objekt automatisch mit Status-Tracking
    """


def biome_generator():
    """
    Path: core/biome_generator.py#
    date changed: 06.08.2025

    Funktionsweise: Klassifikation von Biomen mit BaseGenerator-Integration und TerrainData-ähnlicher Struktur
    - BiomeClassificationSystem erbt von BaseGenerator für einheitliche Generator-Architektur
    - Hauptfunktion: generate_biomes() koordiniert alle Biom-Klassifikationsschritte
    - Progressive Berechnung: calculate_base_classification() → calculate_super_biome_overrides() → calculate_proximity_biomes() → calculate_elevation_biomes() → calculate_supersampling()
    - BiomeData-Container für strukturierte Datenorganisation mit Status-System und Cache-Invalidation
    - Gauß-basierte Klassifizierung mit Gewichtungen je Biomtyp
    - Verwendung eines vektorbasierten Klassifikators für performante Zuordnung auf großen Karten
    - Zwei-Ebenen-System: Base-Biomes und Super-Biomes
    - Supersampling für weichere Übergänge zwischen allen Biomen (die vier dominantesten Anteile pro Zelle)
    - Super-Biomes überschreiben Base-Biomes basierend auf speziellen Bedingungen
    - LOD-System mit Supersampling nur bei LOD256+ für Performance-Optimierung
    - Optional Dependencies mit intelligenten Fallback-Werten (soil_moist_map, water_biomes_map)
    - Threading-Support für Hintergrund-Berechnung mit CPU-Yields

    Parameter Input (aus value_default.py BIOME):
    - biome_wetness_factor (Gewichtung der Bodenfeuchtigkeit)
    - biome_temp_factor (Gewichtung der Temperaturwerte)
    - sea_level (Meeresspiegel-Höhe in Metern)
    - bank_width (Radius für Ufer-Biome in Pixeln)
    - edge_softness (Globaler Weichheits-Faktor für alle Super-Biome-Übergänge, 0.1-2.0)
    - alpine_level (Basis-Höhe für Alpine-Zone in Metern)
    - snow_level (Basis-Höhe für Schneegrenze in Metern)
    - cliff_slope (Grenzwert für Klippen-Klassifikation in Grad)

    data_manager Input:
    - map_seed (Globaler Karten-Seed für reproduzierbare Zufallswerte)
    - heightmap (2D-Array in meter Altitude) - REQUIRED
    - slopemap (2D-Array in m/m mit dz/dx, dz/dy) - REQUIRED
    - temp_map (2D-Array in °C) - REQUIRED
    - soil_moist_map (2D-Array in Bodenfeuchtigkeit %) - OPTIONAL (Fallback: Höhen-basiert)
    - water_biomes_map (2D-Array mit Wasser-Klassifikation: 0=kein Wasser, 1=Creek, 2=River, 3=Grand River, 4=Lake) - OPTIONAL (Fallback: alle 0)

    Output:
    - biome_map (2D-Array mit Index der jeweils dominantesten Biomklasse)
    - biome_map_super (2D-Array, 2x supersampled zur Darstellung gemischter Biome)
    - super_biome_mask (2D-Array, Maske welche Pixel von Super-Biomes überschrieben wurden)

    BASE_BIOME_CLASSIFICATIONS = {
        'ice_cap': 'T=-40--5 | soil_moist=0-300 | h=0-8000 | slope=0-90',
        'tundra': 'T=-15-5 | soil_moist=100-600 | h=0-2000 | slope=0-30',
        'taiga': 'T=-10-15 | soil_moist=300-1200 | h=50-2500 | slope=0-45',
        'grassland': 'T=0-25 | soil_moist=200-800 | h=10-1500 | slope=0-15',
        'temperate_forest': 'T=5-25 | soil_moist=600-2000 | h=0-2000 | slope=0-60',
        'mediterranean': 'T=8-30 | soil_moist=300-900 | h=0-1200 | slope=0-45',
        'desert': 'T=10-50 | soil_moist=0-250 | h=0-2000 | slope=0-30',
        'semi_arid': 'T=5-35 | soil_moist=200-600 | h=0-1800 | slope=0-25',
        'tropical_rainforest': 'T=20-35 | soil_moist=1500-4000 | h=0-1500 | slope=0-70',
        'tropical_seasonal': 'T=18-35 | soil_moist=800-2000 | h=0-1200 | slope=0-40',
        'savanna': 'T=15-35 | soil_moist=400-1200 | h=0-1800 | slope=0-20',
        'montane_forest': 'T=0-20 | soil_moist=800-3000 | h=800-3500 | slope=5-80',
        'swamp': 'T=5-35 | soil_moist=800-3000 | h=0-200 | slope=0-5',
        'coastal_dunes': 'T=5-35 | soil_moist=300-1500 | h=0-100 | slope=5-45',
        'badlands': 'T=-5-45 | soil_moist=0-400 | h=200-2500 | slope=15-90'
    }

    SUPER_BIOME_CONDITIONS = {
        'ocean': {
            'condition': 'lokales Minimum + Randverbindung + h < sea_level',
            'description': 'Flood-Fill von lokalen Minima die mit Kartenrand verbunden sind und unter sea_level liegen',
            'priority': 0
        },
        'lake': {
            'condition': 'water_biomes_map == 4',
            'description': 'Direkt aus water_biomes_map übernommen',
            'priority': 1
        },
        'grand_river': {
            'condition': 'water_biomes_map == 3',
            'description': 'Direkt aus water_biomes_map übernommen',
            'priority': 2
        },
        'river': {
            'condition': 'water_biomes_map == 2',
            'description': 'Direkt aus water_biomes_map übernommen',
            'priority': 3
        },
        'creek': {
            'condition': 'water_biomes_map == 1',
            'description': 'Direkt aus water_biomes_map übernommen',
            'priority': 4
        },
        'cliff': {
            'condition': 'slope_degrees > cliff_slope',
            'description': 'Klippe ab einem Grenzwert cliff_slope mit weichen Übergängen',
            'priority': 5,
            'soft_transition': True,
            'probability_formula': 'sigmoid((slope_degrees - cliff_slope) / edge_softness)'
        },
        'beach': {
            'condition': 'Nähe zu Ocean + h <= sea_level + 5',
            'description': 'Innerhalb bank_width Distanz zu Ocean-Pixeln mit weichen Rändern',
            'priority': 6,
            'soft_transition': True,
            'probability_formula': 'max(0, 1 - (distance_to_ocean / bank_width)^edge_softness)'
        },
        'lake_edge': {
            'condition': 'Nähe zu Lake + nicht selbst Lake',
            'description': 'Innerhalb bank_width Distanz zu Lake-Pixeln mit weichen Rändern',
            'priority': 7,
            'soft_transition': True,
            'probability_formula': 'max(0, 1 - (distance_to_lake / bank_width)^edge_softness)'
        },
        'river_bank': {
            'condition': 'Nähe zu River/Grand River/Creek + nicht selbst Wasser',
            'description': 'Innerhalb bank_width Distanz zu Fließgewässern mit weichen Rändern',
            'priority': 8,
            'soft_transition': True,
            'probability_formula': 'max(0, 1 - (distance_to_water / bank_width)^edge_softness)'
        },
        'snow_level': {
            'condition': 'h > snow_level + 500*(1 + temp_map(x,y)/10)',
            'description': 'Schneegrenze mit graduellen, temperaturabhängigen Übergängen',
            'priority': 9,
            'soft_transition': True,
            'probability_formula': 'sigmoid((h - (snow_level + 500*(1 + temp/10))) / (100 * edge_softness))'
        },
        'alpine_level': {
            'condition': 'h > alpine_level + 500*(1 + temp_map(x,y)/10)',
            'description': 'Alpine Zone mit graduellen, temperaturabhängigen Übergängen',
            'priority': 10,
            'soft_transition': True,
            'probability_formula': 'sigmoid((h - (alpine_level + 500*(1 + temp/10))) / (200 * edge_softness))'
        }
    }

    Beschreibung der Biom-Zuweisung:

    Schritt 1: Base-Biome Klassifikation
    Für alle Pixel wird zunächst das Base-Biome durch Gauß-basierte Klassifikation bestimmt:
    - Slope wird berechnet: slope_degrees = arctan(sqrt((dz/dx)² + (dz/dy)²)) * 180/π
    - Für jedes Base-Biome wird die Gauß-Passung für alle 4 Parameter berechnet
    - Das Biom mit dem höchsten kombinierten Gewicht wird als Base-Biome zugewiesen
    - Gewichtung: Temperatur (30%), Niederschlag (35%), Höhe (20%), Slope (15%)

    Schritt 2: Super-Biome Überschreibung
    Super-Biomes überschreiben Base-Biomes in folgender Prioritätsreihenfolge:

    2.1: Ocean Detection
    - Identifikation aller lokalen Minima in heightmap
    - Flood-Fill Algorithmus startet von Kartenrändern
    - Nur Minima die (a) mit Rand verbunden sind UND (b) h < sea_level erfüllen werden zu Ocean
    - Binnengewässer werden NICHT zu Ocean

    2.2: Water-Biomes aus water_biomes_map
    - Direkte Übertragung: water_biomes_map == 1 → Creek
    - Direkte Übertragung: water_biomes_map == 2 → River
    - Direkte Übertragung: water_biomes_map == 3 → Grand River
    - Direkte Übertragung: water_biomes_map == 4 → Lake

    2.3: Proximity-basierte Super-Biomes
    - Beach: Gaussian-Filter um alle Ocean-Pixel mit Radius bank_width + h <= sea_level + 5
    - Lake Edge: Gaussian-Filter um alle Lake-Pixel mit Radius bank_width, außer Lake-Pixel selbst
    - River Bank: Gaussian-Filter um alle Creek/River/Grand River-Pixel mit Radius bank_width, außer Wasser-Pixel selbst

    Schritt 3: Supersampling mit diskretisierter Zufalls-Rotation
    Beide Base-Biomes und Super-Biomes werden 2x2 supersampled mit optimierter Rotation:

    3.1: Diskretisierte Rotations-Zuweisung
    - Rotations-Seed für Pixel (x,y): rotation_hash = (map_seed + 12345 + x*997 + y*991) % 4
    - Rotation-Mapping:
      - hash % 4 == 0: 0° Rotation (Sub-Pixel-Reihenfolge: TL, TR, BL, BR)
      - hash % 4 == 1: 90° Rotation (Sub-Pixel-Reihenfolge: BL, TL, BR, TR)
      - hash % 4 == 2: 180° Rotation (Sub-Pixel-Reihenfolge: BR, BL, TR, TL)
      - hash % 4 == 3: 270° Rotation (Sub-Pixel-Reihenfolge: TR, BR, TL, BL)
    - Große Primzahlen (997, 991) + Konstante (12345) sorgen für maximale Streuung
    - Benachbarte Pixel haben garantiert unterschiedliche Rotations-Bereiche
    - Verhindert zusammenhängende Bereiche mit gleicher Sub-Pixel-Anordnung

    3.2: Wahrscheinlichkeitsbasierte Super-Biome mit einheitlicher Softness
    Für Super-Biomes mit soft_transition=True wird edge_softness global angewendet:
    - **Cliff:** sigmoid((slope_degrees - cliff_slope) / edge_softness)
    - **Bank-Biomes:** max(0, 1 - (distance / bank_width)^edge_softness)
    - **Höhen-Biomes:**
      - Snow: sigmoid((height_diff) / (100 * edge_softness))
      - Alpine: sigmoid((height_diff) / (200 * edge_softness))
    - **Einheitliche Kontrolle:** Ein Parameter steuert alle Übergangs-Weichheiten
    - **Konsistenz:** Alle Super-Biomes haben ähnlich weiche/harte Übergänge

    3.3: Sub-Pixel Zuweisung mit optimierter Zufälligkeit
    - Sub-Pixel-Seed: sub_seed = (map_seed + 54321 + x*4 + y*4 + rotated_sub_index*3571) % 1000
    - Wahrscheinlichkeit: probability = sub_seed / 1000.0 (Bereich 0.000-0.999)
    - Wenn probability < Super-Biome-Wahrscheinlichkeit: Super-Biome zugewiesen
    - Primzahl 3571 sorgt für unkorrelierte Sub-Pixel-Wahrscheinlichkeiten
    - Konstante 54321 trennt Rotations- und Wahrscheinlichkeits-Zufälligkeit

    Vorteile des optimierten Systems:
    - **Einheitliche Softness-Kontrolle:** Ein Parameter `edge_softness` steuert alle Super-Biome-Übergänge
    - **Optimierte Zufälligkeit:** Große Primzahlen und Konstanten maximieren räumliche Streuung
    - **Diskretisierte Rotation:** Benachbarte Pixel haben garantiert unterschiedliche Rotations-Bereiche
    - **Reproduzierbare Ergebnisse:** `map_seed` sorgt für konsistente, aber variierte Zufälligkeit
    - **Keine Korrelation:** Verschiedene Zufallsaspekte (Rotation, Wahrscheinlichkeit) sind unabhängig
    - **Performance-optimiert:** Integer-Arithmetik und Modulo-Operationen für schnelle Berechnung
    - **Natürliche Muster:** Eliminiert künstliche Repetition durch mathematisch optimierte Streuung

    Klassen:
    BiomeClassificationSystem (BaseGenerator)
        Funktionsweise: Hauptklasse für Biom-Klassifikation mit BaseGenerator-API und LOD-System
        Aufgabe: Koordiniert Base-Biome und Super-Biome Zuordnung mit Supersampling und Progress-Updates
        Methoden: generate(), _execute_generation(), _load_default_parameters(), _get_dependencies()

    BaseBiomeClassifier
        Funktionsweise: Gauß-basierte Klassifizierung von 15 Grundbiomen mit LOD-Optimierung
        Aufgabe: Erstellt biome_map basierend auf Höhe, Temperatur, Niederschlag und Slope
        Methoden: calculate_gaussian_fitness(), weight_environmental_factors(), assign_dominant_biome()

    SuperBiomeOverrideSystem
        Funktionsweise: Überschreibt Base-Biomes mit speziellen Bedingungen (Ocean, Cliff, Beach, etc.)
        Aufgabe: Erstellt super_biome_mask für prioritätsbasierte Biom-Überschreibung
        Methoden: detect_ocean_connectivity(), apply_proximity_biomes(), calculate_elevation_biomes()

    SupersamplingManager
        Funktionsweise: 2x2 Supersampling mit diskretisierter Zufalls-Rotation (nur bei LOD256+)
        Aufgabe: Erstellt biome_map_super für weiche Übergänge zwischen Biomen bei höheren LODs
        Methoden: apply_rotational_supersampling(), calculate_soft_transitions(), optimize_spatial_distribution()

    ProximityBiomeCalculator
        Funktionsweise: Berechnet Proximity-basierte Super-Biomes (Beach, Lake Edge, River Bank)
        Aufgabe: Erstellt weiche Übergänge um Gewässer mit konfigurierbarem edge_softness
        Methoden: calculate_distance_fields(), apply_gaussian_proximity(), blend_with_base_biomes()
    calculate_base_classification():
        Status: base_classification_valid, biome_count, fitness_calculated, gaussian_applied
        Aufgabe: Gauß-basierte Klassifikation aller 15 Base-Biomes durch Environmental-Factor-Weighting
        Input: temp_map, soil_moist_map, heightmap, slopemap, biome_wetness_factor, biome_temp_factor
        Output: base_biome_map (dominante Base-Biomes), fitness_maps (Fitness-Werte aller Biome)

    calculate_super_biome_overrides():
        Status: water_biomes_applied, ocean_detected, water_override_count
        Aufgabe: Direkte Überschreibung durch Wasser-Biomes (Ocean/Lake/Rivers) mit höchster Priorität
        Input: water_biomes_map, heightmap, sea_level
        Output: water_override_mask (Wasser-Biome-Positionen)

    calculate_proximity_biomes():
        Status: proximity_calculated, distance_fields_valid, beach_count, lake_edge_count, river_bank_count
        Aufgabe: Proximity-basierte Super-Biomes (Beach/Lake Edge/River Bank) mit weichen Übergängen
        Input: water_biomes_map, heightmap, bank_width, edge_softness
        Output: proximity_masks (Wahrscheinlichkeits-Maps für Ufer-Biome)

    calculate_elevation_biomes():
        Status: elevation_biomes_valid, cliff_detected, snow_applied, alpine_applied
        Aufgabe: Höhen-basierte Super-Biomes (Cliff/Snow/Alpine) mit temperaturabhängigen Übergängen
        Input: heightmap, temp_map, slopemap, cliff_slope, snow_level, alpine_level, edge_softness
        Output: elevation_masks (Wahrscheinlichkeits-Maps für Höhen-Biome)

    calculate_supersampling():
        Status: supersampling_valid, rotation_applied, sub_pixel_calculated, enhancement_level
        Aufgabe: 2x2 Supersampling mit diskretisierter Zufalls-Rotation (nur bei LOD256+)
        Input: integrated_biome_map, all_super_masks, map_seed, lod_level
        Output: biome_map_super (2x supersampeltes Biom-Map für weiche Übergänge)

    BiomeData-Container:
        Funktionsweise: Container für alle Biome-Daten mit Status-System und LOD-Management
        Aufgabe: Speichert alle internen und externen Maps mit Validity-State und Parameter-Hash
        Attribute: biome_map, biome_map_super, super_biome_mask, fitness_maps, proximity_masks, elevation_masks
        Status-Attribute: base_classification_valid, water_biomes_applied, proximity_calculated, elevation_biomes_valid, supersampling_valid
        LOD-Tracking: lod_level, actual_size, supersampling_enabled, validity_state, parameter_hash
        Methods: is_step_valid(step), invalidate_step(step), get_step_status(step)

    BaseGenerator-Integration:
    - _load_default_parameters(): Lädt BIOME-Parameter aus value_default.py
    - _get_dependencies(): Holt required + optional Dependencies mit intelligenten Fallback-Werten
    - _execute_generation(): Führt generate_biomes() mit Progress-Updates für jeden calculate-Schritt aus
    - _save_to_data_manager(): Speichert BiomeData-Objekt automatisch mit Status-Tracking
    """


def value_default():
    """
    Path: gui/config/value_default.py

    Funktionsweise: Zentrale Parameter-Defaults für alle Slider und Controls
    - Min/Max/Step/Default Werte für alle Generator-Parameter
    - Organisiert nach Generator-Typen (TERRAIN, GEOLOGY, etc.)
    - Validation-Rules und Parameter-Constraints
    - Einheitliche Decimal-Precision und Suffix-Definitionen

    Struktur:
    class TERRAIN:
        SIZE = {"min": 64, "max": 512, "default": 256, "step": 32}
        HEIGHT = {"min": 0, "max": 400, "default": 100, "step": 10, "suffix": "m"}
        OCTAVES = {"min": 1, "max": 10, "default": 4}
        # etc.

    class GEOLOGY:
        HARDNESS = {"min": 1, "max": 100, "default": 50}
        # etc.
    """


def gui_default():
    """"
    Path: gui/config/gui_default.py

    Funktionsweise: GUI-Layout und Styling-Konfiguration
    - Window-Größen und Positionen für alle Tabs
    - Button-Styling (Farben, Größen, Fonts)
    - Canvas-Konfiguration (Split-Ratios, Render-Settings)
    - Color-Schemes und Theme-Definitionen

    Struktur:

    class WindowSettings:
        MAIN_MENU = {"width": 800, "height": 600}
        MAP_EDITOR = {"width": 1500, "height": 1000}

    class ButtonSettings:
        PRIMARY = {"color": "#27ae60", "hover": "#229954"}
        SECONDARY = {"color": "#3498db", "hover": "#2980b9"}

    class CanvasSettings:
        SPLIT_RATIO = 0.7  # 70% Canvas, 30% Controls
    """

def data_manager():
    """
    Path: gui/managers/data_manager.py

    Funktionsweise: Zentrale Datenverwaltung für alle Tabs mit Validity-System und einheitlichem LOD-Management
    - Memory-effiziente numpy Array-Referenzen (keine Kopien) mit automatischem Garbage-Collection
    - Einheitliches LOD-System: LOD64 (64x64) → LOD128 (128x128) → LOD256 (256x256) → FINAL (512x512+) für alle Generatoren
    - Dependency-Tracking zwischen Generator-Outputs mit automatischer Impact-Propagation
    - Cross-Tab Data-Sharing ohne Pickle-Serialisierung über Signal-basierte Communication
    - Automatic Cache-Invalidation bei Parameter-Änderungen mit Validity-State-Management
    - TerrainData-Integration mit kompletter Metadaten-Speicherung und LOD-Level-Tracking
    - Parameter-Hash-Vergleich für intelligente Cache-Validation
    - Memory-Usage-Tracking für Performance-Monitoring pro LOD-Level

    Kommunikationskanäle:
    - Signals: data_updated, cache_invalidated, dependency_changed für Cross-Tab-Updates
    - Input: Generator-Data von allen Core-Modulen über set_[generator]_data() Methoden
    - Output: Cached Data für alle Tabs über get_[generator]_data() Methoden
    - LOD-Management: Einheitliches LOD-System ohne Translation - alle Generatoren nutzen LOD64/LOD128/LOD256/FINAL

    Validity-System:
    - Parameter-Hash-Tracking: MD5-Hash aller Parameter für Change-Detection pro LOD-Level
    - Cache-Timestamp-Management: Verfolgt letzte Änderung für jeden Generator, Data-Key und LOD-Level
    - Validity-State-Propagation: Invalidiert nachgelagerte Generatoren basierend auf Impact-Matrix
    - Cross-Generator-Dependencies: Verwaltet komplette Dependency-Chain von Terrain bis Settlement mit LOD-Awareness
    - Memory-Leak-Prevention: Automatisches Cleanup von invaliden Cache-Einträgen pro LOD-Level

    LOD-Level-Management:
    - LOD64: Basis-Level für alle Generatoren (64x64 für die meisten, shadowmap immer 64x64)
    - LOD128: Erhöhte Auflösung (128x128) mit erweiterten Features (mehr Sonnenwinkel, etc.)
    - LOD256: Hohe Auflösung (256x256) für detaillierte Previews
    - FINAL: Maximale Auflösung (512x512+) für finale Ausgabe
    - LOD-Hierarchy-Validation: Stellt sicher dass höhere LODs auf niedrigeren basieren

    Generator-Data-Management:
    - Terrain: heightmap, slopemap, shadowmap mit TerrainData-Objekt-Support und LOD-Metadaten
    - Geology: rock_map (RGB), hardness_map mit Mass-Conservation-Validation pro LOD
    - Weather: wind_map (Vektoren), temp_map, precip_map, humid_map mit Atmospheric-Consistency-Checks
    - Water: water_map, flow_map, soil_moist_map, water_biomes_map mit Hydrologic-Cycle-Validation
    - Biome: biome_map, biome_map_super, super_biome_mask mit Climate-Consistency-Checks
    - Settlement: settlement_list, landmark_list, plot_map, civ_map mit Terrain-Suitability-Validation

    Performance-Features:
    - Memory-Usage-Monitoring: get_memory_usage() für alle Generator-Arrays pro LOD-Level
    - LOD-Specific-Caching: Separate Cache-Verwaltung für jeden LOD-Level
    - Progressive-Loading: Lädt höhere LODs nur bei Bedarf
    - Garbage-Collection-Optimization: Automatisches Cleanup nicht referenzierter Arrays pro LOD
    - Cross-Tab-Reference-Counting: Verhindert vorzeitige Garbage-Collection aktiv genutzter Daten
    """


def generation_orchestrator():
    """
    Path: gui/managers/generation_orchestrator.py

    Funktionsweise: Zentrale Orchestrierung aller Generator-Berechnungen mit LOD-System, Dependency-Management und Validity-Tracking
    - Verwaltet komplette Dependency-Chain: Terrain → Geology → Weather → Water → Biome → Settlement
    - LOD-Progression-System: LOD64 → LOD128 → LOD256 → FINAL mit Background-Threading und Interrupt-Support
    - Intelligente Dependency-Queue verhindert Deadlocks und löst Dependencies automatisch auf

    - Parameter-Impact-Matrix definiert welche Parameter-Änderungen nachgelagerte Generatoren invalidieren
    - Cross-Generator Validity-Tracking und automatische Re-Berechnung bei kritischen Parameter-Änderungen
    - Threading-Koordination für parallele und sequenzielle Generation mit Thread-Safety
    - Memory-Management und Performance-Optimierung für große Datenmengen
    - 8-Generator-Thread-Status-Display für UI-Integration mit Queue-Position und ETA

    Kommunikation:
    - Input: Parameter-Änderungen von allen Tabs über OrchestratorRequestBuilder
    - Output: Signals für UI-Updates, DataManager-Updates, Progress-Callbacks mit Thread-safe Qt-Signal-System
    - Threading: Background-Threads für LOD-Enhancement ohne UI-Blocking mit CPU-Priority-Management
    - Dependencies: Koordiniert mit DataManager für Cross-Tab Data-Sharing und Validity-Propagation

    Parameter-Impact-Matrix (definiert Cache-Invalidation-Strategien):
    - Terrain-Changes: High-Impact (map_seed, size, amplitude) → invalidiert alle nachgelagerten Generatoren
    - Geology-Changes: Medium-Impact (hardness) → invalidiert Water, Biome, Settlement
    - Weather-Changes: Medium-Impact (precipitation, temperature) → invalidiert Water, Biome, Settlement
    - Water-Changes: Low-Impact (erosion) → invalidiert nur Biome, Settlement
    - Biome-Changes: Low-Impact → invalidiert nur Settlement
    - Settlement-Changes: No-Impact → invalidiert nichts (letzter in Chain)

    LOD-Progression-Logic:
    - Adaptive-LOD-Selection: Startet bei niedrigstem benötigtem LOD basierend auf verfügbaren Dependencies
    - Progressive-Enhancement: Jeder LOD-Level wird sofort an UI weitergegeben für Preview
    - Interrupt-Support: Parameter-Änderungen stoppen Generation nach aktuellem LOD-Step
    - Memory-Pressure-Adaptation: Reduziert automatisch Target-LOD bei Memory-Constraints
    - Quality-vs-Performance-Balancing: Intelligente LOD-Auswahl basierend auf System-Performance

    Queue-System mit Deadlock-Prevention:
    - Dependency-Resolution: Kontinuierliche Queue-Verarbeitung alle 2 Sekunden
    - Priority-Based-Scheduling: Terrain hat höchste Priority, Settlement niedrigste
    - Timeout-Management: 5min Generation-Timeout, 10min Queue-Timeout mit automatischem Cleanup
    - Thread-Pool-Management: Max 3 parallele Generationen mit CPU-Core-Balancing
    - Request-Validation: Typ-sichere Request-Validation mit Parameter-Range-Checks

    Design-Pattern:
    - Observer-Pattern für Tab-Communication mit Signal-basierter Loose-Coupling
    - Command-Pattern für Generation-Requests mit Undo/Redo-Support
    - Strategy-Pattern für verschiedene LOD-Levels mit adaptiver Algorithm-Selection
    - Factory-Pattern für Generator-Instanzen mit Lazy-Loading
    - Queue-Pattern für Dependency-Resolution mit Priority-Scheduling
    - State-Machine-Pattern für Generation-States mit Error-Recovery
    """


def orchestrator_manager():
    """

    Path: gui/managers/orchestrator_manager.py

    Funktionsweise: Eliminiert Code-Duplikation der Orchestrator-Integration zwischen allen Generator-Tabs
    Aufgabe: Standard-Orchestrator-Handler, Request-Building, Signal-Management, Error-Handling für wiederverwendbare Integration
    Features: Thread-safe UI-Updates, Request-Validation, Batch-Operations, Multi-Tab-Coordination

    Komponenten:
    StandardOrchestratorHandler:
        Funktionsweise: Wiederverwendbare Orchestrator-Integration für alle Generator-Tabs
        Aufgabe: Eliminiert Code-Duplikation zwischen Generator-Tabs mit einheitlichen Signal-Handlern
        Features: Thread-safe Qt-Signal-Updates, Custom-UI-Method-Mapping, Connection-Status-Tracking
        Signal-Integration: generation_started, generation_completed, generation_progress, lod_progression_completed
        UI-Update-Methods: Configurable method-mapping für verschiedene Tab-Implementierungen

    OrchestratorRequestBuilder:
        Funktionsweise: Builder-Pattern für typ-sichere Orchestrator-Requests
        Aufgabe: Konsistente Request-Erstellung, Parameter-Validation, Generator-spezifische Default-Values
        Features: Terrain/Geology-spezialisierte Builder, Parameter-Range-Validation, Batch-Request-Support
        Validation: Request-Completeness-Check, LOD-Validation, Priority-Range-Check, Parameter-Constraint-Validation
        Templates: build_terrain_request(), build_geology_request(), build_standard_request(), build_batch_request()

    OrchestratorErrorHandler:
        Funktionsweise: Error-Handling für Orchestrator-Integration mit Retry-Logic
        Aufgabe: Error-Classification, Automatic-Retry, Parameter-Correction, Recovery-Strategies
        Features: Memory-Error-Recovery (LOD-Downgrade), Parameter-Error-Correction, Network-Timeout-Retry
        Error-Types: memory_error, parameter_error, retryable, critical mit entsprechenden Recovery-Strategies
        Retry-Logic: Max 3 Retries mit exponential backoff, Parameter-Correction bei bekannten Error-Patterns

    OrchestratorIntegrationManager:
        Funktionsweise: High-Level Manager für komplette Multi-Tab Orchestrator-Integration
        Aufgabe: Koordiniert Handler, Request-Builder und Error-Handler für alle Generator-Tabs
        Features: Multi-Tab-Registration, Batch-Generation-Requests, Status-Monitoring, Resource-Cleanup
        Tab-Management: register_tab(), request_generation(), request_batch_generation(), cleanup_all_handlers()
        Status-Monitoring: get_generation_status(), connection-status für alle registrierten Tabs

    Factory-Functions:
        create_standard_orchestrator_integration(): Factory für vorkonfigurierten IntegrationManager
        setup_tab_orchestrator_integration(): Setup für einzelnen Tab mit Orchestrator-Integration
        register_standard_map_generator_tabs(): Bulk-Registration aller Standard-Generator-Tabs

    Thread-Safety und Performance:
    - Qt-Signal-System für Thread-safe UI-Updates mit QMetaObject.invokeMethod()
    - Connection-Status-Tracking für Debugging und Health-Monitoring
    - Automatic-Cleanup bei Tab-Destruction oder App-Shutdown
    - Memory-Leak-Prevention durch proper Signal-Disconnection
    """


def shader_manager():
    """
    Path: gui/managers/shader_manager.py

    Funktionsweise: GPU-Compute Management für Performance-kritische Operationen
    - OpenGL Compute Shader für Parallel-Processing
    - Fallback auf CPU für Systeme ohne GPU-Support
    - Optimierte Operationen: Noise-Generation, Erosion, Biome-Blending
    - Memory-Management zwischen GPU und CPU

    Einsatzgebiete:
    - Terrain: Multi-Octave Simplex-Noise parallel
    - Water: Hydraulic Erosion simulation
    - Weather: Wind-Field calculations
    - Biome: Vegetation distribution

    Kommunikationskanäle:
    - Input: Large numpy arrays für GPU-Processing
    - Output: Processed arrays zurück an Data-Manager
    """


def navigation_manager():
    """
    Path: gui/managers/navigation_manager.py

    Funktionsweise: Zentrale Tab-Navigation und Parameter-Persistierung mit Dependency-Validation
    - Tab-Reihenfolge: main_menu → terrain → geology → settlement → weather → water → biome → overview
    - Automatische Parameter-Speicherung vor Tab-Wechsel mit Validity-Checks
    - Window-Geometrie Persistierung zwischen Sessions
    - Dependency-Validation vor Navigation mit User-friendly Error-Messages
    - Graceful Cleanup und Resource-Management bei Tab-Wechseln
    - Cross-Tab-Parameter-Transfer über DataManager-Integration

    Kommunikationskanäle:
    - Signals: tab_changed, parameters_saved, validation_failed für Cross-Component-Communication
    - Config: gui_default.py für Window-Settings und Default-Geometries
    - Data: Koordination mit DataManager für Parameter-Transfer und Dependency-Checks
    - Validation: Nutzt DataManager.check_dependencies() für vollständige Dependency-Validation

    Tab-Dependencies (definiert Navigation-Constraints):
    - main_menu: [] (keine Dependencies)
    - terrain: [] (Basis-Generator)
    - geology: [terrain] (braucht heightmap)
    - weather: [terrain] (braucht heightmap für orographic effects)
    - water: [terrain, geology, weather] (braucht heightmap, hardness_map, precipitation)
    - biome: [terrain, weather, water] (braucht heightmap, temperature, soil_moisture)
    - settlement: [terrain, water, biome] (braucht heightmap, water_biomes, biome_map)
    - overview: [terrain, geology, settlement, weather, water, biome] (braucht alle für Export)

    Navigation-Validation:
    - Dependency-Completeness-Check: Prüft ob alle required Outputs verfügbar sind
    - LOD-Sufficiency-Check: Mindestens LOD64 muss für alle Dependencies vorhanden sein
    - Parameter-Consistency-Check: Validiert dass alle Dependencies mit aktuellen Parametern berechnet sind
    - Error-Recovery-Suggestions: Schlägt automatische Re-Generation fehlender Dependencies vor

    Window-Management:
    - Multi-Tab-Geometry-Persistence: Separate Geometrie für jeden Tab
    - Resolution-Adaptation: Passt Window-Size an Display-Resolution an
    - Multi-Monitor-Support: Merkt sich Monitor-Assignment für jeden Tab
    - Graceful-Tab-Cleanup: Speichert Parameter und räumt Ressourcen auf bei Tab-Wechsel
    """


def main_menu():
    """
    Path: gui/tabs/main_menu.py

    Funktionsweise: Hauptmenü mit eleganter Navigation
    - BaseButton Widgets für einheitliche Button-Darstellung
    - Navigation über NavigationManager (kein direktes Tab-Loading)
    - Gradient-Hintergrund und responsive Layout
    - Status-Display für verfügbare Core-Module

    Kommunikationskanäle:
    - Config: gui_default.py für Layout und Button-Styling
    - Navigation: navigation_manager.navigate_to_tab('terrain')
    - Buttons: Map-Editor (functional), Laden/Settings (non-functional), Beenden
    """


def base_tab():
    """
    Path: gui/tabs/base_tab.py

    Funktionsweise: Basis-Klasse für alle Map-Editor Tabs
    - Standardisiertes 70/30 Layout (Canvas links, Controls rechts)
    - Gemeinsame Auto-Simulation Controls für alle Tabs
    - Input-Status Display (verfügbare Dependencies)
    - Observer-Pattern für Cross-Tab Updates
    - Einheitliche Navigation (Prev/Next Buttons)
    - Performance-optimiertes Debouncing-System

    Kommunikationskanäle:
    - Signals: data_updated, parameter_changed, validation_status_changed
    - Config: gui_default.py für Layout-Settings
    - Data: data_manager für Input/Output
    - Navigation: navigation_manager für Tab-Wechsel

    Gemeinsame Features:
    - setup_common_ui() → 70/30 Layout
    - setup_auto_simulation() → Auto-Update Checkbox + Manual Button
    - setup_input_status() → Dependency-Status Widget
    - setup_navigation() → Prev/Next Navigation
    """


def terrain_tab():
    """

    Path: gui/tabs/terrain_tab.py
    Funktionsweise: Terrain-Editor mit GenerationOrchestrator und StandardOrchestratorHandler Integration
    - Erbt von BaseMapTab für gemeinsame Features
    - GenerationOrchestrator für einheitliche Generator-Architektur mit Dependency-Management
    - StandardOrchestratorHandler eliminiert Code-Duplikation zwischen Generator-Tabs
    - OrchestratorRequestBuilder für typ-sichere Request-Erstellung mit Terrain-spezifischer Validation
    - Live 2D/3D Preview über map_display_2d/3d.py mit Display-Modi (Height/Slope/Shadow)
    - Real-time Terrain-Statistics mit Performance-optimierter Berechnung und Caching
    - Parameter-Update-Manager gegen Race-Conditions mit Debouncing-System
    - Generation-Interrupt-Logic mit Validity-System und automatischer Cache-Invalidation
    - Output: TerrainData-Objekt über DataManager für nachfolgende Generatoren
    - Map-Size-Synchronisation zwischen Tabs über DataManager.sync_map_size()

    UI-Layout:
    Control Panel (rechts, 30% Breite):
      - Generate Terrain Button: Manual Generation-Trigger mit Priority-Handling

      - System Status Widget:
        * LOD-Level-Display für Heightmap/Slopemap/Shadowmap mit Validity-State
        * Generation-Status: Queued/Generating/Completed/Failed mit Progress-Details
        * Dependency-Status: Waiting für Dependencies, Error-States mit Recovery-Suggestions
        * Queue-Position und estimated completion Zeit
      - Progress Bar: LOD-progression mit aktueller Phase (Heightmap/Slopes/Shadows) und Detail-Messages
      - Terrain Parameters: Slider für alle Parameter mit RandomSeedButton für map_seed, Parameter-Validation mit Cross-Parameter-Constraints
      - Terrain Statistics: Höhenverteilung, Steigungsstatistiken, Verschattungsinfo mit LOD-Info und Performance-Metriken

    Hauptpanel (links, 70% Breite):
      - Oben: 2D/3D Toggle + Display-Optionen
        * Height/Slope: Radio-Buttons (exklusiv)
        * Shadow: Checkbox (Overlay-Modus) mit Sonnenwinkel-Slider (1-7 Winkel)
            (Falls der aktive Winkel noch nicht berechnet wurde, sollte die Anzeige nicht dargestellt werden.)
        * Contour Lines, Grid-Overlay: Checkboxes für zusätzliche Overlays
      - Mitte: Display-Widget mit automatischer Colorbar und LOD-Info-Overlay
      - Unten: Display-Tools (Measure Distance, Height-Profile, Export-View, etc.)

    Kommunikationskanäle:
    - Config: value_default.TERRAIN für alle Parameter-Ranges und Validation-Rules
    - Core: GenerationOrchestrator.request_generation("terrain") mit OrchestratorRequestBuilder
    - Handler: StandardOrchestratorHandler für einheitliche Signal-Integration und UI-Updates
    - BaseGenerator-Integration: Nutzt einheitliche Generator-Architektur mit automatischem Dependency-Tracking
    - Signals: terrain_data_updated, validity_changed (für nachfolgende Tabs)
    - Output: TerrainData → DataManager.set_terrain_data_complete() für geology/settlement/weather
    - Map-Size-Sync: DataManager.sync_map_size() für Tab-übergreifende Größen-Konsistenz mit Validity-Checks

    Parameter: size, amplitude, octaves, frequency, persistence, lacunarity, redistribute_power, map_seed

    LOD-System Integration:
    - System Status zeigt aktuelles LOD-Level und Validity-State für alle Terrain-Daten
    - Generation läuft LOD64→LOD128→LOD256→FINAL mit Validity-Checks nach jedem Level
    - Parameter-Änderungen invalidieren Daten und stoppen Generation nach aktuellem LOD-Step
    - Display arbeitet immer mit besten verfügbaren LOD-Daten, zeigt LOD-Info in UI
    - Cache-Invalidation propagiert automatisch zu nachgelagerten Generatoren über Impact-Matrix

    Generation-Flow mit Validity-System:
    1. Parameter-Änderung → ParameterUpdateManager (Debouncing) → Validity-Check gegen cached Parameters
    2. Parameter-Impact-Analysis → Cache-Invalidation für betroffene nachgelagerte Generatoren
    3. Generate Button / Auto-Simulation → OrchestratorRequestBuilder.build_terrain_request()
    4. Request-Validation → GenerationOrchestrator.request_generation() mit Dependency-Queue
    5. LOD-progression mit Interrupt-Support bei Parameter-Änderungen und Validity-Re-Check
    6. Progress-Updates → StandardOrchestratorHandler → System Status + Progress Bar Updates
    7. Completion → Display-Update + Statistics-Update + Map-Size-Sync + Validity-State-Update
    8. Validity-System stellt Konsistenz zwischen allen LOD-Levels sicher und triggert nachgelagerte Re-Generation bei kritischen Änderungen

    Error-Handling und Recovery:
    - OrchestratorErrorHandler mit Retry-Logic für Memory-Errors (LOD-Downgrade) und Parameter-Errors (Auto-Correction)
    - Generation-Timeout-Detection mit automatischem Cleanup und User-Notification
    - Invalid-Parameter-Detection mit User-friendly Error-Messages und Correction-Suggestions
    - GPU-Fallback-Notification wenn ShaderManager auf CPU-Processing zurückfällt
    """


def geology_tab():
    """
    Path: gui/tabs/geology_tab.py
    date changed: 06.08.2025

    Funktionsweise: Geology-Editor mit terrain_tab-ähnlicher UI-Struktur
    - Erbt von BaseMapTab für gemeinsame Features (70/30 Layout, Navigation, etc.)
    - UI-Struktur: Generate Button → LOD/Status-Display → Parameter-Panel → Statistics → Navigation (fixiert unten)
    - GenerationOrchestrator Integration mit StandardOrchestratorHandler
    - Real-time Status-Display für alle calculate-Schritte mit Progress und Validity-State
    - Live Rock-Distribution Preview und Mass-Conservation Monitoring

    UI-Layout:
    Control Panel (rechts, 30% Breite):
      - Generate Geology Button: Manual Generation-Trigger mit Priority-Handling
      - System Status Widget:
        * LOD-Level-Display für alle GeologyData-Komponenten mit Status-Icons
        * Step-Status: elevation_classification → slope_hardening → geological_zones → deformation → hardness
        * Generation-Status: Queued/Generating/Completed/Failed mit Detail-Messages
        * Validity-State für jeden Berechnungsschritt mit Error-Recovery-Suggestions
      - Progress Bar: Step-progression mit aktueller Phase und Detail-Messages
      - Geology Parameters: Hardness-Slider und Deformation-Parameter mit Live-Preview
      - Rock Distribution Statistics: Mass-Conservation Status, Gesteinsverteilung-Preview, Härte-Statistiken

    Hauptpanel (links, 70% Breite):
      - Oben: 2D/3D Toggle + Display-Optionen (Heightmap/Hardness/Rock Types)
      - Mitte: Display-Widget mit Rock-Map-Visualization und LOD-Info-Overlay
      - Unten: Display-Tools und 3D-Terrain-Overlay-Controls
    """


def settlement_tab():
    """
    Path: gui/tabs/settlement_tab.py

    Funktionsweise: Settlement-Editor mit terrain_tab-ähnlicher UI-Struktur und vollständiger Core-Integration
    - Erbt von BaseMapTab für gemeinsame Features (70/30 Layout, Navigation, etc.)
    - UI-Struktur: Generate Button → LOD/Status-Display → Parameter-Panel → Statistics → Navigation (fixiert unten)
    - GenerationOrchestrator Integration mit StandardOrchestratorHandler
    - Real-time Status-Display für alle calculate-Schritte mit Progress und Validity-State
    - Live Settlement-Preview und 3D-Visualization mit Terrain-Integration
    - Input: heightmap, slopemap, water_map von vorherigen Generatoren
    - Intelligent Settlement-Placement mit Terrain-Suitability Analysis
    - Road-Network mit Pathfinding und Spline-Interpolation
    - Plotnode-System mit Delaunay-Triangulation
    - 3D-Markers für Villages/Landmarks/Roadsites auf Terrain
    - Civilization-Map für späteres Biome-System

    UI-Layout:
    Control Panel (rechts, 30% Breite):
      - Generate Settlement Button: Manual Generation-Trigger mit Priority-Handling
      - System Status Widget:
        * LOD-Level-Display für alle SettlementData-Komponenten mit Status-Icons
        * Step-Status: terrain_suitability → settlements → road_network → roadsites → civilization_mapping → landmarks → plots
        * Generation-Status: Queued/Generating/Completed/Failed mit Detail-Messages
        * Multi-Dependency-Status für alle Required Input-Maps mit Recovery-Suggestions
        * Validity-State für jeden Berechnungsschritt mit Error-Recovery-Suggestions
      - Progress Bar: Step-progression mit aktueller Phase und Detail-Messages
      - Settlement Parameters:
        * Location Counts: settlements, landmarks, roadsites, plotnodes mit Live-Preview
        * Civilization Influence: civ_influence_decay, terrain_factor_villages
        * Road Network: road_slope_to_distance_ratio
        * Wilderness & Plots: landmark_wilderness, plotsize
      - Settlement Statistics:
        * Parameter Preview: Settlement-Counts, Suitability-Stats
        * Generation Results: Actual Counts, Road Length, Plot Count, Avg Suitability
      - Civilization Influence Widget: Influence-Parameter, Decay-Preview, Civilization-Statistics

    Hauptpanel (links, 70% Breite):
      - Oben: 2D/3D Toggle + Display-Optionen
        * Terrain Suitability: Radio-Button für Suitability-Map-Display
        * Settlements & Landmarks: Radio-Button mit Typ-Filtern (Settlements/Landmarks/Roadsites)
        * Road Network: Radio-Button für Road-Visualization
        * Civilization Map: Radio-Button für civ_map-Display
        * Plot Boundaries: Radio-Button für Plot-System-Display
        * 3D Visualization: Checkboxes für 3D Terrain Overlay und Settlement Markers
      - Mitte: Display-Widget mit Settlement-Visualization und Multi-Layer-Overlays
      - Unten: Display-Tools und Settlement-Legend mit Location-Type-Definitions

    Kommunikationskanäle:
    - Config: value_default.SETTLEMENT für alle Parameter-Ranges und Validation-Rules
    - Core: GenerationOrchestrator.request_generation("settlement") mit OrchestratorRequestBuilder
    - Handler: StandardOrchestratorHandler für einheitliche Signal-Integration und UI-Updates
    - BaseGenerator-Integration: Nutzt einheitliche Generator-Architektur mit automatischem Dependency-Tracking
    - Signals: settlement_data_updated, validity_changed (für nachfolgende Tabs)
    - Output: SettlementData → DataManager.set_settlement_data() für biome-Generator Dependencies
    - Dependencies: heightmap, slopemap, water_map mit Multi-Dependency-Status-Widget

    Parameter: settlements, landmarks, roadsites, plotnodes, civ_influence_decay, terrain_factor_villages, road_slope_to_distance_ratio, landmark_wilderness, plotsize

    LOD-System Integration:
    - System Status zeigt aktuelles LOD-Level und Validity-State für alle Settlement-Daten
    - Generation läuft LOD64→LOD128→LOD256→FINAL mit Validity-Checks nach jedem Level
    - Parameter-Änderungen invalidieren Daten und stoppen Generation nach aktuellem LOD-Step
    - Display arbeitet immer mit besten verfügbaren LOD-Daten, zeigt LOD-Info in UI
    - Cache-Invalidation propagiert automatisch zu nachgelagerten Generatoren über Impact-Matrix

    Generation-Flow mit Validity-System:
    1. Parameter-Änderung → ParameterUpdateManager (Debouncing) → Validity-Check gegen cached Parameters
    2. Parameter-Impact-Analysis → Cache-Invalidation für betroffene nachgelagerte Generatoren
    3. Generate Button / Auto-Simulation → OrchestratorRequestBuilder.build_settlement_request()
    4. Request-Validation → GenerationOrchestrator.request_generation() mit Dependency-Queue
    5. LOD-progression mit Interrupt-Support bei Parameter-Änderungen und Validity-Re-Check
    6. Progress-Updates → StandardOrchestratorHandler → System Status + Progress Bar Updates
    7. Completion → Display-Update + Statistics-Update + Validity-State-Update
    8. Validity-System stellt Konsistenz zwischen allen LOD-Levels sicher und triggert nachgelagerte Re-Generation bei kritischen Änderungen

    Visualization-Modi:
    - Terrain Suitability: Zeigt combined_suitability_map mit optimalen Settlement-Bereichen
    - Settlements & Landmarks: 3D-Markers auf Terrain mit Typ-Filtern und Radius-Visualization
    - Road Network: Spline-geglättete Straßenverläufe mit Pathfinding-Visualization
    - Civilization Map: civ_map-Heatmap mit Wilderness-Bereichen und Einfluss-Gradienten
    - Plot Boundaries: Delaunay-Triangulation und Plot-Grenzen mit Node-Optimization-Display

    3D-Integration:
    - 3D Terrain Overlay: Heightmap als 3D-Mesh mit Settlement-Positionen
    - Settlement Markers: 3D-Symbole für Villages/Landmarks/Roadsites mit Größen-Skalierung
    - Road Visualization: 3D-Pfade entlang Terrain mit Elevation-Profile
    - Civilization Influence: 3D-Heatmap-Overlay auf Terrain mit Höhen-abhängigem Decay

    Error-Handling und Recovery:
    - OrchestratorErrorHandler mit Retry-Logic für Memory-Errors (LOD-Downgrade) und Parameter-Errors (Auto-Correction)
    - Generation-Timeout-Detection mit automatischem Cleanup und User-Notification
    - Invalid-Parameter-Detection mit User-friendly Error-Messages und Correction-Suggestions
    - Dependency-Missing-Recovery mit automatischen Re-Generation-Suggestions vorheriger Generatoren
    """


def weather_tab():
    """
    Path: gui/tabs/weather_tab.py
    date changed: 06.08.2025

    Funktionsweise: Weather-Editor mit terrain_tab-ähnlicher UI-Struktur
    - Erbt von BaseMapTab für gemeinsame Features (70/30 Layout, Navigation, etc.)
    - UI-Struktur: Generate Button → LOD/Status-Display → Parameter-Panel → Statistics → Navigation (fixiert unten)
    - GenerationOrchestrator Integration mit StandardOrchestratorHandler
    - Real-time Status-Display für alle calculate-Schritte mit Progress und Validity-State
    - Live Climate-Preview und 3D Wind-Vector-Visualization

    UI-Layout:
    Control Panel (rechts, 30% Breite):
      - Generate Weather Button: Manual Generation-Trigger mit Priority-Handling
      - System Status Widget:
        * LOD-Level-Display für alle WeatherData-Komponenten mit Status-Icons
        * Step-Status: temperature → pressure_gradients → wind_field → moisture_transport → precipitation
        * Generation-Status: Queued/Generating/Completed/Failed mit Detail-Messages
        * GPU-Shader-Status für alle 8 Weather-Shader mit Performance-Metriken
      - Progress Bar: Step-progression mit aktueller Phase und Detail-Messages
      - Weather Parameters: Temperature-System und Wind-System Parameter mit Live-Preview
      - Climate Statistics: Temperature-Range, Niederschlags-Total, Wind-Stärke, Climate-Classification

    Hauptpanel (links, 70% Breite):
      - Oben: 2D/3D Toggle + Display-Optionen (Temperature/Precipitation/Humidity/Wind Field)
      - Mitte: Display-Widget mit Weather-Map-Visualization und 3D-Wind-Vectors-Overlay
      - Unten: Display-Tools und Orographic-Effects-Highlighting
    """


def water_tab():
    """
    Path: gui/tabs/water_tab.py
    date changed: 06.08.2025

    Funktionsweise: Water-Editor mit terrain_tab-ähnlicher UI-Struktur
    - Erbt von BaseMapTab für gemeinsame Features (70/30 Layout, Navigation, etc.)
    - UI-Struktur: Generate Button → LOD/Status-Display → Parameter-Panel → Statistics → Navigation (fixiert unten)
    - GenerationOrchestrator Integration mit StandardOrchestratorHandler
    - Real-time Status-Display für alle calculate-Schritte mit Progress und Validity-State
    - Live Hydrology-Statistics und GPU-Shader-Performance-Monitoring

    UI-Layout:
    Control Panel (rechts, 30% Breite):
      - Generate Hydrology Button: Manual Generation-Trigger mit Priority-Handling
      - System Status Widget:
        * LOD-Level-Display für alle WaterData-Komponenten mit Status-Icons
        * Step-Status: lake_detection → flow_network → manning_flow → erosion_sedimentation → soil_moisture → evaporation
        * Generation-Status: Queued/Generating/Completed/Failed mit Detail-Messages
        * Multi-Dependency-Status für alle 8 Required Input-Maps mit Recovery-Suggestions
        * GPU-Shader-Status für alle 8 Hydrology-Shader mit Performance-Metriken
      - Progress Bar: Step-progression mit aktueller Phase und Detail-Messages
      - Hydrology Parameters: Lake-Detection, Flow-Dynamics, Erosion/Sedimentation, Evaporation-Parameter
      - Hydrology Statistics: Water-Coverage, Flow-Dynamics, Erosion/Sedimentation-Balance, Mass-Conservation-Status

    Hauptpanel (links, 70% Breite):
      - Oben: 2D/3D Toggle + Display-Optionen (Water Depth/Flow/Erosion/Soil Moisture/etc.)
      - Mitte: Display-Widget mit Water-System-Visualization und River-Network-Overlay
      - Unten: Display-Tools und 3D-Terrain-Overlay mit Erosion-Highlighting
    """


def biome_tab():
    """
    Path: gui/tabs/biome_tab.py
    date changed: 06.08.2025

    Funktionsweise: Biome-Editor mit terrain_tab-ähnlicher UI-Struktur und Export-Features
    - Erbt von BaseMapTab für gemeinsame Features (70/30 Layout, Navigation, etc.)
    - UI-Struktur: Generate Button → LOD/Status-Display → Parameter-Panel → Statistics → Export → Navigation (fixiert unten)
    - GenerationOrchestrator Integration mit StandardOrchestratorHandler
    - Real-time Status-Display für alle calculate-Schritte mit Progress und Validity-State
    - Live Biome-Preview und World-Statistics mit Export-Funktionalität
    - Komplexeste Dependencies: 5 Input-Maps mit Optional-Fallback-Handling

    UI-Layout:
    Control Panel (rechts, 30% Breite):
      - Generate Biomes Button: Manual Generation-Trigger mit Priority-Handling
      - System Status Widget:
        * LOD-Level-Display für alle BiomeData-Komponenten mit Status-Icons
        * Step-Status: base_classification → super_biome_overrides → proximity_biomes → elevation_biomes → supersampling
        * Generation-Status: Queued/Generating/Completed/Failed mit Detail-Messages
        * Multi-Dependency-Status für alle 5 Input-Maps (3 required, 2 optional) mit Fallback-Info
        * Supersampling-Status: Zeigt ob 2x2 Enhancement aktiv (nur LOD256+)
      - Progress Bar: Step-progression mit aktueller Phase und Detail-Messages
      - Biome Parameters: Climate-Weighting, Water-Features, Elevation-Zones, Edge-Softness mit Live-Preview
      - Biome Classification Statistics: Base-Biome-Distribution, Super-Biome-Override-Count, Diversity-Index, Climate-Classification
      - World Export Widget: PNG-Maps, JSON-Data, 3D-OBJ-Terrain Export mit Options

    Hauptpanel (links, 70% Breite):
      - Oben: 2D/3D Toggle + Display-Optionen (Base Biomes/Super Biomes/Override Mask)
      - Mitte: Display-Widget mit Biome-Map-Visualization und Multi-Layer-Overlays (Settlements/Rivers/Contours)
      - Unten: Display-Tools und Biome-Legend-Dialog mit allen 26 Biome-Definitionen

    Besondere Features:
      - Biome Legend Dialog: Übersicht aller 15 Base-Biomes + 11 Super-Biomes mit Definitionen
      - World Statistics Widget: Comprehensive World-Overview aller Generator-Outputs
      - Export-System: Multi-Format-Export (PNG/JSON/OBJ) mit Parameter-Sets für Reproduzierbarkeit
      - Optional Dependencies Handling: Intelligente UI-Anpassung wenn Water/Settlement-Generatoren nicht gelaufen sind
      - Live Parameter Validation: Real-time Validation-Messages mit Error-Recovery-Suggestions
    """

def overview_tab():
    """
    Path: gui/tabs/overview_tab.py

    Funktionsweise: Finale Welt-Übersicht und Export
    - High-Quality Rendering aller Generator-Outputs
    - Export in verschiedene Formate (PNG, OBJ, JSON)
    - Welt-Statistiken und Zusammenfassung
    - Parameter-Set Export für Reproduzierbarkeit

    Kommunikationskanäle:
    - Input: Alle Generator-Outputs von data_manager
    - Export: Datei-Ausgabe in verschiedene Formate
    - Display: Finale High-Resolution Welt-Darstellung
    """


def error_handler():
    """
    Path: gui/utils/error_handler.py

    Funktionsweise: Zentrale Fehlerbehandlung für gesamte Anwendung
    - Exception-Catching und User-friendly Error-Messages
    - Log-System für Debugging
    - Fallback-Strategien bei Core-Module Fehlern
    - Recovery-Mechanismen für Corrupted States

    Features:
    - @error_handler Decorator für kritische Methoden
    - Automatic Error-Reporting mit Context-Information
    - User-Notification System mit Resolution-Suggestions
    """


def performance_handler():
    """
    Path: gui/utils/performance_handler.py

    Funktionsweise: Performance-Monitoring und Optimierung
    - @performance_tracked Decorator für Method-Timing
    - @debounced_method Decorator für Live-Update Optimization
    - Memory-Usage Tracking für große Arrays
    - Automatic LOD-Selection basierend auf Performance

    Features:
    - Real-time Performance-Metrics Display
    - Automatic Performance-Tuning (LOD-Adjustment)
    - Memory-Leak Detection für lange Sessions
    """


def map_display_2d():
    """
    Path: gui/widgets/map_display_2d.py

    Funktionsweise: 2D Map-Rendering mit Matplotlib Integration
    - Heightmap-Visualization mit Contour-Lines
    - Multi-Layer Rendering (Terrain + Overlays)
    - Interactive Tools (Zoom, Pan, Measure)
    - Export-Quality Rendering für High-Resolution Output

    Kommunikationskanäle:
    - Input: numpy arrays von data_manager
    - Config: gui_default.CanvasSettings für Render-Parameter
    - Output: Interactive 2D-Display mit Tool-Integration
    """


def map_display_3d():
    """
    Path: gui/widgets/map_display_3d.py

    Funktionsweise: 3D Terrain-Rendering mit OpenGL
    - Real-time 3D Heightmap-Mesh Rendering
    - Camera-Controls (Orbit, Zoom, Pan)
    - Multi-Layer Support (Terrain + Settlements + Water + Wind-Vectors)
    - Lighting und Shading für realistisches Appearance

    Kommunikationskanäle:
    - Input: heightmaps und Overlay-Data von data_manager
    - GPU: shader_manager für optimierte Rendering-Pipeline
    - Controls: Camera-Settings aus gui_default.py
    """


def widgets():
    """
    Path: gui/widgets/widgets.py

    Funktionsweise: Wiederverwendbare UI-Komponenten
    - BaseButton mit konfigurierbarem Styling
    - ParameterSlider mit value_default.py Integration
    - StatusIndicator für Input-Dependencies
    - ProgressBar für Tab-Navigation
    - AutoSimulationPanel für alle Tabs

    Kommunikationskanäle:
    - Config: gui_default.py für Styling, value_default.py für Parameter-Ranges
    - Signals: Standard Qt-Signals für Value-Changes
    - Features: Consistent Styling, Validation, Error-States
    """