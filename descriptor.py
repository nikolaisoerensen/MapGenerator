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
    date_changed: 18.08.2025

    Parameter Input:
    - map_size, amplitude, octaves, frequency, persistence, lacunarity, redistribute_power, map_seed

    Output:
    - TerrainData-Objekt mit heightmap, slopemap, shadowmap, validity_state und LOD-Metadaten
    - DataLODManager-Storage für nachfolgende Generatoren (geology, weather, water, biome, settlement)

    LOD-System (Numerisch):
    - lod_level 1: 32x32 (1 Sonnenwinkel - Mittag)
    - lod_level 2: 64x64 (3 Sonnenwinkel - Vormittag/Mittag/Nachmittag)
    - lod_level 3: 128x128 (5 Sonnenwinkel + Morgen/Abend)
    - lod_level 4: 256x256 (7 Sonnenwinkel + Dämmerung)
    - lod_level 5+: bis map_size erreicht, 7 Sonnenwinkel konstant

    Validity-System:
    - Parameter-Change-Detection: Erkennt signifikante Parameter-Änderungen für Cache-Invalidation
    - LOD-Consistency-Checks: Stellt sicher dass alle LOD-Level mit gleichen Parametern berechnet sind
    - Dependency-Invalidation: Invalidiert nachgelagerte Generatoren bei kritischen Parameter-Änderungen
    - State-Tracking: validity_flags in TerrainData für jeden LOD-Level und Output-Type

    Fallback-System (3-stufig):
    - GPU-Shader (Optimal): ShaderManager für parallele Multi-Octave-Noise-Berechnung
    - CPU-Fallback (Gut): Optimierte NumPy-Implementierung mit Multiprocessing
    - Simple-Fallback (Minimal): Direkte Implementierung im Generator, wenige Zeilen

    Graceful-Degradation-Strategy:
    1. ShaderManager-Unavailability: Automatic CPU-Fallback ohne User-Intervention
    2. CPU-Processing-Errors: Simple-Fallback mit reduzierter Qualität aber garantierter Completion
    3. Memory-Constraints: LOD-Size-Reduction und progressive Quality-Degradation
    4. Parameter-Invalidity: Default-Value-Substitution mit Warning-Logging
    5. Critical-Failures: Minimal-Result-Generation (flat heightmap) für System-Continuity

    Error-Recovery-Mechanisms:
    - Exception-Safe-Operations: Alle kritischen Methoden mit try/except und Fallback-Returns
    - Resource-Cleanup: Garantierte Memory/GPU-Resource-Freigabe auch bei Exceptions
    - State-Consistency: Partial-Results werden validiert und repariert vor DataLODManager-Storage
    - User-Notification: Error-Details an DataLODManager für UI-Status-Display, aber keine Generation-Stopps

    Performance-Characteristics:
    - GPU-Accelerated: 10-50x speedup für Multi-Octave-Noise bei großen LODs
    - CPU-Optimized: Vectorized NumPy-Operations, Multiprocessing für Parallelisierung
    - Memory-Efficient: LOD-based progressive allocation, automatic cleanup bei LOD-Completion
    - Cache-Friendly: Parameter-Hash-basierte Result-Caching, LOD-Interpolation für upgrades

    Klassen:
    TerrainData
        Funktionsweise: Container für alle Terrain-Daten mit Validity-System und Cache-Management
        Aufgabe: Speichert Heightmap, Slopemap, Shadowmap mit LOD-Level, Validity-State und Parameter-Hash
        Attribute: heightmap, slopemap, shadowmap, lod_level, actual_size, validity_state, parameter_hash, calculated_sun_angles, parameters
        Validity-Methods: is_valid(), invalidate(), validate_against_parameters(), get_validity_summary()

    BaseTerrainGenerator
        Funktionsweise: Hauptklasse für Terrain-Generierung mit numerischem LOD-System und Manager-Integration
        Aufgabe: Koordiniert alle Terrain-Generierungsschritte, verwaltet Parameter und LOD-Progression
        External-Interface: calculate_heightmap(parameters, lod_level) - wird von GenerationOrchestrator aufgerufen
        Internal-Methods: _coordinate_generation(), _validate_parameters(), _create_terrain_data()
        Manager-Integration: DataLODManager für Storage, ShaderManager für Performance-Optimierung
        Threading: Läuft in GenerationOrchestrator-Background-Threads mit LOD-Progression
        Error-Handling: Graceful Degradation bei Shader/Generator-Fehlern, vollständige Fallback-Kette

    SimplexNoiseGenerator
        Funktionsweise: Erzeugt OpenSimplex-Noise mit 3-stufiger Fallback-Strategie (siehe Fallback-System)
        Aufgabe: Basis-Noise-Funktionen für Heightmap-Generation mit Performance-Optimierung
        Methoden: noise_2d(), multi_octave_noise(), ridge_noise()
        LOD-Optimiert: generate_noise_grid() für Batch-Verarbeitung, interpolate_existing_grid() für LOD-Upgrades
        Spezifische Fallbacks:
          - GPU-Optimal: request_noise_generation() für parallele Multi-Octave-Berechnung
          - CPU-Fallback: Optimierte NumPy-Implementierung mit vectorization
          - Simple-Fallback: Direkte Random-Noise-Generation (5-10 Zeilen)

    ShadowCalculator
        Funktionsweise: Berechnet Verschattung mit Raycasts für LOD-spezifische Sonnenwinkel
        Aufgabe: Erstellt shadowmap (konstant 64x64) für Weather-System und visuelle Darstellung
        Methoden: calculate_shadows(heightmap, lod_level, parameters), raycast_shadow(), combine_shadow_angles()
        LOD-System: get_sun_angles_for_lod() - 1,3,5,7 Sonnenwinkel je nach LOD-Level
        Progressive-Enhancement: Berechnet nur neue Sonnenwinkel bei LOD-Upgrades, kombiniert mit bestehenden Shadows
        Spezifische Fallbacks:
          - GPU-Optimal: Parallele Raycast-Berechnung für alle Sonnenwinkel
          - CPU-Fallback: Optimierte CPU-Raycast-Implementierung
          - Simple-Fallback: Einfache Height-Difference-Shadow-Approximation

    SlopeCalculator
        Funktionsweise: Berechnet Steigungsgradienten (dz/dx, dz/dy) aus Heightmap
        Aufgabe: Erstellt slopemap für Geology-Generator und visuelle Darstellung
        Methoden: calculate_slopes(heightmap, parameters), gradient_magnitude(), validate_slopes()
        Output-Format: 3D-Array (H,W,2) mit dz/dx und dz/dy Komponenten
        Validation: Gradient-Range-Checks und Consistency mit heightmap-Shape
        Spezifische Fallbacks:
          - GPU-Optimal: Parallele Gradient-Berechnung mit GPU-Compute-Shader
          - CPU-Fallback: NumPy gradient() mit optimierten Parametern
          - Simple-Fallback: Einfache Finite-Difference-Approximation

    Integration und Datenfluss:
    GenerationOrchestrator → BaseTerrainGenerator.calculate_heightmap(parameters, lod_level)
                          → SimplexNoiseGenerator.generate_noise() → ShaderManager-Request
                          → SlopeCalculator.calculate_slopes() → ShaderManager-Request
                          → ShadowCalculator.calculate_shadows() → ShaderManager-Request
                          → TerrainData-Assembly → DataLODManager.store_result()

    Output-Datenstrukturen:
    - heightmap: 2D numpy.float32 array, Elevation in Metern
    - slopemap: 3D numpy.float32 array (H,W,2), dz/dx und dz/dy Gradienten
    - shadowmap: 3D numpy.float32 array (H,W,angles), Shadow-Values [0-1] für 1-7 Sonnenwinkel
    - TerrainData: Validity-State, Parameter-Hash, LOD-Metadata, Performance-Stats
    """

def geology_generator():
    """
    Path: core/geology_generator.py

    Funktionsweise: Geologische Schichten und Gesteinstypen mit DataLODManager-Integration und 3-stufigem Fallback-System
    - GeologyGenerator koordiniert geologische Simulation mit numerischem LOD-System
    - Rock-Type Klassifizierung (sedimentary, metamorphic, igneous) basierend auf geologischer Simulation
    - Mass-Conservation-System (R+G+B=255) für Erosions-/Sedimentations-Kompatibilität
    - Hardness-Map-Generation für Water-Generator und nachfolgende Systeme

    Parameter Input:
    - sedimentary_hardness, igneous_hardness, metamorphic_hardness [0-100]
    - ridge_warping, bevel_warping, metamorphic_foliation, metamorphic_folding, igneous_flowing [0.0-1.0]

    Dependencies (über DataLODManager):
    - heightmap (von terrain_generator)
    - slopemap (von terrain_generator)

    Output:
    - GeologyData-Objekt mit rock_map, hardness_map, validity_state und LOD-Metadaten
    - DataLODManager-Storage für nachfolgende Generatoren (water, biome, settlement)

    LOD-System (Numerisch):
    - LOD-Progression entsprechend Terrain-LOD-System (32→64→128→256→512→1024→2048)
    - Progressive Enhancement: Deformation-Detail und Zone-Complexity steigen mit LOD-Level
    - Geological-Zones: Detaillierung von einfachen Height-Zones zu komplexen Multi-Layer-Noise

    Fallback-System (3-stufig):
    - GPU-Shader (Optimal): Parallele Noise-Generation und geologische Zone-Berechnung
    - CPU-Fallback (Gut): Optimierte NumPy-Implementierung mit vectorization
    - Simple-Fallback (Minimal): Höhen-basierte Linear-Classification ohne Noise-Zones

    Rock-Distribution-Algorithmus:
    1. Höhen-basierte Basis-Verteilung mit verzerrter Noise-Funktion
    2. Steile Hänge: Härtere Gesteine (Igneous/Metamorphic) werden bevorzugt
    3. Geologische Zonen: Drei Simplex-Funktionen für Sedimentary (>0.2), Metamorphic (>0.6), Igneous (>0.8)
    4. RGB-Gewichtung: R=Sedimentary, G=Igneous, B=Metamorphic mit R+G+B=255 Massenerhaltung
    5. Tektonische Deformation: Ridge/Bevel-Warping, Metamorphic-Foliation/Folding, Igneous-Flowing

    Graceful-Degradation-Strategy:
    1. ShaderManager-Unavailability: Automatic CPU-Fallback für alle geologischen Operationen
    2. Input-Data-Problems: Default-Rock-Distribution basierend auf Height-Percentiles
    3. Memory-Constraints: LOD-Size-Reduction und simplified Deformation-Processing
    4. Parameter-Invalidity: Default-Hardness-Values und minimal Deformation-Effects
    5. Critical-Failures: Uniform-Rock-Distribution (33% je Typ) für System-Continuity

    Error-Recovery-Mechanisms:
    - Exception-Safe-Operations: Alle Geology-Operations mit try/except und Fallback-Returns
    - Input-Data-Repair: Corrupt heightmap/slopemap-Pixel werden interpoliert oder auf Defaults gesetzt
    - Mass-Conservation-Enforcement: Garantierte R+G+B=255 auch bei Calculation-Errors
    - State-Consistency: Partial-Results werden validiert und repariert vor DataLODManager-Storage

    Performance-Characteristics:
    - GPU-Accelerated: 5-20x speedup für Multi-Zone-Noise und Deformation-Processing
    - CPU-Optimized: Vectorized NumPy-Operations, optimierte SciPy-Interpolation
    - Memory-Efficient: LOD-based progressive allocation, Input-Data-Sharing mit Terrain-Generator
    - Cache-Friendly: Parameter-Hash-basierte Result-Caching, Height-Data-Reuse zwischen LODs

    Klassen:
    GeologyGenerator
        Funktionsweise: Hauptklasse für geologische Schichten und Gesteinstyp-Verteilung
        Aufgabe: Koordiniert Gesteinsverteilung, Härte-Berechnung und Mass-Conservation
        External-Interface: calculate_geology(heightmap, slopemap, parameters, lod_level) - wird von GenerationOrchestrator aufgerufen
        Internal-Methods: _coordinate_geology_generation(), _validate_input_data(), _create_geology_data()
        Dependencies: Prüft heightmap/slopemap-Verfügbarkeit und Shape-Consistency über DataLODManager
        Error-Handling: Graceful Degradation bei Input-Inconsistencies, vollständige Fallback-Kette

    RockTypeClassifier
        Funktionsweise: Klassifiziert Gesteinstypen basierend auf Höhe, Steigung und geologischen Zonen
        Aufgabe: Erstellt rock_map mit RGB-Kanälen für drei Gesteinstypen
        Methoden: classify_by_elevation(), apply_slope_hardening(), blend_geological_zones()
        Geological-Zones: Drei unabhängige Simplex-Noise-Layers für realistische geologische Verteilung
        Spezifische Fallbacks:
          - GPU-Optimal: Parallele Multi-Zone-Noise-Berechnung und Blend-Operations
          - CPU-Fallback: NumPy-vectorized Zone-Classification mit optimierten Noise-Functions
          - Simple-Fallback: Linear Height-based Classification (Sedimentary: 0-30%, Metamorphic: 30-70%, Igneous: 70-100%)

    MassConservationManager
        Funktionsweise: Stellt sicher dass R+G+B immer 255 ergibt für Massenerhaltung
        Aufgabe: Verwaltet Gesteins-Massenverteilung für spätere Erosion im Water-Generator
        Methoden: normalize_rock_masses(), validate_conservation(), redistribute_masses()
        Conservation-Algorithm: Proportionale Skalierung wenn R+G+B != 255, Fallback zu (85,85,85) bei R+G+B=0
        Output-Validation: Range-Checks [0-255], Consistency mit Rock-Map, NaN-Detection und Repair

    HardnessCalculator
        Funktionsweise: Berechnet Hardness-Map aus Rock-Map und Hardness-Parametern
        Aufgabe: Erstellt hardness_map für Water-Generator Erosions-Simulation
        Methoden: calculate_hardness_map(), validate_hardness_ranges(), apply_hardness_modifiers()
        Formula: hardness_map(x,y) = (R*sed_hardness + G*ign_hardness + B*met_hardness) / 255
        Output-Validation: Range-Checks [0-100], Consistency mit Rock-Map, NaN-Detection und Repair
        Spezifische Fallbacks:
          - GPU-Optimal: Parallele Weighted-Sum-Berechnung für alle Pixel
          - CPU-Fallback: NumPy-Dot-Product-Operations für effiziente Hardness-Calculation
          - Simple-Fallback: Durchschnitts-Hardness-Wert für gesamte Map

    TectonicDeformationProcessor
        Funktionsweise: Appliziert tektonische Verformung auf geologische Verteilung
        Aufgabe: Ridge/Bevel-Warping, Metamorphic-Foliation/Folding, Igneous-Flowing für realistische Geologie
        Methoden: apply_ridge_warping(), apply_bevel_warping(), process_metamorphic_effects(), process_igneous_flowing()
        Deformation-Parameters: ridge_warping, bevel_warping, metamorphic_foliation/folding, igneous_flowing
        Spezifische Fallbacks:
          - GPU-Optimal: Parallele Deformation-Field-Calculation und Texture-Warping
          - CPU-Fallback: SciPy-based Deformation mit optimierten Interpolation-Methods
          - Simple-Fallback: Lineare Height-based Deformation ohne komplexe Warping

    Integration und Datenfluss:
    GenerationOrchestrator → GeologyGenerator.calculate_geology(heightmap, slopemap, parameters, lod_level)
                          → RockTypeClassifier.classify_rock_types() → ShaderManager-Request
                          → TectonicDeformationProcessor.apply_deformation() → ShaderManager-Request
                          → MassConservationManager.normalize_masses() → ShaderManager-Request
                          → HardnessCalculator.calculate_hardness() → ShaderManager-Request
                          → GeologyData-Assembly → DataLODManager.store_result()

    Output-Datenstrukturen:
    - rock_map: 3D numpy.uint8 array (H,W,3), RGB-Kanäle für Sedimentary/Igneous/Metamorphic mit R+G+B=255
    - hardness_map: 2D numpy.float32 array, Gesteinshärte-Werte [0-100] für Erosions-Simulation
    - GeologyData: Validity-State, Parameter-Hash, LOD-Metadata, Mass-Conservation-Info, Performance-Stats
    """

def settlement_generator():
    """
    Path: core/settlement_generator.py
    
    Funktionsweise: Intelligente Settlement-Platzierung
    - Terrain-Suitability Analysis (Steigung, Höhe, Wasser-Nähe)
        Suitability-Map wird mit diesen Einflussgrößen erzeugt.
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
    
    data_manager Input:
        - heightmap
        - slopemap
        - water_map
    
    Parameter Input:
        - map_seed
        - settlements, landmarks, roadsites, plotnodes: number of each type
        - civ_influence_decay: Influence around Locationtypes decays of distance
        - terrain_factor_villages: terrain influence on settlement suitability
        - road_slope_to_distance_ratio: rather short roads or steep roads
        - landmark_wilderness: wilderness area size by changing cutoff-threshold
        - plotsize: how much accumulated civ-value to form plot
    
    Output:
        - settlement list
        - landmark list
        - roadsite list
        - plot_map
        - civ_map
    
    Klassen:
    SettlementGenerator  
        Funktionsweise: Hauptklasse für intelligente Settlement-Platzierung und Civilization-Mapping
        Aufgabe: Koordiniert alle Settlement-Aspekte und erstellt civ_map
        Methoden: generate_settlements(), create_road_network(), place_landmarks(), generate_plots()
    
    TerrainSuitabilityAnalyzer
        Funktionsweise: Analysiert Terrain-Eignung für Settlements basierend auf Steigung, Höhe, Wasser-Nähe
        Aufgabe: Erstellt Suitability-Map für optimale Settlement-Platzierung
        Methoden: analyze_slope_suitability(), calculate_water_proximity(), evaluate_elevation_fitness()
    
    PathfindingSystem   
        Funktionsweise: Findet Wege geringsten Widerstands zwischen Settlements für Straßen
        Aufgabe: Erstellt realistische Straßenverbindungen mit Spline-Interpolation
        Methoden: find_least_resistance_path(), apply_spline_smoothing(), calculate_movement_cost()
    
    CivilizationInfluenceMapper    
        Funktionsweise: Berechnet civ_map durch radialen Decay von Settlement/Road/Landmark-Punkten
        Aufgabe: Erstellt realistische Zivilisations-Verteilung mit Decay-Kernels
        Methoden: apply_settlement_influence(), calculate_road_influence(), apply_decay_kernel()
    
    PlotNodeSystem    
        Funktionsweise: Generiert Plotnodes mit Delaunay-Triangulation und Grundstücks-Bildung
        Aufgabe: Erstellt Grundstücks-System für späteres Gameplay
        Methoden: generate_plot_nodes(), create_delaunay_triangulation(), merge_to_plots(), optimize_node_positions()
    """

def weather_generator():
    """
    Path: core/weather_generator.py

    Funktionsweise: Dynamisches Wetter- und Feuchtigkeitssystem
    - Terrain-basierte Temperaturberechnung (Altitude, Sonneneinstrahlung, Breitengrad (Input: heightmap, shade_map) der Luft.
    - Windfeld-Simulation mit Ablenkung um Berge und Abbremsen durch Geländegradienten-Analyse (slopemap).
        - Konvektion: Wärmetransport durch Windströme
        - Orographischer Regen: Feuchtigkeitstransport
        - Precipation durch Abkühlen/Aufsteigen der Luft
        - Evaporation durch Bodenfeuchtigkeit (Input: soil_moist_map)
        - OpenSimplex-Noise für  Variation in Temperatur-, Feuchte- und Luftdruck an Kartengrenze
        - Regen wird ausgelöst durch Erreichen eines Feuchtigkeitsgrenzwertes von über 1.0. Dieser berechnet sich aus rho_max = 5*exp(0.06*T) mit T aus temp_map und der Luftfeuchtigkeit in der jeweiligen Zelle.

    Parameter Input:
    - air_temp_entry (Lufttemperatur bei Karteneintritt)
    - solar_power (max. solare Gewinne (default 20°C))
    - altitude_cooling (Abkühlen der Luft pro 100 m Altitude(default 6°C))
    - thermic_effect (Thermische Verformung der Windvektoren durch shademap)
    - wind_speed_factor (Windgeschwindigkeit je Luftdruckdifferenz)
    - terrain_factor (Einfluss von Terrain auf Wind und Temperatur)
    - flow_direction, flow_accumulation (optional: Wasserfluss-Informationen)

    data_manager Input:
    - map_seed
    - heightmap
    - shade_map
    - soil_moist_map

    Output:
    - wind_map (Windvektoren (Richtung und Geschwindigkeit)) in m/s
    - temp_map (2D-Feld, Lufttemperatur) in °C
    - precip_map (2D-Feld, Regenfall) in gH20/m2
    - humid_map (2D-Feld, Luftfeuchte) in gH20/m3

    Beschreibung der Berg-Wind-Simulation
    Schritt 1: Temperaturberechnung
    Zunächst wird die Lufttemperatur für jede Zelle der Simulation basierend auf drei Hauptfaktoren berechnet. Die Altitude (z) führt zu einer starken Temperaturabnahme von 60°C pro Kilometer Höhe - das ist zehnmal stärker als in der Realität, um dramatischere Effekte zu erzielen. Die Sonneneinstrahlung I(x,y) wird aus einer speziell vorbereiteten Shademap importiert, die sechs verschiedene Sonnenwinkel über den Tag gewichtet kombiniert. Hohe Sonnenstände erhalten dabei mehr Gewichtung. Bei maximaler Sonneneinstrahlung (Wert 1) steigt die Temperatur um 10°C, bei minimaler Einstrahlung (Wert 0) fällt sie um 10°C. Der Breitengrad (y-Position) simuliert den Äquator-zu-Pol-Gradienten: An der Südseite der Karte (y=0) bleibt die Basistemperatur unverändert, während sie zur Nordseite hin (y=map_size) um 5°C ansteigt.

    Schritt 2: Windfeld-Grundsimulation
    Die eigentliche Windsimulation wird durch einen konstanten Druckgradienten von West nach Ost initialisiert. An der Westseite der Karte wird ein erhöhter Luftdruck angelegt, der nach Osten hin kontinuierlich abnimmt. Um natürlichere Strömungsmuster zu erzeugen, wird dieser Grunddruckgradient mit Simplex-Noise moduliert, wodurch turbulente Variationen entstehen. Die gesamte Welt wird in ein regelmäßiges Gitter von Simulationszellen unterteilt, wobei jede Zelle eine Luftsäule vom Gelände bis zur maximalen Simulationshöhe repräsentiert. Ein spezieller GPU-Shader berechnet dann für jede Zelle die Druckverteilung und die daraus resultierenden Windgeschwindigkeiten in alle drei Raumrichtungen.

    Schritt 3: Geländeinteraktion
    Das Windfeld interagiert dynamisch mit der Geländetopographie. Düseneffekte entstehen automatisch in engen Tälern, wo die Windgeschwindigkeit aufgrund der Kontinuitätsgleichung zunimmt. Blockierungseffekte treten auf, wenn Luftmassen auf Berghänge treffen - der Wind wird dann entweder nach oben abgelenkt oder um die Hindernisse herumgeleitet. Orographische Hebung an Luvhängen führt zu Aufwinden, während Leewirbel auf der windabgewandten Seite von Bergen entstehen. Die Temperaturunterschiede zwischen sonnenexponierten und schattigen Hängen verstärken diese Effekte zusätzlich durch thermisch induzierte Hangwinde.

    Schritt 4: Numerische Integration
    Die zeitliche Entwicklung des Windfeldes wird durch die Navier-Stokes-Gleichungen für inkompressible Strömungen gesteuert. Dabei werden Advektionsterme (Wind transportiert sich selbst), Druckgradientenkräfte und Viskositätseffekte berücksichtigt. Die Kontinuitätsgleichung sorgt dafür, dass die Massenerhaltung eingehalten wird, was besonders wichtig ist, um realistische Strömungsmuster um Geländehindernisse zu erzeugen. Jeder Simulationsschritt aktualisiert sowohl die Windgeschwindigkeiten als auch die Druckverteilung konsistent.

    Benötigte Shader für die Berg-Wind-Simulation
    Shader 1: Temperatur-Berechnung (temperatureCalculation.frag)
    Dieser Shader berechnet die Lufttemperatur für jede Zelle basierend auf den Eingabeparametern. Er liest die Heightmap aus, um die Höhenabhängige Abkühlung (altitude_cooling) anzuwenden, sampelt die Shade-Map für die solare Erwärmung (solar_power) und berücksichtigt den Breitengrad-Gradienten über die Y-Position. OpenSimplex-Noise wird verwendet, um natürliche Temperaturschwankungen an den Kartengrenzen zu erzeugen. Der Shader gibt ein 2D-Temperaturfeld aus, das als Grundlage für alle thermischen Berechnungen dient.

    Shader 2: Windfeld-Basis (windFieldGeneration.frag)
    Dieser Shader erzeugt das grundlegende Windfeld durch Druckgradienten von West nach Ost. Er berechnet Druckdifferenzen und konvertiert diese über den wind_speed_factor in Windgeschwindigkeiten. OpenSimplex-Noise moduliert die Druckverteilung für natürliche Turbulenz. Der Shader berücksichtigt Geländeablenkung durch die Slope-Map und den terrain_factor, wodurch Wind um Berge herumgeleitet und in Tälern beschleunigt wird. Die Ausgabe ist ein 2D-Vektorfeld mit horizontalen Windkomponenten.

    Shader 3: Thermische Konvektion (thermalConvection.frag)
    Dieser Shader berechnet thermisch induzierte Windkomponenten basierend auf Temperaturdifferenzen. Er liest das Temperaturfeld aus dem vorherigen Shader und berechnet lokale Temperaturgradienten. Aufwinde entstehen über warmen Bereichen (hohe Shade-Map-Werte), Abwinde über kalten Bereichen. Der thermic_effect-Parameter steuert die Stärke dieser thermischen Verformung. Der Shader modifiziert das bestehende Windfeld durch Überlagerung der konvektiven Komponenten und erzeugt realistische Hangwind-Systeme.

    Shader 4: Feuchtigkeits-Transport (moistureTransport.frag)
    Dieser Shader simuliert den Transport von Wasserdampf durch das Windfeld. Er liest die Soil-Moisture-Map für die Evaporation, das aktuelle Feuchtigkeitsfeld und das Windfeld für den Advektionstransport. Evaporation wird basierend auf Bodenfeuchte und lokaler Temperatur berechnet. Der Shader implementiert eine Advektionsgleichung, die Wasserdampf entsprechend der Windrichtung und -geschwindigkeit transportiert. Diffusionseffekte glätten extreme Feuchtigkeitsgradienten für realistische Verteilungen.

    Shader 5: Niederschlags-Berechnung (precipitationCalculation.frag)
    Dieser Shader bestimmt, wo und wie viel Niederschlag fällt. Er berechnet die maximale Wasserdampfdichte rho_max = 5*exp(0.06*T) für jede Zelle basierend auf der lokalen Temperatur. Wenn die relative Luftfeuchtigkeit den Wert 1.0 überschreitet, wird Niederschlag ausgelöst. Der Shader berücksichtigt orographische Hebung durch Windgeschwindigkeit und Geländesteigung. Latente Wärmefreisetzung bei der Kondensation wird zurück an den Temperatur-Shader gegeben. Die Ausgabe ist ein 2D-Niederschlagsfeld.

    Shader 6: Orographische Effekte (orographicEffects.frag)
    Dieser spezialisierte Shader berechnet geländeinduzierte Wetterphänomene. Er analysiert Windrichtung relativ zur Geländeorientierung, um Luv- und Lee-Bereiche zu identifizieren. Staueffekte werden an Luvhängen durch verstärkte Aufwinde simuliert. Föhneffekte entstehen durch trockenadiabatische Erwärmung auf der Leeseite. Der Shader modifiziert sowohl Temperatur als auch Feuchtigkeit basierend auf der Geländeinteraktion und erzeugt charakteristische Regenschatten-Muster.

    Shader 7: System-Integration (weatherIntegration.frag)
    Dieser zentrale Shader führt alle Komponenten zusammen und berechnet die zeitliche Entwicklung des Wettersystems. Er implementiert Rückkopplungsschleifen zwischen Temperatur, Wind und Feuchtigkeit. Konvergenz- und Divergenzzonen werden identifiziert und verstärken lokale Wetterphänomene. Der Shader aktualisiert alle Felder konsistent und sorgt für Massenerhaltung bei Feuchtigkeit und Energieerhaltung bei thermischen Prozessen. Er koordiniert die Ausgabe der finalen wind_map, temp_map, precip_map und humid_map.

    Shader 8: Boundary-Conditions (boundaryConditions.frag)
    Dieser Shader verwaltet die Randbedingungen an den Kartengrenzen. Er implementiert kontinuierliche Wetterfront-Einträge mit den konfigurierten Parametern (air_temp_entry, etc.). OpenSimplex-Noise erzeugt realistische Wetterfront-Variationen. Der Shader sorgt für konsistente Übergänge zwischen den Kartenrändern und dem Innenbereich und verhindert künstliche Artefakte an den Grenzen. Periodische Randbedingungen können für endlose Karten implementiert werden.

    Zusätzliche Utility-Shader:
    Gradient-Berechnung (gradientCalculation.frag): Berechnet Höhen- und Temperaturgradienten für die anderen Shader
    Noise-Generation (noiseGeneration.frag): Erzeugt OpenSimplex-Noise-Felder für natürliche Variationen
    Debug-Visualisierung (debugVisualization.frag): Stellt verschiedene Datenfelder für die Entwicklung visuell dar

    Alle Shader arbeiten im Ping-Pong-Verfahren zwischen mehreren Framebuffern, um zeitliche Entwicklung zu simulieren und Rückkopplungseffekte zu ermöglichen.

    Klassen:
    WeatherSystemGenerator
        Funktionsweise: Hauptklasse für dynamisches Wetter- und Feuchtigkeitssystem
        Aufgabe: Koordiniert Temperatur, Wind, Niederschlag und Luftfeuchtigkeit
        Methoden: generate_weather_system(), update_weather_cycle(), integrate_atmospheric_effects()

    TemperatureCalculator
        Funktionsweise: Berechnet Lufttemperatur basierend auf Altitude, Sonneneinstrahlung und Breitengrad
        Aufgabe: Erstellt temp_map als Grundlage für alle Weather-Berechnungen
        Methoden: calculate_altitude_cooling(), apply_solar_heating(), add_latitude_gradient()

    WindFieldSimulator
        Funktionsweise: Simuliert Windfeld mit Ablenkung um Berge und Geländegradienten-Analyse
        Aufgabe: Erstellt wind_map für Konvektion und Orographischen Regen
        Methoden: simulate_pressure_gradients(), apply_terrain_deflection(), calculate_thermal_effects()

    PrecipitationSystem
        Funktionsweise: Berechnet Niederschlag durch Feuchtigkeits-Transport und Kondensation
        Aufgabe: Erstellt precip_map basierend auf Luftfeuchtigkeit und Geländeeffekten
        Methoden: calculate_orographic_precipitation(), simulate_moisture_transport(), trigger_precipitation_events()

    AtmosphericMoistureManager
        Funktionsweise: Verwaltet Luftfeuchtigkeit durch Evaporation und Konvektion
        Aufgabe: Erstellt humid_map mit realistischer Feuchtigkeits-Verteilung
        Methoden: calculate_evaporation(), transport_moisture(), apply_humidity_diffusion()
    """

def water_generator():
    """
    Path: core/water_generator.py

    Funktionsweise: Dynamisches Hydrologiesystem mit Erosion und Sedimentation
    - Lake-Detection durch Jump Flooding Algorithm für parallele Senken-Identifikation
    - Flussnetzwerk-Aufbau durch Steepest Descent mit Upstream-Akkumulation
    - Strömungsberechnung nach Manning-Gleichung mit adaptiven Querschnitten
    - Bodenfeuchtigkeit durch Gaussian-Diffusion von Gewässern
    - Stream Power Erosion mit Hjulström-Sundborg Transport
    - Realistische Sedimentation mit Transportkapazitäts-Überschreitung
    - Evaporation nach Penman-Gleichung mit Wind- und Temperatureffekten

    Parameter Input:
    - lake_volume_threshold (Mindestvolumen für Seebildung, default 0.1m)
    - rain_threshold (Niederschlagsschwelle für Quellbildung, default 5.0 gH2O/m²)
    - manning_coefficient (Rauheitskoeffizient für Fließgeschwindigkeit, default 0.03)
    - erosion_strength (Erosionsintensität-Multiplikator, default 1.0)
    - sediment_capacity_factor (Transportkapazitäts-Faktor, default 0.1)
    - evaporation_base_rate (Basis-Verdunstungsrate, default 0.002 m/Tag)
    - diffusion_radius (Bodenfeuchtigkeit-Ausbreitungsradius, default 5.0 Pixel)
    - settling_velocity (Sediment-Sinkgeschwindigkeit, default 0.01 m/s)

    data_manager Input:
    - map_seed
    - heightmap
    - slopemap
    - hardness_map (Gesteinshärte-Verteilung)
    - rock_map (RGB-Feld, Gesteinsmassen - R=Sedimentary, G=Igneous, B=Metamorphic mit R+G+B=255)
    - precip_map (2D-Feld, Niederschlag in gH2O/m²)
    - temp_map (2D-Feld, Lufttemperatur in °C)
    - wind_map (2D-Feld, Windvektoren in m/s)
    - humid_map (2D-Feld, Luftfeuchtigkeit in gH2O/m³)

    Output:
    - water_map (2D-Feld, Gewässertiefen) in m
    - flow_map (2D-Feld, Volumenstrom) in m³/s
    - flow_speed (2D-Feld, Fließgeschwindigkeit) in m/s
    - cross_section (2D-Feld, Flusquerschnitt) in m²
    - soil_moist_map (2D-Feld, Bodenfeuchtigkeit) in %
    - erosion_map (2D-Feld, Erosionsrate) in m/Jahr
    - sedimentation_map (2D-Feld, Sedimentationsrate) in m/Jahr
    - rock_map_updated (RGB-Feld, Gesteinsmassen-Verteilung) - R=Sedimentary, G=Igneous, B=Metamorphic mit R+G+B=255
    - evaporation_map (2D-Feld, Verdunstung) in gH2O/m²/Tag
    - ocean_outflow (Skalär, Wasserabfluss ins Meer) in m³/s
    - water_biomes_map (2D-Array mit Wasser-Klassifikation: 0=kein Wasser, 1=Creek, 2=River, 3=Grand River, 4=Lake)

    Beschreibung der Hydrologischen Simulation

    Schritt 1: Lake-Detection und Floodfill
    Das System identifiziert zunächst alle lokalen Minima in der Heightmap als potenzielle Seestandorte. Der Jump Flooding Algorithm (JFA) wird verwendet, um diese Senken parallel zu füllen. JFA arbeitet in logarithmischer Zeit O(log n) und ist hochgradig parallelisierbar, da jeder Pixel unabhängig operiert. In der Initialisierungsphase markiert jedes lokale Minimum sich selbst als "Lake-Seed". In den folgenden Iterationen propagieren diese Seeds ihre Informationen mit exponentiell abnehmenden Sprungdistanzen (512, 256, 128, 64, ..., 1 Pixel). Jeder Pixel prüft dabei seine Nachbarn in der aktuellen Sprungdistanz und übernimmt den nächstgelegenen Seed, falls die Höhendifferenz positiv ist (Wasser kann dorthin fließen). Der lake_volume_threshold bestimmt dabei, wie viel Höhendifferenz nötig ist, um eine Senke als See zu klassifizieren.

    Schritt 2: Flussnetzwerk-Aufbau durch Steepest Descent
    Nach der See-Identifikation werden alle Niederschlagsquellen (Zellen mit precip > rain_threshold) mit ihren Zielen verknüpft. Der Steepest Descent Algorithmus folgt dabei dem steilsten Gradienten bergab bis zum nächsten lokalen Minimum oder Kartenrand. Anders als der klassische Dijkstra-Algorithmus ist diese Methode vollständig parallelisierbar, da jede Zelle unabhängig ihre optimale Fließrichtung berechnen kann. Die Upstream-Akkumulation wird iterativ berechnet: Zellen sammeln Wassermengen von allen Upstream-Zellen, die zu ihnen fließen. Dieser Prozess wird solange wiederholt, bis ein stabiler Zustand erreicht ist. Wasser, das die Kartenränder verlässt, wird als ocean_outflow akkumuliert und an den Biome Generator weitergegeben.

    Schritt 3: Manning-Strömungsberechnung mit adaptiven Querschnitten
    Die Fließgeschwindigkeit wird nach der Manning-Gleichung berechnet: v = (1/n) * R^(2/3) * S^(1/2), wobei n der Manning-Koeffizient, R der hydraulische Radius und S die Sohlneigung ist. Für jeden Flussabschnitt wird der optimale Querschnitt durch iterative Lösung der Kontinuitätsgleichung Q = A * v ermittelt. Das System unterscheidet dabei zwischen verschiedenen Geländeformen: In engen Tälern entstehen tiefe, schmale Flüsse, während in weiten Ebenen breite, flache Gewässer entstehen. Die Tal-Breite wird durch Analyse der lokalen Höhenprofile bestimmt - das System sucht in alle Richtungen nach Geländeanstiegen und passt das Breite-zu-Tiefe-Verhältnis entsprechend an.

    Schritt 4: Bodenfeuchtigkeit durch Gaussian-Diffusion
    Die Bodenfeuchtigkeit wird durch realistische Diffusion von Gewässern in das umgebende Terrain berechnet. Zwei Gaussian-Filter verschiedener Größen simulieren dabei unterschiedliche Ausbreitungsmechanismen: Ein enger Filter (Radius 2-3 Pixel) repräsentiert kapillare Ausbreitung, ein weiter Filter (Radius 5-10 Pixel) simuliert Grundwasser-Effekte. Die finale Bodenfeuchtigkeit ist das Maximum aus beiden Filtern und der direkten Wasserpräsenz, wodurch Flussufer die maximale Feuchtigkeit erhalten, während die Feuchtigkeit zur Umgebung hin exponentiell abnimmt.

    Schritt 5: Stream Power Erosion
    Die Erosion folgt dem Stream Power Gesetz: E = K * (τ - τc), wobei τ die Scherspannung und τc die kritische Scherspannung ist. Die Scherspannung wird berechnet als τ = ρ * g * h * S (Dichte * Gravitation * Wassertiefe * Sohlneigung). Die kritische Scherspannung hängt von der Gesteinshärte ab, die aus der hardness_map ausgelesen wird. Nur wenn die Scherspannung die kritische Schwelle überschreitet, findet Erosion statt. Die Erosionsrate ist proportional zur Überschuss-Energie und der Fließgeschwindigkeit im Quadrat, wird aber durch die Gesteinshärte dividiert.

    Schritt 6: Gesteinsmassen-Transport mit Massenerhaltung
    Das Erosions- und Sedimentationssystem arbeitet mit einem physikalisch korrekten Massentransport-Modell. Die rock_map speichert die Gesteinsmassen als RGB-Werte, wobei R+G+B immer 255 ergibt (normierte Massenerhaltung). Erosion transportiert Material proportional zu den lokalen Gesteinsanteilen - wenn an einer Stelle 60% Sedimentary, 30% Igneous und 10% Metamorphic vorhanden sind, wird erodiertes Material in genau diesem Verhältnis abtransportiert. Der Transport erfolgt entlang der Fließrichtungen mit einer distanz- und geschwindigkeitsabhängigen Transporteffizienz. Sedimentation lagert das transportierte Material an Stellen geringerer Transportkapazität ab, wobei die ursprünglichen Gesteinsverhältnisse erhalten bleiben. Nach jedem Zeitschritt wird die gesamte rock_map renormiert, so dass R+G+B=255 bleibt - dadurch kann Material weder verschwinden noch entstehen, sondern nur umverteilt werden.

    Schritt 7: Atmosphärische Evaporation
    Die Evaporation wird basierend auf atmosphärischen Bedingungen berechnet, nicht nach der Penman-Gleichung. Die Verdunstungsrate hängt von drei Hauptfaktoren ab: Luftfeuchtigkeit (feuchte Luft kann weniger Wasserdampf aufnehmen), Temperatur (warme Luft hat höhere Aufnahmekapazität) und Windgeschwindigkeit (Luftbewegung beschleunigt den Dampftransport). Die maximale Verdunstung ergibt sich aus der Sättigungsdampfdichte der Luft minus der aktuellen Feuchtigkeit. Wind verstärkt diesen Effekt durch verbesserten Dampftransport von der Wasseroberfläche weg. Die verfügbare Wasseroberfläche bestimmt schließlich, wie viel tatsächlich verdunsten kann.

    Benötigte Shader für die Hydrologische Simulation

    Shader 1: Jump Flooding Lake Detection (jumpFloodLakes.frag)
    Dieser Shader implementiert den Jump Flooding Algorithm für parallele Lake-Detection. In der Initialisierungsphase (u_pass_number = 0) identifiziert er lokale Minima durch Vergleich mit allen 8 Nachbarzellen. Nur Minima mit ausreichender Tiefe (lake_volume_threshold) werden als Lake-Seeds markiert. In den Propagationsphasen springen die Seeds mit exponentiell abnehmenden Distanzen und übertragen ihre Informationen an alle Zellen, die höher liegen und somit potentielle Einzugsgebiete darstellen. Der Shader gibt für jede Zelle die Position des nächstgelegenen Lake-Seeds und die Wassertiefe aus.

    Shader 2: Steepest Descent Flow Network (steepestDescentFlow.frag)
    Dieser Shader berechnet das Flussnetzwerk durch Steepest Descent Analyse. Für jede Zelle wird der steilste Gradient zu allen 8 Nachbarn berechnet und als Fließrichtung gespeichert. Niederschlagsquellen (precipitation > rain_threshold) erhalten eine initiale Wassermenge. Die Upstream-Akkumulation erfolgt iterativ: Jede Zelle sammelt Wasser von allen Nachbarzellen, deren Fließrichtung zu ihr zeigt. Lakes fungieren als Senken und akkumulieren Wasser ohne Weiterleitung. Der Shader trackt auch Wasser, das die Kartenränder verlässt.

    Shader 3: Manning Stream Flow Calculation (streamFlowCalculation.frag)
    Dieser komplexe Shader löst die Manning-Gleichung für jeden Flussabschnitt. Er berechnet zunächst eine erste Schätzung für Breite und Tiefe basierend auf dem Volumenstrom, dann iteriert er zur optimalen Lösung unter Berücksichtigung des hydraulischen Radius. Die Geländekonfinierung wird durch Analyse der lokalen Höhenprofile bestimmt - der Shader sucht in alle Richtungen nach Geländeanstiegen und passt das Breite-zu-Tiefe-Verhältnis entsprechend an. Die finale Ausgabe umfasst Fließgeschwindigkeit, Querschnittsfläche, Breite und Tiefe.

    Shader 4: Gaussian Soil Moisture Diffusion (soilMoistureCalculation.frag)
    Dieser Shader berechnet die Bodenfeuchtigkeit durch gewichtete Gaussian-Diffusion von allen Gewässern in der Umgebung. Er implementiert einen variablen Radius-Filter, der für jeden Pixel die Beiträge aller Gewässer im Diffusionsradius summiert. Die Gewichtung folgt einer Gaussian-Funktion exp(-0.5 * (d/σ)²), wobei d die Distanz und σ die Standardabweichung ist. Direkte Wasserpräsenz (Flüsse, Seen) erhält maximale Feuchtigkeit, während die Umgebung graduell abnimmt.

    Shader 5: Stream Power Erosion (erosionCalculation.frag)
    Dieser Shader implementiert das Stream Power Erosionsmodell. Er berechnet die Scherspannung τ = ρ * g * h * S für jede Zelle und vergleicht sie mit der gesteinsspezifischen kritischen Scherspannung aus der hardness_map. Die Erosionsrate folgt der Formel E = K * (τ - τc) * v², ist aber auf realistische Maximalwerte begrenzt. Der Shader berücksichtigt auch eine Minimalgeschwindigkeit für Erosion - sehr langsame Flüsse erodieren nicht, unabhängig von ihrer Größe.

    Shader 6: Hjulström Sedimentation (sedimentationCalculation.frag)
    Dieser Shader simuliert Sedimenttransport und -ablagerung nach dem Hjulström-Diagramm. Er sammelt Sediment von allen Upstream-Zellen und berechnet die lokale Transportkapazität als Funktion der Fließgeschwindigkeit (Kapazität ∝ v^2.5). Wenn die Sedimentlast die Kapazität überschreitet oder die Geschwindigkeit zu gering wird, erfolgt Sedimentation. Der Shader implementiert auch bevorzugte Sedimentation in Flussbiegungen und Konfluenzen durch Analyse der lokalen Strömungsgeometrie.

    Shader 8: Gesteinsmassen-Transport (rockMassTransport.frag)
    Dieser Shader berechnet den physikalisch korrekten Transport von Gesteinsmassen durch das Flusssystem. Er liest die aktuellen Gesteinsmassen aus der rock_map (RGB-Format) und berechnet Erosion proportional zu den lokalen Anteilen. Das erodierte Material wird entlang der Fließrichtungen transportiert, wobei die Transporteffizienz von Distanz und Fließgeschwindigkeit abhängt. Der Shader sammelt Material von allen Upstream-Zellen und berechnet die lokale Sedimentation. Die Ausgabe sind Delta-Werte für jeden Gesteinstyp, die in einem zweiten Shader zur Massenerhaltung normiert werden.

    Shader 9: Gesteinsmassen-Normierung (rockMassNormalization.frag)
    Dieser kritische Shader stellt die Massenerhaltung im Gesteinssystem sicher. Er wendet die berechneten Delta-Werte auf die aktuelle rock_map an, verhindert negative Gesteinsmassen und normiert alle RGB-Werte so, dass ihre Summe immer 255 ergibt. Dies garantiert, dass Gesteinsmasse weder verloren geht noch neu entsteht, sondern nur zwischen den Pixeln umverteilt wird. Bei vollständiger Erosion wird eine Gleichverteilung (85, 85, 85) als Fallback verwendet.

    Shader 7: Atmosphärische Evaporation (atmosphericEvaporation.frag)
    Dieser Shader berechnet die Evaporation basierend auf atmosphärischen Sättigungseffekten. Er liest humid_map, temp_map und wind_map vom data_manager und berechnet die maximale Wasserdampfdichte nach der Magnus-Formel. Die aktuelle Luftfeuchtigkeit bestimmt die verbleibende Aufnahmekapazität, die Temperatur erhöht die maximale Kapazität exponentiell, und der Wind beschleunigt den Massentransfer linear. Die verfügbare Wasseroberfläche wird aus Stream- und Lake-Daten ermittelt. Der Shader gibt die tatsächliche Verdunstungsrate zurück, die physikalisch durch die Atmosphäre begrenzt ist.

    Zusätzliche Utility-Shader:
    Terrain Confinement Analysis (terrainConfinement.frag): Analysiert lokale Talformen für realistische Flussquerschnitte
    Sediment Gaussian Distribution (sedimentDistribution.frag): Verteilt abgelagertes Sediment realistisch über die Umgebung
    Flow Accumulation Convergence (flowConvergence.frag): Überprüft Konvergenz der iterativen Upstream-Akkumulation
    Hydraulic Geometry Optimization (hydraulicGeometry.frag): Optimiert Breite-zu-Tiefe-Verhältnis basierend auf Geländeform

    Alle Shader arbeiten im Multi-Pass-Verfahren mit Ping-Pong-Buffering für iterative Berechnungen und zeitliche Entwicklung des hydrologischen Systems.

    Klassen:
    HydrologySystemGenerator
        Funktionsweise: Hauptklasse für dynamisches Hydrologiesystem mit Erosion und Sedimentation
        Aufgabe: Koordiniert alle hydrologischen Prozesse und Massentransport
        Methoden: generate_hydrology_system(), simulate_water_cycle(), update_erosion_sedimentation()

    LakeDetectionSystem
        Funktionsweise: Identifiziert Seen durch Jump Flooding Algorithm für parallele Senken-Identifikation
        Aufgabe: Findet alle potentiellen Seestandorte und deren Einzugsgebiete
        Methoden: detect_local_minima(), apply_jump_flooding(), classify_lake_basins()

    FlowNetworkBuilder
        Funktionsweise: Baut Flussnetzwerk durch Steepest Descent mit Upstream-Akkumulation
        Aufgabe: Erstellt flow_map und water_biomes_map mit realistischen Flusssystemen
        Methoden: calculate_steepest_descent(), accumulate_upstream_flow(), classify_water_bodies()

    ManningFlowCalculator
        Funktionsweise: Berechnet Strömung nach Manning-Gleichung mit adaptiven Querschnitten
        Aufgabe: Erstellt flow_speed und cross_section für realistische Fließgeschwindigkeiten
        Methoden: solve_manning_equation(), optimize_channel_geometry(), calculate_hydraulic_radius()

    ErosionSedimentationSystem
        Funktionsweise: Simuliert Stream Power Erosion mit Hjulström-Sundborg Transport
        Aufgabe: Modifiziert heightmap und rock_map durch realistische Erosions-/Sedimentationsprozesse
        Methoden: calculate_stream_power(), transport_sediment(), apply_mass_conservation()

    SoilMoistureCalculator
        Funktionsweise: Berechnet Bodenfeuchtigkeit durch Gaussian-Diffusion von Gewässern
        Aufgabe: Erstellt soil_moist_map für Biome-System und Weather-Evaporation
        Methoden: apply_gaussian_diffusion(), calculate_groundwater_effects(), integrate_moisture_sources()

    EvaporationCalculator
        Funktionsweise: Berechnet Evaporation nach atmosphärischen Bedingungen
        Aufgabe: Erstellt evaporation_map basierend auf temp_map, humid_map und wind_map
        Methoden: calculate_atmospheric_evaporation(), apply_wind_effects(), limit_by_available_water()
    """

def biome_generator():
    """
    Path: core/biome_generator.py

    Funktionsweise: Klassifikation von Biomen auf Basis von Höhe, Temperatur, Niederschlag und Slope
    - Gauß-basierte Klassifizierung mit Gewichtungen je Biomtyp
    - Verwendung eines vektorbasierten Klassifikators für performante Zuordnung auf großen Karten
    - Zwei-Ebenen-System: Base-Biomes und Super-Biomes
    - Supersampling für weichere Übergänge zwischen allen Biomen (die vier dominantesten Anteile pro Zelle)
    - Super-Biomes überschreiben Base-Biomes basierend auf speziellen Bedingungen

    Parameter Input:
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
    - heightmap (2D-Array in meter Altitude)
    - slopemap (2D-Array in m/m mit dz/dx, dz/dy)
    - temp_map (2D-Array in °C)
    - soil_moist_map (2D-Array in Bodenfeuchtigkeit %)
    - water_biomes_map (2D-Array mit Wasser-Klassifikation: 0=kein Wasser, 1=Creek, 2=River, 3=Grand River, 4=Lake)

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
    BiomeClassificationSystem
        Funktionsweise: Hauptklasse für Biom-Klassifikation basierend auf Klimadaten
        Aufgabe: Koordiniert Base-Biome und Super-Biome Zuordnung mit Supersampling
        Methoden: classify_biomes(), apply_supersampling(), integrate_super_biomes()

    BaseBiomeClassifier
        Funktionsweise: Gauß-basierte Klassifizierung von 15 Grundbiomen
        Aufgabe: Erstellt biome_map basierend auf Höhe, Temperatur, Niederschlag und Slope
        Methoden: calculate_gaussian_fitness(), weight_environmental_factors(), assign_dominant_biome()

    SuperBiomeOverrideSystem
        Funktionsweise: Überschreibt Base-Biomes mit speziellen Bedingungen (Ocean, Cliff, Beach, etc.)
        Aufgabe: Erstellt super_biome_mask für prioritätsbasierte Biom-Überschreibung
        Methoden: detect_ocean_connectivity(), apply_proximity_biomes(), calculate_elevation_biomes()

    SupersamplingManager
        Funktionsweise: 2x2 Supersampling mit diskretisierter Zufalls-Rotation
        Aufgabe: Erstellt biome_map_super für weiche Übergänge zwischen Biomen
        Methoden: apply_rotational_supersampling(), calculate_soft_transitions(), optimize_spatial_distribution()

    ProximityBiomeCalculator
        Funktionsweise: Berechnet Proximity-basierte Super-Biomes (Beach, Lake Edge, River Bank)
        Aufgabe: Erstellt weiche Übergänge um Gewässer mit konfigurierbarem edge_softness
        Methoden: calculate_distance_fields(), apply_gaussian_proximity(), blend_with_base_biomes()
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
        SIZE = {"min": 32, "max": 2048, "default": 256, "step": 32}
        HEIGHT = {"min": 0, "max": 400, "default": 100, "step": 10, "suffix": "m"}
        OCTAVES = {"min": 1, "max": 10, "default": 4}
        # etc.

    class GEOLOGY:
        HARDNESS = {"min": 1, "max": 100, "default": 50}
        # etc.
    """

def gui_default():
    """
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


def data_lod_manager():
    """
    Path: gui/managers/data_lod_manager.py
    date_changed: 18.08.2025

    =========================================================================================
    DATA LOD MANAGER - INTEGRIERTE DATENVERWALTUNG UND RESOURCE-MANAGEMENT
    =========================================================================================

    Parameter Input:
    - lod_config: map_size_min, target_map_size für LOD-Progression
    - memory_thresholds: tracking_threshold_mb, cleanup_threshold_mb, force_gc_threshold_mb
    - tab_connections: generator_type → tab_instance mapping für Signal-Integration

    Output:
    - Zentrale Datenverwaltung für alle 6 Generator-Typen mit numerischem LOD-System
    - Signal-Hub für Tab-Kommunikation und Dependency-Management
    - Resource-Tracker für Memory-Leak-Prevention und automatisches Cleanup
    - Display-Manager für optimierte UI-Updates mit Change-Detection

    LOD-System (Numerisch):
    - lod_level 1: map_size_min (default 32x32)
    - lod_level 2: 64x64 (Verdopplung bis target_map_size erreicht)
    - lod_level 3: 128x128
    - lod_level n: erweiterbar, lod_size bleibt bei target_map_size fixiert
    - Proportionale Generator-Skalierung: Weather CFD-Zellen, Settlement Nodes, etc.

    DEPENDENCY MATRIX (Zentral):
    ---------------------------
    DEPENDENCY_MATRIX = {
        "terrain": [],                                    # Basis-Generator
        "geology": ["terrain"],                          # heightmap, slopemap
        "weather": ["terrain"],                          # heightmap für orographic effects
        "water": ["terrain", "geology", "weather"],     # heightmap, hardness_map, precipitation
        "biome": ["terrain", "weather", "water", "geology"], # vollständige Umwelt-Dependencies
        "settlement": ["terrain", "geology", "water", "biome"] # alle Dependencies für Settlement-Planning
    }

    Dependency-Resolution:
    - check_dependencies(generator_type, lod_level): Validiert alle Required-Dependencies
    - dependencies_satisfied Signal-Emission bei Completion
    - Auto-Generation-Triggering für nachgelagerte Generatoren

    MEMORY-MANAGEMENT:
    ------------------
    Grenzwerte (für 2048x2048 Maps optimiert):
    - Tracking-Threshold: 50 MB
    - Cleanup-Threshold: 200 MB
    - Force-GC-Threshold: 500 MB für kritische Memory-Situationen

    Resource-Type-Management:
    - WeakReference-basiertes Tracking aller großen Arrays
    - Age-based Cleanup: Ressourcen >2h automatisch bereinigt
    - Size-based Cleanup: Ressourcen >200MB bei Memory-Warnings
    - Automatic Garbage-Collection bei Force-GC-Threshold

    SIGNAL ARCHITECTURE:
    --------------------
    Outgoing Signals (Hub → Tabs/UI):
    - tab_status_updated(tab, status_dict): Status-Changes für einzelne Tabs
    - dependencies_satisfied(tab, lod_level): Dependency-Requirements erfüllt
    - shader_info_updated(generator_type, shader_stats): GPU/CPU-Fallback-Status
    - memory_warning(current_usage_mb, threshold_mb): Memory-Management-Warnings

    Incoming Signals (Tabs → Hub):
    - on_tab_lod_started(tab, lod_level, lod_size): Generation gestartet
    - on_tab_lod_progress(tab, lod_level, progress_percent): Progress-Update
    - on_tab_lod_completed(tab, lod_level, success, data_keys): Generation fertig
    - on_shader_status_changed(generator, operation, gpu_used, performance): ShaderManager-Updates

    Legacy Compatibility:
    - data_updated(generator_type, data_key): Für bestehende Tab-Integration
    - cache_invalidated(generator_type): Cache-Management
    - lod_data_stored(generator_type, lod_level, data_keys): Neue LOD-Daten

    Klassen:
    --------

    DataLODManager
        Funktionsweise: Zentrale Koordination aller vier Subsysteme mit einheitlicher API
        Aufgabe: Datenverwaltung, Signal-Hub, Resource-Tracking, Display-Optimierung
        Methoden: set_*_data_lod(), get_*_data(), connect_tab_to_hub(), check_dependencies()
        LOD-Integration: Numerische LOD-Level für alle Generator-Typen, proportionale Skalierung
        Memory-Management: Integrierte Resource-Tracker mit automatischem Cleanup
        Threading: Thread-safe Operations durch QMutex, Signal-basierte Tab-Communication

    LODDataStorage
        Funktionsweise: Hierarchische Datenspeicherung mit numerischen LOD-Levels
        Aufgabe: Effiziente Speicherung und Abruf von Generator-Daten nach LOD-Level
        Methoden: store_data(), get_data(), get_available_lods(), cleanup_old_lods()
        Data-Structure: nested dict {generator_type: {lod_level: {data_key: array}}}
        Cache-Management: Parameter-Hash-basierte Invalidation, Memory-Usage-Tracking

    CommunicationHub
        Funktionsweise: Signal-basierte Koordination zwischen Tabs mit Dependency-Management
        Aufgabe: Status-Tracking, Dependency-Resolution, Auto-Generation-Triggering
        Methoden: connect_tab(), emit_status_update(), check_tab_dependencies()
        Status-States: idle/pending/running/success/failure pro Tab/LOD-Level
        Dependency-Logic: DEPENDENCY_MATRIX-basierte Validation mit LOD-Level-Checks

    ResourceTracker
        Funktionsweise: WeakReference-basiertes Tracking mit automatischem Cleanup
        Aufgabe: Memory-Leak-Prevention, Resource-Monitoring, Performance-Optimierung
        Methoden: register_resource(), cleanup_resources(), get_memory_usage()
        Tracking-Types: numpy-arrays, gpu-textures, custom-resources mit Cleanup-Callbacks
        Cleanup-Strategies: age-based, size-based, force-gc bei kritischen Memory-Levels

    DisplayUpdateManager
        Funktionsweise: Hash-basierte Change-Detection für optimierte UI-Updates
        Aufgabe: Verhindert unnötige Re-Renderings, Performance-Monitoring
        Methoden: needs_update(), mark_updated(), clear_cache()
        Hash-Optimization: Sample-based Hashing für große Arrays (>1M Elemente)
        Cache-Management: Pending-Updates-Prevention, automatische Cache-Bereinigung

    Fallback-System (3-stufig):
    ---------------------------
    - Primary: Vollständige LOD-Integration mit allen Subsystemen
    - Secondary: Legacy-Kompatibilität ohne LOD-Features aber mit Resource-Management
    - Minimal: Basic Data-Storage ohne Signal-Integration oder Optimization

    Error-Recovery-Mechanisms:
    - Exception-Safe-Operations: Alle kritischen Methoden mit Fallback-Returns
    - Resource-Cleanup: Garantierte Memory-Freigabe auch bei Exceptions
    - Signal-Resilience: Tab-Communication funktioniert auch bei Hub-Fehlern
    - Data-Integrity: Atomic Operations mit Rollback bei Critical-Failures

    Performance-Characteristics:
    - LOD-Caching: Numerische LODs 30% schneller als String-Vergleiche
    - Memory-Efficiency: WeakReference ohne Memory-Overhead, Sample-based Hashing
    - Signal-Performance: Direct pyqtSlot-Connections, Debounced Updates
    - Resource-Optimization: Automatic Cleanup verhindert Memory-Accumulation

    Usage-Pattern:
    ```python
    # Factory-Setup mit konfigurierten Defaults
    data_lod_manager = create_integrated_data_lod_manager(memory_threshold_mb=200)

    # LOD-basierte Data-Storage
    data_lod_manager.set_terrain_data_lod("heightmap", heightmap_array, lod_level=3, parameters)

    # Legacy-kompatible Data-Retrieval
    heightmap = data_lod_manager.get_terrain_data("heightmap")  # höchstes verfügbares LOD
    ```

    Integration-Points:
    - BaseMapTab: ResourceTracker und DisplayUpdateManager ersetzen base_tab.py Implementierungen
    - ShaderManager: Performance-Tracking für LOD-Optimierung, GPU-Memory-Coordination
    - GenerationOrchestrator: Dependency-Matrix als Single-Source-of-Truth, LOD-Status-Updates

    =========================================================================================
    """

def parameter_manager():
    """
    Pfad: gui/managers/parameter_manager.py
    Änderungsdatum: 08.08.2025

    ═══════════════════════════════════════════════════════════════════════════════

    PARAMETER MANAGER v2.0 - MANUAL-ONLY SYSTEM

    Input: Tab-Parameter-Änderungen, Export/Import-Anfragen, Preset-Operationen
    Output: Cache-Invalidation-Signale, Parameter-Validation, JSON-Export-Dateien
    Kern: Cross-Tab-Synchronisation ohne automatische Generation-Auslösung

    ═══════════════════════════════════════════════════════════════════════════════

    1. ÜBERSICHT UND ZIELSETZUNG

    Der ParameterManager koordiniert Parameter zwischen sechs Generator-Tabs durch
    ein integriertes Kommunikationssystem mit Export/Import-Management und
    Preset-System. Die Hauptänderung von automatischem zu manual-only
    Generation-System eliminiert Timing-Probleme bei vollständiger
    Parameter-Koordination.

    Kernfunktionen:
    - Cross-Tab Parameter-Broadcasting mit intelligenter Change-Tracking
    - JSON-basierte Export/Import-Funktionalität mit Metadaten-Management
    - Kategorisiertes Preset-System mit Such- und Tag-Funktionalität
    - Parameter-Validation mit Constraint-System für alle Generator-Types
    - Cache-Invalidation ohne automatische Generation-Auslösung

    Neue Arbeitsweise:
    Der überarbeitete Ablauf folgt dem Muster Parameter-Change führt zu
    Cache-Invalidation, gefolgt von manuellem "Berechnen"-Click. Dies eliminiert
    Race-Conditions und ungewollte automatische Generierungen vollständig.

    ═══════════════════════════════════════════════════════════════════════════════

    2. KERNÄNDERUNGEN VON VERSION 1.0

    Entfernte Komponenten:
    Das System eliminiert den Auto-Generation-Timer, Debounced Updates und
    Race-Condition-Prevention-Mechanismen. Diese Komponenten führten zu
    unvorhersagbaren Generierungs-Zyklen und erschwerten die Benutzersteuerung.

    Hinzugefügte Komponenten:
    Neu implementiert wurden die Impact-Classification-Matrix für intelligente
    Cache-Invalidation und eine Manual-Only Signal-Architektur. Diese Ergänzungen
    ermöglichen präzise Steuerung der Generierungsabläufe durch den Benutzer.

    Unveränderte Komponenten:
    Das Export/Import-System und Preset-Management bleiben funktional identisch
    zur Version 1.0. Diese bewährten Komponenten bieten weiterhin vollständige
    Parameter-Serialisierung und Template-Generierung.

    ═══════════════════════════════════════════════════════════════════════════════

    3. ARCHITEKTUR-KOMPONENTEN

    ParameterCommunicationHub:
    Diese Kernkomponente übernimmt Cross-Tab Parameter-Broadcasting mit
    intelligenter Change-Tracking. Der Hub implementiert broadcast_change für
    Tab-übergreifende Parameter-Synchronisation, add_listener für
    Event-Registration und get_dependencies für Abhängigkeits-Abfragen.
    Die Integration löst cache_invalidation_requested Signale an den
    DataLODManager aus.

    ExportImportManager:
    Verantwortlich für JSON-basierte Parameter-Serialisierung und
    Template-Generierung. Implementiert export_json für vollständige
    Parameter-Ausgabe, import_json für Daten-Wiederherstellung, create_template
    für Vorlagen-Erstellung und validate_import für Integritätsprüfung.
    Die Integration erfolgt durch datei-basierte Speicherung mit Metadaten
    und Versionsverwaltung.

    PresetManager:
    Ermöglicht kategorisierte Preset-Speicherung mit umfassendem Such- und
    Tag-System. Bietet save_preset für Konfigurationsspeicherung, load_preset
    für Wiederherstellung, search_presets für Suchfunktionalität und
    list_categories für Organisationsstruktur. Die Integration erfolgt durch
    datei-basierte Speicherung im presets-Verzeichnis mit JSON-Format.

    ParameterUpdateManager:
    Spezialisiert auf Cache-Invalidation ohne automatische Generation.
    Implementiert request_cache_invalidation für explizite Cache-Markierung
    und track_parameter_changes für Änderungs-Verfolgung. Die Integration
    eliminiert das Timer-System vollständig und ermöglicht direkte
    Cache-Markierung.

    ═══════════════════════════════════════════════════════════════════════════════

    4. SIGNAL-SYSTEM UND KOMMUNIKATION

    Information-Only Signals:
    Das neue Signal-System basiert auf drei Hauptsignalen ohne
    Generation-Trigger-Funktionalität. Das parameter_changed Signal übermittelt
    Tab-Name, Parameter-Name, alten Wert und neuen Wert für
    Tab-übergreifende Information. Das cache_invalidation_requested Signal
    kommuniziert Source-Tab und betroffene Generatoren für intelligente
    Cache-Verwaltung. Das manual_generation_requested Signal überträgt
    Generator-Type und Parameter-Dictionary für explizite Generierungs-Anfragen.

    Entfernte Signals:
    Die Signals validation_requested, generation_requested und update_completed
    wurden vollständig entfernt, da sie automatische Generierungen auslösten
    und damit dem Manual-Only-Prinzip widersprachen.

    Signal-Integration-Pattern:
    Tabs registrieren sich für Information-Only Signals über
    cache_invalidation_requested und parameter_changed Verbindungen.
    Der Ablauf erfolgt durch Tab-Registration mit Name, Instanz und
    Dependencies, gefolgt von Parameter-Änderungen über set_parameter,
    Cache-Invalidation-Signalen und manueller Generierungs-Anfrage durch
    Benutzer-Klick.

    ═══════════════════════════════════════════════════════════════════════════════

    5. PARAMETER-IMPACT-KLASSIFIKATION

    Impact-Matrix-Struktur:
    Das System implementiert eine umfassende Impact-Matrix für intelligente
    Cache-Invalidation. Jeder Generator-Type definiert drei Impact-Kategorien:
    High-Impact-Parameter, Medium-Impact-Parameter und Low-Impact-Parameter.

    Terrain-Impact-Klassifikation:
    High-Impact-Parameter umfassen map_seed, size, amplitude, octaves und
    frequency, da diese die grundlegende Terrain-Struktur bestimmen.
    Medium-Impact-Parameter beinhalten persistence und lacunarity für
    Oberflächendetails. Low-Impact-Parameter wie redistribute_power
    beeinflussen nur lokale Eigenschaften.

    Geology-Impact-Klassifikation:
    High-Impact-Parameter umfassen sedimentary_hardness, igneous_hardness
    und metamorphic_hardness, da diese die geologische Grundstruktur definieren.
    Medium-Impact-Parameter beinhalten ridge_warping und bevel_warping für
    Oberflächenmodifikationen. Low-Impact-Parameter wie metamorphic_foliation
    und igneous_flowing beeinflussen nur visuelle Details.

    Weather-Impact-Klassifikation:
    High-Impact-Parameter umfassen temperature_range und precipitation_base
    für grundlegende Klimacharakteristika. Medium-Impact-Parameter beinhalten
    wind_strength und humidity_base für sekundäre Klimaeffekte.
    Low-Impact-Parameter wie weather_seed beeinflussen nur Variationen.

    Water-Impact-Klassifikation:
    High-Impact-Parameter umfassen water_threshold und flow_iterations für
    fundamentale Wassersystem-Eigenschaften. Medium-Impact-Parameter beinhalten
    manning_coefficient und erosion_factor für Fließverhalten.
    Low-Impact-Parameter wie river_width_factor beeinflussen nur visuelle Aspekte.

    Biome-Impact-Klassifikation:
    High-Impact-Parameter umfassen temperature_factor und precipitation_factor
    für Grundökosystem-Definition. Medium-Impact-Parameter beinhalten
    elevation_factor und latitude_factor für sekundäre Biom-Beeinflussung.
    Low-Impact-Parameter wie biome_smoothing beeinflussen nur Übergänge.

    Settlement-Impact-Klassifikation:
    High-Impact-Parameter umfassen settlement_count und city_ratio für
    Grundsiedlungsstruktur. Medium-Impact-Parameter beinhalten road_density
    und trade_factor für Infrastruktur-Details. Low-Impact-Parameter wie
    settlement_seed beeinflussen nur Positionsvariationen.

    Dependency-Hierarchie-System:
    Das System implementiert eine strenge Dependency-Hierarchie. Terrain bildet
    die Basis ohne Dependencies. Geology und Weather hängen von Terrain ab.
    Water benötigt Terrain und Geology. Biome integriert Terrain, Geology,
    Water und Weather. Settlement als komplexeste Schicht benötigt Terrain,
    Geology, Water und Biome.

    Cache-Invalidation-Effekte:
    High-Impact-Parameter invalidieren alle nachgelagerten Generatoren entsprechend
    der Dependency-Hierarchie. Medium-Impact-Parameter invalidieren nur direkte
    Abhängigkeiten. Low-Impact-Parameter erfordern manuelle Regeneration ohne
    automatische Invalidation.

    ═══════════════════════════════════════════════════════════════════════════════

    6. TAB-INTEGRATIONS-PATTERN

    BaseMapTab-Delegation-System:
    Jeder BaseMapTab erhält Zugriff auf den globalen ParameterManager über
    get_global_parameter_manager. Parameter-Abfragen erfolgen über
    get_tab_parameters mit Tab-Namen-Spezifikation. Parameter-Änderungen
    werden über set_parameter mit Tab-Name, Parameter-Name und neuem Wert
    übermittelt.

    Signal-Integration-Mechanismus:
    Tabs registrieren sich für cache_invalidation_requested Signale über
    on_cache_invalidated Handler. External Parameter-Changes werden über
    parameter_changed Signale und on_external_parameter_changed Handler verarbeitet.
    Diese Integration ermöglicht vollständige Tab-übergreifende Synchronisation.

    Registration-und-Ablauf-Pattern:
    Der Ablauf beginnt mit Tab-Registration über register_tab mit Name, Instanz
    und Dependencies. Parameter-Änderungen lösen set_parameter-Aufrufe aus.
    Der Manager sendet cache_invalidation_requested mit Source-Tab und betroffenen
    Generatoren. Benutzer-Klicks auf "Berechnen" führen zu manual_generation_requested
    an den GenerationOrchestrator.

    ═══════════════════════════════════════════════════════════════════════════════

    7. PARAMETER-VALIDATION-SYSTEM

    Constraint-System-Implementierung:
    Das System implementiert umfassende Parameter-Constraints für alle
    Generator-Types durch Lambda-Funktionen und Range-Validierung.

    Terrain-Constraints:
    Size-Parameter müssen zwischen 64 und 2048 liegen und Power-of-2-Werte sein.
    Amplitude-Parameter erfordern Werte zwischen 0 und 1000. Octaves-Parameter
    müssen zwischen 1 und 10 liegen. Frequency-Parameter erfordern positive
    Fließkomma-Werte. Persistence und Lacunarity müssen zwischen 0 und 2 liegen.

    Geology-Constraints:
    Hardness-Parameter für Sedimentary, Igneous und Metamorphic müssen zwischen
    0 und 100 liegen. Ridge-Warping und Bevel-Warping erfordern Werte zwischen
    0.0 und 1.0. Foliation und Flowing-Parameter müssen zwischen 0 und 100 liegen.

    Weather-Constraints:
    Temperature-Range-Parameter müssen zwischen -50 und 60 Grad liegen.
    Precipitation-Base erfordert Werte zwischen 0 und 5000 Millimeter.
    Wind-Speed-Parameter müssen zwischen 0 und 200 km/h liegen.
    Humidity-Parameter erfordern Werte zwischen 0 und 100 Prozent.

    Water-Constraints:
    Water-Threshold-Parameter müssen zwischen 0.0 und 1.0 liegen.
    Manning-Coefficient erfordert Werte zwischen 0.01 und 0.1.
    Flow-Iterations müssen zwischen 1 und 1000 liegen.
    Erosion-Factor-Parameter erfordern Werte zwischen 0.0 und 2.0.

    Biome-Constraints:
    Temperature-Factor und Precipitation-Factor müssen zwischen 0.0 und 2.0 liegen.
    Elevation-Levels erfordern Werte zwischen 0 und 5000 Meter.
    Latitude-Factor muss zwischen -90 und 90 Grad liegen.
    Smoothing-Parameter erfordern Werte zwischen 0.0 und 1.0.

    Settlement-Constraints:
    Settlement-Count muss zwischen 1 und 50 liegen. City-Ratio erfordert Werte
    zwischen 0.0 und 2.0. Road-Density muss zwischen 0.0 und 1.0 liegen.
    Trade-Factor erfordert Werte zwischen 0.0 und 3.0.

    Validation-Prozess-Stufen:
    Individual Parameter-Validation führt Range- und Type-Checks für jeden
    Parameter durch. Cross-Parameter-Validation prüft Consistency zwischen
    verwandten Parametern. Dependency-Validation überprüft Parameter-Compatibility
    zwischen abhängigen Tabs. Error-Aggregation erstellt comprehensive
    Error-Reports für UI-Display.

    ═══════════════════════════════════════════════════════════════════════════════

    8. EXPORT/IMPORT UND PRESET-SYSTEM

    JSON-Export-Format-Struktur:
    Das vollständige Export-Format umfasst Metadata-Sektion mit Export-Timestamp,
    Parameter-Dependencies-Mapping, Map-Generator-Version, exportierten Tab-Liste,
    Gesamt-Parameter-Anzahl und Export-Format-Version. Die Parameters-Sektion
    enthält Tab-kategorisierte Parameter-Collections mit allen relevanten Werten.

    Metadata-Komponenten:
    Export-Timestamp dokumentiert Erstellungszeit im ISO-Format.
    Parameter-Dependencies-Mapping zeigt Abhängigkeitsbeziehungen zwischen Tabs.
    Map-Generator-Version identifiziert Kompatibilität. Exported-Tabs-Liste
    spezifiziert inkludierte Tab-Namen. Total-Parameters zählt exportierte
    Parameter. Export-Format-Version ermöglicht Rückwärtskompatibilität.

    Template-System-Funktionalität:
    Template-Generation erstellt Vorlagen für verschiedene Map-Types mit
    Default-Parameter-Werten. Template-Info-Sektion enthält Name, Description,
    Included-Tabs und Creation-Date. Parameters-Sektion liefert
    Default-Werte für alle spezifizierten Tabs.

    Import-Validation-Mechanismen:
    Format-Version-Compatibility-Checks prüfen Systemkompatibilität.
    Missing-Parameter-Detection identifiziert fehlende Werte mit
    Default-Value-Substitution. Tab-Availability-Validation bestätigt
    Tab-Existenz. Parameter-Range-Validation prüft gegen aktuelle Constraints.
    Dependency-Consistency-Validation überprüft Abhängigkeits-Integrität.

    Preset-System-Struktur:
    Preset-Info-Sektion enthält Name, Description, Creation-Date, Category,
    Tags-Array, Parameter-Count und File-Version. Parameters-Sektion speichert
    Tab-kategorisierte Parameter-Werte. Das System unterstützt fünf
    Hauptkategorien mit spezialisierten Tag-Systemen.

    Preset-Kategorien-System:
    General-Kategorie umfasst Mixed-Parameter für verschiedene Map-Types.
    Terrain-Kategorie fokussiert auf Terrain-spezifische Presets wie Mountains,
    Plains und Islands. Weather-Kategorie enthält Weather-Pattern wie Tropical,
    Arid und Temperate. Biome-Kategorie speichert Ecosystem-focused Presets
    wie Desert, Forest und Tundra. Settlement-Kategorie umfasst
    Civilization-Pattern wie Medieval, Modern und Sparse.

    ═══════════════════════════════════════════════════════════════════════════════

    9. PERFORMANCE UND FEHLERBEHANDLUNG

    Performance-Charakteristiken:
    Change-Detection erfolgt via Hash-Comparison mit O(1)-Komplexität für
    einzelne Parameter. Eliminierte Timer reduzieren Event-Spam von über
    100 Events pro Sekunde auf direkte Invalidation. Parameter-Cache eliminiert
    redundante Tab-Queries mit 95-prozentiger Cache-Hit-Rate. Preset-Cache
    mit automatischem Cleanup verhindert Memory-Accumulation.
    Parameter-History-Limit von 1000 Events verhindert unbegrenztes Memory-Growth.

    Memory-Efficiency-Mechanismen:
    WeakReference-Pattern für Tab-Registration verhindert Memory-Leaks bei
    Tab-Destruction. JSON-Streaming für Large-Parameter-Set Export/Import
    unterstützt Files über 1MB. Lazy-Loading von Preset-Files erfolgt bei
    First-Access. Batch-Export reduziert I/O-Operations um 80 Prozent durch
    Kombinierte Schreibvorgänge.

    Fehlerbehandlung-Strategien:
    Graceful Degradation bei einzelnen Tab-Registration-Failures ermöglicht
    Partial-System-Operation. Parameter-Default-Substitution bei Import-Errors
    erhält Systemfunktionalität. Constraint-Violation-Recovery durch
    Value-Clamping zu gültigen Ranges korrigiert ungültige Eingaben.
    Backup-Creation vor destructive Operations wie Preset-Overwrite schützt
    vor Datenverlust. Atomic-Write-Pattern für Data-Integrity bei Export/Import
    verhindert korrupte Dateien.

    Debugging-und-Monitoring-Funktionalität:
    Parameter-Change-History ermöglicht Debugging mit get_parameter_change_history
    für spezifische Tab-Names mit konfigurierbarem Limit. Die Funktion liefert
    ParameterChangeEvent-Listen mit Timestamp, Old-Value und New-Value.
    Cross-Tab Parameter-Dependencies werden über get_dependency_parameters
    abgefragt und liefern abhängige Parameter-Dictionaries.
    Export-Success-Rate-Tracking über get_export_statistics zeigt
    Successful-Exports, Failed-Exports und Average-Size-KB.

    ═══════════════════════════════════════════════════════════════════════════════

    10. DATALODMANAGER-INTEGRATION

    Integration-Pattern-Mechanismen:
    Parameter-Changes lösen cache_invalidated Signale im DataLODManager aus
    für koordinierte Cache-Verwaltung. Export-Data enthält LOD-Status-Information
    vom DataLODManager für vollständige System-State-Speicherung.
    Preset-Application koordiniert mit LOD-System für optimale Performance
    durch intelligente Cache-Priorisierung. Parameter-History korreliert mit
    Generation-History für umfassendes System-Debugging.

    Signal-Koordination:
    Das System koordiniert cache_invalidation_requested Signale mit dem
    DataLODManager für intelligente LOD-Invalidation. Parameter-Impact-Level
    bestimmt LOD-Invalidation-Scope basierend auf Dependency-Hierarchie.
    Manual-Generation-Requests werden an GenerationOrchestrator weitergeleitet
    mit vollständiger Parameter-Context-Information.

    ═══════════════════════════════════════════════════════════════════════════════
    """

def generation_orchestrator():
    """
    Path: gui/managers/generation_orchestrator.py
    Date Changed: 08.08.2025

    ═══════════════════════════════════════════════════════════════════════════════

    1. ÜBERBLICK UND ZIELSETZUNG

    Der GenerationOrchestrator fungiert als zentrale Koordinationsstelle für alle
    sechs Map-Generatoren des Systems. Seine Hauptaufgabe besteht darin, komplexe
    Generierungsprozesse intelligent zu orchestrieren, dabei Abhängigkeiten
    automatisch aufzulösen und eine optimale Ressourcennutzung sicherzustellen.

    Kernfunktionen:
    - Zentrale Koordination aller Map-Generatoren (Terrain, Geology, Weather, Water, Biome, Settlement)
    - Intelligente Abhängigkeitsauflösung basierend auf der DataLODManager-Dependency-Matrix
    - Parallelverarbeitung ohne UI-Blockierung durch Background-Threading
    - Automatische Qualitätsprogression von niedrigen zu hohen LOD-Stufen
    - Robuste Fehlerbehandlung mit Graceful Degradation

    Architektonische Positionierung:
    Der Orchestrator sitzt zwischen den Map-Tabs (Benutzerebene) und den
    Core-Generatoren (Berechnungsebene). Er transformiert Benutzeranfragen in
    strukturierte Generierungssequenzen und koordiniert dabei die komplexen
    Interdependenzen zwischen verschiedenen Map-Komponenten.

    ═══════════════════════════════════════════════════════════════════════════════

    2. BENUTZERINTERAKTION UND ARBEITSWEISE

    Manual-Only-Triggering-Prinzip:
    Das System folgt einem strikten Manual-Only-Ansatz: Generierungen werden
    ausschließlich durch explizite Benutzeraktion ("Berechnen"-Button) ausgelöst.
    Parameter-Änderungen führen niemals zu automatischen Neu-Generierungen, was
    ungewollte Berechnungen und Ressourcenverschwendung verhindert.

    Request-Flow:
    Der typische Ablauf beginnt mit einem Klick auf "Berechnen" in einem BaseMapTab.
    Der Orchestrator erstellt daraufhin eine strukturierte Anfrage, prüft
    Abhängigkeiten und startet bei erfüllten Voraussetzungen sofort die
    Background-Generierung. Falls Dependencies fehlen, wird die Anfrage in die
    Dependency-Queue eingereiht.

    Ununterbrechbare Background-Verarbeitung:
    Einmal gestartete Generierungen laufen vollständig im Hintergrund ab und können
    durch Benutzeraktionen nicht unterbrochen werden. Tab-Wechsel, Parameter-Änderungen
    oder andere UI-Interaktionen haben keinen Einfluss auf laufende Berechnungen.
    Dies gewährleistet Konsistenz und verhindert inkomplette Generierungszustände.

    Kontinuierliche UI-Updates:
    Trotz der ununterbrechbaren Verarbeitung erhält die Benutzeroberfläche
    kontinuierliche Updates über den Fortschritt. Nach jeder abgeschlossenen
    LOD-Stufe werden die Ergebnisse sofort in der UI sichtbar, sodass Benutzer
    den Qualitätsfortschritt in Echtzeit verfolgen können.

    ═══════════════════════════════════════════════════════════════════════════════

    3. GENERATOREN UND ABHÄNGIGKEITEN

    Die sechs Map-Generatoren:
    Das System koordiniert folgende Generatoren in ihrer natürlichen
    Abhängigkeitsreihenfolge:

    - Terrain (Basis-Layer): Höhenkarten, Slopes, Shadows - bildet die geografische
      Grundlage für alle anderen Generatoren.

    - Geology: Gesteinsverteilung basierend auf Terrain-Eigenschaften - bestimmt
      Bodenbeschaffenheit und geologische Strukturen.

    - Weather: Klimamuster abhängig von Terrain-Features - beeinflusst Niederschlag,
      Temperature und Windverhältnisse.

    - Water: Wassersysteme basierend auf Terrain, Geology und Weather - umfasst
      Flüsse, Seen und Grundwasser.

    - Biome: Vegetationszonen abhängig von Terrain, Weather und Water - definiert
      Ökosysteme und Landschaftscharakter.

    - Settlement: Siedlungsstrukturen basierend auf Terrain, Water und Biome -
      platziert menschliche Aktivitäten optimal.

    Abhängigkeitshierarchie:
    Die Generatoren sind in einer strengen Hierarchie organisiert. Terrain bildet
    die Basis ohne Dependencies. Geology und Weather hängen direkt von Terrain ab.
    Water benötigt alle drei Vorgänger. Biome integriert Terrain, Weather und Water.
    Settlement als komplexeste Schicht benötigt Terrain, Water und Biome.

    Automatische Kettengenerierung:
    Der Orchestrator überwacht kontinuierlich den Completion-Status aller Generatoren.
    Sobald alle Dependencies für einen nachgelagerten Generator erfüllt sind, startet
    automatisch dessen Generierung. Dies ermöglicht effiziente Batch-Verarbeitung
    ohne manuelle Intervention bei jeder Abhängigkeitsstufe.

    Dependency-Resolution-Logic:
    Das System verwendet die DataLODManager-Dependency-Matrix als Single-Source-of-Truth
    für alle Abhängigkeitsbeziehungen. Eine kontinuierliche Queue-Resolution prüft
    alle zwei Sekunden verfügbare Requests und startet diese bei erfüllten
    Voraussetzungen. Timeout-Management verhindert Deadlocks bei unerfüllbaren
    Dependencies.

    ═══════════════════════════════════════════════════════════════════════════════

    4. LOD-PROGRESSION UND QUALITÄTSSTUFEN

    Stufenweise Qualitätsverbesserung:
    Jede Generierung durchläuft automatisch eine LOD-Progression von niedrigen zu
    hohen Qualitätsstufen. Typischerweise beginnt der Prozess bei LOD 1 und steigert
    sich schrittweise bis zur maximalen Map-Size. Dies ermöglicht schnelle erste
    Ergebnisse mit kontinuierlicher Qualitätsverbesserung.

    Thread-per-LOD-Execution:
    Jede LOD-Stufe wird in einem eigenen Thread verarbeitet, was maximale
    Parallelisierung ermöglicht. Während höhere LOD-Stufen noch berechnet werden,
    sind niedrigere bereits verfügbar und können in der UI angezeigt werden.

    Incremental Result-Storage:
    Nach Completion jeder LOD-Stufe werden die Ergebnisse automatisch im DataManager
    gespeichert. Dies ermöglicht sofortige UI-Updates und stellt sicher, dass auch
    bei Unterbrechungen bereits berechnete Qualitätsstufen erhalten bleiben.

    Smart LOD-Sequence-Creation:
    Das System erkennt bereits vorhandene LOD-Stufen und generiert nur fehlende
    Levels. Wenn beispielsweise LOD 1 und 2 bereits existieren, startet die
    Progression direkt bei LOD 3. Dies optimiert Performance und vermeidet
    redundante Berechnungen.

    UI-Integration der Progression:
    Die Benutzeroberfläche erhält nach jeder LOD-Completion ein Update mit der
    besten verfügbaren Qualität. Benutzer sehen sofort Verbesserungen, ohne auf
    die finale Completion warten zu müssen. Progress-Updates informieren über
    aktuell verarbeitete LOD-Stufe und geschätzte Completion-Zeit.

    ═══════════════════════════════════════════════════════════════════════════════

    5. PARALLELVERARBEITUNG UND PERFORMANCE

    Threading-Architektur:
    Das System basiert auf QThread-Background-Processing mit einem Thread-Pool-Pattern.
    Maximal drei Generierungen laufen parallel, um optimale Ressourcennutzung ohne
    System-Überlastung zu gewährleisten. QMutex-Protection sichert Thread-Safety
    bei gleichzeitigen Datenzugriffen.

    Parallellisierungs-Strategie:
    Die Parallelverarbeitung erfolgt sowohl zwischen verschiedenen Generatoren als
    auch zwischen verschiedenen LOD-Stufen desselben Generators. Dies maximiert die
    Ausnutzung moderner Multi-Core-Systeme und minimiert Wartezeiten.

    Performance-Monitoring:
    Der Orchestrator sammelt kontinuierlich Performance-Metriken: Generation-Timings
    pro LOD-Stufe, Memory-Usage pro Thread und System-Resource-Utilization. Diese
    Daten ermöglichen Performance-Optimierung und frühzeitige Erkennung von Bottlenecks.

    Memory-Management-Integration:
    Enge Integration mit dem DataLODManager's ResourceTracker ermöglicht intelligente
    Memory-Allocation. Bei Memory-Warnings wird automatisch Garbage-Collection
    ausgelöst und Large-Array-Processing speziell behandelt. Thread-Completion
    triggert automatische Memory-Cleanup.

    Resource-Coordination:
    Der Orchestrator koordiniert sich mit anderen System-Komponenten bezüglich
    Ressourcennutzung. ShaderManager-Integration optimiert Thread-Allocation
    basierend auf GPU-Usage. DataManager-Coordination verhindert simultane
    Large-Data-Operations.

    ═══════════════════════════════════════════════════════════════════════════════

    6. INTEGRATION UND DATENVERWALTUNG

    DataLODManager-Integration:
    Der Orchestrator nutzt den DataLODManager als zentrale Datenquelle und -senke.
    Alle Dependencies werden über dessen DEPENDENCY_MATRIX aufgelöst. Generated
    Results werden automatisch im DataManager gespeichert, kategorisiert nach
    Generator-Type und LOD-Level.

    Signal-System für UI-Updates:
    Eine harmonisierte Signal-Architektur ermöglicht direkte Integration mit
    BaseMapTabs ohne komplexe Handler-Classes. Standardisierte Signals wie
    generation_completed, lod_progression_completed und generation_progress
    gewährleisten konsistente UI-Updates across alle Tab-Types.

    Automatic Result-Storage:
    Nach jeder LOD-Completion werden Ergebnisse automatisch im DataManager
    persistiert. Terrain-Results als Heightmaps, Geology-Results als Rock-Maps,
    Weather-Results als Climate-Data etc. Dies eliminiert manuelle DataManager-Calls
    in den Tabs.

    Tab-Communication-Patterns:
    Tabs verbinden sich direkt mit Orchestrator-Signals für Updates. Request-Submission
    erfolgt über typisierte Builder-Pattern, die Parameter-Validation und
    Default-Value-Injection handhaben. Error-Propagation liefert actionable
    Information für User-Feedback.

    Cross-Component-Coordination:
    Der Orchestrator koordiniert sich mit anderen Manager-Komponenten für optimale
    System-Performance. Cache-Invalidation bei Parameter-Changes wird mit dem
    DataManager koordiniert. Shader-Resource-Allocation erfolgt in Abstimmung mit
    dem ShaderManager.

    ═══════════════════════════════════════════════════════════════════════════════

    7. ROBUSTHEIT UND FEHLERBEHANDLUNG

    Graceful Degradation:
    Das System ist darauf ausgelegt, auch bei partiellen Fehlern funktionsfähig
    zu bleiben. Einzelne Generator-Failures führen nicht zum System-Ausfall.
    Automatic Fallback auf Legacy-Generation-Methods erfolgt bei kritischen
    Orchestrator-Problemen. UI-Responsiveness bleibt auch bei
    Background-Generation-Overload erhalten.

    Recovery-Mechanismen:
    Transiente Fehler werden automatisch mit bis zu drei Retry-Versuchen behandelt.
    Thread-Pool-Recovery nach Thread-Crashes stellt kontinuierliche Verfügbarkeit
    sicher. Queue-State-Recovery nach Memory-Shortage verhindert Request-Loss.
    Background-Thread-Monitoring mit Health-Checks erkennt frühzeitig Probleme.

    Timeout-Management:
    Mehrstufiges Timeout-Management verhindert System-Blockierung:
    - Request-Timeout von fünf Minuten pro Generation-Request
    - Queue-Timeout von zehn Minuten maximaler Verweilzeit
    - Thread-Timeout von drei Sekunden für graceful Termination
    - Deadlock-Prevention durch kontinuierliche Queue-Resolution

    Error-Propagation und User-Feedback:
    Fehler werden mit detailliertem Context an die UI propagiert. Error-Messages
    enthalten actionable Information für Benutzer. Partial-Success-Scenarios werden
    klar kommuniziert. Critical-Error-Recovery-Suggestions helfen bei Problemlösung.

    System-Health-Monitoring:
    Kontinuierliche Überwachung der System-Gesundheit durch:
    - Health-Checks aller Background-Threads
    - Memory-Usage-Monitoring mit Warning-Thresholds
    - Queue-Length-Monitoring zur Deadlock-Prevention
    - Performance-Metrics-Collection für Trend-Analysis

    Shutdown-Gracefully:
    Bei Application-Shutdown werden alle Background-Threads graceful terminiert.
    Laufende Generierungen werden bis Completion fortgesetzt oder sauber abgebrochen.
    Partial-Results werden gespeichert. Resource-Cleanup erfolgt vollständig ohne
    Memory-Leaks.

    ═══════════════════════════════════════════════════════════════════════════════
    """

def shader_manager():
    """
    Path: gui/managers/shader_manager.py
    Date Changed: 08.08.2025

    ═══════════════════════════════════════════════════════════════════════════════

    SHADER MANAGER - GPU-COMPUTE INTEGRATION UND VALIDATION

    Input: Generator-Requests für GPU-beschleunigte Operationen
    Output: GPU-Berechnungen mit Success/Failure-Status und Datenvalidation
    Kern: GPU-Shader-Verwaltung mit Execution-Monitoring und Datenvalidierung

    ═══════════════════════════════════════════════════════════════════════════════

    1. ÜBERSICHT UND ZIELSETZUNG

    Der ShaderManager bietet GPU-Compute-Beschleunigung für performance-kritische
    Operationen aller sechs Generatoren mit fokussierter GPU-Shader-Verwaltung.
    Die Integration erfolgt über Generator-Requests mit klarer Success/Failure-Rückgabe
    und umfassender Shader-Execution-Validation. Fallback-Logik verbleibt vollständig
    in den Generatoren.

    Kernfunktionen:
    - GPU-Compute-Beschleunigung mit OpenGL Compute Shadern
    - GPU-Verfügbarkeits-Prüfung und Shader-Compilation-Management
    - Shader-Execution-Monitoring und Datenvalidierung
    - Success/Failure-Rückgabe für Generator-Fallback-Entscheidungen
    - Performance-Monitoring und automatische GPU-Optimierung

    Architektonische Positionierung:
    Der ShaderManager fungiert als reine GPU-Compute-Schnittstelle zwischen
    den Generatoren und der GPU-Hardware. Er konzentriert sich ausschließlich
    auf GPU-Shader-Verwaltung, während Generatoren eigenständig CPU-Fallbacks
    und Simple-Fallbacks implementieren.

    ═══════════════════════════════════════════════════════════════════════════════

    2. CORE ARCHITECTURE UND KOMPONENTEN

    Performance-Layer-Struktur:
    Der ShaderManager organisiert sich in drei Hauptkomponenten: GPU-Shader-Management
    für Shader-Compilation und -Execution, Execution-Validation für
    Datenintegritäts-Prüfung und Performance-Monitor für kontinuierliche
    GPU-Optimierung.

    GPU-Shader-Management-Komponente:
    Diese Kernkomponente verwaltet OpenGL Compute Shader für parallele
    Berechnung mit 10-50x speedup für Multi-Octave-Noise bei großen LODs.
    LOD-spezifische Shader-Varianten optimieren Performance basierend auf
    Auflösung. GPU-Memory-Management koordiniert sich mit DataLODManager
    für optimale Ressourcennutzung. Shader-Compilation-Status wird kontinuierlich
    überwacht.

    Execution-Validation-Komponente:
    Diese Komponente überwacht Shader-Execution-Status und validiert
    Ergebnis-Daten auf Korrektheit. GPU-Execution-Monitoring erkennt
    Shader-Crashes oder Timeouts. Datenvalidierung prüft Output-Arrays
    auf NaN-Werte, Infinite-Values und Range-Violations. Memory-Corruption-Detection
    identifiziert fehlerhafte GPU-Memory-Operations.

    Performance-Monitor-Komponente:
    Diese Komponente überwacht GPU-Performance-Metriken, sammelt
    Execution-Timing-Stats und protokolliert GPU-Memory-Usage für
    kontinuierliche System-Optimierung. Hardware-Capability-Detection
    ermöglicht optimale Shader-Variant-Selection.

    ═══════════════════════════════════════════════════════════════════════════════

    3. GPU-SHADER-VERWALTUNG UND VALIDATION

    GPU-Shader-Stufe (Optimale Performance):
    Der ShaderManager konzentriert sich ausschließlich auf GPU-Shader-Management
    für maximale Performance-Vorteile. OpenGL Compute Shader ermöglichen
    parallele Berechnung mit dramatischen Performance-Steigerungen.
    Multi-Octave-Noise bei großen LODs erreicht 10-50x speedup gegenüber
    CPU-Implementierung. LOD-spezifische Shader-Varianten optimieren Performance
    basierend auf Auflösung und Komplexität.

    GPU-Verfügbarkeits-Prüfung:
    Das System prüft kontinuierlich GPU-Verfügbarkeit und Shader-Compilation-Status.
    OpenGL-Context-Validation stellt sicher, dass GPU-Operationen möglich sind.
    Shader-Compilation-Error-Detection identifiziert problematische Shader.
    Hardware-Capability-Assessment ermittelt optimale Shader-Variants für
    spezifische GPU-Hardware.

    Shader-Execution-Monitoring:
    Während der Shader-Ausführung überwacht das System kontinuierlich
    Execution-Status und Performance-Metriken. GPU-Timeout-Detection
    erkennt hängende Shader-Operations. Memory-Access-Violation-Detection
    identifiziert GPU-Memory-Probleme. Execution-Progress-Tracking
    ermöglicht Cancellation bei Bedarf.

    Datenvalidierung und Integrität:
    Nach Shader-Completion führt das System umfassende Datenvalidierung durch.
    NaN-Detection identifiziert mathematische Fehler in Shader-Output.
    Range-Validation prüft Output-Werte gegen erwartete Parameter-Bereiche.
    Array-Dimension-Validation stellt korrekte Output-Array-Größen sicher.
    Memory-Corruption-Detection erkennt fehlerhafte GPU-Memory-Transfers.

    Success/Failure-Rückgabe-Interface:
    Der ShaderManager liefert klare Success/Failure-Information an Generatoren
    für Fallback-Entscheidungen. Success-Status enthält validierte Output-Daten
    und Performance-Metriken. Failure-Status spezifiziert Fehler-Typ und
    Error-Details für Generator-Debugging. Execution-Timing-Information
    ermöglicht Performance-Optimierung in Generatoren.

    ═══════════════════════════════════════════════════════════════════════════════

    4. GENERATOR-INTEGRATION PATTERN

    Generator-Fallback-Pattern-Implementierung:
    Jeder Generator implementiert eigenständige Fallback-Logik mit CPU-Optimized
    und Simple-Fallback-Methoden. Der Generator prüft zunächst ShaderManager-Verfügbarkeit,
    führt GPU-Request durch und entscheidet basierend auf Success/Failure-Rückgabe
    über Fallback-Activation. CPU-Fallbacks und Simple-Fallbacks sind vollständig
    Generator-internal implementiert.

    TerrainGenerator-Integration-Muster:
    Der TerrainGenerator implementiert calculate_heightmap mit drei Stufen:
    GPU-Request über ShaderManager für multi_octave_noise, CPU-Fallback mit
    optimierten NumPy-Operations bei GPU-Failure, Simple-Fallback mit
    basic noise generation bei CPU-Problemen. Generator entscheidet autonom
    über Fallback-Progression basierend auf ShaderManager-Response.

    ShaderManager-Request-Interface:
    Der ShaderManager-Request erfolgt mit Operation-Type, Input-Data,
    Parameter-Dictionary und LOD-Level-Spezifikation. Response enthält
    Success-Boolean, Output-Data bei Success, Error-Type und Error-Details
    bei Failure, Execution-Time-Metrics für Performance-Tracking.
    Generator implementiert Fallback-Logic basierend auf Response-Status.

    Generator-Autonome-Fallback-Entscheidung:
    Bei ShaderManager-Success verwendet Generator GPU-Output direkt.
    Bei ShaderManager-Failure aktiviert Generator CPU-Fallback mit
    optimierten NumPy-Vectorization und Multiprocessing. Bei CPU-Fallback-Failure
    aktiviert Generator Simple-Fallback mit minimal-code Implementation.
    Jede Fallback-Stufe ist Generator-internal ohne ShaderManager-Dependency.

    ═══════════════════════════════════════════════════════════════════════════════

    5. SHADER OPERATIONS FÜR ALLE GENERATOREN

    Terrain-Generator-Operations:
    Der ShaderManager bietet request_noise_generation für Multi-Octave
    Simplex-Noise mit GPU-parallelisierter Octave-Berechnung.
    Request_shadow_calculation implementiert parallele Raycast-Shadows
    für realistische Beleuchtung. Request_slope_calculation führt
    Gradient-Field-Computation für Slope-Maps durch.

    Geology-Generator-Operations:
    Request_geological_zones implementiert Multi-Zone-Noise-Classification
    für geologische Gesteinsverteilung. Request_deformation_processing
    führt Tectonic-Warping-Operations für geologische Verformung durch.
    Request_mass_conservation implementiert Parallel-Normalization-Operations
    für geologische Konsistenz.

    Weather-Generator-Operations:
    Request_wind_field_calculation implementiert CFD-Based Wind-Simulation
    für realistische Luftströmungen. Request_temperature_distribution
    führt Thermal-Gradient-Processing für Klimazonen durch.
    Request_precipitation_modeling implementiert Orographic-Precipitation-Calculation
    für höhenabhängige Niederschläge.

    Water-Generator-Operations:
    Request_erosion_simulation implementiert Hydraulic-Erosion-Processing
    für realistische Landschaftsformung. Request_flow_accumulation
    führt Watershed-Analysis-Computation für Wassereinzugsgebiete durch.
    Request_water_distribution implementiert Flow-Network-Calculation
    für Flusssystem-Generierung.

    Biome-Generator-Operations:
    Request_biome_blending implementiert Multi-Factor-Biome-Classification
    basierend auf Klima, Höhe und Wasser. Request_vegetation_distribution
    führt Density-Map-Generation für Vegetationsdichte durch.
    Request_biome_transitions implementiert Smooth-Transition-Processing
    für natürliche Biom-Übergänge.

    Settlement-Generator-Operations:
    Request_suitability_analysis implementiert Multi-Criteria-Suitability-Mapping
    für optimale Siedlungsplatzierung. Request_network_optimization
    führt Road/River-Network-Processing für Infrastruktur durch.
    Request_settlement_distribution implementiert Population-Density-Calculation
    für realistische Bevölkerungsverteilung.

    ═══════════════════════════════════════════════════════════════════════════════

    6. LOD-OPTIMIZATION SYSTEM

    LOD-Level-Spezifische Shader-Varianten:
    Das System implementiert verschiedene Shader-Varianten basierend auf
    LOD-Level und Grid-Size. Für niedrige LODs bis Level 2 mit 32x32 und
    64x64 Auflösung werden Fast-Shader-Variants verwendet, die Performance
    über Qualität priorisieren. Mittlere LODs bis Level 4 mit 128x128 und
    256x256 nutzen Balanced-Shader-Variants für optimalen Quality-Performance-Tradeoff.
    Hohe LODs ab Level 5 mit 512x512+ verwenden Quality-Shader-Variants
    für maximale Bildqualität.

    Performance-Adaptive-Selection-Mechanismus:
    Das System führt GPU-Performance-Profiling für den ersten LOD-Level
    durch, um Hardware-Capabilities zu ermitteln. Automatic Fallback-Selection
    basiert auf Performance-Thresholds, die aus diesem Profiling abgeleitet
    werden. Memory-Usage-Monitoring überwacht GPU-Memory-Constraints für
    intelligente Ressourcenallokation. LOD-Level-Performance-Caching
    speichert Ergebnisse für zukünftige Requests.

    Adaptive-Optimization-Strategien:
    Das System lernt aus Performance-Daten und passt Shader-Selection
    dynamisch an. Bei wiederholten Performance-Problemen erfolgt automatischer
    Downgrade zu einfacheren Shader-Varianten. Memory-Constraints führen
    zu progressiver LOD-Size-Reduction bis GPU-Memory-Fit erreicht wird.
    Thread-Allocation wird basierend auf GPU-Utilization optimiert.

    Quality-Performance-Balancing:
    Für jeden LOD-Level wird der optimale Quality-Performance-Balance
    automatisch ermittelt. Fast-Variants reduzieren Octave-Count und
    Sampling-Rate für maximale Speed. Balanced-Variants maintainen
    algorithmic correctness bei moderater Performance. Quality-Variants
    implementieren Full-Resolution Processing mit allen Features.

    ═══════════════════════════════════════════════════════════════════════════════

    7. SIGNAL INTEGRATION UND DATALODMANAGER-HUB

    Outgoing-Signals-Architektur:
    Das Signal-System kommuniziert kontinuierlich mit dem DataLODManager
    über vier Hauptsignale. Shader_status_changed übermittelt Generator-Name,
    Operation-Type, GPU-Usage-Status und Performance-Information.
    Shader_performance_optimized kommuniziert Generator, LOD-Level und
    angewendete Optimization. GPU_memory_warning übermittelt aktuelle
    Memory-Usage und Threshold-Werte. Fallback_strategy_activated
    kommuniziert Generator, Operation und aktivierte Fallback-Stufe.

    DataLODManager-Integration-Pattern:
    Der DataLODManager empfängt Shader-Performance-Information über
    shader_status_changed Signale und integriert diese in das globale
    Performance-Monitoring. Performance-Information wird an UI-Komponenten
    weitergeleitet über shader_info_updated Signale. Memory-Warnings
    triggern koordinierte Resource-Cleanup-Aktionen zwischen ShaderManager
    und DataLODManager.

    UI-Status-Display-Integration:
    Das System bietet umfassende UI-Integration für Benutzer-Information.
    Active-Fallback-Strategy-Display zeigt aktuell verwendete Performance-Stufe.
    GPU-Utilization-Stats ermöglichen Performance-Monitoring durch Benutzer.
    Memory-Usage-Display zeigt GPU-Memory-Consumption für Resource-Management.
    Shader-Compilation-Status bietet detaillierte Information für Debugging.

    Signal-Flow-Koordination:
    Shader-Requests triggern Performance-Monitoring-Signals an DataLODManager.
    Memory-Warnings vom DataLODManager führen zu Shader-Memory-Cleanup.
    LOD-Progression-Signals koordinieren Shader-Optimization mit
    Generation-Progress. Error-Signals ermöglichen koordinierte
    Fehlerbehandlung zwischen Komponenten.

    ═══════════════════════════════════════════════════════════════════════════════

    8. GPU-EXECUTION-VALIDATION UND ERROR-DETECTION

    GPU-Execution-Monitoring-System:
    Das System überwacht kontinuierlich GPU-Shader-Execution für Fehler-Detection
    und Performance-Tracking. Execution-Timeout-Detection erkennt hängende
    Shader-Operations mit konfigurierbaren Timeout-Werten. GPU-Memory-Access-Monitoring
    identifiziert Memory-Violations und Invalid-Pointer-Access.
    Shader-Crash-Detection erkennt GPU-Driver-Exceptions und Hardware-Failures.

    Datenvalidierung-nach-Execution:
    Nach Shader-Completion führt das System umfassende Output-Datenvalidierung
    durch. NaN-Value-Detection identifiziert mathematische Errors in
    Floating-Point-Calculations. Infinite-Value-Detection erkennt
    Division-by-Zero und Overflow-Conditions. Range-Validation prüft
    Output-Values gegen erwartete Parameter-Ranges für Plausibility-Checks.

    Memory-Integrity-Validation:
    Das System validiert GPU-Memory-Integrity vor und nach Shader-Execution.
    Buffer-Size-Validation stellt korrekte Input/Output-Buffer-Dimensionen sicher.
    Memory-Corruption-Detection identifiziert fehlerhafte GPU-Memory-Transfers.
    Data-Type-Validation prüft korrekte Data-Type-Mapping zwischen CPU und GPU.

    Error-Classification-und-Reporting:
    Errors werden in kategorisierte Types klassifiziert für spezifische
    Generator-Response. Hardware-Errors indizieren GPU-Driver-Probleme
    oder Hardware-Failures. Shader-Compilation-Errors zeigen Code-Probleme
    in Shader-Implementation. Data-Validation-Errors indizieren Input-Parameter-Probleme
    oder Algorithm-Bugs. Performance-Errors zeigen Timeout oder Resource-Exhaustion.

    Automatic-Error-Recovery-Mechanisms:
    Bei detektierten Errors führt das System automatische Recovery-Versuche durch.
    Memory-Cleanup bei Memory-Corruption resettet GPU-Memory-State.
    Shader-Recompilation bei Compilation-Errors versucht Error-Correction.
    Context-Reset bei Hardware-Errors reinitialisiert GPU-Context.
    Timeout-Extension bei Performance-Errors erweitert Execution-Limits.

    ═══════════════════════════════════════════════════════════════════════════════

    9. PERFORMANCE MONITORING UND OPTIMIZATION

    Performance-Metrics-Collection-System:
    Das System sammelt umfassende Performance-Metriken für kontinuierliche
    Optimierung. GPU-Execution-Times werden per Operation in Listen gespeichert
    für Trend-Analysis. CPU-Fallback-Times ermöglichen Vergleichsanalysen
    zwischen Performance-Stufen. Memory-Usage-Peaks dokumentieren
    Peak-Memory-Consumption per Operation für Memory-Optimization.

    Fallback-Frequency-Analysis:
    Fallback-Frequencies werden per Operation und Strategy-Type getrackt
    für System-Health-Monitoring. LOD-Performance-Scaling dokumentiert
    Performance-Verhältnis zwischen LOD-Levels für Optimization-Decisions.
    Hardware-Specific-Performance-Profiles ermöglichen adaptive Strategy-Selection.

    Adaptive-Optimization-Implementation:
    Performance-Threshold-Learning ermöglicht automatische Fallback-Selection
    basierend auf Historical-Performance-Data. LOD-Level-Performance-Prediction
    optimiert Thread-Allocation für maximale Effizienz. Memory-Usage-Prediction
    ermöglicht proaktive GPU-Memory-Management. Generator-Specific-Performance-Profiling
    ermöglicht Custom-Optimizations per Generator-Type.

    Resource-Management-Coordination:
    GPU-Texture-Memory-Pooling implementiert wiederverwendbare Resource-Allocation.
    Automatic-GPU-Memory-Cleanup erfolgt bei DataLODManager-Memory-Warnings.
    CPU-Memory-Coordination mit DataLODManager-Resource-Tracker optimiert
    System-Resource-Usage. Thread-Pool-Coordination mit GenerationOrchestrator
    verhindert Resource-Conflicts.

    Real-Time-Performance-Adaptation:
    Das System passt Performance-Strategies in Echtzeit basierend auf
    aktueller System-Load an. Dynamic-Quality-Adjustment reduziert
    Shader-Complexity bei Performance-Bottlenecks. Automatic-LOD-Downgrade
    bei Memory-Constraints erhält System-Stability. Predictive-Optimization
    basierend auf Historical-Data minimiert Fallback-Frequency.

    ═══════════════════════════════════════════════════════════════════════════════

    10. SYSTEM-RESILIENCE UND GPU-MANAGEMENT

    GPU-Hardware-Resilience-Strategies:
    Das System implementiert umfassende Resilience-Mechanisms für
    GPU-Hardware-Probleme. GPU-Driver-Crash-Recovery reinitialisiert
    GPU-Context nach Driver-Failures. Hardware-Failure-Detection
    identifiziert defekte GPU-Hardware durch Error-Pattern-Analysis.
    Automatic-GPU-Disable bei wiederholten Hardware-Failures schützt
    System-Stability.

    Shader-Compilation-Error-Management:
    Bei Shader-Compilation-Problemen führt das System systematische
    Error-Analysis durch. Syntax-Error-Detection identifiziert
    Code-Probleme in Shader-Source. Hardware-Compatibility-Issues
    werden durch Feature-Detection erkannt. Automatic-Shader-Fallback
    zu einfacheren Shader-Variants bei Compilation-Failures.

    Memory-Management-Resilience:
    GPU-Memory-Management implementiert robuste Error-Recovery für
    Memory-Probleme. Memory-Allocation-Failure-Recovery reduziert
    Buffer-Sizes und versucht erneute Allocation. Memory-Leak-Detection
    überwacht GPU-Memory-Usage für Leak-Prevention.
    Automatic-Memory-Cleanup bei kritischen Memory-Levels.

    Performance-Degradation-Handling:
    Bei Performance-Problemen implementiert das System adaptive
    Response-Strategies. Timeout-Extension für langsame GPU-Hardware
    erweitert Execution-Limits. Quality-Reduction reduziert
    Shader-Complexity bei Performance-Bottlenecks.
    Automatic-LOD-Downgrade bei persistenten Performance-Issues.

    System-State-Recovery:
    Das System implementiert umfassende State-Recovery-Mechanisms.
    GPU-Context-Reinitialization nach Context-Loss stellt
    Functionality wieder her. Shader-Program-Reload nach GPU-Reset
    recompiliert alle Shaders. Buffer-Reallocation nach Memory-Corruption
    reinitialisiert GPU-Memory-Structures.

    ═══════════════════════════════════════════════════════════════════════════════

    11. INTEGRATION SUMMARY UND SYSTEM-BENEFITS

    Generator-ShaderManager-Interface-Flow:
    Der überarbeitete Ablauf beginnt mit Generator-Method-Call für spezifische
    GPU-Operation. ShaderManager-Request mit Operation-Parameters wird erstellt.
    ShaderManager prüft GPU-Verfügbarkeit und führt Shader-Execution durch.
    Success/Failure-Response mit Daten oder Error-Information wird an Generator
    zurückgegeben. Generator implementiert eigenständige Fallback-Logic
    basierend auf ShaderManager-Response.

    Klare Verantwortlichkeitstrennung:
    ShaderManager konzentriert sich ausschließlich auf GPU-Shader-Management,
    Execution-Monitoring und Datenvalidierung. Generatoren implementieren
    vollständige Fallback-Logic mit CPU-Optimized und Simple-Fallback-Methoden.
    Diese Trennung reduziert Complexity und verbessert Maintainability
    durch klare Interface-Definitionen.

    System-Benefits-durch-Fokussierung:
    Performance-Benefit durch spezialisierte GPU-Shader-Optimization ohne
    Fallback-Overhead. Reliability-Benefit durch robuste Shader-Execution-Validation
    und Error-Detection. Maintainability-Benefit durch klare Verantwortlichkeitstrennung
    zwischen ShaderManager und Generatoren. Scalability-Benefit durch
    fokussierte GPU-Resource-Management.

    Generator-Autonomie-Vorteile:
    Generator-autonome Fallback-Implementation ermöglicht spezifische
    Optimization pro Generator-Type. CPU-Fallback-Methods können
    Generator-spezifisch optimiert werden. Simple-Fallback-Implementation
    kann Domain-Knowledge des Generators nutzen. Fallback-Timing-Control
    liegt vollständig beim Generator.

    Development-und-Debugging-Benefits:
    Simplified-ShaderManager-Interface reduziert Integration-Complexity.
    Clear-Error-Reporting ermöglicht präzises Debugging von GPU-Issues.
    Generator-focused-Fallback-Logic vereinfacht Generator-Development.
    Separated-Concerns erleichtern Unit-Testing und System-Validation.

    ═══════════════════════════════════════════════════════════════════════════════
    Der komplette Ablauf beginnt mit Generator-Method-Call für spezifische
    Operation. ShaderManager-Request mit Fallback-Function wird erstellt.
    3-stufiges Fallback-System garantiert Operation-Success: GPU-Shader
    bei Verfügbarkeit, CPU-Fallback bei GPU-Problemen, Simple-Fallback
    als Ultimate-Guarantee. Performance-Stats werden an DataLODManager
    übermittelt für UI-Status-Update.

    System-Benefits-Realization:
    Performance-Benefit durch GPU-Acceleration erreicht dramatische
    Speedups wo Hardware verfügbar ist. Reliability-Benefit durch
    Guaranteed-Success eliminiert Generation-Failures vollständig.
    Scalability-Benefit durch LOD-optimierte Shader-Variants passt
    Performance an Anforderungen an. Maintainability-Benefit durch
    Simple-Fallbacks direkt in Generatoren reduziert System-Complexity.

    User-Experience-Optimization:
    Transparent-Fallback ohne Generation-Failures bietet nahtlose
    User-Experience unabhängig von Hardware-Capabilities. Performance-Feedback
    informiert Benutzer über aktuelle System-Performance ohne Technical-Overwhelm.
    Quality-Options ermöglichen User-Control über Performance-Quality-Tradeoffs.
    Debugging-Support assistiert bei Performance-Optimization und Troubleshooting.

    Development-Benefits:
    Unified-Interface für alle Generator-Types reduziert Integration-Complexity.
    Standardized-Fallback-Pattern vereinfacht Generator-Development.
    Performance-Monitoring-Integration ermöglicht Data-driven-Optimization.
    Error-Handling-Abstraction reduziert Fehlerbehandlung in Generatoren.

    ═══════════════════════════════════════════════════════════════════════════════
    """

def navigation_manager():
    """
    Path: gui/managers/navigation_manager.py

    Funktionsweise: Tab-Navigation und Fenster-Geometrie-Verwaltung ohne Interferenz mit Background-Generation
    - NavigationManager koordiniert Tab-Wechsel mit Multi-Monitor-Support und Geometrie-Persistierung
    - Tab-Sequence-Management mit Previous/Next-Navigation und Keyboard-Shortcuts
    - Window-Geometry-System für Tab-spezifische Fenster-Konfigurationen
    - Navigation-State-Tracking für UI-Control-Updates und User-Experience

    Parameter Input:
    - target_tab [string]: Ziel-Tab aus TAB_SEQUENCE (main_menu, terrain, geology, water, biome, settlement, overview)
    - save_geometry [boolean]: Geometrie vor Tab-Wechsel speichern
    - restore_history [boolean]: Navigation-Historie wiederherstellen
    - keyboard_navigation [boolean]: Keyboard-Shortcut-Unterstützung aktivieren

    Dependencies:
    - Keine direkten Dependencies zu DataLODManager, ParameterManager oder GenerationOrchestrator
    - Read-Only Signal-Empfang für Status-Display von anderen Managern
    - UI-Integration über MainWindow-Interface

    Output:
    - NavigationResult-Objekt mit current_tab, previous_tab, navigation_success
    - WindowGeometry-Daten für Tab-spezifische Fenster-Restaurierung
    - NavigationState-Updates für UI-Control-Synchronisation

    Tab-Navigation-System (Simplified):
    - Tab-Sequence: main_menu → terrain → geology → water → biome → settlement → overview
    - Bidirectional-Navigation mit Previous/Next-Controls
    - Direct-Tab-Access über Tab-Names ohne Dependency-Validation
    - Background-Generation-Independence: Navigation unterbricht keine laufenden Prozesse

    Fallback-System (3-stufig):
    - Standard-Navigation: Vollständige Geometrie-Persistierung mit Multi-Monitor-Support
    - Reduzierte-Navigation: Basis-Tab-Wechsel ohne komplexe Geometrie-Features bei Speicher-Problemen
    - Minimal-Navigation: Einfacher Tab-Wechsel ohne Persistierung bei kritischen Fehlern

    Window-Geometry-Management:
    1. Tab-spezifische Geometrie-Speicherung (Position, Größe, Monitor-Assignment)
    2. Multi-Monitor-Unterstützung mit automatischer Monitor-Detection
    3. Resolution-Change-Adaptation ohne Navigation-Unterbrechung
    4. Splitter-Position-Persistierung für Canvas/Control-Panel-Aufteilung
    5. Minimized/Maximized-State-Recovery pro Tab

    Navigation-State-Tracking:
    1. Current-Tab-Management ohne Business-Logic-Validation
    2. Navigation-History für Back/Forward-Funktionalität
    3. Recently-Used-Tabs für Quick-Access-Features
    4. Tab-Status-Display basierend auf Signal-Inputs von anderen Managern

    Background-Generation-Independence:
    - Tab-Wechsel haben keine Auswirkungen auf laufende Generation-Threads
    - Background-Processes empfangen weiterhin Updates unabhängig vom aktiven Tab
    - Navigation-Events triggern keine Parameter-Validierung oder Dependency-Checks
    - Graceful-Navigation ohne Unterbrechung von Workflows

    Error-Recovery-Mechanisms:
    - Invalid-Tab-Recovery mit Fallback zu main_menu
    - Corrupted-Geometry-Recovery mit Default-Window-Configuration
    - Monitor-Unavailability-Recovery mit Primary-Monitor-Fallback
    - Config-File-Corruption-Recovery mit Built-in-Default-Values

    Performance-Characteristics:
    - Navigation-Operations unter 10ms für instant User-Response
    - Minimal Memory-Footprint durch Elimination komplexer Business-Logic
    - Cached-Geometry-Lookups für Performance-optimierte Window-Restaurierung
    - Non-Blocking-Operations während Navigation-Prozessen

    Klassen:
    NavigationManager
        Funktionsweise: Hauptklasse für Tab-Navigation und Window-Management
        Aufgabe: Koordiniert Tab-Wechsel, Geometrie-Persistierung und UI-State-Updates
        External-Interface: navigate_to_tab(target_tab), get_current_tab(), save_window_state() - wird von MainWindow aufgerufen
        Internal-Methods: _coordinate_navigation(), _validate_tab_target(), _update_ui_controls()
        Dependencies: Keine direkten Dependencies, empfängt Status-Signals von anderen Managern
        Error-Handling: Graceful Degradation bei Navigation-Problemen, Fallback-Chain ohne Workflow-Disruption

    TabSequenceManager
        Funktionsweise: Verwaltet Tab-Reihenfolge und Navigation-Logik für Previous/Next-Operations
        Aufgabe: Bestimmt Target-Tab basierend auf aktueller Position und Navigation-Direction
        Methoden: get_next_tab(), get_previous_tab(), get_tab_index(), validate_tab_sequence()
        Tab-Sequence: Linear-Array mit Wrap-Around-Logic für seamless Navigation
        Spezifische Fallbacks:
          - Standard-Sequence: Vollständige TAB_SEQUENCE mit allen Generator-Tabs
          - Reduced-Sequence: Basis-Tabs (main_menu, terrain, overview) bei Memory-Constraints
          - Minimal-Sequence: Nur main_menu und overview bei kritischen System-Zuständen

    WindowGeometryController
        Funktionsweise: Speichert und restauriert Tab-spezifische Fenster-Geometrie mit Multi-Monitor-Support
        Aufgabe: Geometrie-Persistierung, Monitor-Assignment, Resolution-Adaptation
        Methoden: save_current_geometry(), restore_geometry_for_tab(), adapt_to_display_change()
        Geometry-Storage: Tab-spezifische Dictionaries mit width, height, x, y, monitor_id
        Spezifische Fallbacks:
          - Full-Geometry: Komplette Geometrie-Persistierung mit Multi-Monitor-Daten
          - Basic-Geometry: Standard-Größe ohne Position-Tracking bei Storage-Problemen
          - Default-Geometry: Fixed-Layout (1400x900) bei Geometry-System-Failures

    NavigationStateTracker
        Funktionsweise: Verfolgt Navigation-Zustand und Historie für UI-Control-Updates
        Aufgabe: Current-Tab-Management, Navigation-History, UI-Status-Synchronisation
        Methoden: update_current_tab(), add_to_history(), get_navigation_state(), reset_navigation_state()
        State-Management: Current-Tab, Previous-Tab, Navigation-Direction, History-Stack
        Spezifische Fallbacks:
          - Full-Tracking: Komplette Navigation-Historie mit Timestamps und User-Actions
          - Basic-Tracking: Current/Previous-Tab ohne detaillierte Historie bei Memory-Limits
          - Minimal-Tracking: Nur Current-Tab ohne Historie bei kritischen Zuständen

    NavigationControlsManager
        Funktionsweise: Verwaltet Navigation-UI-Controls und Keyboard-Shortcuts
        Aufgabe: Button-State-Updates, Keyboard-Handling, Control-Synchronisation mit Navigation-State
        Methoden: update_button_states(), handle_keyboard_shortcuts(), sync_with_navigation_state()
        Control-Elements: Previous/Next-Buttons, Tab-Indicators, Keyboard-Shortcuts (Ctrl+PageUp/Down)
        Spezifische Fallbacks:
          - Full-Controls: Alle Navigation-Controls mit Keyboard-Shortcuts und Visual-Indicators
          - Basic-Controls: Previous/Next-Buttons ohne komplexe Keyboard-Handling
          - Minimal-Controls: Nur Direct-Tab-Selection ohne Navigation-Enhancements

    Integration und Datenfluss:
    MainWindow → NavigationManager.navigate_to_tab(target_tab)
              → TabSequenceManager.determine_target_tab() → Validation
              → WindowGeometryController.save_current_geometry() → Persistence
              → NavigationStateTracker.update_navigation_state() → State-Updates
              → NavigationControlsManager.update_ui_controls() → UI-Synchronisation
              → MainWindow.switch_active_tab() → UI-Display-Update

    Signal-Architecture:

    Outgoing Signals (NavigationManager → UI):
    tab_changed = pyqtSignal(str, str)              # (from_tab, to_tab)
    navigation_state_updated = pyqtSignal(dict)     # (current_tab, can_go_back, can_go_forward)
    window_geometry_changed = pyqtSignal(str, dict) # (tab_name, geometry_data)
    navigation_error = pyqtSignal(str, str)         # (error_type, error_message)

    Incoming Signals (Andere Manager → NavigationManager für Status-Display):
    data_lod_manager.tab_status_changed.connect(navigation_manager.update_tab_status_indicators)
    generation_orchestrator.background_status.connect(navigation_manager.update_generation_indicators)
    parameter_manager.parameter_validation_state.connect(navigation_manager.update_parameter_indicators)

    Output-Datenstrukturen:
    - NavigationResult: success_state, current_tab, previous_tab, navigation_time, error_info
    - WindowGeometry: width, height, x_position, y_position, monitor_id, maximized_state
    - NavigationState: current_tab, tab_history, can_navigate_back, can_navigate_forward, active_shortcuts
    - ControlState: button_states, keyboard_enabled, visual_indicators, status_messages
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
    Path: gui/tabs/base_tab_refactored.py
    date_changed: 07.08.2025

    =========================================================================================
    BASEMAPTAB - VOLLSTÄNDIGE REFACTORED IMPLEMENTATION
    =========================================================================================

    ARCHITECTURE OVERVIEW:
    ----------------------

    BaseMapTab ist die fundamentale Basis-Klasse für alle spezialisierten Map-Editor Tabs.
    Sie implementiert ein standardisiertes 70/30 Layout-System, Cross-Tab-Kommunikation,
    automatische Generation-Workflows und robustes Resource-Management.

    DESIGN PRINCIPLES:
    ------------------
    1. MODULARE ARCHITEKTUR: Jede Funktionalität ist in separate, testbare Methoden aufgeteilt
    2. RESOURCE SAFETY: Systematisches Tracking und Cleanup aller UI-Ressourcen
    3. ERROR RESILIENCE: Jede kritische Operation hat Exception-Handling
    4. SIGNAL ORCHESTRATION: Standardisierte Cross-Tab-Kommunikation über Signals
    5. MEMORY EFFICIENCY: Change-Detection verhindert unnötige Display-Updates
    6. EXTENSIBILITY: Sub-Classes können jede Funktionalität selektiv überschreiben

    CORE COMPONENTS:
    ----------------

    ┌─────────────────┬─────────────────────────────────────────────────────────────┐
    │ COMPONENT       │ RESPONSIBILITY                                               │
    ├─────────────────┼─────────────────────────────────────────────────────────────┤
    │ Layout System   │ 70/30 Canvas/Control Split mit dynamischer Größenanpassung │
    │ Display Stack   │ 2D/3D View-Toggle mit Fallback-Management                  │
    │ Auto-Simulation │ Parameter-Change-Detection und Auto-Generation             │
    │ Orchestrator    │ Integration mit GenerationOrchestrator für LOD-Progression │
    │ Resource Track  │ Memory-Leak-Prevention durch systematisches Cleanup        │
    │ Signal Hub      │ Cross-Tab-Communication und Data-Change-Notifications      │
    └─────────────────┴─────────────────────────────────────────────────────────────┘

    SIGNAL ARCHITECTURE:
    --------------------

    OUTGOING SIGNALS (BaseMapTab → Other Components):
    • data_updated(generator_type: str, data_key: str)
      - Emittiert wenn Tab neue Daten generiert hat
      - Andere Tabs können darauf reagieren und ihre Dependencies prüfen

    • parameter_changed(generator_type: str, parameters: dict)
      - Emittiert bei Parameter-Änderungen
      - Triggert Auto-Simulation in anderen Tabs falls aktiviert

    • validation_status_changed(generator_type: str, is_valid: bool, messages: list)
      - Emittiert bei Dependency-Status-Änderungen
      - Navigation-Manager kann Tab-Verfügbarkeit entsprechend anpassen

    • generation_completed(generator_type: str, success: bool)
      - Emittiert nach Abschluss einer Generation
      - Ermöglicht Chain-Generationen zwischen Tabs

    INCOMING SIGNALS (Other Components → BaseMapTab):
    • data_manager.data_updated → on_external_data_updated()
      - Reagiert auf Daten-Updates von anderen Generatoren
      - Prüft Dependencies und triggert Display-Updates

    • data_manager.cache_invalidated → on_cache_invalidated()
      - Reagiert auf Cache-Invalidierung
      - Refresht Display falls nötig

    • generation_orchestrator.* → _on_generation_*()
      - Empfängt LOD-Progression-Updates vom Orchestrator
      - Aktualisiert UI-Status und Progress-Anzeigen

    UI ARCHITECTURE:
    ----------------

    MAIN LAYOUT STRUCTURE:
    ┌──────────────────────────────────────────────────────────────────────────────────┐
    │ BaseMapTab (QWidget)                                                             │
    │ ┌──────────────────────────────────────────────────────────────────────────────┐ │
    │ │ QHBoxLayout (main_layout)                                                    │ │
    │ │ ┌───────────────────────────┬──────────────────────────────────────────────┐ │ │
    │ │ │ Canvas Container (70%)    │ Control Widget (30%)                         │ │ │
    │ │ │ ┌───────────────────────┐ │ ┌──────────────────────────────────────────┐ │ │ │
    │ │ │ │ View Toggle Buttons   │ │ │ QVBoxLayout                              │ │ │ │
    │ │ │ └───────────────────────┘ │ │ ┌──────────────────────────────────────┐ │ │ │ │
    │ │ │ ┌───────────────────────┐ │ │ │ QScrollArea                          │ │ │ │ │
    │ │ │ │ QStackedWidget        │ │ │ │ ┌──────────────────────────────────┐ │ │ │ │ │
    │ │ │ │ ├─ 2D Display         │ │ │ │ │ Control Panel (QWidget)          │ │ │ │ │ │
    │ │ │ │ └─ 3D Display         │ │ │ │ │ ┌─ Parameter Controls (Sub-Class)│ │ │ │ │ │
    │ │ │ └───────────────────────┘ │ │ │ │ ├─ Auto-Simulation Panel         │ │ │ │ │ │
    │ │ └───────────────────────────┘ │ │ │ └─ Error Status Display          │ │ │ │ │ │
    │ │                               │ │ └──────────────────────────────────┘ │ │ │ │ │
    │ │                               │ └──────────────────────────────────────┘ │ │ │ │
    │ │                               │ ┌──────────────────────────────────────┐ │ │ │ │
    │ │                               │ │ Navigation Panel (not scrollable)    │ │ │ │ │
    │ │                               │ └──────────────────────────────────────┘ │ │ │ │
    │ │                               └──────────────────────────────────────────┘ │ │ │
    │ └────────────────────────────────────────────────────────────────────────────┘ │ │
    │                                                                                │ │
    │ QSplitter (Canvas ↔ Control resizable, 70/30 ratio maintained)                 │ │
    └────────────────────────────────────────────────────────────────────────────────┘ │
                                                                                       │
    └──────────────────────────────────────────────────────────────────────────────────┘

    CONTROL PANEL COMPOSITION:
    ┌─────────────────────────────────────┐
    │ QScrollArea (scrollable content)    │
    │ ┌─────────────────────────────────┐ │
    │ │ Control Panel (QWidget)         │ │  ← Sub-Classes add content here
    │ │ ┌─ Custom Parameter Controls   │ │  ← Via control_panel.layout()
    │ │ ├─ Auto-Simulation Panel       │ │  ← Standardized across all tabs
    │ │ └─ Error Status Display        │ │  ← Shows dependency/generation status
    │ └─────────────────────────────────┘ │
    └─────────────────────────────────────┘
    ┌─────────────────────────────────────┐
    │ Navigation Panel (fixed)            │  ← Always visible, not scrollable
    └─────────────────────────────────────┘

    LIFECYCLE MANAGEMENT:
    ---------------------

    INITIALIZATION SEQUENCE:
    1. __init__(): Core attribute setup, dependency injection
    2. setup_ui(): UI structure creation with error handling
       ├─ setup_layout_structure(): 70/30 split, splitter configuration
       ├─ setup_display_stack(): 2D/3D views with fallback handling
       ├─ setup_control_panel(): Scrollable parameter area
       ├─ setup_navigation_panel(): Fixed navigation (not scrollable)
       ├─ setup_auto_simulation(): Auto-generation controls
       └─ setup_error_handling(): Status display and error recovery
    3. setup_signals(): Cross-tab communication setup
    4. setup_standard_orchestrator_handlers(): LOD-progression integration

    CLEANUP SEQUENCE:
    1. cleanup_resources(): Systematic resource deallocation
       ├─ _safe_disconnect_orchestrator_signals(): Exception-safe disconnects
       ├─ _safe_disconnect_data_manager_signals(): Exception-safe disconnects
       ├─ resource_tracker.force_cleanup_all(): Memory leak prevention
       ├─ parameter_manager.cleanup(): Parameter state cleanup
       └─ Force garbage collection for large arrays

    DATA RESET SEQUENCE (NEW):
    1. reset_all_data(): Complete data reset while keeping parameters
       ├─ data_manager.clear_all_data(): Clear all generated data
       ├─ _reset_display_content(): Clear display content
       ├─ Generation state reset: Stop timers, reset flags
       └─ UI status reset: Ready state, clear error messages

    GENERATION WORKFLOW:
    --------------------

    STANDARD GENERATION FLOW:
    1. User Action/Auto-Trigger → generate()
    2. State validation and reset (prevents double-generation)
    3. Check for tab-specific generator method (generate_[tab_name]_system)
    4. Start performance timing
    5. Execute generation with error handling
    6. Update UI status and progress (via orchestrator signals)
    7. End timing and cleanup state

    AUTO-SIMULATION FLOW:
    1. Parameter change detected → auto_simulation_timer starts
    2. Timer expires → on_auto_simulation_triggered()
    3. For auto-generation tabs (geology+): Check dependencies
    4. If dependencies satisfied: Trigger generation
    5. Update dependent tabs via signals

    AUTO-TAB-GENERATION (NEW FEATURE):
    - Tabs: Geology, Settlement, Weather, Water, Biome
    - Behavior: Automatic generation on tab switch OR manual generate click
    - Dependency: Only executes if required dependencies are available
    - Integration: check_input_dependencies() → generate() → signal emission

    DISPLAY SYSTEM:
    ---------------

    VIEW SWITCHING ARCHITECTURE:
    • 2D/3D Toggle: QStackedWidget with DisplayWrapper abstraction
    • Fallback System: Graceful degradation when display classes unavailable
    • State Management: Active/inactive display coordination
    • Memory Management: Texture cleanup on view switches

    DISPLAY UPDATE OPTIMIZATION:
    • Change Detection: Hash-based comparison prevents unnecessary updates
    • Resource Tracking: Systematic cleanup of display resources
    • Memory Management: Garbage collection for large array updates
    • Error Resilience: Failed updates don't crash the UI

    RENDERING PIPELINE:
    1. update_display_mode() → _render_current_mode()
    2. Change detection via DisplayUpdateManager
    3. Cleanup old textures if update needed
    4. Apply new data to display
    5. Mark as updated in cache
    6. Force GC for large arrays (>50MB)

    EXTENSIBILITY FOR SUB-CLASSES:
    ------------------------------

    REQUIRED IMPLEMENTATIONS:
    • generate_[tab_name]_system(): Core generation logic
    • required_dependencies: List[str]: Input dependency specification
    • create_parameter_controls(): UI controls for generator parameters

    OPTIONAL OVERRIDES:
    • create_visualization_controls(): Custom display mode buttons
    • update_display_mode(): Custom display rendering logic
    • check_input_dependencies(): Custom dependency validation
    • on_external_data_updated(): Custom cross-tab data handling

    STANDARD INTEGRATION POINTS:
    • control_panel.layout(): Add custom parameter controls
    • setup_standard_orchestrator_handlers(generator_type): LOD integration
    • Signal connections: Connect to data_updated, parameter_changed etc.

    PARAMETER INTEGRATION:
    • parameter_manager: Handles slider persistence and change detection
    • Auto-simulation: Automatic triggering on parameter changes
    • Validation: Dependency checking and UI state updates

    ERROR HANDLING STRATEGY:
    ------------------------

    DEFENSIVE PROGRAMMING:
    • Every critical operation wrapped in try/except
    • Graceful degradation: Continue operation even if non-critical parts fail
    • Resource safety: Cleanup guaranteed even on exceptions
    • User feedback: Clear error messages in UI status displays

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

    Funktionsweise: Terrain-Editor mit vollständiger BaseMapTab-Integration und Core-Generator-Anbindung
    - Erbt alle BaseMapTab-Features: 70/30 Layout, 2D/3D Toggle, Manual-Generation, Resource-Management
    - Terrain-Generator Integration über BaseTerrainGenerator aus core/terrain_generator.py
    - Einheitliche Generation über BaseMapTab.generate() → calculate_heightmap(), calculate_shadows(), calculate_slopes()
    - StandardOrchestratorHandler über BaseMapTab.setup_standard_orchestrator_handlers("terrain")
    - Parameter-Integration über BaseMapTab's ParameterManager
    - Verwendet LOD-System aus DataLODManager:
        - LOD-Progression: Automatische Verdopplung 64→128→256→512→1024→2048 bis map_size erreicht
        - LOD-Level: 1,2,3,4,5,6,7+ (mindestens 7 für alle Sonnenstände und theoretisch weitere bis map_size)
    - Output: heightmap, slopemap, shadowmap für nachfolgende Generatoren
    - Map-Size-Synchronisation über DataLODManager.sync_map_size() für Tab-übergreifende Konsistenz

    UI-Layout (erweitert BaseMapTab):
    Control Panel Parameter-Sektion:
      - Terrain Parameters GroupBox mit Verwendung von value_default.py für "min", "max", "default", "step" (und "suffix"
      für Amplitude) und widgets.py für ParameterSlider und RandomSeedButton
        * Map Size: ParameterSlider
        * Height Amplitude: ParameterSlider
        * Detail Octaves: ParameterSlider
        * Base Frequency: ParameterSlider
        * Detail Persistence: ParameterSlider
        * Frequency Scaling: ParameterSlider
        * Height Redistribution: ParameterSlider
        * Map Seed: ParameterSlider + RandomSeedButton

      - Generation Control GroupBox:
        * "Berechnen"-Button: Manual Generation-Trigger
        * Generation Progress: ProgressBar mit LOD-Phase-Details
        * System Status: LOD-Level-Display, Generation-Status

      - Terrain Statistics GroupBox:
        * Height Range: Min/Max/Mean/StdDev
        * Slope Statistics: Max-Gradient, Mean-Slope in Degrees
        * Shadow Coverage: Min/Max/Mean Shadow-Values über alle Sonnenwinkel
        * Performance Metrics: Data-Size, Generation-Time

    Canvas Visualization Controls:
      - Terrain-spezifische Controls (nur 2D):
        * Height/Slope: Radio-Buttons (exklusiv, nur für Terrain-2D-Display)
     (erweitert BaseMapTab):
      - Standard BaseMapTab Controls (2D+3D):
        * Shadow: Checkbox (Overlay) + Shadow-Angle-Slider (0-6, entspricht 7 Sonnenstände)
        * Contour Lines: Checkbox (Overlay mit Heightmap)
        * Grid Overlay: Checkbox (für Measurement und Scale-Reference)

    calculate_heightmap():

    Input: Parameter-Set (size, amplitude, octaves, frequency, persistence, lacunarity, redistribute_power, map_seed) + current_lod_level
    Core-Call: BaseTerrainGenerator.calculate_heightmap(parameters, lod_level)
    Heightmap-Generation: Multi-Scale Simplex-Noise mit Octave-Layering und Progressive-Interpolation
    Ridge-Warping: Deformation mittels ridge warping für natürliche Landschaftsformen
    Height-Redistribution: Höhen-Redistribution für natürliche Höhenverteilung basierend auf redistribute_power
    Validation: Shape/Range/NaN-Checks für generated heightmap
    Output-Storage: DataLODManager.set_terrain_data_lod("heightmap", array, lod_level, parameters)
    Signal-Emission: data_updated("terrain", "heightmap")

    calculate_slopes():

    Input: heightmap + terrain-parameters für Gradient-Berechnung
    Core-Call: BaseTerrainGenerator.calculate_slopes(heightmap, parameters)
    Slope-Calculation: dz/dx und dz/dy Gradienten-Berechnung über heightmap
    Gradient-Magnitude: Berechnung der Slope-Intensity für Steigungsstatistiken
    Output-Format: 3D-Array (H,W,2) mit dz/dx und dz/dy Komponenten
    Validation: Gradient-Range-Checks und Consistency mit heightmap-Shape
    Output-Storage: DataLODManager.set_terrain_data_lod("slopemap", array, lod_level, parameters)
    Signal-Emission: data_updated("terrain", "slopemap")

    calculate_shadows():

    Input: heightmap + sun_angles (LOD-spezifische Sonnenwinkel-Anzahl) + shadow-parameters
    Core-Call: BaseTerrainGenerator.calculate_shadows(heightmap, sun_angles, parameters)
    Shadow-Calculation: Raycast-Shadow-Berechnung für multiple Sonnenwinkel mit GPU-Shader-Support
    LOD-Progressive-Shadows: 1,3,5,7 Sonnenwinkel je nach LOD-Level (Mittag→Vormittag/Nachmittag→Morgen/Abend→Dämmerung)
    Multi-Angle-Combination: Kombination aller Sonnenwinkel zu finaler Shadowmap
    Output-Format: 3D-Array (H,W,angles) mit Shadow-Values [0-1] für jeden Sonnenwinkel
    Validation: Shadow-Range-Checks [0-1] und Angle-Consistency
    Output-Storage: DataLODManager.set_terrain_data_lod("shadowmap", array, lod_level, parameters)
    Signal-Emission: data_updated("terrain", "shadowmap")

    Parameter-Spezifika:
    - Map Size: Power-of-2 Validation (64,128,256,512,1024,2048), Map-Size-Sync zu anderen Tabs
    - Seed-System: RandomSeedButton aus widgets.py generiert Seeds
    - Cross-Parameter-Constraints: Octaves vs Size Validation
    - Parameter-Widgets: Alle ParameterSlider und RandomSeedButton aus widgets.py

    Kommunikationskanäle:
    - Config: value_default.TERRAIN für Parameter-Ranges, Validation-Rules, Default-Values
    - Core: core/terrain_generator.py → BaseTerrainGenerator, SimplexNoiseGenerator, ShadowCalculator
    - Manager: GenerationOrchestrator.request_generation("terrain") mit OrchestratorRequestBuilder
    - Data: DataLODManager.set_terrain_data_complete() mit TerrainData für geology/settlement/weather
    - Signals: BaseMapTab-Signals + terrain-spezifische Validity-Updates
    - Display: BaseMapTab._render_current_mode() für Height/Slope/Shadow-Rendering
    - Widgets: widgets.py für ParameterSlider, RandomSeedButton, StatusIndicator

    Generation-Flow:
    1. "Berechnen"-Button → generate()
    1a. calculate_heightmap() → DataLODStorage → Signal-Emission
    1b. calculate_slopes() → DataLODStorage → Signal-Emission
    1c. calculate_shadows() → DataLODStorage → Signal-Emission
    2. LOD-Progression über GenerationOrchestrator mit incremental Results
    3. Display-Update über BaseMapTab._render_current_mode() nach jedem LOD
    4. Statistics-Update in Real-time, Map-Size-Sync zu dependent Tabs
    5. Signal-Emission für Cross-Tab-Dependencies (geology, weather, water, biome, settlement)

    Error-Handling:
    - Parameter-Validation: Range-Checks, Cross-Parameter-Constraints
    - Generation-Validation: TerrainData Shape/Range/NaN-Checks
    - LOD-Consistency: Validity-Checks zwischen LOD-Levels
    - GPU-Fallback: ShaderManager-Integration mit CPU-Fallback-Notification
    - Memory-Management: Large-Array-Detection, Force-GC für >50MB Arrays

    Output-Datenstrukturen:
    - heightmap: 2D numpy.float32 array, Elevation in Metern
    - slopemap: 3D numpy.float32 array (H,W,2), dz/dx und dz/dy Gradienten
    - shadowmap: 3D numpy.float32 array (H,W,7), Shadow-Values [0-1] für 7 Sonnenwinkel
    - TerrainData: Container mit allen Arrays + LOD-Metadata + Validity-State + Parameter-Hash
    """

def geology_tab():
    """
    Path: gui/tabs/geology_tab.py

    Funktionsweise: Geology-Editor mit BaseMapTab-Integration und geologischer Simulation
    - Erbt alle BaseMapTab-Features: 70/30 Layout, 2D/3D Toggle, Manual-Generation, Resource-Management
    - Geology-Generator Integration über GeologyGenerator aus core/geology_generator.py
    - Input-Dependencies: heightmap, slopemap von terrain_tab über DataLODManager
    - Einheitliche Generation über BaseMapTab.generate() → calculate_rockmap(), calculate_massconservation(),
    calculate_hardnessmap()
    - Rock-Type-Classification: Sedimentary/Igneous/Metamorphic mit Mass-Conservation (Red (Sedimentary)+Green (Igneous)
     + Blue(Metamorphic) = 255)
    - LOD-Progression: Automatische Verdopplung bis map_size erreicht (entsprechend Terrain)
    - Output: rock_map (RGB), hardness_map für water_generator und nachfolgende Systeme

    UI-Layout (erweitert BaseMapTab):
    Control Panel Parameter-Sektion mit "min", "max", "default", "step" aus value_default.py
    und ParameterSlider aus widgets.py:
      - Rock Hardness Parameters GroupBox:
        * Sedimentary Hardness: ParameterSlider
        * Igneous Hardness: ParameterSlider
        * Metamorphic Hardness: ParameterSlider

      - Tectonic Deformation Parameters GroupBox:
        * Ridge Warping: ParameterSlider
        * Bevel Warping: ParameterSlider
        * Metamorphic Foliation: ParameterSlider
        * Metamorphic Folding: ParameterSlider
        * Igneous Flowing: ParameterSlider

      - Generation Control GroupBox:
        * "Berechnen"-Button: Manual Generation-Trigger  (erweitert BaseTab)
        * Generation Progress: ProgressBar mit Geology-Phase-Details  (erweitert BaseTab)
        * Input Dependencies: StatusIndicator für heightmap/slopemap-Verfügbarkeit

      - Rock Distribution Widget:
        * Hardness Preview: Progress-Bars für Sedimentary/Igneous/Metamorphic
        * Distribution Statistics: Prozentuale Verteilung nach Generation
        * Mass Conservation Status: StatusIndicator für R+G+B=255 Validation

    Canvas Visualization Controls (erweitert BaseMapTab):
      - Geology-spezifische Controls:
        * Rock Types/Hardness: Radio-Buttons (2D+3D)
      - Standard BaseMapTab Controls (2D+3D):
        * Shadow: Checkbox (Overlay über alle Modi, nutzt Terrain-Shadowmap)
        * Contour Lines: Checkbox (Overlay, nutzt Terrain-Heightmap) und erzeugt einen Sonnenwinkel-Slider (1-7)
        * Grid Overlay: Checkbox (für alle Modi)

    Core-Funktionalität:
    calculate_rockmap():

    - Input-Validation: heightmap/slopemap Shape/Range/Consistency-Checks
    - Input: Parameter-Set + heightmap + slopemap von terrain_tab
    - Core-Call: GeologyGenerator.calculate_rock_distribution(heightmap, slopemap, parameters)
    - Rock-Classification: Höhen-basierte + Steigungs-basierte + Noise-basierte Verteilung
    - Geological-Zones: Sedimentary/Igneous/Metamorphic-Zone-Calculation mit Simplex-Noise
    - Ridge-Warping: Tektonische Verformung für härtere Gesteine in steilen Hängen
    - Deformation-Effects: Bevel-Warping, Metamorphic-Foliation/Folding, Igneous-Flowing
    - Output-Storage: DataLODManager.set_geology_data_lod("rock_map", array, lod_level, parameters)
    - Signal-Emission: data_updated("geology", "rock_map")

    calculate_hardnessmap():

    - Input: rock_map + hardness-Parameters (sedimentary/igneous/metamorphic_hardness)
    - Core-Call: GeologyGenerator.calculate_hardness_map(rock_map, hardness_parameters)
    - Hardness-Calculation: Gewichtete Hardness-Map aus RGB-rock_map und hardness-Parametern
    - Formula: hardness_map(x,y) = (R*sed_hardness + G*ign_hardness + B*met_hardness) / 255
    - Output-Storage: DataLODManager.set_geology_data_lod("hardness_map", array, lod_level, parameters)
    - Signal-Emission: data_updated("geology", "hardness_map")

    calculate_massconservation():

    - Input: rock_map mit potentiell inkonsistenten RGB-Werten
    - Core-Call: MassConservationManager.normalize_rock_masses(rock_map)
    - Mass-Conservation: R+G+B=255 Enforcement für alle Pixel
    - Normalization: Proportionale Skalierung wenn R+G+B != 255
    - Fallback: Gleichverteilung (85,85,85) bei R+G+B=0 Pixeln
    - Output-Update: Überschreibt rock_map in DataLODManager mit korrigierten Werten
    - Statistics-Update: Rock-Distribution-Widget mit korrigierter Verteilung

    Dependency-Management:
    - Required Dependencies: ["heightmap", "slopemap"] von terrain_tab
    - check_input_dependencies(): Prüft Terrain-Data-Verfügbarkeit, Shape-Consistency, Data-Quality
    - Dependency-Status-Display: StatusIndicator mit missing/invalid Inputs und Recovery-Suggestions

    Kommunikationskanäle:
    - Config: value_default.GEOLOGY für Parameter-Ranges, Hardness-Defaults, Validation-Rules
    - Core: core/geology_generator.py → GeologyGenerator, RockTypeClassifier, MassConservationManager
    - Input: DataLODManager.get_terrain_data("heightmap/slopemap") für Basis-Terrain-Daten
    - Output: DataLODManager.set_geology_data_lod() für rock_map/hardness_map zu water_generator
    - Manager: GenerationOrchestrator.request_generation("geology") nach Dependency-Erfüllung
    - Signals: BaseMapTab-Signals + geology-spezifische rock_map/hardness_map Updates
    - Widgets: widgets.py für ParameterSlider, StatusIndicator, ProgressBar

    Display-Modi (erweitert BaseMapTab):
    - Height Mode: Terrain-Heightmap als Basis-Layer
    - Rock Types Mode: RGB rock_map mit Color-Coding (Rot=Sedimentary, Grün=Igneous, Blau=Metamorphic)
    - Hardness Mode: Hardness-Map mit Grayscale/Color-Coding (Blau=weich, Rot=hart)
    - 3D Terrain Overlay: Kombiniert Rock-Classification mit 3D-Terrain-Rendering
    - Shadow/Contour/Grid Overlays: Nutzen Terrain-Data über alle Geology-Modi

    LOD-System Integration:
    - Progressive Generation: Automatische LOD-Progression im Hintergrund entsprechend Terrain
    - LOD-Size-Verdopplung: 64→128→256→512→1024→2048 bis map_size erreicht
    - LOD-Level-Nummerierung: 1,2,3,4,5,6+ entsprechend verfügbaren Terrain-LODs
    - Incremental Updates: UI-Update nach jedem LOD-Level
    - Display-Optimization: Immer bestes verfügbares LOD für Display

    Generation-Flow:
    1. Dependency-Check → heightmap/slopemap-Validation → "Berechnen"-Button
    2. generate() → calculate_geology() → GeologyGenerator.generate()
    3. Rock-Classification → Mass-Conservation → Hardness-Calculation
    4. Validation → Repair → DataLODManager-Storage → Signal-Emission
    5. Display-Update → Statistics-Update → Cross-Tab-Notification für Dependencies

    Output-Datenstrukturen:
    - rock_map: 3D numpy.uint8 array (H,W,3), RGB-Kanäle für Sedimentary/Igneous/Metamorphic mit R+G+B=255
    - hardness_map: 2D numpy.float32 array, Gesteinshärte-Werte [0-100] für Erosions-Simulation
    - GeologyData: Container mit rock_map/hardness_map + Validation-State + Mass-Conservation-Info
    """

def weather_tab():
    """
    Path: gui/tabs/weather_tab.py

    Funktionsweise: Wetter-System mit 3D Wind-Visualization
    - Climate-Modeling basierend auf Terrain (Orographic Effects)
    - 3D Wind-Vector Display über Heightmap
    - Live Climate-Classification Display
    - Temperature/Precipitation Field Generation

    Kommunikationskanäle:
    - Input: heightmap für orographic effects
    - Core: weather_generator für Climate-Simulation
    - Output: temperature_field, precipitation_field, wind_vectors → data_manager
    """

def water_tab():
    """
    Path: gui/tabs/water_tab.py

    Funktionsweise: Wassersystem mit River-Networks und Erosion
    - Input: Heightmap, Precipitation, Rock-Hardness
    - River-Generation durch Flow-Accumulation
    - Lake-Placement und Water-Table Simulation
    - Erosion-Simulation modifiziert Heightmap

    Kommunikationskanäle:
    - Input: heightmap, precipitation_field, rock_hardness von data_manager
    - Core: water_generator für Hydrologie
    - Output: river_network, lakes, modified_heightmap → data_manager
    """

def biome_tab():
    """
    Path: gui/tabs/biome_tab.py

    Funktionsweise: Finale Biome-Klassifizierung und Welt-Übersicht
    - Input: Alle vorherigen Generator-Outputs
    - Whittaker-Biome Classification (Temperature vs Precipitation)
    - Integration aller Systeme in finale Welt-Darstellung
    - Export-Funktionalität für komplette Welt

    Kommunikationskanäle:
    - Input: heightmap, temperature_field, precipitation_field, settlements, water
    - Core: biome_generator für Ecosystem-Classification
    - Output: final_world_map mit allen Layern integriert
    """

def settlement_tab():
    """
    Path: gui/tabs/settlement_tab.py

    Funktionsweise: Settlement-Platzierung mit Terrain-Integration
    - Input: Heightmap, optional Rock-Map für Suitability
    - Intelligent Settlement-Placement (Steigung, Wasser-Nähe)
    - 3D-Markers für Villages/Landmarks auf Terrain
    - Road-Network Visualization

    Kommunikationskanäle:
    - Input: heightmap, rock_map von data_manager
    - Core: settlement_generator für Placement-Algorithm
    - Output: settlement_positions, road_network → data_manager
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