"""
    Descriptor file:
    Descriptions for all scripts as methods (to find them faster)
"""


def main():
    """
    Path: main.py
    date_changed: 25.08.2025

    Funktionsweise: Programm-Einstiegspunkt für QApplication-Setup
    - Initialisiert QApplication mit Standard-Konfiguration
    - Startet MainMenuWindow als zentraler Entry-Point
    - Basic Error-Handling für kritische Startup-Fehler
    - Clean Application-Exit nach Benutzer-Beendigung

    Kommunikationskanäle:
    - Config: gui_default.py für Application-Settings
    - UI: Direkte Instanziierung von MainMenuWindow
    - Navigation: MainMenuWindow → NavigationManager für Tab-Navigation

    Architektur-Prinzipien:
    - Dependency-Checks erfolgen in den jeweiligen Core-Generatoren
    - Manager werden lazy initialisiert bei erster Verwendung
    - Fehlerbehandlung delegiert an error_handler.py
    - Minimale Startup-Komplexität für schnellen Application-Start
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
        Attribute: heightmap, slopemap, shadowmap, lod_level, actual_size, validity_state, parameter_hash,
        calculated_sun_angles, parameters
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
    date_changed: 18.08.2025

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
        External-Interface: calculate_geology(heightmap, slopemap, parameters, lod_level) - wird von
        GenerationOrchestrator aufgerufen
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
        Settlements: Städte oder Dörfer die an bestimmten Orten vorkommen können (Täler, flache Hügel). Settlements
        verringern die Terrainverformung in der Nähe etwas. Je nach Radius (Siedlungsgröße) ist der Einfluss auf die
        Umgebung größer/kleiner. Die Form der Stadt soll zB Linsenförmig sein und über die Slopemap erzeugt werden.
        Zwischen Settlements gibt es einen Minimalabstand je nach map_size und Anzahl von Settlements. Innerhalb der
        Stadtgrenzen ist civ_map = 1, außerhalb nimmt der Einfluss ab.
        Roads: Nachdem Settlements entstanden sind werden die ersten Wege zwischen den Ortschaften geplottet. Dazu soll
        der Weg des geringsten Widerstands gefunden werden (Pathfinding via slopemap-cost). Danach werden die Straßen
        etwas gebogen über sanfte Splineinterpolation zwischen zB jedem 3.Waypoint. Erzeugen sehr geringen Einfluss
        entlang der Wege (z.B. 0.3).
        Roadsites: z.B. Taverne, Handelsposten, Wegschrein, Zollhaus, Galgenplatz, Markt, besondere Industrie.
        Entstehen in einem Bereich von 30%-70% Weglänge zwischen Settlements entlang von Roads. Der civ_map-Einfluss
        ist wesentlich geringer als der von Städten.
        Landmarks: z.B. Burgen, Kloster, mystische Stätte etc. entstehen in Regionen mit einem
        civ_map value < thresholds (landmark_wilderness). Erzeugen einen ähnlich geringen Einfluss wie Roadsites.
        Außerdem werden beide nur in niedrigeren Höhen und Slopes generiert.
        Wilderness: Bereiche unterhalb eines civ_map-Werts unterhalb von 0.2 werden genullt und als Wilderness
        deklariert. Hier spawnen keine Plotnodes. Hier sollen in der  späteren Spielentwicklung Questevents stattfinden.
        civ_map-Logik: civ_map wird mit 0.0 initialisiert. Jeder Quellpunkt trägt akkumulativ zum civ-Wert bei.
        Einflussverteilung um Quellpunkt über radialen Decay-Kernel (z.B. Gauß, linear fallend oder benutzerdefinierte
        Kurve). Decay ist stärker an Hanglagen, so dass Zivilisation nicht auf Berge reicht. Decayradius und Initialwert
        abhängig von Location-Typ: Stadt-Grenzpunkte starten bei 0.8 (innerhalb der Stadt ist 1.0), Roadwaypoints
        addieren 0.2 bis max. 0.5, Roadsite/Landmarks 0.4. Optional bei sehr hohen Berechnungzeiten kann die
        Einflussverteilung mit GPU-Shadermasken erfolgen.
        Plotnodes: Es wird eine feste Anzahl an Plotnodes generiert (plotnodes-parameter). Gleichmäßige Verteilung auf
        alle Bereiche außerhalb von Städten und Wilderness. Die Plotnodes verbinden sich mit mit Nachbarnodes über
        Delaunay-Triangulation. Dann verbinden sich die Delaunay-Dreiecke mit benachbarten Dreiecken zu Grundstücken.
        Die Plotnode-Civwerte werden zusammengerechnet und wenn sie einen Wert (plot-size-parameter) überschreiten ist
        die Größe erreicht. So werden Grundstücke in Region mit hohem Civ-wert kleiner. Über Abstoßungslogik können die
        Nodes "physisch" umarrangiert werden. Kanten mit geringem Winkel sollen sich glätten und die Zwischenpunkte
        können verschwinden. Sehr spitze Winkel lockern sich ebenso.
        Plotnode-Eigenschaften:
            node_id, node_location, connector_id (list of nodes), connector_distance (x,y entfernung),
            connector_elevation (akkumulierter höhenunterschied zu connector), connector_movecost (movecost
            abhängig von biomes)
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
    date_changed: 25.08.2025

    Funktionsweise: Dynamisches Wetter- und Feuchtigkeitssystem
    - Terrain-basierte Temperaturberechnung (Altitude, Sonneneinstrahlung, Breitengrad (Input: heightmap, shade_map) der Luft.
    - Windfeld-Simulation mit Ablenkung um Berge und Abbremsen durch Geländegradienten-Analyse (slopemap).
        - Konvektion: Wärmetransport durch Windströme
        - Orographischer Regen: Feuchtigkeitstransport
        - Precipation durch Abkühlen/Aufsteigen der Luft
        - Evaporation durch Bodenfeuchtigkeit (Input: soil_moist_map)
        - OpenSimplex-Noise für  Variation in Temperatur-, Feuchte- und Luftdruck an Kartengrenze
        - Regen wird ausgelöst durch Erreichen eines Feuchtigkeitsgrenzwertes von über 1.0. Dieser berechnet sich aus
        rho_max = 5*exp(0.06*T) mit T aus temp_map und der Luftfeuchtigkeit in der jeweiligen Zelle.

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
    Zunächst wird die Lufttemperatur für jede Zelle der Simulation basierend auf drei Hauptfaktoren berechnet.
    Die Altitude (z) führt zu einer starken Temperaturabnahme von 60°C pro Kilometer Höhe - das ist zehnmal stärker
    als in der Realität, um dramatischere Effekte zu erzielen. Die Sonneneinstrahlung I(x,y) wird aus einer speziell
    vorbereiteten Shademap importiert, die sechs verschiedene Sonnenwinkel über den Tag gewichtet kombiniert. Hohe
    Sonnenstände erhalten dabei mehr Gewichtung. Bei maximaler Sonneneinstrahlung (Wert 1) steigt die Temperatur um
    10°C, bei minimaler Einstrahlung (Wert 0) fällt sie um 10°C. Der Breitengrad (y-Position) simuliert den
    Äquator-zu-Pol-Gradienten: An der Südseite der Karte (y=0) bleibt die Basistemperatur unverändert, während sie
    zur Nordseite hin (y=map_size) um 5°C ansteigt.

    Schritt 2: Windfeld-Grundsimulation
    Die eigentliche Windsimulation wird durch einen konstanten Druckgradienten von West nach Ost initialisiert. An
    der Westseite der Karte wird ein erhöhter Luftdruck angelegt, der nach Osten hin kontinuierlich abnimmt. Um
    natürlichere Strömungsmuster zu erzeugen, wird dieser Grunddruckgradient mit Simplex-Noise moduliert, wodurch
    turbulente Variationen entstehen. Die gesamte Welt wird in ein regelmäßiges Gitter von Simulationszellen unterteilt,
    wobei jede Zelle eine Luftsäule vom Gelände bis zur maximalen Simulationshöhe repräsentiert. Ein spezieller
    GPU-Shader berechnet dann für jede Zelle die Druckverteilung und die daraus resultierenden Windgeschwindigkeiten
    in alle drei Raumrichtungen.

    Schritt 3: Geländeinteraktion
    Das Windfeld interagiert dynamisch mit der Geländetopographie. Düseneffekte entstehen automatisch in engen Tälern,
    wo die Windgeschwindigkeit aufgrund der Kontinuitätsgleichung zunimmt. Blockierungseffekte treten auf, wenn
    Luftmassen auf Berghänge treffen - der Wind wird dann entweder nach oben abgelenkt oder um die Hindernisse
    herumgeleitet. Orographische Hebung an Luvhängen führt zu Aufwinden, während Leewirbel auf der windabgewandten
    Seite von Bergen entstehen. Die Temperaturunterschiede zwischen sonnenexponierten und schattigen Hängen verstärken
    diese Effekte zusätzlich durch thermisch induzierte Hangwinde.

    Schritt 4: Numerische Integration
    Die zeitliche Entwicklung des Windfeldes wird durch die Navier-Stokes-Gleichungen für inkompressible Strömungen
    gesteuert. Dabei werden Advektionsterme (Wind transportiert sich selbst), Druckgradientenkräfte und
    Viskositätseffekte berücksichtigt. Die Kontinuitätsgleichung sorgt dafür, dass die Massenerhaltung eingehalten wird,
    was besonders wichtig ist, um realistische Strömungsmuster um Geländehindernisse zu erzeugen. Jeder
    Simulationsschritt aktualisiert sowohl die Windgeschwindigkeiten als auch die Druckverteilung konsistent.

    Benötigte Shader für die Berg-Wind-Simulation
    Shader 1: Temperatur-Berechnung (temperatureCalculation.frag)
    Dieser Shader berechnet die Lufttemperatur für jede Zelle basierend auf den Eingabeparametern. Er liest die
    Heightmap aus, um die Höhenabhängige Abkühlung (altitude_cooling) anzuwenden, sampelt die Shade-Map für die solare
    Erwärmung (solar_power) und berücksichtigt den Breitengrad-Gradienten über die Y-Position. OpenSimplex-Noise wird
    verwendet, um natürliche Temperaturschwankungen an den Kartengrenzen zu erzeugen. Der Shader gibt ein
    2D-Temperaturfeld aus, das als Grundlage für alle thermischen Berechnungen dient.

    Shader 2: Windfeld-Basis (windFieldGeneration.frag)
    Dieser Shader erzeugt das grundlegende Windfeld durch Druckgradienten von West nach Ost. Er berechnet
    Druckdifferenzen und konvertiert diese über den wind_speed_factor in Windgeschwindigkeiten. OpenSimplex-Noise
    moduliert die Druckverteilung für natürliche Turbulenz. Der Shader berücksichtigt Geländeablenkung durch die
    Slope-Map und den terrain_factor, wodurch Wind um Berge herumgeleitet und in Tälern beschleunigt wird. Die
    Ausgabe ist ein 2D-Vektorfeld mit horizontalen Windkomponenten.

    Shader 3: Thermische Konvektion (thermalConvection.frag)
    Dieser Shader berechnet thermisch induzierte Windkomponenten basierend auf Temperaturdifferenzen. Er liest das
    Temperaturfeld aus dem vorherigen Shader und berechnet lokale Temperaturgradienten. Aufwinde entstehen über warmen
    Bereichen (hohe Shade-Map-Werte), Abwinde über kalten Bereichen. Der thermic_effect-Parameter steuert die Stärke
    dieser thermischen Verformung. Der Shader modifiziert das bestehende Windfeld durch Überlagerung der konvektiven
    Komponenten und erzeugt realistische Hangwind-Systeme.

    Shader 4: Feuchtigkeits-Transport (moistureTransport.frag)
    Dieser Shader simuliert den Transport von Wasserdampf durch das Windfeld. Er liest die Soil-Moisture-Map für die
    Evaporation, das aktuelle Feuchtigkeitsfeld und das Windfeld für den Advektionstransport. Evaporation wird basierend
    auf Bodenfeuchte und lokaler Temperatur berechnet. Der Shader implementiert eine Advektionsgleichung, die
    Wasserdampf entsprechend der Windrichtung und -geschwindigkeit transportiert. Diffusionseffekte glätten extreme
    Feuchtigkeitsgradienten für realistische Verteilungen.

    Shader 5: Niederschlags-Berechnung (precipitationCalculation.frag)
    Dieser Shader bestimmt, wo und wie viel Niederschlag fällt. Er berechnet die maximale Wasserdampfdichte
    rho_max = 5*exp(0.06*T) für jede Zelle basierend auf der lokalen Temperatur. Wenn die relative Luftfeuchtigkeit
    den Wert 1.0 überschreitet, wird Niederschlag ausgelöst. Der Shader berücksichtigt orographische Hebung durch
    Windgeschwindigkeit und Geländesteigung. Latente Wärmefreisetzung bei der Kondensation wird zurück an den
    Temperatur-Shader gegeben. Die Ausgabe ist ein 2D-Niederschlagsfeld.

    Shader 6: Orographische Effekte (orographicEffects.frag)
    Dieser spezialisierte Shader berechnet geländeinduzierte Wetterphänomene. Er analysiert Windrichtung relativ zur
    Geländeorientierung, um Luv- und Lee-Bereiche zu identifizieren. Staueffekte werden an Luvhängen durch verstärkte
    Aufwinde simuliert. Föhneffekte entstehen durch trockenadiabatische Erwärmung auf der Leeseite. Der Shader
    modifiziert sowohl Temperatur als auch Feuchtigkeit basierend auf der Geländeinteraktion und erzeugt
    charakteristische Regenschatten-Muster.

    Shader 7: System-Integration (weatherIntegration.frag)
    Dieser zentrale Shader führt alle Komponenten zusammen und berechnet die zeitliche Entwicklung des Wettersystems.
    Er implementiert Rückkopplungsschleifen zwischen Temperatur, Wind und Feuchtigkeit. Konvergenz- und Divergenzzonen
    werden identifiziert und verstärken lokale Wetterphänomene. Der Shader aktualisiert alle Felder konsistent und
    sorgt für Massenerhaltung bei Feuchtigkeit und Energieerhaltung bei thermischen Prozessen. Er koordiniert die
    Ausgabe der finalen wind_map, temp_map, precip_map und humid_map.

    Shader 8: Boundary-Conditions (boundaryConditions.frag)
    Dieser Shader verwaltet die Randbedingungen an den Kartengrenzen. Er implementiert kontinuierliche
    Wetterfront-Einträge mit den konfigurierten Parametern (air_temp_entry, etc.). OpenSimplex-Noise erzeugt
    realistische Wetterfront-Variationen. Der Shader sorgt für konsistente Übergänge zwischen den Kartenrändern und
    dem Innenbereich und verhindert künstliche Artefakte an den Grenzen. Periodische Randbedingungen können für
    endlose Karten implementiert werden.

    Zusätzliche Utility-Shader:
    Gradient-Berechnung (gradientCalculation.frag): Berechnet Höhen- und Temperaturgradienten für die anderen Shader
    Noise-Generation (noiseGeneration.frag): Erzeugt OpenSimplex-Noise-Felder für natürliche Variationen
    Debug-Visualisierung (debugVisualization.frag): Stellt verschiedene Datenfelder für die Entwicklung visuell dar

    Alle Shader arbeiten im Ping-Pong-Verfahren zwischen mehreren Framebuffern, um zeitliche Entwicklung zu simulieren
    und Rückkopplungseffekte zu ermöglichen.

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
    date_changed: 25.08.2025

    Funktionsweise: Dynamisches Hydrologiesystem mit Erosion, Sedimentation und bidirektionaler Terrain-Modifikation
    - Lake-Detection durch Jump Flooding Algorithm für parallele Senken-Identifikation
    - Flussnetzwerk-Aufbau durch Steepest Descent mit Upstream-Akkumulation
    - Strömungsberechnung nach Manning-Gleichung mit adaptiven Querschnitten
    - Bodenfeuchtigkeit durch Gaussian-Diffusion von Gewässern
    - Stream Power Erosion mit iterativer Terrain-Modifikation über LOD-Levels
    - Realistische Sedimentation mit kumulative Akkumulation
    - Evaporation basierend auf statischen Weather-Daten (temp_map, wind_map, humid_map)

    Parameter Input:
    - lake_volume_threshold (Mindestvolumen für Seebildung, default 0.1m)
    - rain_threshold (Niederschlagsschwelle für Quellbildung, default 5.0 gH2O/m²)
    - manning_coefficient (Rauheitskoeffizient für Fließgeschwindigkeit, default 0.03)
    - erosion_strength (Erosionsintensitäts-Multiplikator, default 1.0)
    - sediment_capacity_factor (Transportkapazitäts-Faktor, default 0.1)
    - evaporation_base_rate (Basis-Verdunstungsrate, default 0.002 m/Tag)
    - diffusion_radius (Bodenfeuchtigkeit-Ausbreitungsradius, default 5.0 Pixel)
    - settling_velocity (Sediment-Sinkgeschwindigkeit, default 0.01 m/s)
    - erosion_iterations_per_lod (Anzahl Erosions-Zyklen pro LOD-Level, default 10)
    - water_seed (Reproduzierbare Zufallsvariation)

    Dependencies (über DataLODManager):
    - heightmap (von terrain_generator für Orographic-Effects und Flow-Pathfinding)
    - hardness_map (von geology_generator für Erosions-Resistance)
    - precip_map (von weather_generator für Precipitation-driven Water-Sources)
    - temp_map (von weather_generator für Temperature-based Evaporation)
    - wind_map (von weather_generator für Wind-enhanced Evaporation)

    Output:
    - WaterData-Objekt mit water_map, flow_map, flow_speed, soil_moist_map, water_biomes_map, erosion_map, sedimentation_map, validity_state und LOD-Metadaten
    - Bidirektionale Terrain-Integration: erosion_map und sedimentation_map für DataLODManager.composite_heightmap
    - DataLODManager-Storage für nachfolgende Generatoren (biome, settlement)

    LOD-System (Numerisch mit iterativer Erosion):
    - lod_level 1: map 32x32 → Erosion 10 Iterations für große Erosions-Effekte
    - lod_level 2: map 64x64 → Erosion 10 Iterations mit Upsampling vorheriger Erosion-Daten
    - lod_level 3: map 128x128 → Erosion 10 Iterations mit kumulative Erosions-Akkumulation
    - lod_level 4: map 256x256 → Erosion 10 Iterations mit verfeinerte Flow-Pathfinding
    - lod_level 5: map 512x512 → Erosion 10 Iterations mit detaillierte Sediment-Transport
    - lod_level 6: map 1024x1024 → Erosion 10 Iterations mit hochauflösende Hydrologie
    - lod_level 7: map 2048x2048 → Erosion 10 Iterations mit finale Erosions-Details

    Fallback-System (3-stufig):
    - GPU-Shader (Optimal): Jump-Flooding, Flow-Network und Erosion-Simulation mit Compute-Shadern
    - CPU-Fallback (Gut): Optimierte NumPy-Implementierung mit Multiprocessing für parallele Operations
    - Simple-Fallback (Minimal): Vereinfachte Hydrologische Simulation ohne komplexe CFD-Berechnungen

    Graceful-Degradation-Strategy:
    1. GPU-Memory-Exhaustion: Automatic CPU-Fallback für große Arrays und komplexe Berechnungen
    2. Multi-Dependency-Input-Problems: Fallback zu vereinfachten Parametern bei fehlenden Weather/Geology-Daten
    3. Erosion-Numerical-Instability: Simplified-Erosion-Model ohne komplexe Sediment-Transport
    4. Flow-Pathfinding-Convergence-Failure: Static-Flow-Paths ohne dynamische Re-Pathfinding
    5. Critical-Memory-Failures: Minimal-Water-System (statische Seen ohne Flow-Simulation)

    Error-Recovery-Mechanisms:
    - Multi-Input-Validation: Konsistenz-Checks zwischen heightmap, hardness_map und Weather-Daten
    - Erosion-Stability-Monitoring: Überwachung auf numerical instabilities mit Auto-Correction
    - Flow-Network-Topology-Validation: Repair korrupter Flow-Paths mit alternative Pathfinding
    - Mass-Conservation-Enforcement: Erhaltung der Wasser-Masse bei allen Berechnungen
    - LOD-Upsampling-Validation: Konsistenz-Checks bei Erosion-Daten-Interpolation zwischen LOD-Levels

    Performance-Characteristics:
    - GPU-Accelerated: 15-40x speedup für Jump-Flooding und parallele Erosion-Simulation
    - CPU-Optimized: Vectorized NumPy-Operations für Flow-Networks, optimierte SciPy für Erosion-Math
    - Memory-Efficient: Progressive Erosion-Akkumulation, LOD-based Upsampling mit bicubic Interpolation
    - Cache-Friendly: Composite-Heightmap-Caching im DataLODManager, Parameter-Hash-basierte Invalidation

    Hydrologische Simulation-Pipeline:

    Schritt 1: Multi-Dependency Input-Integration
    Der Generator sammelt alle erforderlichen Input-Daten von verschiedenen Generatoren über DataLODManager.
    Heightmap bildet die topographische Basis, hardness_map von Geology bestimmt Erosions-Resistance,
    precip_map von Weather definiert Niederschlags-Quellen, temp_map/wind_map beeinflussen Evaporation.
    Input-Validation prüft Konsistenz zwischen allen Datenquellen und LOD-Level-Kompatibilität.

    Schritt 2: Lake-Detection durch Jump Flooding Algorithm
    Jump Flooding Algorithm identifiziert alle lokalen Minima parallel in logarithmischer Zeit O(log n).
    Initialisierung markiert jedes lokale Minimum als Lake-Seed mit lake_volume_threshold-Validation.
    Propagation-Phase: Seeds propagieren exponentiell abnehmende Sprungdistanzen (512→256→128→...→1).
    Flood-Fill erfolgt nur in Bereiche mit positiver Höhendifferenz zum Seed-Punkt.
    See-Klassifikation basiert auf akkumuliertem Volumen und Verbindung zu Kartenrändern.

    Schritt 3: Flow-Network durch Steepest Descent
    Alle Niederschlagsquellen (precip > rain_threshold) werden mit Zielen verknüpft über steepste Gradienten.
    Steepest Descent folgt dem steilsten Gradienten bergab bis zum nächsten lokalen Minimum oder Kartenrand.
    Upstream-Akkumulation sammelt Wassermengen iterativ von allen upstream-Zellen.
    Flow-Paths sind vollständig parallelisierbar da jede Zelle unabhängig ihre Richtung berechnet.
    Ocean-Outflow akkumuliert Wasser das Kartenränder verlässt für Biome-Generator-Integration.

    Schritt 4: Manning-Strömungsberechnung mit adaptiven Querschnitten
    Fließgeschwindigkeit nach Manning-Gleichung: v = (1/n) * R^(2/3) * S^(1/2).
    Hydraulischer Radius R und Sohlneigung S aus lokaler Topographie und Flow-Accumulation.
    Optimaler Querschnitt durch iterative Lösung der Kontinuitätsgleichung Q = A * v.
    Gelände-Form-Analyse: enge Täler → tiefe schmale Flüsse, weite Ebenen → breite flache Gewässer.
    Tal-Breite-Detection durch lokale Höhenprofile in alle Richtungen mit Gelände-Anstiegs-Suche.

    Schritt 5: Iterative Erosion-Sedimentation über LOD-Levels
    Pro LOD-Level: 10 Erosions-Iterationen für kumulative Landschafts-Modifikation.
    Stream Power Erosion: E = K * (τ - τc) wobei τ = ρ * g * h * S (Scherspannung).
    Kritische Scherspannung τc basiert auf hardness_map von Geology-Generator.
    Sediment-Transport entlang Flow-Paths mit distanz-abhängiger Transport-Effizienz.
    Sedimentation bei Transport-Kapazitäts-Überschreitung oder Geschwindigkeits-Reduktion.
    Kumulative Akkumulation: erosion_map und sedimentation_map sammeln alle Änderungen über Iterationen.

    Schritt 6: LOD-Progression mit Erosion-Upsampling
    Nach LOD-Completion: Erosion/Sedimentation-Daten werden für nächstes LOD upgesampled.
    Bicubic-Interpolation für glatte Erosions-Pattern-Erhaltung bei höherer Auflösung.
    Keine Mass-Conservation-Skalierung erforderlich da Erosion Höhen-Änderungen in Metern repräsentiert.
    Kombinierte Flow-Pathfinding: neue Slopemap basiert auf composite_heightmap (base + erosion + sedimentation).
    Progressive Erosions-Enhancement: höhere LODs bauen auf bereits erodierte Landschaft auf.

    Schritt 7: Bodenfeuchtigkeit durch Gaussian-Diffusion
    Gaussian-Filter verschiedener Größen simulieren Ausbreitungs-Mechanismen um Gewässer.
    Enger Filter (2-3 Pixel): kapillare Ausbreitung für direkte Ufer-Feuchtigkeit.
    Weiter Filter (5-10 Pixel): Grundwasser-Effekte für regionale Feuchtigkeit.
    Maximum aus beiden Filtern plus direkte Wasser-Präsenz für finale soil_moist_map.
    Exponentieller Feuchtigkeits-Abfall mit Distanz zu Gewässern für realistische Verteilung.

    Schritt 8: Water-Biomes-Klassifikation
    Einfache regelbasierte Klassifikation: 0=kein Wasser, 1=Creek, 2=River, 3=Grand River, 4=Lake.
    Klassifikation basiert auf water_depth und flow_rate: Seen sind statisch (flow_rate < 0.1).
    Creek: flow_rate < 1.0, River: flow_rate 1.0-5.0, Grand River: flow_rate > 5.0.
    Integration mit Lake-Detection-Results für konsistente Gewässer-Typisierung.

    Schritt 9: Evaporation basierend auf statischen Weather-Daten
    Evaporation verwendet statische Snapshots von temp_map, wind_map, humid_map.
    Temperatur-abhängige maximale Wasserdampf-Kapazität nach Magnus-Formel.
    Wind-verstärkte Evaporation durch verbesserten Dampf-Transport von Oberflächen.
    Atmosphärische Sättigungs-Limits: feuchte Luft kann weniger zusätzlichen Wasserdampf aufnehmen.
    Verfügbare Wasser-Oberfläche aus water_map bestimmt tatsächliche Evaporation-Rate.

    Schritt 10: Bidirektionale Terrain-Integration
    erosion_map und sedimentation_map werden an DataLODManager für composite_heightmap übertragen.
    DataLODManager kombiniert automatisch: composite_heightmap = base_heightmap + erosion_map + sedimentation_map.
    Alle nachfolgenden Generatoren erhalten transparent die composite_heightmap für realistische Post-Erosion-Simulation.
    Signal-Emission: composite_heightmap_updated informiert abhängige Generatoren über Terrain-Änderungen.

    Benötigte Shader für Hydrologische Simulation:

    Shader 1: Jump Flooding Lake Detection (jumpFloodLakes.frag)
    Implementiert parallelen Jump Flooding Algorithm für O(log n) Lake-Detection mit GPU-Optimierung.
    Initialisierungs-Phase identifiziert lokale Minima durch 8-Nachbar-Vergleich mit lake_volume_threshold.
    Propagations-Phasen: exponentiell abnehmende Sprung-Distanzen mit Lake-Seed-Information-Transfer.
    Output: Lake-Position, Wasser-Tiefe und Einzugsgebiet-Zuordnung für jede Zelle.

    Shader 2: Steepest Descent Flow Network (steepestDescentFlow.frag)
    Berechnet Flow-Network durch parallele Steepest Descent Analyse für jede Zelle.
    Steilster Gradient zu allen 8 Nachbarn bestimmt Fließrichtung mit Niederschlags-Quellen-Integration.
    Iterative Upstream-Akkumulation sammelt Wasser von allen upstream-Zellen über mehrere Pässe.
    Lake-Integration: Seen fungieren als Senken ohne Weiterleitung, Ocean-Outflow-Tracking.

    Shader 3: Manning Flow Calculation (manningFlowCalculation.frag)
    Löst Manning-Gleichung für optimale Fließgeschwindigkeit und Querschnitt-Geometrie.
    Gelände-Konfinierung-Analyse: lokale Höhenprofile bestimmen Tal-Breite für Breite-zu-Tiefe-Verhältnis.
    Iterative Optimierung von Breite/Tiefe unter Berücksichtigung hydraulischen Radius.
    Output: Fließgeschwindigkeit, Querschnittsfläche, Kanal-Geometrie für realistische Strömung.

    Shader 4: Stream Power Erosion (streamPowerErosion.frag)
    Implementiert Stream Power Erosion-Model mit Scherspannung-basierter Erosions-Rate.
    Hardness-Map-Integration: kritische Scherspannung basiert auf geologischer Gesteins-Härte.
    Erosions-Rate-Berechnung: E = K * (τ - τc) * v² mit Geschwindigkeits-Abhängigkeit.
    Kumulative Erosions-Akkumulation über mehrere Iterationen pro LOD-Level.

    Shader 5: Sediment Transport (sedimentTransport.frag)
    Simuliert Sediment-Transport entlang Flow-Paths mit Transport-Kapazitäts-Berechnung.
    HjulstrÖm-inspirierte Transport-Kapazität: Kapazität ∝ v^2.5 für realistische Sediment-Limits.
    Sedimentation bei Kapazitäts-Überschreitung oder Geschwindigkeits-Reduktion in Fluss-Biegungen.
    Distanz-abhängige Transport-Effizienz für realistische Sediment-Ablagerung-Patterns.

    Shader 6: Gaussian Soil Moisture (soilMoistureGaussian.frag)
    Berechnet Bodenfeuchtigkeit durch gewichtete Gaussian-Diffusion von allen Gewässern.
    Multipler-Radius-Filter: enge Filter für kapillare Effekte, weite Filter für Grundwasser.
    Gaussian-Gewichtung: exp(-0.5 * (d/σ)²) mit Distanz d und Standard-Abweichung σ.
    Maximum-Kombination verschiedener Filter plus direkte Wasser-Präsenz für finale Feuchtigkeit.

    Shader 7: Water Biomes Classification (waterBiomesClassification.frag)
    Regelbasierte Klassifikation von Gewässer-Types basierend auf Tiefe und Fließgeschwindigkeit.
    Lake-Detection-Integration für konsistente statische Gewässer-Identifikation.
    Flow-Rate-Thresholds für Creek/River/Grand River-Unterscheidung mit Konsistenz-Checks.

    Shader 8: Atmospheric Evaporation (atmosphericEvaporation.frag)
    Berechnet Evaporation basierend auf statischen Weather-Daten mit Magnus-Formel für Sättigung.
    Wind-Enhanced-Evaporation: lineare Wind-Verstärkung für verbesserten Dampf-Transport.
    Temperatur-Exponential-Abhängigkeit für realistische Evaporation-Rates.
    Verfügbare Wasser-Oberfläche limitiert tatsächliche Evaporation unabhängig von Atmosphäre.

    Shader 9: Erosion Data Upsampling (erosionUpsampling.frag)
    Spezialisiert auf bicubic Interpolation von Erosion/Sedimentation-Daten zwischen LOD-Levels.
    Pattern-Preservation: Erhaltung von Erosions-Strukturen bei Auflösungs-Verdopplung.
    Keine Mass-Conservation-Skalierung da Erosion absolute Höhen-Änderungen repräsentiert.

    Zusätzliche Utility-Shader:
    Composite Heightmap Generator (compositeHeightmap.frag): Kombiniert base + erosion + sedimentation
    Flow Pathfinding Convergence (flowPathfindingConvergence.frag): Überwacht Upstream-Akkumulation-Konvergenz
    Terrain Modification Validator (terrainModificationValidator.frag): Validiert Erosions-Plausibilität

    Alle Shader arbeiten im Multi-Pass-Verfahren mit Ping-Pong-Buffering für iterative Erosions-Zyklen
    und koordinierte LOD-Progression mit automatischer Upsampling-Integration.

    Klassen:
    HydrologySystemGenerator
        Funktionsweise: Hauptklasse für komplexes hydrologisches System mit bidirektionaler Terrain-Integration
        Aufgabe: Koordiniert Lake-Detection, Flow-Networks, Erosion-Sedimentation und Weather-Coupling
        External-Interface: calculate_water_system(heightmap, hardness_map, precip_map, temp_map, wind_map, parameters, lod_level)
        Internal-Methods: _coordinate_hydrology_generation(), _validate_multi_input_data(), _create_water_data()
        Erosion-Management: Iterative Erosion-Cycles pro LOD mit kumulative Akkumulation und Upsampling
        Error-Handling: Graceful Degradation bei Multi-Input-Failures, vollständige 3-stufige Fallback-Kette

    LakeDetectionSystem
        Funktionsweise: Jump Flooding Algorithm für parallele See-Identifikation mit GPU-Optimierung
        Aufgabe: Findet lokale Minima, klassifiziert Seen und bestimmt Einzugsgebiete
        Methoden: detect_local_minima(), apply_jump_flooding_passes(), classify_lake_basins(), validate_lake_topology()
        Optimization: Logarithmische Zeit-Komplexität O(log n) durch exponentiell abnehmende Sprung-Distanzen
        Spezifische Fallbacks:
          - GPU-Optimal: Parallele Jump-Flooding mit Compute-Shadern für alle Sprung-Phasen
          - CPU-Fallback: Optimierte NumPy-Implementierung mit Multiprocessing für Parallelisierung
          - Simple-Fallback: Direkte lokale Minima-Suche ohne komplexe Flood-Fill-Algorithmus

    FlowNetworkBuilder
        Funktionsweise: Steepest Descent Flow-Network mit Upstream-Akkumulation für realistische Flusssysteme
        Aufgabe: Erstellt flow_map und water_biomes_map durch topographie-basierte Flow-Pathfinding
        Methoden: calculate_steepest_descent(), accumulate_upstream_flow(), classify_water_bodies(), track_ocean_outflow()
        Integration: Precipitation-Sources von Weather-Generator, Lake-Sinks von Lake-Detection-System
        Spezifische Fallbacks:
          - GPU-Optimal: Parallele Steepest-Descent mit simultaner Upstream-Akkumulation
          - CPU-Fallback: Vectorized NumPy-Operations für Flow-Direction-Berechnung
          - Simple-Fallback: Vereinfachte Drainage-Network ohne komplexe Upstream-Akkumulation

    ManningFlowCalculator
        Funktionsweise: Manning-Gleichungs-Solver mit adaptiven Querschnitten für realistische Strömungs-Simulation
        Aufgabe: Berechnet flow_speed und cross_section für physikalisch korrekte Fließgeschwindigkeiten
        Methoden: solve_manning_equation(), optimize_channel_geometry(), calculate_hydraulic_radius(), analyze_terrain_confinement()
        Gelände-Integration: Tal-Breite-Analysis für natürliche Breite-zu-Tiefe-Verhältnisse
        Spezifische Fallbacks:
          - GPU-Optimal: Parallele Manning-Gleichungs-Lösung mit iterativer Querschnitt-Optimierung
          - CPU-Fallback: SciPy-Optimization für Kanal-Geometrie mit vectorized Berechnungen
          - Simple-Fallback: Fixed Breite-zu-Tiefe-Verhältnisse ohne Gelände-spezifische Optimierung

    ErosionSedimentationSystem
        Funktionsweise: Stream Power Erosion mit iterativen Terrain-Modifikationen über LOD-Levels
        Aufgabe: Modifiziert Landschaft durch erosion_map und sedimentation_map mit realistischem Sediment-Transport
        Methoden: calculate_stream_power(), simulate_erosion_iterations(), transport_sediment(), apply_sedimentation(), accumulate_terrain_changes()
        Iteration-Management: 10 Erosions-Zyklen pro LOD mit kumulative Landschafts-Veränderung
        Spezifische Fallbacks:
          - GPU-Optimal: Parallele Stream-Power-Berechnung mit simultaner Erosion-Sedimentation
          - CPU-Fallback: NumPy-vectorized Erosion-Math mit optimierten Sediment-Transport-Algorithmen
          - Simple-Fallback: Vereinfachte Erosion ohne komplexe Transport-Kapazitäts-Berechnungen

    SoilMoistureCalculator
        Funktionsweise: Gaussian-Diffusion-basierte Bodenfeuchtigkeit mit Multi-Radius-Filter-System
        Aufgabe: Erstellt soil_moist_map durch realistische Feuchtigkeits-Ausbreitung von Gewässern
        Methoden: apply_gaussian_diffusion(), combine_multi_radius_filters(), calculate_groundwater_effects(), integrate_direct_water_presence()
        Filter-System: Kombiniert kapillare und Grundwasser-Effekte für natürliche Feuchtigkeits-Verteilung
        Spezifische Fallbacks:
          - GPU-Optimal: Parallele Multi-Radius Gaussian-Filter mit optimierten Convolution-Operationen
          - CPU-Fallback: SciPy-Gaussian-Filter mit optimierten Multi-Pass-Verfahren
          - Simple-Fallback: Lineare Distanz-basierte Feuchtigkeit ohne Gaussian-Diffusion-Komplexität

    EvaporationCalculator
        Funktionsweise: Atmosphäre-gekoppelte Evaporation basierend auf statischen Weather-Daten
        Aufgabe: Berechnet evaporation_map durch Integration von temp_map, wind_map und humid_map
        Methoden: calculate_atmospheric_evaporation(), apply_magnus_formula(), enhance_wind_effects(), limit_by_available_water()
        Weather-Integration: Statische Snapshots ohne Real-time-Kopplung für Performance-Optimierung
        Spezifische Fallbacks:
          - GPU-Optimal: Parallele Magnus-Formel-Berechnung mit Wind-Enhancement-Integration
          - CPU-Fallback: Vectorized Atmospheric-Calculations mit optimierten Sättigungs-Berechnungen
          - Simple-Fallback: Fixed Evaporation-Rate ohne atmosphärische Komplexität

    BiDirectionalTerrainIntegrator
        Funktionsweise: Koordiniert bidirektionale Terrain-Modifikation zwischen Water-Generator und DataLODManager
        Aufgabe: Überträgt erosion_map/sedimentation_map für composite_heightmap-Erstellung
        Methoden: transfer_erosion_data(), coordinate_composite_heightmap_updates(), validate_terrain_modifications(), signal_terrain_changes()
        Integration: Transparente composite_heightmap-Bereitstellung für nachfolgende Generatoren
        DataLODManager-Coordination: Automatische Cache-Invalidation und Signal-Emission bei Terrain-Änderungen

    LODProgressionManager
        Funktionsweise: Verwaltet LOD-Progression mit Erosion-Upsampling und kumulative Akkumulation
        Aufgabe: Koordiniert Erosion-Iterationen pro LOD und Upsampling für nächstes LOD
        Methoden: manage_lod_progression(), upsample_erosion_data(), accumulate_cumulative_changes(), validate_lod_consistency()
        Upsampling-Strategy: Bicubic-Interpolation für glatte Erosions-Pattern-Erhaltung
        Performance-Optimization: Minimiert Memory-Usage durch progressive Erosion-Akkumulation

    Integration und Datenfluss:
    GenerationOrchestrator → HydrologySystemGenerator.calculate_water_system(multi_input_data, parameters, lod_level)
                          → LakeDetectionSystem.detect_lakes() → ShaderManager-Request
                          → FlowNetworkBuilder.build_flow_network() → ShaderManager-Request
                          → ManningFlowCalculator.calculate_flow_speeds() → ShaderManager-Request
                          → ErosionSedimentationSystem.simulate_erosion_iterations() → ShaderManager-Request (10x per LOD)
                          → SoilMoistureCalculator.calculate_soil_moisture() → ShaderManager-Request
                          → EvaporationCalculator.calculate_evaporation() → ShaderManager-Request
                          → BiDirectionalTerrainIntegrator.transfer_terrain_modifications()
                          → LODProgressionManager.manage_lod_progression()
                          → WaterData-Assembly → DataLODManager.set_water_data_lod() + composite_heightmap_update

    Output-Datenstrukturen:
    - water_map: 2D numpy.float32 array, Gewässertiefen in Metern
    - flow_map: 2D numpy.float32 array, Volumenstrom in m³/s
    - flow_speed: 2D numpy.float32 array, Fließgeschwindigkeit in m/s
    - soil_moist_map: 2D numpy.float32 array, Bodenfeuchtigkeit in Prozent
    - water_biomes_map: 2D numpy.uint8 array, Gewässer-Klassifikation (0-4)
    - erosion_map: 2D numpy.float32 array, kumulative Erosion in Metern (negative Werte)
    - sedimentation_map: 2D numpy.float32 array, kumulative Sedimentation in Metern (positive Werte)
    - evaporation_map: 2D numpy.float32 array, Verdunstungsrate in gH2O/m²/Tag
    - WaterData: Validity-State, Parameter-Hash, LOD-Metadata, Erosion-Statistics, Performance-Stats

    Spezielle Herausforderungen (für spätere Lösung):
    - Dynamic Flow-Pathfinding: Flow-Paths müssen sich während Erosion-Iterationen anpassen da sich die Topographie ändert
    - Cross-LOD Erosion-Balance: Parameter-Tuning erforderlich damit niedrige LODs nicht gesamte Landschaft erodieren
    - Performance-Optimization: 10 Erosion-Iterations × 7 LOD-Levels = 70 komplexe Berechnungs-Zyklen pro Generation
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
        OCTAVES = {"min": 1, "max": 10, "default": 4}1
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
    - LOD-Memory-Optimization: Automatisches Löschen niedrigerer LODs, wenn höhere LOD erfolgreich erstellt

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
    Generation-Trigger-Funktionalität.
    - parameter_changed:
        Signal übermittelt Tab-Name, Parameter-Name, alten Wert und neuen Wert für
        Tab-übergreifende Information.
    - cache_invalidation_requested:
        Signal kommuniziert Source-Tab und betroffene
        Generatoren für intelligente Cache-Verwaltung.
    - manual_generation_requested:
        Signal überträgt Generator-Type und Parameter-
        Dictionary für explizite Generierungs-Anfragen.

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
    Status: eventuell einige falsche Inhalte

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
    date_changed: 24.08.2025

    ## ÜBERSICHT UND ZIELSETZUNG

    BaseMapTab ist die fundamentale Basis-Klasse für alle spezialisierten Map-Editor Tabs.
    Sie implementiert ein standardisiertes 70/30 Layout-System, Cross-Tab-Kommunikation
    und robustes UI-Management mit klarer Trennung zwischen UI-Layer und Business-Logic.

    ═══════════════════════════════════════════════════════════════════════════════
    ## KERNVERANTWORTLICHKEITEN

    **UI-Layout und Display-Management:**
    - Standardisiertes 70/30 Layout mit QSplitter (Canvas links, Controls rechts)
    - 2D/3D Display-Stack mit Toggle-Controls und Fallback-Management
    - Parameter-UI-Controls als Proxy zum ParameterManager
    - Display-Update-Koordination ohne eigene Business-Logic

    **Manager-Integration als Proxy:**
    - ParameterManager: UI-Parameter ↔ Storage-Parameter Synchronisation
    - GenerationOrchestrator: Generation-Requests mit Parameter-Weiterleitung
    - DataLODManager: Display-Data-Retrieval für UI-Updates
    - NavigationManager: Tab-Navigation und Window-Geometry

    ═══════════════════════════════════════════════════════════════════════════════
    ## SIGNAL-ARCHITEKTUR

    **Outgoing Signals (UI-Events):**
    - `generate_requested(generator_type: str)` → GenerationOrchestrator
    - `parameter_ui_changed(param_name: str, value)` → ParameterManager
    - `display_mode_changed(mode: str)` → self.update_display()

    **Incoming Signals (UI-Updates):**
    - `parameter_manager.parameter_changed` → update_parameter_ui()
    - `data_lod_manager.data_updated` → update_display()
    - `generation_orchestrator.generation_progress` → update_progress_ui()

    ═══════════════════════════════════════════════════════════════════════════════
    ## UI-ARCHITEKTUR

    **Main Layout Structure:**
    BaseMapTab (QWidget)
    ├── QHBoxLayout (70/30 Split mit QSplitter)
    │   ├── Canvas Container (70%)
    │   │   ├── View Toggle Buttons (2D/3D)
    │   │   └── QStackedWidget (Display-Switcher)
    │   └── Control Widget (30%)
    │       ├── QScrollArea (Parameter-Controls - scrollable)
    │       └── Navigation Panel (Fixed - not scrollable)

    **Control Panel Composition:**
    QScrollArea (scrollable content)
    ├── Control Panel (QWidget) ← Sub-Classes add content here
    │   ├── Custom Parameter Controls ← Via control_panel.layout()
    │   ├── Auto-Simulation Panel ← Standardized across all tabs
    │   └── Error Status Display ← Shows dependency/generation status
    └── Navigation Panel (fixed) ← Always visible, not scrollable

    ═══════════════════════════════════════════════════════════════════════════════
    ## LIFECYCLE-MANAGEMENT

    **Initialization Sequence:**
    1. `__init__()`: Core attribute setup, Manager-Referenzen
    2. `setup_ui()`: UI structure creation mit Exception-Handling
       - `setup_layout_structure()`: 70/30 Split, Splitter-Konfiguration
       - `setup_display_stack()`: 2D/3D Views mit Fallback-Handling
       - `setup_control_panel()`: Scrollable Parameter-Area
       - `setup_navigation_panel()`: Fixed Navigation (not scrollable)
       - `setup_auto_simulation()`: Auto-Generation Controls
    3. `setup_signals()`: Manager-Signal-Verbindungen
    4. `setup_generation_handlers()`: GenerationOrchestrator Integration

    **Cleanup Sequence:**
    1. `cleanup_ui_resources()`: UI-spezifische Resource-Deallocation
    2. `disconnect_manager_signals()`: Exception-safe Signal-Disconnects
    3. Parameter-UI-State reset ohne Manager-Interferenz

    ═══════════════════════════════════════════════════════════════════════════════
    ## DISPLAY-SYSTEM

    **View Switching Architecture:**
    - 2D/3D Toggle: QStackedWidget mit DisplayWrapper-Abstraction
    - Fallback System: Graceful Degradation bei Display-Class-Unavailability
    - State Management: Active/Inactive Display-Coordination
    - Memory Management: Texture-Cleanup bei View-Switches

    **Display Update Pipeline:**
    1. `update_display_mode()` → `_render_current_mode()`
    2. Change Detection via Hash-Comparison (verhindert unnötige Updates)
    3. Data-Retrieval via DataLODManager.get_*_data()
    4. Apply new Data zu Active Display
    5. UI-Status Updates (Progress, Statistics)

    ═══════════════════════════════════════════════════════════════════════════════
    ## EXTENSIBILITY FÜR SUB-CLASSES

    **Required Implementations:**
    - `generator_type`: str = "terrain"/"geology" etc.
    - `required_dependencies`: List[str] = Dependencies von anderen Generatoren
    - `create_parameter_controls()`: UI-Controls für Generator-Parameter

    **Optional Overrides:**
    - `create_visualization_controls()`: Custom Display-Mode Buttons
    - `update_display_mode()`: Custom Display-Rendering Logic
    - `check_input_dependencies()`: Custom Dependency-Validation
    - `on_external_data_updated()`: Custom Cross-Tab Data-Handling

    **Standard Integration Points:**
    - `control_panel.layout()`: Add custom Parameter-Controls
    - `setup_generation_handlers(generator_type)`: GenerationOrchestrator Integration
    - Signal connections: Connect zu Manager-Signals für Updates

    ═══════════════════════════════════════════════════════════════════════════════
    ## PARAMETER-INTEGRATION

    **ParameterManager Integration:**
    - UI-Controls → ParameterManager.set_parameter()
    - Parameter-Changes ← ParameterManager.parameter_changed Signal
    - Validation-Display ← ParameterManager.validation_status_changed
    - Auto-Simulation Trigger ← ParameterManager Auto-Generation Logic

    **Parameter-UI-Proxy Pattern:**
    # UI-Control Change
    def on_slider_changed(self, value):
        self.parameter_manager.set_parameter(
            self.generator_type, "amplitude", value
        )

    # Manager Update
    def on_parameter_changed(self, generator, param_name, value):
        if generator == self.generator_type:
            self.update_parameter_ui(param_name, value)

    ═══════════════════════════════════════════════════════════════════════════════
    ## ERROR-HANDLING STRATEGY

    **Defensive Programming:**
    - Jede kritische Operation wrapped in try/except
    - Graceful Degradation: Continue operation auch bei non-critical Failures
    - UI-Safety: Cleanup guaranteed auch bei Exceptions
    - User Feedback: Clear Error-Messages in UI-Status-Displays

    **Manager-Integration Safety:**
    - Manager-Unavailability: UI funktioniert auch ohne Manager-Verbindungen
    - Signal-Resilience: UI-Updates funktionieren auch bei Signal-Failures
    - Resource-Safety: UI-Cleanup unabhängig von Manager-States

    ═══════════════════════════════════════════════════════════════════════════════
    ## KLASSEN

    **BaseMapTab**
    - Funktionsweise: Vereinfachte Basis-Klasse für UI-Management ohne Business-Logic
    - Aufgabe: Layout, Display-Coordination, Parameter-UI-Proxy zu Managern
    - Methoden: setup_ui(), create_parameter_controls(), update_display()
    - Manager-Integration: Reine Proxy-Funktionen zu ParameterManager und GenerationOrchestrator

    **ParameterUIProxy**
    - Funktionsweise: Verbindet Parameter-UI-Controls mit ParameterManager
    - Aufgabe: UI-Control-Creation, Value-Updates, Validation-Display
    - Methoden: create_sliders(), update_ui_values(), show_validation_errors()

    **DisplayCoordinator**
    - Funktionsweise: Koordiniert 2D/3D Display-Updates ohne Business-Logic
    - Aufgabe: Display-Mode-Switching, Overlay-Management, Update-Coordination
    - Methoden: switch_display_mode(), apply_overlays(), refresh_display()
    """

def terrain_tab():
    """
    Path: gui/tabs/terrain_tab.py
    Date changed: 24.08.2025

    ═══════════════════════════════════════════════════════════════════════════════
    ## ÜBERSICHT UND ZIELSETZUNG

    TerrainTab implementiert die Terrain-Generator UI mit vollständiger BaseMapTab-Integration
    und direkter Anbindung an den TerrainGenerator aus core/terrain_generator.py. Als Basis-Generator
    ohne Dependencies liefert er heightmap, slopemap und shadowmap für alle nachgelagerten Systeme.

    ═══════════════════════════════════════════════════════════════════════════════
    ## CORE-GENERATOR INTEGRATION

    **BaseTerrainGenerator Anbindung:**
    - Core-Call: BaseTerrainGenerator.calculate_heightmap(parameters, lod_level)
    - Heightmap-Generation: Multi-Scale Simplex-Noise mit Octave-Layering
    - Ridge-Warping: Deformation für natürliche Landschaftsformen
    - Height-Redistribution: Natürliche Höhenverteilung basierend auf redistribute_power

    **TerrainData Output:**
    - heightmap: 2D numpy.float32 array, Elevation in Metern
    - slopemap: 3D numpy.float32 array (H,W,2), dz/dx und dz/dy Gradienten
    - shadowmap: 3D numpy.float32 array (H,W,7), Shadow-Values [0-1] für 7 Sonnenwinkel

    ═══════════════════════════════════════════════════════════════════════════════
    ## UI-LAYOUT (erweitert BaseMapTab)

    **Control Panel Parameter-Sektion:**
    Terrain Parameters GroupBox:
    ├── Map Size: ParameterSlider (value_default.TERRAIN.SIZE)
    ├── Height Amplitude: ParameterSlider (value_default.TERRAIN.HEIGHT) + suffix:"m"
    ├── Detail Octaves: ParameterSlider (value_default.TERRAIN.OCTAVES)
    ├── Base Frequency: ParameterSlider (value_default.TERRAIN.FREQUENCY)
    ├── Detail Persistence: ParameterSlider (value_default.TERRAIN.PERSISTENCE)
    ├── Frequency Scaling: ParameterSlider (value_default.TERRAIN.LACUNARITY)
    ├── Height Redistribution: ParameterSlider (value_default.TERRAIN.REDISTRIBUTE)
    └── Map Seed: ParameterSlider + RandomSeedButton (widgets.py)

    Generation Control GroupBox:
    ├── "Berechnen"-Button: Manual Generation-Trigger
    ├── Generation Progress: ProgressBar mit LOD-Phase-Details
    └── System Status: LOD-Level-Display, Generation-Status

    Terrain Statistics GroupBox:
    ├── Height Range: Min/Max/Mean/StdDev
    ├── Slope Statistics: Max-Gradient, Mean-Slope in Degrees
    ├── Shadow Coverage: Min/Max/Mean Shadow-Values über alle Sonnenwinkel
    └── Performance Metrics: Data-Size, Generation-Time

    **Canvas Visualization Controls:**
    Terrain-spezifische Controls (nur 2D):
    └── Height/Slope: Radio-Buttons (exklusiv für Terrain-2D-Display)

    Standard BaseMapTab Controls (2D+3D):
    ├── Shadow: Checkbox (Overlay) + Shadow-Angle-Slider (0-6, entspricht 7 Sonnenstände)
    ├── Contour Lines: Checkbox (Overlay mit Heightmap)
    └── Grid Overlay: Checkbox (für Measurement und Scale-Reference)

    ═══════════════════════════════════════════════════════════════════════════════
    ## GENERATION-WORKFLOW

    **Generation-Flow:**
    1. "Berechnen"-Button → generate_terrain_system()
    2. Parameter-Validation → GenerationOrchestrator.request_generation("terrain")
    3. BaseTerrainGenerator.calculate_heightmap() → TerrainData Creation
    4. LOD-Progression über GenerationOrchestrator mit incremental Results
    5. DataLODManager.set_terrain_data_lod() → Signal-Emission
    6. Display-Update über BaseMapTab._render_current_mode() nach jedem LOD
    7. Statistics-Update in Real-time, Map-Size-Sync zu dependent Tabs

    **LOD-System Integration:**
    - LOD-Progression: 64→128→256→512→1024→2048 bis map_size erreicht
    - LOD-Level: 1,2,3,4,5,6+ (numerisches System entsprechend DataLODManager)
    - Progressive Enhancement: UI-Update nach jedem LOD-Level
    - Shadow-Progression: 1,3,5,7 Sonnenwinkel je nach LOD-Level

    ═══════════════════════════════════════════════════════════════════════════════
    ## PARAMETER-SYSTEM

    **Map-Size Synchronisation:**
    - Power-of-2 Validation (64,128,256,512,1024,2048)
    - Map-Size-Sync zu anderen Tabs über DataLODManager.sync_map_size()
    - Cross-Parameter-Constraints: Octaves vs Size Validation

    **Seed-System:**
    - RandomSeedButton aus widgets.py generiert Seeds
    - Reproduzierbare Results über map_seed Parameter
    - Seed-Validation und Range-Checks

    ═══════════════════════════════════════════════════════════════════════════════
    ## MANAGER-INTEGRATION

    **Kommunikationskanäle:**
    - Config: value_default.TERRAIN für Parameter-Ranges, Validation-Rules, Default-Values
    - Core: core/terrain_generator.py → BaseTerrainGenerator, SimplexNoiseGenerator, ShadowCalculator
    - Manager: GenerationOrchestrator.request_generation("terrain") mit Parameter-Set
    - Data: DataLODManager.set_terrain_data_lod() für geology/settlement/weather Dependencies
    - Widgets: widgets.py für ParameterSlider, RandomSeedButton, StatusIndicator

    **Signal-Flow:**
    TerrainTab.generate_requested("terrain")
    → GenerationOrchestrator.request_generation("terrain", parameters)
    → BaseTerrainGenerator.calculate_heightmap()
    → DataLODManager.set_terrain_data_lod()
    → data_updated("terrain") Signal
    → TerrainTab.on_data_updated()
    → TerrainTab.update_display()


    ═══════════════════════════════════════════════════════════════════════════════
    ## ERROR-HANDLING UND VALIDATION

    **Parameter-Validation:**
    - Range-Checks über value_default.TERRAIN Constraints
    - Cross-Parameter-Constraints (Octaves vs Size)
    - Real-time Validation-Feedback in UI

    **Generation-Validation:**
    - TerrainData Shape/Range/NaN-Checks
    - LOD-Consistency zwischen LOD-Levels
    - GPU-Fallback über ShaderManager mit CPU-Fallback-Notification

    **Memory-Management:**
    - Large-Array-Detection für >50MB Arrays
    - Force-GC nach Generation-Completion
    - Resource-Cleanup über DataLODManager.ResourceTracker

    ═══════════════════════════════════════════════════════════════════════════════
    ## ABHÄNGIGKEITEN UND OUTPUT

    **Dependencies:**
    - required_dependencies: [] (Basis-Generator ohne Input-Dependencies)
    - No Dependency-Checks erforderlich

    **Output für nachgelagerte Generatoren:**
    - heightmap → geology_generator, weather_generator, water_generator
    - slopemap → geology_generator, settlement_generator
    - shadowmap → alle Generatoren für Overlay-Visualisierung
    - Map-Size-Synchronisation → alle abhängigen Tabs
    """

def geology_tab():
    """
    Path: gui/tabs/geology_tab.py
    Date Changed: 24.08.2025

    ## ÜBERSICHT UND ZIELSETZUNG

    GeologyTab implementiert die Geology-Generator UI mit BaseMapTab-Integration und geologischer
    Simulation basierend auf Terrain-Input. Erzeugt Rock-Type-Classification und Hardness-Maps
    für realistische geologische Strukturen mit Mass-Conservation-System.

    ## CORE-GENERATOR INTEGRATION

    **GeologyGenerator Anbindung:**
    - Core-Call: GeologyGenerator.calculate_geology(heightmap, slopemap, parameters, lod_level)
    - Rock-Classification: Sedimentary/Igneous/Metamorphic mit Mass-Conservation (R+G+B=255)
    - Geological-Zones: Multi-Zone Simplex-Noise für realistische Gesteinsverteilung
    - Tectonic-Deformation: Ridge/Bevel-Warping, Metamorphic-Foliation/Folding, Igneous-Flowing

    **GeologyData Output:**
    - rock_map: 3D numpy.uint8 array (H,W,3), RGB-Kanäle für Sedimentary/Igneous/Metamorphic mit R+G+B=255
    - hardness_map: 2D numpy.float32 array, Gesteinshärte-Werte [0-100] für Erosions-Simulation

    ## UI-LAYOUT (erweitert BaseMapTab)

    **Control Panel Parameter-Sektion:**
    Rock Hardness Parameters GroupBox:
    ├── Sedimentary Hardness: ParameterSlider (value_default.GEOLOGY.SEDIMENTARY_HARDNESS)
    ├── Igneous Hardness: ParameterSlider (value_default.GEOLOGY.IGNEOUS_HARDNESS)
    └── Metamorphic Hardness: ParameterSlider (value_default.GEOLOGY.METAMORPHIC_HARDNESS)

    Tectonic Deformation Parameters GroupBox:
    ├── Ridge Warping: ParameterSlider (value_default.GEOLOGY.RIDGE_WARPING)
    ├── Bevel Warping: ParameterSlider (value_default.GEOLOGY.BEVEL_WARPING)
    ├── Metamorphic Foliation: ParameterSlider (value_default.GEOLOGY.METAMORPHIC_FOLIATION)
    ├── Metamorphic Folding: ParameterSlider (value_default.GEOLOGY.METAMORPHIC_FOLDING)
    └── Igneous Flowing: ParameterSlider (value_default.GEOLOGY.IGNEOUS_FLOWING)

    Generation Control GroupBox:
    ├── "Berechnen"-Button: Manual Generation-Trigger
    ├── Generation Progress: ProgressBar mit Geology-Phase-Details
    └── Input Dependencies: StatusIndicator für heightmap/slopemap-Verfügbarkeit

    Rock Distribution Widget:
    ├── Hardness Preview: Progress-Bars für Sedimentary/Igneous/Metamorphic
    ├── Distribution Statistics: Prozentuale Verteilung nach Generation
    └── Mass Conservation Status: StatusIndicator für R+G+B=255 Validation

    **Canvas Visualization Controls:**
    Geology-spezifische Controls:
    └── Rock Types/Hardness: Radio-Buttons (2D+3D)

    Standard BaseMapTab Controls (2D+3D):
    ├── Shadow: Checkbox (Overlay über alle Modi, nutzt Terrain-Shadowmap)
    ├── Contour Lines: Checkbox (Overlay, nutzt Terrain-Heightmap) + Shadow-Angle-Slider (0-6)
    └── Grid Overlay: Checkbox (für alle Modi)

    ## GENERATION-WORKFLOW

    **Generation-Flow:**
    1. Dependency-Check → heightmap/slopemap-Validation → "Berechnen"-Button
    2. generate_geology_system() → GenerationOrchestrator.request_generation("geology")
    3. GeologyGenerator.calculate_geology() → Rock-Classification + Mass-Conservation + Hardness-Calculation
    4. LOD-Progression über GenerationOrchestrator entsprechend Terrain-LODs
    5. DataLODManager.set_geology_data_lod() → Signal-Emission
    6. Display-Update → Statistics-Update → Cross-Tab-Notification für Dependencies

    **LOD-System Integration:**
    - LOD-Progression: Automatische Verdopplung bis map_size erreicht (entsprechend Terrain)
    - LOD-Size-Verdopplung: 64→128→256→512→1024→2048 bis map_size erreicht
    - LOD-Level-Nummerierung: 1,2,3,4,5,6+ entsprechend verfügbaren Terrain-LODs
    - Progressive Enhancement: Deformation-Detail und Zone-Complexity steigen mit LOD-Level

    ## DEPENDENCY-MANAGEMENT

    **Input Dependencies:**
    - required_dependencies: ["heightmap", "slopemap"] von terrain_tab
    - check_input_dependencies(): Prüft Terrain-Data-Verfügbarkeit, Shape-Consistency, Data-Quality
    - Dependency-Status-Display: StatusIndicator mit missing/invalid Inputs und Recovery-Suggestions

    **Input-Validation:**
    - heightmap/slopemap Shape/Range/Consistency-Checks
    - Data-Quality-Validation (NaN-Detection, Range-Validation)
    - LOD-Level-Consistency zwischen Terrain- und Geology-Data

    ## MASS-CONSERVATION-SYSTEM

    **R+G+B=255 Enforcement:**
    - Mass-Conservation: Proportionale Skalierung wenn R+G+B != 255
    - Fallback: Gleichverteilung (85,85,85) bei R+G+B=0 Pixeln
    - Real-time Validation: Mass-Conservation-Status in UI
    - Error-Recovery: Automatic Repair bei Mass-Conservation-Violations

    ## MANAGER-INTEGRATION

    **Kommunikationskanäle:**
    - Config: value_default.GEOLOGY für Parameter-Ranges, Hardness-Defaults, Validation-Rules
    - Core: core/geology_generator.py → GeologyGenerator, RockTypeClassifier, MassConservationManager
    - Input: DataLODManager.get_terrain_data("heightmap/slopemap") für Basis-Terrain-Daten
    - Output: DataLODManager.set_geology_data_lod() für rock_map/hardness_map zu water_generator
    - Manager: GenerationOrchestrator.request_generation("geology") nach Dependency-Erfüllung
    - Widgets: widgets.py für ParameterSlider, StatusIndicator, ProgressBar

    **Signal-Flow:**
    GeologyTab.generate_requested("geology")
    → GenerationOrchestrator.request_generation("geology", parameters)
    → GeologyGenerator.calculate_geology()
    → DataLODManager.set_geology_data_lod()
    → data_updated("geology") Signal
    → GeologyTab.on_data_updated()
    → GeologyTab.update_display()

    ## DISPLAY-MODI (erweitert BaseMapTab)

    **Visualization Modes:**
    - Height Mode: Terrain-Heightmap als Basis-Layer
    - Rock Types Mode: RGB rock_map mit Color-Coding (Rot=Sedimentary, Grün=Igneous, Blau=Metamorphic)
    - Hardness Mode: Hardness-Map mit Grayscale/Color-Coding (Blau=weich, Rot=hart)
    - 3D Terrain Overlay: Kombiniert Rock-Classification mit 3D-Terrain-Rendering

    **Overlay-System:**
    - Shadow/Contour/Grid Overlays: Nutzen Terrain-Data über alle Geology-Modi
    - Display-Optimization: Immer bestes verfügbares LOD für Display

    ## ERROR-HANDLING UND VALIDATION

    **Parameter-Validation:**
    - Range-Checks über value_default.GEOLOGY Constraints
    - Hardness-Parameter [0-100] Validation
    - Deformation-Parameter [0.0-1.0] Validation

    **Geology-Data-Validation:**
    - Rock-Map RGB-Range-Checks [0-255]
    - Mass-Conservation-Validation (R+G+B=255)
    - Hardness-Map Range-Checks [0-100]

    **Error-Recovery:**
    - Invalid-Input-Recovery mit Default-Rock-Distribution
    - Mass-Conservation-Repair bei R+G+B != 255
    - Automatic Fallback bei Critical-Generation-Failures

    ## ABHÄNGIGKEITEN UND OUTPUT

    **Dependencies:**
    - required_dependencies: ["heightmap", "slopemap"] von terrain_tab
    - Dependency-Validation erforderlich vor Generation

    **Output für nachgelagerte Generatoren:**
    - rock_map → water_generator für Erosions-Simulation
    - hardness_map → water_generator für variable Erosions-Resistance
    - Geological-Data → settlement_generator für Suitability-Analysis
    """

def weather_tab():
   """
   Path: gui/tabs/weather_tab.py
   Date Changed: 24.08.2025

   ═══════════════════════════════════════════════════════════════════════════════
   ## ÜBERSICHT UND ZIELSETZUNG
   ═══════════════════════════════════════════════════════════════════════════════

   WeatherTab implementiert die Weather-Generator UI mit BaseMapTab-Integration und
   dynamischer Klimasimulation basierend auf Terrain-Input. Erzeugt realistische
   Temperatur-, Niederschlags-, Wind- und Feuchtigkeitsfelder durch CFD-basierte
   Atmosphären-Simulation mit orographischen Effekten.

   ═══════════════════════════════════════════════════════════════════════════════
   ## CORE-GENERATOR INTEGRATION
   ═══════════════════════════════════════════════════════════════════════════════

   **WeatherSystemGenerator Anbindung:**
   - Core-Call: WeatherSystemGenerator.generate_weather_system(heightmap, shadowmap, parameters, lod_level)
   - Climate-Modeling: Terrain-basierte Temperaturberechnung mit Altitude/Solar/Latitude-Effekten
   - Wind-Field-Simulation: CFD-basierte Luftströmung mit Berg-Ablenkung und Düseneffekten
   - Orographischer Niederschlag: Luv-/Lee-Effekte mit Feuchtigkeitstransport
   - Atmospheric-Moisture: Evaporation/Kondensation-Zyklen mit Temperatur-Kopplung

   **WeatherData Output:**
   - temp_map: 2D numpy.float32 array, Lufttemperatur in °C
   - precip_map: 2D numpy.float32 array, Niederschlag in gH2O/m²
   - wind_map: 3D numpy.float32 array (H,W,2), Windvektoren (u,v) in m/s
   - humid_map: 2D numpy.float32 array, Luftfeuchtigkeit in gH2O/m³

   ═══════════════════════════════════════════════════════════════════════════════
   ## UI-LAYOUT (erweitert BaseMapTab)
   ═══════════════════════════════════════════════════════════════════════════════

   **Control Panel Parameter-Sektion:**
   Climate Base Parameters GroupBox:
   ├── Air Temperature Entry: ParameterSlider (value_default.WEATHER.AIR_TEMP_ENTRY) + suffix:"°C"
   ├── Solar Power: ParameterSlider (value_default.WEATHER.SOLAR_POWER) + suffix:"°C"
   ├── Altitude Cooling: ParameterSlider (value_default.WEATHER.ALTITUDE_COOLING) + suffix:"°C/100m"
   └── Latitude Effect: ParameterSlider (value_default.WEATHER.LATITUDE_EFFECT) + suffix:"°C"

   Wind System Parameters GroupBox:
   ├── Wind Speed Factor: ParameterSlider (value_default.WEATHER.WIND_SPEED_FACTOR) + suffix:"m/s/Pa"
   ├── Thermic Effect: ParameterSlider (value_default.WEATHER.THERMIC_EFFECT) + suffix:"factor"
   ├── Terrain Factor: ParameterSlider (value_default.WEATHER.TERRAIN_FACTOR) + suffix:"factor"
   └── Flow Direction: ParameterSlider (value_default.WEATHER.FLOW_DIRECTION) + suffix:"degrees"

   Atmospheric Parameters GroupBox:
   ├── Base Humidity: ParameterSlider (value_default.WEATHER.BASE_HUMIDITY) + suffix:"gH2O/m³"
   ├── Evaporation Rate: ParameterSlider (value_default.WEATHER.EVAPORATION_RATE) + suffix:"factor"
   ├── Condensation Threshold: ParameterSlider (value_default.WEATHER.CONDENSATION_THRESHOLD) + suffix:"factor"
   └── Weather Seed: ParameterSlider + RandomSeedButton (widgets.py)

   Generation Control GroupBox:
   ├── "Berechnen"-Button: Manual Generation-Trigger
   ├── Generation Progress: ProgressBar mit CFD-Iteration-Details
   └── Input Dependencies: StatusIndicator für heightmap/shadowmap-Verfügbarkeit

   Climate Statistics GroupBox:
   ├── Temperature Range: Min/Max/Mean/StdDev in °C
   ├── Precipitation Stats: Total/Max/Mean/Distribution in mm/Jahr
   ├── Wind Statistics: Max-Speed, Mean-Speed, Dominant-Direction
   ├── Humidity Distribution: Min/Max/Mean Luftfeuchtigkeit
   └── Performance Metrics: CFD-Iterations, Solver-Time, Convergence-Rate


   **Canvas Visualization Controls:**
   Weather-spezifische Controls (2D+3D):
   └── Temperature/Precipitation/Wind/Humidity: Radio-Buttons (exklusiv für Weather-Modi)

   Standard BaseMapTab Controls (2D+3D):
   ├── Shadow: Checkbox (Overlay über alle Modi, nutzt Terrain-Shadowmap) + Shadow-Angle-Slider (0-6)
   ├── Contour Lines: Checkbox (Overlay mit Heightmap für alle Weather-Modi)
   ├── Grid Overlay: Checkbox (für alle Modi)
   └── Wind Vectors: Checkbox (3D-Arrows über Terrain in allen Modi)


   ═══════════════════════════════════════════════════════════════════════════════
   ## GENERATION-WORKFLOW
   ═══════════════════════════════════════════════════════════════════════════════

   **Generation-Flow:**
   1. Dependency-Check → heightmap/shadowmap-Validation → "Berechnen"-Button
   2. generate_weather_system() → GenerationOrchestrator.request_generation("weather")
   3. WeatherSystemGenerator.generate_weather_system() → CFD-Simulation + Climate-Modeling
   4. LOD-Progression über GenerationOrchestrator mit steigender CFD-Komplexität
   5. DataLODManager.set_weather_data_lod() → Signal-Emission
   6. Display-Update → Statistics-Update → Cross-Tab-Notification für Dependencies

   **LOD-System Integration:**
   - LOD-Progression: CFD-Grid skaliert automatisch bis map_size erreicht (entsprechend Terrain)
   - LOD-Grid-Scaling: 32x32→64x64→128x128→256x256→512x512→1024x1024→2048x2048
   - LOD-Level-Nummerierung: 1,2,3,4,5,6+ entsprechend verfügbaren Terrain-LODs
   - Progressive Enhancement: CFD-Solver-Iterations steigen von 3→5→7→10→15→20 mit LOD-Level
   - Performance-Scaling: Wind-Field-Auflösung und Atmospheric-Detail steigen progressiv

   ═══════════════════════════════════════════════════════════════════════════════
   ## DEPENDENCY-MANAGEMENT
   ═══════════════════════════════════════════════════════════════════════════════

   **Input Dependencies:**
   - required_dependencies: ["heightmap", "shadowmap"] von terrain_tab
   - check_input_dependencies(): Prüft Terrain-Data-Verfügbarkeit, Orographic-Data-Quality, Solar-Data-Consistency
   - Dependency-Status-Display: StatusIndicator mit missing/invalid Inputs und Recovery-Suggestions

   **Input-Validation:**
   - heightmap Shape/Range/Orographic-Suitability-Checks für Wind-Deflection
   - shadowmap Solar-Data-Quality-Validation für Temperature-Calculation
   - Data-Quality-Validation (NaN-Detection, Physical-Range-Validation)
   - LOD-Level-Consistency zwischen Terrain- und Weather-Data

   **Orographic Effects Integration:**
   - Terrain-Height → Altitude-Cooling (6°C/100m default)
   - Terrain-Slope → Wind-Deflection und Orographic-Lifting
   - Shadow-Data → Solar-Heating-Variation über Tagesverlauf
   - Valley-Detection → Wind-Channeling und Temperature-Inversion

   ═══════════════════════════════════════════════════════════════════════════════
   ## MANAGER-INTEGRATION
   ═══════════════════════════════════════════════════════════════════════════════

   **Kommunikationskanäle:**
   - Config: value_default.WEATHER für Parameter-Ranges, Climate-Defaults, CFD-Settings
   - Core: core/weather_generator.py → WeatherSystemGenerator, TemperatureCalculator, WindFieldSimulator
   - Input: DataLODManager.get_terrain_data("heightmap/shadowmap") für Orographic-Base-Data
   - Output: DataLODManager.set_weather_data_lod() für temp_map/precip_map/wind_map zu water_generator
   - Manager: GenerationOrchestrator.request_generation("weather") nach Dependency-Erfüllung
   - Widgets: widgets.py für ParameterSlider, StatusIndicator, ProgressBar, RandomSeedButton

   **Signal-Flow:**
   WeatherTab.generate_requested("weather")
   ↑ GenerationOrchestrator.request_generation("weather", parameters)
   ↑ WeatherSystemGenerator.generate_weather_system()
   ↑ DataLODManager.set_weather_data_lod()
   ↑ data_updated("weather") Signal
   ↑ WeatherTab.on_data_updated()
   ↑ WeatherTab.update_display()

   ═══════════════════════════════════════════════════════════════════════════════
   ## DISPLAY-MODI (erweitert BaseMapTab)
   ═══════════════════════════════════════════════════════════════════════════════

   **Visualization Modes:**
   - Temperature Mode: temp_map mit Color-Coding (Blau=-20°C → Rot=50°C)
   - Precipitation Mode: precip_map mit Niederschlags-Intensität (Weiß=0mm → Dunkelblau=200mm)
   - Wind Mode: wind_map als 2D-Vektorfeld + 3D-Wind-Arrows über Terrain-Mesh
   - Humidity Mode: humid_map mit Feuchtigkeits-Gradienten (Gelb=trocken → Grün=feucht)

   **3D-Integration:**
   - Wind-Vectors: 3D-Arrows skaliert nach wind_map-Magnitude über 3D-Terrain
   - Temperature-Overlay: Color-coded Temperature auf 3D-Terrain-Surface
   - Precipitation-Visualization: Animated Rain-Effects basierend auf precip_map
   - Atmospheric-Effects: Fog/Mist-Rendering basierend auf Humidity-Levels

   **Overlay-System:**
   - Shadow/Contour/Grid Overlays: Nutzen Terrain-Data über alle Weather-Modi
   - Display-Optimization: Immer bestes verfügbares LOD für Display
   - Multi-Layer-Rendering: Kombiniert Weather-Data mit Terrain-Base-Layer

   ═══════════════════════════════════════════════════════════════════════════════
   ## ERROR-HANDLING UND VALIDATION
   ═══════════════════════════════════════════════════════════════════════════════

   **Parameter-Validation:**
   - Range-Checks über value_default.WEATHER Constraints
   - Temperature-Parameter [-50°C bis +60°C] Validation
   - Wind-Parameter [0-200 km/h] Physical-Plausibility-Checks
   - CFD-Parameter [Iteration-Limits, Convergence-Criteria] Technical-Validation

   **Weather-Data-Validation:**
   - Temperature-Map Physical-Range-Checks [-50°C bis +60°C]
   - Precipitation-Map Range-Validation [0-500mm Niederschlag]
   - Wind-Map Velocity-Limit-Checks [0-200km/h Maximum]
   - Humidity-Map Saturation-Limit-Validation [0-100% relative Feuchtigkeit]

   **CFD-System-Validation:**
   - Solver-Convergence-Monitoring für Wind-Field-Stability
   - Mass-Conservation-Checks für Atmospheric-Moisture-Transport
   - Energy-Conservation-Validation für Temperature-Heat-Balance
   - Numerical-Stability-Checks für CFD-Iteration-Convergence

   **Error-Recovery:**
   - Invalid-Input-Recovery mit Default-Climate-Conditions
   - CFD-Solver-Recovery bei Numerical-Instability mit Simplified-Physics
   - Automatic Fallback bei Critical-Generation-Failures zu Linear-Climate-Model

   ═══════════════════════════════════════════════════════════════════════════════
   ## ABHÄNGIGKEITEN UND OUTPUT
   ═══════════════════════════════════════════════════════════════════════════════

   **Dependencies:**
   - required_dependencies: ["heightmap", "shadowmap"] von terrain_tab
   - Dependency-Validation erforderlich vor Generation
   - Orographic-Data-Quality-Checks für realistische Climate-Effects

   **Output für nachgelagerte Generatoren:**
   - temp_map → biome_generator für Temperature-based Biome-Classification
   - precip_map → water_generator für Precipitation-driven River-Generation
   - precip_map → biome_generator für Whittaker-Biome-Classification
   - wind_map → water_generator für Wind-driven Evaporation-Calculation
   - humid_map → water_generator für Atmospheric-Moisture-Cycling
   - Climate-Data → settlement_generator für Climate-Suitability-Analysis
   """

def water_tab():
   """
   Path: gui/tabs/water_tab.py
   Date Changed: 24.08.2025

   ═══════════════════════════════════════════════════════════════════════════════
   ## ÜBERSICHT UND ZIELSETZUNG
   ═══════════════════════════════════════════════════════════════════════════════

   WaterTab implementiert die Water-Generator UI mit BaseMapTab-Integration und
   komplexer hydrologischer Simulation basierend auf Terrain-, Geology- und Weather-Input.
   Erzeugt realistische Wassersysteme mit Erosion, Sedimentation und bidirektionaler
   Terrain-Modifikation durch physikalisch korrekte Sediment-Transport-Simulation.

   ═══════════════════════════════════════════════════════════════════════════════
   ## CORE-GENERATOR INTEGRATION
   ═══════════════════════════════════════════════════════════════════════════════

   **HydrologySystemGenerator Anbindung:**
   ├ Core-Call: HydrologySystemGenerator.generate_hydrology_system(heightmap, hardness_map, precip_map, temp_map, wind_map, parameters, lod_level)
   ├ Lake-Detection: Jump Flooding Algorithm für parallele Senken-Identifikation
   ├ Flow-Network: Steepest Descent mit Upstream-Akkumulation für Flusssysteme
   ├ Erosion-Simulation: Stream Power Erosion mit HjulstrÃ¶m-Sundborg Transport
   ├ Terrain-Feedback: Bidirektionale heightmap-Modifikation durch Erosions-/Sedimentationsprozesse

   **HydrologyData Output:**
   ├ water_map: 2D numpy.float32 array, Gewässertiefen in m
   ├ flow_map: 2D numpy.float32 array, Volumenstrom in m³/s
   ├ flow_speed: 2D numpy.float32 array, Fließgeschwindigkeit in m/s
   ├ soil_moist_map: 2D numpy.float32 array, Bodenfeuchtigkeit in %
   ├ erosion_map: 2D numpy.float32 array, Erosionsrate in m/Jahr
   ├ sedimentation_map: 2D numpy.float32 array, Sedimentationsrate in m/Jahr
   ├ water_biomes_map: 2D numpy.uint8 array, Wasser-Klassifikation (0=kein Wasser, 1=Creek, 2=River, 3=Grand River, 4=Lake)
   ├ heightmap_modified: 2D numpy.float32 array, durch Erosion/Sedimentation modifizierte Höhenkarte

   ═══════════════════════════════════════════════════════════════════════════════
   ## UI-LAYOUT (erweitert BaseMapTab)
   ═══════════════════════════════════════════════════════════════════════════════

   **Control Panel Parameter-Sektion:**

   ├ Hydrology Base Parameters GroupBox:
   ├── Lake Volume Threshold: ParameterSlider (value_default.WATER.LAKE_VOLUME_THRESHOLD) + suffix:"m"
   ├── Rain Threshold: ParameterSlider (value_default.WATER.RAIN_THRESHOLD) + suffix:"gH2O/m²"
   ├── Manning Coefficient: ParameterSlider (value_default.WATER.MANNING_COEFFICIENT) + suffix:"roughness"
   ├── Flow Iterations: ParameterSlider (value_default.WATER.FLOW_ITERATIONS) + suffix:"cycles"

   ├ Erosion & Sedimentation Parameters GroupBox:
   ├── Erosion Strength: ParameterSlider (value_default.WATER.EROSION_STRENGTH) + suffix:"factor"
   ├── Sediment Capacity Factor: ParameterSlider (value_default.WATER.SEDIMENT_CAPACITY_FACTOR) + suffix:"factor"
   ├── Settling Velocity: ParameterSlider (value_default.WATER.SETTLING_VELOCITY) + suffix:"m/s"
   ├── Transport Efficiency: ParameterSlider (value_default.WATER.TRANSPORT_EFFICIENCY) + suffix:"factor"

   ├ Atmospheric Coupling Parameters GroupBox:
   ├── Evaporation Base Rate: ParameterSlider (value_default.WATER.EVAPORATION_BASE_RATE) + suffix:"m/Tag"
   ├── Diffusion Radius: ParameterSlider (value_default.WATER.DIFFUSION_RADIUS) + suffix:"pixel"
   ├── Soil Moisture Factor: ParameterSlider (value_default.WATER.SOIL_MOISTURE_FACTOR) + suffix:"factor"
   ├── Water Seed: ParameterSlider + RandomSeedButton (widgets.py)

   ├ Generation Control GroupBox:
   ├── "Berechnen"-Button: Manual Generation-Trigger
   ├── Generation Progress: ProgressBar mit Hydrology-Phase-Details (Lake-Detection→Flow-Network→Erosion→Sedimentation)
   ├── Input Dependencies: StatusIndicator für heightmap/hardness_map/precip_map-Verfügbarkeit
   ├── Terrain Modification: StatusIndicator für heightmap-Änderungen durch Erosion

   ├ Hydrology Statistics GroupBox:
   ├── Water Coverage: Prozent der Karte mit Wasserkörpern, aufgeteilt nach Biome-Types
   ├── Flow Statistics: Max/Mean Flow-Speed, Total Discharge, Longest River-Length
   ├── Lake Statistics: Lake-Count, Total Lake-Area, Largest Lake-Size
   ├── Erosion Impact: Total Eroded Volume, Max Erosion-Rate, Sedimentation Balance
   ├── Soil Moisture: Min/Max/Mean Bodenfeuchtigkeit, Moisture-Distribution
   ├── Performance Metrics: Jump-Flooding-Time, Flow-Iterations, Erosion-Cycles

   **Canvas Visualization Controls:**

   ├ Water-spezifische Controls (2D+3D):
   ├── Water Depth/Flow Speed/Erosion/Soil Moisture: Radio-Buttons (exklusiv für Water-Modi)
   ├── Water Biomes: Radio-Button (Creek/River/Grand River/Lake Classification-Display)

   ├ Standard BaseMapTab Controls (2D+3D):
   ├── Shadow: Checkbox (Overlay über alle Modi, nutzt Terrain-Shadowmap) + Shadow-Angle-Slider (0-6)
   ├── Contour Lines: Checkbox (nutzt Original- oder Modified-Heightmap je nach Modus)
   ├── Grid Overlay: Checkbox (für alle Modi)
   ├── Flow Vectors: Checkbox (2D-Arrows für Flow-Direction, 3D-Animated-Rivers)
   ├── Erosion Overlay: Checkbox (Red-Zones für aktive Erosion, Blue-Zones für Sedimentation)

   ═══════════════════════════════════════════════════════════════════════════════
   ## GENERATION-WORKFLOW
   ═══════════════════════════════════════════════════════════════════════════════

   **Generation-Flow:**
   ├ Multi-Dependency-Check → heightmap/hardness_map/precip_map/temp_map/wind_map-Validation → "Berechnen"-Button
   ├ generate_hydrology_system() → GenerationOrchestrator.request_generation("water")
   ├ HydrologySystemGenerator.generate_hydrology_system() → Lake-Detection + Flow-Network + Erosion-Simulation
   ├ LOD-Progression über GenerationOrchestrator mit steigender Hydrology-Komplexität
   ├ DataLODManager.set_water_data_lod() + set_terrain_data_lod(modified_heightmap) → Dual-Signal-Emission
   ├ Display-Update → Statistics-Update → Cross-Tab-Notification für Terrain-Modification

   **LOD-System Integration:**
   ├ LOD-Progression: Hydrology-Complexity skaliert automatisch bis map_size erreicht (entsprechend Terrain)
   ├ LOD-Grid-Scaling: 32x32→64x64→128x128→256x256→512x512→1024x1024→2048x2048
   ├ LOD-Level-Nummerierung: 1,2,3,4,5,6+ entsprechend verfügbaren Input-Dependencies-LODs
   ├ Progressive Enhancement: Jump-Flooding-Precision, Flow-Iterations und Erosion-Cycles steigen mit LOD
   ├ Performance-Scaling: Lake-Detection-Precision (3→5→7→10→15→20→25 Jump-Passes), Flow-Network-Detail steigt progressiv

   **Terrain-Feedback-Workflow:**
   ├ Original-Heightmap → Erosion-Calculation → Modified-Heightmap
   ├ DataLODManager.set_terrain_data_lod("heightmap_original") → Backup für andere Generatoren
   ├ DataLODManager.set_terrain_data_lod("heightmap_modified") → Updated für nachfolgende Systeme
   ├ Version-Control: Biome/Settlement-Generatoren können Original oder Modified wählen

   ═══════════════════════════════════════════════════════════════════════════════
   ## DEPENDENCY-MANAGEMENT
   ═══════════════════════════════════════════════════════════════════════════════

   **Input Dependencies:**
   ├ required_dependencies: ["heightmap", "hardness_map", "precip_map", "temp_map", "wind_map"]
   ├ Primary-Dependencies: terrain_tab → heightmap/slopemap
   ├ Secondary-Dependencies: geology_tab → hardness_map/rock_map
   ├ Tertiary-Dependencies: weather_tab → precip_map/temp_map/wind_map
   ├ check_input_dependencies(): Multi-Generator Cross-Validation mit Consistency-Checks

   **Input-Validation:**
   ├ heightmap Shape/Range/Hydrology-Suitability-Checks für Flow-Calculation
   ├ hardness_map Geological-Consistency-Validation mit rock_map für Erosion-Resistance
   ├ precip_map Precipitation-Range-Validation [0-500mm] für realistische Hydrology
   ├ temp_map/wind_map Atmospheric-Data-Quality für Evaporation-Calculation
   ├ Cross-Generator-Consistency: LOD-Level-Matching zwischen allen Input-Dependencies
   ├ Physical-Plausibility: Erosion-Parameters vs. Geological-Hardness Consistency-Checks

   **Complex-Dependency-Status-Display:**
   ├ Multi-Generator-StatusIndicator mit Individual-Dependency-Status
   ├ Data-Quality-Assessment pro Input-Dependency mit Recovery-Suggestions
   ├ Cross-Validation-Results für Input-Consistency zwischen Generatoren
   ├ LOD-Compatibility-Status zwischen verschiedenen Input-Sources

   ═══════════════════════════════════════════════════════════════════════════════
   ## MANAGER-INTEGRATION
   ═══════════════════════════════════════════════════════════════════════════════

   **Kommunikationskanäle:**
   ├ Config: value_default.WATER für Parameter-Ranges, Hydrology-Defaults, Erosion-Settings
   ├ Core: core/water_generator.py → HydrologySystemGenerator, LakeDetectionSystem, ErosionSedimentationSystem
   ├ Multi-Input: DataLODManager.get_terrain_data() + get_geology_data() + get_weather_data()
   ├ Dual-Output: DataLODManager.set_water_data_lod() + set_terrain_data_lod(modified) für Terrain-Feedback
   ├ Manager: GenerationOrchestrator.request_generation("water") nach Multi-Dependency-Erfüllung
   ├ Widgets: widgets.py für ParameterSlider, StatusIndicator, ProgressBar, RandomSeedButton

   **Signal-Flow:**
   ├ WaterTab.generate_requested("water")
   ├ GenerationOrchestrator.request_generation("water", parameters)
   ├ HydrologySystemGenerator.generate_hydrology_system()
   ├ DataLODManager.set_water_data_lod() + set_terrain_data_lod(modified_heightmap)
   ├ data_updated("water") + data_updated("terrain_modified") Signals
   ├ WaterTab.on_data_updated() + Cross-Tab-Updates für Terrain-Modification
   ├ WaterTab.update_display()

   ═══════════════════════════════════════════════════════════════════════════════
   ## DISPLAY-MODI (erweitert BaseMapTab)
   ═══════════════════════════════════════════════════════════════════════════════

   **Visualization Modes:**
   ├ Water Depth Mode: water_map mit Color-Coding (Transparent=0m → Dunkelblau=20m Tiefe)
   ├ Flow Speed Mode: flow_speed mit Velocity-Visualization (Blau=langsam → Rot=schnell)
   ├ Erosion Rate Mode: erosion_map mit Erosion-Intensity (Grün=Sedimentation → Rot=Erosion)
   ├ Soil Moisture Mode: soil_moist_map mit Feuchtigkeit-Gradienten (Gelb=trocken → Blau=feucht)
   ├ Water Biomes Mode: water_biomes_map mit Biome-Classification-Color-Coding

   **3D-Integration:**
   ├ Animated Rivers: 3D-Flow-Vectors mit Movement-Animation basierend auf flow_speed
   ├ Water-Surface-Rendering: Reflective Water-Surface über water_map-Areas
   ├ Erosion-Visualization: Real-time Height-Changes durch Erosion/Sedimentation
   ├ Terrain-Modification-Display: Original vs. Modified Heightmap Toggle-Option
   ├ Lake-3D-Rendering: Realistic Water-Bodies mit Depth-based Transparency

   **Overlay-System:**
   ├ Shadow/Contour/Grid Overlays: Nutzen Original- oder Modified-Terrain-Data je nach User-Choice
   ├ Flow-Vector-Overlay: 2D-Arrows für Flow-Direction, skaliert nach flow_map-Magnitude
   ├ Erosion-Overlay: Color-coded Erosion/Sedimentation-Zones über alle Modi
   ├ Multi-Layer-Rendering: Kombiniert Water-Data mit Original/Modified-Terrain-Base-Layer

   ═══════════════════════════════════════════════════════════════════════════════
   ## ERROR-HANDLING UND VALIDATION
   ═══════════════════════════════════════════════════════════════════════════════

   **Parameter-Validation:**
   ├ Range-Checks über value_default.WATER Constraints
   ├ Hydrology-Parameter [Lake-Threshold 0.1-10.0m, Manning 0.01-0.1] Physical-Validation
   ├ Erosion-Parameter [Erosion-Strength 0.0-5.0, Sediment-Capacity 0.0-1.0] Geological-Plausibility
   ├ Cross-Parameter-Constraints: Erosion-Strength vs. Hardness-Map-Values Consistency

   **Multi-Input-Validation:**
   ├ Heightmap Physical-Range-Checks für realistische Hydrology-Simulation
   ├ Hardness-Map vs. Rock-Map Consistency-Validation für Erosion-Resistance
   ├ Weather-Data Atmospheric-Plausibility für Evaporation-Calculation
   ├ Multi-Generator LOD-Level-Consistency zwischen allen Input-Dependencies

   **Hydrology-System-Validation:**
   ├ Water-Mass-Conservation-Checks für Input/Output-Balance
   ├ Flow-Network-Topology-Validation für physically-correct River-Systems
   ├ Erosion-Stability-Monitoring für Numerical-Stability bei Terrain-Modification
   ├ Lake-Detection-Consistency-Validation für Jump-Flooding-Algorithm-Results

   **Advanced-Error-Recovery:**
   ├ Multi-Input-Recovery mit Fallback-Strategies pro Dependency-Type
   ├ Erosion-Stability-Recovery bei Numerical-Instabilities mit Simplified-Physics
   ├ Flow-Network-Recovery bei Topology-Errors mit Alternative-Pathfinding
   ├ Terrain-Modification-Rollback bei Critical-Heightmap-Corruption

   ═══════════════════════════════════════════════════════════════════════════════
   ## ABHÄNGIGKEITEN UND OUTPUT
   ═══════════════════════════════════════════════════════════════════════════════

   **Dependencies:**
   ├ required_dependencies: ["heightmap", "hardness_map", "precip_map", "temp_map", "wind_map"]
   ├ Complex-Multi-Generator-Dependencies: terrain + geology + weather
   ├ Dependency-Validation erforderlich für alle Input-Sources vor Generation
   ├ Cross-Generator-Consistency-Checks für realistic Hydrology-Simulation

   **Output für nachgelagerte Generatoren:**
   ├ water_map → biome_generator für Water-Proximity Biome-Classification
   ├ soil_moist_map → biome_generator für Moisture-based Vegetation-Distribution
   ├ water_biomes_map → settlement_generator für Water-Access Settlement-Suitability
   ├ flow_map → settlement_generator für River-Transport-Route-Analysis
   ├ heightmap_modified → biome_generator/settlement_generator für Updated-Terrain-Base
   ├ erosion_map → settlement_generator für Geological-Stability-Assessment

   **Bidirectional-Terrain-Feedback:**
   ├ Original-Heightmap → Preserved für Generator-Compatibility
   ├ Modified-Heightmap → Available für Realistic-Post-Erosion-Simulation
   ├ Version-Control-System → Nachfolgende Generatoren können Original/Modified wählen
   ├ Terrain-Change-Notification → Cross-Tab-Updates für Terrain-Modification-Awareness
   """

def biome_tab():
   """
   Path: gui/tabs/biome_tab.py
   Date Changed: 24.08.2025

   ═══════════════════════════════════════════════════════════════════════════════
   ## ÜBERSICHT UND ZIELSETZUNG
   ═══════════════════════════════════════════════════════════════════════════════

   BiomeTab implementiert die Biome-Generator UI mit BaseMapTab-Integration und
   komplexer Ökosystem-Klassifikation basierend auf allen vorherigen Generator-Outputs.
   Erzeugt finale Biome-Verteilung durch 15 Base-Biome-Classification nach Whittaker-Diagramm
   mit 11 Super-Biome-Override-System und 2x2 Supersampling für weiche Übergänge.

   ═══════════════════════════════════════════════════════════════════════════════
   ## CORE-GENERATOR INTEGRATION
   ═══════════════════════════════════════════════════════════════════════════════

   **BiomeClassificationSystem Anbindung:**
   ├ Core-Call: BiomeClassificationSystem.classify_biomes(heightmap, temp_map, precip_map, soil_moist_map, water_biomes_map, parameters, lod_level)
   ├ Base-Biome-Classification: Gauß-basierte Klassifizierung von 15 Grundbiomen nach Klima/Höhe/Feuchtigkeit
   ├ Super-Biome-Override: 11 spezielle Biome (Ocean, Lake, River, Cliff, Beach, Alpine, etc.) überschreiben Base-Biome
   ├ Supersampling-System: 2x2 Supersampling mit diskretisierter Zufalls-Rotation für weiche Biome-Übergänge
   ├ Proximity-Biome-System: Ufer-Biome (Beach, Lake Edge, River Bank) mit konfigurierbarem edge_softness

   **BiomeData Output:**
   ├ biome_map: 2D numpy.uint8 array, Index der dominantesten Biom-Klasse
   ├ biome_map_super: 2D numpy.uint8 array, 2x supersampled zur Darstellung gemischter Biome
   ├ super_biome_mask: 2D numpy.bool array, Maske welche Pixel von Super-Biomes überschrieben wurden
   ├ biome_statistics: Dict mit Biome-Verteilungs-Prozenten und Ökosystem-Metriken
   ├ climate_classification: 2D numpy.uint8 array, Whittaker-Klimazone-Zuordnung

   ═══════════════════════════════════════════════════════════════════════════════
   ## UI-LAYOUT (erweitert BaseMapTab)
   ═══════════════════════════════════════════════════════════════════════════════

   **Control Panel Parameter-Sektion:**

   ├ Biome Classification Parameters GroupBox:
   ├── Temperature Factor: ParameterSlider (value_default.BIOME.BIOME_TEMP_FACTOR) + suffix:"weight"
   ├── Precipitation Factor: ParameterSlider (value_default.BIOME.BIOME_WETNESS_FACTOR) + suffix:"weight"
   ├── Elevation Factor: ParameterSlider (value_default.BIOME.ELEVATION_FACTOR) + suffix:"weight"
   ├── Soil Moisture Factor: ParameterSlider (value_default.BIOME.SOIL_MOISTURE_FACTOR) + suffix:"weight"

   ├ Super-Biome Thresholds GroupBox:
   ├── Sea Level: ParameterSlider (value_default.BIOME.SEA_LEVEL) + suffix:"m"
   ├── Alpine Level: ParameterSlider (value_default.BIOME.ALPINE_LEVEL) + suffix:"m"
   ├── Snow Level: ParameterSlider (value_default.BIOME.SNOW_LEVEL) + suffix:"m"
   ├── Cliff Slope: ParameterSlider (value_default.BIOME.CLIFF_SLOPE) + suffix:"degrees"

   ├ Transition Quality Parameters GroupBox:
   ├── Edge Softness: ParameterSlider (value_default.BIOME.EDGE_SOFTNESS) + suffix:"factor"
   ├── Bank Width: ParameterSlider (value_default.BIOME.BANK_WIDTH) + suffix:"pixel"
   ├── Supersampling Quality: ParameterSlider (value_default.BIOME.SUPERSAMPLING_QUALITY) + suffix:"level"
   ├── Biome Seed: ParameterSlider + RandomSeedButton (widgets.py)

   ├ Generation Control GroupBox:
   ├── "Berechnen"-Button: Manual Generation-Trigger
   ├── Generation Progress: ProgressBar mit Biome-Phase-Details (Base-Classification→Super-Override→Supersampling→Statistics)
   ├── Input Dependencies: StatusIndicator für heightmap/temp_map/precip_map/soil_moist_map/water_biomes_map-Verfügbarkeit
   ├── Multi-Generator-Status: StatusIndicator für terrain+geology+weather+water Dependency-Chain

   ├ Biome Distribution Statistics GroupBox:
   ├── Base-Biome-Distribution: Progress-Bars für alle 15 Base-Biome mit Prozent-Anteilen
   ├── Super-Biome-Coverage: Ocean/Lake/River/Alpine Coverage-Statistiken
   ├── Climate-Zone-Distribution: Whittaker-Klimazone-Prozente (Arctic, Temperate, Tropical, etc.)
   ├── Diversity-Metrics: Shannon-Diversity-Index, Biome-Richness, Transition-Smoothness
   ├── Quality-Metrics: Supersampling-Effectiveness, Edge-Transition-Quality
   ├── Performance-Stats: Classification-Time, Supersampling-Time, Memory-Usage

   **Canvas Visualization Controls:**

   ├ Biome-spezifische Controls (2D+3D):
   ├── Base Biomes/Super Biomes/Mixed View: Radio-Buttons (exklusiv für Biome-Display-Modi)
   ├── Climate Zones: Radio-Button (Whittaker-Klimazone-Display)
   ├── Biome Transitions: Checkbox (Supersampled vs. Discrete Biome-Boundaries)

   ├ Standard BaseMapTab Controls (2D+3D):
   ├── Shadow: Checkbox (Overlay über alle Modi, nutzt Terrain-Shadowmap) + Shadow-Angle-Slider (0-6)
   ├── Contour Lines: Checkbox (nutzt Original/Modified-Heightmap je nach Water-Tab-Status)
   ├── Grid Overlay: Checkbox (für alle Modi)
   ├── Water Overlay: Checkbox (Water-Biomes-Overlay über alle Biome-Modi)
   ├── Settlement Overlay: Checkbox (Settlement-Points-Overlay für Ökosystem-Human-Interaction)

   ═══════════════════════════════════════════════════════════════════════════════
   ## GENERATION-WORKFLOW
   ═══════════════════════════════════════════════════════════════════════════════

   **Generation-Flow:**
   ├ Complex-Multi-Dependency-Check → heightmap+temp_map+precip_map+soil_moist_map+water_biomes_map-Validation → "Berechnen"-Button
   ├ generate_biome_system() → GenerationOrchestrator.request_generation("biome")
   ├ BiomeClassificationSystem.classify_biomes() → Base-Classification + Super-Override + Supersampling + Statistics
   ├ LOD-Progression über GenerationOrchestrator mit steigender Classification-Precision und Supersampling-Quality
   ├ DataLODManager.set_biome_data_lod() → Signal-Emission
   ├ Display-Update → Statistics-Update → Cross-Tab-Notification für Final-World-Assembly

   **LOD-System Integration:**
   ├ LOD-Progression: Biome-Classification-Precision skaliert automatisch bis map_size erreicht (entsprechend allen Input-Dependencies)
   ├ LOD-Grid-Scaling: 32x32→64x64→128x128→256x256→512x512→1024x1024→2048x2048
   ├ LOD-Level-Nummerierung: 1,2,3,4,5,6+ entsprechend verfügbaren Multi-Generator-Input-LODs
   ├ Progressive Enhancement: Classification-Accuracy, Super-Biome-Detail und Supersampling-Quality steigen mit LOD
   ├ Performance-Scaling: Gauß-Fitting-Precision (5→10→15→20→25→30 Samples), Super-Biome-Calculation-Detail steigt progressiv

   **Final-Integration-Workflow:**
   ├ Multi-Generator-Data-Assembly → Climate+Geological+Hydrological+Terrain-Integration
   ├ Ecosystem-Coherence-Validation → Cross-System-Consistency-Checks
   ├ Biome-Transition-Optimization → Natural-Boundary-Enhancement durch Supersampling
   ├ Final-World-State-Preparation → Export-Ready Biome-Map für Overview-Tab

   ═══════════════════════════════════════════════════════════════════════════════
   ## DEPENDENCY-MANAGEMENT
   ═══════════════════════════════════════════════════════════════════════════════

   **Input Dependencies (komplexeste aller Generatoren):**
   ├ required_dependencies: ["heightmap", "temp_map", "precip_map", "soil_moist_map", "water_biomes_map"]
   ├ Primary-Dependencies: terrain_tab → heightmap für Elevation-based Biome-Classification
   ├ Secondary-Dependencies: weather_tab → temp_map/precip_map für Whittaker-Climate-Classification
   ├ Tertiary-Dependencies: water_tab → soil_moist_map/water_biomes_map für Moisture/Proximity-based Biome-Modulation
   ├ Optional-Dependencies: geology_tab → rock_map für Geological-Biome-Influence (future enhancement)
   ├ check_input_dependencies(): Most-Complex Multi-Generator Cross-Validation mit Climate-Geological-Hydrological-Consistency

   **Input-Validation:**
   ├ heightmap Elevation-Range-Validation für Alpine/Snow-Level-Classification
   ├ temp_map/precip_map Climate-Data-Quality für Whittaker-Biome-Matrix-Classification
   ├ soil_moist_map Moisture-Range-Validation [0-100%] für Wetland/Desert-Classification
   ├ water_biomes_map Water-Body-Classification-Consistency für Proximity-Super-Biomes
   ├ Cross-System-Physical-Plausibility: Temperature vs. Precipitation vs. Elevation Consistency
   ├ Multi-Generator-LOD-Consistency: Matching LOD-Levels zwischen allen 4 Input-Sources

   **Complex-Multi-Generator-Status-Display:**
   ├ 4-Generator-Dependencies-StatusIndicator mit Individual-Input-Quality-Assessment
   ├ Climate-Data-Quality-Matrix mit Temperature/Precipitation-Data-Validation
   ├ Hydrological-Data-Consistency mit Water-System-Integration-Status
   ├ Cross-System-Validation-Results für Physical-Plausibility zwischen allen Input-Systems

   ═══════════════════════════════════════════════════════════════════════════════
   ## MANAGER-INTEGRATION
   ═══════════════════════════════════════════════════════════════════════════════

   **Kommunikationskanäle:**
   ├ Config: value_default.BIOME für Parameter-Ranges, Classification-Defaults, Supersampling-Settings
   ├ Core: core/biome_generator.py → BiomeClassificationSystem, BaseBiomeClassifier, SuperBiomeOverrideSystem, SupersamplingManager
   ├ Multi-Input: DataLODManager.get_terrain_data() + get_weather_data() + get_water_data() für Complete-Ecosystem-Data
   ├ Final-Output: DataLODManager.set_biome_data_lod() für Final-World-State
   ├ Manager: GenerationOrchestrator.request_generation("biome") nach All-Dependencies-Erfüllung
   ├ Widgets: widgets.py für ParameterSlider, StatusIndicator, ProgressBar, RandomSeedButton

   **Signal-Flow:**
   ├ BiomeTab.generate_requested("biome")
   ├ GenerationOrchestrator.request_generation("biome", parameters)
   ├ BiomeClassificationSystem.classify_biomes()
   ├ DataLODManager.set_biome_data_lod()
   ├ data_updated("biome") Signal
   ├ BiomeTab.on_data_updated() + Overview-Tab-Notification für Final-World-Assembly
   ├ BiomeTab.update_display()

   ═══════════════════════════════════════════════════════════════════════════════
   ## DISPLAY-MODI (erweitert BaseMapTab)
   ═══════════════════════════════════════════════════════════════════════════════

   **Visualization Modes:**
   ├ Base Biomes Mode: biome_map mit 15-Color-Coded Base-Biomes (Ice Cap, Tundra, Taiga, Desert, Rainforest, etc.)
   ├ Super Biomes Mode: super_biome_mask mit 11-Color-Coded Override-Biomes (Ocean, Lake, River, Cliff, Beach, Alpine, etc.)
   ├ Mixed View Mode: biome_map_super mit Supersampled-Blending zwischen Base- und Super-Biomes
   ├ Climate Zones Mode: climate_classification mit Whittaker-Climate-Zone-Display (Arctic, Boreal, Temperate, Tropical, Arid)
   ├ Biome Transitions Mode: Edge-Softness-Visualization mit Transition-Quality-Assessment

   **3D-Integration:**
   ├ 3D-Biome-Texturing: Realistic-Biome-Textures auf 3D-Terrain-Mesh basierend auf biome_map
   ├ Vegetation-Height-Simulation: Biome-appropriate Vegetation-Height-Modulation (Forest=tall, Desert=low, etc.)
   ├ Ecosystem-3D-Visualization: Biome-specific 3D-Elements (Trees, Grass, Rocks, Snow) als Procedural-Details
   ├ Climate-3D-Effects: Temperature/Precipitation-based Visual-Effects (Snow, Rain, Mist) über 3D-Scene
   ├ Supersampling-3D-Blending: Smooth-Biome-Transitions in 3D-Rendering durch biome_map_super

   **Overlay-System:**
   ├ Shadow/Contour/Grid Overlays: Nutzen Modified-Terrain-Data von Water-Tab für Post-Erosion-Display
   ├ Water-System-Overlay: water_biomes_map-Integration für Water-Biome-Interaction-Display
   ├ Settlement-Overlay: Settlement-Points für Human-Ecosystem-Interaction-Visualization
   ├ Multi-Layer-Ecosystem-Rendering: Kombiniert Biome-Data mit allen anderen Generator-Outputs

   ═══════════════════════════════════════════════════════════════════════════════
   ## ERROR-HANDLING UND VALIDATION
   ═══════════════════════════════════════════════════════════════════════════════

   **Parameter-Validation:**
   ├ Range-Checks über value_default.BIOME Constraints
   ├ Classification-Factor-Parameter [0.0-3.0] für Temperature/Precipitation/Elevation-Weights
   ├ Elevation-Threshold-Parameter [0-8000m] für Alpine/Snow-Level Physical-Plausibility
   ├ Transition-Quality-Parameter [0.1-2.0] für Edge-Softness und Supersampling-Settings

   **Multi-Input-Ecosystem-Validation:**
   ├ Climate-Data Physical-Range-Validation für realistische Biome-Classification
   ├ Elevation vs. Temperature Consistency-Checks für Alpine-Biome-Logic
   ├ Precipitation vs. Soil-Moisture Hydrological-Consistency für Wetland/Desert-Classification
   ├ Water-System vs. Proximity-Biome Consistency für Beach/Lake-Edge/River-Bank-Logic

   **Biome-Classification-Validation:**
   ├ Whittaker-Matrix-Consistency-Checks für Climate-Zone-Biome-Mapping
   ├ Super-Biome-Override-Logic-Validation für Priority-System-Consistency
   ├ Supersampling-Quality-Assessment für Transition-Smoothness-Metrics
   ├ Ecosystem-Coherence-Validation für Cross-Biome-Boundary-Natural-Transitions

   **Advanced-Ecosystem-Error-Recovery:**
   ├ Multi-Input-Recovery mit Intelligent-Fallback pro Climate/Geological/Hydrological-Input
   ├ Classification-Failure-Recovery mit Simplified-Biome-Mapping bei Complex-System-Failures
   ├ Supersampling-Recovery mit Quality-Degradation bei Memory/Performance-Constraints
   ├ Ecosystem-Consistency-Recovery mit Default-Biome-Distributions bei Cross-System-Conflicts

   ═══════════════════════════════════════════════════════════════════════════════
   ## ABHÄNGIGKEITEN UND OUTPUT
   ═══════════════════════════════════════════════════════════════════════════════

   **Dependencies (komplexeste aller Generatoren):**
   ├ required_dependencies: ["heightmap", "temp_map", "precip_map", "soil_moist_map", "water_biomes_map"]
   ├ Complete-Multi-Generator-Dependencies: terrain + weather + water (+ optional geology)
   ├ Dependency-Validation erforderlich für alle 4-5 Input-Sources vor Final-Generation
   ├ Cross-System-Ecosystem-Consistency-Checks für realistic Final-World-Assembly

   **Output für nachgelagerte Systeme:**
   ├ biome_map → settlement_generator für Biome-Suitability Settlement-Placement
   ├ biome_map_super → overview_tab für High-Quality Final-World-Rendering
   ├ climate_classification → settlement_generator für Climate-Zone-based Settlement-Types
   ├ biome_statistics → overview_tab für World-Summary und Export-Metadata
   ├ Final-Ecosystem-State → overview_tab für Complete-World-Export-Assembly

   **Final-World-Integration:**
   ├ Complete-Ecosystem-Assembly → Integration aller Generator-Outputs in coherent World-State
   ├ Export-Ready-Data-Preparation → High-Quality-Biome-Maps für verschiedene Export-Formate
   ├ World-Coherence-Validation → Final-Cross-System-Consistency für realistic World-Simulation
   ├ Overview-Tab-Integration → Finale World-Visualization und Multi-Format-Export-Preparation
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
    date_changed: 24.08.2025

    Funktionsweise: Erweiterte wiederverwendbare UI-Komponenten für Manual-Only System
    - Eliminiert Auto-Simulation Components (AutoSimulationPanel, ParameterUpdateManager)
    - Standard-Widgets mit verbesserter User-Experience (No-Wheel-Events)
    - Fokus auf wiederverwendbare Core-Components für alle Tabs

    Kern-Widgets (aktiv verwendet):
    - BaseButton mit konfigurierbarem Styling für alle Tabs
    - ParameterSlider mit value_default.py Integration und No-Wheel-Events (Focus-Required)
    - RandomSeedButton für alle Generator Seed-Parameter
    - StatusIndicator für Input-Dependencies und Generation-Status
    - ProgressBar für LOD-Progression Display
    - QComboBox mit No-Wheel-Events (Focus-Required) für Dropdown-Selections

    Spezielle Widgets:
    - BiomeLegendDialog für alle 26 Biome-Typen Anzeige (15 Base + 11 Super)
    - DisplayWrapper für einheitliche 2D/3D Display-API mit Fallback-Handling
    - DependencyResolver für robuste Cross-Tab Dependency-Resolution

    UI-Enhancement Features:
    - Alle Parameter-Controls blockieren Mausrad-Events außer bei Focus
    - Verhindert ungewollte Parameter-Änderungen beim Scrollen
    - Einheitliche Styling-API über gui_default.py
    - Memory-Management und Thread-Safety Verbesserungen

    Kommunikationskanäle:
    - Config: gui_default.py für Styling, value_default.py für Parameter-Ranges
    - Signals: Standard Qt-Signals für Manual-Generation-Requests
    - Manager-Integration: Direkte ParameterManager Kommunikation ohne Auto-Trigger
    - Thread-Safety: QMutex-Protection für Background-Generation-Coordination
    """