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

    Funktionsweise: Geologische Schichten und Gesteinstypen
    - Rock-Type Klassifizierung (sedimentary, metamorphic, igneous) basierend auf geologischer Simulation mit folgenden Techniken:
        - die höhen werden genommen und dann mit einer neuen verzerrten noisefunktion multipliziert, so dass die höhen auch variieren für eine erste verteilung (weiche übergänge)
        - in besonders steile hänge werden härtere steine mit reingemischt.
        - geologische zonen werden in bestimmte zonen addiert und subtrahiert. d.h. je eine simplex funktion von 0 bis 1 bei der nur werte über 0.2 für sedimentär, eine andere simplex über 0.6 für metamorph und eine dritte simplex mit über 0.8 für igneous.
        - alle drei maps werden zu einer RGB-Karte gewichtet addiert und die Summe der einzelnen R, G und B ergeben zusammen IMMER 255, so dass die Massenerhaltung bestehen bleibt, aber die Verhältnisse sich jeweils ändern können (später bei Erosion und Sedimentation durch Wasser (water_generator.py)).
        - aus den Gesteinstypen wird entsprechend dem Eingabeparameter rock_type"_hardness" (mit sedimentary, igneous, metamorphic für rock_type) eine hardness_map(x,y) erstellt mit den jeweiligen Verhältnissen aus rock_map

    Parameter Input:
    - rock_types, hardness_values, ridge_warping, bevel_warping, metamorph_foliation, metamorph_folding, igneous_flowing

    data_manager Input:
    - heightmap

    Output:
    - rock_map RGB array
    - hardness_map array

    Klassen:
    GeologyGenerator
        Funktionsweise: Hauptklasse für geologische Schichten und Gesteinstyp-Verteilung
        Aufgabe: Koordiniert Gesteinsverteilung und Härte-Berechnung
        Methoden: generate_rock_distribution(), calculate_hardness_map(), apply_geological_zones()

    RockTypeClassifier
        Funktionsweise: Klassifiziert Gesteinstypen (sedimentary, metamorphic, igneous) basierend auf Höhe und Steigung
        Aufgabe: Erstellt rock_map mit RGB-Kanälen für drei Gesteinstypen
        Methoden: classify_by_elevation(), apply_slope_hardening(), blend_geological_zones()

    MassConservationManager
        Funktionsweise: Stellt sicher dass R+G+B immer 255 ergibt für Massenerhaltung
        Aufgabe: Verwaltet Gesteins-Massenverteilung für spätere Erosion
        Methoden: normalize_rock_masses(), validate_conservation(), redistribute_masses()
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
        SIZE = {"min": 64, "max": 512, "default": 256, "step": 32}
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

    Funktionsweise: Geologie-Editor mit 3D Textured Terrain
    - Input: Heightmap von terrain_tab
    - Core-Integration: geology_generator.py
    - 3D-Rendering mit Gesteinstyp-Texturen
    - Parameter: Rock-Types, Hardness-Values, Tektonische Deformation

    Kommunikationskanäle:
    - Input: heightmap von data_manager
    - Core: geology_generator für Rock-Classification
    - Output: rock_map, hardness_map → data_manager
    - Config: value_default.GEOLOGY
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