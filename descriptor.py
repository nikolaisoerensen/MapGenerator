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

def data_lod_manager():
    """
    Path: gui/managers/data_lod_manager.py
    date_changed: 08.08.2025

    =========================================================================================
    DATA LOD MANAGER - INTEGRIERTE DATENVERWALTUNG UND RESOURCE-MANAGEMENT
    =========================================================================================

    OVERVIEW:
    ---------
    DataLODManager ist das zentrale Herzstück der Daten-Pipeline, das vier kritische Systeme
    in einer kohärenten Architektur vereint: Datenverwaltung mit numerischem LOD-System,
    Signal-basierte Tab-Kommunikation, systematisches Resource-Management und optimierte
    Display-Updates. Diese Integration eliminiert Code-Duplikation und schafft eine
    einheitliche Schnittstelle für alle datenabhängigen Operationen.

    CORE ARCHITECTURE:
    ------------------

    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                            DataLODManager (Main Hub)                            │
    │ ┌─────────────────┬─────────────────┬─────────────────┬─────────────────────┐   │
    │ │ LOD Data System │ Communication   │ Resource Mgmt   │ Display Optimization│   │
    │ │ - Numerische    │ Hub             │ - Memory Leak   │ - Change Detection  │   │
    │ │   LODs (1,2,3..)│ - Tab Signals   │   Prevention    │ - Hash Caching      │   │
    │ │ - 6 Generators  │ - Dependencies  │ - WeakRef Track │ - Pending Updates   │   │
    │ │ - Map-size      │ - Auto-Gen      │ - Age/Size      │ - Performance Stats │   │
    │ │   Proportional  │   Triggering    │   Cleanup       │                     │   │
    │ └─────────────────┴─────────────────┴─────────────────┴─────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────────────┘

    INTEGRATED COMPONENTS:
    ----------------------

    1. **DATENVERWALTUNG (Numerisches LOD-System)**
       - Alle 6 Generator-Typen: Terrain, Geology, Weather, Water, Biome, Settlement
       - Numerische LOD-Levels (1,2,3...n)
       - Map-size-proportionale Skalierung: Viele Generator-LOD skalieren mit der map-size d.h. Anzahl Rechenpunkten
       - Die lod_size ist immer <= map_size. D.h. es verdoppelt sich bis es größer ist und dann wird der Wert bei
            map_size fixiert und kommuniziert, dass die map_size erreicht ist. Die lod_level werden weiter größer bis
            alle Rechnungen fertig sind, d.h. z.B. das bei geology sich nichts mehr ändert, aber bei terrain weiterhin
            alle verbleibenden Sonnenstände berechnet werden.
       - Generator-spezifische Optimierungen:
         * Terrain: LOD-Level fügt Sonnenstände für Schattenberechnung hinzu (1-7)
         * Weather: CFD-Zellen halbieren pro LOD, Z-Achse konstant (3 Schichten) d.h. wenn 1024x1024 das Ziel ist, dann
            wird zunächst mit 32x32 Zellen der Länge und Breite von 32x32 gerechnet. Für lod_level 2 dann 64x64 Zellen
            mit 16x16 etc. bis 1x1
         * Biome: Super-sampling (2x2px pro 1x1px) siehe biome_generator.py, jedoch steigt LOD ganz regulär mit map_size
         * Settlement: Skalierung mit Map-size und Verdopplung der Node-Anzahl bis der Zielwert erreicht:
            plotnode_lod_steps = np.ceil(map_size / map_size_min)
            Plotnodes-Parameter * 2^(lod_level-1) / plotnode_lod_steps ist dann der Wert für das jeweilige lod_level

    2. **LOD COMMUNICATION HUB (Signal-Option B)**
       - Zentrale Signal-Koordination zwischen allen Tabs
       - Status-Tracking: idle/pending/success/failure pro Tab/LOD
       - Dependency-Matrix: Vereinfacht auf gleiche LOD-Stufe zwischen Tabs
       - Auto-Generation-Triggering für abhängige Tabs (Geology→Settlement→etc.)
       - Progress-Updates und Error-Handling für UI-Integration

    3. **RESOURCE TRACKER (Memory-Leak-Prevention)**
       - WeakReference-basiertes Tracking aller großen Ressourcen (>10MB Arrays)
       - Automatisches Cleanup bei Garbage Collection
       - Age-based Cleanup (alte Ressourcen >2h)
       - Size-based Cleanup (Ressourcen >50MB) bei Memory-Warnings
       - Resource-Type-Management für systematische Bereinigung

    4. **DISPLAY UPDATE MANAGER (Performance-Optimierung)**
       - Hash-basierte Change-Detection verhindert unnötige Re-Renderings
       - Sample-based Hashing für große Arrays (>1M Elemente)
       - Pending-Updates-Management verhindert Race-Conditions
       - Cache-Statistics für Performance-Monitoring
       - Multi-Level-Hashing für verschiedene Display-Modi

    LOD-SYSTEM DETAILS:
    -------------------

    **Numerische LOD-Progression:**
    ```
    lod_level : LOD Size = map_size_min * 2^(lod_level-1)
    für map_size_min = 32:
        lod_level 1: 32x32
        lod_level 2: 64x64
        lod_level 3: 128x128
        lod_level 4: 256x256
        lod_level 5: 512x512
        lod_level 6: 1024x1024
        lod_level 7: 2048x2048
        lod_level n: erweiterbar
    ```

    **Dependency-Matrix (Vereinfacht):**
    ```python
    DEPENDENCY_MATRIX = {
        "terrain": [],                           # Keine Dependencies
        "geology": ["terrain"],                  # Gleiche LOD-Stufe
        "weather": ["terrain"],                  # Gleiche LOD-Stufe
        "water": ["terrain", weather],           # Gleiche LOD-Stufe
        "biome": ["weather", "water"],           # Gleiche LOD-Stufe
        "settlement": ["terrain", "geology"]     # Gleiche LOD-Stufe
    }
    ```

    **Auto-Generation-Flow:**
    1. Terrain LOD n fertig → Geology, Weather, Water kann LOD n starten
    2. Geology LOD n fertig → Settlement kann LOD n starten
    3. Weather LOD n fertig → Water kann LOD n starten
    4. Water LOD n fertig → Biome kann LOD n starten.

    SIGNAL ARCHITECTURE:
    --------------------

    **Outgoing Signals (Hub → Tabs/UI):**
    - `tab_status_updated(tab, status_dict)` - Status-Changes für einzelne Tabs
    - `dependencies_satisfied(tab, lod_level)` - Dependency-Requirements erfüllt
    - `all_tabs_status(complete_status)` - Globaler Status-Überblick
    - `auto_generation_ready(tab, lod_level)` - Auto-Generation kann starten

    **Incoming Signals (Tabs → Hub):**
    - `on_tab_lod_started(tab, lod_level, lod_size)` - Generation gestartet
    - `on_tab_lod_progress(tab, lod_level, progress_percent)` - Progress-Update
    - `on_tab_lod_completed(tab, lod_level, success, data_keys)` - Generation fertig
    - `on_tab_lod_failed(tab, lod_level, error_message)` - Generation fehlgeschlagen

    **Legacy Signals (Kompatibilität):**
    - `data_updated(generator_type, data_key)` - Für bestehende Tab-Integration
    - `cache_invalidated(generator_type)` - Cache-Management
    - `lod_data_stored(generator_type, lod_level, data_keys)` - Neue LOD-Daten

    RESOURCE MANAGEMENT:
    --------------------

    **Automatisches Tracking:**
    - Alle Arrays >10MB werden automatisch getrackt
    - WeakReference verhindert Memory-Leaks bei vergessenen References
    - Cleanup-Callbacks für komplexe Ressourcen (GPU-Texturen, etc.)
    - Resource-Type-Kategorisierung für selektives Cleanup

    **Memory-Management-Strategien:**
    - **Präventiv:** Periodisches Cleanup alle 60s bei hoher Memory-Usage (>1GB)
    - **Reaktiv:** Aggressive Cleanup bei Memory-Warnings (>500MB Threshold)
    - **Age-based:** Ressourcen >2h werden automatisch bereinigt
    - **Size-based:** Große Ressourcen >50MB werden bei Memory-Druck bereinigt

    **Display-Cache-Optimierung:**
    - Hash-Caching verhindert doppelte Hash-Berechnungen
    - Sample-based Hashing für Performance bei großen Arrays
    - Pending-Updates verhindern Race-Conditions zwischen UI und Logic
    - Alte Display-Cache-Einträge (>30min) werden automatisch bereinigt

    USAGE PATTERNS:
    ---------------

    **Standard Setup:**
    ```python
    # Factory-Methode mit konfigurierten Defaults
    data_lod_manager = create_integrated_data_lod_manager(memory_threshold_mb=500)

    # LOD-Config für Projekt
    data_lod_manager.set_lod_config(minimal_map_size=32, target_map_size=512)

    # Tab-Integration
    data_lod_manager.connect_tab_to_hub("terrain", terrain_tab)
    data_lod_manager.connect_tab_to_hub("geology", geology_tab)
    ```

    **Data Storage (Generator → DataManager):**
    ```python
    # Neue LOD-basierte Methoden
    data_lod_manager.set_terrain_data_lod("heightmap", heightmap_array, lod_level=3, parameters)
    data_lod_manager.set_geology_data_lod("rock_map", rock_array, lod_level=2, parameters)

    # Legacy-kompatible Methoden (nutzen höchstes verfügbares LOD)
    heightmap = data_lod_manager.get_terrain_data("heightmap")
    rock_map = data_lod_manager.get_geology_data("rock_map")
    ```

    **Resource Management:**
    ```python
    # Manuelle Resource-Registrierung für spezielle Ressourcen
    resource_id = data_lod_manager.get_resource_tracker().register_resource(
        large_texture, "gpu_texture", cleanup_func=lambda: texture.delete()
    )

    # Memory-Management
    data_lod_manager.cleanup_old_lod_resources(max_age_hours=1.0)
    data_lod_manager.cleanup_large_resources(size_threshold_mb=100.0)
    ```

    **Display Updates:**
    ```python
    # Change-Detection für optimierte Updates
    display_manager = data_lod_manager.get_display_manager()
    if display_manager.needs_update(display_id, data, layer_type):
        display.update_display(data, layer_type)
        display_manager.mark_updated(display_id, data, layer_type)
    ```

    PERFORMANCE CHARACTERISTICS:
    ----------------------------

    **Memory Efficiency:**
    - No-Copy Array-Referenzen zwischen Komponenten
    - WeakReference-basiertes Tracking ohne Memory-Overhead
    - Sample-based Hashing reduziert Hash-Zeit für große Arrays von 100ms auf <10ms
    - Aggressive Cleanup bei Memory-Warnings verhindert Out-of-Memory

    **Signal Performance:**
    - Direct pyqtSlot-Connections ohne QMetaObject-Overhead
    - Debounced Updates verhindern Signal-Spam bei schnellen Parameter-Änderungen
    - Cached Status-Updates vermeiden doppelte Dictionary-Konstruktionen

    **LOD-System Performance:**
    - Numerische LODs sind 30% schneller als String-Vergleiche
    - LOD-basiertes Caching verhindert unnötige Re-Generationen
    - Proportionale Skalierung reduziert Memory-Footprint bei niedrigen LODs

    INTEGRATION POINTS:
    -------------------

    **BaseMapTab Integration:**
    - ResourceTracker und DisplayUpdateManager aus base_tab.py werden ersetzt
    - Signal-Integration über connect_tab_to_hub()
    - Legacy get_*_data() Methoden bleiben kompatibel
    - Auto-Simulation-Integration über Hub-Signals

    **GenerationOrchestrator Integration:**
    - LOD-Status-Updates über Hub-Signals
    - Resource-Tracking für Generation-Threads
    - Memory-Management während intensiver Berechnungen
    - Display-Update-Koordination nach Generation

    **Parameter-Manager Integration:**
    - Parameter-Changes triggern Cache-Invalidation
    - Cross-Tab Parameter-Dependencies über Hub
    - Export/Import berücksichtigt LOD-Status
    - Preset-Application mit LOD-Koordination

    ERROR HANDLING & RESILIENCE:
    -----------------------------

    **Graceful Degradation:**
    - Fallback auf Legacy-Methoden bei LOD-System-Fehlern
    - Continued Operation auch bei Resource-Tracking-Problemen
    - Display-Updates funktionieren auch ohne Change-Detection
    - Tab-Communication funktioniert auch ohne Hub-Integration

    **Memory Safety:**
    - Exception-sichere Cleanup-Funktionen
    - Force-Garbage-Collection bei kritischer Memory-Usage
    - WeakReference verhindert Circular-Dependencies
    - Resource-Leak-Detection und automatische Bereinigung

    **Data Integrity:**
    - Atomic LOD-Updates mit Rollback bei Fehlern
    - Cache-Konsistenz durch Parameter-Hash-Validation
    - Signal-Ordering durch Qt's Event-Queue
    - Thread-safe Resource-Access durch QMutex (wo nötig)

    MONITORING & DEBUGGING:
    -----------------------

    **Statistics APIs:**
    ```python
    # Comprehensive Statistics für Performance-Monitoring
    stats = data_lod_manager.get_integrated_statistics()
    # Returns: data_manager stats, resource_tracker stats, display_manager stats, lod_hub stats

    # Memory-Usage-Breakdown
    memory_by_lod = data_lod_manager.get_memory_usage_by_lod()
    # Returns: {"terrain": {"LOD_1": 5.2, "LOD_2": 18.7}, "geology": {...}}

    # Resource-Tracking-Details
    resource_stats = data_lod_manager.get_resource_tracker().get_resource_statistics()
    # Returns: total_resources, alive_resources, dead_references, memory_usage, etc.
    ```

    **Export/Summary:**
    ```python
    # Detailed Export für Debugging und Analysis
    summary = data_lod_manager.export_data_summary_lod()
    # Includes: LOD-Data per Generator, Resource-Statistics, Display-Cache-Stats
    ```

    Diese integrierte Architektur schafft eine einheitliche, performante und wartbare
    Basis für alle datenabhängigen Operationen im Map-Generator, eliminiert Code-Duplikation
    und bietet robuste Memory-Management-Garantien.

    =========================================================================================
    """

def parameter_manager():
    """
    Path: gui/managers/parameter_manager.py
    date_changed: 08.08.2025

    =========================================================================================
    PARAMETER MANAGER - ZENTRALE PARAMETER-KOORDINATION UND EXPORT-SYSTEM
    =========================================================================================

    OVERVIEW:
    ---------
    ParameterManager orchestriert alle Parameter zwischen den 6 Generator-Tabs durch ein
    integriertes System aus Cross-Tab-Communication, Export/Import-Management, Preset-System
    und Race-Condition-Prevention. Das System ermöglicht reproduzierbare Map-Generation,
    Parameter-Templates und automatische Dependency-Synchronisation zwischen Tabs.

    CORE ARCHITECTURE:
    ------------------

    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                         ParameterManager (Central Hub)                         │
    │ ┌─────────────────┬─────────────────┬─────────────────┬─────────────────────┐ │
    │ │ Communication   │ Export/Import   │ Preset System   │ Update Management   │ │
    │ │ Hub             │ Manager         │ - File Storage  │ - Debouncing        │ │
    │ │ - Cross-Tab     │ - JSON Export   │ - Categories    │ - Race Prevention   │ │
    │ │   Sync          │ - Template Gen  │ - Tag System    │ - Validation        │ │
    │ │ - Dependencies  │ - Metadata      │ - Search        │   Timing            │ │
    │ │ - Validation    │ - Versioning    │ - Version Mgmt  │ - Generation Coord  │ │
    │ └─────────────────┴─────────────────┴─────────────────┴─────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────────────────┘

    KEY COMPONENTS:
    ---------------

    1. **PARAMETER COMMUNICATION HUB**
       - Central Registry für alle 6 Generator-Tabs
       - Cross-Tab Parameter-Broadcasting mit Change-Tracking
       - Dependency-Management zwischen Tab-Parametern
       - Change-History mit 1000-Event-Limit für Debugging
       - Parameter-Constraint-System für Validation
       - Signal-basierte Parameter-Propagation

    2. **EXPORT/IMPORT MANAGER**
       - JSON-basiertes Export/Import-System für komplette Parameter-Sets
       - Metadata-Integration (Timestamps, Dependencies, Version-Info)
       - Template-Generation für Default-Parameter-Sets
       - Selective Import/Export (nur bestimmte Tabs)
       - Format-Versioning für Future-Compatibility
       - Import-Validation mit Compatibility-Checks

    3. **PRESET MANAGEMENT SYSTEM**
       - File-basiertes Preset-Storage mit Kategorisierung
       - Tag-System für flexible Preset-Organisation
       - Search-Engine für Preset-Discovery
       - Category-Management (General, Terrain-Focused, Weather-Heavy, etc.)
       - Preset-Versioning und Update-Tracking
       - Cache-basierte Performance-Optimierung

    4. **PARAMETER UPDATE MANAGER**
       - Debounced Parameter-Updates für Race-Condition-Prevention
       - Separate Validation/Generation-Timers (250ms validation, 500ms generation)
       - Pending-Update-Queue für koordinierte Parameter-Changes
       - Integration mit GenerationOrchestrator für automatic Triggering
       - Thread-safe Parameter-State-Management

    SIGNAL ARCHITECTURE:
    --------------------

    **Parameter Communication Signals:**
    ```python
    # Cross-Tab Parameter-Broadcasting
    parameter_changed = pyqtSignal(str, str, object, object)  # (tab, param, old_val, new_val)
    tab_parameters_updated = pyqtSignal(str, dict)            # (tab_name, all_parameters)
    validation_status_changed = pyqtSignal(str, bool, list)   # (tab_name, is_valid, errors)

    # Update Management Signals
    validation_requested = pyqtSignal()    # Debounced validation trigger
    generation_requested = pyqtSignal()    # Debounced generation trigger
    update_completed = pyqtSignal()        # Update cycle completion
    ```

    **Integration with DataLODManager:**
    - Parameter-Changes trigger Cache-Invalidation in DataLODManager
    - Cross-Tab Dependencies coordinate with LOD-Dependency-System
    - Export/Import includes LOD-Status-Information
    - Preset-Application coordinates with Generation-Orchestrator

    PARAMETER DEPENDENCY SYSTEM:
    ----------------------------

    **Standard Map-Generator Dependencies:**
    ```python
    # Hierarchical Parameter-Dependencies
    dependencies = {
        'terrain': [],                                    # Base Parameters
        'geology': ['terrain'],                          # Needs terrain parameters
        'water': ['terrain', 'geology'],                # Complex dependencies
        'weather': ['terrain'],                         # Environmental parameters
        'biome': ['terrain', 'geology', 'water', 'weather'], # High-level integration
        'settlement': ['terrain', 'geology', 'water', 'biome'] # Final layer
    }
    ```

    **Dependency-Driven Parameter-Sync:**
    - Automatic Parameter-Propagation bei Dependency-Changes
    - Constraint-Validation basierend auf Dependent-Parameter-Values
    - Change-Listener-System für automatic Cross-Tab-Updates
    - Dependency-Graph-Validation für Circular-Dependency-Prevention

    EXPORT/IMPORT SYSTEM:
    ---------------------

    **JSON Export Format:**
    ```json
    {
      "metadata": {
        "export_timestamp": "2024-01-15T10:30:00",
        "parameter_dependencies": {...},
        "map_generator_version": "1.0",
        "exported_tabs": ["terrain", "geology", "weather"],
        "total_parameters": 45,
        "export_format_version": "1.0"
      },
      "parameters": {
        "terrain": {
          "size": 512, "amplitude": 100, "octaves": 6,
          "frequency": 0.01, "persistence": 0.5
        },
        "geology": {
          "sedimentary_hardness": 30, "igneous_hardness": 80
        }
        // ... etc für alle Tabs
      }
    }
    ```

    **Template System:**
    ```python
    # Template-Generation für verschiedene Map-Types
    def export_parameter_template(filename, tab_names):
        template_data = {
            "template_info": {
                "name": "Parameter Template",
                "description": "Template with default parameter values",
                "included_tabs": tab_names
            },
            "parameters": get_default_parameters_for_tabs(tab_names)
        }
    ```

    **Import Validation:**
    - Format-Version-Compatibility-Checks
    - Missing-Parameter-Detection mit Default-Value-Substitution
    - Tab-Availability-Validation
    - Parameter-Range-Validation gegen aktuelle Constraints
    - Dependency-Consistency-Validation

    PRESET MANAGEMENT:
    ------------------

    **File-Based Storage:**
    ```
    presets/
    ├── general_preset_name.json
    ├── terrain_mountain_preset.json
    ├── weather_tropical_preset.json
    └── biome_desert_preset.json
    ```

    **Preset Structure:**
    ```json
    {
      "preset_info": {
        "name": "Mountain Terrain",
        "description": "High-altitude mountain terrain with rocky geology",
        "creation_date": "2024-01-15T10:30:00",
        "category": "terrain",
        "tags": ["mountains", "rocky", "high-altitude"],
        "parameter_count": 15,
        "file_version": "1.0"
      },
      "parameters": {
        "terrain": {...},
        "geology": {...}
      }
    }
    ```

    **Search & Discovery:**
    ```python
    # Tag-based Search
    mountain_presets = preset_manager.search_presets("mountains")

    # Category-filtered Listing
    terrain_presets = preset_manager.list_presets(category="terrain")

    # Advanced Search (name + description + tags)
    results = preset_manager.search_presets("rocky mountain")
    ```

    **Preset Categories:**
    - **general:** Mixed-parameter presets für verschiedene Map-Types
    - **terrain:** Terrain-focused presets (mountains, plains, islands)
    - **weather:** Weather-pattern presets (tropical, arid, temperate)
    - **biome:** Ecosystem-focused presets (desert, forest, tundra)
    - **settlement:** Civilization-pattern presets (medieval, modern, sparse)

    PARAMETER CONSTRAINTS & VALIDATION:
    -----------------------------------

    **Constraint System:**
    ```python
    # Parameter-Constraint-Registration
    hub.add_parameter_constraint("terrain", "size",
        lambda x: 64 <= x <= 2048 and (x & (x-1)) == 0)  # Power of 2
    hub.add_parameter_constraint("terrain", "amplitude", lambda x: 0 < x <= 1000)
    hub.add_parameter_constraint("geology", "sedimentary_hardness", lambda x: 0 <= x <= 100)
    ```

    **Validation Process:**
    1. **Individual Parameter-Validation:** Range/Type-Checks per Parameter
    2. **Cross-Parameter-Validation:** Consistency-Checks zwischen related Parameters
    3. **Dependency-Validation:** Dependent-Tab Parameter-Compatibility
    4. **Constraint-Function-Execution:** Custom Validation-Logic per Parameter
    5. **Error-Aggregation:** Comprehensive Error-Reporting für UI-Display

    **Standard Constraints:**
    - **Terrain:** Size (Power-of-2, 64-2048), Amplitude (1-1000), Octaves (1-10)
    - **Geology:** Hardness-Values (0-100), Warping-Factors (0.0-1.0)
    - **Weather:** Temperature-Range (-50°C to 60°C), Wind-Speed (0-200 km/h)
    - **Water:** Threshold-Values (0.0-1.0), Manning-Coefficient (0.01-0.1)
    - **Biome:** Factor-Values (0.0-2.0), Level-Values (0-5000m)
    - **Settlement:** Count-Values (1-50), Ratio-Values (0.0-2.0)

    UPDATE MANAGEMENT & RACE-CONDITION-PREVENTION:
    ----------------------------------------------

    **Debounced Update System:**
    ```python
    class ParameterUpdateManager:
        def __init__(self, debounce_ms=500):
            self.validation_timer = QTimer()    # 250ms debounce
            self.generation_timer = QTimer()    # 500ms debounce

        def request_validation(self):      # Fast validation
        def request_generation(self):      # Slower generation
    ```

    **Update Coordination:**
    1. **Parameter-Change-Detection:** UI-Slider-Changes trigger debounced Updates
    2. **Validation-Phase:** Fast Parameter-Validation (250ms debounce)
    3. **Generation-Phase:** Slower Generation-Triggering (500ms debounce)
    4. **Cross-Tab-Propagation:** Parameter-Changes propagate zu dependent Tabs
    5. **Generation-Coordination:** Integration mit GenerationOrchestrator

    **Race-Condition-Prevention:**
    - **Pending-Update-Flags:** Prevent multiple simultaneous Updates
    - **Timer-Coordination:** Validation-Timer cancels Generation-Timer
    - **State-Synchronization:** Thread-safe Parameter-State-Management
    - **Signal-Ordering:** Qt's Event-Queue garantiert Signal-Order

    INTEGRATION PATTERNS:
    ---------------------

    **Tab Integration:**
    ```python
    # Tab-Registration mit Dependencies
    parameter_hub.register_tab("terrain", terrain_tab, dependencies=[])
    parameter_hub.register_tab("geology", geology_tab, dependencies=["terrain"])

    # Parameter-Change-Broadcasting
    def on_parameter_changed(self, param_name, new_value):
        parameter_hub.broadcast_parameter_change(self.tab_name, param_name, new_value)

    # Cross-Tab Parameter-Listening
    parameter_hub.add_parameter_change_listener("geology", self.on_terrain_parameters_changed)
    ```

    **Export/Import Integration:**
    ```python
    # Complete Parameter-Export
    export_manager.export_parameters_json("my_map_config.json", include_metadata=True)

    # Selective Import
    export_manager.import_parameters_json("template.json",
        selective_import=["terrain", "geology"])

    # Preset-Management
    preset = export_manager.create_parameter_preset("Mountain Map",
        description="Rocky mountain terrain", tab_filter=["terrain", "geology"])
    preset_manager.save_preset("mountain_map", preset, category="terrain",
        tags=["mountains", "rocky"])
    ```

    **DataLODManager Integration:**
    - Parameter-Changes trigger cache_invalidated Signal in DataLODManager
    - Export-Data includes LOD-Status-Information from DataLODManager
    - Preset-Application coordinates mit LOD-System für optimale Performance
    - Parameter-History correlates mit Generation-History für Debugging

    PERFORMANCE CHARACTERISTICS:
    ----------------------------

    **Parameter-Sync Performance:**
    - Change-Detection via Hash-Comparison (O(1) für einzelne Parameter)
    - Debounced Updates reduzieren Signal-Spam von 100+ Events/s auf <2 Events/s
    - Parameter-Cache eliminiert redundante Tab-Queries
    - Lazy-Loading von Preset-Files bei First-Access

    **Memory Efficiency:**
    - Parameter-History-Limit (1000 Events) verhindert unbegrenztes Growth
    - WeakReference-Pattern für Tab-Registration prevents Memory-Leaks
    - JSON-Streaming für Large-Parameter-Set Export/Import
    - Preset-Cache mit automatic Cleanup für alte Entries

    **File I/O Optimization:**
    - Batch-Export reduziert I/O-Operations
    - Preset-Cache eliminiert redundante File-Reads
    - Atomic-Write-Pattern für Data-Integrity bei Export/Import
    - Background-Loading für Preset-Discovery-Performance

    MONITORING & DEBUGGING:
    -----------------------

    **Parameter-Change-History:**
    ```python
    # Detaillierte Change-History für Debugging
    history = parameter_hub.get_parameter_change_history(tab_name="terrain", limit=50)
    # Returns: List[ParameterChangeEvent] mit timestamp, old_value, new_value

    # Cross-Tab Parameter-Dependencies
    deps = parameter_hub.get_dependency_parameters("biome")
    # Returns: {"terrain": {...}, "weather": {...}, "water": {...}}
    ```

    **Export/Import Statistics:**
    ```python
    # Export-Success-Rate-Tracking
    export_stats = export_manager.get_export_statistics()
    # {"successful_exports": 45, "failed_exports": 2, "average_size_kb": 12.5}

    # Import-Compatibility-Analysis
    compatibility = export_manager.analyze_import_compatibility("old_config.json")
    # {"compatible": True, "missing_parameters": [], "version_issues": []}
    ```

    **Preset-Usage-Analytics:**
    ```python
    # Preset-Usage-Statistics
    usage_stats = preset_manager.get_usage_statistics()
    # {"most_used_presets": [...], "category_distribution": {...}, "search_terms": [...]}
    ```

    ERROR HANDLING & RESILIENCE:
    -----------------------------

    **Graceful Degradation:**
    - Continued Operation auch bei einzelnen Tab-Registration-Failures
    - Parameter-Sync funktioniert auch bei partial Tab-Availability
    - Export/Import mit partial Success (successful Tabs werden processed)
    - Preset-System funktioniert auch bei einzelnen corrupted Preset-Files

    **Recovery Mechanisms:**
    - Automatic Parameter-Default-Substitution bei Import-Errors
    - Constraint-Violation-Recovery durch Value-Clamping
    - Corrupted-Preset-Recovery durch Backup-Creation
    - Cross-Tab-Sync-Recovery bei Signal-Connection-Problems

    **Data Integrity:**
    - Atomic-Export/Import verhindert partial File-Corruption
    - Parameter-Constraint-Validation verhindert Invalid-States
    - Backup-Creation vor destructive Operations (Preset-Overwrite, etc.)
    - Change-History-Persistence für Post-Mortem-Analysis

    Diese Parameter-Management-Architektur schafft eine robuste, benutzerfreundliche
    Basis für reproduzierbare Map-Generation mit flexiblem Export/Import-System und
    intelligenter Cross-Tab-Parameter-Coordination.
    """

def generation_orchestrator():
    """
    Path: gui/managers/generation_orchestrator.py
    date_changed: 08.08.2025

    =========================================================================================
    GENERATION ORCHESTRATOR - ZENTRALE GENERATION-KOORDINATION UND THREADING
    =========================================================================================

    OVERVIEW:
    ---------
    GenerationOrchestrator koordiniert alle 6 Map-Generatoren (Terrain, Geology, Weather,
    Water, Biome, Settlement) durch ein intelligentes Threading-System mit Dependency-
    Resolution, LOD-Progression und Parameter-Impact-Analyse. Das System ermöglicht
    parallele Background-Generierung ohne UI-Blocking und automatische Cache-Invalidation
    bei Parameter-Änderungen.

    CORE ARCHITECTURE:
    ------------------

    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                       GenerationOrchestrator (Central Hub)                     │
    │ ┌─────────────────┬─────────────────┬─────────────────┬─────────────────────┐ │
    │ │ Request Queue   │ LOD Progression │ Threading Mgmt  │ Parameter Impact    │ │
    │ │ - Dependency    │ - Auto LOD      │ - 3 Parallel    │ - Change Analysis   │ │
    │ │   Resolution    │ - Incremental   │   Generations   │ - Cache Invalidation│ │
    │ │ - Priority      │   UI Updates    │ - Background    │ - Downstream Effects│ │
    │ │   Handling      │ - DataManager   │   Processing    │ - Dependency Chain  │ │
    │ │ - Timeout Mgmt  │                 │ - Thread Safety │   Invalidation      │ │
    │ └─────────────────┴─────────────────┴─────────────────┴─────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────────────────┘

    KEY SYSTEMS:
    ------------

    1. **DEPENDENCY QUEUE SYSTEM**
       - Intelligente Request-Queue mit automatischer Dependency-Resolution
       - Priority-basierte Verarbeitung (Terrain=10, Geology=8, Settlement=4)
       - Parallel-Processing-Limit (max 3 gleichzeitige Generationen)
       - Timeout-Management (5min per Request, 10min Queue-Timeout)
       - Deadlock-Prevention durch kontinuierliche Queue-Resolution

    2. **LOD PROGRESSION ENGINE**
       - Automatische LOD-Sequence-Erstellung (LOD64→LOD128→LOD256→FINAL)
       - Incremental UI-Updates nach jeder LOD-Stufe
       - DataManager-Integration für automatisches Result-Storage
       - Skip-Logic für bereits vorhandene LOD-Level
       - Thread-per-LOD-Level für maximale Parallelisierung

    3. **PARAMETER IMPACT MATRIX**
       - Kategorisierung aller Parameter nach Impact-Level (high/medium/low)
       - Automatic downstream Dependency-Invalidation
       - Smart Cache-Management basierend auf Parameter-Changes
       - Generator-spezifische Impact-Regeln pro Parameter
       - Cross-Generator Parameter-Effect-Propagation

    4. **THREAD MANAGEMENT & STATE TRACKING**
       - Thread-Pool für Background-Generation ohne UI-Blocking
       - Comprehensive State-Tracking für alle 6 Generator-Threads
       - Performance-Monitoring (Generation-Timings, Memory-Usage)
       - Thread-Safety durch QMutex und Qt's Signal-System
       - Graceful Thread-Termination bei App-Shutdown

    SIGNAL ARCHITECTURE (Harmonized):
    ----------------------------------

    **Standardized Tab-Compatible Signals:**
    ```python
    # Homogene Signals für alle Tab-Typen (kompatibel mit BaseMapTab)
    generation_completed = pyqtSignal(str, dict)    # (result_id, result_data)
    lod_progression_completed = pyqtSignal(str, str) # (result_id, lod_level)
    generation_progress = pyqtSignal(int, str)       # (progress, message)

    # Extended Orchestrator Signals
    dependency_invalidated = pyqtSignal(str, list)   # (generator_type, affected_generators)
    batch_generation_completed = pyqtSignal(bool, str) # (success, summary_message)
    queue_status_changed = pyqtSignal(list)          # (thread_status_list)
    ```

    Diese harmonisierte Signal-Architektur eliminiert die Notwendigkeit für komplexe
    StandardOrchestratorHandler und ermöglicht direkte Signal-Connections in Tabs.

    REQUEST PROCESSING FLOW:
    ------------------------

    **1. Request Submission:**
    ```python
    # Tab → Orchestrator
    request = OrchestratorRequestBuilder.build_terrain_request(parameters, target_lod="FINAL")
    request_id = orchestrator.request_generation(request)
    ```

    **2. Dependency Resolution:**
    - Request wird in DependencyQueue eingereiht
    - Kontinuierliche Prüfung auf verfügbare Dependencies (alle 2s)
    - Priority-basierte Start-Reihenfolge

    **3. LOD Progression Execution:**
    - Automatic LOD-Sequence: LOD64 → LOD128 → LOD256 → FINAL
    - Thread-per-LOD-Level mit incremental Results
    - DataManager-Integration für automatic Result-Storage
    - UI-Updates nach jeder abgeschlossenen LOD-Stufe

    **4. Result Processing:**
    - Automatic DataManager-Storage basierend auf Generator-Type
    - Signal-Emission für Tab-Updates
    - Queue-Resolution-Triggering für abhängige Requests
    - Performance-Metrics-Collection

    DEPENDENCY MANAGEMENT:
    ----------------------

    **Hierarchical Dependency Tree:**
    ```python
    dependency_tree = {
        GeneratorType.TERRAIN: set(),                                           # Base Layer
        GeneratorType.GEOLOGY: {GeneratorType.TERRAIN},                       # Depends on Terrain
        GeneratorType.WEATHER: {GeneratorType.TERRAIN},                       # Depends on Terrain
        GeneratorType.WATER: {GeneratorType.TERRAIN, GeneratorType.GEOLOGY,   # Complex Dependencies
                              GeneratorType.WEATHER},
        GeneratorType.BIOME: {GeneratorType.TERRAIN, GeneratorType.WEATHER,   # High-Level Integration
                             GeneratorType.WATER},
        GeneratorType.SETTLEMENT: {GeneratorType.TERRAIN, GeneratorType.WATER, # Final Layer
                                  GeneratorType.BIOME}
    }
    ```

    **Dependency Resolution Logic:**
    - Breadth-First-Search durch Dependency-Tree
    - Minimum-LOD-Requirements pro Dependency
    - Automatic Retry bei temporär nicht verfügbaren Dependencies
    - Timeout-basierte Deadlock-Prevention

    PARAMETER IMPACT SYSTEM:
    ------------------------

    **Impact Classification Matrix:**
    ```python
    impact_matrix = {
        GeneratorType.TERRAIN: {
            "high_impact": ["map_seed", "size", "amplitude", "octaves", "frequency"],
            "medium_impact": ["persistence", "lacunarity"],
            "low_impact": ["redistribute_power"]
        },
        # ... similar für alle 6 Generator-Types
    }
    ```

    **Impact Processing:**
    1. **Parameter-Change-Detection:** Hash-basierte Comparison mit cached Parameters
    2. **Impact-Level-Determination:** Lookup in Generator-spezifischer Impact-Matrix
    3. **Downstream-Effect-Calculation:** Traversierung des Dependency-Trees
    4. **Cache-Invalidation:** Selective oder Complete basierend auf Impact-Level
    5. **Dependency-Chain-Invalidation:** Cascading Invalidation für betroffene Generatoren

    **Impact-Level Effects:**
    - **High-Impact:** Complete Cache-Invalidation aller downstream Dependencies
    - **Medium-Impact:** Invalidation direkter Dependencies only
    - **Low-Impact:** No automatic Invalidation (Manual regeneration required)

    LOD PROGRESSION DETAILS:
    ------------------------

    **Progression Strategy:**
    ```python
    # Automatic LOD-Sequence basierend auf DataManager-Status
    def create_lod_sequence(generator_type, target_lod):
        available_lods = datamanager.get_available_lods(generator_type)
        missing_lods = calculate_missing_sequence(available_lods, target_lod)
        return missing_lods  # Nur fehlende LODs werden generiert
    ```

    **Thread-per-LOD Execution:**
    - Jede LOD-Stufe läuft in eigenem GenerationThread
    - Incremental Result-Storage nach jeder LOD-Completion
    - UI-Updates mit bestem verfügbarem LOD (immediate feedback)
    - Automatic Next-LOD-Triggering bei Success

    **DataManager Integration:**
    ```python
    # Automatic Result-Storage basierend auf Generator-Type
    def save_generation_result_to_data_manager(generator_type, lod_level, result_data):
        if generator_type == "terrain":
            datamanager.set_terrain_data_complete(result_data.generator_output, parameters)
        elif generator_type == "geology":
            datamanager.set_geology_data("rock_map", result_data.rock_map, parameters)
        # ... etc für alle Generator-Types
    ```

    THREADING & PERFORMANCE:
    ------------------------

    **Thread Management:**
    - QThread-basierte Background-Processing
    - Thread-Pool-Pattern mit QMutex-Protection
    - Graceful Thread-Termination (3s timeout, dann force-kill)
    - Thread-State-Tracking für UI-Display (6-Thread-Status-Grid)

    **Performance Optimizations:**
    - Lazy-Loading von Generator-Instanzen (Import only when needed)
    - Memory-Usage-Tracking pro Generation
    - Generation-Timing-Statistics für Performance-Analysis
    - Thread-Reuse für Sequential LOD-Processing

    **Memory Management:**
    - Integration mit DataLODManager's ResourceTracker
    - Automatic Memory-Cleanup bei Thread-Completion
    - Large-Array-Detection und specialized Handling
    - Force-Garbage-Collection bei Memory-Warnings

    QUEUE MANAGEMENT & RESILIENCE:
    ------------------------------

    **Queue Processing:**
    ```python
    # Kontinuierliche Queue-Resolution (alle 2s)
    def resolve_dependency_queue():
        available_requests = queue.get_available_requests(
            completed_generators=get_completed_generators(),
            active_limit=3  # Max 3 parallele Generationen
        )
        for request in available_requests:
            start_lod_progression(request)
    ```

    **Timeout & Error Handling:**
    - **Request-Timeout:** 5 Minuten per Generation-Request
    - **Queue-Timeout:** 10 Minuten maximale Queue-Verweilzeit
    - **Thread-Timeout:** 3 Sekunden für graceful Thread-Termination
    - **Deadlock-Prevention:** Kontinuierliche Queue-Resolution verhindert Starvation

    **Error Recovery:**
    - Automatic Retry für transiente Fehler (Import-Errors, Memory-Shortage)
    - Graceful Degradation bei Generator-Unavailability
    - Error-Propagation mit detailliertem Error-Context
    - Cleanup-on-Error verhindert Resource-Leaks

    REQUEST BUILDER SYSTEM:
    -----------------------

    **Type-Safe Request Construction:**
    ```python
    # Generator-spezifische Request-Builder mit Parameter-Validation
    terrain_request = OrchestratorRequestBuilder.build_terrain_request(
        parameters={"size": 512, "amplitude": 100, "octaves": 6},
        target_lod="FINAL",
        source_tab="terrain"
    )

    # Batch-Request für Multiple Generators
    batch_request = OrchestratorRequestBuilder.build_batch_request([
        terrain_request, geology_request, weather_request
    ])
    ```

    **Parameter Validation:**
    - Generator-spezifische Parameter-Range-Validation
    - Default-Value-Injection für missing Parameters
    - Type-Checking und Value-Constraint-Enforcement
    - Invalid-Request-Rejection mit detailed Error-Messages

    INTEGRATION PATTERNS:
    ---------------------

    **Tab Integration (Simplified):**
    ```python
    # Direkte Signal-Connection ohne Handler-Classes
    orchestrator.generation_completed.connect(tab.on_generation_completed)
    orchestrator.lod_progression_completed.connect(tab.on_lod_progression_completed)
    orchestrator.generation_progress.connect(tab.update_progress_bar)

    # Request-Submission
    request_id = orchestrator.request_generation(
        OrchestratorRequestBuilder.build_terrain_request(tab.get_parameters())
    )
    ```

    **DataManager Integration:**
    - Automatic Result-Storage eliminiert manual DataManager-Calls in Tabs
    - LOD-Status-Synchronization zwischen Orchestrator und DataManager
    - Cache-Invalidation-Coordination bei Parameter-Changes
    - Memory-Usage-Coordination für Large-Generation-Results

    **UI Integration:**
    - 6-Thread-Status-Display für Live-Generation-Monitoring
    - Queue-Status-Updates für User-Feedback
    - Progress-Updates mit LOD-specific Detail-Messages
    - Error-Display mit actionable Error-Information

    MONITORING & DEBUGGING:
    -----------------------

    **Performance Metrics:**
    ```python
    # Generation-Timing-Statistics
    generation_timings = orchestrator.generation_timings
    # {generator_type: {lod_level: duration_seconds}}

    # Memory-Usage-Tracking
    memory_usage = orchestrator.get_memory_usage_summary()
    # {"data_manager_usage": {...}, "active_threads": 3, "processing_requests": 2}

    # Thread-State-Monitoring
    thread_states = orchestrator.state_tracker.get_all_thread_status()
    # [{"generator": "Terrain", "status": "Generating", "progress": 45, "runtime": "2.3s"}]
    ```

    **Queue Analytics:**
    ```python
    # Queue-Status für Performance-Analysis
    queue_status = orchestrator.dependency_queue.get_queue_status()
    # {"total_queued": 5, "by_generator": {"terrain": 1, "geology": 2, "water": 2}}

    # Timeout-Analysis
    timed_out_requests = orchestrator.dependency_queue.get_timed_out_requests(current_time)
    ```

    ERROR HANDLING & RESILIENCE:
    -----------------------------

    **Graceful Degradation:**
    - Continued Operation auch bei einzelnen Generator-Failures
    - Automatic Fallback auf Legacy-Generation-Methods bei Orchestrator-Problems
    - Partial-Success-Handling bei Batch-Generations
    - UI-Responsiveness auch bei Background-Generation-Overload

    **Recovery Mechanisms:**
    - Automatic Request-Retry für transiente Failures
    - Thread-Pool-Recovery nach Thread-Crashes
    - Queue-State-Recovery nach Memory-Shortage
    - DataManager-Integration-Recovery bei Connection-Problems

    Diese Orchestrator-Architektur ermöglicht robuste, performante Background-Generierung
    aller Map-Components mit automatischer Dependency-Resolution und intelligenter
    Resource-Management-Integration.

    =========================================================================================
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

    ERROR CATEGORIES:
    • Setup Errors: UI creation failures → Minimal fallback UI
    • Display Errors: Rendering failures → Continue with degraded display
    • Generation Errors: Algorithm failures → Clear error status, enable retry
    • Resource Errors: Memory/cleanup failures → Force cleanup and logging

    MEMORY MANAGEMENT:
    ------------------

    RESOURCE TRACKING SYSTEM:
    • ResourceTracker: Systematic registration and cleanup of all resources
    • Weak references: Prevent circular dependencies
    • Cleanup functions: Custom cleanup logic per resource type
    • Age-based cleanup: Automatic cleanup of old resources

    MEMORY OPTIMIZATION:
    • Display change detection: Prevents unnecessary texture uploads
    • Large array handling: Force garbage collection for >50MB arrays
    • Resource pooling: Reuse display resources where possible
    • Exception safety: Cleanup guaranteed even on errors

    PERFORMANCE CONSIDERATIONS:
    ---------------------------

    UI RESPONSIVENESS:
    • Auto-simulation debouncing: Prevents excessive generation triggers
    • Progressive loading: LOD-based generation with incremental updates
    • Background processing: Non-blocking generation workflows
    • Change detection: Skip rendering if data hasn't changed

    MEMORY EFFICIENCY:
    • Reference sharing: numpy arrays shared between components (no copies)
    • Lazy loading: Resources created only when needed
    • Systematic cleanup: Prevent memory leaks through resource tracking
    • Garbage collection: Forced GC for large operations

    THREAD SAFETY:
    • Signal-based communication: Qt's thread-safe signal system
    • UI updates: All UI changes in main thread via pyqtSlot
    • Resource access: Coordinated through data_manager
    • State synchronization: Generation state managed centrally

    =========================================================================================
    USAGE EXAMPLES FOR SUB-CLASSES:
    =========================================================================================

    MINIMAL TAB IMPLEMENTATION:
    ```python
    class TerrainTab(BaseMapTab):
        def __init__(self, data_manager, navigation_manager, shader_manager, generation_orchestrator):
            # Required dependencies for this tab
            self.required_dependencies = []  # No dependencies for terrain (base generator)

            super().__init__(data_manager, navigation_manager, shader_manager, generation_orchestrator)

            # Setup orchestrator integration
            self.setup_standard_orchestrator_handlers("terrain")

            # Create parameter controls
            self.create_parameter_controls()

        def create_parameter_controls(self):
            # Add custom parameter sliders/controls to self.control_panel.layout()
            pass

        def generate_terrain_system(self):
            # Core generation logic
            # Should emit self.data_updated.emit("terrain", "heightmap") when done
            pass
    ```

    ADVANCED TAB WITH CUSTOM DISPLAYS:
    ```python
    class GeologyTab(BaseMapTab):
        def __init__(self, data_manager, navigation_manager, shader_manager, generation_orchestrator):
            self.required_dependencies = ["heightmap", "slopemap"]  # Requires terrain data

            super().__init__(data_manager, navigation_manager, shader_manager, generation_orchestrator)
            self.setup_standard_orchestrator_handlers("geology")
            self.create_parameter_controls()

        def create_visualization_controls(self):
            # Override to add custom display modes (rock_map, hardness_map)
            widget = super().create_visualization_controls()

            self.rock_radio = QRadioButton("Rock Types")
            self.hardness_radio = QRadioButton("Hardness")

            # Add to layout and connect signals
            return widget

        def update_display_mode(self):
            # Custom rendering logic for geology-specific displays
            if self.rock_radio.isChecked():
                rock_data = self.data_manager.get_geology_data("rock_map")
                current_display = self.get_current_display()
                if rock_data is not None and current_display:
                    self._render_current_mode(1, current_display, rock_data, "rock_types")
            # ... etc

        def generate_geology_system(self):
            # Geology generation logic
            # Emits data_updated signals for rock_map, hardness_map etc.
            pass
    ```

    INTEGRATION CHECKLIST FOR NEW TABS:
    • ✓ Define required_dependencies list
    • ✓ Call setup_standard_orchestrator_handlers(generator_type)
    • ✓ Implement generate_[tab_name]_system() method
    • ✓ Create parameter controls and add to control_panel.layout()
    • ✓ Override visualization controls if custom display modes needed
    • ✓ Override update_display_mode() if custom rendering needed
    • ✓ Emit data_updated signals when generation completes
    • ✓ Handle parameter changes and auto-simulation appropriately

    =========================================================================================

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