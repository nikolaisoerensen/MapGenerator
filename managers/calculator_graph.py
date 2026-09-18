"""
Path: managers/calculator_graph.py

Funktionsweise: Zentrale Graph-Definition der 29 echten Rechenklassen (Calculators)
aus docs/generation_pipeline_dependencies.md - feingranularer als die bisherige
6-Generator-Sicht des Orchestrators (dessen dependency_tree/DependencyQueue nur
"Terrain/Geology/Weather/Water/Biome/Settlement" als Knoten kennt).

Aufgabe: Grundlage für den LOD-Lockstep-Umbau (Tracker #16, siehe
docs/generation_pipeline_dependencies.md und docs/session_handover_2026-07-08.md).
Zwei Scheduler-Klassen mit unterschiedlichem Zweck:
- CalculatorRoundScheduler: verwaltet nur GENERATOR-INTERNE Teilmengen (z.B. Terrain:
  noise -> redistribution -> {slope, shadow}), genutzt von core/*_generator.py als
  Zwischenschritt beim Zerlegen jedes einzelnen Generators.
- CalculatorDispatcher: verwaltet den KOMPLETTEN Graph generatorübergreifend und
  löst den bisherigen 6-Knoten dependency_tree in generation_orchestrator.py ab -
  damit kann z.B. Settlement komplett starten, ohne auf Biome zu warten (kein
  settlement.*-Knoten hängt von einem biome.*-Output ab, siehe
  [[project-settlement-plot-physics-rebuild]] - settlement.plot_nodes brauchte
  früher biome_map, seit dem Umbau auf PlotPhysicsSystem nicht mehr).

water.erosion_feedback (Water->Terrain-Rückkopplung, "Problem #22" in den Docs) ist
laut Dokumentation bekannt kaputt (produziert keine Wirkung) und deshalb hier
bewusst NICHT als aktiver Knoten aufgenommen. settlement.city_blocks/
settlement.landscape_voronoi wurden im Zuge des Settlement-Plot-Physics-Umbaus
entfernt (siehe oben) - Zähler in der Doku ("29 aktive Calculators") ist damit
veraltet, aktuelle Zahl siehe len(CALCULATOR_GRAPH).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass(frozen=True)
class CalculatorSpec:
    calculator_id: str
    generator: str  # "terrain"/"geology"/"weather"/"water"/"biome"/"settlement" (GeneratorType.value)
    depends_on: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)


_CALCULATOR_SPECS = [
    # --- Terrain (#1-#4) ---
    CalculatorSpec("terrain.noise", "terrain", [], ["noise_grid"]),
    # ridge_map (-1 in Kerben, +1 auf Kaemmen) kommt seit 2026-07-30 aus dem
    # ATEF-Erosionsfilter, der in _calc_redistribution() mitlaeuft (SPEZIFIKATION
    # §9). Bewusst kein eigener Knoten: der Filter liefert die endgueltige
    # Gelaendeform, und 20+ Lesestellen holen die Heightmap von hier - sie alle
    # umzuhaengen ist das Risiko aus §4.5. Noch von niemandem gelesen.
    # river_mask/river_order kommen seit 2026-07-30 aus dem Flussnetz-Skelett,
    # das ebenfalls in _calc_redistribution() mitlaeuft (SPEZIFIKATION §12).
    # Noch von niemandem gelesen und nicht als Anzeige-Layer registriert.
    # river_generation seit 2026-08-05: 3 = Makro (Strom), 2 = Meso, 1 = Mikro
    # (Bach), 0 = kein Fluss. Ein NEUER Output statt einer geaenderten
    # Bedeutung von river_order - Water und Biome sollen einen Trog von einem
    # Bach unterscheiden koennen, ohne dass bestehende Leser umlernen muessen.
    # region_map seit 2026-08-06: 0..8, welche der neun Regionen an diesem Pixel
    # fuehrt (Reihenfolge von alle_regionen(), Nordwest nach Suedost). Sie
    # entsteht aus DEMSELBEN Gewichtsfeld, das auch die Gelaendeparameter
    # traegt - deshalb haengt sie an terrain und nicht an settlement, obwohl
    # settlement ihr Hauptleser ist. Ein zweiter Aufruf von voronoi_regionen()
    # an anderer Stelle waere eine zweite Wahrheit (§4.5); der Regional-Reiter
    # tat bis zu diesem Datum genau das.
    # NUR bei aktiver Weltkarte belegt - im alten Pfad gibt es keine Regionen.
    CalculatorSpec("terrain.redistribution", "terrain", ["terrain.noise"],
                   ["heightmap", "ridge_map", "river_mask", "river_order",
                    "river_generation", "region_map", "klima_map", "spielkarte"]),
    CalculatorSpec("terrain.slope", "terrain", ["terrain.redistribution"], ["slopemap"]),
    CalculatorSpec("terrain.shadow", "terrain", ["terrain.redistribution"], ["shadowmap"]),

    # --- Geology (3D-Gesteinsstapel-Rework, siehe core/geology_generator.py
    # und den zugehörigen Umsetzungsplan) ---
    # layer_thickness braucht jetzt auch terrain.slope (Slope-Verdünnung der
    # Schichtdicke, Teil-2-Rework Punkt A4).
    CalculatorSpec("geology.layer_thickness", "geology",
                   ["terrain.redistribution", "terrain.slope"], ["layer_thickness"]),
    # stack_deformation (vormals tectonic_delta) enthält jetzt zusätzlich den
    # Terrain-Hub-Anteil (terrain_hub_delta) - wirkt NUR auf den Ausbiss,
    # NICHT mehr additiv in height_delta (siehe geology.intrusions unten).
    CalculatorSpec("geology.tectonic_displacement", "geology", ["terrain.redistribution"],
                   ["stack_deformation", "terrain_hub_delta", "tilt_delta", "fold_delta",
                    "fault_delta", "fault_distance_map"]),
    CalculatorSpec("geology.outcrop", "geology",
                   ["geology.layer_thickness", "geology.tectonic_displacement", "terrain.redistribution"],
                   ["layer_id_map", "layer_boundaries"]),
    # Keine direkte Abhängigkeit mehr von geology.tectonic_displacement -
    # height_delta kommt jetzt ausschließlich aus der Intrusions-Dom-Hebung
    # selbst, stack_deformation wird hier nicht mehr gelesen (nur noch
    # transitiv über geology.outcrop gebraucht).
    CalculatorSpec("geology.intrusions", "geology", ["geology.outcrop"],
                   ["layer_id_map", "intrusion_distance_map", "intrusion_delta", "height_delta"]),
    # Rein Gesteinstyp-Klassifikation (Relief-Proxy aus der Terrain-Heightmap),
    # trägt bewusst KEINEN Höhenbeitrag bei - würde sich sonst mit Waters
    # erosion_map/sedimentation_map verdoppeln (siehe Umsetzungsplan Punkt 8).
    # Braucht deshalb auch KEINE neue water.*-Abhängigkeit.
    CalculatorSpec("geology.sediment_overlay", "geology", ["geology.intrusions"], ["layer_id_map"]),
    CalculatorSpec("geology.metamorphic_overprint", "geology",
                   ["geology.tectonic_displacement", "geology.intrusions"], ["metamorphic_grade_map"]),
    CalculatorSpec("geology.rock_color", "geology",
                   ["geology.sediment_overlay", "geology.metamorphic_overprint"], ["rock_map"]),
    CalculatorSpec("geology.hardness", "geology",
                   ["geology.sediment_overlay", "geology.metamorphic_overprint"], ["hardness_map"]),

    # --- Weather (#11-#14) ---
    # erosion.hydraulic als Abhaengigkeit: weather.temperature ist die Wurzel
    # der Weather-Kette (wind/humidity/precipitation haengen an ihr), und sie
    # liest ihre Heightmap ueber get_calculator_combined_heightmap(). Ohne
    # diese Kante rechnet das gesamte Wetter auf dem UNerodierten Gelaende -
    # genau der Zustand, der bis 2026-07-28 galt, als die Erosion noch hinter
    # Weather im Water-Block lag. Sie ist damit die Kante, wegen der die
    # Erosion ueberhaupt ein eigener Generator geworden ist.
    #
    # OFFEN und bewusst nicht mitgezogen: terrain.shadow wirft seine Schatten
    # weiterhin auf dem unerodierten Gelaende. Das zu aendern hiesse, einen
    # TERRAIN-Knoten auf die Erosion warten zu lassen - machbar (kein Zyklus,
    # die Erosion braucht keine shadowmap), aber ein Eingriff in die
    # Generator-Reihenfolge, der eine eigene Runde verdient.
    # biome.preseed_hint ergaenzt 2026-07-29: die Verdunstung, die den
    # Feuchtegehalt der Atmosphaere speist, hing an einem FESTEN 50-%-Wert
    # (weather_generator.py, soil_moisture_norm). Damit war der groessere Teil
    # der Feuchte breitengrad-UNabhaengig, und der subtropische
    # Trockenguertel kam im Niederschlag nie an - gemessen trug die
    # Klimatologie nur 32-42 % zur Impfung bei, der Rest war die Konstante.
    #
    # Das Pre-Biome kennt Breitengrad und Topografie und liefert ueber
    # _BIOME_MOISTURE_CAPACITY, wieviel Wasser der Untergrund ueberhaupt
    # halten kann (Wueste 20, Sumpf 95, Fels 15). Sand kann Feuchte eben
    # nicht weit transportieren.
    #
    # KEIN ZYKLUS, nachgerechnet: biome.preseed_hint haengt ausschliesslich an
    # terrain.redistribution und erosion.slope, nie an weather.*. Genau dafuer
    # wurde der Vorab-Schaetzwert gebaut - er bricht die Henne-Ei-Beziehung
    # zwischen Biom und Feuchte auf.
    CalculatorSpec("weather.temperature", "weather",
                   ["terrain.redistribution", "terrain.shadow", "erosion.hydraulic",
                    "biome.preseed_hint"],
                   ["temp_map"]),
    CalculatorSpec("weather.wind", "weather",
                   ["terrain.redistribution", "weather.temperature", "terrain.shadow"], ["wind_map"]),
    CalculatorSpec("weather.humidity", "weather",
                   ["terrain.redistribution", "weather.temperature", "weather.wind"], ["humid_map"]),
    CalculatorSpec("weather.precipitation", "weather",
                   ["weather.humidity", "weather.temperature", "weather.wind", "terrain.redistribution"],
                   ["precip_map"]),

    # --- Erosion (eigener Generator seit 2026-07-28) ---
    #
    # EIN Knoten fuer den kompletten Feld-Erosionslauf (Pipe-Hydraulik +
    # mitstroemendes Sedimentfeld + Boeschungswinkel, siehe
    # core/erosion_generator.py). Ein einzelner Knoten, weil die acht Passes
    # einen gemeinsamen, eng gekoppelten Zustand teilen - sie ueber mehrere
    # Knoten zu trennen hiesse, Hoehe/Wasser/Fluss/Sediment jede Runde durch
    # den Storage zu schleusen.
    #
    # WARUM ZWISCHEN GEOLOGY UND WEATHER: bis 2026-07-28 lag die Erosion im
    # Water-Block und damit HINTER weather.*. Temperatur, Wind und
    # Niederschlag rechneten also auf dem UNerodierten Gelaende - obwohl die
    # Erosion per Nutzer-Vorgabe ohnehin nicht vom Regen abhaengen darf. Der
    # eigene Knoten mit ausschliesslich terrain/geology-Abhaengigkeiten dreht
    # das um: ab hier sehen ALLE nachgelagerten Generatoren die tatsaechlichen
    # Taeler. Die Regen-Entkopplung bleibt strukturell erzwungen, weil
    # weather.* hier gar nicht auftaucht.
    CalculatorSpec("erosion.hydraulic", "erosion",
                   ["terrain.redistribution", "geology.hardness"],
                   ["erosion_map", "sedimentation_map",
                    "thermal_erosion_map", "thermal_deposition_map",
                    "sediment_load_map", "water_depth_map", "flow_velocity_map"]),

    # Der Slope NACH der Erosion (2026-07-28).
    #
    # terrain.slope rechnet auf dem UNerodierten Gelaende - das muss so
    # bleiben, weil geology.layer_thickness ihn braucht und Geology vor der
    # Erosion laeuft. terrain.slope einfach hinter die Erosion zu haengen
    # waere ein ZYKLUS, nachgerechnet:
    #
    #   terrain.slope           <- erosion.hydraulic   (die gewuenschte Kante)
    #   erosion.hydraulic       <- geology.hardness
    #   geology.hardness        <- ... <- geology.layer_thickness
    #   geology.layer_thickness <- terrain.slope
    #
    # Deshalb ein ZWEITER Knoten statt eines verschobenen. Er rechnet mit
    # demselben SlopeCalculator, nur auf der kombinierten (erodierten)
    # Heightmap. Alles, was nach der Erosion kommt, liest ab hier diesen -
    # sonst kennen Biome und Settlement die frisch eingeschnittenen Rinnen und
    # Kaemme nicht, auf denen sie ihre Entscheidungen treffen.
    CalculatorSpec("erosion.slope", "erosion", ["erosion.hydraulic"], ["slopemap"]),

    # --- Water (#15-#21, #22 erosion_feedback bewusst ausgeschlossen - siehe Docstring) ---
    #
    # ZWINGENDE REIHENFOLGE PRO LOD-RUNDE (Nutzer-Vorgabe 2026-07-27):
    #   erosion.hydraulic -> lake_detection -> flow_network -> manning_flow ->
    #   {soil_moisture, evaporation}
    #
    # Erst wird das GELAENDE geformt (Droplet-Erosion + Boeschungswinkel,
    # beide UNABHAENGIG vom Weather-Niederschlag), danach wird der
    # Wasserkreislauf mit dem ECHTEN Regen auf genau diesem veraenderten
    # Gelaende simuliert. Diese Reihenfolge steht bewusst hier im Graph und
    # nicht nur in HydrologySystemGenerator._execute_generation(): der
    # CalculatorDispatcher leitet die Ausfuehrungsreihenfolge AUSSCHLIESSLICH
    # aus depends_on ab und startet alle in derselben Runde bereiten Knoten
    # parallel als eigene Threads. Ohne die lake_detection -> thermal_erosion-
    # Kante wurden lake_detection und erosion_sedimentation gleichzeitig
    # bereit - Seen wurden dann je nach Thread-Timing auf dem NICHT erodierten
    # Gelaende gesucht und flow_network simulierte auf einer Heightmap, die
    # die Erosion dieser Runde mal enthielt und mal nicht (nicht
    # reproduzierbare Ergebnisse zwischen zwei identischen Laeufen).
    #
    # Droplet-Erosion-Umbau 2026-07-25 (siehe core/water_generator.py
    # DropletErosionSystem): vollstaendig entkoppelt von water.flow_network -
    # Partikel spawnen hoehen-gewichtet direkt aus der Heightmap, brauchen
    # weder simulierten Abfluss noch terrain.slope (der Droplet-Gradient
    # kommt direkt aus bilinearer Heightmap-Abtastung). Dieser Knoten haengt
    # deshalb bewusst NICHT von weather.precipitation ab - Erosion modelliert
    # geologische Zeit und ist vom heutigen Wetter unabhaengig. Die
    # Reihenfolge-Regression in smoke_test_water_pipeline_order.py sichert
    # beide Eigenschaften (Reihenfolge UND Regen-Entkopplung) strukturell ab.
    # === ALTBESTAND DROPLET-EROSION (stillgelegt 2026-07-28) ===
    # Ersetzt durch den Knoten erosion.hydraulic oben (Feldverfahren in
    # core/erosion_generator.py). Kann samt DropletErosionSystem und den
    # zugehoerigen _calc_*-Methoden in core/water_generator.py vollstaendig
    # geloescht werden, sobald das Feldmodell freigegeben ist.
    # CalculatorSpec("water.erosion_sedimentation", "water",
    #                ["terrain.redistribution", "geology.hardness"],
    #                ["erosion_map", "sedimentation_map"]),
    # Böschungswinkel-Erosion ("Phase 6", siehe core/water_generator.py
    # ThermalErosionSystem) - läuft NACH water.erosion_sedimentation (Fluss-
    # Erosion schneidet zuerst das V-Kerbtal, Thermal Erosion kollabiert/
    # verbreitert es danach je nach Härte, siehe ThermalErosionSystem-
    # Docstring). Eigener Knoten statt in water.erosion_sedimentation
    # eingefaltet (Nutzer-Entscheidung 2026-07-25) - separat einstellbar/
    # anzeigbar. Letzter Knoten, der die Heightmap veraendert: ab hier ist das
    # Gelaende dieser LOD-Runde final.
    # === ALTBESTAND (stillgelegt 2026-07-28) ===
    # Die Boeschungswinkel-Erosion ist jetzt ein Pass INNERHALB von
    # erosion.hydraulic (dort umschaltbar zwischen dem hier verwendeten
    # Gather-Verfahren und der Flux-Variante des Vorbilds). ThermalErosionSystem
    # selbst bleibt in Gebrauch - der Erosion-Generator benutzt seine
    # Konstanten und seine Haerte-Winkel-Beziehung.
    # CalculatorSpec("water.thermal_erosion", "water",
    #                ["water.erosion_sedimentation", "geology.hardness"],
    #                ["thermal_erosion_map", "thermal_deposition_map"]),
    # Pipe-Modell-Umbau 2026-07-25 (D8-Steilster-Abstieg -> virtuelle-Rohre-
    # Hydraulik, siehe core/water_generator.py PipeFlowSimulator): lake_detection
    # liefert kein full_basin_map mehr (die ungefilterte Wasserscheiden-
    # Zuordnung wurde nur für die jetzt entfallene Wasserscheiden-Umleitung
    # gebraucht - Senken füllen/laufen im Pipe-Modell strukturell von selbst
    # über). water.steepest_descent (D8-Fließrichtung) entfällt komplett -
    # es gibt kein "die eine Richtung" mehr unter kontinuierlichem Fluss.
    # erosion.hydraulic als Abhaengigkeit: Senken werden auf dem fertig
    # erodierten Gelaende gesucht (siehe Reihenfolge-Block oben). Diese Kante
    # ersetzt die frueher hier stehende auf water.thermal_erosion und ist der
    # Grund, warum der GESAMTE Water-Block das erodierte Gelaende sieht - die
    # uebrigen Water-Knoten erben sie transitiv.
    CalculatorSpec("water.lake_detection", "water",
                   ["terrain.redistribution", "erosion.hydraulic"],
                   ["lake_map"]),
    # water_depth/velocity_x/velocity_y kommen jetzt DIREKT aus der
    # Pipe-Simulation (echte simulierte Werte, siehe PipeFlowSimulator) -
    # nicht mehr aus water.manning_flow (dessen Kanalgeometrie-Suche würde
    # sonst eine zweite, abweichende "Wahrheit" für dieselbe physikalische
    # Größe liefern). Erbt die Erosions-Reihenfolge transitiv ueber
    # water.lake_detection. weather.precipitation ist hier - und NUR hier -
    # die Wasserquelle: der Kreislauf laeuft mit dem echten Regen.
    # water_biomes_map ist auf dieser Stufe die ZENTRALLINIE (ein Pixel
    # breit, rein akkumulationsbasiert klassifiziert); die raeumlich
    # ausgedehnte, gemalte Fassung liefert water.manning_flow.
    # weather.temperature/wind/humidity treiben die potentielle Verdunstung -
    # die zweite Senke des Modells neben dem Randabfluss (siehe
    # core/water_generator.py EvaporationCalculator.
    # calculate_potential_evaporation()). Sie haengt bewusst NUR von
    # Wetterdaten ab und nicht von der Wasser-Klassifikation, sonst entstuende
    # ein Zyklus mit water.evaporation. Ohne diese Senke konnte Wasser in
    # abflusslosen Becken nur steigen.
    CalculatorSpec("water.flow_network", "water",
                   ["terrain.redistribution", "weather.precipitation", "weather.temperature",
                    "weather.wind", "weather.humidity", "water.lake_detection"],
                   ["flow_accumulation", "water_biomes_map", "water_depth", "velocity_x", "velocity_y",
                    "ocean_outflow", "evaporated_volume", "depth_state", "flux_state"]),
    # Geschrumpft auf reine Fluss-Breiten-Malerei (calculate_channel_width/
    # paint_channel_width) - die frühere unabhängige Manning-Kanalgeometrie-
    # Suche für flow_speed/water_depth ist gelöscht (redundant seit beide
    # Größen direkt aus water.flow_network kommen, siehe
    # ManningFlowCalculator-Docstring). cross_section wird jetzt aus der
    # Kontinuitätsgleichung der simulierten Werte abgeleitet, braucht daher
    # kein terrain.slope mehr als eigene Kante.
    #
    # water_biomes_map ist ein EIGENER Output dieses Knotens (die gemalte
    # Fassung), kein Ueberschreiben von water.flow_networks gleichnamigem
    # Output mehr: get_calculator_output() ist auf (LOD, calculator_id, key)
    # geschluesselt, beide Fassungen existieren also unabhaengig
    # nebeneinander. Vorher schrieb dieser Knoten in den Output-Slot eines
    # FREMDEN Knotens - da water.evaporation/biome.super_override/
    # settlement.suitability nur von water.flow_network abhingen, konnten sie
    # in derselben Runde parallel laufen und je nach Thread-Timing die
    # Zentrallinie ODER die gemalte Fassung lesen. Alle Konsumenten der
    # FINALEN Wasser-Klassifikation haengen jetzt explizit von diesem Knoten
    # ab; wer bewusst die Zentrallinie braucht (Boden-Feuchte-Quellflaeche),
    # liest weiterhin water.flow_network.
    CalculatorSpec("water.manning_flow", "water",
                   ["water.flow_network", "terrain.redistribution"],
                   ["cross_section", "channel_width", "water_biomes_map"]),
    # weather.temperature zusaetzlich zur Wasser-Klassifikation: treibt den
    # biom-abhaengigen Trocknungs-Term (siehe Biome-Preseed-Plan Punkt C) -
    # weather.temperature haengt selbst nur von terrain.* ab, kein Zyklus.
    # Haengt von BEIDEN Wasser-Knoten ab: water.flow_network fuer die
    # Zentrallinie (100%-Feuchte-Quellflaeche, entkoppelt die Feuchte-
    # Ausdehnung von der visuellen Flussbreite) und water.manning_flow fuer
    # die gemalte Klassifikation.
    # terrain.shadow zusaetzlich (2026-07-27): die Besonnung steuert, wie
    # stark ein Hang austrocknet - sonnenabgewandte Haenge und von
    # Nachbarbergen verschattete Lagen bleiben feuchter (Nutzer-Vorgabe:
    # "die ebenen, nordhänge (bereiche wo wenig sonne hinkommt) sind eher
    # feucht"). Die shadowmap enthaelt die tatsaechliche Beleuchtung inklusive
    # Verschattung durch Nachbarberge und ist damit die bessere Quelle als
    # eine reine Hangausrichtung aus der slopemap.
    CalculatorSpec("water.soil_moisture", "water",
                   ["water.manning_flow", "water.flow_network", "weather.temperature",
                    "terrain.shadow"],
                   ["soil_moist_map"]),
    CalculatorSpec("water.evaporation", "water",
                   ["weather.temperature", "weather.wind", "weather.humidity", "water.manning_flow"],
                   ["evaporation_map"]),

    # --- Biome (#23-#27, + preseed_hint neu) ---
    # Billiger Vorab-Biome-Schaetzwert NUR aus Slope+Breitengrad (kein
    # Gauss-Fitness), haengt bewusst NUR von terrain.* ab (nicht von water.*/
    # biome.base_classification) - loest das Henne-Ei-Problem "echte Biome-
    # Klassifikation braucht water.soil_moisture, water.soil_moisture
    # bräuchte fuer eine biom-abhaengige Kapazitaet wiederum den Biome-Typ"
    # fuer die allererste LOD-Runde (siehe Biome-Preseed-Plan Punkt B). Ab
    # LOD 2 nutzt water.soil_moisture stattdessen die ECHTE biome_map der
    # Vorstufe (biome.integrate_layers bei lod_level-1).
    CalculatorSpec("biome.preseed_hint", "biome",
                   ["terrain.redistribution", "erosion.slope"], ["preseed_biome_map"]),
    CalculatorSpec("biome.base_classification", "biome",
                   ["terrain.redistribution", "weather.temperature", "weather.precipitation",
                    "water.soil_moisture"], ["base_biome_map"]),
    # water.manning_flow statt water.flow_network: Biome brauchen die FINALE,
    # gemalte Wasser-Klassifikation (Fluss in voller Breite), nicht die
    # ein Pixel breite Zentrallinie - siehe water.manning_flow-Kommentar oben.
    CalculatorSpec("biome.super_override", "biome",
                   ["terrain.redistribution", "weather.temperature", "water.manning_flow",
                    "water.soil_moisture"], ["super_biome_mask", "super_biome_probabilities"]),
    CalculatorSpec("biome.integrate_layers", "biome",
                   ["biome.base_classification", "biome.super_override"], ["biome_map"]),
    CalculatorSpec("biome.supersampling", "biome", ["biome.integrate_layers"], ["biome_map_super"]),
    CalculatorSpec("biome.climate_classification", "biome",
                   ["weather.temperature", "weather.precipitation"], ["climate_classification"]),

    # --- Settlement (#28-#34) ---
    # water.manning_flow statt water.flow_network: Siedlungseignung bewertet
    # die Naehe zu tatsaechlichen Gewaesserflaechen, also die FINALE gemalte
    # Klassifikation - siehe water.manning_flow-Kommentar oben.
    CalculatorSpec("settlement.suitability", "settlement",
                   ["terrain.redistribution", "erosion.slope", "water.manning_flow"],
                   ["combined_suitability_map"]),
    CalculatorSpec("settlement.settlements", "settlement",
                   ["settlement.suitability", "terrain.redistribution"], ["settlement_list"]),
    CalculatorSpec("settlement.city_boundary", "settlement",
                   ["settlement.settlements", "terrain.redistribution", "erosion.slope"],
                   ["city_mask", "city_cost_map"]),
    # settlement.city_blocks/settlement.landscape_voronoi (CityBlockSystem/
    # LandscapeVoronoiSystem) entfernt - vollständig durch settlement.plot_nodes
    # (PlotPhysicsSystem) ersetzt, siehe [[project-settlement-plot-physics-rebuild]].
    CalculatorSpec("settlement.pathfinding", "settlement",
                   # biome.integrate_layers ergaenzt 2026-07-28: der Pfadfinder
                   # liest biome_map fuer seine MoveCost-Berechnung
                   # (core/settlement_generator.py:3278), hat das aber nie
                   # deklariert. Ohne die Kante durfte Settlement in derselben
                   # Runde wie Biome laufen und je nach Thread-Timing die
                   # hoehenbasierte Notfall-Ersatzkarte
                   # (_create_fallback_biome_map) statt der echten Biome
                   # benutzen - reproduzierbar war das Ergebnis so nicht.
                   #
                   # Aufgefallen ist es erst, als der Dependency-Tree des
                   # Orchestrators aus diesem Graph ABGELEITET wurde statt von
                   # Hand gepflegt: die Handtabelle fuehrte biome bei
                   # settlement, der Graph nicht. Die Handtabelle hatte recht.
                   #
                   # settlement.city_boundary ergaenzt (Ticket #73, Anschluss-
                   # punkte der Wege an der Stadtgrenze, docs/SIEDLUNGEN_ENTWURF.md
                   # §6.1): _calc_pathfinding schneidet die geroutete `roads`-
                   # Liste gegen city_mask, braucht city_mask also als
                   # deklarierte Eingabe statt sie ungefragt vom Data-LOD-
                   # Manager zu holen - siehe die Lehre zu biome_map oben, die
                   # sich hier fast wortgleich wiederholt haette.
                   ["settlement.settlements", "settlement.city_boundary", "erosion.slope",
                    "biome.integrate_layers"], ["roads", "sea_roads", "road_entry_points"]),
    # settlement.outer_roads ENTFERNT (2026-08-10, OFFENE_PUNKTE 5.11): verband
    # Siedlungen mit dem KARTENRAND - eine Insel/Region hat kein sinnvolles
    # "Draussen". docs/SIEDLUNGEN_ENTWURF.md kennt diese Anbindung nicht.
    CalculatorSpec("settlement.roadsites", "settlement", ["settlement.pathfinding"], ["roadsite_list"]),
    CalculatorSpec("settlement.civ_influence", "settlement",
                   ["terrain.redistribution", "erosion.slope", "settlement.settlements",
                    "settlement.pathfinding", "settlement.roadsites"], ["civ_map"]),
    CalculatorSpec("settlement.landmarks", "settlement",
                   ["settlement.civ_influence", "terrain.redistribution", "erosion.slope"], ["landmark_list"]),
    CalculatorSpec("settlement.landmark_roads", "settlement",
                   ["settlement.landmarks", "settlement.pathfinding", "erosion.slope"], ["landmark_roads"]),
    # settlement.plot_nodes (PlotPhysicsSystem, siehe [[project-settlement-plot-physics-rebuild]]) -
    # läuft NUR am finalen LOD (Guard innerhalb von _calc_plot_nodes selbst, kein
    # Dispatcher-Feature dafür vorhanden) - braucht city_mask (settlement.city_boundary,
    # für die Stadt/Wildnis-Unterscheidung) statt biome_map/pathfinding wie zuvor.
    CalculatorSpec("settlement.plot_nodes", "settlement",
                   ["settlement.civ_influence", "settlement.settlements", "settlement.city_boundary",
                    "terrain.redistribution"],
                   ["plot_nodes", "plots", "plot_map", "plot_edges", "plot_node_positions"]),
]

CALCULATOR_GRAPH: Dict[str, CalculatorSpec] = {spec.calculator_id: spec for spec in _CALCULATOR_SPECS}

# Tatsächliche Zählung: 34 durchnummerierte Schritte (#1-#34) aus
# docs/generation_pipeline_dependencies.md, minus dem bekannt kaputten #22
# (water.erosion_feedback, oben ausgeschlossen), plus geology.faceted_boundaries
# (in den Docs nicht erfasst, aber real im Code vorhanden - siehe core/geology_generator.py),
# plus 5 neue Settlement-Knoten (city_boundary, city_blocks, landscape_voronoi,
# outer_roads, landmark_roads) aus dem Settlement-Rework (siehe docs/backlog.md
# Ticket #4) = 39 aktive Knoten. Davon 2 (city_blocks, landscape_voronoi) im
# Zuge von [[project-settlement-plot-physics-rebuild]] wieder entfernt (durch
# settlement.plot_nodes/PlotPhysicsSystem vollständig ersetzt) = 37 aktive Knoten.
# Geology-3D-Gesteinsstapel-Rework (siehe core/geology_generator.py) ersetzte die
# 7 alten Geology-Knoten (classify_elevation...hardness) durch 8 neue
# (layer_thickness...hardness) = netto +1 -> 38 aktive Knoten. Biome-Preseed-
# Plan fuegte biome.preseed_hint hinzu = netto +1 -> 39 aktive Knoten.
# Pipe-Modell-Umbau 2026-07-25 entfernte water.steepest_descent (D8-
# Fliessrichtung, siehe core/water_generator.py PipeFlowSimulator) = netto -1
# -> 38 aktive Knoten. Derselbe Umbau fuegte water.thermal_erosion hinzu
# (Boeschungswinkel-Erosion, "Phase 6") = netto +1 -> 39 aktive Knoten.
# Erosion-Umbau 2026-07-28 (Partikel- -> Feldverfahren, eigener Generator,
# siehe core/erosion_generator.py): water.erosion_sedimentation und
# water.thermal_erosion stillgelegt, dafuer erosion.hydraulic neu = netto -1
# -> 38 aktive Knoten. Am selben Tag kam erosion.slope dazu (Hangneigung NACH
# der Erosion, siehe dortiger Kommentar - ein ZWEITER Slope-Knoten, weil ein
# verschobener terrain.slope einen Zyklus ueber geology.layer_thickness
# ergaebe) = netto +1 -> 39 aktive Knoten. Wegenetz-Umbau 2026-08-10
# (docs/SIEDLUNGEN_ENTWURF.md, OFFENE_PUNKTE 5.11) entfernte
# settlement.outer_roads (Anbindung an den Kartenrand - eine Insel/Region hat
# kein sinnvolles "Draussen") = netto -1 -> 38 aktive Knoten.
assert len(CALCULATOR_GRAPH) == 38, f"Erwartet 38 aktive Calculators, gefunden {len(CALCULATOR_GRAPH)}"


class CalculatorRoundScheduler:
    """
    Führt eine Teilmenge von CALCULATOR_GRAPH-Knoten rundenweise aus - pro Runde
    genau ein LOD-Level, in Abhängigkeits-Reihenfolge. Abhängigkeiten AUSSERHALB
    der verwalteten Teilmenge (z.B. geology.classify_elevation haengt von
    terrain.redistribution ab, aber dieser Scheduler verwaltet evtl. nur die
    Geology-Knoten) gelten als bereits erfüllt - der Aufrufer ist dafür
    verantwortlich, den Executor erst zu starten, wenn diese externen
    Abhängigkeiten tatsächlich für das gewünschte LOD vorliegen (das übernimmt
    heute weiterhin generation_orchestrator.py's bestehende Generator-Dispatch-
    Logik, solange nicht alle 6 Generatoren zerlegt sind).
    """

    def __init__(self, calculator_ids: List[str], executors: Dict[str, Callable[[dict], None]]):
        unknown = [cid for cid in calculator_ids if cid not in CALCULATOR_GRAPH]
        if unknown:
            raise ValueError(f"Unbekannte Calculator-IDs: {unknown}")
        missing_executors = [cid for cid in calculator_ids if cid not in executors]
        if missing_executors:
            raise ValueError(f"Fehlende Executor-Funktionen für: {missing_executors}")

        self.calculator_ids = list(calculator_ids)
        self.executors = executors
        self.completed_lod: Dict[str, int] = {cid: 0 for cid in calculator_ids}

    def run_round(self, target_lod: int, context: dict) -> dict:
        """
        Führt alle verwalteten Calculators für GENAU ein LOD-Level aus.
        context: gemeinsames Dict, aus dem Executor-Funktionen ihre Eingaben lesen
        und in das sie ihre Outputs schreiben (Data-Key -> Wert). Jeder Executor
        bekommt exakt dieses eine context-Dict übergeben.
        Returns: aktualisiertes context-Dict.
        """
        remaining = set(self.calculator_ids)
        while remaining:
            ready = [
                cid for cid in remaining
                if all(
                    dep not in self.calculator_ids or self.completed_lod[dep] >= target_lod
                    for dep in CALCULATOR_GRAPH[cid].depends_on
                )
            ]
            if not ready:
                raise RuntimeError(
                    f"Zirkuläre oder von außerhalb dieser Scheduler-Instanz "
                    f"unerfüllbare Abhängigkeit unter verbleibenden Knoten: {remaining}")

            for cid in ready:
                self.executors[cid](context)
                self.completed_lod[cid] = target_lod
                remaining.remove(cid)

        return context


# EIN-RUNDEN-BETRIEB (2026-07-28, Stufe 2 der LOD-Aufloesung)
#
# True  = die Pipeline laeuft GENAU EINMAL, direkt in der Zielaufloesung.
# False = das fruehere Verhalten, LOD-Leiter von 32 px in Verdopplungen hoch.
#
# Warum: die Leiter war dafuer da, nach wenigen Sekunden eine grobe Vorschau zu
# liefern. Diesen Gegenwert hat sie verloren - die Erosion rechnet ohnehin nur
# noch in der letzten Runde (die groben LOD-1-Strukturen liessen sich einbacken
# und durch keine Verfeinerung mehr entfernen), Settlement ebenso. Gleichzeitig
# geht ein grosser Teil der behobenen Fehler auf sie zurueck: Erosion blieb auf
# LOD 3 stehen waehrend alles andere LOD 5 anstrebte (fuenf gleichzeitige
# Timeouts ohne Fehlermeldung), die zurueckgenommene LOD-Eviction, Regen pro
# Schritt statt pro Sekunde, Konvergenz pro Schritt statt pro Sekunde,
# Partikeldichte je LOD.
#
# WAS DAS AENDERT: acht Stellen im Code lesen `lod_level - 1`, also das
# Ergebnis der VORIGEN Runde - die Leiter trug damit echte Rueckkopplungen:
#
#   weather_generator.py:768       Bodenfeuchte -> Verdunstung zurueck ins Wetter
#   weather_generator.py:750/752/754, 810, 934   eigene Monatsschichten/Feuchte
#   water_generator.py:3442        die ECHTE biome_map fuer die Austrocknung
#   water_generator.py:3145/3148   Pipe-Zustand als Warmstart
#
# Der Speicher sucht ABWAERTS (get_calculator_output: range(lod, 0, -1)), diese
# Stellen bekommen also sauber None statt versehentlich die eigene Runde. Jede
# hat einen dokumentierten Fallback, weil LOD 1 noch nie eine Vorstufe hatte -
# der Ein-Runden-Betrieb ist damit exakt der LOD-1-Codepfad in voller
# Aufloesung. Wetter startet aus Rauschen statt aus geerbten Schichten, die
# Bodenfeuchte-Kopplung faellt auf ihren 50%-Platzhalter, Water startet mit
# trockener Karte.
#
# Diese Rueckkopplungen kommen in Stufe 3 als EXPLIZITE Schleife zurueck. Ihre
# Zahl haengt dann an einem Regler statt zufaellig an der Zahl der
# Aufloesungsstufen.
#
# Die Konstante bleibt als Schalter stehen: sie ist die Gegenprobe, mit der
# sich beide Verhalten direkt vergleichen lassen.
SINGLE_ROUND_PIPELINE = True


# RUECKKOPPLUNGS-DURCHGAENGE (2026-07-28, Stufe 3 der LOD-Aufloesung)
#
# Weather, Water und Biome bilden einen echten KREIS im Modell:
#
#   Weather -> Water   (Regen speist den Kreislauf)
#   Water   -> Biome   (Bodenfeuchte bestimmt den Biomtyp)
#   Biome   -> Water   (der Biomtyp steuert die Austrocknung)
#   Water   -> Weather (Bodenfeuchte verdunstet zurueck in die Luft)
#
# Eine lineare Reihenfolge kann das nicht ausdruecken. Die LOD-Leiter hat den
# Kreis bisher aufgebrochen, indem jede Runde die Werte der vorigen benutzte -
# die Zahl der Durchgaenge hing damit zufaellig an der Zahl der
# Aufloesungsstufen. Hier ist sie eine Zahl.
#
# DEFAULT 1, ALSO KEINE RUECKKOPPLUNG - und das ist eine Messung, keine
# Bequemlichkeit.
#
# GEMESSEN (128 px, Default-Parameter, Mittelwerte der ganzen Karte):
#
#     Passes   temp   precip   soil    Aenderung zum vorigen Lauf
#        1     4.421  14.342   8.091   -
#        2     0.712   2.313   2.811   temp  84%  precip  84%  soil 110%
#        3     0.947   0.548   1.966   temp  62%  precip  81%  soil 119%
#        4     0.639   0.386   1.780   temp  21%  precip  59%  soil 137%
#        5     0.590   0.336   1.396   temp   8%  precip  52%  soil  51%
#
# Die Schleife KONVERGIERT NICHT. Sie trocknet monoton aus, und die Aenderung
# je Durchgang liegt beim fuenften noch ueber 50%. Der Kreis ist als
# MITKOPPLUNG verdrahtet, ohne rueckstellenden Term:
#
#     weniger Bodenfeuchte -> weniger Verdunstung -> weniger Luftfeuchte
#     -> weniger Niederschlag -> weniger Bodenfeuchte
#
# Solange das so ist, waere jede Zahl groesser 1 willkuerlich: das Ergebnis
# haengt dann daran, wie oft man gedreht hat, nicht am Modell. Ein Default von
# 3 haette genau das ausgeliefert.
#
# Mit 1 ist der Zustand wohldefiniert und stabil: Wetter kennt die Bodenfeuchte
# nicht, Water benutzt die Pre-Biome-Karte - dieselben Platzhalter, die frueher
# in LOD 1 galten.
#
# WAS OFFEN BLEIBT: der Kreis braucht einen daempfenden Term (z.B. Verdunstung
# aus offenem Wasser und Ozean-Zufuhr als feuchte Quelle, die nicht von der
# Bodenfeuchte abhaengt), bevor man ihn schliessen kann. Der Mechanismus hier
# ist fertig und getestet - er wartet nur auf ein Modell, das ihn vertraegt.
#
# EIGENER MESSFEHLER, festgehalten damit er nicht wiederholt wird: eine erste
# Konvergenzmessung schien sauber einzuschwingen (temp 4.42 -> 2.16 -> 2.21 ->
# 2.20). Sie lief aber, BEVOR die Lesestellen von "voriges LOD" auf "voriger
# Durchgang" umgestellt waren - gemessen wurden also Wiederholungen OHNE
# Rueckkopplung. Eine Messung am halb umgebauten Stand ist keine Messung.
FEEDBACK_PASSES = 1

# Die Generatoren, die den Kreis bilden. Alles, was NUR von ihnen abhaengt
# (heute Settlement), laeuft erst nach dem letzten Durchgang - siehe
# CalculatorDispatcher._generators_after_feedback().
FEEDBACK_GENERATORS = ("weather", "water", "biome")


class CalculatorDispatcher:
    """
    Globaler Runden-Scheduler über den KOMPLETTEN CALCULATOR_GRAPH (alle 6
    Generatoren gemeinsam) - löst den bisherigen 6-Knoten dependency_tree in
    GenerationOrchestrator ab (Tracker #16 / LOD-Lockstep-Umbau).

    Kernidee: "Runde N" heißt nicht "alle Knoten werden gleichzeitig auf LOD N
    gebracht", sondern "alle Knoten, die für LOD N bereit sind/werden, laufen -
    in Abhängigkeits-Kaskade - bis nichts mehr für LOD N bereit wird, DANN erst
    beginnt Runde N+1". Ein Knoten wie geology.classify_elevation kann also
    innerhalb derselben Runde laufen wie terrain.redistribution, sobald diese
    fertig ist (kein künstliches Warten auf einen globalen Rundenabschluss) -
    entscheidend ist nur, dass KEIN Knoten LOD N+1 erreicht, bevor nicht jeder
    für LOD N erreichbare Knoten sein LOD N abgeschlossen hat.

    Nutzt bewusst NUR die primitiven Bausteine (get_ready_nodes/mark_completed),
    nicht einen synchronen "alles durchlaufen"-Loop als einzige Schnittstelle -
    so kann die Anbindung in GenerationOrchestrator (Task 18) jeden bereiten
    Knoten als eigenen asynchronen Thread dispatchen und mark_completed() erst
    aus dessen Qt-Completion-Signal aufrufen, statt blockierend zu warten.
    run_all_rounds() ist eine synchrone Convenience-Variante für Tests und
    einfache Nicht-GUI-Nutzung.
    """

    def __init__(self, executors: Dict[str, Callable[[str, int], None]]):
        """
        executors: calculator_id -> callable(calculator_id, lod_level) -> None.
        Jeder Executor ist dafür verantwortlich, seinen Output selbst über
        DataLODManager.set_calculator_output() zu persistieren (siehe
        core/*_generator.py _calc_*-Methoden nach der Umstellung in den
        Tasks 12-17).
        """
        missing = [cid for cid in CALCULATOR_GRAPH if cid not in executors]
        if missing:
            raise ValueError(f"Fehlende Executor-Funktionen für: {missing}")

        self.executors = executors
        self.completed_lod: Dict[str, int] = {cid: 0 for cid in CALCULATOR_GRAPH}
        self.target_lod: Dict[str, int] = {cid: 0 for cid in CALCULATOR_GRAPH}  # 0 = nicht angefragt
        self._next_round = 1  # Fortsetzungspunkt für wiederholte run_all_rounds()-Aufrufe
        # Laufender Rueckkopplungs-Durchgang (siehe FEEDBACK_PASSES).
        self._feedback_pass = 1
        # Einmal ableiten statt bei jeder Bereitschaftspruefung neu.
        self._after_feedback = set(self._generators_after_feedback())

    def request(self, generator: str, target_lod: int):
        """
        Setzt das Ziel-LOD für alle Calculator-Knoten EINES Generators (z.B. wenn
        ein Tab "Generieren" klickt oder Auto-Start beim App-Start alle 6 anfragt).
        Ein Knoten mit target_lod=0 gilt als nicht angefragt und wird vom
        Scheduler übersprungen (bleibt ewig "nicht bereit" für abhängige Knoten).
        """
        for cid, spec in CALCULATOR_GRAPH.items():
            if spec.generator == generator:
                self.target_lod[cid] = max(self.target_lod[cid], target_lod)

    def get_ready_nodes(self, round_n: int) -> List[str]:
        """
        Alle Knoten, die JETZT für Runde round_n ausgeführt werden können:
        eigenes target_lod >= round_n, noch nicht auf round_n abgeschlossen, und
        alle Abhängigkeiten haben round_n bereits erreicht. Ein Knoten, dessen
        Generator nie angefragt wurde (target_lod=0 bei einer Abhängigkeit),
        wird nie bereit - das ist beabsichtigt (kein Auto-Request von Upstream-
        Generatoren hier, das entscheidet der Aufrufer/Auto-Start explizit).
        """
        ready = []
        for cid, spec in CALCULATOR_GRAPH.items():
            if self.target_lod[cid] < round_n:
                continue
            if self._is_held_back(cid):
                continue
            if self.completed_lod[cid] >= round_n:
                continue
            if all(self.completed_lod[dep] >= round_n for dep in spec.depends_on):
                ready.append(cid)
        return ready

    def mark_completed(self, calculator_id: str, round_n: int):
        """Markiert einen Knoten als für round_n abgeschlossen (nach Thread-Completion)."""
        self.completed_lod[calculator_id] = max(self.completed_lod[calculator_id], round_n)

    def reset_completed(self, calculator_id: str):
        """
        Setzt einen Knoten auf completed_lod=0 zurück (Parameter-Änderung/
        Invalidierung, siehe GenerationOrchestrator.reset_lod_status()). MUSS
        über diese Methode laufen statt completed_lod[...] direkt zu schreiben:
        get_next_ready_batch() lässt _next_round unbegrenzt weiterlaufen, sobald
        einmal is_fully_done() erreicht wurde (_next_round steht dann auf
        höchstem target_lod + 1). Ein Knoten, dessen completed_lod danach ohne
        Rewind auf 0 zurückfällt, hat für IMMER target_lod < round_n - er wird
        von der "bereits erledigt"-Prüfung nie wieder erfasst, obwohl
        is_fully_done() ihn weiterhin als offen zählt: get_next_ready_batch()
        läuft dann in eine echte Endlosschleife auf dem aufrufenden (GUI-)Thread.
        Das rundenweise Vorlaufen weiter unten holt den zurückgesetzten Knoten
        günstig wieder ein, da bereits abgeschlossene Runden anderer Knoten
        sofort übersprungen werden (completed_lod >= round_n).
        """
        self.completed_lod[calculator_id] = 0
        self._next_round = 1

    @staticmethod
    def _generators_after_feedback() -> List[str]:
        """
        Generatoren, die (transitiv) auf dem Rueckkopplungs-Block aufbauen,
        aber nicht selbst dazugehoeren - heute nur Settlement.

        ABGELEITET aus CALCULATOR_GRAPH statt als Liste gepflegt: derselbe
        Fehlertyp wie beim Auto-Start, beim Dependency-Tree und bei den
        Generator-Parametern hat in diesem Projekt schon vier Mal zugeschlagen.
        Sie duerfen erst NACH dem letzten Durchgang rechnen, sonst treffen sie
        ihre Entscheidungen auf einem Zwischenstand, der sich danach noch
        aendert.
        """
        upstream = {}
        for cid, spec in CALCULATOR_GRAPH.items():
            upstream.setdefault(spec.generator, set()).update(
                CALCULATOR_GRAPH[dep].generator
                for dep in spec.depends_on if dep in CALCULATOR_GRAPH)

        def haengt_am_kreis(generator, gesehen=None):
            gesehen = gesehen if gesehen is not None else set()
            for oben in upstream.get(generator, ()):
                if oben in FEEDBACK_GENERATORS:
                    return True
                if oben not in gesehen:
                    gesehen.add(oben)
                    if haengt_am_kreis(oben, gesehen):
                        return True
            return False

        return sorted(generator for generator in upstream
                      if generator not in FEEDBACK_GENERATORS
                      and haengt_am_kreis(generator))

    def _is_held_back(self, calculator_id: str) -> bool:
        """
        Wartet dieser Knoten auf den letzten Rueckkopplungs-Durchgang?

        Settlement (und alles Kuenftige hinter dem Kreis) soll GENAU EINMAL
        laufen, und zwar auf dem eingeschwungenen Stand. Ohne diese Bremse
        rechnet es im ersten Durchgang auf Zwischenwerten, die sich danach noch
        aendern - gemessen lief es zweimal statt einmal, das erste Mal
        vollstaendig umsonst.
        """
        if self._feedback_pass >= FEEDBACK_PASSES:
            return False
        return CALCULATOR_GRAPH[calculator_id].generator in self._after_feedback

    def _nodes_of(self, generators) -> List[str]:
        return [cid for cid, spec in CALCULATOR_GRAPH.items()
                if spec.generator in generators]

    def _begin_next_feedback_pass(self) -> bool:
        """
        Startet den naechsten Rueckkopplungs-Durchgang, falls noch einer
        aussteht. Return: True, wenn etwas zurueckgesetzt wurde.

        Die Knoten des Kreises werden auf "nicht gerechnet" zurueckgesetzt und
        laufen erneut - diesmal lesen sie im Speicher die Werte des vorigen
        Durchgangs vor, statt auf ihre Anfangs-Platzhalter zu fallen. Genau das
        tat die LOD-Leiter nebenbei, nur ohne dass jemand die Zahl der
        Durchgaenge bestimmen konnte.

        Die nachgelagerten Generatoren werden ZUSAMMEN mit dem letzten
        Durchgang zurueckgesetzt: sie haengen an Knoten des Kreises und koennen
        deshalb ohnehin erst starten, wenn der fertig ist.
        """
        if self._feedback_pass >= FEEDBACK_PASSES:
            return False

        self._feedback_pass += 1
        neu_zu_rechnen = list(FEEDBACK_GENERATORS)
        if self._feedback_pass == FEEDBACK_PASSES:
            neu_zu_rechnen += self._generators_after_feedback()

        for cid in self._nodes_of(neu_zu_rechnen):
            if self.target_lod[cid] > 0:
                self.completed_lod[cid] = 0
        self._next_round = 1
        return True

    def _start_round(self) -> int:
        """
        Die erste Runde, die ueberhaupt gerechnet wird.

        Im Ein-Runden-Betrieb ist das das hoechste angefragte Ziel-LOD - alle
        groeberen Stufen entfallen. Ohne ihn bleibt es bei 1, also der
        vollstaendigen Leiter.

        Bewusst das MAXIMUM ueber alle angefragten Knoten und nicht je Knoten
        sein eigenes Ziel: get_ready_nodes() verlangt, dass jede Abhaengigkeit
        dieselbe Runde erreicht hat. Ein Knoten mit niedrigerem Ziel wuerde
        seine Abnehmer sonst dauerhaft blockieren. In der Praxis fragt der
        Auto-Start ohnehin alle Generatoren mit demselben Ziel an, und
        request() kann Ziele nur anheben, nie senken.
        """
        if not SINGLE_ROUND_PIPELINE:
            return 1
        requested = [lod for lod in self.target_lod.values() if lod > 0]
        return max(requested) if requested else 1

    def get_pending_nodes(self) -> List[str]:
        """Alle angefragten, aber noch nicht auf ihr Ziel-LOD gebrachten Knoten."""
        return [
            cid for cid in CALCULATOR_GRAPH
            if self.target_lod[cid] > 0 and self.completed_lod[cid] < self.target_lod[cid]
        ]

    def is_fully_done(self) -> bool:
        """
        True, wenn jeder angefragte Knoten sein Ziel-LOD erreicht hat UND kein
        Rueckkopplungs-Durchgang mehr aussteht.

        Der zweite Teil ist nicht kosmetisch: jeder Aufrufer - der synchrone
        run_all_rounds() ebenso wie advance_calculator_dispatch() in der GUI -
        benutzt diese Methode als Abbruchbedingung. Meldete sie schon nach dem
        ersten Durchgang "fertig", fragte niemand mehr nach neuen Knoten, und
        die Schleife in get_next_ready_batch() kaeme nie zum Zug (gemessen:
        alle Generatoren liefen genau einmal, auch mit FEEDBACK_PASSES=3).
        """
        if self.get_pending_nodes():
            return False
        # Nur wenn ueberhaupt etwas angefragt wurde - sonst waere ein frisch
        # gebauter Dispatcher nie "fertig".
        if any(lod > 0 for lod in self.target_lod.values()):
            return self._feedback_pass >= FEEDBACK_PASSES
        return True

    @property
    def current_round(self) -> int:
        """Die Runde, die get_next_ready_batch() als nächstes zu vervollständigen versucht."""
        return self._next_round

    def get_next_ready_batch(self) -> List[str]:
        """
        Nicht-blockierende Kernmethode für asynchronen Dispatch (siehe
        GenerationOrchestrator.advance_calculator_dispatch(), Task 18): advanced
        self._next_round über bereits vollständig abgeschlossene Runden hinweg
        und gibt die aktuell bereiten Knoten für die (ggf. neu erreichte)
        nächste Runde zurück, OHNE sie selbst auszuführen - das übernimmt der
        Aufrufer (z.B. über eigene QThreads), der danach mark_completed()
        aufruft und diese Methode erneut abfragt (current_round liefert die
        Runde, für die die zurückgegebenen IDs bereit sind).

        Gibt eine leere Liste zurück, wenn entweder alles erledigt ist
        (is_fully_done() prüfen) oder aktuell nichts Neues bereit ist (z.B. weil
        alles gerade angeforderte bereits läuft/in einem Thread hängt, oder ein
        echtes Deadlock vorliegt, weil eine Abhängigkeit nie angefragt wurde).
        """
        # Im Ein-Runden-Betrieb (siehe SINGLE_ROUND_PIPELINE) werden die
        # Runden 1..Ziel-1 uebersprungen: gerechnet wird ausschliesslich in der
        # Zielaufloesung. Die Startrunde wird hier bestimmt und nicht im
        # Konstruktor, weil die Ziel-LODs erst durch request() feststehen.
        hoechstes_ziel = max(self.target_lod.values(), default=0)
        round_n = max(self._next_round, self._start_round())
        # MUSS zurueckgeschrieben werden: der Aufrufer liest current_round und
        # ruft damit mark_completed(cid, runde). Bliebe _next_round auf 1,
        # waehrend hier bereits Runde 5 verteilt wird, markierte er die falsche
        # Runde - die Knoten kaemen nie auf ihr Ziel-LOD und die Schleife
        # liefe endlos (gemessen: 200 Durchlaeufe ohne Fortschritt).
        self._next_round = round_n
        while True:
            # Zurueckgehaltene Knoten zaehlen hier NICHT als offen. Sonst
            # waere die Runde nie "vollstaendig erledigt", der naechste
            # Durchgang wuerde nie angestossen und die Pipeline bliebe stehen -
            # ein Deadlock ohne Fehlermeldung, also genau die Sorte, die dieses
            # Projekt schon zweimal getroffen hat.
            still_pending_for_round = [
                cid for cid in CALCULATOR_GRAPH
                if self.target_lod[cid] >= round_n
                and self.completed_lod[cid] < round_n
                and not self._is_held_back(cid)
            ]

            if not still_pending_for_round:
                # Runde round_n ist (soweit überhaupt angefragt) vollständig erledigt
                round_n += 1
                self._next_round = round_n

                # BEWUSST get_pending_nodes() und NICHT is_fully_done():
                # letzteres meldet erst "fertig", wenn auch alle
                # Rueckkopplungs-Durchgaenge durch sind - und genau die sollen
                # hier ja erst angestossen werden. Mit is_fully_done() an
                # dieser Stelle entsteht ein Zirkelschluss: der Zweig wird nie
                # betreten, der `continue` darunter zaehlt round_n endlos hoch,
                # und der Aufruf kehrt nicht mehr zurueck (beim Bauen prompt
                # passiert - der Prozess hing ohne jede Ausgabe).
                if not [cid for cid in self.get_pending_nodes()
                        if not self._is_held_back(cid)]:
                    # Alle Knoten am Ziel - steht noch ein Durchgang aus? Dann
                    # laeuft der Kreis Weather/Water/Biome erneut, diesmal mit
                    # den Werten des vorigen Durchgangs statt mit Platzhaltern.
                    if self._begin_next_feedback_pass():
                        round_n = max(self._next_round, self._start_round())
                        self._next_round = round_n
                        continue
                    return []

                # SCHRANKE. Oberhalb des hoechsten Ziel-LODs kann nie wieder
                # etwas offen sein - ohne diese Zeile zaehlt die Schleife
                # round_n endlos hoch, sobald die Abbruchbedingung darueber aus
                # irgendeinem Grund nicht greift. Genau das ist beim Bauen
                # dieser Rueckkopplung ZWEIMAL passiert (einmal ueber
                # is_fully_done(), einmal ueber zurueckgehaltene Knoten in
                # get_pending_nodes()) - beide Male hing der Prozess stumm.
                # Ein Aufruf, der nicht zurueckkehrt, ist schlimmer als ein
                # falsches Ergebnis: man sieht ihm nicht an, was fehlt.
                if round_n > hoechstes_ziel + 1:
                    raise RuntimeError(
                        "get_next_ready_batch kommt nicht voran: Runde {}, "
                        "Durchgang {}/{}, offen: {}".format(
                            round_n, self._feedback_pass, FEEDBACK_PASSES,
                            self.get_pending_nodes()[:5]))
                continue

            return self.get_ready_nodes(round_n)

    def run_all_rounds(self, max_rounds: int = 1000):
        """
        Synchrone Convenience-Variante: dispatcht Runde für Runde blockierend,
        bis kein Knoten mehr Fortschritt machen kann. Für Tests/einfache
        Nicht-GUI-Nutzung - die echte GUI-Anbindung (Task 18) nutzt
        get_next_ready_batch()/mark_completed() direkt für asynchronen
        Thread-Dispatch, ohne zu blockieren.

        Wiederholt aufrufbar (z.B. erst request("terrain", 1), run_all_rounds(),
        dann später request("terrain", 2), run_all_rounds() erneut, wenn neue
        Ziele nachträglich gesetzt werden) - merkt sich die zuletzt erreichte
        Runde in self._next_round, damit ein erneuter Aufruf nicht fälschlich
        sofort abbricht, nur weil Runde 1 schon vollständig erledigt war.
        """
        iterations = 0

        while iterations < max_rounds:
            iterations += 1

            ready = self.get_next_ready_batch()
            if not ready:
                break

            round_n = self._next_round
            for cid in ready:
                self.executors[cid](cid, round_n)
                self.mark_completed(cid, round_n)

        if not self.is_fully_done():
            raise RuntimeError(
                f"run_all_rounds beendet ohne alle Ziele zu erreichen (evtl. max_rounds zu "
                f"niedrig oder nie angefragte Abhängigkeit): {self.get_pending_nodes()}")
