"""
Path: gui/OldManagers/calculator_graph.py

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
    CalculatorSpec("terrain.redistribution", "terrain", ["terrain.noise"], ["heightmap"]),
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
    CalculatorSpec("weather.temperature", "weather",
                   ["terrain.redistribution", "terrain.shadow", "erosion.hydraulic"],
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
                   ["terrain.redistribution", "terrain.slope"], ["preseed_biome_map"]),
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
                   ["terrain.redistribution", "terrain.slope", "water.manning_flow"],
                   ["combined_suitability_map"]),
    CalculatorSpec("settlement.settlements", "settlement",
                   ["settlement.suitability", "terrain.redistribution"], ["settlement_list"]),
    CalculatorSpec("settlement.city_boundary", "settlement",
                   ["settlement.settlements", "terrain.redistribution", "terrain.slope"],
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
                   ["settlement.settlements", "terrain.slope",
                    "biome.integrate_layers"], ["roads"]),
    CalculatorSpec("settlement.outer_roads", "settlement",
                   ["settlement.settlements", "settlement.suitability", "terrain.slope"], ["outer_roads"]),
    CalculatorSpec("settlement.roadsites", "settlement", ["settlement.pathfinding"], ["roadsite_list"]),
    CalculatorSpec("settlement.civ_influence", "settlement",
                   ["terrain.redistribution", "terrain.slope", "settlement.settlements",
                    "settlement.pathfinding", "settlement.roadsites"], ["civ_map"]),
    CalculatorSpec("settlement.landmarks", "settlement",
                   ["settlement.civ_influence", "terrain.redistribution", "terrain.slope"], ["landmark_list"]),
    CalculatorSpec("settlement.landmark_roads", "settlement",
                   ["settlement.landmarks", "settlement.pathfinding", "terrain.slope"], ["landmark_roads"]),
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
# -> 38 aktive Knoten.
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

    def get_pending_nodes(self) -> List[str]:
        """Alle angefragten, aber noch nicht auf ihr Ziel-LOD gebrachten Knoten."""
        return [
            cid for cid in CALCULATOR_GRAPH
            if self.target_lod[cid] > 0 and self.completed_lod[cid] < self.target_lod[cid]
        ]

    def is_fully_done(self) -> bool:
        """True, wenn jeder angefragte Knoten sein Ziel-LOD erreicht hat."""
        return len(self.get_pending_nodes()) == 0

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
        round_n = self._next_round
        while True:
            still_pending_for_round = [
                cid for cid, spec in CALCULATOR_GRAPH.items()
                if self.target_lod[cid] >= round_n and self.completed_lod[cid] < round_n
            ]

            if not still_pending_for_round:
                # Runde round_n ist (soweit überhaupt angefragt) vollständig erledigt
                round_n += 1
                self._next_round = round_n
                if self.is_fully_done():
                    return []
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
