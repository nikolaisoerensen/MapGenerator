"""
Regressionstest fuer die STRUKTURELLEN Zusagen des Water-Systems
(2026-07-27). Nicht Teil einer Test-Suite - manuell ueber das gemeinsame venv
laufen lassen, siehe CLAUDE.md.

Deckt vier Eigenschaften ab, die vorher nirgends geprueft wurden und deren
Verletzung jeweils monatelang unbemerkt blieb:

(A) REIHENFOLGE + REGEN-ENTKOPPLUNG (Nutzer-Vorgabe 2026-07-27)
    Pro LOD-Runde muss gelten: erst erodieren + sedimentieren, dann
    Boeschungswinkel, DANACH Seen/Wasserkreislauf mit dem echten Regen.
    Der CalculatorDispatcher leitet die Reihenfolge ausschliesslich aus
    depends_on ab und startet alle in derselben Runde bereiten Knoten
    parallel - ohne die passenden Kanten wurden Seen auf dem NICHT erodierten
    Gelaende gesucht. Zusaetzlich darf water.erosion_sedimentation NIE von
    weather.precipitation abhaengen.

(B) RESET
    Parameter aendern und zurueckstellen darf das Ergebnis nicht veraendern.
    invalidate_cache_lod() leerte frueher nur den Domain-Storage und liess
    den Calculator-Zwischenstand stehen - die naechste Generierung erodierte
    dadurch auf dem bereits erodierten Gelaende weiter.

(C) DETERMINISMUS
    Gleiche Eingaben, gleiche Parameter -> bitidentische Outputs.

(D) SKALIERUNG mit map_size und map_distance_km
    Fluss-Anteil, Erosion pro Pixel und Wassertiefe muessen innerhalb eines
    Toleranzbands bleiben, statt sich um Groessenordnungen zu verschieben.
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

from core.water_generator import HydrologySystemGenerator
from gui.OldManagers.calculator_graph import CALCULATOR_GRAPH, CalculatorDispatcher
from gui.OldManagers.data_lod_manager import DataLODManager


def check(label, condition):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}")
    return bool(condition)


# =============================================================================
# (A) Reihenfolge + Regen-Entkopplung
# =============================================================================

def run_execution_order_and_rain_decoupling():
    ok = True

    # --- Regen-Entkopplung: statisch am Graph pruefbar ---
    # Erosion-Umbau 2026-07-28: das Gelaende formt jetzt der eigene Knoten
    # erosion.hydraulic (core/erosion_generator.py), der VOR Weather laeuft.
    # Die Regen-Entkopplung ist damit noch staerker strukturell verankert als
    # vorher - weather.* kann gar nicht mehr davorstehen.
    erosion_deps = set(CALCULATOR_GRAPH["erosion.hydraulic"].depends_on)
    ok &= check(
        f"erosion.hydraulic haengt NICHT von weather.* ab (deps={sorted(erosion_deps)})",
        not any(dep.startswith("weather.") for dep in erosion_deps))

    def ancestors(node, seen=None):
        seen = seen if seen is not None else set()
        for dep in CALCULATOR_GRAPH[node].depends_on:
            if dep not in seen:
                seen.add(dep)
                ancestors(dep, seen)
        return seen

    # Der eigentliche Gewinn des Umbaus: Weather rechnet nicht mehr auf dem
    # unerodierten Gelaende. Bis 2026-07-28 lag die Erosion im Water-Block und
    # damit HINTER Weather - Temperatur, Wind und Niederschlag sahen die
    # Taeler nie.
    for downstream in ("weather.temperature", "weather.precipitation",
                       "water.flow_network", "biome.base_classification"):
        ok &= check(
            f"{downstream} sieht das erodierte Gelaende",
            "erosion.hydraulic" in ancestors(downstream))

    ok &= check(
        "water.flow_network haengt von weather.precipitation ab (echter Regen im Kreislauf)",
        "weather.precipitation" in CALCULATOR_GRAPH["water.flow_network"].depends_on)

    # --- Reihenfolge: dynamisch ueber den echten Dispatcher ---
    # Jeder Knoten wird "ausgefuehrt", indem er nur seine Runde protokolliert.
    execution_log = []

    def make_executor(calculator_id):
        def executor(context):
            execution_log.append((context["round"], calculator_id))
        return executor

    dispatcher = CalculatorDispatcher({cid: make_executor(cid) for cid in CALCULATOR_GRAPH})
    for generator in ("terrain", "geology", "erosion", "weather", "water", "biome", "settlement"):
        dispatcher.request(generator, 3)

    round_n = 1
    while not dispatcher.is_fully_done():
        ready = dispatcher.get_ready_nodes(round_n)
        if not ready:
            round_n += 1
            if round_n > 10:
                break
            continue
        for cid in ready:
            execution_log.append((round_n, cid))
            dispatcher.mark_completed(cid, round_n)

    # Pro Runde: an welcher Position lief welcher Knoten?
    order_ok = True
    rounds_checked = 0
    for target_round in sorted({r for r, _ in execution_log}):
        in_round = [cid for r, cid in execution_log if r == target_round]
        positions = {cid: i for i, cid in enumerate(in_round)}
        required = ["erosion.hydraulic", "weather.temperature",
                    "water.lake_detection", "water.flow_network",
                    "water.manning_flow", "water.soil_moisture"]
        if not all(cid in positions for cid in required):
            continue
        rounds_checked += 1
        sequence = [positions[cid] for cid in required]
        if sequence != sorted(sequence):
            order_ok = False
            print(f"    Runde {target_round}: falsche Reihenfolge "
                  f"{[(cid, positions[cid]) for cid in required]}")

    ok &= check(f"Erosion- und Water-Knoten laufen in jeder Runde in der vorgeschriebenen "
                f"Reihenfolge ({rounds_checked} Runden geprueft)",
                order_ok and rounds_checked > 0)

    # Kernaussage: lake_detection darf NIE bereit sein, bevor die Erosion
    # dieselbe Runde abgeschlossen hat.
    ok &= check("water.lake_detection haengt direkt von erosion.hydraulic ab",
                "erosion.hydraulic" in CALCULATOR_GRAPH["water.lake_detection"].depends_on)
    return ok


def run_every_generator_is_reachable():
    """
    Jeder Generator im Graph muss vom Auto-Start auch ANGEFRAGT werden, und
    jeder Knoten muss erreichbar sein.

    Der Anlass: die Auto-Start-Liste in gui/map_editor.py war fest verdrahtet
    auf sechs Generatoren. Der neue Erosion-Generator fehlte darin, sein Knoten
    blieb auf Ziel-LOD 0 - und weil weather.temperature seit dem Umbau darauf
    wartet, stand die gesamte Pipeline nach Terrain und Geology still. In der
    laufenden App sichtbar als "13 / 111 LOD-Runden, 12%", ohne Fehlermeldung.

    Ein Deadlock durch einen vergessenen Listeneintrag ist die unangenehmste
    Sorte Fehler: nichts stuerzt ab, nichts wird geloggt, es passiert nur
    nichts mehr. Deshalb wird hier geprueft, dass die Listen AUS dem Graphen
    bzw. aus GeneratorType abgeleitet sind statt danebenzuliegen.
    """
    from gui.OldManagers.generation_orchestrator import GeneratorType
    from gui.widgets.pipeline_status_panel import GENERATOR_ORDER

    graph_generators = {spec.generator for spec in CALCULATOR_GRAPH.values()}
    enum_generators = {generator.value for generator in GeneratorType}

    ok = check(
        f"jeder Generator im Graph steht in GeneratorType "
        f"(Graph: {sorted(graph_generators)})",
        graph_generators <= enum_generators)
    ok &= check(
        f"der Pipeline-Status zeigt jeden Generator (Panel: {GENERATOR_ORDER})",
        graph_generators <= set(GENERATOR_ORDER))

    # Dynamisch: werden ALLE Knoten fertig, wenn alle Generatoren angefragt
    # werden - und bleibt umgekehrt etwas haengen, wenn einer fehlt?
    def run_dispatcher(requested):
        dispatcher = CalculatorDispatcher({cid: (lambda ctx: None) for cid in CALCULATOR_GRAPH})
        for generator in requested:
            dispatcher.request(generator, 2)
        for round_n in range(1, 12):
            while True:
                ready = dispatcher.get_ready_nodes(round_n)
                if not ready:
                    break
                for cid in ready:
                    dispatcher.mark_completed(cid, round_n)
        return dispatcher.is_fully_done()

    ok &= check("mit allen Generatoren laeuft die Pipeline vollstaendig durch",
                run_dispatcher(sorted(enum_generators)))

    # Gegenprobe: genau das Weglassen von "erosion" muss den Stillstand
    # reproduzieren - sonst wuerde der Test oben nichts beweisen.
    without_erosion = sorted(enum_generators - {"erosion"})
    ok &= check("ohne den Erosion-Generator bleibt die Pipeline stehen "
                "(Gegenprobe: der Test misst wirklich etwas)",
                not run_dispatcher(without_erosion))
    return ok


def run_dependency_tree_matches_graph():
    """
    Der Dependency-Tree des Orchestrators muss aus CALCULATOR_GRAPH abgeleitet
    sein - und zwar so, dass er JEDEN Generator kennt.

    Der Anlass (2026-07-28, in der laufenden App): die Tabelle stand als
    Literal im Konstruktor und kannte den Erosion-Generator nicht. Eine
    Terrain-Aenderung invalidierte water/weather/settlement/biome/geology, aber
    NICHT erosion. Terrain rechnete auf das neue Ziel-LOD hoch, Erosion blieb
    auf ihrem alten Stand, und alles hinter ihr wartete auf eine Runde, die nie
    kam. Im Log fuenf gleichzeitige "Generation ... timed out" - und davor
    keine einzige Fehlermeldung.

    Beim Ableiten fiel ausserdem eine FEHLENDE Graph-Kante auf: Settlement
    liest biome_map (core/settlement_generator.py:3278), deklarierte aber
    keinen Biome-Knoten. Der Graph hatte unrecht, die alte Handtabelle recht.
    Beides ist behoben; dieser Test haelt beide Seiten zusammen.
    """
    from gui.OldManagers.generation_orchestrator import (
        GenerationOrchestrator, GeneratorType)

    tree = GenerationOrchestrator._derive_dependency_tree()

    ok = check("der Dependency-Tree kennt jeden GeneratorType "
               "({} Eintraege)".format(len(tree)),
               set(tree) == set(GeneratorType))

    # Erosion formt das Gelaende - Weather und Water MUESSEN dahinter haengen,
    # sonst rechnen sie auf dem unerodierten Gelaende.
    for downstream in (GeneratorType.WEATHER, GeneratorType.WATER):
        ok &= check("{} haengt von erosion ab".format(downstream.value),
                    GeneratorType.EROSION in tree[downstream])

    # Gegenprobe: jede Kante im Baum muss eine ECHTE Knotenkante im Graph
    # haben. Ein Baum, der einfach alles mit allem verbindet, wuerde die
    # Pruefungen oben ebenfalls bestehen - und jede Parameteraenderung die
    # halbe Pipeline neu rechnen lassen.
    edges = {(spec.generator, CALCULATOR_GRAPH[dep].generator)
             for spec in CALCULATOR_GRAPH.values()
             for dep in spec.depends_on if dep in CALCULATOR_GRAPH}
    spurious = [(gen.value, up.value) for gen, ups in tree.items()
                for up in ups if (gen.value, up.value) not in edges]
    ok &= check("keine erfundenen Kanten im Baum "
                "(gefunden: {})".format(spurious or "keine"), not spurious)

    # Settlement liest biome_map - die Kante muss im GRAPH stehen, damit die
    # Reihenfolge erzwungen ist, nicht nur die Invalidierung.
    ok &= check("settlement haengt im Graph von biome ab",
                GeneratorType.BIOME in tree[GeneratorType.SETTLEMENT])
    return ok


def run_terrain_forming_generators_refresh_all_tabs():
    """
    Jeder Generator, dessen Ergebnis in die KOMBINIERTE Heightmap einfliesst,
    muss in BaseMapTab._TERRAIN_FORMING_GENERATORS stehen.

    Der Anlass: ein Tab aktualisiert seine Anzeige normalerweise nur, wenn sein
    EIGENER Generator meldet. Jeder Tab baut sein 3D-Mesh aber aus der
    kombinierten Heightmap - und die aendert sich auch, wenn ein FREMDER
    Generator rechnet. Der neue Erosion-Generator blieb dadurch fuer alle
    anderen Tabs unsichtbar: "Heightmap Combined" enthielt die Erosion, das
    3D-Mesh zeigte weiter das unerodierte Gelaende.

    Geprueft wird gegen die Datenschluessel, die get_terrain_data_combined()
    tatsaechlich liest - so faellt ein kuenftiger gelaendeformender Generator
    hier auf, statt still ein veraltetes Mesh stehen zu lassen.
    """
    from gui.tabs.base_tab import BaseMapTab
    from gui.OldManagers.data_lod_manager import DATA_KEY_TO_TAB_MAPPING

    # Die Karten, die get_terrain_data_combined() verrechnet.
    combined_keys = ("heightmap", "height_delta", "erosion_map", "sedimentation_map",
                     "thermal_erosion_map", "thermal_deposition_map")
    owners = set()
    for key in combined_keys:
        owner = DATA_KEY_TO_TAB_MAPPING.get(key)
        if owner:
            owners.add(owner)
    # height_delta steht nicht in der Tabelle - es gehoert Geology.
    owners.add("geology")

    declared = set(BaseMapTab._TERRAIN_FORMING_GENERATORS)
    missing = owners - declared
    return check(
        "alle gelaendeformenden Generatoren loesen in JEDEM Tab ein "
        "Display-Update aus (erwartet {}, fehlen: {})".format(
            sorted(owners), sorted(missing)),
        not missing)


# =============================================================================
# Gemeinsamer Aufbau fuer (B)-(D)
# =============================================================================

def _make_inputs(size, seed=9):
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    heightmap = (300.0 + 900.0 * np.exp(-((x - size * 0.4) ** 2 + (y - size * 0.4) ** 2)
                                         / (2 * (size * 0.2) ** 2))
                 + 40.0 * rng.randn(size, size)).astype(np.float32)
    return {
        'heightmap': heightmap,
        'hardness_map': np.full((size, size), 50.0, dtype=np.float32),
        'precip_map': np.full((size, size), 8.0, dtype=np.float32),
        'temp_map': np.full((size, size), 15.0, dtype=np.float32),
        'wind_map': np.zeros((size, size, 2), dtype=np.float32),
        'humid_map': np.full((size, size), 50.0, dtype=np.float32),
    }


def _default_parameters():
    return {
        'lake_volume_threshold': 5000.0,
        'river_abundance': 0.10,
        'erosion_strength': 2.5,
        'sediment_capacity_factor': 4.0,
        'settling_velocity': 0.3,
        'thermal_erosion_strength': 1.0,
        'diffusion_radius': 2.0,
        'evaporation_base_rate': 0.002,
    }


def _set_target_lod(dlm, lod):
    """Ziel-LOD für alle Water-Knoten setzen - genau das tut
    GenerationOrchestrator.request_generation() in der echten Pipeline.

    Ohne diesen Schritt hält HydrologySystemGenerator._is_final_lod() keine
    Runde für final und Erosion/Böschungswinkel liefern durchgehend Nullkarten
    (seit 2026-07-27, siehe dortigen Docstring). Ein Test, der das vergisst,
    misst am Erosionspfad nichts und ist trotzdem grün - deshalb steht das
    hier explizit und wird unten auch geprüft.
    """
    for cid, spec in CALCULATOR_GRAPH.items():
        if spec.generator == "water":
            dlm.set_calculator_target_lod(cid, lod)


def _generate(size, map_distance_km, parameters, lod=2):
    dlm = DataLODManager()
    dlm.set_map_distance_km(map_distance_km)
    _set_target_lod(dlm, lod)
    generator = HydrologySystemGenerator(map_seed=7, data_lod_manager=dlm)
    dependencies = _make_inputs(size)
    water_data = generator._execute_generation(lod, dependencies, parameters)
    return water_data, dlm


# =============================================================================
# (B) Reset
# =============================================================================

def run_reset_restores_identical_result():
    """Parameter aendern und exakt zurueckstellen -> identisches Ergebnis.

    Prueft den Pfad, der bei JEDER Parameter-Aenderung laeuft
    (DataLODManager.invalidate_cache_lod), nicht das grobe clear_all_data().
    """
    size = 48
    parameters = _default_parameters()

    dlm = DataLODManager()
    dlm.set_map_distance_km(10.0)
    _set_target_lod(dlm, 2)
    generator = HydrologySystemGenerator(map_seed=7, data_lod_manager=dlm)
    dependencies = _make_inputs(size)

    baseline = generator._execute_generation(2, dependencies, parameters)
    # Die Erosions- und Thermal-Karten sind hier entfallen: Water formt seit
    # 2026-07-28 kein Gelaende mehr (siehe core/erosion_generator.py). Geprueft
    # wird jetzt der Wasserkreislauf selbst - er ist der Teil, dessen
    # Zwischenzustand (depth_state/flux_state) einen verworfenen Lauf
    # ueberleben koennte.
    baseline_depth = float(np.sum(baseline.water_map))
    baseline_flow = float(np.sum(baseline.flow_map))
    baseline_moisture = float(np.sum(baseline.soil_moist_map))

    # Zwischendurch mit anderen Parametern rechnen, danach invalidieren -
    # exakt das, was der GenerationOrchestrator bei einer Slider-Aenderung tut.
    other = dict(parameters, erosion_strength=5.0)
    generator._execute_generation(2, dependencies, other)
    dlm.invalidate_cache_lod("water")
    # Ziel-LOD nachziehen, wie es GenerationOrchestrator.request_generation()
    # in der echten Pipeline unmittelbar nach einer Invalidierung tut - ohne
    # ein gesetztes Ziel-LOD faende _is_final_lod() keins und der zweite Lauf
    # wuerde gar nicht erodieren, waere also nicht mit dem ersten vergleichbar.
    #
    # Dass dieser Test ueberhaupt greift, haengt daran, dass die beiden
    # gelaendeformenden Knoten ihren eigenen Vorstand selbst wegraeumen (siehe
    # HydrologySystemGenerator.TERRAIN_FORMING_CALCULATORS). invalidate_cache_lod()
    # fasst den Calculator-Storage bewusst NICHT mehr an - der pauschale Purge
    # riss laufenden Threads anderer Generatoren die Daten weg (Begruendung in
    # DataLODManager.invalidate_cache_lod).
    _set_target_lod(dlm, 2)

    repeat = generator._execute_generation(2, dependencies, parameters)
    repeat_depth = float(np.sum(repeat.water_map))
    repeat_flow = float(np.sum(repeat.flow_map))
    repeat_moisture = float(np.sum(repeat.soil_moist_map))

    ok = check(f"water_map.sum() nach Reset identisch "
               f"(vorher={baseline_depth:.3f}, nachher={repeat_depth:.3f})",
               np.isclose(baseline_depth, repeat_depth, rtol=1e-6))
    ok &= check(f"flow_map.sum() nach Reset identisch "
                f"(vorher={baseline_flow:.3f}, nachher={repeat_flow:.3f})",
                np.isclose(baseline_flow, repeat_flow, rtol=1e-6))
    ok &= check(f"soil_moist_map.sum() nach Reset identisch "
                f"(vorher={baseline_moisture:.3f}, nachher={repeat_moisture:.3f})",
                np.isclose(baseline_moisture, repeat_moisture, rtol=1e-6))
    return ok


# =============================================================================
# (C) Determinismus
# =============================================================================

def run_determinism():
    """Zwei unabhaengige Laeufe mit identischen Eingaben -> bitidentisch.

    Faengt insbesondere die Wettlaufsituation ab, in der Konsumenten je nach
    Thread-Timing die Fluss-Zentrallinie oder die gemalte Fassung sahen.
    """
    parameters = _default_parameters()
    first, _ = _generate(48, 10.0, parameters)
    second, _ = _generate(48, 10.0, parameters)

    ok = True
    for key in HydrologySystemGenerator.WATER_DATA_KEYS:
        a = getattr(first, key)
        b = getattr(second, key)
        if isinstance(a, np.ndarray):
            identical = np.array_equal(a, b)
        else:
            identical = a == b
        ok &= check(f"{key} bitidentisch ueber zwei Laeufe", identical)
    return ok


# =============================================================================
# (D) Skalierung
# =============================================================================

def run_scale_invariance():
    parameters = _default_parameters()

    print("  --- map_size-Variation (map_distance 10 km) ---")
    by_size = {}
    for size in (48, 96, 192):
        water_data, _ = _generate(size, 10.0, parameters, lod=3)
        pixels = water_data.water_biomes_map.size
        by_size[size] = {
            "river_fraction": float(np.mean(water_data.water_biomes_map > 0)),
            "mean_depth": float(np.mean(water_data.water_map)),
        }
        print(f"    size={size:4d}: Fluss-Anteil={by_size[size]['river_fraction']:.3f}  "
              f"mittlere Tiefe={by_size[size]['mean_depth']:.4f} m")

    ok = True
    # Zuerst: rechnet der Wasserkreislauf ueberhaupt? Ohne diese Zusicherung
    # waeren alle Stabilitaets-Aussagen unten trivial erfuellt, sobald der Lauf
    # still nichts mehr liefert - genau das ist beim Erosions-Umbau 2026-07-28
    # passiert (ein NameError im Zusammenbau wurde von einem inzwischen
    # entfernten `except Exception`-Zweig in Nullkarten verwandelt).
    ok &= check("der Wasserkreislauf liefert ueberhaupt Wasser (sonst misst der Test nichts)",
                all(v["mean_depth"] > 0.0 for v in by_size.values()))

    # ERSATZLOS ENTFALLEN: erosion_per_pixel. Water erodiert seit 2026-07-28
    # nicht mehr - das macht der eigene Erosion-Generator
    # (core/erosion_generator.py). Die Skalierung der Erosion ueber map_size
    # und map_distance ist dort neu zu leisten und noch offen; sie gehoert in
    # einen eigenen Erosions-Skalierungstest, nicht hierher.
    for metric, tolerance in (("river_fraction", 3.0),):
        values = [v[metric] for v in by_size.values()]
        low, high = min(values), max(values)
        ratio = high / max(low, 1e-9)
        ok &= check(f"{metric} bleibt ueber map_size stabil "
                    f"(Faktor {ratio:.2f}, erlaubt < {tolerance})", ratio < tolerance)

    print("  --- map_distance-Variation (map_size 96) ---")
    by_distance = {}
    for distance in (2.0, 10.0, 50.0):
        water_data, _ = _generate(96, distance, parameters, lod=3)
        by_distance[distance] = {
            "river_fraction": float(np.mean(water_data.water_biomes_map > 0)),
            "mean_depth": float(np.mean(water_data.water_map)),
        }
        print(f"    map_distance={distance:5.1f} km: "
              f"Fluss-Anteil={by_distance[distance]['river_fraction']:.3f}  "
              f"mittlere Tiefe={by_distance[distance]['mean_depth']:.4f} m")

    # Toleranz bewusst weiter als bei map_size: map_distance aendert bei
    # gleichbleibender TERRAIN.AMPLITUDE die tatsaechliche Gelaendeform. Eine
    # 2-km-Karte mit 4000 m Relief ist Hochgebirge, eine 50-km-Karte mit
    # demselben Relief ein sanftes Huegelland - dass dort unterschiedlich viel
    # Wasser stehen bleibt und sich sammelt, ist richtig und nicht der Fehler,
    # den dieser Test sucht. Ueber den 25-fachen map_distance-Bereich liegt
    # der gemessene Unterschied bei etwa Faktor 3.3; als Regression gilt eine
    # Groessenordnung, nicht diese physikalisch erwartete Reaktion.
    #
    # Der eigentliche Befund, den dieser Test absichert: VOR der Korrektur der
    # Naesse-Maske (siehe WET_CELL_DISCHARGE_FRACTION) lag der Fluss-Anteil in
    # ALLEN Konfigurationen bei exakt river_abundance (0.100) - vollkommen
    # unabhaengig von Terrain, Klima und Kartengroesse. Ein Ergebnis, das sich
    # hier ueberhaupt nicht mehr bewegt, ist deshalb genauso verdaechtig wie
    # eines, das explodiert.
    river_values = [v["river_fraction"] for v in by_distance.values()]
    ratio = max(river_values) / max(min(river_values), 1e-9)
    ok &= check(f"Fluss-Anteil reagiert auf map_distance, ohne zu entgleisen "
                f"(Faktor {ratio:.2f}, erlaubt 1.05 .. 10.0)", 1.05 < ratio < 10.0)
    return ok


# =============================================================================
# (E) Einzel-Neuberechnung nach vollem Lauf
# =============================================================================

def run_single_generator_rerun_after_full_run():
    """
    Nach einem vollstaendigen Lauf bis LOD 3 muss ein EINZELN neu angefragter
    Generator wieder bei LOD 1 durchlaufen koennen - seine Upstream-Daten
    gehoeren Generatoren, die dabei NICHT neu rechnen und deren LOD-1-Stand
    deshalb erhalten bleiben muss.

    Genau das hat eine LOD-Eviction (2026-07-27, wieder entfernt) gebrochen:
    sie verwarf alles unterhalb der vorletzten Stufe, worauf der Klick auf
    GENERIEREN im Water-Tab mit
      "erosion.hydraulic failed at LOD 1:
       fehlende Dependencies: heightmap, hardness_map"
    abbrach. get_calculator_output() sucht nur abwaerts - lag die Heightmap
    nur noch bei LOD 2/3 vor, fand eine LOD-1-Anfrage nichts.
    Siehe DataLODManager, Abschnitt "KEINE LOD-EVICTION".
    """
    size = 48
    parameters = _default_parameters()

    dlm = DataLODManager()
    dlm.set_map_distance_km(10.0)
    _set_target_lod(dlm, 3)
    generator = HydrologySystemGenerator(map_seed=7, data_lod_manager=dlm)
    dependencies = _make_inputs(size)

    # Voller Lauf ueber die LOD-Kette, wie beim Auto-Start.
    for lod in (1, 2, 3):
        generator._execute_generation(lod, dependencies, parameters)

    ok = check("Upstream-Heightmap bei LOD 1 nach vollem Lauf noch vorhanden",
               dlm.get_calculator_output("terrain.redistribution", "heightmap", 1) is not None)
    ok &= check("Upstream-Haerte bei LOD 1 nach vollem Lauf noch vorhanden",
                dlm.get_calculator_output("geology.hardness", "hardness_map", 1) is not None)

    # Einzelne Neuberechnung: nur Water invalidieren, Upstream bleibt stehen.
    dlm.invalidate_cache_lod("water")
    ok &= check("nach Water-Invalidierung ist die Upstream-Heightmap bei LOD 1 UNBERUEHRT",
                dlm.get_calculator_output("terrain.redistribution", "heightmap", 1) is not None)

    try:
        from core.erosion_generator import ErosionSystemGenerator
        erosion_generator = ErosionSystemGenerator(data_lod_manager=dlm)
        erosion_generator.set_active_parameters({"max_steps": 50})
        erosion_generator._calc_hydraulic("erosion.hydraulic", 1)
        rerun_ok = dlm.get_calculator_output(
            "erosion.hydraulic", "erosion_map", 1) is not None
        ok &= check("erosion.hydraulic laeuft bei LOD 1 erneut durch", rerun_ok)
    except Exception as e:
        ok &= check(f"erosion.hydraulic laeuft bei LOD 1 erneut durch "
                    f"({type(e).__name__}: {e})", False)
    return ok


if __name__ == "__main__":
    results = {
        "execution_order_and_rain_decoupling": run_execution_order_and_rain_decoupling(),
        "every_generator_is_reachable": run_every_generator_is_reachable(),
        "dependency_tree_matches_graph": run_dependency_tree_matches_graph(),
        "terrain_forming_generators_refresh_all_tabs":
            run_terrain_forming_generators_refresh_all_tabs(),
        "reset_restores_identical_result": run_reset_restores_identical_result(),
        "determinism": run_determinism(),
        "scale_invariance": run_scale_invariance(),
        "single_generator_rerun_after_full_run": run_single_generator_rerun_after_full_run(),
    }
    print("\n=== SUMMARY ===")
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)
