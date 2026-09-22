"""
Throwaway headless smoke test for the reworked 3D-Gesteinsstapel Geology
generator (core/geology_generator.py). Not part of the test suite - run
manually via the shared venv, see CLAUDE.md.
"""
import sys
import traceback

import numpy as np

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

from managers.data_lod_manager import DataLODManager
from core.geology_generator import GeologySystemGenerator, N_LAYERS
from gui.config.value_default import GEOLOGY


def make_synthetic_heightmap(size, seed=1):
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 6, size)
    y = np.linspace(0, 6, size)
    X, Y = np.meshgrid(x, y)
    base = 1500 + 900 * np.sin(X) * np.cos(Y) + 200 * rng.randn(size, size)
    return base.astype(np.float32)


def make_synthetic_slopemap(heightmap):
    """Echte Slope aus der Heightmap ableiten statt All-Null - slopemap wird
    jetzt für die Slope-Verdünnung der Schichtdicke tatsächlich konsumiert
    (Teil-2-Rework Punkt A4), ein konstantes Null-Slopemap würde diesen
    Codepfad nie ausüben."""
    dzdy, dzdx = np.gradient(heightmap.astype(np.float64))
    return np.stack([dzdx, dzdy], axis=-1).astype(np.float32)


def full_params():
    p = {}
    for key, cfg_name in [
        ('sedimentary_hardness', 'SEDIMENTARY_HARDNESS'), ('igneous_hardness', 'IGNEOUS_HARDNESS'),
        ('metamorphic_hardness', 'METAMORPHIC_HARDNESS'), ('tilt_intensity', 'TILT_INTENSITY'),
        ('tilt_direction', 'TILT_DIRECTION'), ('fold_intensity', 'FOLD_INTENSITY'),
        ('fold_detail', 'FOLD_DETAIL'), ('fault_intensity', 'FAULT_INTENSITY'),
        ('fault_detail', 'FAULT_DETAIL'), ('fault_edge_softness', 'FAULT_EDGE_SOFTNESS'),
        ('intrusion_density', 'INTRUSION_DENSITY'), ('intrusion_size', 'INTRUSION_SIZE'),
        ('intrusion_detail', 'INTRUSION_DETAIL'),
        ('metamorphic_overprint_intensity', 'METAMORPHIC_OVERPRINT_INTENSITY'),
        ('foliation_detail', 'FOLIATION_DETAIL'),
    ]:
        p[key] = getattr(GEOLOGY, cfg_name)["default"]
    # Alle Effekt-Stärken deutlich über 0 setzen, damit jeder Codepfad
    # (Tilt/Fold/Fault/Intrusion/Metamorphic/Foliation) tatsächlich ausgeführt wird.
    p['tilt_intensity'] = 8.0
    p['fault_intensity'] = 120.0
    p['intrusion_density'] = 0.5
    return p


def check(label, condition):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def run_hardness_variability():
    """
    Feinschliff-Runde: die 13 ROCK_LAYERS haben jetzt einen festen
    RockLayer.hardness_factor statt alle sedimentären Schichten dieselbe
    Kategorie-Härte zu teilen (Nutzer-Feedback: "zu unvariabel"). Bei
    Default-Parametern auf einer Heightmap, die mehrere Schichten
    ausbeißen lässt, muss hardness_map eine deutliche Streuung zeigen.
    """
    size = 256
    dlm = DataLODManager()
    dlm.set_map_distance_km(10.0)
    geo = GeologySystemGenerator(map_seed=21, data_lod_manager=dlm)
    heightmap = make_synthetic_heightmap(size, seed=5)
    slopemap = make_synthetic_slopemap(heightmap)
    params = full_params()
    result = geo.calculate_geology(heightmap, slopemap, params, lod_level=5)

    std = float(np.std(result.hardness_map))
    unique_layers = len(np.unique(result.layer_id_map))
    ok = check(f"mehrere Schichten beißen aus (gefunden: {unique_layers})", unique_layers >= 2)
    ok &= check(f"hardness_map Standardabweichung deutlich > 0 (std={std:.2f})", std > 3.0)
    return ok


def run_intrusion_irregularity():
    """
    Feinschliff-Runde: intrusion_detail soll bei hohen Werten (0.9) deutlich
    unförmigere Blob-Ränder erzeugen als bei niedrigen (0.1) - Nutzer-
    Feedback: "selbst auf 0.9 sind die Blobs noch sehr rund". Gleicher Seed
    -> identische Blob-Position/-Radius in beiden Läufen (intrusion_detail
    beeinflusst keine RandomState-Aufrufe), daher isoliert die Differenz
    zwischen den beiden Läufen exakt den Effekt der Rand-Amplitude.
    """
    from core.geology_generator import _build_intrusion_field

    shape = (256, 256)
    map_distance_km = 10.0
    map_seed = 33
    # intrusion_density so gewählt, dass genau 1 Blob entsteht (round(0.2*6)=1)
    # - vereinfacht die Analyse (keine Überlagerung mehrerer Blobs).
    low, _ = _build_intrusion_field(shape, map_distance_km, map_seed, 1000.0, 0.2, 1.0, 0.1)
    high, _ = _build_intrusion_field(shape, map_distance_km, map_seed, 1000.0, 0.2, 1.0, 0.9)

    diff_std = float(np.std(high - low))
    return check(f"intrusion_detail 0.1->0.9 aendert die Randkontur deutlich (std diff={diff_std:.4f} km)",
                 diff_std > 0.05)


def run_basic_multi_resolution():
    all_ok = True
    for size in (128, 256, 512):
        for map_distance_km in (5.0, 10.0, 50.0):
            dlm = DataLODManager()
            dlm.set_map_distance_km(map_distance_km)
            geo = GeologySystemGenerator(map_seed=42, data_lod_manager=dlm)

            heightmap = make_synthetic_heightmap(size)
            slopemap = make_synthetic_slopemap(heightmap)
            params = full_params()

            try:
                result = geo.calculate_geology(heightmap, slopemap, params, lod_level=5)
            except Exception as e:
                print(f"[FAIL] size={size} km={map_distance_km}: exception {e}")
                traceback.print_exc()
                all_ok = False
                continue

            label = f"size={size} km={map_distance_km}"
            all_ok &= check(f"{label}: rock_map shape", result.rock_map.shape == (size, size, 3))
            all_ok &= check(f"{label}: hardness_map shape", result.hardness_map.shape == (size, size))
            all_ok &= check(f"{label}: layer_id_map shape", result.layer_id_map.shape == (size, size))
            all_ok &= check(f"{label}: height_delta shape", result.height_delta.shape == (size, size))
            all_ok &= check(f"{label}: no NaN in hardness_map", not np.isnan(result.hardness_map).any())
            all_ok &= check(f"{label}: no NaN in height_delta", not np.isnan(result.height_delta).any())
            all_ok &= check(f"{label}: hardness in [1,100]",
                             bool(np.all((result.hardness_map >= 1.0) & (result.hardness_map <= 100.0))))
            all_ok &= check(f"{label}: layer_id in [0,N_LAYERS]",
                             bool(np.all((result.layer_id_map >= 0) & (result.layer_id_map <= N_LAYERS))))
            all_ok &= check(f"{label}: layer_boundaries shape",
                             result.layer_boundaries.shape == (N_LAYERS, size, size))
            all_ok &= check(f"{label}: fault_distance_map present", result.fault_distance_map is not None)
            all_ok &= check(f"{label}: intrusion_distance_map present", result.intrusion_distance_map is not None)
            all_ok &= check(f"{label}: metamorphic_grade_map present", result.metamorphic_grade_map is not None)
            all_ok &= check(f"{label}: delta_components has 5 keys",
                             result.delta_components is not None and
                             set(result.delta_components.keys()) ==
                             {"terrain_hub", "tilt", "fold", "fault", "intrusion"})
            all_ok &= check(f"{label}: is_valid()", result.is_valid())
    return all_ok


def run_height_delta_always_zero():
    """
    Nutzer-Korrektur (nach Runde 1 dieses Rework, siehe height_delta-Docstring
    in core/geology_generator.py): height_delta ist jetzt IMMER Null - auch
    Intrusionen tragen keinen Höhenbeitrag zur sichtbaren Karte mehr bei
    (die frühere gekappte Dom-Hebung erzeugte eine Terrain-Erhebung, die
    nicht gewollt war). Intrusionen wirken nur noch auf layer_id_map
    (Durchbruch/Ausbiss) - intrusion_delta bleibt als reines Diagnose-Feld
    (delta_components["intrusion"]) erhalten und ist weiterhin klar von Null
    verschieden, nur eben nicht mehr Teil von height_delta. Ersetzt den
    früheren Test aus Runde 2 (dort war height_delta == intrusion_delta).
    """
    from managers.calculator_graph import CALCULATOR_GRAPH
    sediment_spec = CALCULATOR_GRAPH["geology.sediment_overlay"]
    ok = check("sediment_overlay output_keys excludes height_delta",
               "height_delta" not in sediment_spec.output_keys)

    size = 128
    dlm = DataLODManager()
    dlm.set_map_distance_km(10.0)
    geo = GeologySystemGenerator(map_seed=7, data_lod_manager=dlm)
    heightmap = make_synthetic_heightmap(size, seed=2)
    slopemap = make_synthetic_slopemap(heightmap)
    params = full_params()
    result = geo.calculate_geology(heightmap, slopemap, params, lod_level=5)

    ok &= check("height_delta ist komplett Null (Geology traegt keinen Hoehenbeitrag mehr bei)",
                bool(np.allclose(result.height_delta, 0.0, atol=1e-9)))
    # Terrain-Hub/Tilt/Fold/Fault/Intrusion sind bei den Testparametern klar
    # von Null verschieden (full_params() setzt tilt/fault/fold/intrusion
    # ungleich 0) - alle bleiben als reine Diagnose-Felder erhalten, obwohl
    # keines davon mehr height_delta beeinflusst.
    ok &= check("tilt/fold/fault/intrusion delta_components sind nicht trivial Null",
                bool(np.any(result.delta_components["tilt"] != 0) or
                     np.any(result.delta_components["fold"] != 0) or
                     np.any(result.delta_components["fault"] != 0) or
                     np.any(result.delta_components["intrusion"] != 0)))
    return ok


def run_terrain_hub_high_terrain_shows_older_rock():
    """
    Kernanforderung der Terrain-Kopplungs-Runde: ein Punkt, der deutlich
    ÜBER seinem eigenen regional geglätteten Terrain-Mittel liegt (ein
    "Berg"), muss eine ÄLTERE (niedrigerer Index) Schicht zeigen als ein
    Punkt in seiner flachen, unauffälligen Umgebung - sonst hätte sich der
    Terrain-Hub-Effekt beim Ausbiss-Vergleich herausgekürzt oder falsch
    herum ausgewirkt (siehe _compute_outcrop()-Docstring, Punkt 2:
    Index-Spiegelung). Tektonik-Effekte werden für diesen isolierten
    Nachweis auf 0 gesetzt, damit NUR Terrain-Hub + Slope-Verdünnung den
    Unterschied erzeugen.
    """
    size = 256
    map_distance_km = 10.0

    base_height = 800.0
    heightmap = np.full((size, size), base_height, dtype=np.float64)
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size // 2, size // 2
    dist = np.hypot(xx - cx, yy - cy)
    peak_radius_px = size * 0.03
    peak_height = 1800.0
    heightmap += peak_height * np.exp(-(dist / peak_radius_px) ** 2)
    heightmap = heightmap.astype(np.float32)
    slopemap = make_synthetic_slopemap(heightmap)

    dlm = DataLODManager()
    dlm.set_map_distance_km(map_distance_km)
    geo = GeologySystemGenerator(map_seed=13, data_lod_manager=dlm)
    params = full_params()
    params['tilt_intensity'] = 0.0
    params['fold_intensity'] = 0.0
    params['fault_intensity'] = 0.0
    params['intrusion_density'] = 0.0

    result = geo.calculate_geology(heightmap, slopemap, params, lod_level=5)

    peak_layer_id = int(result.layer_id_map[cy, cx])
    far_layer_id = int(result.layer_id_map[10, 10])

    ok = check(f"Peak (layer_id={peak_layer_id}) zeigt aeltere Schicht als flache Umgebung (layer_id={far_layer_id})",
               peak_layer_id < far_layer_id)
    return ok


def run_lod_invariance():
    """Direkter Sprung auf lod_level=3 (128px) muss dasselbe height_delta/
    layer_id_map liefern wie das schrittweise Durchlaufen von LOD1(32px)->
    LOD2(64px)->LOD3(128px) - Kernanforderung aus der 3D-Stack-Diskussion."""
    target_size = 128
    map_distance_km = 10.0
    params = full_params()
    heightmap_target = make_synthetic_heightmap(target_size, seed=3)
    slopemap_target = make_synthetic_slopemap(heightmap_target)

    # Szenario A: direkter Sprung
    dlm_a = DataLODManager()
    dlm_a.set_map_distance_km(map_distance_km)
    geo_a = GeologySystemGenerator(map_seed=99, data_lod_manager=dlm_a)
    result_a = geo_a.calculate_geology(heightmap_target, slopemap_target, params, lod_level=3)

    # Szenario B: schrittweise LOD1(32)->LOD2(64)->LOD3(128, identische Heightmap)
    dlm_b = DataLODManager()
    dlm_b.set_map_distance_km(map_distance_km)
    geo_b = GeologySystemGenerator(map_seed=99, data_lod_manager=dlm_b)
    heightmap_32 = make_synthetic_heightmap(32, seed=3)
    heightmap_64 = make_synthetic_heightmap(64, seed=3)
    geo_b.calculate_geology(heightmap_32, make_synthetic_slopemap(heightmap_32), params, lod_level=1)
    geo_b.calculate_geology(heightmap_64, make_synthetic_slopemap(heightmap_64), params, lod_level=2)
    result_b = geo_b.calculate_geology(heightmap_target, slopemap_target, params, lod_level=3)

    ok = check("LOD-invariance: height_delta identical",
               bool(np.allclose(result_a.height_delta, result_b.height_delta, atol=1e-4)))
    ok &= check("LOD-invariance: layer_id_map identical",
                bool(np.array_equal(result_a.layer_id_map, result_b.layer_id_map)))
    ok &= check("LOD-invariance: hardness_map identical",
                bool(np.allclose(result_a.hardness_map, result_b.hardness_map, atol=1e-4)))
    return ok


def run_map_seed_propagation():
    """Nutzer-Bug-Report: Intrusion/Fault-Lines/Tilt sahen bei jeder Map
    identisch aus, auch nach einer Map-Seed-Aenderung. Root Cause: der
    GenerationOrchestrator instanziiert GeologySystemGenerator lazy OHNE
    map_seed-Konstruktor-Argument (bleibt beim Default 42) und cached die
    Instanz fuer die gesamte App-Session - self.map_seed aenderte sich nie.
    Fix: DataLODManager.get_map_seed()/set_map_seed() spiegeln Terrains
    map_seed-Parameter (analog zu map_distance_km/map_latitude), Geology.
    set_active_parameters() liest das bei JEDER Generierungsrunde neu und
    re-seedet thickness_builder/displacement_builder, falls es sich
    geaendert hat. Test: EINE GeologySystemGenerator-Instanz (wie im echten
    Cache), zwei calculate_geology()-Aufrufe mit unterschiedlichem
    dlm.set_map_seed() dazwischen - Intrusion/Fault/Tilt muessen sich
    unterscheiden."""
    size = 96
    dlm = DataLODManager()
    dlm.set_map_distance_km(10.0)
    geo = GeologySystemGenerator(data_lod_manager=dlm)  # kein map_seed-Argument, wie im echten Cache
    heightmap = make_synthetic_heightmap(size, seed=3)
    slopemap = make_synthetic_slopemap(heightmap)
    params = full_params()

    dlm.set_map_seed(111)
    result_a = geo.calculate_geology(heightmap, slopemap, params, lod_level=4)
    dlm.set_map_seed(222)
    result_b = geo.calculate_geology(heightmap, slopemap, params, lod_level=4)

    ok = check("geo.map_seed folgt dlm.get_map_seed() (222 nach zweitem Aufruf)",
               geo.map_seed == 222)
    ok &= check("intrusion_distance_map unterscheidet sich zwischen Seed 111 und 222",
                not np.allclose(result_a.intrusion_distance_map, result_b.intrusion_distance_map))
    ok &= check("fault delta_component unterscheidet sich zwischen Seed 111 und 222",
                not np.allclose(result_a.delta_components["fault"], result_b.delta_components["fault"]))
    ok &= check("fold delta_component unterscheidet sich zwischen Seed 111 und 222",
                not np.allclose(result_a.delta_components["fold"], result_b.delta_components["fold"]))
    return ok


if __name__ == "__main__":
    results = {
        "multi_resolution": run_basic_multi_resolution(),
        "height_delta_always_zero": run_height_delta_always_zero(),
        "terrain_hub_high_terrain_shows_older_rock": run_terrain_hub_high_terrain_shows_older_rock(),
        "lod_invariance": run_lod_invariance(),
        "hardness_variability": run_hardness_variability(),
        "intrusion_irregularity": run_intrusion_irregularity(),
        "map_seed_propagation": run_map_seed_propagation(),
    }
    print("\n=== SUMMARY ===")
    overall = True
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        overall &= ok
    sys.exit(0 if overall else 1)
