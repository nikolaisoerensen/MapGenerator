"""
Path: tests/smoke_test_adaptive_terrain_mesh.py

Prueft gui/widgets/adaptive_terrain_mesh.py headless (reine Numpy-Logik, kein
Qt/OpenGL noetig): Risslosigkeit (Wasserdichtigkeit), Dreiecks-Reduktion in
flachen Bereichen, Determinismus, Rand-Faelle (nicht geeignete Heightmap-Groessen)
und grobe Performance bei realistischer Map-Groesse.
"""
import sys
import time
import numpy as np

sys.path.insert(0, ".")

from gui.widgets.adaptive_terrain_mesh import (
    ist_fuer_adaptives_mesh_geeignet,
    baue_adaptives_mesh_roh,
    build_adaptive_mesh,
)


def pruefe_wasserdicht(positionen, dreiecke, N):
    """Jede innere Kante muss von genau 2 Dreiecken geteilt werden, jede
    Aussenkante (beide Endpunkte auf demselben Rand des Gesamt-Quadrats) von
    genau 1 - alles andere ist ein Riss/T-Junction."""
    kanten = {}
    for a, b, c in dreiecke:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            kanten[key] = kanten.get(key, 0) + 1

    def ist_aussenkante(u, v):
        (x1, y1), (x2, y2) = positionen[u], positionen[v]
        return ((x1 == 0 and x2 == 0) or (x1 == N and x2 == N) or
                (y1 == 0 and y2 == 0) or (y1 == N and y2 == N))

    risse = []
    for (u, v), n in kanten.items():
        erwartet = 1 if ist_aussenkante(u, v) else 2
        if n != erwartet:
            risse.append((positionen[u], positionen[v], n))
    return risse


def test_precondition():
    assert ist_fuer_adaptives_mesh_geeignet(np.zeros((128, 128))) is True  # echte map_size dieses Projekts
    assert ist_fuer_adaptives_mesh_geeignet(np.zeros((1024, 1024))) is True
    assert ist_fuer_adaptives_mesh_geeignet(np.zeros((130, 130))) is False  # 130 keine 2er-Potenz
    assert ist_fuer_adaptives_mesh_geeignet(np.zeros((128, 256))) is False  # nicht quadratisch
    print("OK precondition")


def test_watertight_flat():
    H = np.zeros((128, 128), dtype=np.float32)
    roh = baue_adaptives_mesh_roh(H, fehler_toleranz_m=1.0, min_leaf_size=1)
    assert roh is not None
    positionen, dreiecke, blaetter, N = roh
    assert len(blaetter) == 1, f"flache Karte sollte 1 Blatt ergeben, war {len(blaetter)}"
    assert len(dreiecke) == 2
    risse = pruefe_wasserdicht(positionen, dreiecke, N)
    assert not risse, f"Risse gefunden: {risse[:5]}"
    print(f"OK watertight_flat ({len(dreiecke)} Dreiecke)")


def test_watertight_cliff():
    size = 128
    H = np.zeros((size, size), dtype=np.float32)
    H[:, size // 2:] = 500.0  # scharfe Klippe in der Mitte
    roh = baue_adaptives_mesh_roh(H, fehler_toleranz_m=5.0, min_leaf_size=1)
    assert roh is not None
    positionen, dreiecke, blaetter, N = roh
    risse = pruefe_wasserdicht(positionen, dreiecke, N)
    assert not risse, f"Risse an der Klippe: {risse[:5]}"
    voll = 2 * (size - 1) * (size - 1)
    assert len(dreiecke) < voll * 0.5, (
        f"Erwartete deutliche Reduktion abseits der Klippe, war {len(dreiecke)}/{voll}")
    print(f"OK watertight_cliff ({len(dreiecke)}/{voll} Dreiecke, {len(blaetter)} Blaetter)")


def test_watertight_random_low_tolerance():
    rng = np.random.RandomState(42)
    size = 64
    H = rng.uniform(0, 1000, size=(size, size)).astype(np.float32)
    roh = baue_adaptives_mesh_roh(H, fehler_toleranz_m=0.01, min_leaf_size=1)
    assert roh is not None
    positionen, dreiecke, blaetter, N = roh
    risse = pruefe_wasserdicht(positionen, dreiecke, N)
    assert not risse, f"Risse bei Zufallsrauschen (niedrige Toleranz): {risse[:5]}"
    print(f"OK watertight_random_low_tolerance ({len(dreiecke)} Dreiecke, {len(blaetter)} Blaetter)")


def test_watertight_random_high_tolerance():
    rng = np.random.RandomState(7)
    size = 64
    H = rng.uniform(0, 1000, size=(size, size)).astype(np.float32)
    roh = baue_adaptives_mesh_roh(H, fehler_toleranz_m=50.0, min_leaf_size=1)
    assert roh is not None
    positionen, dreiecke, blaetter, N = roh
    risse = pruefe_wasserdicht(positionen, dreiecke, N)
    assert not risse, f"Risse bei Zufallsrauschen (hohe Toleranz): {risse[:5]}"
    print(f"OK watertight_random_high_tolerance ({len(dreiecke)} Dreiecke, {len(blaetter)} Blaetter)")


def test_mixed_flat_and_detail():
    """Realistischeres Bild: grossteils flaches 'Meer', eine detaillierte 'Kueste'-Ecke."""
    size = 256
    H = np.zeros((size, size), dtype=np.float32)
    rng = np.random.RandomState(3)
    ecke = rng.uniform(0, 800, size=(64, 64)).astype(np.float32)
    H[:64, :64] = ecke
    roh = baue_adaptives_mesh_roh(H, fehler_toleranz_m=8.0, min_leaf_size=1)
    assert roh is not None
    positionen, dreiecke, blaetter, N = roh
    risse = pruefe_wasserdicht(positionen, dreiecke, N)
    assert not risse, f"Risse im gemischten Bild: {risse[:5]}"
    voll = 2 * (size - 1) * (size - 1)
    reduktion = len(dreiecke) / voll
    assert reduktion < 0.15, f"Erwartete starke Reduktion, war {reduktion:.3f} ({len(dreiecke)}/{voll})"
    print(f"OK mixed_flat_and_detail ({len(dreiecke)}/{voll} Dreiecke, Reduktion {reduktion:.3%})")


def test_determinism():
    rng = np.random.RandomState(99)
    H = rng.uniform(0, 500, size=(128, 128)).astype(np.float32)
    roh1 = baue_adaptives_mesh_roh(H, fehler_toleranz_m=6.0, min_leaf_size=1)
    roh2 = baue_adaptives_mesh_roh(H, fehler_toleranz_m=6.0, min_leaf_size=1)
    assert roh1[0] == roh2[0]
    assert roh1[1] == roh2[1]
    print("OK determinism")


def test_full_vertex_array_shape():
    rng = np.random.RandomState(5)
    H = rng.uniform(0, 500, size=(128, 128)).astype(np.float32)
    ergebnis = build_adaptive_mesh(H, terrain_scale_factor=0.1, terrain_height_scale=0.001,
                                    fehler_toleranz_m=6.0, min_leaf_size=1)
    assert ergebnis is not None
    vertices, indices, stats = ergebnis
    assert vertices.dtype == np.float32
    assert indices.dtype == np.uint32
    assert vertices.size % 8 == 0
    anzahl_vertices = vertices.size // 8
    assert indices.max() < anzahl_vertices
    assert indices.min() >= 0
    assert np.isfinite(vertices).all()
    normalen = vertices.reshape(-1, 8)[:, 3:6]
    laengen = np.linalg.norm(normalen, axis=1)
    assert np.allclose(laengen, 1.0, atol=1e-4), "Normalen sind nicht normiert"
    print(f"OK full_vertex_array_shape (stats={stats})")


def test_unsuitable_returns_none():
    H = np.zeros((130, 130), dtype=np.float32)
    assert baue_adaptives_mesh_roh(H, 5.0) is None
    assert build_adaptive_mesh(H, 0.1, 0.001, 5.0) is None
    print("OK unsuitable_returns_none")


def test_performance_realistic_size():
    size = 512  # typische map_size-Groessenordnung dieses Projekts
    rng = np.random.RandomState(11)
    # Realistischeres Terrain: glattes Rauschen statt weissem Rauschen, sonst
    # ist JEDER Punkt maximal detailliert und die adaptive Reduktion greift nicht.
    from scipy.ndimage import gaussian_filter
    roh_rauschen = rng.uniform(0, 1, size=(size, size)).astype(np.float32)
    H = gaussian_filter(roh_rauschen, sigma=8.0) * 1000.0
    # eine scharfe Klippe einbauen, damit auch echte Detailbereiche vorkommen
    H[200:260, 200:260] += 400.0

    start = time.time()
    roh = baue_adaptives_mesh_roh(H, fehler_toleranz_m=6.0, min_leaf_size=1)
    dauer = time.time() - start
    assert roh is not None
    positionen, dreiecke, blaetter, N = roh
    risse = pruefe_wasserdicht(positionen, dreiecke, N)
    assert not risse, f"Risse bei realistischer Groesse: {risse[:5]}"
    voll = 2 * (size - 1) * (size - 1)
    print(f"OK performance_realistic_size: {dauer:.2f}s, {len(dreiecke)}/{voll} Dreiecke "
          f"({len(dreiecke)/voll:.1%}), {len(blaetter)} Blaetter")
    assert dauer < 30.0, f"Zu langsam fuer eine Regenerierung: {dauer:.2f}s"


if __name__ == "__main__":
    test_precondition()
    test_watertight_flat()
    test_watertight_cliff()
    test_watertight_random_low_tolerance()
    test_watertight_random_high_tolerance()
    test_mixed_flat_and_detail()
    test_determinism()
    test_full_vertex_array_shape()
    test_unsuitable_returns_none()
    test_performance_realistic_size()
    print("\nAlle Tests bestanden.")
