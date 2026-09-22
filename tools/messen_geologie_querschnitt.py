"""
Wegwerf-Messskript zu Ticket #62 ("Den Geologie-Querschnitt groeber
rechnen"), nicht Teil des Programms.

Misst die Rechenzeit von MapDisplay2D._render_geology_cross_section() bei
512 und 1024 px, mit synthetischen aber realistisch geformten Daten
(N_LAYERS=13 Schichtgrenzen, Terrainhoehe, Intrusions-Abstandskarte), und
vergleicht sie mit einer auf "ziel_punkte" Stuetzstellen reduzierten
Variante. Ergebnis und Interpretation stehen in docs/archiv/2026-08-25_AUFRAEUMPLAN.md
Abschnitt 4.6.

Laeuft OHNE QApplication: _render_geology_cross_section() benutzt aus
`self` ausschliesslich `self.ax` (siehe Quelltext) - ein reines
matplotlib-Axes-Objekt (Agg-Backend) reicht als Attrappe. Deshalb ist
QT_QPA_PLATFORM=offscreen hier unproblematisch (siehe CLAUDE.md-Warnung
dazu: die betrifft nur GPUWorker/QOffscreenSurface/QOpenGLContext, die
hier gar nicht beteiligt sind - reines matplotlib, kein Shader-Dispatch).

Aufruf (aus der Projektwurzel):
    .venv/Scripts/python.exe .scratch/geologie-querschnitt-messung/messen_querschnitt.py
"""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gui.widgets.map_display_2d import MapDisplay2D
from core.geology_layers import N_LAYERS


def baue_daten(size, seed=20260917):
    rng = np.random.default_rng(seed)
    H = W = size

    # Terrain: Rauschsumme, grob wie eine echte Heightmap (mehrere Oktaven)
    terrain = np.zeros((H, W), dtype=np.float64)
    coordsY, coordsX = np.meshgrid(np.linspace(0, 8, H), np.linspace(0, 8, W), indexing="ij")
    for octave, amp in [(1, 400.0), (2, 200.0), (4, 80.0), (8, 30.0)]:
        terrain += amp * np.sin(coordsX * octave + rng.uniform(0, 6)) * np.cos(coordsY * octave * 1.3 + rng.uniform(0, 6))
    terrain += 300.0  # ungefaehre mittlere Hoehe

    # Schichtgrenzen: N_LAYERS aufsteigende Flaechen, leicht deformiert/gefaltet,
    # wie boundaries_deformed = boundaries_base + stack_deformation in
    # core/geology_generator.py._compute_outcrop().
    layer_boundaries = np.zeros((N_LAYERS, H, W), dtype=np.float32)
    base = -3000.0
    for i in range(N_LAYERS):
        base += rng.uniform(100.0, 350.0)
        fold = 60.0 * np.sin(coordsX * (0.5 + 0.1 * i) + i) + 40.0 * np.cos(coordsY * 0.7 + i)
        layer_boundaries[i] = base + fold

    # Intrusions-Abstandskarte: signierter Abstand in km, negativ = innerhalb
    cx, cy = W * 0.6, H * 0.4
    dist_px = np.sqrt((coordsX * W / 8 - cx) ** 2 + (coordsY * H / 8 - cy) ** 2)
    map_distance_km = 21.3
    dist_km = dist_px * (map_distance_km / W)
    intrusion_distance_map = (dist_km - 3.0).astype(np.float32)  # Blob mit Radius ~3km

    return layer_boundaries, terrain.astype(np.float32), intrusion_distance_map


class _FakeDisplay:
    """Attrappe fuer MapDisplay2D: nur `.ax` wird von
    _render_geology_cross_section() gelesen/beschrieben."""
    def __init__(self, ax):
        self.ax = ax


def render_original(fake, payload):
    MapDisplay2D._render_geology_cross_section(fake, payload)


def render_grob(fake, payload, ziel_punkte):
    """Wie render_original(), aber das 1D-Profil wird VOR dem Plotten auf
    `ziel_punkte` gleichverteilte Stuetzstellen entlang der Schnittlinie
    reduziert (np.linspace-Indizes, keine Mittelung - erhaelt scharfe
    Kanten/Faults besser als eine gemittelte Dezimierung)."""
    from core.geology_layers import ALL_ROCK_TYPES, BASALT_INTRUSION

    layer_boundaries = payload["layer_boundaries"]
    terrain_height = payload["terrain_height"]
    intrusion_distance_map = payload.get("intrusion_distance_map")
    axis = payload.get("axis", "x")
    position = float(np.clip(payload.get("position", 0.5), 0.0, 1.0))

    height, width = terrain_height.shape[:2]
    if axis == "x":
        row = int(round(position * (height - 1)))
        boundaries_slice = layer_boundaries[:, row, :]
        terrain_slice = terrain_height[row, :].astype(np.float64)
        intrusion_slice = intrusion_distance_map[row, :] if intrusion_distance_map is not None else None
        full_n = width
        x_label = f"X (row Y={row})"
    else:
        col = int(round(position * (width - 1)))
        boundaries_slice = layer_boundaries[:, :, col]
        terrain_slice = terrain_height[:, col].astype(np.float64)
        intrusion_slice = intrusion_distance_map[:, col] if intrusion_distance_map is not None else None
        full_n = height
        x_label = f"Y (col X={col})"

    n = min(ziel_punkte, full_n)
    idx = np.linspace(0, full_n - 1, n).round().astype(np.int64)
    coord = idx.astype(np.float64)
    boundaries_slice = boundaries_slice[:, idx]
    terrain_slice = terrain_slice[idx]
    if intrusion_slice is not None:
        intrusion_slice = intrusion_slice[idx]

    ax = fake.ax
    ax.set_aspect("auto")
    stack_floor = float(np.min(boundaries_slice)) - 50.0
    root_floor = stack_floor
    lower_true = np.full_like(terrain_slice, stack_floor)
    for i, layer in enumerate(ALL_ROCK_TYPES[:N_LAYERS]):
        upper_true = boundaries_slice[i].astype(np.float64)
        lower_display = np.minimum(lower_true, terrain_slice)
        upper_display = np.minimum(upper_true, terrain_slice)
        color = tuple(c / 255.0 for c in layer.color)
        ax.fill_between(coord, lower_display, upper_display, color=color, label=layer.name, linewidth=0)
        lower_true = upper_true

    if intrusion_slice is not None:
        inside = intrusion_slice < 0
        if np.any(inside):
            penetration = np.clip(-intrusion_slice, 0.0, None)
            root_depth = stack_floor - penetration * 300.0
            root_floor = float(np.min(root_depth[inside]))
            basalt_color = tuple(c / 255.0 for c in BASALT_INTRUSION.color)
            ax.fill_between(coord, root_depth, terrain_slice, where=inside,
                             color=basalt_color, label=BASALT_INTRUSION.name, linewidth=0)

    ax.plot(coord, terrain_slice, color="black", linewidth=1.5, label="Terrain")
    y_min = min(float(np.min(terrain_slice)), float(np.min(boundaries_slice[0])), root_floor) - 50.0
    y_max = max(float(np.max(terrain_slice)), float(np.max(boundaries_slice[-1]))) + 50.0
    ax.set_xlim(float(coord[0]), float(coord[-1]))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Elevation (m)")
    ax.legend(loc="upper right", fontsize=6, ncol=2)


def zeitmessung(fn, wiederholungen=60):
    """Misst NUR den Redraw, mit wiederverwendeter Figure/Achse - wie die
    echte App: map_display_2d.py legt Figure/Canvas einmal pro Tab an und
    ruft je Update nur self.ax.clear() + _render_geology_cross_section().
    Ein neues Figure je Iteration wuerde die matplotlib-Fixkosten
    (Objektaufbau) mitmessen und den datengroessenabhaengigen Anteil
    verdecken."""
    fig = plt.figure(figsize=(12, 8), dpi=100)
    ax = fig.add_subplot(111)
    fake = _FakeDisplay(ax)
    zeiten = []
    for _ in range(wiederholungen):
        ax.clear()
        t0 = time.perf_counter()
        fn(fake)
        t1 = time.perf_counter()
        zeiten.append(t1 - t0)
    plt.close(fig)
    return zeiten


def bild_speichern(fn):
    fig = plt.figure(figsize=(12, 8), dpi=100)
    ax = fig.add_subplot(111)
    fake = _FakeDisplay(ax)
    fn(fake)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


if __name__ == "__main__":
    # Canvas ist 12in x 100dpi = 1200 px breit (Figure(figsize=(12,8), dpi=100)
    # in map_display_2d.py) - das ist die tatsaechlich SICHTBARE Aufloesung
    # der X-Achse. ziel_punkte etwas darunter, hier 300, als konkreter
    # Testwert fuer "grob" (Faktor ~1.7-3.4 unter der Kartenaufloesung).
    ZIEL_PUNKTE = 300

    for size in (512, 1024):
        layer_boundaries, terrain, intrusion = baue_daten(size)
        payload = {
            "layer_boundaries": layer_boundaries,
            "terrain_height": terrain,
            "intrusion_distance_map": intrusion,
            "axis": "x",
            "position": 0.5,
        }

        zeiten_original = zeitmessung(lambda fake: render_original(fake, payload))
        zeiten_grob = zeitmessung(lambda fake: render_grob(fake, payload, ZIEL_PUNKTE))

        med_orig = np.median(zeiten_original) * 1000
        med_grob = np.median(zeiten_grob) * 1000
        min_orig = min(zeiten_original) * 1000
        min_grob = min(zeiten_grob) * 1000
        print(f"\n=== map_size={size} ===")
        print(f"Original   (n={size:4d} Punkte): median={med_orig:.2f} ms  min={min_orig:.2f} ms  "
              f"max={max(zeiten_original)*1000:.2f} ms  std={np.std(zeiten_original)*1000:.2f} ms")
        print(f"Grob       (n={min(ZIEL_PUNKTE,size):4d} Punkte): median={med_grob:.2f} ms  min={min_grob:.2f} ms  "
              f"max={max(zeiten_grob)*1000:.2f} ms  std={np.std(zeiten_grob)*1000:.2f} ms")
        ersparnis_median = (1 - med_grob / med_orig) * 100 if med_orig > 0 else 0.0
        ersparnis_min = (1 - min_grob / min_orig) * 100 if min_orig > 0 else 0.0
        print(f"Ersparnis (Median): {ersparnis_median:.1f} %  (absolut {med_orig-med_grob:.2f} ms je Redraw)")
        print(f"Ersparnis (Min/Bestfall, ruhigstes Signal): {ersparnis_min:.1f} %  (absolut {min_orig-min_grob:.2f} ms je Redraw)")

        # Bildvergleich (visuelle Pruefung "sieht unveraendert aus")
        bild_orig = bild_speichern(lambda fake: render_original(fake, payload))
        bild_grob = bild_speichern(lambda fake: render_grob(fake, payload, ZIEL_PUNKTE))
        diff = np.abs(bild_orig.astype(np.int32) - bild_grob.astype(np.int32))
        print(f"Bildvergleich: mean|diff|={diff.mean():.3f}  max|diff|={diff.max()}  "
              f"Anteil Pixel mit diff>10: {(diff.max(axis=-1) > 10).mean()*100:.3f} %")
