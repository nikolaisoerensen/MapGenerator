"""
Path: scratch_erosion_lab.py

WERKZEUG, kein Test. Fährt Parameter-Varianten der Feld-Erosion über die GPU,
misst sie mit Kennzahlen, die das AUSSEHEN beschreiben, und schreibt einen
Kontaktabzug zum Anschauen.

WARUM ES DAS BRAUCHT: die bisherigen Qualitätskennzahlen (Top-5%-Anteil,
Kanalnetz, Ebenenanteil) haben nicht bemerkt, dass die Böschungserosion das
ganze Gelände in 45°-Pyramiden verwandelt hatte. Sie messen, WIE VIEL erodiert
wurde und WO - nicht, ob das Ergebnis wie eine Landschaft aussieht. Ein Bild
sagte in einer Sekunde, was fünf Zahlen verschwiegen haben.

Die vier Kennzahlen hier sind deshalb bewusst auf FORM ausgelegt:

  anisotropy       Richtungsabhängigkeit der Hänge. Gittereffekte erzeugen
                   Vorzugsrichtungen bei 0/45/90 Grad; eine echte Landschaft
                   ist richtungslos. Das ist die Zahl, die die Pyramiden
                   gefunden hätte.
  drainage_density Anteil der Fläche, der zu einem Gerinne gehört - das
                   klassische Mass für "wie fein ist die Landschaft zertalt".
  slope_area_beta  Steigung der log-log-Beziehung zwischen Einzugsgebiet und
                   Hangneigung. In fluvial geformten Landschaften liegt sie
                   um -0.5; sie ist die Standard-Signatur dafür, dass Wasser
                   und nicht Rauschen das Gelände geformt hat.
  hypsometric      Lage der Höhenverteilung zwischen Minimum und Maximum.
                   Junge Landschaften liegen hoch (>0.5), reife tief (<0.45).

Aufruf:
    .venv\\Scripts\\python.exe scratch_erosion_lab.py            # Standard-Sweep
    .venv\\Scripts\\python.exe scratch_erosion_lab.py baseline   # nur Referenz
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "erosion_lab")


# =============================================================================
# KENNZAHLEN
# =============================================================================

def anisotropy(terrain):
    """
    Wie stark bevorzugen die Hänge bestimmte Richtungen? 0 = richtungslos.

    Gittereffekte (4-Nachbarschaft) erzeugen Häufungen bei 0/45/90 Grad. Weil
    eine Hangrichtung und ihre Gegenrichtung dieselbe Struktur beschreiben,
    wird der Winkel modulo 90 Grad genommen und über 18 Fächer histogrammiert;
    zurückgegeben wird die Streuung relativ zum Mittel.

    Referenzwerte: das unerodierte Simplex-Terrain liegt bei ~0.05, das
    Pyramiden-Gelände der 4-Nachbar-Böschungserosion lag bei ~0.35.
    """
    gy, gx = np.gradient(terrain.astype(np.float64))
    magnitude = np.hypot(gx, gy)
    strong = magnitude > np.percentile(magnitude, 60)
    if strong.sum() < 100:
        return 0.0
    angles = np.degrees(np.arctan2(gy[strong], gx[strong])) % 90.0
    histogram, _ = np.histogram(angles, bins=18, range=(0.0, 90.0))
    share = histogram / max(histogram.sum(), 1)
    return float(share.std() / max(share.mean(), 1e-12))


def flow_accumulation(terrain):
    """
    D8-Einzugsgebiet je Zelle (in Zellen), von hoch nach tief aufsummiert.

    Bewusst D8 und nicht das Pipe-Modell: hier soll die FORM des fertigen
    Geländes vermessen werden, unabhängig davon, womit sie erzeugt wurde.
    """
    height, width = terrain.shape
    accumulation = np.ones(terrain.size, dtype=np.float64)
    flat = terrain.astype(np.float64).ravel()
    order = np.argsort(flat)[::-1]

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    distances = [np.sqrt(2), 1.0, np.sqrt(2), 1.0, 1.0, np.sqrt(2), 1.0, np.sqrt(2)]

    for index in order:
        y, x = divmod(int(index), width)
        best_slope, best_target = 0.0, -1
        for (dy, dx), distance in zip(offsets, distances):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            slope = (flat[index] - flat[ny * width + nx]) / distance
            if slope > best_slope:
                best_slope, best_target = slope, ny * width + nx
        if best_target >= 0:
            accumulation[best_target] += accumulation[index]
    return accumulation.reshape(terrain.shape)


def drainage_density(accumulation, channel_threshold=25.0):
    """Flächenanteil, dessen Einzugsgebiet gross genug für ein Gerinne ist."""
    return float((accumulation >= channel_threshold).mean())


def slope_area_beta(terrain, accumulation, meters_per_pixel):
    """
    Exponent der log-log-Beziehung Hangneigung ~ Einzugsgebiet^beta.

    In fluvial geformten Landschaften ist er deutlich negativ (typisch -0.3 bis
    -0.7): je groesser das Einzugsgebiet, desto flacher der Kanal. Bei reinem
    Rauschen liegt er nahe 0 - es gibt dort keinen Zusammenhang zwischen
    Einzugsgebiet und Neigung, weil kein Wasser das Gelaende geformt hat.
    Diese Zahl unterscheidet also "erodiert" von "nur verrauscht".
    """
    gy, gx = np.gradient(terrain.astype(np.float64))
    slope = np.hypot(gx, gy) / meters_per_pixel
    mask = (accumulation >= 10.0) & (slope > 1e-6)
    if mask.sum() < 200:
        return 0.0
    x = np.log10(accumulation[mask])
    y = np.log10(slope[mask])
    return float(np.polyfit(x, y, 1)[0])


def hypsometric_integral(terrain):
    """(Mittel - Minimum) / (Maximum - Minimum). Reife Landschaften < 0.45."""
    low, high = float(terrain.min()), float(terrain.max())
    if high - low < 1e-9:
        return 0.5
    return (float(terrain.mean()) - low) / (high - low)


def measure(terrain, meters_per_pixel):
    accumulation = flow_accumulation(terrain)
    return {
        "anisotropy": anisotropy(terrain),
        "drainage": drainage_density(accumulation),
        "beta": slope_area_beta(terrain, accumulation, meters_per_pixel),
        "hypso": hypsometric_integral(terrain),
    }


# =============================================================================
# AUFBAU
# =============================================================================

# FESTER Seed fuer alle Laeufe dieses Werkzeugs.
#
# TERRAIN.MAP_SEED wird bei JEDEM Programmstart neu gewuerfelt (siehe
# value_default.py). Fuer die App ist das richtig, fuer eine Messreihe ist es
# fatal: zwei Sweeps liefen auf verschiedenen Gelaenden (Relief 1486 gegen
# 1566 m) und waren damit nicht vergleichbar - dieselbe Variante bekam einmal
# Anisotropie 0.159 und einmal 0.458.
LAB_SEED = 424242


def build_terrain(size=256, lod=4, terrain_overrides=None):
    """Gelaende aus dem ECHTEN Terrain-Generator mit den echten Defaults,
    aber festem Seed (siehe LAB_SEED)."""
    from gui.OldManagers.data_lod_manager import DataLODManager
    from gui.config.value_default import TERRAIN
    from core.terrain_generator import BaseTerrainGenerator

    manager = DataLODManager()
    manager.set_map_distance_km(TERRAIN.MAP_DISTANCE_KM["default"])
    parameters = {key.lower(): getattr(TERRAIN, key)["default"] for key in
                  ("AMPLITUDE", "OCTAVES", "FREQUENCY", "PERSISTENCE",
                   "LACUNARITY", "REDISTRIBUTE_POWER", "MAP_SEED")}
    parameters["map_size"] = size
    parameters["map_seed"] = LAB_SEED
    parameters.update(terrain_overrides or {})

    generator = BaseTerrainGenerator(data_lod_manager=manager)
    generator.set_active_parameters(parameters)
    for node in ("terrain.noise", "terrain.redistribution"):
        manager.set_calculator_target_lod(node, lod)
    generator._calc_noise("terrain.noise", lod)
    generator._calc_redistribution("terrain.redistribution", lod)
    heightmap = manager.get_calculator_output("terrain.redistribution", "heightmap", lod)
    return heightmap.astype(np.float32), TERRAIN.MAP_DISTANCE_KM["default"] * 1000.0 / size


def default_parameters():
    from gui.config.value_default import EROSION
    values = {key.lower(): getattr(EROSION, key)["default"]
              for key in dir(EROSION) if key.isupper()}
    values["thermal_variant"] = "gather"
    return values


# =============================================================================
# SWEEP
# =============================================================================

def run_variants(variants, size=256, terrain_overrides=None):
    from PyQt6.QtGui import QGuiApplication
    application = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841
    from gui.OldManagers.shader_manager import ShaderManager
    from core.erosion_generator import HydraulicFieldSimulator

    terrain, meters_per_pixel = build_terrain(size, terrain_overrides=terrain_overrides)
    shader_manager = ShaderManager()
    print("Terrain {}x{}, {:.0f} m/px, Relief {:.0f} m".format(
        size, size, meters_per_pixel, float(terrain.max() - terrain.min())))

    baseline = measure(terrain, meters_per_pixel)
    print("\n{:34s} {:>7s} {:>9s} {:>7s} {:>7s} {:>7s} {:>6s}".format(
        "Variante", "Aniso", "Drainage", "beta", "hypso", "Schritte", "Zeit"))
    print("{:34s} {:7.3f} {:9.3f} {:7.3f} {:7.3f} {:>7s} {:>6s}".format(
        "(unerodiert)", baseline["anisotropy"], baseline["drainage"],
        baseline["beta"], baseline["hypso"], "-", "-"))

    results = [("unerodiert", terrain, baseline)]
    for label, overrides in variants:
        parameters = default_parameters()
        # Schluessel in GROSSBUCHSTABEN sind Klassen-KONSTANTEN des Simulators
        # (Schwellen, Bezugsgroessen), keine Slider. Sie ueberschreiben zu
        # koennen ist der Unterschied zwischen "am Regler drehen" und "am
        # Modell schrauben" - ein Kalibrier-Werkzeug muss beides erlauben.
        constants = {k: v for k, v in overrides.items() if k.isupper()}
        parameters.update({k: v for k, v in overrides.items() if not k.isupper()})

        simulator = HydraulicFieldSimulator(shader_manager=shader_manager)
        for name, value in constants.items():
            assert hasattr(simulator, name), "unbekannte Konstante " + name
            setattr(simulator, name, value)

        started = time.time()
        outcome = simulator.simulate(
            terrain, np.full(terrain.shape, 50.0, dtype=np.float32),
            parameters, meters_per_pixel)
        after = (terrain.astype(np.float64)
                 - outcome["erosion_map"] + outcome["sedimentation_map"])
        metrics = measure(after, meters_per_pixel)
        print("{:34s} {:7.3f} {:9.3f} {:7.3f} {:7.3f} {:7d} {:5.0f}s".format(
            label, metrics["anisotropy"], metrics["drainage"], metrics["beta"],
            metrics["hypso"], outcome["steps_taken"], time.time() - started))
        results.append((label, after, metrics))
    return results, meters_per_pixel


def contact_sheet(results, filename):
    """Alle Varianten als schattiertes Relief nebeneinander."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    light = LightSource(azdeg=315, altdeg=45)
    columns = min(4, len(results))
    rows = (len(results) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(7 * columns, 7 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, (label, terrain, metrics) in zip(axes, results):
        axis.imshow(light.shade(terrain, cmap=plt.cm.terrain,
                                vert_exag=0.15, blend_mode="overlay"))
        axis.set_title("{}\nAniso {:.3f} | Drainage {:.3f} | beta {:.2f}".format(
            label, metrics["anisotropy"], metrics["drainage"], metrics["beta"]),
            fontsize=11)
        axis.axis("off")
    for axis in axes[len(results):]:
        axis.axis("off")

    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=55, bbox_inches="tight")
    print("\nKontaktabzug: {}".format(path))
    return path


# Der Standard-Sweep. Jede Zeile ist eine Hypothese darueber, was dem Zielbild
# naeher kommt - nicht ein blindes Abrastern des Parameterraums.
STANDARD_SWEEP = [
    ("Default", {}),
    ("ohne Boeschung", {"thermal_strength": 0.0}),
    ("ohne Glaettung", {"smoothing": 0.0}),
    ("mehr Abtrag (Ks 1.5)", {"erosion_strength": 1.5}),
    ("mehr Ablagerung (Kd 1.5)", {"deposition_rate": 1.5}),
    ("mehr Kapazitaet (Kc 3)", {"erosion_capacity": 3.0}),
    ("mehr Regen (5)", {"rainfall": 5.0}),
    ("laenger (Konvergenz 1e-7)", {"convergence_threshold": 1e-7}),
]

# Zweiter Sweep, aus dem Ergebnis des ersten abgeleitet: die Erosion war dort
# durchweg ZU SCHWACH - alle Varianten sahen dem unerodierten Gelaende aehnlich,
# waehrend das Zielbild vollstaendig von der Erosion gepraegt ist. Sichtbare
# Verzweigung zeigten nur die Laeufe mit mehr Wasser und mehr Kapazitaet.
# Hier wird deshalb in genau diese Richtung gedrueckt, inklusive der
# Modell-Konstanten, die den Abtrag begrenzen.
STRONGER_SWEEP = [
    ("Default (Referenz)", {}),
    ("Regen 5 + Kc 3", {"rainfall": 5.0, "erosion_capacity": 3.0}),
    ("Regen 5 + Kc 3, ohne Boeschung",
     {"rainfall": 5.0, "erosion_capacity": 3.0, "thermal_strength": 0.0}),
    ("Regen 5, Schwelle 0.2",
     {"rainfall": 5.0, "EROSION_THRESHOLD_DISCHARGE": 0.2}),
    ("Regen 5, Bezugssaeule 5 m",
     {"rainfall": 5.0, "CAPACITY_REFERENCE_M": 5.0}),
    ("Regen 5, Kc 3, Saeule 5 m, lang",
     {"rainfall": 5.0, "erosion_capacity": 3.0, "CAPACITY_REFERENCE_M": 5.0,
      "convergence_threshold": 1e-7}),
    ("alles stark, ohne Boeschung",
     {"rainfall": 5.0, "erosion_capacity": 3.0, "CAPACITY_REFERENCE_M": 5.0,
      "EROSION_THRESHOLD_DISCHARGE": 0.2, "thermal_strength": 0.0,
      "convergence_threshold": 1e-7}),
    ("alles stark, mit Boeschung",
     {"rainfall": 5.0, "erosion_capacity": 3.0, "CAPACITY_REFERENCE_M": 5.0,
      "EROSION_THRESHOLD_DISCHARGE": 0.2, "convergence_threshold": 1e-7}),
]

# Dritter Sweep, aus Sweep 2 abgeleitet: mehr Kraft hat das Gelaende nicht
# zertalt, sondern PLANIERT (hypsometrischer Wert 0.249 -> 0.485, die Taeler
# wurden zugeschuettet). Das Material bleibt gleich nebenan liegen, statt die
# Karte zu verlassen. Der Regler dafuer ist die Ablagerungsrate: je kleiner,
# desto laenger bleibt die Fracht in Schwebe und desto weiter kommt sie.
TRANSPORT_SWEEP = [
    ("Default (Referenz)", {}),
    ("Kd 0.05 - Fracht bleibt lange", {"deposition_rate": 0.05}),
    ("Kd 0.05 + Regen 5", {"rainfall": 5.0, "deposition_rate": 0.05}),
    ("Kd 0.05 + Regen 5 + Kc 3",
     {"rainfall": 5.0, "deposition_rate": 0.05, "erosion_capacity": 3.0}),
    ("Kd 0.02 + Regen 5 + Kc 3, lang",
     {"rainfall": 5.0, "deposition_rate": 0.02, "erosion_capacity": 3.0,
      "convergence_threshold": 1e-7}),
    ("Kd 0.05, Schwelle 0.2, Regen 5",
     {"rainfall": 5.0, "deposition_rate": 0.05,
      "EROSION_THRESHOLD_DISCHARGE": 0.2}),
    ("Kd 0.05, Regen 5, ohne Boeschung",
     {"rainfall": 5.0, "deposition_rate": 0.05, "thermal_strength": 0.0}),
    ("Kd 0.05, Regen 5, ohne Glaettung",
     {"rainfall": 5.0, "deposition_rate": 0.05, "smoothing": 0.0}),
]

# Die aus Sweep 3 gewonnene beste Kombination - Ausgangspunkt fuer alles
# Weitere. Wenig Ablagerung (die Fracht soll die Karte verlassen, nicht
# nebenan liegenbleiben), niedrige Gerinne-Schwelle und mehr Wasser.
BEST_SO_FAR = {"rainfall": 5.0, "deposition_rate": 0.05,
               "EROSION_THRESHOLD_DISCHARGE": 0.2}

# Vierter Sweep: liegt es noch am EINGANGSGELAENDE? Der Default
# REDISTRIBUTE_POWER=3.5 druckt fast die ganze Karte in die Tiefe und laesst
# nur einzelne Gipfel stehen - im Kontaktabzug von Sweep 3 gut zu sehen als
# breite, glatte Becken, in denen es nichts mehr zu zertalen gibt. Das
# Zielbild ist dagegen durchgehend gebirgig.
TERRAIN_SWEEP = [
    ("beste Erosion, Terrain-Default", dict(BEST_SO_FAR)),
]

SWEEPS = {"standard": STANDARD_SWEEP, "stronger": STRONGER_SWEEP,
          "transport": TRANSPORT_SWEEP, "terrain": TERRAIN_SWEEP}

# Gelaende-Varianten fuer den vierten Sweep: dieselbe Erosion, anderes Relief.
TERRAIN_VARIANTS = [
    ("Redistribution 3.5 (Default)", {"redistribute_power": 3.5}),
    ("Redistribution 2.0", {"redistribute_power": 2.0}),
    ("Redistribution 1.2 - gebirgig", {"redistribute_power": 1.2}),
    ("Redistribution 1.2 + 6 Oktaven",
     {"redistribute_power": 1.2, "octaves": 6}),
]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "standard"
    if name == "abnahme":
        # Abnahme: die AUSGELIEFERTEN Defaults, ohne jede Uebersteuerung.
        # Alles, was oben als Uebersteuerung gewonnen wurde, steht jetzt in
        # gui/config/value_default.py bzw. als Konstante im Simulator - dieser
        # Lauf beweist, dass das auch wirklich so angekommen ist.
        results, _ = run_variants([("Defaults 2026-07-28", {})], size=512,
                                  terrain_overrides={"redistribute_power": 1.2})
        contact_sheet(results, "abnahme_defaults.png")
        return 0
    if name == "schwelle":
        # Gemessen: der Durchfluss-Median ist bei 128/256/512 px praktisch
        # gleich (11.73 / 11.91 / 11.79) - die Schwelle haengt also NICHT an
        # der Aufloesung, meine Vermutung dazu war falsch. Sie haengt am
        # REGEN: mit Rate 5.0 liegt der Median bei ~11.8, da beisst eine
        # Schwelle von 0.1 oder 0.6 kaum noch. Also direkt vergleichen, mit
        # allen anderen Werten auf dem neuen Stand.
        base = dict(BEST_SO_FAR, thermal_strength=0.0, smoothing=0.0)
        variants = [("Schwelle {}".format(t),
                     dict(base, EROSION_THRESHOLD_DISCHARGE=t))
                    for t in (0.6, 0.4, 0.2, 0.1)]
        results, _ = run_variants(variants, size=512,
                                  terrain_overrides={"redistribute_power": 1.2})
        contact_sheet(results, "sweep_schwelle.png")
        return 0
    if name == "boeschung":
        # Boeschung 0.05 loescht die Rinnen aus. Grund ist der Massstab: die
        # Schwelle ist `Zellbreite * tan(Winkel)` = 20 m * tan(30 Grad) ~ 11 m,
        # eine frisch eingeschnittene Rinne ist tiefer - und rutscht deshalb
        # sofort wieder zu. Frage: rettet ein steilerer Winkel sie?
        base = dict(BEST_SO_FAR, thermal_strength=0.0, smoothing=0.0,
                    EROSION_THRESHOLD_DISCHARGE=0.1)
        variants = [
            ("Schwelle 0.1, ohne Boeschung", dict(base)),
            ("Schwelle 0.05", dict(base, EROSION_THRESHOLD_DISCHARGE=0.05)),
            ("Boeschung 0.15, Winkel x2.0",
             dict(base, thermal_strength=0.15, talus_angle_scale=2.0)),
            ("Boeschung 0.05, Winkel x2.0",
             dict(base, thermal_strength=0.05, talus_angle_scale=2.0)),
        ]
        results, _ = run_variants(variants, size=512,
                                  terrain_overrides={"redistribute_power": 1.2})
        contact_sheet(results, "sweep_boeschung.png")
        return 0
    if name == "finale":
        # Die Gegenprobe "verursacher" hat das Zielbild geliefert: OHNE
        # Boeschung und OHNE Glaettung entsteht die dichte, verzweigte
        # Zertalung ueber die ganze Karte. Beide Passes verschmieren die
        # Rinnen, sobald sie eingeschnitten sind.
        #
        # WICHTIG - meine Kennzahlen haben das falsch bewertet: die
        # Siegervariante lag bei Drainage 0.177 (schlechtester Wert) und
        # Aniso 0.454. Nur `beta` hat sie erkannt (-0.419, mit Abstand die
        # klarste fluviale Signatur). Die Drainage-Dichte haengt an einer
        # festen Schwelle auf die Fliessakkumulation und misst deshalb eher,
        # wie viel Wasser sich buendelt, nicht wie fein das Netz ist. Der
        # Kontaktabzug bleibt das entscheidende Instrument.
        #
        # Hier: wieviel Boeschung/Glaettung vertraegt das Bild noch?
        base = dict(BEST_SO_FAR, thermal_strength=0.0, smoothing=0.0)
        variants = [
            ("Ziel: ohne beides", dict(base)),
            ("+ Boeschung 0.05", dict(base, thermal_strength=0.05)),
            ("+ Boeschung 0.15", dict(base, thermal_strength=0.15)),
            ("+ Glaettung 0.1", dict(base, smoothing=0.1)),
            ("Schwelle 0.1 statt 0.2",
             dict(base, EROSION_THRESHOLD_DISCHARGE=0.1)),
            ("Regen 10 statt 5", dict(base, rainfall=10.0)),
        ]
        results, _ = run_variants(variants, size=512,
                                  terrain_overrides={"redistribute_power": 1.2})
        contact_sheet(results, "sweep_finale.png")
        return 0
    if name == "verursacher":
        # Bei 512 px steigt die Anisotropie auf 0.580 (bei 256 px: 0.144).
        # Eine physikalische Erosion wird auf einem feineren Gitter nicht
        # richtungsabhaengiger - also ist es ein Gitterartefakt. Welcher Pass?
        # Dieselbe Gegenprobe, die schon die 45-Grad-Pyramiden gefunden hat:
        # Verdaechtige einzeln abschalten und die Kennzahl messen.
        variants = [
            ("alles an", dict(BEST_SO_FAR)),
            ("ohne Boeschung", dict(BEST_SO_FAR, thermal_strength=0.0)),
            ("ohne Glaettung", dict(BEST_SO_FAR, smoothing=0.0)),
            ("ohne beides", dict(BEST_SO_FAR, thermal_strength=0.0, smoothing=0.0)),
        ]
        results, _ = run_variants(variants, size=512,
                                  terrain_overrides={"redistribute_power": 1.2})
        contact_sheet(results, "sweep_verursacher.png")
        return 0
    if name == "aufloesung":
        # Letzte offene Frage: kommt die fehlende DICHTE des Zielbildes einfach
        # aus der Aufloesung? Bei 256 px und 39 m/px ist eine Rinne wenige
        # Zellen breit - unterhalb dessen kann das Gitter keine Verzweigung
        # mehr darstellen. Deshalb dieselben Parameter zweimal, nur die
        # Kantenlaenge verdoppelt. Gelaende: gebirgig, wie in Sweep 4 gewonnen.
        for px in (256, 512):
            results, _ = run_variants([("beste Erosion", dict(BEST_SO_FAR))],
                                      size=px,
                                      terrain_overrides={"redistribute_power": 1.2})
            contact_sheet(results, "sweep_aufloesung_{}.png".format(px))
        return 0
    if name == "terrain":
        # Hier variiert das GELAENDE, nicht die Erosion - deshalb ein eigener
        # Ablauf, der die beste bekannte Erosion konstant haelt.
        collected = []
        for label, overrides in TERRAIN_VARIANTS:
            results, _ = run_variants([("beste Erosion", dict(BEST_SO_FAR))],
                                      terrain_overrides=overrides)
            collected.append((label + " (unerodiert)", results[0][1], results[0][2]))
            collected.append((label + " (erodiert)", results[1][1], results[1][2]))
        contact_sheet(collected, "sweep_terrain.png")
        return 0
    variants = SWEEPS.get(name, STANDARD_SWEEP)
    results, _ = run_variants(variants)
    contact_sheet(results, "sweep_{}.png".format(name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
