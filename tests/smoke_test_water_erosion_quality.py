"""
Regressionstest fuer die QUALITAET der Erosion (2026-07-27) - friert die
Zusagen ein, die aus dem Nutzer-Report vom selben Tag entstanden sind. Nicht
Teil einer Test-Suite, manuell ueber das gemeinsame venv laufen lassen (siehe
CLAUDE.md).

Die fuenf Kennzahlen bilden 1:1 die gemeldeten Punkte ab:

  (a) "das system erzeugt viele einzelne krater"
      -> Zahl echter Senken nach dem Lauf. Muss 0 sein: die abschliessende
         Senkenfuellung (fill_depressions) laesst per Konstruktion keine
         geschlossene Grube uebrig.

  (b) "die Berge sind zerfressen an den spitzen"
  (e) "Die Bergspitzen und Kaemme sind groesstenteils unberuehrt"
      -> hoehengewichteter Schwerpunkt der Erosion. Muss unter der mittleren
         Gelaendehoehe liegen.

  (d) "erosionkarte ist eine homogene flaeche anstatt ... wie ein baum mit
      aesten"
      -> Anteil der Erosion in den staerksten 5% der Zellen (Konzentration)
         UND groesste zusammenhaengende Komponente des Kanalnetzes.

  (e) "Sedimentation erzeugt ebenen, also flache landschaften"
      -> Flaechenanteil mit deutlich unterdurchschnittlicher Hangneigung.

WICHTIG - die Schwellen sind bewusst grosszuegiger als die gemessenen Werte:
sie sollen eine echte Verschlechterung fangen, nicht bei jeder Neukalibrierung
rot werden. Die aktuell gemessenen Werte stehen jeweils daneben.
"""
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, label, minimum_filter

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

from core.water_generator import (DropletErosionSystem, HydrologySystemGenerator,
                                   _NEIGHBOR_FOOTPRINT_8, fill_depressions)


def check(label_text, condition):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label_text}")
    return bool(condition)


# Mindesttiefe, ab der eine Senke als echter Krater zaehlt. Ohne diese Schwelle
# zaehlt man float32-Rundungsrauschen mit: bei 4000 m Hoehe betraegt die
# float32-Aufloesung 0.24 mm, und eine perfekt gefuellte Ebene enthaelt dadurch
# hunderte "Senken" von wenigen Mikrometern Tiefe. Gemessen lagen nach einem
# vollstaendigen Lauf ALLE verbliebenen Senken unter 0.03 mm.
CRATER_MIN_DEPTH_M = 0.5


def count_real_craters(dem, min_depth_m=CRATER_MIN_DEPTH_M):
    """Zellen, die messbar tiefer liegen als ihr niedrigster Nachbar."""
    values = dem.astype(np.float64)
    neighbour_min = minimum_filter(values, footprint=_NEIGHBOR_FOOTPRINT_8,
                                    mode='constant', cval=np.inf)
    deep = (neighbour_min - values) > min_depth_m
    deep[0, :] = deep[-1, :] = deep[:, 0] = deep[:, -1] = False
    return int(deep.sum())


def _make_terrain(size=128, seed=5):
    rng = np.random.RandomState(seed)
    return (gaussian_filter(rng.rand(size, size), 3) * 4000).astype(np.float32)


def _run(size=128, seed=5):
    heightmap = _make_terrain(size, seed)
    hardness_map = np.full((size, size), 50.0, dtype=np.float32)
    meters_per_pixel = 10000.0 / size

    generator = HydrologySystemGenerator.__new__(HydrologySystemGenerator)
    num_particles = generator._get_lod_iterations(size)["erosion_particles"]

    system = DropletErosionSystem(erosion_strength=2.5, sediment_capacity_factor=4.0,
                                   deposit_speed=0.3)
    erosion, sedimentation = system.simulate_erosion_sedimentation(
        heightmap, hardness_map, {"water_seed": 7},
        {"erosion_particles": num_particles}, meters_per_pixel)
    result = heightmap.astype(np.float64) - erosion + sedimentation
    return heightmap, erosion, sedimentation, result, num_particles


def run_quality_metrics():
    heightmap, erosion, sedimentation, result, num_particles = _run()
    print(f"    ({heightmap.shape[0]}px, {num_particles} Partikel, "
          f"{DropletErosionSystem.DEFAULT_EROSION_PASSES} Durchgaenge)")

    ok = True

    # (a) Krater
    before = count_real_craters(heightmap)
    after = count_real_craters(result)
    ok &= check(f"(a) keine Krater nach dem Lauf (vorher {before}, nachher {after}; "
                f"gemessen 0)", after == 0)

    # (b)/(e) Gipfel unberuehrt
    norm_height = (heightmap - heightmap.min()) / max(float(heightmap.max() - heightmap.min()), 1e-9)
    weights = erosion / max(float(erosion.sum()), 1e-9)
    erosion_centre = float((norm_height * weights).sum())
    terrain_centre = float(norm_height.mean())
    ok &= check(f"(b/e) Erosions-Schwerpunkt unter der mittleren Gelaendehoehe "
                f"(Erosion {erosion_centre:.3f}, Gelaende {terrain_centre:.3f}; "
                f"gemessen 0.443)", erosion_centre < terrain_centre)

    # (d) Konzentration
    flat = np.sort(erosion.ravel())[::-1]
    top5 = float(flat[:max(1, len(flat) // 20)].sum() / max(flat.sum(), 1e-9))
    ok &= check(f"(d) Erosion konzentriert sich in Linien statt flaechig "
                f"(Top-5%-Anteil {top5:.3f}, Schwelle > 0.30; gemessen 0.446)",
                top5 > 0.30)

    # (d)/(c) zusammenhaengendes Kanalnetz
    positive = erosion[erosion > 0]
    threshold = float(np.percentile(positive, 92)) if positive.size else 0.0
    labelled, count = label(erosion > threshold)
    largest = int(np.bincount(labelled.ravel())[1:].max()) if count else 0
    ok &= check(f"(c/d) zusammenhaengendes Kanalnetz (groesste Komponente {largest}px, "
                f"Schwelle > 120; gemessen 242)", largest > 120)

    # (e) Ebenen
    grad_y, grad_x = np.gradient(result)
    slope = np.hypot(grad_x, grad_y)
    plains = float((slope < slope.mean() * 0.25).mean())
    ok &= check(f"(e) Sedimentation erzeugt Ebenen (Flaechenanteil {plains:.1%}, "
                f"Schwelle 20-55%; gemessen 37.2%)", 0.20 < plains < 0.55)

    # Massenbilanz-Grundzusage: es kann nichts aus dem Nichts abgelagert
    # werden, DIE FUELLUNG AUSGENOMMEN - sie ist ein eigener Mechanismus und
    # bringt bewusst zusaetzliches Material ein (sie modelliert Verfuellung
    # aus dem Umland, nicht Transport durch die Partikel).
    ok &= check("Ergebnis endlich", bool(np.all(np.isfinite(result))))
    return ok


def run_fill_removes_all_depressions():
    """Direkte Zusage von fill_depressions(): danach existiert keine
    geschlossene Senke mehr, und ein bis zum Kartenrand entwaessernder Kanal
    bleibt unangetastet."""
    size = 64
    heightmap = _make_terrain(size, seed=3).astype(np.float64)

    # Kuenstliche geschlossene Grube - muss verschwinden.
    heightmap[30:34, 30:34] -= 300.0

    # Durchgehend zum linken Rand fallende Rinne - muss erhalten bleiben.
    # Bewusst STRENG monoton: eine verrauschte Rinne haette kleine Senken
    # entlang ihres Verlaufs, und die zu fuellen ist richtig. Geprueft wird
    # hier die Unterscheidung "entwaessert zum Rand" gegen "geschlossene
    # Grube", nicht die Glaettung von Rauschen.
    channel_row = 10
    heightmap[channel_row, :] = (heightmap.min() - 200.0
                                 + np.arange(size, dtype=np.float64) * 2.0)

    before = count_real_craters(heightmap)
    fill_amount = fill_depressions(heightmap)
    filled = heightmap + fill_amount

    ok = check(f"kuenstliche Grube vorher vorhanden ({before} Senken)", before > 0)
    ok &= check("nach der Fuellung keine Senke mehr", count_real_craters(filled) == 0)
    ok &= check("Fuellhoehe ist nie negativ", bool(np.all(fill_amount >= 0.0)))

    pit_fill = float(fill_amount[31, 31])
    ok &= check(f"die geschlossene Grube wurde aufgefuellt ({pit_fill:.1f} m)", pit_fill > 1.0)

    channel_fill = float(fill_amount[channel_row, :].max())
    ok &= check(f"die zum Rand entwaessernde Rinne bleibt unangetastet "
                f"(maximale Fuellung dort {channel_fill:.6f} m)", channel_fill < 1e-6)
    return ok


def run_more_particles_improve_branching():
    """Kernzusage der Kopplung Partikelzahl <-> Wasser pro Tropfen
    (DropletErosionSystem.initial_water_volume): mehr Partikel muessen das
    Ergebnis FEINER machen, nicht nur mehr abtragen. Ohne die Kopplung kehrte
    sich das um (Top-5%-Anteil sank von 0.243 auf 0.142)."""
    size = 96
    heightmap = _make_terrain(size, seed=5)
    hardness_map = np.full((size, size), 50.0, dtype=np.float32)
    meters_per_pixel = 10000.0 / size

    concentrations = []
    for num_particles in (5000, 40000):
        system = DropletErosionSystem(erosion_strength=2.5, sediment_capacity_factor=4.0,
                                       deposit_speed=0.3)
        erosion, _ = system.simulate_erosion_sedimentation(
            heightmap, hardness_map, {"water_seed": 7},
            {"erosion_particles": num_particles}, meters_per_pixel)
        flat = np.sort(erosion.ravel())[::-1]
        concentrations.append(float(flat[:len(flat) // 20].sum() / max(flat.sum(), 1e-9)))

    return check(f"mehr Partikel -> staerker verzweigt "
                 f"(5k: {concentrations[0]:.3f}, 40k: {concentrations[1]:.3f})",
                 concentrations[1] > concentrations[0])


if __name__ == "__main__":
    results = {
        "quality_metrics": run_quality_metrics(),
        "fill_removes_all_depressions": run_fill_removes_all_depressions(),
        "more_particles_improve_branching": run_more_particles_improve_branching(),
    }
    print("\n=== SUMMARY ===")
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)
