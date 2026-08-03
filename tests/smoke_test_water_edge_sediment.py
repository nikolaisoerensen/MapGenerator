"""
Throwaway headless smoke test for droplet-based hydraulic erosion
(core/water_generator.py DropletErosionSystem). Not part of the test suite -
run manually via the shared venv, see CLAUDE.md.

Komplett neu geschrieben 2026-07-25 fuer den Eulerian -> Droplet-Umbau (siehe
Plan "Water: Droplet-basierte Erosion") - die vorherige Datei testete
ausschliesslich _transport_sediment_maccormack(), das ersatzlos gestrichen
wurde. Deckt jetzt ab:
- (a) Massenerhaltung wird ABSICHTLICH NICHT erzwungen (Nutzer-Vorgabe
  2026-07-25: "ich brauche keine strikte Massenerhaltung, ich will das wie
  Sebastian Lague") - sedimentation_map.sum() darf/soll unter
  erosion_map.sum() liegen. Eine frühere Fassung dieser Datei testete das
  Gegenteil (erzwungene exakte Bilanz durch Zwangs-Absetzung der Restfracht)
  - das war der eigentliche Grund fuer sichtbare kuenstliche Huegel und
  wurde wieder entfernt, siehe DropletErosionSystem-Klassen-Docstring.
- (b) Partikel, die ueber den Kartenrand laufen oder ihr Lebenszeit-Budget
  aufbrauchen, VERWERFEN ihre Restfracht (identisch zur Referenz Erosion.cs),
  statt sie zwangsweise irgendwo abzusetzen.
- (c) Haerte-Differenzierung: identisches Terrain/Spawn-Seed, weiches vs.
  hartes Gestein - weiches Gestein muss messbar mehr Bruttoerosion zeigen.
- (d) GLEICHVERTEILTE Spawn-Verteilung (seit 2026-07-27; vorher
  hoehengewichtet, siehe run_uniform_spawn_distribution()).
- (e) Erosions-Schwerpunkt liegt unter der mittleren Gelaendehoehe - die
  eigentliche Zusage hinter (d): Gipfel bleiben unberuehrt, bergab nimmt die
  Erosion zu.
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

from core.water_generator import DropletErosionSystem


def check(label, condition):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def _make_slope_terrain(size=32, seed=23):
    """Gleichmaessig geneigtes Gelaende (hoch rechts, tief links) mit etwas
    Rauschen - treibt Partikel systematisch Richtung linkem Kartenrand,
    deckt sowohl Rand-Austritt als auch normale Ablagerung im Inneren ab."""
    rng = np.random.RandomState(seed)
    x = np.arange(size, dtype=np.float64)[None, :]
    heightmap = (100.0 + 4.0 * x + 3.0 * rng.randn(size, size)).astype(np.float32)
    return heightmap


def run_mass_conservation_not_enforced():
    """Nutzer-Vorgabe 2026-07-25: 'ich brauche keine strikte Massenerhaltung,
    ich will das wie Sebastian Lague' - sedimentation_map.sum() darf NIE
    groesser sein als erosion_map.sum() (es kann nichts aus dem Nichts
    abgelagert werden), muss aber auch NICHT gleich sein. Auf normalem
    Terrain mit genug Lebenszeit-Budget deponieren die meisten Partikel den
    Grossteil ihrer Fracht normal (Referenz-Formel, kein Unterschied zu
    vorher) - das ist hier kein Fehlschlag, nur die scharfe Gleichheits-
    Erwartung einer frueheren Fassung dieses Tests war falsch."""
    size = 40
    rng = np.random.RandomState(31)
    heightmap = (200.0 + 50.0 * rng.randn(size, size)).astype(np.float32)
    hardness_map = (1.0 + 99.0 * rng.rand(size, size)).astype(np.float32)
    meters_per_pixel = 15.0

    system = DropletErosionSystem(erosion_strength=2.5, sediment_capacity_factor=4.0, deposit_speed=0.3)

    ok = True
    for num_droplets in (50, 300, 1000):
        spawn_positions = system._sample_spawn_positions(heightmap, num_droplets, {"water_seed": 99})
        erosion_map, sedimentation_map = system.simulate_droplets_only(
            heightmap, hardness_map, spawn_positions, meters_per_pixel)
        total_erosion = float(erosion_map.sum())
        total_sedimentation = float(sedimentation_map.sum())

        ok &= check(f"num_droplets={num_droplets}: abgelagert <= abgetragen "
                    f"(abgetragen={total_erosion:.4f}, abgelagert={total_sedimentation:.4f})",
                    total_sedimentation <= total_erosion + 1e-6)
        ok &= check(f"num_droplets={num_droplets}: kein NaN/Inf",
                    bool(np.all(np.isfinite(erosion_map))) and bool(np.all(np.isfinite(sedimentation_map))))
    return ok


def run_edge_exit_discards_remaining_sediment():
    """Kernnachweis fuer den 2026-07-25-Fix (Nutzer-Report 'hier bilden sich
    Huegel', Ursache war eine faelschlich erzwungene exakte Massenerhaltung):
    Partikel, die durch das systematische Gefaelle ueber den linken
    Kartenrand laufen, muessen ihre Restfracht VERWERFEN statt sie
    zwangsweise irgendwo abzusetzen - identisch zur Referenz Erosion.cs.
    Auf diesem stark zum Rand hin geneigten Terrain erreichen viele Partikel
    den Rand, bevor sie ihre volle Fracht normal deponieren konnten - die
    Gesamt-Sedimentation muss deshalb MESSBAR unter der Gesamt-Erosion
    liegen (Beweis, dass tatsaechlich verworfen statt zwangsabgesetzt wird)."""
    size = 32
    heightmap = _make_slope_terrain(size=size, seed=23)
    hardness_map = np.full((size, size), 20.0, dtype=np.float32)
    meters_per_pixel = 10.0

    system = DropletErosionSystem(erosion_strength=3.0, sediment_capacity_factor=4.0, deposit_speed=0.3)
    spawn_positions = system._sample_spawn_positions(heightmap, 500, {"water_seed": 5})
    erosion_map, sedimentation_map = system.simulate_droplets_only(
        heightmap, hardness_map, spawn_positions, meters_per_pixel)

    total_erosion = float(erosion_map.sum())
    total_sedimentation = float(sedimentation_map.sum())

    ok = check("Erosion tatsaechlich stattgefunden (Terrain hat Gefaelle)", bool(total_erosion > 0.01))
    # Schwelle 2026-07-27 von 0.9 auf 0.99 angehoben: der Partikel-Lebensweg
    # ist seit der Aufloesungs-Entkopplung in METERN definiert
    # (DropletErosionSystem.DROPLET_MAX_LIFETIME_M) und ergibt bei den
    # 10 m/px dieses Tests statt der frueheren 30 Schritte deren 234. Mit dem
    # laengeren Weg deponieren die Partikel unterwegs deutlich mehr ihrer
    # Fracht regulaer, bevor sie den Rand erreichen - der verworfene Rest
    # faellt entsprechend kleiner aus (gemessen 5.7% statt vorher >10%).
    # Geprueft wird weiterhin die EIGENSCHAFT (am Rand wird verworfen, nicht
    # zwangsabgesetzt), nicht die alte, an eine feste Schrittzahl gebundene
    # Groesse des Verlusts. Exakte Gleichheit waere der Fehlerfall.
    ok &= check(f"Sedimentation liegt messbar unter Erosion (Restfracht am Rand verworfen, nicht "
                f"zwangsabgesetzt): abgetragen={total_erosion:.4f}, abgelagert={total_sedimentation:.4f} "
                f"({100 * total_sedimentation / total_erosion:.1f}% erhalten)",
                total_sedimentation < 0.99 * total_erosion)
    ok &= check("Ergebnis endlich",
                bool(np.all(np.isfinite(erosion_map))) and bool(np.all(np.isfinite(sedimentation_map))))
    return ok


def run_hardness_differentiation():
    """Kernnachweis 'haertere Materialien bleiben besser zurueck': identisches
    Terrain, identischer Spawn-Seed (also identische Partikel-Startpunkte),
    nur die Haerte unterscheidet sich - weiches Gestein muss messbar mehr
    Bruttoerosion zeigen als hartes Gestein."""
    size = 40
    rng = np.random.RandomState(17)
    heightmap = (300.0 + 400.0 * np.exp(-((np.arange(size)[:, None] - size / 2) ** 2
                                           + (np.arange(size)[None, :] - size / 2) ** 2) / (2 * (size * 0.2) ** 2))
                 + 10.0 * rng.randn(size, size)).astype(np.float32)
    hardness_soft = np.full((size, size), 2.0, dtype=np.float32)
    hardness_hard = np.full((size, size), 95.0, dtype=np.float32)
    meters_per_pixel = 10.0

    system = DropletErosionSystem(erosion_strength=2.5, sediment_capacity_factor=4.0, deposit_speed=0.3)
    spawn_positions = system._sample_spawn_positions(heightmap, 600, {"water_seed": 42})

    erosion_soft, _ = system.simulate_droplets_only(heightmap, hardness_soft, spawn_positions, meters_per_pixel)
    erosion_hard, _ = system.simulate_droplets_only(heightmap, hardness_hard, spawn_positions, meters_per_pixel)

    total_soft = float(erosion_soft.sum())
    total_hard = float(erosion_hard.sum())

    ok = check(f"weiches Gestein zeigt messbar mehr Bruttoerosion als hartes Gestein "
               f"(weich={total_soft:.4f}, hart={total_hard:.4f})", total_soft > 1.3 * total_hard)
    ok &= check("beide Ergebnisse endlich",
                bool(np.all(np.isfinite(erosion_soft))) and bool(np.all(np.isfinite(erosion_hard))))
    return ok


def run_uniform_spawn_distribution():
    """
    Partikel spawnen GLEICHVERTEILT (DROPLET_SPAWN_HEIGHT_POWER = 0, seit
    2026-07-27).

    Vorher pruefte dieser Test das Gegenteil ('Partikel fallen auf die Berge',
    >90% in der Hochzone). Diese Vorgabe wurde zurueckgenommen: gemessen
    starteten damit 69% aller Partikel in der oberen Gelaendehaelfte und
    zerkraterten genau die Gipfel (Nutzer-Report 'die Berge sind zerfressen an
    den Spitzen'). Regen faellt real ueberall gleich; dass die Erosion trotzdem
    bergab zunimmt, ergibt sich von selbst aus Geschwindigkeit, Kapazitaet und
    der Buendelung der Fliesswege - siehe DropletErosionSystem-Docstring und
    run_erosion_focuses_downhill() unten, das genau diese Wirkung nachweist.

    Auf einem Terrain mit klarer Hoch-/Tiefzone (rechte Haelfte hoch, linke
    tief, beide flaechengleich) muessen die Spawns entsprechend etwa 50/50
    liegen.
    """
    size = 50
    heightmap = np.full((size, size), 100.0, dtype=np.float32)
    heightmap[:, size // 2:] = 900.0

    system = DropletErosionSystem()
    spawn_positions = system._sample_spawn_positions(heightmap, 20000, {"water_seed": 3})
    fraction_in_high_zone = float(np.mean(spawn_positions[:, 0] >= size / 2))

    return check(f"Spawns gleichverteilt ueber beide Zonen "
                 f"({100 * fraction_in_high_zone:.1f}% in der Hochzone, erwartet 48-52%)",
                 0.48 < fraction_in_high_zone < 0.52)


def run_erosion_focuses_downhill():
    """
    Die eigentliche Zusage hinter dem gleichverteilten Spawn (Nutzer-Punkt (e):
    'Die Bergspitzen und Kaemme sind groesstenteils unberuehrt ... und alle
    Orte die von dort bergab sind haben mehr und mehr Erosion').

    Geprueft auf einem einzelnen Kegelberg: der hoehengewichtete Schwerpunkt
    der Erosion muss deutlich UNTER der Kegelspitze liegen. Das ist die
    Eigenschaft, die der frueher hoehengewichtete Spawn zerstoert hat.
    """
    size = 64
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    r = np.hypot(xx - size / 2, yy - size / 2)
    heightmap = (1000.0 * np.clip(1.0 - r / (size * 0.45), 0.0, 1.0)).astype(np.float32)
    hardness_map = np.full((size, size), 50.0, dtype=np.float32)
    meters_per_pixel = 10000.0 / size

    system = DropletErosionSystem(erosion_strength=2.5, sediment_capacity_factor=4.0, deposit_speed=0.3)
    spawn_positions = system._sample_spawn_positions(heightmap, 4000, {"water_seed": 11})
    erosion_map, _ = system.simulate_droplets_only(
        heightmap, hardness_map, spawn_positions, meters_per_pixel)

    norm_height = (heightmap - heightmap.min()) / max(float(heightmap.max() - heightmap.min()), 1e-9)
    weights = erosion_map / max(float(erosion_map.sum()), 1e-9)
    erosion_centre = float((norm_height * weights).sum())
    terrain_centre = float(norm_height.mean())

    return check(f"Erosions-Schwerpunkt liegt unter der mittleren Gelaendehoehe "
                 f"(Erosion bei {erosion_centre:.3f}, Gelaende bei {terrain_centre:.3f})",
                 erosion_centre < terrain_centre)


if __name__ == "__main__":
    results = {
        "mass_conservation_not_enforced": run_mass_conservation_not_enforced(),
        "edge_exit_discards_remaining_sediment": run_edge_exit_discards_remaining_sediment(),
        "hardness_differentiation": run_hardness_differentiation(),
        "uniform_spawn_distribution": run_uniform_spawn_distribution(),
        "erosion_focuses_downhill": run_erosion_focuses_downhill(),
    }
    print("\n=== SUMMARY ===")
    overall = True
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        overall &= ok
    sys.exit(0 if overall else 1)
