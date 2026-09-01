"""
Path: tools/regionen_lab.py

WERKZEUG, um die Reglergroessen an echten Landschaften zu eichen.

Fuenf Referenzen des Nutzers (2026-07-30), jeweils als Hoehenkarte aus grosser
Hoehe, rund 200 km Bildbreite. Gesucht ist nicht die Nachbildung dieser Bilder,
sondern ein AUSSCHNITT von 25 x 25 km im jeweiligen Stil - und damit ein
Gefuehl dafuer, welche Meterwerte welche Landschaft ergeben.

Erst seit der Skalenverknuepfung (SPEZIFIKATION §10) ist das ueberhaupt
sinnvoll: vorher haetten dieselben Regler bei 25 km eine andere Landschaft
ergeben als bei 15 km, weil Grundformen und Rinnen am Bildausschnitt hingen
statt an der Wirklichkeit.

ALLE ZAHLEN SIND VORSCHLAEGE UND ZU BESTAETIGEN. Sie stammen aus dem
Augenschein der fuenf Bilder plus dem, was ueber die Gegenden allgemein bekannt
ist - nicht aus Hoehenmodellen. Die Bilder selbst kann ich nicht vermessen.

BEKANNTE EINSCHRAENKUNG: TERRAIN.BASE_ELEVATION_M liegt fest bei 100 m, und
`amplitude` ist die GIPFELHOEHE, nicht die Hoehendifferenz. Jede Region hier
beginnt deshalb bei 100 m, auch das Flachland, das real bei 0-20 m liegt. Die
RELIEFS stimmen, die absoluten Basishoehen nicht. Solange amplitude < 100 m
sogar das Gelaende umdreht (offener Fehler, siehe Bericht), ist das nicht
umgehbar.

Aufruf:
    .venv\\Scripts\\python.exe tools/regionen_lab.py
    .venv\\Scripts\\python.exe tools/regionen_lab.py 512
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "lab_output", "regionen_lab")

AUSSCHNITT_KM = 25.0
SEED = 20260730


# =============================================================================
# DIE FUENF REGIONEN
# =============================================================================
# basis_m/gipfel_m -> amplitude; das Relief ist die Differenz.
# feature_size_m   -> Groesse der Grundformen (Massive, Ruecken, Becken)
# gully_size_m     -> Groesse der Erosionsrinnen
#
# Die Faustregel, die sich beim Ansehen der Bilder ergibt: feature_size ist der
# Abstand der grossen Taeler, gully_size der der Seitenrinnen. Ihr VERHAELTNIS
# macht den Charakter - Hochgebirge ~4:1, dicht zergliedertes Nebelrode
# ~8:1, Flachland spielt keine Rolle, weil die Staerke gegen null geht.

REGIONEN = {
    "1_alpen": {
        "bemerkung": "Oetztal/Inntal: Trogtaeler, scharfe Grate, grosse Massive",
        "basis_m": 1000.0, "gipfel_m": 3500.0,
        "feature_size_m": 9000.0,     # Abstand der grossen Taeler
        "gully_size_m": 2200.0,
        "octaves": 2, "redistribute_power": 1.6,
        "strength": 0.30, "detail": 1.8, "gully_weight": 0.45,
        "ridge_rounding": 0.0,        # scharfe Grate
        "crease_rounding": 0.35,      # Trogtal, keine Kerbe
        "gully_octaves": 5,
    },
    "2_mittelgebirge": {
        "bemerkung": "Plateau mit dichter dendritischer Zertalung, geringes Relief",
        "basis_m": 200.0, "gipfel_m": 550.0,
        "feature_size_m": 7000.0,
        "gully_size_m": 900.0,        # dicht - Verhaeltnis ~8:1
        "octaves": 1, "redistribute_power": 1.0,
        "strength": 0.26, "detail": 2.4,   # Rinnen laufen weit auf flaches Land
        "gully_weight": 0.7,
        "ridge_rounding": 0.5,        # weiche Ruecken
        "crease_rounding": 0.1,
        "gully_octaves": 6,
    },
    "3_plattland": {
        "bemerkung": "Norddeutsche Tiefebene: fast kein Relief, traege Maeander",
        "basis_m": 100.0, "gipfel_m": 165.0,
        "feature_size_m": 12000.0,
        "gully_size_m": 2500.0,
        "octaves": 1, "redistribute_power": 0.8,
        "strength": 0.10,             # kaum Erosionssignatur
        "detail": 1.0, "gully_weight": 0.8,
        "ridge_rounding": 0.9, "crease_rounding": 0.7,
        "gully_octaves": 3,
    },
    "4_fjordland": {
        "bemerkung": "Westnorwegen: Hochflaeche, tief eingeschnittene Troege",
        "basis_m": 150.0, "gipfel_m": 1500.0,
        "feature_size_m": 8000.0,
        "gully_size_m": 2600.0,
        "octaves": 1, "redistribute_power": 0.55,   # Hochflaeche statt Gipfel
        "strength": 0.34,             # tiefe Einschnitte
        "detail": 1.1,                # nur die Haupttroege, nicht die Flaeche
        "gully_weight": 0.35,
        "ridge_rounding": 0.7,        # abgeschliffene Hochflaeche
        "crease_rounding": 0.3,
        "gully_octaves": 4,
    },
    "5_vietnam": {
        "bemerkung": "Kalk-/Bergland: enge dichte Kegel und Ruecken",
        "basis_m": 120.0, "gipfel_m": 1100.0,
        "feature_size_m": 5000.0,
        "gully_size_m": 700.0,        # sehr dicht
        "octaves": 2, "redistribute_power": 1.9,   # viel Flaeche tief, Kegel hoch
        "strength": 0.30, "detail": 2.6, "gully_weight": 0.85,
        "ridge_rounding": 0.05,
        "crease_rounding": 0.0,
        "gully_octaves": 6,
    },
}


def baue(name, size):
    from managers.data_lod_manager import DataLODManager
    from gui.config.value_default import TERRAIN
    from core.terrain_generator import BaseTerrainGenerator

    r = REGIONEN[name]
    manager = DataLODManager()
    manager.set_map_distance_km(AUSSCHNITT_KM)

    parameters = {
        "map_size": size,
        "map_distance_km": AUSSCHNITT_KM,
        "map_seed": SEED,
        "amplitude": r["gipfel_m"],
        "feature_size_m": r["feature_size_m"],
        "octaves": r["octaves"],
        "persistence": TERRAIN.PERSISTENCE["default"],
        "lacunarity": TERRAIN.LACUNARITY["default"],
        "redistribute_power": r["redistribute_power"],
        "erosion_filter_strength": r["strength"],
        "erosion_filter_gully_size_m": r["gully_size_m"],
        "erosion_filter_detail": r["detail"],
        "erosion_filter_gully_weight": r["gully_weight"],
        "erosion_filter_ridge_rounding": r["ridge_rounding"],
        "erosion_filter_crease_rounding": r["crease_rounding"],
        "erosion_filter_octaves": r["gully_octaves"],
    }

    lod = int(round(np.log2(max(size, 32) / 32.0))) + 1
    generator = BaseTerrainGenerator(data_lod_manager=manager)
    generator.set_active_parameters(parameters)
    for node in ("terrain.noise", "terrain.redistribution"):
        manager.set_calculator_target_lod(node, lod)
    generator._calc_noise("terrain.noise", lod)
    generator._calc_redistribution("terrain.redistribution", lod)

    z = manager.get_calculator_output("terrain.redistribution", "heightmap", lod)
    assert z.shape == (size, size), "angefragt %d px, bekommen %s" % (size, z.shape)
    return z.astype(np.float64), AUSSCHNITT_KM * 1000.0 / size


def _hillshade(z, mpp, azimut=315.0, hoehe=42.0):
    dy, dx = np.gradient(z, mpp)
    neigung = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspekt = np.arctan2(-dx, dy)
    az, hh = np.deg2rad(360.0 - azimut + 90.0), np.deg2rad(hoehe)
    return np.clip(np.sin(hh) * np.sin(neigung)
                   + np.cos(hh) * np.cos(neigung) * np.cos(az - aspekt), 0, 1)


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 384
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Fuenf Regionen als %.0f x %.0f km, %d px (%.0f m/px)\n"
          % (AUSSCHNITT_KM, AUSSCHNITT_KM, size, AUSSCHNITT_KM * 1000.0 / size))
    print("%-18s %8s %9s %12s %11s %7s" % (
        "Region", "Basis", "Gipfel", "Grundform", "Rinne", "Zeit"))
    print("-" * 72)

    fig, ax = plt.subplots(len(REGIONEN), 2, figsize=(11, 5.2 * len(REGIONEN)),
                           squeeze=False)
    for i, name in enumerate(REGIONEN):
        r = REGIONEN[name]
        start = time.time()
        z, mpp = baue(name, size)
        dauer = time.time() - start
        print("%-18s %7.0fm %8.0fm %11.0fm %10.0fm %6.1fs" % (
            name, z.min(), z.max(), r["feature_size_m"], r["gully_size_m"], dauer))

        # Hoehenkarte in derselben Art wie die Referenzbilder: Farbe = Hoehe,
        # daruebergelegte Schummerung. Ohne die Schummerung sieht man die Form
        # nicht, ohne die Farbe die Hoehenlage.
        schummer = _hillshade(z, mpp)
        b = ax[i, 0].imshow(z, cmap="terrain")
        ax[i, 0].imshow(schummer, cmap="gray", alpha=0.35)
        ax[i, 0].set_title("%s   %.0f - %.0f m  (Relief %.0f m)"
                           % (name, z.min(), z.max(), z.max() - z.min()),
                           fontsize=10)
        fig.colorbar(b, ax=ax[i, 0], shrink=0.8)

        ax[i, 1].imshow(schummer, cmap="gray")
        ax[i, 1].set_title("%s\nGrundform %.0f m, Rinne %.0f m"
                           % (r["bemerkung"], r["feature_size_m"],
                              r["gully_size_m"]), fontsize=8)
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])

    fig.suptitle("Fuenf Referenzlandschaften als %.0f x %.0f km Ausschnitt"
                 % (AUSSCHNITT_KM, AUSSCHNITT_KM), fontsize=12)
    fig.tight_layout()
    ziel = os.path.join(OUTPUT_DIR, "regionen_%dkm_%dpx.png"
                        % (int(AUSSCHNITT_KM), size))
    fig.savefig(ziel, dpi=95)
    plt.close(fig)
    print("\nBild -> %s" % ziel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
