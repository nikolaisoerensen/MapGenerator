"""
Path: tools/erosion_filter_lab.py

WERKZEUG fuer die CPU-Referenz des ATEF-Filters (core/terrain_erosion_filter.py).

Zwei getrennte Fragen, die nicht vermischt werden duerfen:

  "demo"    Ist die PORTIERUNG richtig? Gebaut wird genau das Gelaende des
            Shader-Demonstrationsteils (eigenes fBm mit analytischen
            Ableitungen). Das Bild ist gegen das bekannte Original zu halten.

  "karte"   Wirkt der Filter auf UNSERER Heightmap? Erst wenn die Portierung
            steht, ist ein schlechtes Ergebnis hier eine Aussage ueber die
            Anwendung und nicht ueber den Port.

Diese Trennung ist der Grund, warum es zwei Laeufe gibt: SPEZIFIKATION §5.1.2
verlangt eine Sache auf einmal, und Portierung UND Anwendung gleichzeitig zu
beurteilen war in diesem Projekt schon einmal eine halbe Stunde Suche.

  "kennzahlen"  Was der Filter an den Kennzahlen aus §3.1/§3.2 aendert, mit
                Gegenprobe (Filter aus). Er kann die Entwaesserungs-Kennzahlen
                NICHT verbessern - er bewegt keine Masse. Der Lauf ist da, um
                das zu belegen statt zu behaupten.

Aufruf:
    .venv\\Scripts\\python.exe tools/erosion_filter_lab.py demo
    .venv\\Scripts\\python.exe tools/erosion_filter_lab.py karte
    .venv\\Scripts\\python.exe tools/erosion_filter_lab.py kennzahlen
"""

import os
import sys
import time

import numpy as np

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "lab_output", "erosion_filter_lab")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _hillshade(z, meter_pro_pixel=1.0, azimut=315.0, hoehe=45.0):
    dy, dx = np.gradient(z, meter_pro_pixel)
    neigung = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspekt = np.arctan2(-dx, dy)
    az, hh = np.deg2rad(360.0 - azimut + 90.0), np.deg2rad(hoehe)
    return np.clip(np.sin(hh) * np.sin(neigung)
                   + np.cos(hh) * np.cos(neigung) * np.cos(az - aspekt), 0, 1)


# =============================================================================
# "demo" - ist die Portierung richtig?
# =============================================================================

def lauf_demo(size=384):
    from core.terrain_erosion_filter import demo_heightmap
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt = _plt()

    start = time.time()
    r = demo_heightmap(size=size)
    dauer = time.time() - start

    roh, erodiert = r["height_raw"], r["height_eroded"]
    print("Demonstrationsgelaende %dpx in %.2f s (%.1f us/px)"
          % (size, dauer, 1e6 * dauer / size / size))
    print("  roh       %.4f .. %.4f  (Spanne %.4f)"
          % (roh.min(), roh.max(), roh.max() - roh.min()))
    print("  erodiert  %.4f .. %.4f  (Spanne %.4f)"
          % (erodiert.min(), erodiert.max(), erodiert.max() - erodiert.min()))
    print("  Delta     %.4f .. %.4f, magnitude %.4f"
          % (r["height_delta"].min(), r["height_delta"].max(), r["magnitude"]))
    print("  ridge_map %.3f .. %.3f  (soll etwa -1 .. +1)"
          % (r["ridge_map"].min(), r["ridge_map"].max()))

    # Plausibilitaet der Portierung: harte Grenzen, die ein Tippfehler reisst.
    fehler = []
    if not np.all(np.isfinite(erodiert)):
        fehler.append("nicht-endliche Werte im Ergebnis")
    if r["ridge_map"].min() < -1.6 or r["ridge_map"].max() > 1.6:
        fehler.append("ridge_map weit ausserhalb -1..1 (%.2f..%.2f)"
                      % (r["ridge_map"].min(), r["ridge_map"].max()))
    if abs(r["height_delta"]).max() > 1.0:
        fehler.append("Delta groesser als die gesamte Hoehenspanne")
    for f in fehler:
        print("  FEHLER: %s" % f)

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 10.5))
    ax[0, 0].imshow(_hillshade(roh), cmap="gray")
    ax[0, 0].set_title("Eingang (fBm), Schummerung")
    ax[0, 1].imshow(_hillshade(erodiert), cmap="gray")
    ax[0, 1].set_title("nach dem Filter, Schummerung")
    b = ax[0, 2].imshow(r["height_delta"], cmap="RdBu_r",
                        vmin=-abs(r["height_delta"]).max(),
                        vmax=abs(r["height_delta"]).max())
    ax[0, 2].set_title("Hoehen-Delta")
    fig.colorbar(b, ax=ax[0, 2], shrink=0.8)

    ax[1, 0].imshow(erodiert, cmap="terrain")
    ax[1, 0].set_title("Hoehe nach dem Filter")
    b = ax[1, 1].imshow(r["ridge_map"], cmap="coolwarm", vmin=-1, vmax=1)
    ax[1, 1].set_title("ridge_map  (-1 Kerbe .. +1 Kamm)")
    fig.colorbar(b, ax=ax[1, 1], shrink=0.8)

    mitte = size // 2
    ax[1, 2].plot(roh[mitte], lw=1.0, label="Eingang")
    ax[1, 2].plot(erodiert[mitte], lw=1.4, color="black", label="gefiltert")
    ax[1, 2].set_title("Querschnitt Bildmitte")
    ax[1, 2].legend(fontsize=8)
    ax[1, 2].grid(alpha=0.3)

    for a in ax.ravel()[:5]:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("ATEF CPU-Referenz, Demonstrationsgelaende, %d px" % size)
    fig.tight_layout()
    ziel = os.path.join(OUTPUT_DIR, "demo.png")
    fig.savefig(ziel, dpi=105)
    plt.close(fig)
    print("\nBild -> %s" % ziel)
    return 1 if fehler else 0


# =============================================================================
# "karte" - wirkt der Filter auf unserer Heightmap?
# =============================================================================

def _unser_gelaende(size, amplitude, potenz, octaves=None):
    import tools.erosion_lab as lab
    over = {"amplitude": amplitude, "redistribute_power": potenz}
    if octaves is not None:
        over["octaves"] = octaves
    roh, mpp = lab.build_terrain(size, terrain_overrides=over)
    return roh.astype(np.float64), mpp


def lauf_karte(size=256):
    from core.terrain_erosion_filter import filter_heightmap
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt = _plt()

    faelle = (("04_alpen_wallis", 700.0 + 3800.0, 2.0),
              ("21_bamberg_franken", 230.0 + 300.0, 2.5))

    fig, ax = plt.subplots(len(faelle), 4, figsize=(20, 5.2 * len(faelle)),
                           squeeze=False)
    for r_i, (name, amplitude, potenz) in enumerate(faelle):
        z, mpp = _unser_gelaende(size, amplitude, potenz)
        start = time.time()
        e = filter_heightmap(z, mpp)
        dauer = time.time() - start
        gefiltert = z + e["height_delta"]

        print("%s  %d px, %.0f m/px, %d Oktaven, %.2f s"
              % (name, size, mpp, e["effective_octaves"], dauer))
        print("   Relief  %.0f m -> %.0f m   (Delta %.0f .. %.0f m)"
              % (z.max() - z.min(), gefiltert.max() - gefiltert.min(),
                 e["height_delta"].min(), e["height_delta"].max()))

        ax[r_i, 0].imshow(_hillshade(z, mpp), cmap="gray")
        ax[r_i, 0].set_title("%s: Eingang" % name, fontsize=9)
        ax[r_i, 1].imshow(_hillshade(gefiltert, mpp), cmap="gray")
        ax[r_i, 1].set_title("gefiltert", fontsize=9)
        b = ax[r_i, 2].imshow(e["height_delta"], cmap="RdBu_r",
                              vmin=-abs(e["height_delta"]).max(),
                              vmax=abs(e["height_delta"]).max())
        ax[r_i, 2].set_title("Delta in m", fontsize=9)
        fig.colorbar(b, ax=ax[r_i, 2], shrink=0.75)
        b = ax[r_i, 3].imshow(e["ridge_map"], cmap="coolwarm", vmin=-1, vmax=1)
        ax[r_i, 3].set_title("ridge_map", fontsize=9)
        fig.colorbar(b, ax=ax[r_i, 3], shrink=0.75)
        for a in ax[r_i]:
            a.set_xticks([]); a.set_yticks([])

    fig.suptitle("ATEF-Filter auf den Heightmaps dieses Projekts, %d px" % size)
    fig.tight_layout()
    ziel = os.path.join(OUTPUT_DIR, "karte.png")
    fig.savefig(ziel, dpi=100)
    plt.close(fig)
    print("\nBild -> %s" % ziel)
    return 0


# =============================================================================
# "kennzahlen" - was aendert sich, mit Gegenprobe
# =============================================================================

def lauf_kennzahlen(size=192):
    from core.terrain_erosion_filter import filter_heightmap
    import tools.drainage_lab as dl

    print("Kennzahlen mit und ohne Filter, %d px\n" % size)
    print("%-30s %8s %7s %7s %7s %7s %8s"
          % ("Variante", "Abfluss", "Senken", "Netz", "Nadeln", "beta", "Relief"))
    print("-" * 80)

    # BEIDE Untergrundvarianten: 5 Oktaven ist die heutige Vorgabe, 1 Oktave
    # die, die nach lauf_basis() tatsaechlich benutzt werden soll. Nur auf der
    # verworfenen Konfiguration zu messen waere §4.2 - der teuerste Fehlertyp
    # dieses Projekts.
    for name, amplitude, potenz in (("04 Alpen", 700.0 + 3800.0, 2.0),
                                    ("21 Bamberg", 230.0 + 300.0, 2.5)):
        for okt_name, okt in (("5 Okt", 5), ("1 Okt", 1)):
            z, mpp = _unser_gelaende(size, amplitude, potenz, octaves=okt)
            w = dl.bewerte(z, mpp)
            print("%-30s %7.1f%% %7d %7d %7d %7.3f %8.0f"
                  % ("%s %s ohne Filter" % (name, okt_name),
                     100 * w["abfluss_anteil"], w["senken"],
                     w["netz"], w["nadeln"], w["beta"], w["relief"]))

            e = filter_heightmap(z, mpp)
            gefiltert = z + e["height_delta"]
            w2 = dl.bewerte(gefiltert, mpp)
            print("%-30s %7.1f%% %7d %7d %7d %7.3f %8.0f"
                  % ("%s %s mit Filter" % (name, okt_name),
                     100 * w2["abfluss_anteil"], w2["senken"],
                     w2["netz"], w2["nadeln"], w2["beta"], w2["relief"]))

            # GEGENPROBE: erosion_strength = 0 muss das Gelaende unangetastet
            # lassen. Haelt das nicht, ist das Delta nicht das, was es zu sein
            # behauptet.
            e0 = filter_heightmap(z, mpp, {"erosion_strength": 0.0})
            rest = float(np.abs(e0["height_delta"]).max())
            print("%-30s Delta bei Staerke 0: %.6g m  %s"
                  % ("   Gegenprobe", rest, "ok" if rest < 1e-3 else "FEHLER"))
        print()

    print("Erwartung, vorab notiert: Abfluss und Senken aendern sich kaum bis")
    print("gar nicht zum Besseren. Der Filter bewegt keine Masse und kennt kein")
    print("Routing - er macht das Aussehen, nicht die Entwaesserung (§8).")
    return 0


# =============================================================================
# "basis" - wie glatt muss der Untergrund sein?
# =============================================================================

def lauf_basis(size=256):
    """
    Der Filter erwartet einen GLATTEN Untergrund - sein Demo-Eingang war ein
    einziger weicher Huegel. Unser Terrain-Noise liefert Detail auf allen
    Skalen, und dann sind die Rinnen des Filters kleiner als die vorhandenen
    Formen.

    Geprueft wird deshalb die Oktavenzahl des UNTERGRUNDS, bei sonst gleichen
    Werten, plus der Relieferhalt-Schalter terrain_height_offset[1]. Eine Sache
    auf einmal waere hier zu wenig - die beiden haengen zusammen, weil ein
    konstantes Absenken das Delta dominiert und den Blick auf die Form
    verstellt. Deshalb steht der Relieferhalt in allen Varianten auf 1.0 und
    variiert wird nur der Untergrund; die letzte Zeile ist die Gegenprobe mit
    dem Demo-Wert.
    """
    from core.terrain_erosion_filter import filter_heightmap
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt = _plt()

    amplitude, potenz = 700.0 + 3800.0, 2.0
    # Der Offset steht ueberall auf (0,0) - eine Sache auf einmal. Der erste
    # Durchgang variierte Oktaven UND Offset und war dadurch nicht auswertbar:
    # der Offset-Modus (·, 1.0) setzt den fadeTarget der letzten Oktave ein,
    # also eine hochfrequente Groesse, und ueberdeckte den Oktaveneffekt
    # vollstaendig. Bei THO.y = 0 ist der Offset eine Konstante (magnitude ist
    # ein Skalar) und aendert die Form nicht - fuer uns die richtige Wahl, weil
    # unsere Pipeline die Hoehenspanne ohnehin neu festlegt.
    kein_offset = {"terrain_height_offset": (0.0, 0.0)}
    varianten = (
        ("Untergrund 5 Okt (heute)", {"octaves": 5}, kein_offset),
        ("Untergrund 3 Okt", {"octaves": 3}, kein_offset),
        ("Untergrund 2 Okt", {"octaves": 2}, kein_offset),
        ("Untergrund 1 Okt", {"octaves": 1}, kein_offset),
        # Gegenprobe: derselbe Untergrund, aber die groebere Rinnenskala, die
        # zu einem detailreichen Untergrund passen wuerde.
        ("5 Okt, Rinnen 3x groeber", {"octaves": 5},
         dict(kein_offset, erosion_scale=0.45)),
    )

    print("Einfluss der Untergrund-Glattheit, %d px, Alpental\n" % size)
    print("%-26s %9s %9s %10s %12s" % ("Variante", "Relief in", "Relief out",
                                       "Faktor", "Delta min/max"))
    print("-" * 74)

    fig, ax = plt.subplots(len(varianten), 3, figsize=(15, 4.6 * len(varianten)),
                           squeeze=False)
    for i, (name, terrain_over, filter_over) in enumerate(varianten):
        import tools.erosion_lab as lab
        over = {"amplitude": amplitude, "redistribute_power": potenz}
        over.update(terrain_over)
        z, mpp = lab.build_terrain(size, terrain_overrides=over)
        z = z.astype(np.float64)

        e = filter_heightmap(z, mpp, filter_over)
        gefiltert = z + e["height_delta"]
        relief_in = z.max() - z.min()
        relief_out = gefiltert.max() - gefiltert.min()
        print("%-26s %9.0f %9.0f %9.2fx  %5.0f / %5.0f"
              % (name, relief_in, relief_out, relief_out / relief_in,
                 e["height_delta"].min(), e["height_delta"].max()))

        ax[i, 0].imshow(_hillshade(z, mpp), cmap="gray")
        ax[i, 0].set_title("%s: Eingang" % name, fontsize=9)
        ax[i, 1].imshow(_hillshade(gefiltert, mpp), cmap="gray")
        ax[i, 1].set_title("gefiltert", fontsize=9)
        grenze = max(abs(e["height_delta"]).max(), 1e-6)
        b = ax[i, 2].imshow(e["height_delta"], cmap="RdBu_r",
                            vmin=-grenze, vmax=grenze)
        ax[i, 2].set_title("Delta in m", fontsize=9)
        fig.colorbar(b, ax=ax[i, 2], shrink=0.75)
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])

    fig.suptitle("Wie glatt muss der Untergrund fuer den ATEF-Filter sein?")
    fig.tight_layout()
    ziel = os.path.join(OUTPUT_DIR, "basis.png")
    fig.savefig(ziel, dpi=100)
    plt.close(fig)
    print("\nBild -> %s" % ziel)
    return 0


def main():
    laeufe = {"demo": lauf_demo, "karte": lauf_karte,
              "kennzahlen": lauf_kennzahlen, "basis": lauf_basis}
    if len(sys.argv) < 2 or sys.argv[1] not in laeufe:
        print(__doc__)
        return 1
    return laeufe[sys.argv[1]]()


if __name__ == "__main__":
    raise SystemExit(main())
