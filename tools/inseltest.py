"""
Path: tools/inseltest.py

INSELTEST - das Kuestenmodell an Formen, die es aushalten muss.

Nutzer-Vorgabe 2026-08-18: Inseln verschiedener Groesse (10 km, 3 km, 500 m,
150 m), dann Inseln mit ausgepraegten Halbinseln ("Krakenarme"), erst mit
EINEM Kuestentyp, dann mit allen Typen einer Region, zuletzt mit allen
Regionen. Ergebnis je Fall als PNG, bei jedem Lauf neu geschrieben.

WAS DIESER TEST PRUEFT

Nicht die Rechenzeit und nicht die Dichtheit - dafuer gibt es Smoke-Tests.
Hier geht es um die FORM: sieht eine kleine Insel wie eine kleine Insel aus
oder wie ein abgeschnittener Kegel? Was passiert in der Mitte, wo sich die
Kuesten aller Seiten treffen? Was passiert im Hals einer Halbinsel?

Das sind genau die Stellen, an denen docs/KUESTENMODELL.md Abschnitt 4 und 5
Luecken benennt. Der Test ist also bewusst so gebaut, dass er sie ZEIGT statt
sie zu umgehen.

MASSSTAB, EHRLICH BENANNT

Getestet wird bei guter Aufloesung (der Ausschnitt ist immer 2.5x der
Inseldurchmesser auf 420 px), damit das MODELL beurteilt werden kann und
nicht das Gitter. Was eine Insel bei der echten Kartenaufloesung waere, steht
je Fall in der Bildunterschrift - eine 150-m-Insel ist bei 41.6 m/px rund 3.6
Pixel breit und dort schlicht nicht darstellbar.

Aufruf:
    .venv/Scripts/python.exe tools/inseltest.py
    .venv/Scripts/python.exe tools/inseltest.py --nur rund_3km
"""

import argparse
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.terrain_weltkarte import KUESTEN_ARCHETYPEN, alle_regionen
from core.vektor_kueste import VektorKueste, als_raster

AUSGABE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "inseltest")
GITTER = 420               # Pixel je Kante des Testausschnitts
ECHTE_MPP = 41.6           # Meter je Pixel der echten Karte (512 px)


def _rauschfeld(n, seed, mpp):
    """
    Gelaenderauschen aus DEM OKTAVENSTAPEL DES PROGRAMMS, nicht aus einem
    gefilterten Zufallsfeld.

    Nutzerfrage 2026-08-19: *"warum ist die insel so wenig noisy? kann man
    diese nicht mit perlin noise erzeugen, so wie in der main und die form
    erzwingen?"* - berechtigt. Vorher stand hier ein gaussgefiltertes
    Normalrauschen; das hat EINE Groessenskala und sieht dadurch glatt und
    kuenstlich aus. Das Programm benutzt neun Oktaven von 12 km bis 47 m
    Wellenlaenge.

    Der Ausschnitt der Testinsel ist ein anderer als der der Weltkarte,
    deshalb wird `mpp` durchgereicht (siehe oktavenstapel()). So sieht das
    Testgelaende aus wie das echte, nur auf einem anderen Ausschnitt.
    """
    from core.terrain_weltkarte import GRUNDFORM_M, OKTAVEN, oktavenstapel

    stapel = oktavenstapel(n, int(seed), None, mpp=mpp)
    # Gewichtung wie in weltfeld(): jede Oktave mit rauheit^k, und nur
    # Oktaven, deren Wellenlaenge ueberhaupt in den Ausschnitt passt.
    ausschnitt_m = n * mpp
    feld = np.zeros((n, n), dtype=np.float64)
    summe = 0.0
    rauheit = 0.55
    for k in range(OKTAVEN):
        wellenlaenge = GRUNDFORM_M / (2.0 ** k)
        if wellenlaenge > ausschnitt_m * 1.5:
            continue                      # groesser als die Karte - kein Muster
        if wellenlaenge < 2.0 * mpp:
            break                         # unter Nyquist - nur noch Aliasing
        gewicht = rauheit ** k
        feld += gewicht * stapel[k]
        summe += gewicht
    if summe <= 0:
        return np.zeros((n, n), dtype=np.float64)
    return feld / summe


def insel_bauen(durchmesser_m, ausschnitt_m, seed, arme=0, hoehe_m=180.0,
                schlaengel=0):
    """
    Eine synthetische Insel als Heightmap (Basisgelaende A, ohne Kuestenform).

    `arme` > 0 erzeugt Halbinseln ("Krakenarme") ueber eine winkelabhaengige
    Radiusmodulation. `arme` = 0 gibt eine unfoermige, aber lappenfreie Insel.
    """
    n = GITTER
    mpp = ausschnitt_m / n
    gy, gx = np.mgrid[0:n, 0:n]
    cx = cy = (n - 1) / 2.0
    dx, dy = (gx - cx) * mpp, (gy - cy) * mpp
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)

    rng = np.random.default_rng(seed)
    radius = durchmesser_m / 2.0

    # Unregelmaessiger Rand: wenige niedrige Winkelharmonische.
    stoerung = np.zeros_like(theta)
    for k in range(2, 7):
        stoerung += (rng.normal(0, 1) / k) * np.cos(k * theta + rng.uniform(0, 6.28))
    rand = radius * (1.0 + 0.16 * stoerung)

    if arme:
        # Krakenarme: kraeftige Modulation auf einer festen Armzahl, plus
        # eine Schwelle, damit die Arme schmal werden statt nur wellig.
        arm = np.cos(arme * theta + rng.uniform(0, 6.28))
        rand = rand * (0.58 + 0.62 * np.clip(arm, -1, 1) ** 3 * 0.5 + 0.31 * arm)

    if schlaengel < 0:
        # UNREGELMAESSIG SCHLAENGELND (Nutzer-Vorgabe 2026-08-22): nicht eine
        # Frequenz wie bei `schlaengel > 0`, sondern ein ganzes Band von
        # Winkelharmonischen mit zufaelligen Phasen und abfallender
        # Amplitude. Ergibt eine Kueste, die auf mehreren Groessenskalen
        # gleichzeitig zackt - und damit KEIN Wiederholungsmuster, an dem
        # sich der Test von selbst gruen faerben koennte.
        #
        # Die einzelne Frequenz war genau diese Falle: sie erzeugte 30
        # gleichmaessige Streifen im abgerollten Band, und die stammten aus
        # der Testinsel, nicht aus dem Kuestenmodell.
        wellig = np.zeros_like(theta)
        for k in range(6, 30):
            wellig += (rng.normal(0, 1) / (k ** 0.85)) * np.cos(
                k * theta + rng.uniform(0, 6.28))
        wellig /= max(np.abs(wellig).max(), 1e-9)
        rand = rand * (1.0 + 0.22 * wellig)
    elif schlaengel:
        # ENGE SCHLAENGELLINIE (Nutzerfrage 2026-08-19): der Rand zackt mit
        # hoher Winkelfrequenz, die Insel bleibt aber im Ganzen rund. Die
        # Kuestennormale zeigt dadurch staendig hin und her statt
        # durchgehend zur Mitte - genau der Fall, der gepruft werden soll.
        # Amplitude klein gegen den Radius, damit es Zacken bleiben und
        # keine Arme werden.
        rand = rand * (1.0 + 0.11 * np.cos(schlaengel * theta
                                           + rng.uniform(0, 6.28)))

    # Hoehe: glatte Kuppe plus Rauschen, damit es ein echtes Basisgelaende ist.
    #
    # SMOOTHSTEP STATT POTENZ (2026-08-19): vorher `innen**0.85`, also ein
    # fast linearer Kegel - der lief in der Inselmitte in eine SPITZE, und
    # die sah im Querschnitt aus, als kaeme sie vom Kuestenmodell. Gemessen
    # war sie zu 100 % Basisgelaende (179.14 m in A wie in B, waehrend der
    # Kuestenbeitrag nur 549 m weit reicht und die Mitte 4423 m entfernt
    # liegt). Ein Testgelaende, das selbst einen Fehler zeigt, den es nicht
    # gibt, ist schlechter als keins.
    #
    # Smoothstep hat am Gipfel Steigung null - eine gerundete Kuppe, wie eine
    # echte Insel sie hat.
    innen = np.clip(1.0 - r / np.maximum(rand, 1e-6), 0.0, 1.0)
    kuppe = hoehe_m * (innen * innen * (3.0 - 2.0 * innen))
    # Rauschen mit VOLLER Amplitude, nicht als kleiner Zuschlag - so wie im
    # Programm, wo das Relief aus den Oktaven kommt und die Grossform es nur
    # moduliert.
    rauschen = _rauschfeld(n, seed + 1, mpp) * hoehe_m * 0.55
    H = kuppe + rauschen * innen

    # Ausserhalb des Randes ist See.
    H = np.where(r <= rand, np.maximum(H, 1.0), -20.0 - 30.0 * (r - rand) / max(radius, 1))
    return H.astype(np.float64), mpp


def _region_karte(H, modus, seed):
    """
    Regionszuordnung je Pixel.

    `modus`: "ein" = eine Region ueberall (also alle Archetypen DIESER Region
    stehen zur Wahl), "alle" = die neun Regionen als Winkelsektoren, damit an
    einer einzigen Insel alle Kuestentypen aufeinandertreffen.
    """
    n = H.shape[0]
    if modus == "ein":
        return np.zeros((n, n), dtype=np.int16)
    gy, gx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    winkel = (np.arctan2(gy - c, gx - c) + np.pi) / (2 * np.pi)
    return np.clip((winkel * 9).astype(np.int16), 0, 8)


def _ein_typ_erzwingen(vk, typ_index=0):
    """
    Alle Stationen auf DENSELBEN Archetyp setzen.

    Fuer die ersten Testfaelle: nur ein Kuestentyp, damit die Form der Insel
    beurteilt werden kann und nicht die Typenmischung.
    """
    if vk.saat_baum is None or not len(vk.saat_xy):
        return
    archetypen = KUESTEN_ARCHETYPEN[[r["name"] for _z, _s, r in alle_regionen()][0]]
    typ = archetypen[min(typ_index, len(archetypen) - 1)]
    from core.terrain_weltkarte import KUESTENHOEHE_M, MAX_KLIPPENWINKEL_GRAD
    from core.vektor_kueste import (KUESTEN_SOCKEL_M,
                                    UEBERHOEHUNG_JE_HOEHENFAKTOR)

    # DEN TYP AN DER QUELLE SETZEN UND DIE SEGMENTE NEU BAUEN.
    #
    # ZWEIMAL WAR DAS HIER FALSCH, beide Male derselbe Fehler: der Helfer
    # beschrieb Arrays (`saat_zielhoehe`, `saat_tanwinkel`, ...), die
    # `hoehe()` inzwischen gar nicht mehr liest - erst wanderte die
    # Zielhoehenrechnung in die Hinterlandkopplung, dann die Typwerte in die
    # Segmente. Der erzwungene Typ wirkte dadurch nicht, und die Testbilder
    # zeigten etwas anderes als ihre Ueberschrift behauptete.
    #
    # Jetzt wird der Archetyp dort gesetzt, wo er herkommt, und danach
    # dieselbe Aufbaukette gefahren wie im Programm.
    vk.saat_archetyp = [typ] * len(vk.saat_xy)
    vk._segmente_bauen()
    return typ["name"]


FAELLE = [
    ("rund_10km",  dict(durchmesser_m=10000, ausschnitt_m=25000, arme=0), "ein", True),
    ("rund_3km",   dict(durchmesser_m=3000,  ausschnitt_m=7500,  arme=0), "ein", True),
    ("rund_500m",  dict(durchmesser_m=500,   ausschnitt_m=1250,  arme=0), "ein", True),
    ("rund_150m",  dict(durchmesser_m=150,   ausschnitt_m=375,   arme=0), "ein", True),
    ("arme_10km",  dict(durchmesser_m=10000, ausschnitt_m=25000, arme=5), "ein", True),
    ("arme_3km",   dict(durchmesser_m=3000,  ausschnitt_m=7500,  arme=5), "ein", True),
    ("schlaengel_10km", dict(durchmesser_m=10000, ausschnitt_m=25000,
                             arme=0, schlaengel=26), "ein", True),
    ("region_alle_typen", dict(durchmesser_m=6000, ausschnitt_m=15000, arme=4), "ein", False),
    ("alle_regionen",     dict(durchmesser_m=8000, ausschnitt_m=20000, arme=5), "alle", False),
]


def einen_fall(name, bauargs, region_modus, ein_typ, seed=4711):
    H, mpp = insel_bauen(seed=seed, **bauargs)
    region_map = _region_karte(H, region_modus, seed)

    vk = VektorKueste(H, region_map, seed, welt_km=bauargs["ausschnitt_m"] / 1000.0)
    typname = None
    if ein_typ:
        typname = _ein_typ_erzwingen(vk)
    B = als_raster(vk)

    # Kennzahlen
    land = H > 0
    d = ndimage.distance_transform_edt(land) * mpp
    komp, nk = ndimage.label(land)
    d_max = float(d[land].max()) if land.any() else 0.0

    # Querschnitt durch die Mitte
    mitte = H.shape[0] // 2
    x_m = (np.arange(H.shape[1]) - H.shape[1] / 2) * mpp

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)

    a = fig.add_subplot(gs[0, 0])
    b = a.imshow(H, cmap="terrain", origin="lower")
    a.contour(H, [0], colors="k", linewidths=1)
    fig.colorbar(b, ax=a, shrink=.8, label="m")
    a.set_title("A - Basisgelaende (ohne Kueste)", fontsize=10)
    a.set_xticks([]); a.set_yticks([])

    a = fig.add_subplot(gs[0, 1])
    b = a.imshow(B, cmap="terrain", origin="lower")
    a.contour(B, [0], colors="k", linewidths=1)
    if len(vk.saat_xy):
        a.scatter(vk.saat_xy[:, 0], vk.saat_xy[:, 1], s=3, c="red", zorder=4)
    fig.colorbar(b, ax=a, shrink=.8, label="m")
    a.set_title(f"B - mit Kueste ({len(vk.saat_xy)} Stationen, rot)", fontsize=10)
    a.set_xticks([]); a.set_yticks([])

    a = fig.add_subplot(gs[0, 2])
    diff = B - H
    g = max(np.abs(diff).max(), 1e-9)
    b = a.imshow(diff, cmap="RdBu_r", origin="lower", vmin=-g, vmax=g)
    fig.colorbar(b, ax=a, shrink=.8, label="m")
    a.set_title(f"B - A   (max {g:.0f} m)", fontsize=10)
    a.set_xticks([]); a.set_yticks([])

    a = fig.add_subplot(gs[1, 0])
    hs = ndimage.gaussian_filter(B, 1.0)
    gyy, gxx = np.gradient(hs, mpp)
    schatten = np.clip((gxx * .6 + gyy * .6 + 1) / 2, 0, 1)
    a.imshow(schatten, cmap="gray", origin="lower")
    a.contour(B, [0], colors="cyan", linewidths=1)
    a.set_title("Schattierung von B (Form beurteilen)", fontsize=10)
    a.set_xticks([]); a.set_yticks([])

    a = fig.add_subplot(gs[1, 1])
    a.plot(x_m, H[mitte], lw=1.3, label="A Basis", color="gray", ls="--")
    a.plot(x_m, B[mitte], lw=1.8, label="B mit Kueste")
    a.axhline(0, color="k", lw=.6)
    a.set_xlabel("m von der Inselmitte"); a.set_ylabel("Hoehe (m)")
    a.set_title("Querschnitt durch die Mitte", fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=.3)

    a = fig.add_subplot(gs[1, 2])
    if land.any():
        a.scatter(d[land][::7], B[land][::7], s=1, alpha=.25)
        a.set_xlabel("Abstand zur Kueste (m)"); a.set_ylabel("Hoehe in B (m)")
        a.set_title("Hoehe ueber Kuestenabstand\n(Streuung = Variation laengs)",
                    fontsize=10)
        a.grid(alpha=.3)

    echte_px = bauargs["durchmesser_m"] / ECHTE_MPP
    kopf = (f"{name}   -   Insel {bauargs['durchmesser_m']/1000:.2f} km, "
            f"Ausschnitt {bauargs['ausschnitt_m']/1000:.1f} km, "
            f"{mpp:.1f} m/px   |   auf der echten Karte waeren das "
            f"{echte_px:.0f} px")
    if typname:
        kopf += f"   |   ein Typ: {typname}"
    fig.suptitle(kopf, fontsize=12)
    fig.tight_layout()

    os.makedirs(AUSGABE, exist_ok=True)
    ziel = os.path.join(AUSGABE, f"{name}.png")
    fig.savefig(ziel, dpi=95)
    plt.close(fig)

    return {
        "name": name, "mpp": mpp, "d_max": d_max, "komponenten": int(nk),
        "stationen": len(vk.saat_xy), "echte_px": echte_px,
        "hoehe_A": (float(H[land].min()), float(H[land].max())) if land.any() else (0, 0),
        "hoehe_B": (float(B[land].min()), float(B[land].max())) if land.any() else (0, 0),
        "aenderung_max": float(np.abs(B - H).max()),
        "landanteil": float(land.mean()),
        "ziel": ziel,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nur", default=None)
    args = p.parse_args()

    faelle = [f for f in FAELLE if args.nur is None or f[0] == args.nur]
    if not faelle:
        raise SystemExit(f"Unbekannt. Bekannt: {', '.join(f[0] for f in FAELLE)}")

    print(f"{'Fall':<20}{'m/px':>7}{'echt px':>9}{'Stat.':>7}{'Komp':>6}"
          f"{'d_max':>8}{'Hoehe A':>14}{'Hoehe B':>14}{'max dH':>9}")
    print("-" * 94)
    ergebnisse = []
    for name, bauargs, modus, ein_typ in faelle:
        e = einen_fall(name, bauargs, modus, ein_typ)
        ergebnisse.append(e)
        print(f"{e['name']:<20}{e['mpp']:>7.1f}{e['echte_px']:>9.0f}"
              f"{e['stationen']:>7}{e['komponenten']:>6}{e['d_max']:>8.0f}"
              f"{e['hoehe_A'][0]:>6.0f}-{e['hoehe_A'][1]:<7.0f}"
              f"{e['hoehe_B'][0]:>6.0f}-{e['hoehe_B'][1]:<7.0f}"
              f"{e['aenderung_max']:>9.0f}")
    print(f"\n{len(ergebnisse)} Bilder in {AUSGABE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
