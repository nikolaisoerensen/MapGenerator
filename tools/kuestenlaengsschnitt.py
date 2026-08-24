"""
Path: tools/kuestenlaengsschnitt.py

EINE GANZE KUESTE VERMESSEN - Querschnitte auf den KUESTENNORMALEN, nicht auf
festen Breitengraden.

Nutzerwunsch 2026-08-18: *"koennen wir die gesamte kueste von 'dough' nach
'doolin' nehmen?"*

WARUM NICHT EINFACH PARALLELE SCHNITTE

Der erste Anlauf (`kuestenprofil_holen.py`) legt die Schnitte auf feste
Breitengrade. Solange die Kueste gerade nach Norden laeuft, geht das gut -
sobald sie eine Bucht macht, trifft der Schnitt sie SCHRAEG, und die Wand
erscheint kuenstlich in die Breite gezogen. Gemessen an den 2 km Moher: die
"Strecke bis 90 % Hoehe" streute dadurch zwischen 37 und 279 m, obwohl die
Wand real ueberall aehnlich steil ist. Der Ausreisser sass genau in der
Biegung.

Hier wird deshalb zuerst die Kuestenlinie SELBST bestimmt (Marching Squares
auf der Nullhoehe, also zwischen den Pixeln interpoliert), dann in gleichen
Bogenabstaenden abgeschritten, und an jeder Station laeuft der Schnitt
entlang der oertlichen NORMALEN ins Land. Damit ist "Strecke bis 90 % Hoehe"
eine Eigenschaft der Kueste und nicht des Gitters, auf dem gemessen wurde.

WELCHE SEITE IST LAND

Nicht aus der Umlaufrichtung der Kontur geraten - `find_contours` garantiert
keine einheitliche Orientierung, und eine falsch geratene Seite legt alle
Schnitte ins Wasser. Stattdessen wird beidseitig abgetastet und die hoehere
Seite gewinnt; dieselbe Vorsichtsmassnahme wie in `kuesten_mesh.klippenband()`.

Aufruf:
    .venv/Scripts/python.exe tools/kuestenlaengsschnitt.py doolin
    .venv/Scripts/python.exe tools/kuestenlaengsschnitt.py doolin --abstand 100
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.kuestenprofil_holen import CACHE, dem_holen, _meter_je_grad

# Kuestenstuecke: Kartenausschnitt in Grad.
#
# Je Region des Spiels ein reales Vorbild, dazu Inseln getrennt (Nutzer-
# Vorgabe 2026-08-18: "verschiedene kuesten fuer verschiedene regionen, dann
# separat fuer inseln ein paar beispiele, italien und griechische inseln
# hauptsaechlich, dann nur 1 variante fuer die anderen regionen").
STRECKEN = {
    # ---------------- Festlandskuesten je Region ----------------
    "doolin": {
        "titel": "Dough (Lahinch) bis Doolin, County Clare - "
                 "Liscannor Bay, Hag's Head, Cliffs of Moher",
        "region": "Huegelland",
        "sued": 52.9250, "nord": 53.0250, "west": -9.4700, "ost": -9.3300,
    },
    "moher": {
        "titel": "Cliffs of Moher, Kernstueck",
        "region": "Huegelland",
        "sued": 52.9600, "nord": 52.9820, "west": -9.4450, "ost": -9.4160,
    },
    "lofoten": {
        "titel": "Lofoten bei Reine - Gipfel direkt aus dem Meer",
        "region": "Fjordland",
        "sued": 67.9200, "nord": 68.0400, "west": 13.0000, "ost": 13.2500,
    },
    # FLAECHENGLEICHER AUSSCHNITT zum Fjordland dieser Welt (rund 16.7 km2,
    # davon 11.4 km2 Land). Der grosse `geiranger`-Ausschnitt misst 55.7 km2
    # und enthaelt ein ganzes Fjordsystem samt Hinterland - ein Vergleich der
    # Kuestendichte gegen die Spielregion waere damit sinnlos, weil beide
    # Groessen an der Ausschnittsgroesse haengen (Nutzerauftrag 2026-08-24).
    "geiranger_klein": {
        "titel": "Geirangerfjord, Kernstueck (flaechengleich zum Fjordland)",
        "region": "Fjordland",
        "sued": 62.0850, "nord": 62.1217, "west": 7.0450, "ost": 7.1235,
    },
    "geiranger": {
        "titel": "Geirangerfjord - Fjordwand im Landesinneren",
        "region": "Fjordland",
        "sued": 62.0700, "nord": 62.1300, "west": 7.0000, "ost": 7.1600,
    },
    "stockholm": {
        "titel": "Stockholmer Schaeren - niedrige Felskueste, viele Inseln",
        "region": "Taiga",
        "sued": 59.3000, "nord": 59.4000, "west": 18.7000, "ost": 18.9500,
    },
    "roca": {
        "titel": "Cabo da Roca, Portugal - hohe Atlantikklippe",
        "region": "Atlantikkueste",
        "sued": 38.7500, "nord": 38.8300, "west": -9.5200, "ost": -9.4100,
    },
    "velebit": {
        "titel": "Velebit-Kueste, Kroatien - Gebirge direkt ins Meer",
        "region": "Alpenland",
        "sued": 44.2500, "nord": 44.3600, "west": 14.8800, "ost": 15.0600,
    },
    "ruegen": {
        "titel": "Ruegen, Koenigsstuhl - Kreidekueste",
        "region": "Mittelgebirge",
        "sued": 54.5300, "nord": 54.6000, "west": 13.6000, "ost": 13.7200,
    },
    "schwarzmeer": {
        "titel": "Kap Kaliakra, Bulgarien - Steppenkueste mit Steilkante",
        "region": "Steppe",
        "sued": 43.3400, "nord": 43.4200, "west": 28.4000, "ost": 28.5200,
    },
    "calanques": {
        "titel": "Calanques bei Marseille - Kalkfelskueste",
        "region": "Mittelmeer",
        "sued": 43.1800, "nord": 43.2400, "west": 5.4000, "ost": 5.5400,
    },
    "santorini": {
        "titel": "Santorini - Caldera-Steilwand gegen Aussenkueste",
        "region": "Griechische Inseln",
        "sued": 36.3400, "nord": 36.4800, "west": 25.3300, "ost": 25.4900,
    },

    # ---------------- Inseln ----------------
    "capri": {
        "titel": "Capri, Italien - kleine Felsinsel",
        "region": "Insel/Italien",
        "sued": 40.5300, "nord": 40.5700, "west": 14.1900, "ost": 14.2700,
    },
    "stromboli": {
        "titel": "Stromboli, Italien - Vulkankegel als Insel",
        "region": "Insel/Italien",
        "sued": 38.7600, "nord": 38.8200, "west": 15.1800, "ost": 15.2700,
    },
    "elba": {
        "titel": "Elba (Ostteil), Italien - groessere Insel",
        "region": "Insel/Italien",
        "sued": 42.7200, "nord": 42.8200, "west": 10.1500, "ost": 10.3200,
    },
    "milos": {
        "titel": "Milos, Griechenland - zerklueftete Vulkaninsel",
        "region": "Insel/Griechenland",
        "sued": 36.6600, "nord": 36.7600, "west": 24.3800, "ost": 24.5600,
    },
    "naxos": {
        "titel": "Naxos (Westteil), Griechenland",
        "region": "Insel/Griechenland",
        "sued": 37.0300, "nord": 37.1300, "west": 25.3400, "ost": 25.5000,
    },
    "amorgos": {
        "titel": "Amorgos, Griechenland - schmaler Grat mit Steilkueste",
        "region": "Insel/Griechenland",
        "sued": 36.8000, "nord": 36.8800, "west": 25.8300, "ost": 25.9800,
    },
    "inselchen_taiga": {
        "titel": "Schaereninsel Oestliche Ostsee - flache Felsinsel",
        "region": "Insel/Taiga",
        "sued": 59.4300, "nord": 59.4900, "west": 19.4000, "ost": 19.5200,
    },
    "aran": {
        "titel": "Inisheer (Aran), Irland - flache Kalkinsel",
        "region": "Insel/Huegelland",
        "sued": 53.0400, "nord": 53.0900, "west": -9.5600, "ost": -9.4800,
    },
}

# Wie weit der Schnitt ins Land und ins Meer laeuft, in Metern.
#
# 2026-08-24 VON 900 AUF 2500 m ERHOEHT (Nutzervorgabe: *"dann muessen
# neue profile her ... einfach tiefer ins land gehen"*).
#
# WARUM 900 m ZU WENIG WAR. Das Abbruchkriterium ist "der Anstieg ist auf
# ein Zehntel seines Hoechstwerts gefallen". Bei 900 m Messtiefe konnte
# eine Kueste, die weiter ausholt, dieses Kriterium gar nicht erreichen -
# der Schnitt endete vorher, und die gemessene Reichweite war in
# Wahrheit die Messgrenze. Die Fjordwand kam auf 529 m, also schon nahe
# an der Haelfte des Fensters.
#
# DIE KACHELN TRAGEN DAS: gemessen erlaubt jede Vorbildstrecke ausser
# `moher` mindestens 3300 m ins Land (doolin 4831 m, lofoten 6679 m,
# velebit 6123 m). `moher` ist mit 1001 m die kleine Detailkachel; ihre
# Region wird von `doolin` ohnehin grosszuegiger abgedeckt - genau der
# Ausschnitt "von Dough bis Doolin", den der Nutzer beschrieben hat.
INS_LAND_M = 2500.0
INS_MEER_M = 250.0

# Stuetzstellen je Schnitt. Mit dem groesseren Fenster muessen es mehr
# werden, sonst waere die Abtastung groeber als das DEM: 260 Stellen
# ueber 2750 m sind 10.6 m, COP30 loest 30 m auf - das waere noch
# vertretbar, aber die Reichweitenbestimmung braucht den Anstieg, und der
# wird ueber neun Stellen geglaettet.
STUETZSTELLEN = 600


def zellgroesse_m(kopf):
    """
    (Nord-Sued, Ost-West) Pixelgroesse in Metern.

    DIE PIXEL SIND NICHT QUADRATISCH, und das wurde bis 2026-08-24
    uebersehen. Ein DEM-Raster hat konstante Gradabstaende; ein Grad
    Laenge ist am Aequator 111.3 km, bei 62 Grad Nord aber nur noch
    52 km. Bei Geiranger misst ein Pixel deshalb 31.1 m in Nord-Sued-
    und nur 14.5 m in Ost-West-Richtung.

    GEMESSENE FOLGE des alten `zelle_m = cellsize * 111320` fuer BEIDE
    Richtungen: die Flaeche der Geiranger-Kachel kam auf 118.4 km2 statt
    55.7, jede Ost-West-Strecke war um Faktor 1/cos(Breite) zu gross.
    Alle 27 Vorbilder waren betroffen - von 1.22x (Kreta, 35 Grad) bis
    2.81x (Kola, 69 Grad). Damit waren auch die Kuestenprofile in
    MESS_PROFIL_M_JE_ARCHETYP verzerrt, und zwar je nach Breitengrad
    verschieden stark.
    """
    ns = kopf["cellsize"] * 111320.0
    mitte = kopf.get("yllcorner", 0.0) + 0.5 * kopf.get("nrows", 0) * kopf["cellsize"]
    ow = ns * float(np.cos(np.radians(mitte)))
    return ns, max(ow, 1e-6)


def kuestenlinie(werte, kopf, mindestlaenge_m=500.0, pegel=0.5):
    """
    Die Kuestenlinie(n) als Liste von (M,2)-Zuegen in (spalte, zeile).

    `pegel` bewusst leicht ueber Null: COP30 setzt offenes Meer exakt auf 0,
    eine Kontur genau auf 0 laeuft deshalb streckenweise durch flaches
    Rauschen statt an der Kante.
    """
    from skimage import measure

    # ANISOTROP: je Schritt getrennt nach Zeile und Spalte messen, siehe
    # zellgroesse_m(). Ein Konturschritt nach Osten ist bei 62 Grad Nord
    # nur halb so lang wie einer nach Norden.
    ns_m, ow_m = zellgroesse_m(kopf)
    linien = []
    for kontur in measure.find_contours(np.nan_to_num(werte, nan=-1.0), pegel):
        d = np.diff(kontur, axis=0)
        laenge = float(np.hypot(d[:, 0] * ns_m, d[:, 1] * ow_m).sum())
        if laenge < mindestlaenge_m:
            continue
        linien.append((laenge, np.stack([kontur[:, 1], kontur[:, 0]], axis=1)))
    linien.sort(key=lambda p: -p[0])
    return linien


def _abtasten(werte, spalte, zeile):
    """Bilinear, mit NaN ausserhalb."""
    nrows, ncols = werte.shape
    s = np.clip(spalte, 0, ncols - 1.001)
    z = np.clip(zeile, 0, nrows - 1.001)
    s0, z0 = np.floor(s).astype(int), np.floor(z).astype(int)
    fs, fz = s - s0, z - z0
    s1, z1 = np.minimum(s0 + 1, ncols - 1), np.minimum(z0 + 1, nrows - 1)
    return ((werte[z0, s0] * (1 - fs) + werte[z0, s1] * fs) * (1 - fz)
            + (werte[z1, s0] * (1 - fs) + werte[z1, s1] * fs) * fz)


def schnitte_auf_normalen(werte, kopf, linie, abstand_m=100.0,
                          ins_land_m=INS_LAND_M, ins_meer_m=INS_MEER_M,
                          stuetzstellen=None):
    """
    Je Station ein Schnitt entlang der Kuestennormalen, Meer -> Land.

    Rueckgabe (stationen_xy, bogen_m, strecke_m, profile):
      profile (N, stuetzstellen) - Hoehe, `strecke_m` ist gemeinsam und
      beginnt NEGATIV im Meer, 0 an der Kuestenlinie.
    """
    if stuetzstellen is None:
        stuetzstellen = STUETZSTELLEN
    ns_m, ow_m = zellgroesse_m(kopf)
    # Fuer die Stationsabstaende entlang der Kueste genuegt ein Mittel -
    # die Kueste laeuft in alle Richtungen. Fuer die SCHNITTE weiter unten
    # nicht: dort zaehlt die Richtung der Normalen, und genau dort sass
    # der Fehler.
    zelle_m = 0.5 * (ns_m + ow_m)
    abstand_px = max(1.0, abstand_m / zelle_m)

    schritte = np.hypot(*(np.diff(linie, axis=0).T))
    bogen = np.concatenate([[0.0], np.cumsum(schritte)])
    ziele = np.arange(0.0, bogen[-1], abstand_px)
    sx = np.interp(ziele, bogen, linie[:, 0])
    sy = np.interp(ziele, bogen, linie[:, 1])

    # Tangente aus der geglaetteten Linie - roh ist sie zwischen zwei
    # Konturpunkten fast achsenparallel und die Normale springt.
    fenster = max(3, int(round(abstand_px)))
    kern = np.ones(fenster) / fenster
    gx = np.convolve(np.interp(ziele, bogen, linie[:, 0]), kern, mode="same")
    gy = np.convolve(np.interp(ziele, bogen, linie[:, 1]), kern, mode="same")
    tx = np.gradient(gx)
    ty = np.gradient(gy)
    laenge = np.hypot(tx, ty)
    laenge = np.where(laenge > 1e-9, laenge, 1.0)
    nx, ny = -ty / laenge, tx / laenge          # Normale in der Kartenebene

    # WELCHE SEITE IST LAND - beidseitig abtasten, die hoehere gewinnt.
    probe_px = max(2.0, 150.0 / zelle_m)
    h_plus = _abtasten(werte, sx + nx * probe_px, sy + ny * probe_px)
    h_minus = _abtasten(werte, sx - nx * probe_px, sy - ny * probe_px)
    landseite = np.where(np.nan_to_num(h_plus, nan=-1e9)
                         >= np.nan_to_num(h_minus, nan=-1e9), 1.0, -1.0)
    nx, ny = nx * landseite, ny * landseite

    strecke = np.linspace(-ins_meer_m, ins_land_m, stuetzstellen)
    profile = np.empty((len(ziele), stuetzstellen), dtype=np.float64)

    # HIER SASS DER FEHLER. `schritt_px = strecke / zelle_m` nahm EINE
    # Pixelgroesse fuer alle Richtungen. Ein Schnitt nach Osten lief damit
    # bei 62 Grad Nord doppelt so weit, wie er sollte - das Profil war in
    # Ost-West-Richtung gestreckt.
    #
    # `nx`/`ny` sind Einheitsvektoren im PIXELraum. Wie viele Meter ein
    # Pixelschritt in dieser Richtung wirklich bedeutet, sagt der Satz des
    # Pythagoras auf den beiden Zellgroessen - je Station verschieden,
    # weil jede ihre eigene Normale hat.
    for k in range(len(ziele)):
        meter_je_px = np.hypot(nx[k] * ow_m, ny[k] * ns_m)
        schritt_px = strecke / max(meter_je_px, 1e-6)
        profile[k] = _abtasten(werte,
                               sx[k] + nx[k] * schritt_px,
                               sy[k] + ny[k] * schritt_px)

    return (np.stack([sx, sy], axis=1), ziele * zelle_m, strecke, profile)


def kennzahlen(strecke, profile, anteil=0.9):
    """Je Schnitt: Zielhoehe und Strecke bis `anteil` davon."""
    gipfel = np.nanmax(profile, axis=1)
    land = strecke >= 0
    breite = np.full(len(profile), np.nan)
    for k, p in enumerate(profile):
        ziel = anteil * gipfel[k]
        ueber = np.flatnonzero((p >= ziel) & land)
        if len(ueber):
            breite[k] = strecke[ueber[0]]
    return gipfel, breite


def main():
    p = argparse.ArgumentParser()
    p.add_argument("strecke", nargs="?", default="doolin")
    p.add_argument("--abstand", type=float, default=100.0,
                   help="Stationsabstand entlang der Kueste, Meter")
    p.add_argument("--demtype", default="COP30")
    p.add_argument("--bild", default=None)
    args = p.parse_args()

    if args.strecke not in STRECKEN:
        raise SystemExit(f"Unbekannt. Bekannt: {', '.join(STRECKEN)}")
    s = STRECKEN[args.strecke]

    werte, kopf = dem_holen(s["sued"], s["nord"], s["west"], s["ost"],
                            args.demtype)
    zelle_m = kopf["cellsize"] * 111320.0
    print(f"\n{s['titel']}")
    print(f"  Raster {werte.shape}, Zellweite rund {zelle_m:.0f} m")

    linien = kuestenlinie(werte, kopf)
    if not linien:
        raise SystemExit("Keine Kuestenlinie gefunden.")
    print(f"  {len(linien)} Kuestenzuege, laengster {linien[0][0]/1000:.1f} km")
    laenge_m, linie = linien[0]

    stationen, bogen, strecke, profile = schnitte_auf_normalen(
        werte, kopf, linie, args.abstand)
    gipfel, breite = kennzahlen(strecke, profile)
    print(f"  {len(stationen)} Stationen alle {args.abstand:.0f} m")
    print(f"  Zielhoehe {np.nanmin(gipfel):.0f} .. {np.nanmax(gipfel):.0f} m")
    print(f"  Strecke bis 90 %: Median {np.nanmedian(breite):.0f} m, "
          f"p10 {np.nanpercentile(breite,10):.0f} .. "
          f"p90 {np.nanpercentile(breite,90):.0f} m")
    aend = np.abs(np.diff(gipfel)) / (args.abstand / 1000.0)
    print(f"  Aenderung der Zielhoehe: Median {np.nanmedian(aend):.0f} m/km")

    ziel = args.bild or os.path.join(CACHE, f"laengs_{args.strecke}.png")
    _zeichnen(werte, kopf, linie, stationen, bogen, strecke, profile,
              gipfel, breite, s["titel"], args.demtype, zelle_m, ziel)
    print(f"\n  Bild: {ziel}")
    return 0


def _zeichnen(werte, kopf, linie, stationen, bogen, strecke, profile,
              gipfel, breite, titel, demtype, zelle_m, ziel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(stationen)
    farben = plt.cm.rainbow(np.linspace(0, 1, n))
    fig = plt.figure(figsize=(17.5, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.45, 1.45])

    a = fig.add_subplot(gs[:, 0])
    a.imshow(werte, cmap="Blues_r", vmin=-30, vmax=1)
    a.imshow(np.where(werte <= 0, np.nan, werte), cmap="terrain",
             vmin=0, vmax=220)
    a.plot(linie[:, 0], linie[:, 1], "k-", lw=.6, alpha=.5)
    a.scatter(stationen[:, 0], stationen[:, 1], c=farben, s=5, zorder=4)
    a.set_title(f"Kueste und {n} Stationen\n(Farbe = Position laengs)",
                fontsize=10)
    a.set_xticks([]); a.set_yticks([])

    a = fig.add_subplot(gs[0, 1])
    for p, f in zip(profile, farben):
        a.plot(strecke, p, color=f, lw=.7, alpha=.75)
    a.axvline(0, color="k", lw=.9, ls="--"); a.axhline(0, color="k", lw=.5)
    a.set_xlabel("Abstand von der Kuestenlinie (m)   <- Meer | Land ->")
    a.set_ylabel("Hoehe (m)")
    a.set_title("Auf den Kuestennormalen geschnitten"); a.grid(alpha=.3)

    a = fig.add_subplot(gs[1, 1])
    land = strecke >= 0
    for p, f in zip(profile, farben):
        g = np.nanmax(p)
        if not np.isfinite(g) or g <= 1:
            continue
        a.plot(strecke[land] / strecke[land].max(), np.clip(p[land], 0, None) / g,
               color=f, lw=.7, alpha=.75)
    a.set_xlabel("u  (0 = Wasserlinie)"); a.set_ylabel("h / Zielhoehe")
    a.set_title("Normiert - die Wellenformen"); a.grid(alpha=.3)

    km = bogen / 1000.0
    a = fig.add_subplot(gs[0, 2])
    a.plot(km, gipfel, lw=1.4, color="darkred")
    a.set_xlabel("km laengs der Kueste"); a.set_ylabel("Zielhoehe (m)")
    a.set_title("Zielhoehe laengs"); a.grid(alpha=.3)

    a = fig.add_subplot(gs[1, 2])
    a.plot(km, breite, lw=1.4, color="navy")
    a.set_xlabel("km laengs der Kueste")
    a.set_ylabel("Strecke bis 90 % Hoehe (m)")
    a.set_title("Wie schnell die Wand hochkommt"); a.grid(alpha=.3)

    fig.suptitle(f"{titel}  -  {demtype}, Zellweite rund {zelle_m:.0f} m",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(ziel, dpi=100)


if __name__ == "__main__":
    sys.exit(main())
