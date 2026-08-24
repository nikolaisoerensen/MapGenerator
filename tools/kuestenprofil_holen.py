"""
Path: tools/kuestenprofil_holen.py

ECHTE KUESTENPROFILE VON OPENTOPOGRAPHY HOLEN - als Referenz zum Zeichnen
der Wellenformen (siehe die Kuestentyp-Planung, Wellenform-Modell).

Nutzerfrage 2026-08-18: *"wie koennte man die kontur aus online daten
bekommen? also zB wenn ich einen abschnitt von 2 km von den cliffs of moher
gerne als kontur haette."*

WAS DIESES WERKZEUG TUT

Es laedt ein Hoehenmodell fuer einen Kartenausschnitt, legt eine Schnittlinie
quer zur Kueste hinein und gibt das Hoehenprofil entlang dieser Linie aus -
einmal in echten Metern und einmal auf (u, h) in [0,1] normiert, also genau
in der Form, in der eine Wellenform beschrieben wird.

KEINE NEUE ABHAENGIGKEIT. Die OpenTopography-API kann `AAIGrid` liefern, ein
ESRI-ASCII-Raster - reiner Text mit sechs Kopfzeilen. Das liest numpy direkt.
Der Umweg ueber GeoTIFF haette `rasterio` oder GDAL verlangt (beides nicht
installiert, beides schwergewichtig).

DER API-SCHLUESSEL STEHT NICHT IM QUELLTEXT. Er wird aus der Umgebungs-
variablen OPENTOPOGRAPHY_API_KEY gelesen. Setzen (PowerShell, dauerhaft):

    setx OPENTOPOGRAPHY_API_KEY "dein-schluessel"

Danach ein neues Terminal oeffnen. Fuer nur diese Sitzung:

    $env:OPENTOPOGRAPHY_API_KEY = "dein-schluessel"

EHRLICHE GRENZE, VOR DEM ERSTEN LAUF LESEN

Der beste frei verfuegbare Weltdatensatz (Copernicus, `COP30`) hat 30 m
Rasterweite. Die Cliffs of Moher sind gut 200 m hoch und nahezu senkrecht -
auf der Wand liegen damit ein bis drei Stuetzstellen. Das Ergebnis ist eine
RAMPE, keine Klippe. Und das ist keine Schwaeche der Quelle, sondern
dieselbe strukturelle Grenze wie bei uns: ein Hoehenraster hat einen z-Wert
je (x,y) und kann eine Senkrechte grundsaetzlich nicht abbilden.

Brauchbar ist das Profil also fuer die GROSSFORM - wie breit ist das
Vorland, wie liegt das Plateau dahinter, wie lang laeuft es aus. Die
Steilheit der Wand selbst musst du von Hand setzen. Fuer echte Wandprofile
braeuchte es LiDAR mit 1-2 m (fuer Irland ueber die nationalen Portale zu
suchen, nicht ueber diese API).

Aufruf:
    .venv/Scripts/python.exe tools/kuestenprofil_holen.py moher
    .venv/Scripts/python.exe tools/kuestenprofil_holen.py --liste
"""

import argparse
import math
import os
import sys
import urllib.parse
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_URL = "https://portal.opentopography.org/API/globaldem"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dem_cache")

# Voreingestellte Schnitte: (Name, von_lat, von_lon, nach_lat, nach_lon).
# Die Linie laeuft VOM MEER INS LAND - u=0 ist die Wasserlinie, damit das
# Profil ohne Umdrehen als Wellenform taugt.
SCHNITTE = {
    "moher": {
        "titel": "Cliffs of Moher, Irland (quer zur Kueste)",
        "von": (52.9715, -9.4400), "nach": (52.9715, -9.4180),
    },
    "etretat": {
        "titel": "Falaises d'Etretat, Normandie",
        "von": (49.7100, 0.1900), "nach": (49.7020, 0.2120),
    },
    "sylt": {
        "titel": "Sylt Westkueste (Flachkueste mit Duenen)",
        "von": (54.9050, 8.2900), "nach": (54.9050, 8.3200),
    },
    "geiranger": {
        "titel": "Geirangerfjord, Norwegen (Fjordwand)",
        "von": (62.1000, 7.0800), "nach": (62.1000, 7.1200),
    },
}


def _schluessel():
    key = os.environ.get("OPENTOPOGRAPHY_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "OPENTOPOGRAPHY_API_KEY ist nicht gesetzt.\n"
            "  PowerShell (dauerhaft):  setx OPENTOPOGRAPHY_API_KEY \"...\"\n"
            "  PowerShell (Sitzung):    $env:OPENTOPOGRAPHY_API_KEY = \"...\"\n"
            "Danach ein NEUES Terminal oeffnen, wenn setx benutzt wurde.")
    return key


def _meter_je_grad(lat):
    """Naeherung, genuegt fuer wenige Kilometer."""
    return 111320.0, 111320.0 * math.cos(math.radians(lat))


def dem_holen(sued, nord, west, ost, demtype="COP30"):
    """
    Laedt den Ausschnitt als ESRI-ASCII-Raster und gibt (werte, kopf) zurueck.

    Zwischengespeichert: derselbe Ausschnitt wird nicht zweimal geladen. Die
    API hat ein Kontingent, und beim Ausprobieren verschiedener Schnittlinien
    ist der Ausschnitt fast immer derselbe.
    """
    os.makedirs(CACHE, exist_ok=True)
    name = f"{demtype}_{sued:.4f}_{nord:.4f}_{west:.4f}_{ost:.4f}.asc"
    pfad = os.path.join(CACHE, name)

    if not os.path.exists(pfad):
        params = urllib.parse.urlencode({
            "demtype": demtype, "south": sued, "north": nord,
            "west": west, "east": ost,
            "outputFormat": "AAIGrid", "API_Key": _schluessel()})
        url = f"{API_URL}?{params}"
        print(f"lade {demtype} fuer {sued:.4f}..{nord:.4f} / {west:.4f}..{ost:.4f} ...")
        try:
            with urllib.request.urlopen(url, timeout=120) as antwort:
                inhalt = antwort.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as fehler:
            leib = fehler.read().decode("utf-8", errors="replace")[:400]
            raise SystemExit(f"OpenTopography antwortete {fehler.code}: {leib}")
        if not inhalt.lstrip().lower().startswith("ncols"):
            raise SystemExit(f"Unerwartete Antwort (kein AAIGrid):\n{inhalt[:400]}")
        with open(pfad, "w", encoding="utf-8") as f:
            f.write(inhalt)
        print(f"  gespeichert: {pfad}")
    else:
        print(f"aus dem Zwischenspeicher: {pfad}")

    return asc_lesen(pfad)


def asc_lesen(pfad):
    """ESRI-ASCII-Raster lesen - sechs Kopfzeilen, dann die Werte."""
    kopf = {}
    with open(pfad, "r", encoding="utf-8") as f:
        for _ in range(6):
            teile = f.readline().split()
            kopf[teile[0].lower()] = float(teile[1])
        werte = np.loadtxt(f)
    nodata = kopf.get("nodata_value", -9999.0)
    werte = np.where(werte == nodata, np.nan, werte)
    return werte, kopf


def profil(werte, kopf, von, nach, stuetzstellen=400):
    """
    Hoehenprofil entlang der Linie von->nach.

    Rueckgabe (strecke_m, hoehe_m) - `strecke_m` beginnt bei 0 am Punkt `von`.
    """
    ncols, nrows = int(kopf["ncols"]), int(kopf["nrows"])
    zelle = kopf["cellsize"]
    x0, y0 = kopf["xllcorner"], kopf["yllcorner"]

    t = np.linspace(0.0, 1.0, stuetzstellen)
    lat = von[0] + t * (nach[0] - von[0])
    lon = von[1] + t * (nach[1] - von[1])

    # Rasterindex. Zeile 0 ist die OBERSTE (noerdlichste) Zeile.
    spalte = (lon - x0) / zelle
    zeile = (y0 + nrows * zelle - lat) / zelle

    si = np.clip(np.round(spalte).astype(int), 0, ncols - 1)
    zi = np.clip(np.round(zeile).astype(int), 0, nrows - 1)
    hoehe = werte[zi, si]

    m_lat, m_lon = _meter_je_grad(0.5 * (von[0] + nach[0]))
    dx = (lon - von[1]) * m_lon
    dy = (lat - von[0]) * m_lat
    strecke = np.hypot(dx, dy)
    return strecke, hoehe


def als_wellenform(strecke, hoehe, pegel=0.0):
    """
    Das Profil auf (u, h) in [0,1] normieren - die Form, in der eine
    Wellenform gespeichert wird.

    u = 0 an der Wasserlinie (erster Punkt ueber `pegel`), u = 1 am Ende des
    Landabschnitts. h = 0 an der Wasserlinie, 1 an der Zielhoehe (hier: dem
    hoechsten Punkt des Abschnitts).

    Absichtlich NUR normiert und nicht geglaettet: was hier herauskommt, ist
    eine Messung mit ihrem Rauschen. Die gezeichnete Kurve entsteht daraus
    per Auge, nicht per Filter - siehe Modulkopf zur Aufloesungsgrenze.
    """
    gut = np.isfinite(hoehe)
    if not gut.any():
        return None, None
    land = gut & (hoehe > pegel)
    if not land.any():
        return None, None
    erster = int(np.flatnonzero(land)[0])
    letzter = int(np.flatnonzero(land)[-1])
    if letzter <= erster:
        return None, None

    s = strecke[erster:letzter + 1]
    h = hoehe[erster:letzter + 1]

    # DIE WASSERLINIE INTERPOLIEREN, nicht den ersten Landpunkt nehmen.
    #
    # Bei 31 m Rasterweite liegt der erste Punkt ueber Null an den Cliffs of
    # Moher schon 27 m hoch - die Kurve begann dadurch bei h=0.186 statt bei
    # 0, obwohl (0,0) per Definition die Wasserlinie IST. Der Nulldurchgang
    # zwischen dem letzten See- und dem ersten Landpunkt wird deshalb linear
    # bestimmt und als eigener Anfangspunkt vorangestellt.
    if erster > 0 and np.isfinite(hoehe[erster - 1]):
        h_see, h_land = hoehe[erster - 1], hoehe[erster]
        if h_land != h_see:
            t = (pegel - h_see) / (h_land - h_see)
            s_null = strecke[erster - 1] + t * (strecke[erster] - strecke[erster - 1])
            s = np.concatenate([[s_null], s])
            h = np.concatenate([[pegel], h])

    u = (s - s[0]) / max(s[-1] - s[0], 1e-9)
    spanne = np.nanmax(h) - pegel
    hn = (h - pegel) / max(spanne, 1e-9)
    return u, hn


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("schnitt", nargs="?", default="moher",
                   help="Name eines voreingestellten Schnitts")
    p.add_argument("--liste", action="store_true", help="Schnitte auflisten")
    p.add_argument("--demtype", default="COP30",
                   help="COP30 (Standard), COP90, SRTMGL1, AW3D30, NASADEM ...")
    p.add_argument("--rand", type=float, default=0.004,
                   help="Zusatzrand um die Schnittlinie in Grad")
    p.add_argument("--bild", default=None, help="Profilbild hierhin schreiben")
    args = p.parse_args()

    if args.liste:
        for name, s in SCHNITTE.items():
            print(f"  {name:<12} {s['titel']}")
        return 0

    if args.schnitt not in SCHNITTE:
        raise SystemExit(f"Unbekannt: {args.schnitt}. --liste zeigt alle.")
    s = SCHNITTE[args.schnitt]
    von, nach = s["von"], s["nach"]

    sued = min(von[0], nach[0]) - args.rand
    nord = max(von[0], nach[0]) + args.rand
    west = min(von[1], nach[1]) - args.rand
    ost = max(von[1], nach[1]) + args.rand

    werte, kopf = dem_holen(sued, nord, west, ost, args.demtype)
    strecke, hoehe = profil(werte, kopf, von, nach)

    m_lat, m_lon = _meter_je_grad(von[0])
    laenge = math.hypot((nach[1] - von[1]) * m_lon, (nach[0] - von[0]) * m_lat)
    zelle_m = kopf["cellsize"] * m_lat

    print(f"\n{s['titel']}")
    print(f"  Schnittlaenge {laenge:.0f} m, Rasterweite rund {zelle_m:.0f} m "
          f"-> {laenge/max(zelle_m,1e-9):.0f} echte Stuetzstellen")
    gueltig = np.isfinite(hoehe)
    print(f"  Hoehe {np.nanmin(hoehe):.0f} .. {np.nanmax(hoehe):.0f} m, "
          f"{(~gueltig).sum()} Luecken")

    u, hn = als_wellenform(strecke, hoehe)
    if u is None:
        print("  Kein Landabschnitt auf dieser Linie gefunden.")
        return 1

    print(f"\n  Als Wellenform normiert ({len(u)} Punkte):")
    for anteil in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        k = int(anteil * (len(u) - 1))
        print(f"    u={u[k]:.2f}  h={hn[k]:.3f}")

    steilste = int(np.nanargmax(np.abs(np.gradient(hoehe, strecke))))
    print(f"\n  Steilste Stelle bei {strecke[steilste]:.0f} m: "
          f"{np.degrees(np.arctan(abs(np.gradient(hoehe, strecke)[steilste]))):.0f} Grad "
          f"(bei {zelle_m:.0f} m Rasterweite nach oben begrenzt)")

    ziel = args.bild
    if ziel is None:
        ziel = os.path.join(CACHE, f"profil_{args.schnitt}.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 7))
        ax[0].plot(strecke, hoehe, lw=1.6)
        ax[0].axhline(0, color="k", lw=.6)
        ax[0].set_xlabel("Strecke (m)"); ax[0].set_ylabel("Hoehe (m)")
        ax[0].set_title(f"{s['titel']}  -  {args.demtype}, "
                        f"Rasterweite rund {zelle_m:.0f} m")
        ax[0].grid(alpha=.3)
        ax[1].plot(u, hn, lw=1.8)
        ax[1].set_xlabel("u  (0 = Wasserlinie, 1 = Ende des Abschnitts)")
        ax[1].set_ylabel("h / Zielhoehe")
        ax[1].set_title("normiert - so wuerde die Wellenform aussehen")
        ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(ziel, dpi=100)
        print(f"\n  Bild: {ziel}")
    except Exception as fehler:                              # noqa: BLE001
        print(f"\n  (kein Bild: {fehler})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
