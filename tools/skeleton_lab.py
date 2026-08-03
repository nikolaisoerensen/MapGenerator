"""
Path: tools/skeleton_lab.py

WERKZEUG fuer den Umbau "STRUKTUR VOR NOISE" (2026-07-30).

Ausgangspunkt ist der Stand aus SPEZIFIKATION §7: die Erosion soll die
Entwaesserung aus isotropem fBm HERAUSHOLEN und schafft es nicht - 10-20%
Entwaesserungsanteil, 824-1559 geschlossene Senken, und laenger rechnen macht
es SCHLECHTER (Befund 2). Der Kapazitaetsfaktor ist bei 1.0 geklemmt, eine
Gerinne-Hierarchie ist im Feldmodell gar nicht ausdrueckbar (Befund 4).

Dieses Labor dreht die Richtung um. Das Flussnetz wird ZUERST als Graph
gebaut - vom Auslass am Kartenrand aufwaerts, mit Strahler-Ordnung und
MONOTON steigender Hoehe nach oben - und das Gelaende danach aus dem Abstand
zum Fluss geformt. Noise ist dann Modulation, nicht Basis.

Die falsifizierbare Vorhersage, um die es hier geht (SPEZIFIKATION §4.2/§5.1.4
verlangen genau das VOR der Auswertung):

    Wenn die Hoehe entlang des Netzes per Konstruktion monoton ist, muss der
    ENTWAESSERUNGSANTEIL deutlich ueber den 10-20% des Noise-Pfads liegen, und
    zwar OHNE jede Erosionsiteration.

Traegt die Vorhersage nicht, ist der ganze Umbau falsch und man sieht es hier,
bevor eine Datei in core/ angefasst wird.

Gemessen wird mit dem Werkzeug, das dafuer schon existiert:
tools.drainage_lab.bewerte() - dieselben Kennzahlen wie in §7, damit die
Zahlen direkt vergleichbar sind. Kontaktabzug und Querschnitt gehoeren nach §6
zu jeder Messung und werden hier immer mitgeschrieben.

NICHT umgesetzt, bewusst: die Talform "Tuerme" (Region 06 Guilin). Tuerme sind
eine Eigenschaft der DRAUFSICHT (isolierte Kegel in einer Ebene), nicht des
Querprofils - der Talformer hier kann sie nicht erzeugen, und ein
Schein-Ergebnis dafuer waere schlimmer als die Luecke. Braucht einen eigenen
Mechanismus.

Aufruf:
    .venv\\Scripts\\python.exe tools/skeleton_lab.py talformen
    .venv\\Scripts\\python.exe tools/skeleton_lab.py regionen
    .venv\\Scripts\\python.exe tools/skeleton_lab.py vergleich
"""

import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "lab_output", "skeleton_lab")

LAB_SEED = 20260730


# =============================================================================
# REGIONEN - ein Parametersatz je docs/regionen/NN_*/REGION.md
#
# "Region" heisst hier genau das und nichts mehr: ein Parametersatz fuer EINE
# 15-km-Karte. Kein Voronoi, keine Weltkoordinaten - die Entscheidung des
# Nutzers am 2026-07-30 war, dass die Weltkarte erst dran ist, wenn die
# einzelnen Regionen fuer sich gut aussehen.
#
# basis_m / relief_m stehen so in den REGION.md. Alles andere ist mein
# Vorschlag und ZU BESTAETIGEN - dieselbe Markierung wie in den REGION.md,
# damit nicht gegen geratene Zahlen kalibriert wird.
# =============================================================================

REGIONEN = {
    "04_alpen_wallis": {
        "basis_m": 700.0,           # REGION.md: Talsohle
        "relief_m": 3800.0,         # REGION.md
        # Talform U, glazial ueberformt -> Exponent > 1: flache Sohle, dann
        # steile Flanke.
        "talform_exponent": 2.2,
        "tal_breite_anteil": 0.62,
        "tal_breite_exponent": 0.45,
        "stufen": 0,
        "netz_dichte": 0.55,        # REGION.md: 1 Hauptfluss, 6-10 Seitenbaeche
        "maeander": 0.16,
        "noise_anteil": 0.16,
        "noise_frequenz": 0.055,
    },
    "21_bamberg_franken": {
        "basis_m": 230.0,           # REGION.md: Talsohle Regnitz
        "relief_m": 300.0,          # REGION.md
        # "breite Flussaue, sanfte Muldentaeler" -> breite Sohle, sanfte Flanke.
        "talform_exponent": 1.7,
        "tal_breite_anteil": 0.75,
        "tal_breite_exponent": 0.55,
        "stufen": 0,
        "netz_dichte": 0.7,
        "maeander": 0.26,           # "traeger maeandrierender" Hauptfluss
        "noise_anteil": 0.22,
        "noise_frequenz": 0.075,
    },
    "15_colorado_plateau": {
        "basis_m": 1900.0,          # REGION.md
        "relief_m": 1500.0,         # REGION.md
        # "Schluchten mit Stufenprofil" -> Exponent < 1 (Wand direkt am Fluss)
        # PLUS Stufen: genau der Sawtooth ueber den Flussabstand.
        "talform_exponent": 0.45,
        "tal_breite_anteil": 0.45,
        "tal_breite_exponent": 0.3,
        "stufen": 4,
        "stufen_wandanteil": 0.3,
        "netz_dichte": 0.4,         # "1 grosser, tief eingeschnitten"
        "maeander": 0.12,
        "noise_anteil": 0.08,
        "noise_frequenz": 0.05,
    },
    "17_westsibirien_moor": {
        "basis_m": 120.0,           # REGION.md
        "relief_m": 30.0,           # REGION.md - der Gegentest
        "talform_exponent": 1.4,
        "tal_breite_anteil": 0.85,
        "tal_breite_exponent": 0.6,
        "stufen": 0,
        "netz_dichte": 1.0,         # "viele traege, stark maeandrierend"
        "maeander": 0.42,
        "noise_anteil": 0.35,
        "noise_frequenz": 0.09,
    },
}


def region(name):
    """Parametersatz mit den Vorgaben, die jede Region teilt."""
    grund = {
        "stufen_wandanteil": 0.35,
        "noise_oktaven": 5,
        "noise_persistenz": 0.5,
        "noise_lacunarity": 2.2,
        # Anteil des Reliefs, den das FLUSSNETZ selbst ueberwindet. Der Rest
        # geht in die Flanken. 0.45 heisst: die Quellen liegen auf mittlerer
        # Hoehe, die Kaemme darueber - so sieht ein Hoehenprofil real aus.
        "fluss_relief_anteil": 0.45,
        # Laengsprofil: Steigung fuer Strahler-Ordnung 1, danach je Ordnung
        # flacher. Erzeugt das konkave Profil (steile Baeche, traeger
        # Hauptfluss) und ist relativ, kein Meter-Wert (§4.4).
        "steigung_abfall": 0.55,
        "verzweigungswinkel_grad": 38.0,
        "laengen_verhaeltnis": 0.78,
    }
    grund.update(REGIONEN[name])
    return grund


# =============================================================================
# 1. FLUSSNETZ ALS GRAPH
# =============================================================================

def _scheibe(radius):
    """Offset-Liste einer Scheibe, fuer das Stempeln der Sperrzone."""
    r = int(np.ceil(radius))
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    drin = (dy * dy + dx * dx) <= radius * radius
    return dy[drin], dx[drin]


def wachse_netz(size, p, seed=LAB_SEED):
    """
    Waechst einen Flussbaum VOM AUSLASS AM RAND AUFWAERTS.

    Rueckgabe: dict mit
        y, x        Knotenkoordinaten in Pixeln (float)
        eltern      Index des Knotens FLUSSABWAERTS, -1 fuer den Auslass
        ordnung     Strahler-Ordnung (erst nach strahler() gefuellt)

    Wichtige Eigenschaft, auf der alles Weitere aufbaut: eltern[i] < i. Knoten
    werden in Wachstumsreihenfolge angelegt, ein Kind also immer nach seinem
    Elternknoten. Damit sind Strahler-Ordnung (rueckwaerts) und Hoehenvergabe
    (vorwaerts) je ein einziger Lauf ohne Rekursion und ohne Zyklusgefahr.
    """
    rng = np.random.default_rng(seed)

    schritt = 0.9                                   # px je Knoten
    # netz_dichte steuert AUSSCHLIESSLICH den Mindestabstand zweier Laeufe -
    # das ist die eine Groesse, die "wie dicht" bedeutet.
    min_abstand = max(3.0, 8.0 / max(p["netz_dichte"], 0.15))
    # max_knoten ist eine SICHERHEITSGRENZE, kein Regler: das Wachstum endet
    # von selbst, wenn die Sperrzonen die Karte fuellen. Sie war vorher aus
    # netz_dichte gerechnet und lag bei ~780 Knoten, wo zum Fuellen von 192 px
    # rund 3000 noetig sind - das Netz konnte die Karte nie erreichen.
    max_knoten = int(2.5 * size * size / (min_abstand * schritt))
    max_generation = 11

    sperre = np.zeros((size, size), dtype=bool)
    s_dy, s_dx = _scheibe(min_abstand)

    def stempeln(y, x):
        yy = np.clip(np.round(y).astype(int) + s_dy, 0, size - 1)
        xx = np.clip(np.round(x).astype(int) + s_dx, 0, size - 1)
        sperre[yy, xx] = True

    # Auslass: zufaellige Stelle auf einem zufaelligen Rand, Richtung in die
    # Karte hinein. Nur EIN Auslass - "Rand, Becken spaeter" (Nutzer,
    # 2026-07-30), also findet per Konstruktion alles denselben Weg hinaus.
    seite = int(rng.integers(0, 4))
    lage = float(rng.uniform(0.25, 0.75)) * (size - 1)
    start = {0: (0.0, lage), 1: (size - 1.0, lage),
             2: (lage, 0.0), 3: (lage, size - 1.0)}[seite]
    richtung = {0: 0.0, 1: np.pi, 2: np.pi / 2, 3: -np.pi / 2}[seite]

    ys, xs, eltern = [start[0]], [start[1]], [-1]
    stempeln(np.array([start[0]]), np.array([start[1]]))

    # Ein Ast: (Knotenindex, Richtung, Generation, Segmentlaenge in Schritten)
    segment0 = 0.30 * size / schritt
    aeste = [(0, richtung, 0, segment0)]

    while aeste and len(ys) < max_knoten:
        knoten, phi, generation, laenge = aeste.pop(0)
        y, x = ys[knoten], xs[knoten]
        vorher = knoten
        gelaufen = 0

        while gelaufen < laenge and len(ys) < max_knoten:
            phi += rng.normal(0.0, p["maeander"])
            y_neu = y + schritt * np.cos(phi)
            x_neu = x + schritt * np.sin(phi)

            # Der Rand ist ERLAUBT, nicht verboten: der Auslass sitzt per
            # Definition auf y=0 bzw. x=0, und ein Lauf, der die Karte
            # verlaesst, ist genau das gewuenschte Verhalten (§3.6: "Fluesse
            # verlassen die Karte"). Die frueheren Grenzen 1.0 .. size-2.0
            # liessen den ersten Schritt von 0.9 px vom Auslass aus nicht
            # durch - der Baum bestand aus einem einzigen Knoten.
            if not (0.0 <= y_neu <= size - 1.0 and 0.0 <= x_neu <= size - 1.0):
                break
            # Die Sperrzone des eigenen Segments und des Elternasts liegt
            # zwangslaeufig direkt hinter dem Laeufer. Erst pruefen, wenn er
            # weiter als die Sperrzone gelaufen ist.
            if gelaufen > min_abstand and sperre[int(round(y_neu)), int(round(x_neu))]:
                break

            y, x = y_neu, x_neu
            ys.append(y)
            xs.append(x)
            eltern.append(vorher)
            vorher = len(ys) - 1
            stempeln(np.array([y]), np.array([x]))
            gelaufen += 1

        # Zu kurz geratenes Segment verzweigt nicht - sonst entstehen Bueschel
        # aus Stummeln an jeder Sperrzone.
        if gelaufen < max(3, min_abstand) or generation >= max_generation:
            continue

        winkel = np.deg2rad(p["verzweigungswinkel_grad"])
        neue_laenge = laenge * p["laengen_verhaeltnis"]
        if neue_laenge * schritt < 2.5:
            continue
        # Zwei Zweige als Regel, gelegentlich drei - das haelt das
        # Bifurkationsverhaeltnis bei rund 2.2 und damit im Bereich realer
        # Netze (Horton: 3-5 fuer die Ordnungszahl, das ist nicht dasselbe).
        zweige = 3 if rng.random() < 0.18 else 2
        for k in range(zweige):
            versatz = (k - (zweige - 1) / 2.0) * winkel
            aeste.append((vorher,
                          phi + versatz + rng.normal(0.0, 0.10),
                          generation + 1,
                          neue_laenge))

    return {
        "y": np.array(ys, dtype=np.float64),
        "x": np.array(xs, dtype=np.float64),
        "eltern": np.array(eltern, dtype=np.int64),
        # Der Mindestabstand ist die Bezugsgroesse fuer die Talbreite - die
        # Wasserscheide liegt zwangslaeufig ungefaehr in der Mitte zwischen
        # zwei Laeufen. Als absoluter Meterwert gefuehrt (vorher
        # tal_breite_m) war die Talbreite vom Netz entkoppelt: bei dichtem
        # Netz sass die halbe Karte im geklemmten Bereich d_norm = 1.
        "min_abstand_px": min_abstand,
    }


def strahler(netz):
    """
    Strahler-Ordnung, ein Rueckwaertslauf (eltern[i] < i, siehe wachse_netz).

    Damit hat jedes Segment eine EXPLIZITE Groessenklasse. Genau das fehlt dem
    Feldmodell nach §7 Befund 4: dort ist discharge_factor bei 1.0 geklemmt,
    ein Bach mit 50 Zellen Einzug bekommt dieselbe Kapazitaet wie ein
    Hauptfluss mit 5000. Hier ist die Hierarchie kein Rechenergebnis, sondern
    eine Eigenschaft des Graphen.
    """
    eltern = netz["eltern"]
    n = len(eltern)
    ordnung = np.ones(n, dtype=np.int32)
    # Je Knoten: hoechste Kindordnung und wie oft sie vorkommt.
    beste = np.zeros(n, dtype=np.int32)
    anzahl = np.zeros(n, dtype=np.int32)

    for i in range(n - 1, 0, -1):
        # Knoten i ist fertig: alle seine Kinder haben groesseren Index.
        if anzahl[i] == 0:
            ordnung[i] = 1
        elif anzahl[i] == 1:
            ordnung[i] = beste[i]
        else:
            ordnung[i] = beste[i] + 1

        e = eltern[i]
        if ordnung[i] > beste[e]:
            beste[e], anzahl[e] = ordnung[i], 1
        elif ordnung[i] == beste[e]:
            anzahl[e] += 1

    ordnung[0] = beste[0] + 1 if anzahl[0] > 1 else max(beste[0], 1)
    netz["ordnung"] = ordnung
    return netz


def hoehen(netz, p, meter_pro_pixel):
    """
    Hoehe je Knoten, MONOTON steigend flussaufwaerts.

    Ein Vorwaertslauf: z[i] = z[eltern] + Steigung(Ordnung) * Schrittlaenge.
    Die Steigung faellt mit der Ordnung (steile Baeche, traeger Hauptfluss) -
    das ergibt das konkave Laengsprofil realer Fluesse.

    Danach wird die erreichte Spanne auf das Ziel GESTRECKT, statt sie aus
    Steigung x Schrittzahl herauskommen zu lassen. Dieselbe Entscheidung wie in
    BaseTerrainGenerator._apply_redistribution: die Parameter formen die
    VERTEILUNG, die Spanne ist garantiert. Ein Skalieren mit positivem Faktor
    laesst die Monotonie unberuehrt - die Eigenschaft, auf die es hier ankommt.
    """
    eltern, ordnung = netz["eltern"], netz["ordnung"]
    schritt_m = 0.9 * meter_pro_pixel

    z = np.zeros(len(eltern), dtype=np.float64)
    for i in range(1, len(eltern)):
        steigung = p["steigung_abfall"] ** (ordnung[i] - 1)
        z[i] = z[eltern[i]] + steigung * schritt_m

    ziel = p["fluss_relief_anteil"] * p["relief_m"]
    if z.max() > 1e-9:
        z *= ziel / z.max()
    netz["z"] = p["basis_m"] + z
    return netz


# =============================================================================
# 2. TALFORMER - das Wellenform-Stueck
# =============================================================================

def talformer(d_norm, p):
    """
    Querprofil des Tals ueber dem NORMIERTEN Flussabstand d_norm in [0,1].
    profil(0) = 0 an der Talsohle, profil(1) = 1 am Kamm.

    Ein einziger Exponent deckt die Spalte "Talform" aus SPEZIFIKATION §2
    monoton ab:

        Exponent < 1   Wand direkt am Fluss, dann Plateau   -> Schlucht
        Exponent = 1   gerade Flanke                        -> V
        Exponent > 1   flache Sohle, dann steile Flanke     -> U (glazial)

    `stufen` legt darueber eine Treppe (der Sawtooth-Gedanke): `stufen_wandanteil`
    sagt, welcher Anteil einer Stufe Wand ist und welcher Tritt. Klein =
    ausgepraegte Klippenbaender.

    NICHT enthalten: Tuerme (Region 06). Sie sind eine Eigenschaft der
    Draufsicht, kein Querprofil - siehe Modulkopf.
    """
    profil = np.power(np.clip(d_norm, 0.0, 1.0), p["talform_exponent"])

    stufen = int(p.get("stufen", 0))
    if stufen > 0:
        wand = float(np.clip(p.get("stufen_wandanteil", 0.35), 0.05, 1.0))
        s = profil * stufen
        k = np.floor(s)
        rest = s - k
        # Tritt (flach) bis 1-wand, danach die Wand auf voller Hoehe.
        profil = (k + np.clip((rest - (1.0 - wand)) / wand, 0.0, 1.0)) / stufen

    return np.clip(profil, 0.0, 1.0)


def noise_gewicht(d_norm, exponent=1.4):
    """
    Wo das Noise wirken darf: voll auf der Flanke, auslaufend in der Talsohle
    UND auf dem Kamm. Genau die Vorgabe des Nutzers - der Fake-Erosionseffekt
    soll an beiden Enden auslaufen.
    """
    return np.power(np.sin(np.pi * np.clip(d_norm, 0.0, 1.0)), exponent)


# =============================================================================
# 3. FELD ZUSAMMENSETZEN
# =============================================================================

def baue_gelaende(name, size=192, seed=LAB_SEED, mit_noise=True):
    """
    Vollstaendiger Skelett-Pfad: Netz -> Ordnung -> Hoehen -> Abstandsfeld ->
    Talformer -> Noise-Modulation.

    Rueckgabe: (z, meter_pro_pixel, teile) - teile enthaelt die
    Zwischenergebnisse fuer den Kontaktabzug.
    """
    from gui.config.value_default import TERRAIN

    p = region(name)
    km = TERRAIN.MAP_DISTANCE_KM["default"]
    meter_pro_pixel = km * 1000.0 / size

    netz = hoehen(strahler(wachse_netz(size, p, seed)), p, meter_pro_pixel)

    # Rasterisieren. Bei mehreren Knoten je Pixel gewinnt der TIEFSTE - ein
    # Zusammenfluss darf keine Schwelle im Flussbett erzeugen.
    yi = np.clip(np.round(netz["y"]).astype(int), 0, size - 1)
    xi = np.clip(np.round(netz["x"]).astype(int), 0, size - 1)
    maske = np.zeros((size, size), dtype=bool)
    z_fluss = np.full((size, size), np.inf)
    ordnung_px = np.zeros((size, size), dtype=np.int32)
    reihenfolge = np.argsort(-netz["z"])           # hoch zuerst, tief gewinnt
    for i in reihenfolge:
        maske[yi[i], xi[i]] = True
        z_fluss[yi[i], xi[i]] = netz["z"][i]
        ordnung_px[yi[i], xi[i]] = netz["ordnung"][i]

    # Abstand IN METERN und gleichzeitig der naechste Flussknoten. sampling
    # macht aus dem Pixelabstand einen Meterabstand - eine relative Groesse
    # statt einer Pixelkonstante (§4.4).
    abstand, index = ndimage.distance_transform_edt(
        ~maske, sampling=(meter_pro_pixel, meter_pro_pixel), return_indices=True)
    z_nah = z_fluss[index[0], index[1]]
    ordnung_nah = ordnung_px[index[0], index[1]]

    # Talbreite waechst mit der Flussgroesse - grosse Fluesse, breite Taeler.
    breite = (p["tal_breite_anteil"] * netz["min_abstand_px"] * meter_pro_pixel
              * np.power(np.maximum(ordnung_nah, 1), p["tal_breite_exponent"]))
    d_norm = np.clip(abstand / breite, 0.0, 1.0)

    profil = talformer(d_norm, p)

    # Flankenhoehe so, dass die Spanne exakt basis_m .. basis_m+relief_m wird -
    # dieselbe Garantie, die §3.1 fuer den Noise-Pfad schon als erfuellt fuehrt.
    gipfel = p["basis_m"] + p["relief_m"]
    h_flanke = gipfel - float(netz["z"].max())
    z = z_nah + profil * h_flanke

    rauschen = np.zeros_like(z)
    if mit_noise and p["noise_anteil"] > 0:
        from core.terrain_generator import SimplexNoiseGenerator
        # shader_manager=None -> CPU-Pfad, damit das Labor ohne Qt/GL laeuft.
        gen = SimplexNoiseGenerator(seed=seed, shader_manager=None)
        roh = gen.generate_noise_grid(
            size=size, frequency=p["noise_frequenz"] * (64.0 / size),
            octaves=p["noise_oktaven"], persistence=p["noise_persistenz"],
            lacunarity=p["noise_lacunarity"])
        rauschen = (roh.astype(np.float64) * p["noise_anteil"] * h_flanke
                    * noise_gewicht(d_norm))
        z = z + rauschen

    teile = {
        "netz": netz, "maske": maske, "abstand": abstand, "d_norm": d_norm,
        "profil": profil, "ordnung_nah": ordnung_nah, "rauschen": rauschen,
        "h_flanke": h_flanke, "p": p, "region": name,
    }
    return z, meter_pro_pixel, teile


# =============================================================================
# 4. KENNZAHLEN - dieselben wie in §7, damit die Zahlen vergleichbar sind
# =============================================================================

def bewerte(z, meter_pro_pixel):
    import tools.drainage_lab as dl
    return dl.bewerte(z, meter_pro_pixel)


def kopf():
    print("%-26s %8s %7s %7s %7s %7s %8s %8s"
          % ("Variante", "Abfluss", "Senken", "Netz", "Nadeln", "beta",
             "Relief", "Becken"))
    print("-" * 88)


def zeile(name, w):
    print("%-26s %7.1f%% %7d %7d %7d %7.3f %8.0f %7.1f%%"
          % (name[:26], 100.0 * w["abfluss_anteil"], w["senken"], w["netz"],
             w["nadeln"], w["beta"], w["relief"], 100.0 * w["becken_anteil"]))


# =============================================================================
# 5. BILDER - §6: Kontaktabzug und Querschnitt sind Teil jeder Messung
# =============================================================================

def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _hillshade(z, meter_pro_pixel, azimut=315.0, hoehe=45.0):
    dy, dx = np.gradient(z, meter_pro_pixel)
    neigung = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspekt = np.arctan2(-dx, dy)
    az, hh = np.deg2rad(360.0 - azimut + 90.0), np.deg2rad(hoehe)
    return np.clip(np.sin(hh) * np.sin(neigung)
                   + np.cos(hh) * np.cos(neigung) * np.cos(az - aspekt), 0, 1)


def kontaktabzug(z, meter_pro_pixel, teile, datei):
    plt = _matplotlib()
    netz = teile["netz"]
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 10.5))

    b0 = ax[0, 0].imshow(z, cmap="terrain")
    ax[0, 0].set_title("Hoehe  %.0f - %.0f m" % (z.min(), z.max()))
    fig.colorbar(b0, ax=ax[0, 0], shrink=0.8)

    ax[0, 1].imshow(_hillshade(z, meter_pro_pixel), cmap="gray")
    ax[0, 1].set_title("Schummerung")

    # Netz mit Strahler-Ordnung als Linienstaerke
    ax[0, 2].imshow(z, cmap="Greys_r", alpha=0.45)
    ordnung = netz["ordnung"]
    for o in range(1, int(ordnung.max()) + 1):
        w = ordnung == o
        ax[0, 2].scatter(netz["x"][w], netz["y"][w], s=0.25 * o * o + 0.4,
                         c=[plt.cm.viridis(o / max(ordnung.max(), 1))],
                         linewidths=0)
    ax[0, 2].set_xlim(0, z.shape[1]); ax[0, 2].set_ylim(z.shape[0], 0)
    ax[0, 2].set_title("Netz: %d Knoten, Ordnung bis %d"
                       % (len(ordnung), ordnung.max()))

    b3 = ax[1, 0].imshow(teile["d_norm"], cmap="magma")
    ax[1, 0].set_title("normierter Flussabstand")
    fig.colorbar(b3, ax=ax[1, 0], shrink=0.8)

    b4 = ax[1, 1].imshow(teile["profil"], cmap="magma")
    ax[1, 1].set_title("Talformer  Exponent %.2f, Stufen %d, Talbreite %.2f x Abstand"
                       % (teile["p"]["talform_exponent"], teile["p"].get("stufen", 0),
                          teile["p"]["tal_breite_anteil"]))
    fig.colorbar(b4, ax=ax[1, 1], shrink=0.8)

    # Senken: was NICHT bis zum Rand entwaessert
    import tools.drainage_lab as dl
    empfaenger = dl.d8_receiver(z)
    ist_rand = np.zeros(z.shape, dtype=bool)
    ist_rand[0], ist_rand[-1], ist_rand[:, 0], ist_rand[:, -1] = (True,) * 4
    ziel = np.where(empfaenger >= 0, empfaenger, np.arange(z.size))
    for _ in range(int(np.ceil(np.log2(max(z.size, 2)))) + 1):
        ziel = ziel[ziel]
    entwaessert = ist_rand.ravel()[ziel].reshape(z.shape)
    ax[1, 2].imshow(entwaessert, cmap="RdYlGn", vmin=0, vmax=1)
    ax[1, 2].set_title("entwaessert zum Rand: %.1f%%" % (100.0 * entwaessert.mean()))

    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("%s   %d px, %.0f m/px" % (teile["region"], z.shape[0], meter_pro_pixel))
    fig.tight_layout()
    fig.savefig(datei, dpi=105)
    plt.close(fig)


def querschnitt(faelle, datei):
    """
    Querschnitte uebereinander. §5.1.3: vier Kennzahlen haben die
    45-Grad-Pyramiden nicht gefunden, ein Bild sofort.
    """
    plt = _matplotlib()
    fig, ax = plt.subplots(len(faelle), 1, figsize=(13, 2.6 * len(faelle)),
                           squeeze=False)
    for i, (name, z, mpp) in enumerate(faelle):
        a = ax[i, 0]
        mitte = z.shape[0] // 2
        x_km = np.arange(z.shape[1]) * mpp / 1000.0
        for versatz, farbe, breite in ((0, "black", 1.5),
                                       (-z.shape[0] // 6, "tab:blue", 0.8),
                                       (z.shape[0] // 6, "tab:orange", 0.8)):
            a.plot(x_km, z[mitte + versatz], color=farbe, lw=breite)
        a.set_title("%s   Relief %.0f m" % (name, z.max() - z.min()), fontsize=9)
        a.set_ylabel("m")
        a.grid(alpha=0.3)
    ax[-1, 0].set_xlabel("km")
    fig.tight_layout()
    fig.savefig(datei, dpi=110)
    plt.close(fig)


def talform_tafel(datei):
    """Der Talformer als reine Kurvenschar - was der Exponent tatsaechlich tut."""
    plt = _matplotlib()
    d = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for exponent, etikett in ((0.35, "0.35  Schlucht"), (0.6, "0.6"),
                              (1.0, "1.0  V"), (1.7, "1.7"),
                              (2.5, "2.5  U glazial")):
        ax[0].plot(d, talformer(d, {"talform_exponent": exponent}), label=etikett)
    ax[0].set_title("Exponent: Schlucht - V - U"); ax[0].legend(fontsize=8)
    for stufen, wand in ((3, 0.3), (4, 0.3), (6, 0.2)):
        ax[1].plot(d, talformer(d, {"talform_exponent": 0.45, "stufen": stufen,
                                    "stufen_wandanteil": wand}),
                   label="%d Stufen, Wandanteil %.1f" % (stufen, wand))
    ax[1].plot(d, noise_gewicht(d), "k--", lw=1, label="Noise-Gewicht")
    ax[1].set_title("Stufen (Klippenbaender) + Noise-Gewicht"); ax[1].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("normierter Flussabstand"); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(datei, dpi=110)
    plt.close(fig)


# =============================================================================
# 6. LAEUFE
# =============================================================================

def lauf_talformen():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ziel = os.path.join(OUTPUT_DIR, "talformer.png")
    talform_tafel(ziel)
    print("Talformer-Kurven -> %s" % ziel)
    return 0


def lauf_regionen(size=192):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Skelett-Pfad, %d px, ohne jede Erosionsiteration\n" % size)
    kopf()
    faelle = []
    for name in REGIONEN:
        z, mpp, teile = baue_gelaende(name, size=size)
        zeile(name, bewerte(z, mpp))
        kontaktabzug(z, mpp, teile, os.path.join(OUTPUT_DIR, "%s.png" % name))
        faelle.append((name, z, mpp))
    querschnitt(faelle, os.path.join(OUTPUT_DIR, "querschnitte.png"))
    print("\nBilder -> %s" % OUTPUT_DIR)
    return 0


def lauf_vergleich(size=192):
    """
    Die eigentliche Gegenprobe: Skelett gegen den heutigen Noise-Pfad, auf
    denselben zwei Zielgelaenden wie in §7 (Alpental und Mittelgebirge) und mit
    demselben Messgeraet.

    Erwartung, vor dem Lauf notiert: Abfluss deutlich ueber den 10-20% des
    Noise-Pfads, Senken deutlich unter dessen 824-1559.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import tools.erosion_lab as lab

    print("Skelett gegen Noise, %d px, beide OHNE Erosion\n" % size)
    kopf()
    faelle = []
    paare = (("04_alpen_wallis", 3800.0 + 700.0, 2.0),
             ("21_bamberg_franken", 300.0 + 230.0, 2.5))
    for name, amplitude, potenz in paare:
        roh, mpp = lab.build_terrain(size, terrain_overrides={
            "amplitude": amplitude, "redistribute_power": potenz})
        roh = roh.astype(np.float64)
        zeile("noise  " + name.split("_")[0], bewerte(roh, mpp))
        faelle.append(("noise " + name, roh, mpp))

        z, mpp2, teile = baue_gelaende(name, size=size)
        zeile("skelett " + name.split("_")[0], bewerte(z, mpp2))
        faelle.append(("skelett " + name, z, mpp2))

        # Gegenprobe: Skelett OHNE Noise. Traegt die Entwaesserung das
        # Skelett oder das Rauschen?
        z0, mpp3, _ = baue_gelaende(name, size=size, mit_noise=False)
        zeile("skelett o.Noise " + name.split("_")[0], bewerte(z0, mpp3))

    querschnitt(faelle, os.path.join(OUTPUT_DIR, "vergleich_querschnitte.png"))
    print("\nQuerschnitte -> %s" % os.path.join(OUTPUT_DIR, "vergleich_querschnitte.png"))
    return 0


def main():
    laeufe = {"talformen": lauf_talformen, "regionen": lauf_regionen,
              "vergleich": lauf_vergleich}
    if len(sys.argv) < 2 or sys.argv[1] not in laeufe:
        print(__doc__)
        return 1
    return laeufe[sys.argv[1]]()


if __name__ == "__main__":
    raise SystemExit(main())
