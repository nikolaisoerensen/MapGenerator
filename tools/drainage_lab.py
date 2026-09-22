"""
Path: tools/drainage_lab.py

WERKZEUG fuer zwei Nutzer-Hypothesen zur Erosion (2026-07-30):

(1) "Wasser sucht immer einen Ausweg aus dem lokalen Tal. Flood-fill-maessig
    laeuft es irgendwohin ueber, und erst wenn ein Weg von der Karte hinaus
    gefunden ist, entsteht mit tausenden Partikeln eine Rinne."

    -> Messgroesse ENTWAESSERUNGSANTEIL: welcher Anteil der Karte findet per
    D8-Abfluss einen Weg BIS ZUM KARTENRAND, und welcher endet in einer
    abflusslosen Senke? Das ist die Kennzahl, die im Erosion-Labor bisher
    fehlte - dort wurde nur die GROESSE des Kanalnetzes gemessen, nicht ob es
    irgendwo hinfuehrt.

(2) "Jedes Partikel arbeitet zu viel. Lieber mehr Iterationen als zu starke
    Tropfen."

    -> Sweep bei KONSTANTEM PRODUKT aus Staerke und Schrittzahl. Wenn die
    Hypothese stimmt, verbessern sich Entwaesserung und Nadelzahl, obwohl die
    Gesamtarbeit gleich bleibt.

Zwei Zielgelaende, entsprechend den beiden aktiven Regionen
(docs/regionen/): Alpental mit hohem Relief und Nebelrode mit geringem.
Ein Verfahren muss auf BEIDEN bestehen - eine Aenderung, die das Alpental
verbessert und das Nebelrode zerlegt, ist keine Verbesserung.

Aufruf:
    .venv\\Scripts\\python.exe tools/drainage_lab.py ausgangslage
    .venv\\Scripts\\python.exe tools/drainage_lab.py staerke
"""

import os
import sys

import numpy as np
from scipy import ndimage

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "lab_output", "erosion_lab")

_QT_APP = None


def _qt():
    """Offscreen-Qt fuer den GPU-Pfad. MUSS modulweit gehalten werden, sonst
    raeumt Python es ab waehrend der GL-Kontext lebt - Segfault ohne Meldung."""
    global _QT_APP
    from PyQt6.QtGui import QGuiApplication
    if _QT_APP is None:
        _QT_APP = QGuiApplication.instance() or QGuiApplication([])
    return _QT_APP


# =============================================================================
# KENNZAHLEN
# =============================================================================

def d8_receiver(z):
    """
    Fuer jede Zelle der steilste Nachbar bergab, als flacher Index.
    -1 heisst: kein Nachbar ist tiefer (abflusslose Senke) oder Randzelle,
    die hinausfliesst.
    """
    hoehe, breite = z.shape
    flach = z.ravel()
    empfaenger = np.full(z.size, -1, dtype=np.int64)
    bestes = np.zeros(z.size, dtype=np.float64)

    y, x = np.mgrid[0:hoehe, 0:breite]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            drin = (ny >= 0) & (ny < hoehe) & (nx >= 0) & (nx < breite)
            entfernung = np.sqrt(dy * dy + dx * dx)
            gefaelle = np.zeros_like(z, dtype=np.float64)
            nyc, nxc = np.clip(ny, 0, hoehe - 1), np.clip(nx, 0, breite - 1)
            gefaelle[drin] = (z[drin] - z[nyc[drin], nxc[drin]]) / entfernung
            besser = drin & (gefaelle > bestes.reshape(z.shape))
            ziel = (nyc * breite + nxc).ravel()
            empfaenger[besser.ravel()] = ziel[besser.ravel()]
            bestes.reshape(z.shape)[besser] = gefaelle[besser]
    return empfaenger


def drainage_share(z):
    """
    Anteil der Karte, der bis zum RAND entwaessert - gegen den Anteil, der in
    abflusslosen Senken endet.

    Zeigergestuetzt (pointer jumping): jede Zelle zeigt auf ihren Empfaenger,
    dann wird die Kette log(N)-mal verdoppelt, bis alle beim Endpunkt sind.
    Vektorisiert, kein Python-Loop ueber Zellen.
    """
    hoehe, breite = z.shape
    empfaenger = d8_receiver(z)

    # Randzellen ohne tieferen Nachbarn fliessen HINAUS, innere ohne tieferen
    # Nachbarn sind Senken. Das ist die Unterscheidung, um die es geht.
    ist_rand = np.zeros(z.shape, dtype=bool)
    ist_rand[0], ist_rand[-1], ist_rand[:, 0], ist_rand[:, -1] = True, True, True, True
    ist_rand = ist_rand.ravel()

    ziel = np.where(empfaenger >= 0, empfaenger, np.arange(z.size))
    for _ in range(int(np.ceil(np.log2(max(z.size, 2)))) + 1):
        ziel = ziel[ziel]

    endet_am_rand = ist_rand[ziel]
    senken = int(np.sum((empfaenger < 0) & (~ist_rand)))
    return {
        "abfluss_anteil": float(np.mean(endet_am_rand)),
        "senken": senken,
    }


def river_metrics(z, schwelle=25.0):
    """Netzgroesse, Zusammenfluesse, Randberuehrung, Nadeln."""
    import tools.erosion_lab as lab
    acc = lab.flow_accumulation(z)
    kanal = acc >= schwelle
    marken, n = ndimage.label(kanal, structure=np.ones((3, 3)))
    groessen = np.bincount(marken.ravel())[1:] if n else np.array([0])
    rand = int(kanal[0].sum() + kanal[-1].sum() + kanal[:, 0].sum() + kanal[:, -1].sum())
    d = np.abs(np.diff(z, axis=1))
    relief = float(z.max() - z.min()) or 1.0
    return {
        "netz": int(groessen.max()) if len(groessen) else 0,
        "kanal_anteil": float(kanal.mean()),
        "rand": rand,
        "nadeln": int((d > 0.03 * relief).sum()),
    }


def fill_requirement(z):
    """
    Wie hoch muesste das Wasser in den Becken stehen, damit jedes einen
    Ueberlauf zum Rand hat? Das ist genau das Flood-Fill, das der Nutzer
    beschreibt - hier nur GEMESSEN, nicht angewandt.

    Wird das Ergebnis mit dem tatsaechlichen Wasserstand der Simulation
    verglichen, sagt es, ob die Becken im Modell ueberhaupt bis zum Ueberlauf
    volllaufen. Tun sie es nicht, gibt es keinen Durchfluss, und ohne
    Durchfluss kann am Ueberlaufpunkt keine Rinne entstehen.
    """
    from skimage.morphology import reconstruction
    saat = z.max() * np.ones_like(z)
    saat[0], saat[-1], saat[:, 0], saat[:, -1] = (
        z[0], z[-1], z[:, 0], z[:, -1])
    gefuellt = reconstruction(saat, z, method="erosion")
    fuellung = gefuellt - z
    return {
        "fuell_mittel": float(fuellung.mean()),
        "fuell_max": float(fuellung.max()),
        "becken_anteil": float(np.mean(fuellung > 0.01)),
    }


def bewerte(z, mpp):
    import tools.erosion_lab as lab
    werte = dict(drainage_share(z))
    werte.update(river_metrics(z))
    werte.update(fill_requirement(z))
    werte["beta"] = lab.slope_area_beta(z, lab.flow_accumulation(z), mpp)
    werte["relief"] = float(z.max() - z.min())
    return werte


# =============================================================================
# ZIELGELAENDE - die beiden aktiven Regionen
# =============================================================================

def gelaende(art, size=192):
    """
    art "alpental"      hohes Relief, entspricht 04 Alpen Wallis
    art "mittelgebirge" geringes Relief, entspricht 21 Bamberg
    """
    import tools.erosion_lab as lab
    if art == "alpental":
        return lab.build_terrain(size, terrain_overrides={
            "amplitude": 3800.0, "redistribute_power": 2.0})
    return lab.build_terrain(size, terrain_overrides={
        "amplitude": 400.0, "redistribute_power": 2.5})


def laufe(art, size=192, cpu=False, **overrides):
    import tools.erosion_lab as lab
    from core.erosion_generator import HydraulicFieldSimulator
    from managers.shader_manager import ShaderManager
    _qt()

    t, mpp = gelaende(art, size)
    # shader_manager=None ist die einzige saubere Art, den CPU-Pfad zu
    # erzwingen: has_gpu_path() haengt bewusst an der Dispatch-Registrierung,
    # nicht an der Existenz eines ShaderManagers.
    sim = HydraulicFieldSimulator(
        shader_manager=None if cpu else ShaderManager())
    p = lab.default_parameters()
    p.update(overrides)
    # SPEZIFIKATION §4.2: welcher Pfad laeuft, VOR jeder Schlussfolgerung.
    # Dreimal an einem Tag wurde eine CPU-Aenderung ueber den GPU-Pfad
    # gemessen und das Ergebnis als "unwirksam" gemeldet.
    print("   [Pfad: %s]" % ("GPU" if sim.has_gpu_path() else "CPU"))
    out = sim.simulate(t, np.full(t.shape, 50.0, dtype=np.float32), p, mpp)
    z = t.astype(np.float64) - out["erosion_map"] + out["sedimentation_map"]
    return t.astype(np.float64), z, mpp, out


def kopf():
    print("%-30s %8s %6s %6s %6s %6s %7s %7s %7s"
          % ("Variante", "Abfluss", "Senken", "Netz", "Rand", "Nadeln",
             "beta", "Fuell.", "Becken"))


def zeile(name, w):
    print("%-30s %7.1f%% %6d %6d %6d %6d %7.3f %7.1f %6.1f%%"
          % (name, 100 * w["abfluss_anteil"], w["senken"], w["netz"],
             w["rand"], w["nadeln"], w["beta"], w["fuell_mittel"],
             100 * w["becken_anteil"]))


# =============================================================================
# ABLAEUFE
# =============================================================================

def lauf_ausgangslage():
    """Wo stehen wir - auf beiden Zielgelaenden, vor und nach der Erosion."""
    for art in ("alpental", "mittelgebirge"):
        print("\n=== %s ===" % art.upper())
        roh, nach, mpp, out = laufe(art)
        kopf()
        zeile("roh (vor der Erosion)", bewerte(roh, mpp))
        w = bewerte(nach, mpp)
        zeile("nach der Erosion", w)
        # Die entscheidende Gegenueberstellung: was zum Ueberlaufen NOETIG
        # waere gegen das, was tatsaechlich im Becken steht.
        tiefe = out["water_depth_map"]
        print("   Wasserstand der Simulation: Mittel %.2f m, max %.2f m"
              % (float(tiefe.mean()), float(tiefe.max())))
        print("   noetig zum Ueberlaufen:     Mittel %.2f m, max %.2f m"
              % (w["fuell_mittel"], w["fuell_max"]))
        print("   Schritte %d, konvergiert %s, Relief %.0f m"
              % (out["steps_taken"], out["converged"], w["relief"]))
    return 0


def lauf_staerke():
    """
    Hypothese 2: schwaechere Tropfen, mehr Iterationen - bei KONSTANTEM
    Produkt aus Staerke und Schrittzahl, also gleicher Gesamtarbeit.
    """
    import tools.erosion_lab as lab
    basis = lab.default_parameters()
    ks0, schritte0 = basis["erosion_strength"], basis["max_steps"]
    print("Ausgangswerte: erosion_strength %.2f, max_steps %d (Produkt %.0f)"
          % (ks0, schritte0, ks0 * schritte0))

    for art in ("alpental", "mittelgebirge"):
        print("\n=== %s ===" % art.upper())
        kopf()
        for faktor in (1.0, 0.5, 0.25):
            ks = ks0 * faktor
            schritte = int(schritte0 / faktor)
            # Konvergenzkriterium AUS. Sonst bricht der schwaechere Lauf
            # frueher ab (kleinere Aenderung pro Schritt ist genau das, was
            # variiert wird) und die Schrittzahl waere wirkungslos.
            _, nach, mpp, out = laufe(art, erosion_strength=ks,
                                      max_steps=schritte,
                                      convergence_threshold=1e-12)
            zeile("Ks %.3f x %d Schr." % (ks, out["steps_taken"]),
                  bewerte(nach, mpp))
    return 0


def lauf_wasser():
    """
    Hypothese 1, erster Teil: laufen die Becken ueberhaupt voll?

    Gemessen steht 0.29 m Wasser, wo 13.0 m zum Ueberlaufen noetig waeren.
    Die Verdunstung ist der Verdaechtige - sie nimmt das Wasser weg, bevor
    ein Becken seinen Ueberlaufpunkt erreicht. Ohne Ueberlauf gibt es keinen
    Durchfluss durch das Becken und damit keine Rinne am Ausgang.
    """
    for art in ("alpental", "mittelgebirge"):
        print("\n=== %s ===" % art.upper())
        kopf()
        for verd in (0.015, 0.005, 0.001, 0.0):
            _, nach, mpp, out = laufe(art, evaporation_rate=verd)
            w = bewerte(nach, mpp)
            zeile("Verdunstung %.3f" % verd, w)
            print("      Wasser %.2f m (max %.1f), noetig %.2f m, Schritte %d"
                  % (float(out["water_depth_map"].mean()),
                     float(out["water_depth_map"].max()),
                     w["fuell_mittel"], out["steps_taken"]))
    return 0


def lauf_routing(size=128, schritte=3000):
    """
    Der Versuch zu Hypothese 1: geroutete Einzugsflaeche statt lokalem
    Netto-Fluss.

    LAEUFT AUF DER CPU, weil dort die Aenderung sitzt. Der GPU-Shader kennt
    sie nicht - eine Messung ueber die GPU wuerde "unwirksam" melden, wie es
    bei CHANNEL_WIDTH_SIGMA_PX tatsaechlich passiert ist (Spezifikation §4.2).
    Beide Zeilen derselbe Pfad, damit der Vergleich zaehlt.
    """
    import time
    from core.erosion_generator import HydraulicFieldSimulator as H

    for art in ("alpental", "mittelgebirge"):
        print("\n=== %s (CPU, %d px, %d Schritte) ===" % (art.upper(), size, schritte))
        kopf()
        for name, intervall, krit in (("A ohne Routing", 0, 40.0),
                                      ("B Routing, A_krit 40", 250, 40.0),
                                      ("C Routing, A_krit 150", 250, 150.0)):
            H.DRAINAGE_ROUTING_INTERVAL = intervall
            H.ROUTED_AREA_CRITICAL_CELLS = krit
            t0 = time.time()
            _, nach, mpp, out = laufe(art, size=size, cpu=True,
                                      max_steps=schritte,
                                      convergence_threshold=1e-12)
            zeile(name, bewerte(nach, mpp))
            print("      %.0f s, Bilanz %.1e, Relief %.0f m"
                  % (time.time() - t0, out["mass_balance"],
                     float(nach.max() - nach.min())))
        H.DRAINAGE_ROUTING_INTERVAL = 0
        H.ROUTED_AREA_CRITICAL_CELLS = 40.0
    return 0


def lauf_deckel(size=128, schritte=3000):
    """
    Grabungsklemme einzeln und zusammen mit dem Routing.

    Vier Zeilen, die genau EINEN Unterschied zur Vorzeile haben - sonst sagt
    der Vergleich nicht, welche der beiden Aenderungen gewirkt hat. Bei der
    Erosion haben zwei gleichzeitige Aenderungen (Amplitude und Glaettung)
    schon einmal eine halbe Stunde Fehlersuche gekostet.
    """
    import time
    from core.erosion_generator import HydraulicFieldSimulator as H

    varianten = (
        ("A nichts",                   0.0, 0),
        ("B nur Klemme 1.0",           1.0, 0),
        ("C nur Klemme 0.25",          0.25, 0),
        ("D Klemme 1.0 + Routing",     1.0, 250),
    )
    for art in ("alpental", "mittelgebirge"):
        print("\n=== %s (CPU, %d px, %d Schritte) ===" % (art.upper(), size, schritte))
        kopf()
        for name, klemme, intervall in varianten:
            H.MAX_DIG_TO_NEIGHBOUR_FRACTION = klemme
            H.DRAINAGE_ROUTING_INTERVAL = intervall
            t0 = time.time()
            _, nach, mpp, out = laufe(art, size=size, cpu=True,
                                      max_steps=schritte,
                                      convergence_threshold=1e-12)
            zeile(name, bewerte(nach, mpp))
            print("      %.0f s, %d Schritte, Bilanz %.1e, Relief %.0f m"
                  % (time.time() - t0, out["steps_taken"], out["mass_balance"],
                     float(nach.max() - nach.min())))
        H.MAX_DIG_TO_NEIGHBOUR_FRACTION = 0.0
        H.DRAINAGE_ROUTING_INTERVAL = 0
    return 0


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name = sys.argv[1] if len(sys.argv) > 1 else "ausgangslage"
    return {"ausgangslage": lauf_ausgangslage,
            "staerke": lauf_staerke,
            "wasser": lauf_wasser,
            "routing": lauf_routing,
            "deckel": lauf_deckel}[name]()


if __name__ == "__main__":
    sys.exit(main())
