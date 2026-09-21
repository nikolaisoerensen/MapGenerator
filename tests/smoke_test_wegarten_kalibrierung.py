"""
Ticket #41 ("Wegekostenstaffelung nach Wegart kalibrieren").

WAS HIER GEPRUEFT WIRD. core/wegarten.py ersetzt den einen WEGERABATT-Schritt
in calculate_road_network() durch eine Staffel (core/daten/wegarten.toml):
ein Wegstueck, das oefter Teil einer gebauten Route war, wird selbst billiger
- ein Regelkreis, der auf sein eigenes Ergebnis zurueckwirkt und deshalb
wegdriften kann, wenn die Zahlen nicht stimmen (docs siehe core/wegarten.py-
Modul-Docstring).

Dieser Test baut fuer 3 Seeds x 3 Kartengroessen (128/256/512, siehe
CLAUDE.md "Tests mit den ECHTEN Eingabegroessen bauen" - das sind reale
map_size-Kandidaten dieses Projekts) je ein synthetisches Siedlungsnetz,
laesst calculate_road_network() direkt darueber routen (wie
tests/smoke_test_settlement_roads.py - kein voller Terrain-Lauf noetig, die
Kostenfeld-/Wegarten-Logik haengt nur an heightmap/slopemap/settlements) und
haelt die Kennzahlen aus core.wegarten.wegekennzahlen() gegen das Band in
tests/toleranzen.toml Abschnitt [wegenetz].

ZUSAETZLICH (Abnahmekriterium 6, "sieht nach gewachsenen Wegen aus, nicht
nach einem Stern oder einem Gitter"): eine Bild-Datei wird fuer einen Lauf
gespeichert (Pfad wird ausgegeben, fuer die echte visuelle Bestaetigung durch
den Nutzer), und headless wird zusaetzlich geprueft, dass ueberhaupt
Buendelung stattfindet (Bandwert buendelung_mindestanteil) - ein reiner Stern
haette in JEDEM Lauf 0 % "weg"/"strasse", weil kein Pixel je zweimal benutzt
wird.
"""
import logging
import os
import sys
import tomllib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _lade_band():
    pfad = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toleranzen.toml")
    with open(pfad, "rb") as datei:
        daten = tomllib.load(datei)
    return daten["wegenetz"]


def _baue_siedlungsnetz(size, n_orte, seed):
    """Synthetisches Huegelgelaende + zufaellig platzierte Siedlungen -
    dieselbe Idee wie tests/smoke_test_settlement_roads.py, nur groesser und
    mit realen Kartengroessen statt 40x40/60x60-Miniaturen."""
    from core.settlement_generator import Location, SettlementGenerator

    rng = np.random.RandomState(seed)
    x = np.linspace(0, 6.0, size)
    y = np.linspace(0, 5.0, size)
    heightmap = (20.0 + 15.0 * np.sin(x)[None, :] * np.cos(y)[:, None]).astype(np.float32)
    dz_dy, dz_dx = np.gradient(heightmap)
    slopemap = (np.stack([dz_dx, dz_dy], axis=-1) * 0.05).astype(np.float32)

    xs = rng.uniform(size * 0.1, size * 0.9, n_orte)
    ys = rng.uniform(size * 0.1, size * 0.9, n_orte)
    kulturen = ["A", "B"]
    raenge = ["dorf", "siedlung", "stadt"]
    orte = [
        Location(location_id=i, x=float(xs[i]), y=float(ys[i]), location_type="settlement",
                radius=4.0, civ_influence=0.8, culture=kulturen[i % 2], rank=raenge[i % 3])
        for i in range(n_orte)
    ]

    gen = SettlementGenerator.__new__(SettlementGenerator)
    gen.road_slope_to_distance_ratio = 1.5
    gen.map_seed = seed
    gen._update_progress = None
    gen.logger = logging.getLogger("smoke_test_wegarten_kalibrierung")

    roads, _sea_roads = gen.calculate_road_network(orte, heightmap, slopemap, 5)
    return gen, roads, orte


def _speichere_bild(size, seed, gen, roads, orte, fehler):
    """Rein informativ - Abnahmekriterium 6 verlangt eine gespeicherte
    Bilddatei, ueber die sich das Netz plausibilisieren laesst. Ein
    fehlschlagendes Speichern bricht den Test NICHT ab (matplotlib-Import
    o.ae.) - es ist ein Zusatzbefund, kein Pruefkriterium."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from core.wegarten import rabattfeld_und_stufen

        _rabatt, stufe = rabattfeld_und_stufen(gen.letzte_wegnutzung, gen._wegarten)
        farben = {-1: (0.85, 0.85, 0.85), 0: (0.55, 0.4, 0.25),
                 1: (0.35, 0.35, 0.9), 2: (0.9, 0.2, 0.2)}
        bild = np.zeros(stufe.shape + (3,), dtype=np.float32)
        for wert, farbe in farben.items():
            bild[stufe == wert] = farbe
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(bild, origin="upper")
        ax.scatter([o.x for o in orte], [o.y for o in orte], c="black", s=12, zorder=3)
        ax.set_title("Wegenetz size=%d seed=%d (grau=roh, braun=pfad, "
                     "blau=weg, rot=strasse)" % (size, seed))
        import tempfile
        zielordner = os.path.join(tempfile.gettempdir(), "mapgenerator_wegarten_kalibrierung")
        os.makedirs(zielordner, exist_ok=True)
        ziel = os.path.join(zielordner, "wegenetz_size%d_seed%d.png" % (size, seed))
        fig.savefig(ziel, dpi=110)
        plt.close(fig)
        print("   Bild gespeichert: %s (bitte fuer die echte visuelle "
              "Bestaetigung ansehen)" % ziel)
    except Exception as e:                                    # pragma: no cover
        fehler_hinweis = "Bild konnte nicht gespeichert werden (%s) - kein Testfehler" % e
        print("   " + fehler_hinweis)


def main():
    from core.wegarten import wegekennzahlen

    band = _lade_band()
    fehler = []
    buendelung_gesehen = False

    print("Wegarten-Kalibrierung (Ticket #41) - 3 Kartengroessen x 3 Seeds\n")
    for size in (128, 256, 512):
        for seed in (1, 2, 3):
            gen, roads, orte = _baue_siedlungsnetz(size, 16, seed)
            kz = wegekennzahlen(roads, gen.letzte_wegnutzung, gen._wegarten, orte)
            anteil = kz["anteil_je_wegart"]
            laenge_norm = (kz["gesamtlaenge"] / kz["anzahl_knoten"]) / size
            buendelung = anteil.get("weg", 0.0) + anteil.get("strasse", 0.0)
            buendelung_gesehen = buendelung_gesehen or buendelung >= band["buendelung_mindestanteil"]

            praefix = "size=%-3d seed=%d" % (size, seed)
            print("%s: Wege=%-3d laenge/knoten/size=%.3f  "
                  "pfad=%.2f weg=%.2f strasse=%.2f  umweg=%.3f"
                  % (praefix, kz["anzahl_wege"], laenge_norm,
                     anteil.get("pfad", 0.0), anteil.get("weg", 0.0),
                     anteil.get("strasse", 0.0), kz["mittlerer_umwegfaktor"]))

            if kz["anzahl_knoten"] != 16:
                fehler.append("%s: anzahl_knoten=%s statt 16" % (praefix, kz["anzahl_knoten"]))
            if not (band["laenge_je_knoten_je_kartengroesse_min"] <= laenge_norm
                    <= band["laenge_je_knoten_je_kartengroesse_max"]):
                fehler.append("%s: laenge_je_knoten/size=%.3f ausserhalb [%.2f, %.2f]"
                              % (praefix, laenge_norm,
                                 band["laenge_je_knoten_je_kartengroesse_min"],
                                 band["laenge_je_knoten_je_kartengroesse_max"]))
            for art in ("pfad", "weg", "strasse"):
                wert = anteil.get(art, 0.0)
                lo, hi = band["anteil_%s_min" % art], band["anteil_%s_max" % art]
                if not (lo <= wert <= hi):
                    fehler.append("%s: anteil_%s=%.3f ausserhalb [%.2f, %.2f]"
                                  % (praefix, art, wert, lo, hi))
            umweg = kz["mittlerer_umwegfaktor"]
            if not (band["umwegfaktor_min"] <= umweg <= band["umwegfaktor_max"]):
                fehler.append("%s: umwegfaktor=%.3f ausserhalb [%.2f, %.2f]"
                              % (praefix, umweg, band["umwegfaktor_min"], band["umwegfaktor_max"]))

            if size == 256 and seed == 1:
                _speichere_bild(size, seed, gen, roads, orte, fehler)

    print("")
    if not buendelung_gesehen:
        fehler.append(
            "In KEINEM der 9 Laeufe wurde die Buendelungsschwelle "
            "(weg+strasse >= %.2f) erreicht - das Netz sieht nach einem "
            "Stern aus lauter Einzelverbindungen aus, nicht nach gewachsenen "
            "Wegen (Abnahmekriterium 6)." % band["buendelung_mindestanteil"])

    if fehler:
        print("NICHT IN ORDNUNG - %d Befund(e):" % len(fehler))
        for f in fehler:
            print("   " + f)
        return 1
    print("Alle Zusicherungen erfuellt - der Regelkreis blieb bei allen "
          "9 Kombinationen im Band.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
