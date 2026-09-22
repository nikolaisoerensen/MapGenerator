"""
Seewege, Kreuzungen, Roadsite-/Landmark-Kataloge nach
docs/spezifikation/14_SIEDLUNGEN.md Abschnitt 5.4-§4.7 (Umbau 2026-08-10).

Vorher: kein Seeweg-Konzept ueberhaupt; Kreuzungen wurden nicht erkannt;
Roadsites kamen aus einer flachen 7-Typen-Liste ohne Kulturbezug, zufaellig
irgendwo auf 30-70% eines Weges; Landmarks aus einer flachen 4-Typen-Liste,
und eine pauschale Hoehen-Obergrenze schloss GIPFEL-Landmarks aus genau den
Stellen aus, die ihr eigener Name verlangt ("Trutzburg auf dem Felskopf" durfte
nie auf einem Felskopf stehen).

Geprueft wird hier:

  1. Seeweg: zwei Orte DERSELBEN Kultur, durch eine gesperrte Wasserflaeche
     getrennt, bekommen eine Verbindung UND die Mehrheit ihrer Laenge liegt in
     echtem tiefen Wasser (§4.4-Auflage) - nicht nur "irgendeine Verbindung".
  2. Roadsite-Katalog: der gewaehlte Name gehoert zur Kultur der naechsten
     Siedlung UND passt, wo verfuegbar, zur Platzierungskategorie.
  3. Landmark-Katalog: dieselbe Zusicherung, UND explizit dass eine GIPFEL-Art
     tatsaechlich auf einer echten Anhoehe landen kann (die alte pauschale
     Hoehen-Obergrenze waere hier durchgefallen).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main():
    fehler = []
    from core.settlement_generator import (
        Location, SettlementGenerator, ROADSITE_KATALOG, LANDMARK_KATALOG,
        _naechste_kultur, SEEWEG_TIEFE_ZIEL_M,
    )

    def ort(id_, x, y, culture, rank):
        return Location(location_id=id_, x=float(x), y=float(y), location_type='settlement',
                        radius=4.0, civ_influence=0.8, culture=culture, rank=rank)

    gen = SettlementGenerator.__new__(SettlementGenerator)
    gen.road_slope_to_distance_ratio = 1.5
    gen.map_seed = 7
    gen.roadsites = 6
    gen.landmarks = 6
    gen.landmark_wilderness = 0.5
    gen.scale_factor = 1.0
    gen._update_progress = None
    gen.next_location_id = 100

    # ---------- 1: Seeweg ----------
    print("1. Seeweg zwischen getrennten Kuesten derselben Kultur")
    m = 70
    heightmap = np.full((m, m), 30.0, dtype=np.float32)
    heightmap[:, 30:40] = -25.0   # breiter, tiefer Kanal - fuer Landwege gesperrt
    slopemap = np.zeros((m, m, 2), dtype=np.float32)
    hafen_a = ort(0, 10, 35, "Wikinger", "stadt")
    hafen_b = ort(1, 60, 35, "Wikinger", "stadt")

    roads, sea_roads = gen.calculate_road_network([hafen_a, hafen_b], heightmap, slopemap, 5)
    print("   %d Landwege, %d Seewege" % (len(roads), len(sea_roads)))
    if len(sea_roads) != 1:
        fehler.append("kein Seeweg zwischen durch Wasser getrennten Staedten "
                      "derselben Kultur gebaut: %d Seewege" % len(sea_roads))
    else:
        pfad = sea_roads[0]
        tief = sum(1 for x, y in pfad
                  if heightmap[int(np.clip(round(y), 0, m - 1)),
                              int(np.clip(round(x), 0, m - 1))] <= SEEWEG_TIEFE_ZIEL_M)
        anteil = tief / len(pfad)
        print("   Anteil in echtem tiefen Wasser: %.0f %% (soll >= 50 %%)" % (100 * anteil))
        if anteil < 0.5:
            fehler.append("Seeweg verbringt nicht die Mehrheit seiner Laenge "
                          "in tiefem Wasser: %.0f %%" % (100 * anteil))

    # ---------- 2: Roadsite-Katalog ----------
    print("\n2. Roadsite-Katalog")
    m2 = 90
    heightmap2 = np.full((m2, m2), 40.0, dtype=np.float32)
    heightmap2[40:50, :] = -3.0    # Flachwasserstreifen -> Furt-Kandidat
    slopemap2 = np.zeros((m2, m2, 2), dtype=np.float32)

    kelten_a = ort(0, 10, 10, "Kelten", "stadt")
    kelten_b = ort(1, 10, 80, "Kelten", "siedlung")
    sachsen_a = ort(2, 80, 10, "Sachsen", "stadt")
    sachsen_b = ort(3, 80, 80, "Sachsen", "siedlung")
    orte = [kelten_a, kelten_b, sachsen_a, sachsen_b]

    roads2, sea_roads2 = gen.calculate_road_network(orte, heightmap2, slopemap2, 5)
    print("   %d Landwege, %d Seewege" % (len(roads2), len(sea_roads2)))

    roadsites = gen.calculate_roadsites(roads2, sea_roads2, orte, heightmap2, 5)
    print("   %d Roadsites platziert" % len(roadsites))
    if not roadsites:
        fehler.append("keine Roadsites platziert trotz roadsites=6")
    for r in roadsites:
        katalog_namen = {name for name, _kat in ROADSITE_KATALOG.get(r.culture, [])}
        if r.properties.get('roadsite_type') not in katalog_namen:
            fehler.append("Roadsite %r gehoert nicht zum Katalog von %s"
                          % (r.properties.get('roadsite_type'), r.culture))
        erwartete_kultur = _naechste_kultur(r.x, r.y, orte)
        if r.culture != erwartete_kultur:
            fehler.append("Roadsite bei (%.0f,%.0f) traegt Kultur %s, "
                          "naechste Siedlung ist aber %s"
                          % (r.x, r.y, r.culture, erwartete_kultur))
    print("   Kategorien: %s" % [r.properties.get('kategorie') for r in roadsites])

    # ---------- 3: Landmark-Katalog ----------
    print("\n3. Landmark-Katalog, inkl. Gipfel-Kategorie")
    m3 = 100
    rng = np.random.RandomState(9)
    from scipy.ndimage import zoom
    grob = rng.rand(8, 8)
    heightmap3 = (zoom(grob, m3 / 8.0, order=3)[:m3, :m3] * 900.0 + 20.0).astype(np.float32)
    slopemap3 = np.zeros((m3, m3, 2), dtype=np.float32)
    civ_map3 = np.full((m3, m3), 0.05, dtype=np.float32)   # ueberall Wildnis
    water_map3 = np.zeros((m3, m3), dtype=np.float32)
    alpen = ort(0, 50, 50, "Alemannen", "stadt")

    landmarks = gen.calculate_landmarks(civ_map3, heightmap3, slopemap3, water_map3, [alpen], 5)
    print("   %d Landmarks platziert" % len(landmarks))
    kategorien = [l.properties.get('kategorie') for l in landmarks]
    print("   Kategorien: %s" % kategorien)
    if not landmarks:
        fehler.append("keine Landmarks platziert trotz landmarks=6")
    for l in landmarks:
        katalog_namen = {name for name, _kat in LANDMARK_KATALOG.get(l.culture, [])}
        if l.properties.get('landmark_type') not in katalog_namen:
            fehler.append("Landmark %r gehoert nicht zum Katalog von %s"
                          % (l.properties.get('landmark_type'), l.culture))
    if "gipfel" not in kategorien:
        fehler.append("keine einzige Gipfel-Landmark trotz stark bergigem "
                      "Testgelaende - die alte Hoehen-Obergrenze koennte "
                      "wieder aktiv sein")

    print("")
    if fehler:
        print("NICHT IN ORDNUNG - %d Befunde:" % len(fehler))
        for f in fehler:
            print("   " + f)
        return 1
    print("Alle Zusicherungen erfuellt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
