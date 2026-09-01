"""
See-Voronoi-Gliederung: Seegrad, Zieltiefe, Uferregionen, Seewege ab Grad 1.

ANLASS (2026-08-11). Nutzer-Vorgabe: "falls wir noch keine see voronois haben
dann bitte. und dazu die tiefe und nachbarregionen der seefelder und
seerouten" - docs/OFFENE_PUNKTE.md 3.1 (Seegrad ueber Voronoi-Zellgraph),
3.2 (Schelf durch Gradtabelle ersetzen), 3.3 (Seewege ab Grad 1 statt ab
10m Tiefe), 3.6 (Seezellen mit Uferregionen).

Vorher: der Kuestenschelf war ein reiner Distanzgradient
`-t*(1-exp(-d/L))` ohne jede Zellstruktur; Seewege pruften "tiefes Wasser"
ausschliesslich ueber eine feste Hoehenschwelle (-10m).

Sieben Zusicherungen:
1. Der Seegrad WAECHST mit der Zellentfernung von der Kueste (Breitensuche
   ueber den Zellnachbarschaftsgraphen) - kein Zufallsmuster.
2. Die Zieltiefe je Grad haelt sich ungefaehr an die Standardtabelle
   (0/-40/-90/-150/-200m), abgesehen von der beabsichtigten Glaettung an den
   Zellgrenzen.
3. Uferregionen sind PLAUSIBEL: die naechste Uferregion einer Seezelle stimmt
   ueberwiegend mit der tatsaechlich naechstgelegenen Landregion ueberein
   (Kreuzpruefung gegen eine unabhaengige Distanztransformation je Region).
4. Die Karte bleibt tiling-frei genug fuers Auge - die Zellgroesse (See vs.
   Land) liegt in vergleichbarer Groessenordnung (docs/KLIMA_UND_SEE.md
   Vorbehalt: "das sollte man messen, bevor man es festschreibt").
5. FJORDLAND-Ufer faellt STEILER ab als die Standardtabelle (3.6, Nutzer
   2026-08-07: "das faellt vor dem fjordland steil ab").
6. HUEGELLAND-Ufer bleibt bei Grad 1 FLACHER als die Standardtabelle ("das
   huegelland ab seegrad 2" - erst dort beginnt die eigentliche Vertiefung).
7. SEEEIS liegt ausschliesslich auf See-Zellen, deren naechstes Ufer die
   Morobora ist ("die taiga bekommt seeeis"), mit einer WAHRSCHEINLICHKEIT je
   Seegrad (100/75/50/25 % bei Grad 0-3, 0% ab Grad 4 - Nutzer-Nachbesserung
   2026-08-11: "grade 0 ist 100% eis grade 1 75% chance ... grade 3 ist
   25%"). Ueber mehrere Seeds gemittelt, weil einzelne Grade oft nur eine
   Handvoll Zellen stellen - bei so kleinem n schwankt ein einzelner Seed
   erheblich (eine Karte mit 16 Grad-1-Morobora-Zellen bei p=0.75 kann rein
   zufaellig 5 statt 12 treffen; erst ueber mehrere Karten gemittelt wird die
   Vorgabe selbst pruefbar).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main():
    fehler = []
    from PyQt6.QtGui import QGuiApplication
    _app = QGuiApplication.instance() or QGuiApplication([])

    import core.terrain_weltkarte as rw
    from scipy import ndimage

    SIZE, SEED = 512, 20260804
    H, felder = rw.weltfeld(SIZE, SEED)
    mpp = rw.WELT_KM * 1000.0 / SIZE
    maske = H > 0.0  # Land NACH dem Schelf - fuer diese Pruefungen ausreichend
    seegrad = felder["seegrad"]
    seegrad_tiefe = felder["seegrad_tiefe"]
    ufer_a = felder["ufer_region_a"]

    print("1. Seegrad waechst mit dem Kuestenabstand")
    kuestenabstand_m = ndimage.distance_transform_edt(~maske) * mpp
    baender = [(0, 500), (500, 1500), (1500, 3000), (3000, 6000), (6000, 1e9)]
    mittel = []
    for lo, hi in baender:
        band = (~maske) & (kuestenabstand_m >= lo) & (kuestenabstand_m < hi)
        if np.any(band):
            m = float(seegrad[band].mean())
            mittel.append(m)
            print("   %5d - %6d m: mittlerer Seegrad %.2f (n=%d)"
                  % (lo, min(hi, 99999), m, int(band.sum())))
    for i in range(1, len(mittel)):
        if mittel[i] < mittel[i - 1] - 0.15:
            fehler.append("Seegrad faellt statt zu waechst zwischen Band %d und %d "
                          "(%.2f -> %.2f)" % (i - 1, i, mittel[i - 1], mittel[i]))

    print("")
    print("2. Zieltiefe je Grad (nur Zellen ohne direkten Nachbargrad-Wechsel "
          "in der Naehe, um die Glaettung nicht zu bestrafen)")
    tiefe_soll = rw.TIEFE_JE_SEEGRAD
    for grad in range(5):
        maske_grad = (~maske) & (seegrad == grad)
        # innerer Kern: mindestens 3 Pixel von jeder Gradgrenze entfernt
        kern = ndimage.binary_erosion(maske_grad, iterations=3)
        if kern.sum() < 20:
            continue
        ist = float(seegrad_tiefe[kern].mean())
        soll = tiefe_soll.get(grad, rw.MEERESBODEN_M)
        print("   Grad %d: Zieltiefe ist %.1f m, soll %.1f m" % (grad, ist, soll))
        if abs(ist - soll) > 25.0:
            fehler.append("Grad %d: Zieltiefe %.1f m weicht mehr als 25 m von "
                          "%.1f m ab" % (grad, ist, soll))

    print("")
    print("3. Uferregion A stimmt ueberwiegend mit der tatsaechlich naechsten "
          "Landregion ueberein")
    region_map = felder["regionen"]
    # Unabhaengige Kreuzpruefung: je Regionsindex eine Distanzkarte zu deren
    # LAND-Flaeche, dann je Seepixel das Minimum ueber alle neun - das ist
    # bewusst NICHT derselbe Rechenweg wie seegliederung() (andere Punktmenge,
    # andere Methode), sonst wuerde die Pruefung nur sich selbst bestaetigen.
    entfernung_je_region = np.stack([
        ndimage.distance_transform_edt(~((region_map == i) & maske))
        for i in range(9)
    ], axis=0)
    naechste_region_unabhaengig = np.argmin(entfernung_je_region, axis=0)
    see_maske = ~maske
    uebereinstimmung = float((ufer_a[see_maske] == naechste_region_unabhaengig[see_maske]).mean())
    print("   Uebereinstimmung: %.1f %%" % (100 * uebereinstimmung))
    if uebereinstimmung < 0.6:
        fehler.append("Uferregion A stimmt nur zu %.1f %% mit der unabhaengig "
                      "gemessenen naechsten Landregion ueberein (< 60%%)"
                      % (100 * uebereinstimmung))

    print("")
    print("4. Zellgroesse See vs. Land (Vorbehalt: Kachelung sichtbar?)")
    see_flaeche_km2 = float(see_maske.sum()) * (mpp / 1000.0) ** 2
    land_flaeche_km2 = float(maske.sum()) * (mpp / 1000.0) ** 2
    kante_see_km = np.sqrt(see_flaeche_km2 / 400)
    kante_land_km = np.sqrt(land_flaeche_km2 / 200)
    print("   See:  %.1f km2 / 400 Zellen -> ~%.2f km Kante" % (see_flaeche_km2, kante_see_km))
    print("   Land: %.1f km2 / 200 Zellen -> ~%.2f km Kante" % (land_flaeche_km2, kante_land_km))
    verhaeltnis = kante_see_km / kante_land_km if kante_land_km > 0 else float("nan")
    if verhaeltnis > 2.0:
        fehler.append("See-Zellen sind %.1fx so gross wie Land-Zellen - "
                      "Kachelung im Bild wahrscheinlich, punktzahl_see erhoehen"
                      % verhaeltnis)

    print("")
    print("5-6. Seetyp-Regeln: Skerrheim steiler, Clonagh flacher bei Grad 1")
    regionsnamen = [r["name"] for _z, _s, r in rw.alle_regionen()]

    def _mittlere_tiefe(region_name, grad):
        idx = regionsnamen.index(region_name)
        m = see_maske & (ufer_a == idx) & (seegrad == grad)
        return float(seegrad_tiefe[m].mean()) if m.sum() > 20 else None

    fjord_grad1 = _mittlere_tiefe("Skerrheim", 1)
    huegel_grad1 = _mittlere_tiefe("Clonagh", 1)
    standard_grad1 = tiefe_soll[1]
    print("   Skerrheim Grad 1: %s m (Standard %.0f m)"
          % ("%.1f" % fjord_grad1 if fjord_grad1 is not None else "n/a", standard_grad1))
    print("   Clonagh Grad 1: %s m (Standard %.0f m)"
          % ("%.1f" % huegel_grad1 if huegel_grad1 is not None else "n/a", standard_grad1))
    if fjord_grad1 is not None and fjord_grad1 > standard_grad1 - 15.0:
        fehler.append("Skerrheim faellt bei Grad 1 nicht steiler ab als die "
                      "Standardtabelle (%.1f m, Standard %.1f m)" % (fjord_grad1, standard_grad1))
    if huegel_grad1 is not None and huegel_grad1 < standard_grad1 + 15.0:
        fehler.append("Clonagh bleibt bei Grad 1 nicht flacher als die "
                      "Standardtabelle (%.1f m, Standard %.1f m)" % (huegel_grad1, standard_grad1))

    print("")
    print("7. Seeeis: harte Regeln auf dieser einen Karte, Wahrscheinlichkeit "
          "ueber mehrere Karten gemittelt")
    see_eis = felder["see_eis"]
    taiga_idx = regionsnamen.index("Morobora")
    # Harte Regeln - muessen auf JEDER Karte gelten, kein Sample noetig.
    if bool(np.any(see_eis[maske])):
        fehler.append("Seeeis liegt auch auf Land")
    nicht_taiga_eis = int((see_eis & see_maske & (ufer_a != taiga_idx)).sum())
    if nicht_taiga_eis > 0:
        fehler.append("%d Seeeis-Pixel liegen an einer Nicht-Morobora-Kueste" % nicht_taiga_eis)
    eis_ab_grad_4 = int((see_eis & (seegrad >= 4)).sum())
    if eis_ab_grad_4 > 0:
        fehler.append("%d Eis-Pixel ab Grad 4 (offene See) - Seeweg waere dort "
                      "faelschlich blockiert" % eis_ab_grad_4)

    # WAHRSCHEINLICHKEIT je Grad - ueber mehrere Karten gepoolt, weil ein
    # einzelner Seed oft nur eine Handvoll Zellen je Grad stellt (siehe
    # Docstring). Gepoolte PIXEL-Zahl ist bei sieben Karten gross genug fuer
    # eine ruhige Schaetzung.
    soll_je_grad = rw.EISWAHRSCHEINLICHKEIT_JE_SEEGRAD
    treffer = {g: 0 for g in soll_je_grad}
    gesamt = {g: 0 for g in soll_je_grad}
    for zusatz_seed in (20260804, 12345, 4242, 777, 99, 555, 31415):
        Hs, felders = rw.weltfeld(256, zusatz_seed)
        maskes = Hs > 0.0
        sees = ~maskes
        seegrads = felders["seegrad"]
        ufer_as = felders["ufer_region_a"]
        eiss = felders["see_eis"]
        taiga_sees = sees & (ufer_as == taiga_idx)
        for g in soll_je_grad:
            m = taiga_sees & (seegrads == g)
            gesamt[g] += int(m.sum())
            treffer[g] += int(eiss[m].sum())

    for g in sorted(soll_je_grad):
        n = gesamt[g]
        soll = soll_je_grad[g]
        anteil = treffer[g] / n if n > 0 else float("nan")
        print("   Grad %d: %d/%d Pixel Eis ueber 7 Karten = %.0f %% (soll %.0f %%)"
              % (g, treffer[g], n, 100 * anteil if n else float("nan"), 100 * soll))
        if n >= 200 and abs(anteil - soll) > 0.15:
            fehler.append("Grad %d: gepoolter Eisanteil %.0f %% weicht mehr als "
                          "15 Punkte von der Vorgabe %.0f %% ab (n=%d)"
                          % (g, 100 * anteil, 100 * soll, n))

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
