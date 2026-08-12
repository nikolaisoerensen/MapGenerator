"""
Das Supersampling darf keine Muster erfinden.

ANLASS (2026-08-10). Der Nutzer schickte ein Bild der Biomkarte: quer ueber die
ganze Flaeche, auch ueber offenes Meer, liefen regelmaessige diagonale Baender
in Fluss- und Bachfarben.

URSACHE. Der Teilpixel-Zufallswert lautete

    sub_seed = (seed + 54321 + x * 4 + y * 4 + i * 3571) % 1000

`x * 4 + y * 4` ist `4·(x+y)` - der Wert hing NUR von der Summe der Koordinaten
ab. Jede Diagonale bekam denselben, und wegen `% 1000` wiederholte sich das
alle 250 Diagonalen. Wo eine Super-Biom-Wahrscheinlichkeit ueber null lag,
kippte deshalb ein ganzer Streifen statt eines gestreuten Musters. Derselbe
Ausdruck stand im Shader (shaders/biome/supersampling.comp).

DIESER TEST prueft nicht "sieht gut aus", sondern die EIGENSCHAFT, die fehlte:
der Zufallswert darf mit der Koordinatensumme nicht zusammenhaengen. Genau das
laesst sich messen, und genau das war kaputt.
"""
import os
import sys
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.biome_generator import SupersamplingManager

N = 128
SEED = 20260804


def main():
    fehler = []
    verwalter = SupersamplingManager(biome_seed=SEED, supersampling_quality=1.0)
    yy, xx = np.mgrid[0:N, 0:N]

    print("1. Haengt der Teilpixelwert noch an der Koordinatensumme?")
    for i in range(4):
        w = verwalter._sub_zufall(SEED + 54321, xx, yy, i)
        # Auf jeder Diagonalen x+y = const muesste bei der alten Formel EIN
        # einziger Wert stehen. Gemessen wird die mittlere Streuung innerhalb
        # der Diagonalen gegen die Streuung insgesamt.
        summe = (xx + yy).ravel()
        werte = w.ravel()
        streuungen = []
        for s in range(0, 2 * N - 1, 7):
            gruppe = werte[summe == s]
            if gruppe.size > 4:
                streuungen.append(gruppe.std())
        innen = float(np.mean(streuungen))
        gesamt = float(werte.std())
        print("   Ecke %d: Streuung auf der Diagonalen %.4f, insgesamt %.4f"
              % (i, innen, gesamt))
        if innen < 0.5 * gesamt:
            fehler.append("Ecke %d: der Wert ist auf der Diagonalen fast "
                          "konstant (%.4f gegen %.4f) - das erzeugt Baender"
                          % (i, innen, gesamt))

    print("")
    print("2. Streut er ueberhaupt gleichmaessig?")
    w = verwalter._sub_zufall(SEED + 54321, xx, yy, 0)
    faecher, _ = np.histogram(w, bins=10, range=(0.0, 1.0))
    erwartet = w.size / 10.0
    schiefe = float(np.abs(faecher - erwartet).max() / erwartet)
    print("   groesste Abweichung eines Zehntels: %.1f %%" % (100.0 * schiefe))
    if schiefe > 0.15:
        fehler.append("Verteilung ungleichmaessig: %.1f %% Abweichung"
                      % (100.0 * schiefe))

    print("")
    print("3. Ein Durchlauf auf einer Karte, die UEBERALL etwas Ufer hat")
    biome = np.full((N, N), 5, dtype=np.uint8)
    # 12 % Wahrscheinlichkeit flaechendeckend: bei der alten Formel wurden
    # daraus geschlossene Diagonalen, richtig gestreut sind es Sprenkel.
    karten = {name: np.full((N, N), 0.12, dtype=np.float32)
              for name in ("cliff", "beach", "lake_edge", "river_bank",
                           "snow_level", "alpine_level")}
    t = time.perf_counter()
    gross = verwalter._apply_supersampling_cpu(biome, karten)
    dauer = time.perf_counter() - t
    print("   %s in %.3f s" % (str(gross.shape), dauer))
    if gross.shape != (2 * N, 2 * N):
        fehler.append("falsche Kantenlaenge: %s" % (gross.shape,))

    # Ein Band ist daran zu erkennen, dass in Richtung der Diagonalen kaum
    # gewechselt wird, quer dazu aber sehr wohl.
    anders = gross != biome.repeat(2, 0).repeat(2, 1)
    a = anders[:-1, :-1]
    laengs = float((a != anders[1:, 1:]).mean())    # entlang x+y = const
    quer = float((a != anders[1:, :-1]).mean())     # quer dazu
    print("   Wechsel entlang der Diagonalen %.3f, quer dazu %.3f"
          % (laengs, quer))
    if laengs < 0.5 * quer:
        fehler.append("entlang der Diagonalen wechselt es kaum (%.3f gegen "
                      "%.3f) - die Baender sind zurueck" % (laengs, quer))

    anteil = float(anders.mean())
    print("   %.1f %% der Teilpixel wurden ersetzt" % (100.0 * anteil))
    if not 0.05 < anteil < 0.95:
        fehler.append("Ersetzungsanteil %.2f unplausibel" % anteil)

    print("")
    if fehler:
        print("NICHT IN ORDNUNG - %d Befunde:" % len(fehler))
        for f in fehler:
            print("   " + f)
        return 1
    print("Alle Zusicherungen erfuellt - das Supersampling streut, "
          "statt Baender zu ziehen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
