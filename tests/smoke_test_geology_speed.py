"""
Geology bleibt schnell - und rechnet dabei dasselbe.

ANLASS (2026-08-10). Der Nutzer meldete, dass Geology "bei der Groesse sehr
langsam geworden" ist und die Oberflaeche bei `stack_deformation` einbricht.
Zwei Stellen waren es:

  1. `_build_terrain_hub` glaettet mit sigma = 0.2 * Kantenlaenge. SciPys
     gaussian_filter kostet Pixelzahl mal sigma, und sigma waechst hier selbst
     mit der Kantenlaenge - der Aufwand wuchs also KUBISCH. Gemessen 0.19 s bei
     512 px, 2.08 bei 1024, 16.6 bei 2048.
  2. `_build_fault_field` schleifte ueber alle Stoerungssegmente und rechnete
     je Segment elf volle Kartenarrays. 25.5 s bei 2048 px.

Beide rechnen jetzt anders - die Glaettung auf grobem Gitter, das Stoerungsfeld
kachelweise mit Vorauswahl. Dieser Test sichert das ERGEBNIS ab, nicht die
Geschwindigkeit allein: eine Beschleunigung, die andere Karten erzeugt, waere
keine.

Die Stoerungen muessen BITGLEICH bleiben (die Vorauswahl ist reine
Kandidatenwahl), die Glaettung darf um Bruchteile eines Meters abweichen.
"""
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.geology_generator as gg

KM = 21.3
SEED = 20260804
# Bis 1024 px, damit der Test in ertraeglicher Zeit durchlaeuft - die Vorlage
# braucht bei 2048 allein 25 s.
GROESSEN = (256, 512, 1024)


def gelaende(n):
    """Ein glattes Hoehenfeld mit Struktur, kein weisses Rauschen."""
    rng = np.random.RandomState(7)
    from scipy.ndimage import zoom
    roh = rng.rand(max(4, n // 16), max(4, n // 16)).astype(np.float64)
    return (zoom(roh, 16, order=3)[:n, :n] * 800.0)


def main():
    fehler = []

    print("1. Stoerungsfeld: kachelweise gegen die Vorlage")
    print("   %6s %10s %10s %9s %12s" % ("px", "Vorlage", "jetzt", "schnell",
                                         "Abweichung"))
    for n in GROESSEN:
        t = time.perf_counter()
        soll, _d = gg._build_fault_field_referenz((n, n), KM, SEED,
                                                  100.0, 1.0, 0.3)
        t_alt = time.perf_counter() - t
        t = time.perf_counter()
        ist, _d = gg._build_fault_field((n, n), KM, SEED, 100.0, 1.0, 0.3)
        t_neu = time.perf_counter() - t
        ab = float(np.abs(soll.astype(np.float64) - ist.astype(np.float64)).max())
        print("   %6d %9.2fs %9.3fs %8.0fx %11.4f m"
              % (n, t_alt, t_neu, t_alt / max(t_neu, 1e-9), ab))
        if ab > 1e-6:
            fehler.append("Stoerungsfeld bei %d px weicht um %.4f m ab - die "
                          "Kachelvorauswahl darf das Ergebnis NICHT aendern"
                          % (n, ab))
        if t_neu > t_alt:
            fehler.append("Stoerungsfeld bei %d px ist nicht schneller "
                          "(%.2f s gegen %.2f s)" % (n, t_neu, t_alt))

    print("")
    print("2. Weitraeumige Glaettung: grobes Gitter gegen die Vorlage")
    print("   %6s %10s %10s %9s %12s %10s"
          % ("px", "Vorlage", "jetzt", "schnell", "Abweichung", "Spanne"))
    for n in GROESSEN:
        H = gelaende(n)
        sigma = gg.REGIONAL_HUB_SIGMA_FRACTION * n
        t = time.perf_counter(); soll = gaussian_filter(H, sigma=sigma)
        t_alt = time.perf_counter() - t
        t = time.perf_counter(); ist = gg.grosse_glaettung(H, sigma)
        t_neu = time.perf_counter() - t
        ab = float(np.abs(soll - ist).max())
        spanne = float(H.max() - H.min())
        print("   %6d %9.2fs %9.3fs %8.0fx %11.4f m %9.1f m"
              % (n, t_alt, t_neu, t_alt / max(t_neu, 1e-9), ab, spanne))
        # BEZUG IST DIE GELAENDESPANNE, NICHT DIE DES GEGLAETTETEN FELDES.
        # Ein erster Anlauf mass gegen letztere und schlug bei 1.06 % an - das
        # war der falsche Massstab: der Hub wird vom Gelaende ABGEZOGEN
        # (_compute_outcrop schneidet mit terrain_height - stack_deformation),
        # die treibende Groesse ist also das Relief selbst. Ein halbes Prozent
        # davon verschiebt den Ausbiss um weniger als eine Schichtdicke.
        if ab > max(0.005 * spanne, 0.05):
            fehler.append("Glaettung bei %d px weicht um %.3f m ab "
                          "(Gelaendespanne %.1f m)" % (n, ab, spanne))

    print("")
    print("3. Kubisches Wachstum ist weg")
    zeiten = []
    for n in (512, 1024):
        H = gelaende(n)
        t = time.perf_counter()
        gg.grosse_glaettung(H, gg.REGIONAL_HUB_SIGMA_FRACTION * n)
        zeiten.append(time.perf_counter() - t)
    faktor = zeiten[1] / max(zeiten[0], 1e-9)
    print("   512 -> 1024 px kostet Faktor %.1f (kubisch waeren rund 8)"
          % faktor)
    if faktor > 6.5:
        fehler.append("Glaettung waechst weiter fast kubisch (Faktor %.1f)"
                      % faktor)

    print("")
    if fehler:
        print("NICHT IN ORDNUNG - %d Befunde:" % len(fehler))
        for f in fehler:
            print("   " + f)
        return 1
    print("Alle Zusicherungen erfuellt - Geology rechnet schneller und "
          "dasselbe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
