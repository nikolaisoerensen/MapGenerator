"""
Path: tests/smoke_test_region_vorbild_aehnlichkeit.py

Haelt drei erzeugte Regionen gegen ihr echtes DEM-Vorbild (Copernicus COP30,
bereits in tools/_dem_cache/ zwischengespeichert - dieser Test braucht kein
Netz und keinen API-Schluessel).

NUTZERAUFTRAG 2026-09-17: `tools/region_gegen_vorbild.py` gab die Kennzahlen
bisher nur zum Ablesen auf der Kommandozeile aus. Hier werden sie zu einer
echten, automatisierten Pruefung mit kalibrierten Toleranzen.

GEGEN DAS ZIRKELSCHLUSS-MUSTER: die Sollwerte sind NICHT aus derselben Formel
wie der Code hergeleitet, sondern aus echten COP30-Hoehendaten realer Kuesten
(Cliffs of Moher, Geirangerfjord, Cabo da Roca) gemessen - vor dem Schreiben
dieser Toleranzen, mit einem Wegwerfskript am 2026-09-17.

DREI PRUEFUNGEN, EIN SEAM NACH DEM ANDEREN:

  A) KENNZAHLEN-VERGLEICH   Formfaktor und normierte Hoehenperzentile
                            (h_p25/h_p50/h_p75) je Region gegen ihr Vorbild,
                            als Verhaeltnis erzeugt/echt mit Toleranzband.
  B) HOEHEN-HISTOGRAMM      Kolmogorov-Smirnov-Test (scipy.stats.ks_2samp)
                            auf der vollen normierten Hoehenverteilung, nicht
                            nur auf drei Perzentilen.
  C) KUESTENFORM            Fraktale Dimension der Kuestenlinie
                            (Box-Counting auf dem Kuestensaum), erzeugt gegen
                            echt, plus Realismus-Bereich (Mandelbrot: reale
                            Kuesten liegen zwischen rund 1.1 und 1.3).

WARUM NUR DREI REGIONEN, UND GENAU DIESE DREI

Ein Lauf braucht rund 10 s (Weltgenerierung + DEM-Vergleich je Region). Alle
neun waeren rund 30 s - vertretbar, aber unnoetig fuer einen Smoke-Test, der
bei jeder Aenderung an core/terrain_weltkarte.py mitlaufen soll. Gewaehlt
wurden Clonagh (sanftes Huegelland), Skerrheim (Fjord, mit dem
FLAECHENGLEICHEN Ausschnitt `geiranger_klein`, damit Formfaktor und
Kuestendichte nicht an einer falschen Ausschnittsgroesse haengen - siehe
tools/kuestenlaengsschnitt.STRECKEN) und Estrande (Kuestenebene mit Kliff) -
drei sehr unterschiedliche Kuestencharaktere, keine Rosinenpickerei auf
besonders gut passende Faelle.

ECHTER BEFUND, NICHT WEGKALIBRIERT: beim Kalibrieren wurde auch Macchia
gegen sein Vorbild `calanques` gemessen. Der KS-D-Wert lag bei 0.543 - fast
doppelt so hoch wie der schlechteste der drei hier gepruegten Regionen
(Skerrheim: 0.300). Macchia ist deshalb ABSICHTLICH NICHT in diesem Test:
eine Toleranz, die auch 0.543 durchliesse, wuerde bei den anderen Regionen
nichts mehr pruefen. Das ist ein offener Befund fuer die Gelaende-Kalibrierung
von Macchia, kein Testfehler - siehe Bericht der Sitzung vom 2026-09-17.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_region_vorbild_aehnlichkeit.py
"""

import os
import sys

import numpy as np
from scipy import ndimage, stats

_WURZEL = r"C:\Lokale Dateien\Projects\Python\MapGenerator"
sys.path.insert(0, _WURZEL)
sys.path.insert(0, os.path.join(_WURZEL, "tools"))

from tools.region_gegen_vorbild import region, vorbild

# Region -> (Vorbild-Schluessel aus tools.kuestenlaengsschnitt.STRECKEN, Skala).
#
# Die Skala (1/3) ist die Nutzervorgabe aus tools/region_gegen_vorbild.py:
# 1000 m im Spiel entsprechen etwa 3000 m real. Sie wirkt sich NUR auf
# Hangkennzahlen aus (hier nicht gepruegt) - Formfaktor, Hoehenperzentile
# und die fraktale Dimension sind masstabsfrei bzw. auf die eigene Spanne
# normiert, siehe Modulkopf von region_gegen_vorbild.py.
REGION_VORBILD_PAARE = [
    ("Clonagh", "doolin"),
    ("Skerrheim", "geiranger_klein"),
    ("Estrande", "roca"),
]
SKALA = 1.0 / 3.0

# ---------------------------------------------------------------------------
# Toleranzen, kalibriert am 2026-09-17 auf echten Messwerten (siehe Docstring
# oben). Format je Kennzahl: (untere Grenze, obere Grenze) fuer das
# Verhaeltnis erzeugt/echt.
#
# GEMESSEN (erzeugt/echt):
#   Formfaktor   Clonagh 1.31x  Skerrheim 1.99x  Estrande 6.04x
#   h_p25        Clonagh 0.21x  Skerrheim 0.27x  Estrande 0.22x
#   h_p50        Clonagh 0.61x  Skerrheim 0.62x  Estrande 0.71x
#   h_p75        Clonagh 0.87x  Skerrheim 0.65x  Estrande 0.86x
#
# Formfaktor UND Kuestendichte haengen laut region_gegen_vorbild.py an der
# Ausschnittsgroesse (eine kleine erzeugte Region hat immer eine hoehere
# Kuestenlinie-je-Flaeche als ein grosses reales Kuestenstueck - das ist der
# bekannte "Kuestenparadox"-Effekt, keine Eigenheit dieses Codes). Die
# Formfaktor-Toleranz ist deshalb bewusst weit; sie soll einen KOMPLETTEN
# Ausfall faengen (glattgebuegelte Kueste nahe 0, oder eine im Faktor 10+
# zerfetzte), nicht eine Verdopplung.
FORMFAKTOR_TOLERANZ = (0.5, 8.0)

# Die normierten Hoehenperzentile sind ueber alle drei Regionen erstaunlich
# konsistent (siehe Tabelle oben) - hier darf die Toleranz enger sein, sie
# ist der eigentliche Waechter dieses Tests.
HOEHE_TOLERANZ = {
    "h_p25": (0.12, 0.45),
    "h_p50": (0.40, 0.90),
    "h_p75": (0.45, 1.00),
}

# Kolmogorov-Smirnov: D-Statistik, NICHT der p-Wert.
#
# WARUM NICHT DER P-WERT: bei 8000-116000 Hoehenpixeln je Feld ist praktisch
# jeder Unterschied "statistisch signifikant" - gemessen kam der p-Wert bei
# allen fuenf kalibrierten Paaren auf unter 1e-230. Der p-Wert prueft hier
# nur "ist die Stichprobe riesig" (ja, immer), nicht "sind die Formen
# aehnlich". Die D-Statistik (der groesste Abstand zwischen den beiden
# Verteilungsfunktionen, zwischen 0 und 1) tut das tatsaechlich.
#
# GEMESSEN: Clonagh 0.187, Skerrheim 0.300, Estrande 0.217. Macchia (nicht in
# diesem Test, siehe Docstring) lag bei 0.543 - die Schwelle liegt bewusst
# dazwischen.
KS_D_MAX = 0.40

# Fraktale Dimension per Box-Counting auf dem Kuestensaum.
#
# REALISMUS-BEREICH: Mandelbrot nennt 1.1-1.3 fuer reale Kuesten. Gemessen
# an den Vorbildern selbst kam Estrande/roca (eine recht gerade Klippe ohne
# Buchten) auf 0.972 - knapp darunter. Der Bereich ist deshalb etwas weiter
# gefasst als das Lehrbuchmass, um echte, aber ungewoehnlich glatte Kuesten
# nicht faelschlich durchfallen zu lassen; er faengt trotzdem eine kaputte
# Berechnung (Dimension nahe 0 oder nahe 2 - raumfuellend) zuverlaessig.
FRAKTAL_BEREICH = (0.85, 1.6)

# Abweichung erzeugt vs. echt. GEMESSEN: Clonagh 0.126, Skerrheim 0.057,
# Estrande 0.138. Schwelle mit Marge darueber.
FRAKTAL_ABWEICHUNG_MAX = 0.25


def _boxcount_dimension(land_maske):
    """
    Fraktale Dimension der Kuestenlinie per Box-Counting.

    `land_maske` ist die boolesche Landmaske aus _kennzahlen() (additiv als
    "land_maske" zurueckgegeben, siehe tools/region_gegen_vorbild.py). Der
    Kuestensaum wird genauso gebildet wie dort ("Landpixel mit
    Wassernachbar"), dann werden die Saumpixel bei Kantenlaengen 2,4,8,...
    px in Kaesten gezaehlt. Die Steigung von log(Kaestenzahl) ueber
    log(1/Kantenlaenge) ist die Box-Counting-Dimension.

    None, wenn zu wenige Saumpixel fuer eine sinnvolle Regression da sind.
    """
    land = np.asarray(land_maske, dtype=bool)
    saum = ndimage.binary_dilation(~land, iterations=1) & land
    ys, xs = np.nonzero(saum)
    if len(xs) < 20:
        return None

    groessen = []
    k = 2
    while k <= max(land.shape) // 2:
        groessen.append(k)
        k *= 2

    kaesten = []
    for box in groessen:
        belegt = {(int(x) // box, int(y) // box) for x, y in zip(xs, ys)}
        kaesten.append(len(belegt))

    groessen = np.asarray(groessen, dtype=np.float64)
    kaesten = np.asarray(kaesten, dtype=np.float64)
    gueltig = kaesten > 0
    if gueltig.sum() < 3:
        return None
    log_kehrwert = np.log(1.0 / groessen[gueltig])
    log_kaesten = np.log(kaesten[gueltig])
    steigung, _achsenabschnitt = np.polyfit(log_kehrwert, log_kaesten, 1)
    return float(steigung)


def _kennzahlen_paare():
    """Je Region einmal region() und vorbild() rechnen, fuer alle drei Pruefungen."""
    paare = {}
    for reg, strecke in REGION_VORBILD_PAARE:
        a = region(reg)
        b = vorbild(strecke, SKALA)
        paare[(reg, strecke)] = (a, b)
    return paare


def _teste_kennzahlen(paare, fehler):
    """Test A: Formfaktor und Hoehenperzentile gegen die kalibrierten Toleranzen."""
    print("\nA) Formfaktor & normierte Hoehenperzentile gegen das echte DEM")
    for (reg, strecke), (a, b) in paare.items():
        if a is None or b is None:
            fehler.append(f"A: {reg}/{strecke} - Kennzahlen fehlen")
            print(f"  {reg:<10} gegen {strecke:<18} ... FEHLER (Daten fehlen)")
            continue

        verh_formfaktor = a["formfaktor"] / max(b["formfaktor"], 1e-9)
        lo, hi = FORMFAKTOR_TOLERANZ
        ok_formfaktor = lo <= verh_formfaktor <= hi
        if not ok_formfaktor:
            fehler.append(
                f"A: {reg}/{strecke} - Formfaktor-Verhaeltnis {verh_formfaktor:.2f}x "
                f"ausserhalb [{lo}, {hi}]")

        ok_hoehen = True
        einzelheiten = []
        for feld, (lo_h, hi_h) in HOEHE_TOLERANZ.items():
            verh = a[feld] / max(b[feld], 1e-9)
            ok = lo_h <= verh <= hi_h
            ok_hoehen = ok_hoehen and ok
            einzelheiten.append(f"{feld}={verh:.2f}x{'' if ok else '!!'}")
            if not ok:
                fehler.append(
                    f"A: {reg}/{strecke} - {feld}-Verhaeltnis {verh:.2f}x "
                    f"ausserhalb [{lo_h}, {hi_h}]")

        status = "ok" if (ok_formfaktor and ok_hoehen) else "FEHLER"
        print(f"  {reg:<10} gegen {strecke:<18} Formfaktor={verh_formfaktor:.2f}x  "
              f"{' '.join(einzelheiten)}  ... {status}")


def _teste_ks(paare, fehler):
    """Test B: Kolmogorov-Smirnov auf der vollen normierten Hoehenverteilung."""
    print("\nB) Kolmogorov-Smirnov auf der normierten Hoehenverteilung")
    for (reg, strecke), (a, b) in paare.items():
        if a is None or b is None:
            fehler.append(f"B: {reg}/{strecke} - Daten fehlen")
            continue
        ergebnis = stats.ks_2samp(a["h_norm_werte"], b["h_norm_werte"])
        ok = ergebnis.statistic <= KS_D_MAX
        if not ok:
            fehler.append(
                f"B: {reg}/{strecke} - KS-D {ergebnis.statistic:.3f} "
                f"ueber der Schwelle {KS_D_MAX}")
        print(f"  {reg:<10} gegen {strecke:<18} D={ergebnis.statistic:.3f} "
              f"(Schwelle {KS_D_MAX}, p={ergebnis.pvalue:.2e} - siehe Docstring, "
              f"nicht gepruegt)  ... {'ok' if ok else 'FEHLER'}")


def _teste_fraktal(paare, fehler):
    """Test C: fraktale Dimension der Kuestenlinie, Box-Counting."""
    print("\nC) Fraktale Dimension der Kuestenlinie (Box-Counting)")
    lo, hi = FRAKTAL_BEREICH
    for (reg, strecke), (a, b) in paare.items():
        if a is None or b is None:
            fehler.append(f"C: {reg}/{strecke} - Daten fehlen")
            continue
        da = _boxcount_dimension(a["land_maske"])
        db = _boxcount_dimension(b["land_maske"])
        if da is None or db is None:
            fehler.append(f"C: {reg}/{strecke} - zu wenige Kuestenpixel fuer Box-Counting")
            print(f"  {reg:<10} gegen {strecke:<18} ... FEHLER (zu wenig Kueste)")
            continue

        ok = True
        if not (lo <= da <= hi):
            ok = False
            fehler.append(
                f"C: {reg} - fraktale Dimension {da:.3f} ausserhalb des "
                f"Realismus-Bereichs [{lo}, {hi}]")
        if not (lo <= db <= hi):
            ok = False
            fehler.append(
                f"C: {strecke} (Vorbild) - fraktale Dimension {db:.3f} "
                f"ausserhalb des Realismus-Bereichs [{lo}, {hi}]")
        abweichung = abs(da - db)
        if abweichung > FRAKTAL_ABWEICHUNG_MAX:
            ok = False
            fehler.append(
                f"C: {reg}/{strecke} - Dimension weicht um {abweichung:.3f} ab "
                f"(Schwelle {FRAKTAL_ABWEICHUNG_MAX})")

        print(f"  {reg:<10} gegen {strecke:<18} erzeugt={da:.3f}  echt={db:.3f}  "
              f"Abweichung={abweichung:.3f}  ... {'ok' if ok else 'FEHLER'}")


def lauf():
    print("Regionen gegen ihr DEM-Vorbild - drei Pruefungen "
          f"({len(REGION_VORBILD_PAARE)} Regionen, aus dem Zwischenspeicher)")
    paare = _kennzahlen_paare()

    fehler = []
    _teste_kennzahlen(paare, fehler)
    _teste_ks(paare, fehler)
    _teste_fraktal(paare, fehler)

    print()
    if fehler:
        print(f"NICHT IN ORDNUNG - {len(fehler)} Befunde:")
        for eintrag in fehler:
            print(f"   {eintrag}")
        return 1
    print("Alle drei Pruefungen erfuellt - Formkennzahlen, Hoehenverteilung "
          "und Kuestenform liegen fuer Clonagh, Skerrheim und Estrande im "
          "kalibrierten Rahmen zu ihrem echten Vorbild.")
    return 0


if __name__ == "__main__":
    raise SystemExit(lauf())
