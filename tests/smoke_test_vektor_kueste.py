"""
Path: tests/smoke_test_vektor_kueste.py

Prueft `core/vektor_kueste.py` - die Kueste als Vektor, eine Hoehenfunktion
mit zwei Abtastern (Raster -> Heightmap B, freie Punkte -> Mesh).

WAS HIER GEPRUEFT WIRD, UND WARUM JEDES DAVON

  1. EINE FUNKTION      Beide Abtaster mit demselben `mindest_skala_m` auf
                        denselben Koordinaten muessen BITGLEICH sein. Das ist
                        die Kernzusicherung der ganzen Bauart: gaebe es zwei
                        Implementierungen, liefen Mesh und Heightmap
                        auseinander (02_INVARIANTEN.md 5).
  2. RASTERFREIHEIT     An freien Punkten ZWISCHEN den Pixeln muss messbar
                        mehr Hoehendetail stehen als eine bilineare
                        Interpolation der Rasterkarte hergibt - sonst waere
                        der ganze Vektorweg wirkungslos.
  3. AUFLOESUNG         Die Vektorbeschreibung (Linienlaenge, Saatpunktzahl in
                        METERN) darf nicht an der Pixelzahl haengen
                        (90_MESSPROTOKOLLE.md §10).
  4. DETERMINISMUS      Gleicher Seed -> bitgleiches Ergebnis.
  5. KUESTENLINIE       Die 0-Linie darf sich durch die Umformung NICHT
                        verschieben: Land bleibt Land, See bleibt See. Sonst
                        entwertet jede Formaenderung die Regionseichung
                        (dieselbe Bedingung wie bei `kuestenform` in
                        terrain_weltkarte.py).
  6. RANDFAELLE         Karte ohne Kueste / ganz unter Wasser: Ergebnis statt
                        Absturz.

ECHTE KARTENGROESSEN (256/384/512), nicht 2^n+1 - die Lektion aus 6.16, wo
zehn gruene Tests eine Funktion prueften, die im Programm nie lief.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_vektor_kueste.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.terrain_weltkarte import weltfeld
from core.vektor_kueste import (MESH_MINDEST_SKALA_M, VektorKueste,
                                als_raster, an_punkten)

GROESSEN = (256, 384, 512)
SEED = 20260804

_WELT_CACHE = {}


def welt(size, seed=SEED):
    if (size, seed) not in _WELT_CACHE:
        H, felder = weltfeld(size, seed)
        _WELT_CACHE[(size, seed)] = (np.asarray(H, dtype=np.float64), felder)
    return _WELT_CACHE[(size, seed)]


def _vk(size, seed=SEED):
    H, felder = welt(size, seed)
    return VektorKueste(H, felder["regionen"], seed), H


# --------------------------------------------------------------------- #

def eine_funktion():
    """1. Beide Abtaster, gleicher Parameter, gleiche Punkte -> bitgleich."""
    fehler = []
    for size in GROESSEN:
        vk, _H = _vk(size)
        gy, gx = np.mgrid[0:size, 0:size]
        a = als_raster(vk)
        b = an_punkten(vk, gx.astype(float), gy.astype(float),
                       mindest_skala_m=2.0 * vk.mpp)
        gleich = np.array_equal(a, b)
        print(f"[{'OK' if gleich else 'FEHLER'}] {size} px: Raster- und "
              f"Punktabtaster sind bitgleich")
        if not gleich:
            fehler.append(f"{size} px: Abtaster weichen ab, max "
                          f"{np.abs(a - b).max():.3g} m - es sind ZWEI "
                          f"Funktionen geworden")
    return fehler


def rasterfreiheit():
    """2. An freien Punkten steht Detail, das das Raster nicht hat."""
    fehler = []
    rng = np.random.default_rng(0)
    for size in GROESSEN:
        vk, _H = _vk(size)
        raster = als_raster(vk)
        fx = rng.uniform(1, size - 2, 20000)
        fy = rng.uniform(1, size - 2, 20000)
        frei = an_punkten(vk, fx, fy)

        x0 = np.floor(fx).astype(int)
        y0 = np.floor(fy).astype(int)
        tx, ty = fx - x0, fy - y0
        bilinear = (raster[y0, x0] * (1 - tx) * (1 - ty)
                    + raster[y0, x0 + 1] * tx * (1 - ty)
                    + raster[y0 + 1, x0] * (1 - tx) * ty
                    + raster[y0 + 1, x0 + 1] * tx * ty)
        abweichung = np.abs(frei - bilinear)
        p99 = float(np.percentile(abweichung, 99))
        genug = p99 > 1.0
        print(f"[{'OK' if genug else 'FEHLER'}] {size} px: freie Punkte "
              f"tragen p99 {p99:.1f} m mehr Detail als das Raster "
              f"(max {abweichung.max():.0f} m)")
        if not genug:
            fehler.append(f"{size} px: freie Punkte unterscheiden sich kaum "
                          f"vom Raster (p99 {p99:.2f} m) - der Vektorweg "
                          f"bringt nichts")
    return fehler


def aufloesungsunabhaengig():
    """3. Die Vektorbeschreibung haengt nicht an der Pixelzahl."""
    fehler = []
    laengen, saaten = {}, {}
    for size in GROESSEN:
        vk, _H = _vk(size)
        # ECHTE Bogenlaenge der Kontur in Metern (vk.kuesten_laenge_m), nicht
        # `Punktzahl * mpp`: der Abtastabstand ist bei grober Aufloesung auf
        # 1 px geklemmt, das Produkt maesse dort die Klemme statt der Kueste.
        laengen[size] = vk.kuesten_laenge_m
        saaten[size] = len(vk.saat_xy)

    werte = np.array(list(laengen.values()), dtype=float)
    streuung = float(werte.std() / max(werte.mean(), 1e-9))
    ok = streuung < 0.35
    print(f"[{'OK' if ok else 'FEHLER'}] Kuestenlaenge ueber die Groessen "
          f"stabil: {[f'{v/1000:.1f} km' for v in werte]} "
          f"(Streuung {streuung:.0%})")
    if not ok:
        fehler.append(f"Kuestenlaenge haengt an der Aufloesung "
                      f"(Streuung {streuung:.0%})")

    s = np.array(list(saaten.values()), dtype=float)
    streuung_s = float(s.std() / max(s.mean(), 1e-9))
    ok_s = streuung_s < 0.35
    print(f"[{'OK' if ok_s else 'FEHLER'}] Saatpunktzahl stabil: "
          f"{list(saaten.values())} (Streuung {streuung_s:.0%})")
    if not ok_s:
        fehler.append(f"Saatpunktzahl haengt an der Aufloesung "
                      f"(Streuung {streuung_s:.0%})")
    return fehler


def determinismus():
    """4. Gleicher Seed -> bitgleich."""
    fehler = []
    H, felder = welt(384)
    a = als_raster(VektorKueste(H, felder["regionen"], SEED))
    b = als_raster(VektorKueste(H, felder["regionen"], SEED))
    gleich = np.array_equal(a, b)
    print(f"[{'OK' if gleich else 'FEHLER'}] zweimal gebaut -> bitgleich")
    if not gleich:
        fehler.append("gleicher Seed ergibt verschiedene Kuesten")

    c = als_raster(VektorKueste(H, felder["regionen"], SEED + 1))
    anders = not np.array_equal(a, c)
    print(f"[{'OK' if anders else 'FEHLER'}] anderer Seed -> andere Kueste")
    if not anders:
        fehler.append("Seed wirkt nicht")
    return fehler


def kuestenlinie_bleibt():
    """
    5. Land bleibt Land, See bleibt See.

    Die Umformung darf die 0-Linie nicht verschieben - sonst aendert sich der
    Wasseranteil jeder Region und die gesamte Regionseichung waere entwertet
    (dieselbe Zusicherung, die `kuestenform` in terrain_weltkarte.py gibt).
    """
    fehler = []
    for size in GROESSEN:
        vk, H = _vk(size)
        B = als_raster(vk)
        gekippt = int(((H > 0) != (B > 0)).sum())
        anteil = gekippt / H.size
        ok = anteil < 0.001
        print(f"[{'OK' if ok else 'FEHLER'}] {size} px: {gekippt} Pixel "
              f"wechseln die Seite ({anteil:.3%})")
        if not ok:
            fehler.append(f"{size} px: {anteil:.2%} der Pixel wechseln "
                          f"Land/See - die Kuestenlinie verschiebt sich")
    return fehler


def randfaelle():
    """6. Karte ohne Kueste und Karte ganz unter Wasser."""
    fehler = []
    size = 128
    regionen = np.zeros((size, size), dtype=np.int16)

    for name, H in (("nur Land", np.full((size, size), 500.0)),
                    ("nur See", np.full((size, size), -200.0))):
        try:
            vk = VektorKueste(H, regionen, SEED)
            B = als_raster(vk)
            gut = B.shape == H.shape and np.all(np.isfinite(B))
            print(f"[{'OK' if gut else 'FEHLER'}] {name}: Ergebnis statt "
                  f"Absturz (unveraendert: {np.array_equal(B, H)})")
            if not gut:
                fehler.append(f"{name}: unbrauchbares Ergebnis")
        except Exception as e:                              # noqa: BLE001
            print(f"[FEHLER] {name}: Absturz - {e}")
            fehler.append(f"{name}: Absturz - {e}")
    return fehler


# --------------------------------------------------------------------- #

def lauf():
    gruppen = [
        ("eine_funktion", eine_funktion),
        ("rasterfreiheit", rasterfreiheit),
        ("aufloesungsunabhaengig", aufloesungsunabhaengig),
        ("determinismus", determinismus),
        ("kuestenlinie_bleibt", kuestenlinie_bleibt),
        ("randfaelle", randfaelle),
    ]
    ergebnis = {}
    alle_fehler = []
    for name, fn in gruppen:
        print(f"\n--- {name} ---")
        fehler = fn()
        ergebnis[name] = not fehler
        alle_fehler.extend(fehler)

    print("\n=== SUMMARY ===")
    for name, ok in ergebnis.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if alle_fehler:
        print(f"\nNICHT IN ORDNUNG - {len(alle_fehler)} Befunde:")
        for f in alle_fehler:
            print(f"   {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(lauf())
