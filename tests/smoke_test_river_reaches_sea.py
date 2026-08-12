"""
Jeder Flusslauf muss im Meer enden - keine Ringe.

ANLASS (2026-08-10). Der Nutzer: "die fliessen oft nicht zum meer." Gemessen
bei 384 px, drei Seeds, VOR der Korrektur: 29 bis 47 % aller Netzknoten hingen
in einem RING - ihre Elternkette lief im Kreis statt im Meer zu enden.

URSACHE. `baue_stufe()` (core/terrain_weltfluesse.py) baut den Baum eigentlich
schon meerverwurzelt: eine Ueberquelle haengt mit Kosten ~0 an allen Knoten
unter Wasser, Dijkstra von dort gibt jedem erreichbaren Knoten eine
Elternkette zum Meer - dieselbe Garantie durch Konstruktion, die
redblobgames' Flusswachstum aus dem Vorgehen "von aussen nach innen" zieht
(https://www.redblobgames.com/x/1723-procedural-river-growing/).

Der Abschnitt "Ketten erzwingen" schrieb DANACH Eltern aus der vorigen
Rechenstufe zurueck, ohne zu pruefen, ob das einen Ring schliesst. Zwei
Baeume gemischt sind keiner mehr. Die Tiefenschleife, die das haette auffangen
koennen, bricht bei `d < n` einfach ab und meldet nichts - deshalb blieb es
unbemerkt.

Dieser Test baut das Netz ueber mehrere Seeds und Kartengroessen und folgt bei
JEDEM Knoten die Elternkette bis zum Ende. Zusicherung: sie endet IMMER an
einem Knoten unter Wasser, nie in einem Ring, nie an Land.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.terrain_weltkarte as rw
import core.terrain_weltfluesse as wf

SEEDS = (20260804, 12345, 4242, 777001)
GROESSEN = (256, 384)


def _pruefe(size, seed):
    rw._STAPEL_CACHE.clear()
    H, _felder = rw.weltfeld(size, seed)
    netz = wf.flussnetz(H, seed)
    if netz is None:
        return None

    pk, el = netz["punkte"], netz["eltern"]
    n = len(pk)
    yi = np.clip(np.round(pk[:, 0]).astype(int), 0, size - 1)
    xi = np.clip(np.round(pk[:, 1]).astype(int), 0, size - 1)
    unter_wasser = H[yi, xi] <= 0.0

    meer = sackgasse = ring = 0
    for i in range(n):
        k, gesehen, schritte = i, set(), 0
        while True:
            if el[k] < 0:
                if unter_wasser[k]:
                    meer += 1
                else:
                    sackgasse += 1
                break
            if k in gesehen or schritte > n:
                ring += 1
                break
            gesehen.add(k)
            k, schritte = int(el[k]), schritte + 1

    return n, meer, sackgasse, ring


def main():
    fehler = []
    print("%-8s %-10s %8s %10s %10s %10s"
          % ("Groesse", "Seed", "Knoten", "ins Meer", "Sackgasse", "Ring"))
    print("-" * 60)
    for size in GROESSEN:
        for seed in SEEDS:
            ergebnis = _pruefe(size, seed)
            if ergebnis is None:
                continue
            n, meer, sackgasse, ring = ergebnis
            print("%-8d %-10d %8d %10d %10d %10d"
                  % (size, seed, n, meer, sackgasse, ring))
            if sackgasse or ring:
                fehler.append(
                    "%dpx Seed %d: %d Sackgassen, %d Ringe von %d Knoten - "
                    "nicht jeder Lauf erreicht das Meer"
                    % (size, seed, sackgasse, ring, n))
            if meer != n:
                fehler.append("%dpx Seed %d: nur %d von %d Knoten enden im "
                              "Meer" % (size, seed, meer, n))

    print("")
    if fehler:
        print("NICHT IN ORDNUNG - %d Befunde:" % len(fehler))
        for f in fehler:
            print("   " + f)
        return 1
    print("Alle Zusicherungen erfuellt - jeder Lauf erreicht das Meer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
