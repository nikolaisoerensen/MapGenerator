"""
Path: tests/smoke_test_kuesten_schnitt.py

Prueft `gui/widgets/kuesten_schnitt.py` - das Gitter entlang der 0-Kontur
geschnitten, damit die Kuestenlinie eine echte Meshkante wird.

DIE KERNZUSICHERUNG ist die dritte Gruppe: die Konturvertices duerfen NICHT
auf Pixelecken sitzen. Genau daran ist alles Vorherige gescheitert - das
Quadtree (6.16) hat gemessen 0.000004 px Versatz, also exakt auf der Ecke,
und deshalb folgt seine Kuestensilhouette dem Raster.

Die zweite Kernzusicherung ist die Dichtheit: an einem Schnittverfahren ist
sie nicht selbstverstaendlich, und ein Riss im Gelaende faellt erst im
laufenden 3D-Fenster auf - also genau dort, wo dieses Projekt nicht
automatisch pruefen kann. Gemessen wird deshalb dreifach: keine Kante an mehr
als zwei Dreiecken, Gesamtflaeche exakt gleich der Gitterflaeche (faengt
Loecher UND Ueberlappungen), keine entarteten Dreiecke.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_kuesten_schnitt.py
"""

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.terrain_weltkarte import weltfeld
from core.vektor_kueste import VektorKueste, an_punkten
from gui.widgets.kuesten_schnitt import schnitt_netz

GROESSEN = (256, 384, 512)
SEED = 20260804

_CACHE = {}


def _netz(size):
    if size not in _CACHE:
        H, felder = weltfeld(size, SEED)
        H = np.asarray(H, dtype=np.float64)

        # DIE VORHANDENE VEKTORKUESTE BENUTZEN, keine zweite bauen.
        #
        # Seit `weltfeld()` die Kuestenformung ueber den Vektorweg macht
        # (VEKTOR_KUESTE_AKTIV), ist H bereits geformt. Eine hier neu
        # gebaute `VektorKueste` wuerde die Kueste ein ZWEITES Mal
        # anwenden - gemessen fiel die Konturfreiheit dadurch von 99 % auf
        # 68 %, weil die zweite Anwendung das Gelaende an der Wasserlinie
        # noch einmal flachzieht und der Nulldurchgang dadurch unscharf
        # wird. `weltfeld()` legt das Objekt deshalb in
        # `felder["vektor_kueste"]` ab: EIN Objekt je Karte, von Raster und
        # Mesh gemeinsam benutzt - genau die Bauart aus
        # docs/KUESTENMODELL.md §7.
        vk = felder.get("vektor_kueste")
        if vk is None:
            vk = VektorKueste(H, felder["regionen"], SEED)

        p, h, tri, land = schnitt_netz(
            H, hoehen_fn=lambda x, y: an_punkten(vk, x, y))
        _CACHE[size] = (H, p, h, tri, land)
    return _CACHE[size]


def _flaechen(p, tri):
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


# --------------------------------------------------------------------- #

def zellzerlegung():
    """1. Von Hand nachrechenbare Faelle, inklusive beider Sattelformen."""
    fehler = []
    faelle = [
        ("alles Land", np.ones((3, 3)) * 100.0, 4.0),
        ("senkrechte Kueste", np.array([[-10, -10, 10, 10]] * 4, float), 9.0),
        ("eine Ecke Land", np.array([[10, -10], [-10, -10]], float), 1.0),
        ("Sattel diagonal", np.array([[10, -10], [-10, 10]], float), 1.0),
        ("Sattel gespiegelt", np.array([[-10, 10], [10, -10]], float), 1.0),
    ]
    for name, H, soll in faelle:
        p, _h, tri, _land = schnitt_netz(H)
        A = _flaechen(p, tri)
        flaeche_ok = abs(float(np.abs(A).sum()) - soll) < 1e-9
        entartet = int((np.abs(A) < 1e-12).sum())
        wicklung = bool(np.all(A > 0) or np.all(A < 0))
        ok = flaeche_ok and entartet == 0 and wicklung
        print(f"[{'OK' if ok else 'FEHLER'}] {name}: {len(tri)} Dreiecke, "
              f"Flaeche {np.abs(A).sum():.6f} (soll {soll}), "
              f"entartet {entartet}, Wicklung einheitlich {wicklung}")
        if not ok:
            fehler.append(f"{name}: Flaeche/Entartung/Wicklung nicht in Ordnung")
    return fehler


def dichtheit():
    """2. Keine Risse und keine Ueberlappungen auf echtem Gelaende."""
    fehler = []
    for size in GROESSEN:
        _H, p, _h, tri, _land = _netz(size)
        zaehler = Counter()
        for t in tri:
            for i in range(3):
                zaehler[tuple(sorted((int(t[i]), int(t[(i + 1) % 3]))))] += 1
        zuviel = sum(1 for v in zaehler.values() if v > 2)

        A = _flaechen(p, tri)
        soll = float((size - 1) ** 2)
        flaeche = float(np.abs(A).sum())
        entartet = int((np.abs(A) < 1e-12).sum())
        ok = zuviel == 0 and abs(flaeche - soll) < 1e-6 and entartet == 0
        print(f"[{'OK' if ok else 'FEHLER'}] {size} px: Kanten mit >2 "
              f"Dreiecken {zuviel}, Flaeche {flaeche:.3f} (soll {soll:.0f}), "
              f"entartet {entartet}")
        if not ok:
            fehler.append(f"{size} px: undicht (Kanten {zuviel}, "
                          f"Flaechenfehler {flaeche - soll:+.3g}, "
                          f"entartet {entartet})")
    return fehler


def kontur_frei_vom_raster():
    """
    3. DIE KERNZUSICHERUNG: die Kuestenvertices sitzen NICHT auf Pixelecken.

    Gegenprobe mit den Gittervertices desselben Netzes - die muessen exakt
    auf 0.0 liegen. Ohne diese Gegenprobe wuerde ein Messfehler in der
    Versatzrechnung als Erfolg durchgehen.
    """
    fehler = []
    for size in GROESSEN:
        _H, p, h, tri, _land = _netz(size)
        n_ecken = size * size
        benutzt = np.zeros(len(p), dtype=bool)
        benutzt[tri.ravel()] = True

        kontur = np.arange(n_ecken, len(p))[benutzt[n_ecken:]]
        gitter = np.arange(n_ecken)[benutzt[:n_ecken]]

        def versatz(idx):
            return np.hypot(p[idx, 0] - np.round(p[idx, 0]),
                            p[idx, 1] - np.round(p[idx, 1]))

        v_kontur = versatz(kontur)
        v_gitter = versatz(gitter)
        frei = float((v_kontur > 0.01).mean())
        hoehe_null = bool(np.all(h[kontur] == 0.0))
        gitter_fest = float(np.median(v_gitter)) < 1e-12

        ok = frei > 0.8 and hoehe_null and gitter_fest
        print(f"[{'OK' if ok else 'FEHLER'}] {size} px: {len(kontur)} "
              f"Konturvertices, {frei:.1%} frei vom Raster "
              f"(Median {np.median(v_kontur):.4f} px), Hoehe exakt 0: "
              f"{hoehe_null}, Gittervertices fest: {gitter_fest}")
        if not ok:
            fehler.append(f"{size} px: Kontur nur {frei:.0%} frei, "
                          f"Hoehe-0 {hoehe_null}, Gitter-Gegenprobe "
                          f"{gitter_fest}")
    return fehler


def aufwand():
    """4. Der Schnitt darf das Netz nicht sprengen."""
    fehler = []
    for size in GROESSEN:
        _H, _p, _h, tri, _land = _netz(size)
        voll = 2 * (size - 1) ** 2
        mehr = len(tri) / voll - 1.0
        ok = mehr < 0.25
        print(f"[{'OK' if ok else 'FEHLER'}] {size} px: {len(tri)} Dreiecke, "
              f"{mehr:+.1%} gegenueber dem vollen Gitter ({voll})")
        if not ok:
            fehler.append(f"{size} px: {mehr:+.0%} mehr Dreiecke")
    return fehler


def land_see_getrennt():
    """
    5. Jedes Dreieck liegt ganz auf einer Seite.

    Das ist der Zweck des Schnitts: ein Dreieck, das die Wasserlinie
    ueberspannt, ist genau die Rastertreppe.
    """
    fehler = []
    for size in GROESSEN:
        _H, _p, h, tri, land = _netz(size)
        hoehen = h[tri]
        # Konturvertices liegen auf 0 und zaehlen zu beiden Seiten.
        ueber = (hoehen > 1e-9).any(axis=1)
        unter = (hoehen < -1e-9).any(axis=1)
        gemischt = int((ueber & unter).sum())
        anteil = gemischt / max(len(tri), 1)
        ok = anteil < 0.001
        print(f"[{'OK' if ok else 'FEHLER'}] {size} px: {gemischt} Dreiecke "
              f"ueberspannen die Wasserlinie ({anteil:.4%})")
        if not ok:
            fehler.append(f"{size} px: {anteil:.2%} der Dreiecke ueberspannen "
                          f"die Wasserlinie")
    return fehler


def wicklung_wie_gitter():
    """
    7. Die Wicklung muss zum GITTER passen, nicht nur einheitlich sein.

    Gruppe 1 prueft schon, dass alle Dreiecke gleich herum liegen. Das war
    2026-08-22 gruen, waehrend im laufenden Programm das gesamte Gelaende in
    Streifen zerfiel: der Schnitt wickelte gegen den Uhrzeigersinn, das
    Gitter in map_display_3d._generate_terrain_mesh() im Uhrzeigersinn.

    Mit glFrontFace(GL_CW) + glCullFace(GL_BACK) verschwindet dann alles,
    was zur Kamera zeigt. "Einheitlich" ist also die falsche Frage; die
    richtige ist "einheitlich WIE das Gitter".

    Soll ist deshalb die signierte Flaeche des Gitterdreiecks
    (top_left, bottom_left, top_right) = (x,y), (x,y+1), (x+1,y): negativ.
    """
    fehler = []
    soll = 0.5 * ((0 - 0) * (0 - 0) - (1 - 0) * (1 - 0))   # = -0.5, im UZS

    faelle = [("nur Land", np.full((8, 8), 100.0)),
              ("nur See", np.full((8, 8), -100.0)),
              ("senkrechte Kueste",
               np.tile(np.r_[np.full(4, -20.0), np.full(4, 60.0)], (8, 1))),
              ("Sattel diagonal", np.array([[10, -10], [-10, 10]], float)),
              ("Sattel gespiegelt", np.array([[-10, 10], [10, -10]], float))]
    for name, H in faelle:
        p_, _h, tri, _land = schnitt_netz(H)
        A = _flaechen(p_, tri)
        gleich = float(np.mean(np.sign(A) == np.sign(soll)))
        ok = gleich == 1.0
        print(f"[{'OK' if ok else 'FEHLER'}] {name}: {gleich:.0%} der "
              f"{len(tri)} Dreiecke wie das Gitter gewickelt "
              f"(Median {np.median(A):+.4f}, Gitter {soll:+.1f})")
        if not ok:
            fehler.append(f"{name}: nur {gleich:.0%} wie das Gitter gewickelt "
                          f"- Backface-Culling wirft das Gelaende weg")

    # Gegenprobe auf echtem Gelaende, wo Sattelzellen wirklich vorkommen.
    for size in GROESSEN:
        _H, p_, _h, tri, _land = _netz(size)
        A = _flaechen(p_, tri)
        gleich = float(np.mean(np.sign(A) == np.sign(soll)))
        ok = gleich == 1.0
        print(f"[{'OK' if ok else 'FEHLER'}] {size} px: {gleich:.4%} wie das "
              f"Gitter gewickelt")
        if not ok:
            fehler.append(f"{size} px: nur {gleich:.2%} wie das Gitter")
    return fehler


def randfaelle():
    """6. Karten ohne Kueste."""
    fehler = []
    for name, H in (("nur Land", np.full((16, 16), 100.0)),
                    ("nur See", np.full((16, 16), -100.0))):
        try:
            p, _h, tri, land = schnitt_netz(H)
            A = _flaechen(p, tri)
            ok = (abs(float(np.abs(A).sum()) - 225.0) < 1e-9
                  and len(tri) == 2 * 225)
            print(f"[{'OK' if ok else 'FEHLER'}] {name}: {len(tri)} Dreiecke, "
                  f"Flaeche {np.abs(A).sum():.1f} (soll 225)")
            if not ok:
                fehler.append(f"{name}: unerwartetes Netz")
        except Exception as e:                              # noqa: BLE001
            print(f"[FEHLER] {name}: Absturz - {e}")
            fehler.append(f"{name}: Absturz - {e}")
    return fehler


# --------------------------------------------------------------------- #

def lauf():
    gruppen = [
        ("zellzerlegung", zellzerlegung),
        ("dichtheit", dichtheit),
        ("kontur_frei_vom_raster", kontur_frei_vom_raster),
        ("aufwand", aufwand),
        ("land_see_getrennt", land_see_getrennt),
        ("randfaelle", randfaelle),
        ("wicklung_wie_gitter", wicklung_wie_gitter),
    ]
    ergebnis, alle = {}, []
    for name, fn in gruppen:
        print(f"\n--- {name} ---")
        f = fn()
        ergebnis[name] = not f
        alle.extend(f)

    print("\n=== SUMMARY ===")
    for name, ok in ergebnis.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if alle:
        print(f"\nNICHT IN ORDNUNG - {len(alle)} Befunde:")
        for f in alle:
            print(f"   {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(lauf())
