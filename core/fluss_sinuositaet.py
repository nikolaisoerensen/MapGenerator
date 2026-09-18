"""
Path: core/fluss_sinuositaet.py

Sinuositaet je Fluss aus dem gerasterten Flussnetz (Ticket #33).

Maeander und Breitenvariation der Fluesse sind in docs/SPEZIFIKATION.md als
Ziel formuliert ("Maeander (Sinuositaet der Hauptlaeufe) > 1.2"), aber nie
gemessen worden - ein Eindruck, keine Aussage.

Kennzahl: Sinuositaet = Lauflaenge eines Flusses / Luftlinie zwischen Anfang
und Ende. 1,0 heisst schnurgerade; ab etwa 1,5 gilt ein Lauf als
maeandrierend.

WAS HIER GELESEN WIRD: die vier Rasterkarten, die core/terrain_generator.py
(_weltfluesse) tatsaechlich zurueckgibt - "river_mask", "river_order",
"river_generation" (Knoten "terrain.redistribution"). Der Baum aus Knoten
und Elternzeigern, den core/terrain_weltfluesse.py:flussnetz() intern
aufbaut, wird NICHT nach aussen gereicht (nur zum Rastern benutzt und dann
verworfen) - eine Sinuositaetsmessung ueber die echte Pipeline muss also aus
dem Raster selbst zurueckrechnen, nicht aus dem Baum. Ticket #37
("Flussnetz als Linienzuege exportieren") wuerde genau diese Luecke
schliessen; bis dahin ist das hier die einzige Quelle.

VERFAHREN je Fluss-Ordnung (Strahler-Zahl, "river_order"):

1. Die Maske je Ordnungsstufe isolieren (ein Zusammenfluss hebt die Ordnung
   an, also markiert ein Ordnungswechsel im Raster automatisch das Ende
   eines Laufs und den Anfang des naechsten - kein Zusammenfluss ragt in
   eine andere Ordnungsstufe hinein).
2. Je 8-zusammenhaengender Komponente dieser Teilmaske: die zwei am
   weitesten auseinanderliegenden Pixel suchen (Graph-Distanz per
   Doppel-BFS, nicht Luftlinie) und den kuerzesten Pfad dazwischen
   zurueckverfolgen. Das ist robust gegen kleine Verzweigungen, die die
   Rasterung an einem Zusammenfluss hinterlassen kann.
3. Pfadlaenge = Summe der Schrittweiten entlang des Pfads (orthogonal 1 px,
   diagonal sqrt(2) px), Luftlinie = Euklidischer Abstand der beiden
   Endpunkte - beide mit `meters_per_pixel` in Meter umgerechnet.
"""
from collections import deque

import numpy as np

_NACHBARN_8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1))


def _nachbarn(y, x, hoehe, breite):
    for dy, dx in _NACHBARN_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < hoehe and 0 <= nx < breite:
            yield ny, nx


def _komponenten(maske):
    """8-zusammenhaengende Komponenten einer boolschen Maske, als Liste von
    Pixellisten [(y, x), ...]."""
    besucht = np.zeros_like(maske, dtype=bool)
    hoehe, breite = maske.shape
    komponenten = []
    ys, xs = np.nonzero(maske)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if besucht[y0, x0]:
            continue
        besucht[y0, x0] = True
        stapel = [(y0, x0)]
        pixel = []
        while stapel:
            y, x = stapel.pop()
            pixel.append((y, x))
            for ny, nx in _nachbarn(y, x, hoehe, breite):
                if maske[ny, nx] and not besucht[ny, nx]:
                    besucht[ny, nx] = True
                    stapel.append((ny, nx))
        komponenten.append(pixel)
    return komponenten


def _bfs(start, adjazenz):
    """Kuerzeste-Wege-BFS ab `start`. Gibt (entfernteste_knoten, vorgaenger) zurueck."""
    vorgaenger = {start: None}
    warteschlange = deque([start])
    letzter = start
    while warteschlange:
        knoten = warteschlange.popleft()
        letzter = knoten
        for nachbar in adjazenz[knoten]:
            if nachbar not in vorgaenger:
                vorgaenger[nachbar] = knoten
                warteschlange.append(nachbar)
    return letzter, vorgaenger


def _pfad_zwischen(pixel):
    """Findet in einer Pixelmenge die zwei am weitesten auseinanderliegenden
    Knoten (Graph-Durchmesser per Doppel-BFS) und den Pfad dazwischen, als
    geordnete Liste [(y, x), ...] von einem Ende zum anderen."""
    knoten = set(pixel)
    adjazenz = {}
    for (y, x) in knoten:
        adjazenz[(y, x)] = [n for n in _nachbarn(y, x, 10 ** 9, 10 ** 9)
                             if n in knoten]

    start = pixel[0]
    a, _ = _bfs(start, adjazenz)
    b, vorgaenger = _bfs(a, adjazenz)

    pfad = []
    knoten_akt = b
    while knoten_akt is not None:
        pfad.append(knoten_akt)
        knoten_akt = vorgaenger[knoten_akt]
    pfad.reverse()
    return pfad


def _pfadlaenge_px(pfad):
    if len(pfad) < 2:
        return 0.0
    arr = np.asarray(pfad, dtype=np.float64)
    schritte = np.diff(arr, axis=0)
    return float(np.sqrt((schritte ** 2).sum(axis=1)).sum())


def sinuositaet_je_fluss(river_mask, river_order, meters_per_pixel,
                          region_map=None, min_laenge_px=5):
    """
    Sinuositaet je einzelnem Flusslauf (Reach konstanter Ordnung).

    Returns: Liste von Dicts mit "ordnung", "region_index" (None wenn
    `region_map` nicht gegeben), "anzahl_px", "pfadlaenge_m",
    "luftlinie_m", "sinuositaet".

    `min_laenge_px`: Komponenten mit weniger Pixeln sind Stummel
    (Rasterreste an einer Muendung/einem Kartenrand) und werden
    ausgelassen - eine Sinuositaet ueber 2-3 Pixel ist Rauschen, keine
    Kennzahl.
    """
    river_mask = np.asarray(river_mask)
    river_order = np.asarray(river_order)
    ergebnisse = []

    vorhandene_ordnungen = np.unique(river_order[river_mask > 0])
    for ordnung_wert in vorhandene_ordnungen:
        if ordnung_wert <= 0:
            continue
        teil_maske = (river_mask > 0) & (river_order == ordnung_wert)
        for komponente in _komponenten(teil_maske):
            if len(komponente) < min_laenge_px:
                continue
            pfad = _pfad_zwischen(komponente)
            a, b = pfad[0], pfad[-1]
            luftlinie_px = float(np.hypot(a[0] - b[0], a[1] - b[1]))
            if luftlinie_px <= 0.0:
                continue
            pfadlaenge_px = _pfadlaenge_px(pfad)

            region_index = None
            if region_map is not None:
                region_map = np.asarray(region_map)
                werte = [int(region_map[y, x]) for (y, x) in pfad]
                region_index = max(set(werte), key=werte.count)

            ergebnisse.append({
                "ordnung": int(ordnung_wert),
                "region_index": region_index,
                "anzahl_px": len(pfad),
                "pfadlaenge_m": pfadlaenge_px * meters_per_pixel,
                "luftlinie_m": luftlinie_px * meters_per_pixel,
                "sinuositaet": pfadlaenge_px / luftlinie_px,
            })
    return ergebnisse
