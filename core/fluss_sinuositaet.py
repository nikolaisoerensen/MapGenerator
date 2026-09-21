"""
Path: core/fluss_sinuositaet.py

Sinuositaet der Fluesse - Ticket #33.

Maeander waren in der Spezifikation seit jeher ein Ziel (§3.6: "Sinuositaet
der Hauptlaeufe > 1.2"), aber nie gemessen - Stand dort bis heute "nicht
gemessen". Dieses Modul liefert die Messfunktion; die Ist-Erhebung (Verteilung
nach Flussordnung und Region, 512/1024 px, drei Seeds) steht im
Commit-Text und in docs/TESTBERICHT.md.

WAS SINUOSITAET IST: Lauflaenge eines Flusses geteilt durch die Luftlinie
zwischen seinem Anfang und seinem Ende. 1.0 = schnurgerade, ab rund 1.5 gilt
ein Lauf als maeandrierend.

WELCHES NETZ. Die Weltkarte erzeugt ihr Flussnetz in `core/terrain_weltfluesse.
flussnetz()` (STUFE B, seit 2026-08-05 der Kern der Kartengenerierung, siehe
`terrain_generator._weltfluesse()`) - ein Spannbaum aus Poisson-Disk-Knoten,
dessen Kanten dem Gelaende ausweichen (Kosten steigen mit dem Anstieg). Genau
dieses Ausweichen IST der Maeander in diesem Programm (SPEZIFIKATION §15/§16:
"der Maeander entsteht dabei von selbst - der Lauf geht um den Berg herum").
Die hier gemessene Sinuositaet nimmt deshalb die KNOTENPOLYLINIE des
Spannbaums (`netz["punkte"]`, verbunden ueber `netz["eltern"]`) - nicht die
zusaetzliche Catmull-Rom-Gl'aettung aus `kantenpunkte()`, die nur scharfe
Ecken abrundet (ein reines Zeichen-Feature) und die Lauflaenge messbar NICHT
verlaengert, eher leicht verkuerzt. Ein Fluss, der ueber die Knoten hinweg
schon gerade ist, bleibt nach der Glaettung gerade; einer, der einen Berg
umrundet, bleibt es auch. Die Vereinfachung aendert also nichts an der
Groessenordnung, spart aber die (teurere) Spline-Rekonstruktion in der
Messung.

`core/terrain_river_network.py` traegt ein zweites, LOKALES Flussnetz
("STUFE A", `carve_river_network`, einzelne Kartenkacheln ohne Regionsbezug).
Es ist ueber `gui.config.value_default.FLUSSNETZ_AKTIV` standardmaessig
ABGESCHALTET (siehe `terrain_generator._apply_river_network`) und traegt zur
tatsaechlich erzeugten Weltkarte nichts bei - eine Sinuositaets-Messung dort
waere eine Messung an totem Code. Von dort wird nur `strahler_order()`
wiederverwendet, weil sie unabhaengig vom Netz-Aufbau reine Graphrechnung ist.

EIN "FLUSS" IST HIER: eine maximale Kette von Knoten GLEICHER Strahler-
Ordnung im Spannbaum - von der Quelle (oder der Stelle, an der zwei
gleichgrosse Zufluesse zusammentreffen und die Ordnung ansteigt) bis zur
naechsten Muendung in einen groesseren Fluss oder ins Meer. `strahler_order()`
haengt die Ordnung an die Kante KNOTEN->ELTERNKNOTEN, nicht an den Knoten
isoliert; ein Fluss kann deshalb mehrere Baumkanten lang sein, bevor die
Ordnung wechselt. Jeder Knoten hat genau einen Elternknoten, gehoert also zu
genau einer Kette - Ueberlappungen sind ausgeschlossen.
"""

from typing import Any, Dict, List, Optional

import numpy as np


def sinuositaet_pfad(punkte) -> float:
    """
    Sinuositaet einer Polylinie: Lauflaenge / Luftlinie zwischen erstem und
    letztem Punkt. 1.0 = schnurgerade.

    Gibt np.nan zurueck, wenn der Pfad weniger als zwei Punkte hat oder Anfang
    und Ende zusammenfallen (Luftlinie ~0) - eine Sinuositaet ist dann nicht
    definiert, und eine stille 1.0 waere falsch positiv (sieht aus wie "gerade",
    ist aber "nicht messbar").
    """
    p = np.asarray(punkte, dtype=np.float64)
    if len(p) < 2:
        return float("nan")
    lauflaenge = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    luftlinie = float(np.linalg.norm(p[-1] - p[0]))
    if luftlinie < 1e-9:
        return float("nan")
    return lauflaenge / luftlinie


def fluss_segmente(eltern, order) -> List[List[int]]:
    """
    Zerlegt einen Flussbaum (`eltern` = Elternindex je Knoten, `order` =
    Strahler-Ordnung je Knoten, wie von
    `core.terrain_river_network.strahler_order` geliefert) in einzelne
    "Fluesse" - Ketten gleicher Ordnung, flussaufwaerts nach flussabwaerts
    geordnet.

    Ein Knoten `i` ist KOPF einer Kette, wenn KEIN Kind von `i` dieselbe
    Ordnung traegt wie `i` selbst - das ist entweder eine echte Quelle
    (kein Kind) oder die Stelle, an der zwei gleich grosse Zufluesse
    zusammentreffen und `strahler_order` die Ordnung erhoeht (dort ist `i`
    per Definition groesser als jedes seiner Kinder). Von dort wird
    flussabwaerts gelaufen, solange die Ordnung gleich bleibt; der erste
    Knoten mit anderer Ordnung (Muendung in einen groesseren Fluss) oder das
    Ende der Kette (Auslass, `eltern < 0`) wird noch mitgenommen, weil der
    Fluss dort real endet, auch wenn diese letzte Kante schon einem anderen
    Fluss gehoert.
    """
    eltern = np.asarray(eltern)
    order = np.asarray(order)
    n = len(eltern)

    # Fuer jeden Knoten: hat er ein Kind, dessen Kante dieselbe Ordnung
    # traegt? Dann ist er FORTSETZUNG jenes Kindes, nicht Kopf einer eigenen
    # Kette.
    fortsetzung_von_kind = np.zeros(n, dtype=bool)
    for i in range(n):
        e = int(eltern[i])
        if e >= 0 and order[i] == order[e]:
            fortsetzung_von_kind[e] = True

    segmente = []
    for start in range(n):
        if fortsetzung_von_kind[start]:
            continue
        kette = [start]
        cur = start
        while True:
            p = int(eltern[cur])
            if p < 0:
                break
            kette.append(p)
            if order[p] != order[cur]:
                break
            cur = p
        segmente.append(kette)
    return segmente


def sinuositaet_je_fluss(punkte_px, eltern, order, mpp: float = 1.0,
                         region_map: Optional[np.ndarray] = None
                         ) -> List[Dict[str, Any]]:
    """
    Sinuositaet jedes Flusses im Netz (siehe `fluss_segmente`), mit Ordnung,
    Lauflaenge in Metern und - falls `region_map` gegeben ist - der Region am
    mittleren Knoten der Kette.

    `punkte_px` sind Knotenkoordinaten in PIXELN (wie `netz["punkte"]` aus
    `core.terrain_weltfluesse.flussnetz()`), `mpp` Meter je Pixel. Die
    Sinuositaet selbst ist einheitenlos (ein Verhaeltnis); `mpp` wird nur
    fuer `laenge_m` gebraucht.

    Fluesse mit undefinierter Sinuositaet (siehe `sinuositaet_pfad`) werden
    ausgelassen, nicht mit einer erfundenen Zahl aufgefuellt.

    Returns: Liste von dicts {order, sinuositaet, laenge_m, n_knoten, region,
    knoten}. `region` ist None ohne `region_map`.
    """
    punkte_px = np.asarray(punkte_px, dtype=np.float64)
    segmente = fluss_segmente(eltern, order)
    order = np.asarray(order)

    ergebnis = []
    for kette in segmente:
        if len(kette) < 2:
            continue  # ein einzelner Muendungsknoten ist kein Fluss
        pkt_m = punkte_px[kette] * mpp
        sin = sinuositaet_pfad(pkt_m)
        if not np.isfinite(sin):
            continue
        laenge_m = float(np.sum(np.linalg.norm(np.diff(pkt_m, axis=0), axis=1)))

        region = None
        if region_map is not None:
            mitte = kette[len(kette) // 2]
            y = int(np.clip(round(float(punkte_px[mitte, 0])),
                            0, region_map.shape[0] - 1))
            x = int(np.clip(round(float(punkte_px[mitte, 1])),
                            0, region_map.shape[1] - 1))
            region = int(region_map[y, x])

        ergebnis.append(dict(
            order=int(order[kette[0]]),
            sinuositaet=float(sin),
            laenge_m=laenge_m,
            n_knoten=len(kette),
            region=region,
            knoten=kette,
        ))
    return ergebnis
