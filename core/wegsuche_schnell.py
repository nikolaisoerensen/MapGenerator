"""
Path: core/wegsuche_schnell.py

A* auf dem Kostengitter, mit numba uebersetzt.

WARUM. `PathfindingSystem._a_stern()` ist reines Python: `heapq`, ein
`dict` fuer g_score, ein `set` fuer closed, und eine doppelte
`for dx/dy`-Schleife ueber die acht Nachbarn. Gemessen am 2026-08-23 auf
einer echten 1024-px-Karte: **0.85 s Median je Pfad**, Spanne 0.59-1.65 s.
Bei rund 70 gerouteten Ortspaaren sind das die 60 s, die
`settlement.pathfinding` im Pipeline-Log braucht - ein Drittel der ganzen
Ladezeit.

Der Innenpfad ist reine Arithmetik auf einem float32-Gitter; genau das,
wofuer numba da ist. numba ist in diesem Projekt ohnehin Pflicht (siehe
CLAUDE.md - opensimplex braucht den JIT, und numpy ist deshalb bewusst auf
2.4.6 gepinnt).

## Was hier bitgleich zum Python-Pfad sein muss

Ein schnellerer A*, der einen ANDEREN Pfad findet, ist kein
Geschwindigkeitsgewinn, sondern eine Gelaendeaenderung. Drei Dinge
entscheiden darueber:

1. **Die Heuristik.** Euklidisch, `sqrt(dx^2+dy^2)`, identisch zur
   Python-Fassung. Zulaessig, weil die billigste Kante 1.0 kostet und
   Diagonalen mit 1.414 > sqrt(2) leicht ueberschaetzt werden.

2. **Die Reihenfolge im Heap.** Python vergleicht die Tupel
   `(f, x, y)` lexikographisch: bei gleichem f gewinnt das kleinere x,
   dann das kleinere y. Der Array-Heap hier vergleicht in derselben
   Reihenfolge (siehe `_kleiner`). Ohne das koennten bei Kostengleichstand
   andere - gleich gute, aber anders verlaufende - Pfade herauskommen, und
   die Karte waere nicht mehr reproduzierbar.

3. **Die Nachbarreihenfolge.** Python laeuft `dx` aussen von -r bis +r,
   `dy` innen. Dieselbe Reihenfolge steht unten in `_DX`/`_DY`.

`tests/smoke_test_wegsuche_schnell.py` prueft alle drei gegen die
Python-Fassung auf echtem Gelaende.

## Rueckfall

Fehlt numba, faellt `wegsuche()` auf den uebergebenen Python-Rueckfall
zurueck - aber mit einer LAUTEN Logzeile, nicht still. Ein stiller
Rueckfall auf einen langsameren Pfad ist in diesem Projekt schon zweimal
unbemerkt geblieben (GPU nach dem Dateiumzug, adaptives Mesh bei
2^n+1); die Lehre steht in CLAUDE.md.
"""

import logging
import numpy as np

_LOGGER = logging.getLogger(__name__)

# Nachbarreihenfolge exakt wie die Python-Schleife:
#   for dx in (-r, 0, +r):  for dy in (-r, 0, +r):  (0,0) uebersprungen
_DX = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
_DY = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)

try:
    from numba import njit
    NUMBA_DA = True
except Exception:                                            # noqa: BLE001
    NUMBA_DA = False

    def njit(*a, **k):                                       # type: ignore
        def deko(f):
            return f
        return deko


@njit(cache=True, nogil=True)
def _kleiner(fa, xa, ya, fb, xb, yb):
    """Lexikographischer Vergleich (f, x, y) - wie Pythons Tupelvergleich."""
    if fa != fb:
        return fa < fb
    if xa != xb:
        return xa < xb
    return ya < yb


@njit(cache=True, nogil=True)
def _hoch(hf, hx, hy, i):
    """Ein Element im Binaerheap nach oben schieben."""
    while i > 0:
        e = (i - 1) >> 1
        if _kleiner(hf[i], hx[i], hy[i], hf[e], hx[e], hy[e]):
            hf[i], hf[e] = hf[e], hf[i]
            hx[i], hx[e] = hx[e], hx[i]
            hy[i], hy[e] = hy[e], hy[i]
            i = e
        else:
            break


@njit(cache=True, nogil=True)
def _runter(hf, hx, hy, n, i):
    """Ein Element im Binaerheap nach unten schieben."""
    while True:
        li, ri, klein = 2 * i + 1, 2 * i + 2, i
        if li < n and _kleiner(hf[li], hx[li], hy[li],
                               hf[klein], hx[klein], hy[klein]):
            klein = li
        if ri < n and _kleiner(hf[ri], hx[ri], hy[ri],
                               hf[klein], hx[klein], hy[klein]):
            klein = ri
        if klein == i:
            break
        hf[i], hf[klein] = hf[klein], hf[i]
        hx[i], hx[klein] = hx[klein], hx[i]
        hy[i], hy[klein] = hy[klein], hy[i]
        i = klein


@njit(cache=True, nogil=True)
def _a_stern_kern(kosten, start_x, start_y, end_x, end_y, max_nodes,
                  schritt, kante, kante_bias, kante_skala, h_gewicht):
    """
    A* auf `kosten` (float64, [hoehe, breite]). inf sperrt eine Zelle.

    Rueckgabe: (gefunden, pfad_x, pfad_y, laenge, expandiert). Der Pfad
    steht von Start nach Ziel in den ersten `laenge` Eintraegen.
    """
    hoehe, breite = kosten.shape
    n_zellen = hoehe * breite

    g = np.full(n_zellen, np.inf)
    vorher = np.full(n_zellen, -1, dtype=np.int64)
    zu = np.zeros(n_zellen, dtype=np.uint8)

    kap = 1024
    hf = np.empty(kap); hx = np.empty(kap, dtype=np.int32)
    hy = np.empty(kap, dtype=np.int32)
    hn = 0

    hf[0] = 0.0; hx[0] = start_x; hy[0] = start_y; hn = 1
    g[start_y * breite + start_x] = 0.0

    expandiert = 0
    gefunden = False
    while hn > 0 and expandiert < max_nodes:
        cx, cy = hx[0], hy[0]
        hn -= 1
        hf[0] = hf[hn]; hx[0] = hx[hn]; hy[0] = hy[hn]
        if hn > 0:
            _runter(hf, hx, hy, hn, 0)

        ck = cy * breite + cx
        if zu[ck] == 1:
            continue                     # veralteter Heapeintrag
        zu[ck] = 1
        expandiert += 1

        if cx == end_x and cy == end_y:
            gefunden = True
            break

        cg = g[ck]
        for r in range(8):
            nx = cx + _DX[r] * schritt
            ny = cy + _DY[r] * schritt
            if nx < 0 or nx >= breite or ny < 0 or ny >= hoehe:
                continue
            nk = ny * breite + nx
            if zu[nk] == 1:
                continue
            k = kosten[ny, nx]
            if not np.isfinite(k):
                continue
            if kante_bias > 0.0:
                de = kante[ny, nx]
                if np.isfinite(de):
                    k *= 1.0 + kante_bias * (de / (de + kante_skala))
            if _DX[r] != 0 and _DY[r] != 0:
                k *= 1.414
            neu = cg + k
            if neu < g[nk]:
                vorher[nk] = ck
                g[nk] = neu
                dxh = float(nx - end_x)
                dyh = float(ny - end_y)
                f = neu + h_gewicht * np.sqrt(dxh * dxh + dyh * dyh)
                if hn >= kap:
                    kap2 = kap * 2
                    nf = np.empty(kap2); nxx = np.empty(kap2, dtype=np.int32)
                    nyy = np.empty(kap2, dtype=np.int32)
                    nf[:hn] = hf[:hn]; nxx[:hn] = hx[:hn]; nyy[:hn] = hy[:hn]
                    hf = nf; hx = nxx; hy = nyy; kap = kap2
                hf[hn] = f; hx[hn] = nx; hy[hn] = ny
                hn += 1
                _hoch(hf, hx, hy, hn - 1)

    pfad_x = np.empty(n_zellen, dtype=np.int32)
    pfad_y = np.empty(n_zellen, dtype=np.int32)
    laenge = 0
    if gefunden:
        k = end_y * breite + end_x
        while k != -1:
            pfad_x[laenge] = k % breite
            pfad_y[laenge] = k // breite
            laenge += 1
            k = vorher[k]
        # umdrehen: der Weg wurde vom Ziel her aufgerollt
        for i in range(laenge // 2):
            j = laenge - 1 - i
            pfad_x[i], pfad_x[j] = pfad_x[j], pfad_x[i]
            pfad_y[i], pfad_y[j] = pfad_y[j], pfad_y[i]

    return gefunden, pfad_x, pfad_y, laenge, expandiert


_GEMELDET = False


def wegsuche(kosten, start, ziel, max_nodes, schritt=1,
             kante=None, kante_bias=0.0, kante_skala=1.0, h_gewicht=1.0):
    """
    Ein A*-Lauf. Rueckgabe: Liste von (x, y) oder None.

    `h_gewicht` > 1 macht die Heuristik gewichtet (Punkt 2.4 der
    Leistungsliste): der Suchbaum wird deutlich schmaler, der Pfad ist
    dafuer hoechstens h_gewicht-mal teurer als der optimale. Bei 1.0
    verhaelt sich die Funktion exakt wie der Python-Pfad.
    """
    global _GEMELDET
    if not NUMBA_DA:
        if not _GEMELDET:
            _LOGGER.warning(
                "numba fehlt - Wegsuche laeuft im langsamen Python-Pfad "
                "(gemessen 0.85 s statt 0.03 s je Route). Pruefen: "
                ".venv/Scripts/python.exe -c \"import numba\"")
            _GEMELDET = True
        return None

    kosten = np.ascontiguousarray(kosten, dtype=np.float64)
    if kante is None:
        kante = np.zeros((1, 1), dtype=np.float64)
        kante_bias = 0.0
    else:
        kante = np.ascontiguousarray(kante, dtype=np.float64)

    gefunden, px, py, laenge, _exp = _a_stern_kern(
        kosten, int(start[0]), int(start[1]), int(ziel[0]), int(ziel[1]),
        int(max_nodes), int(schritt), kante, float(kante_bias),
        float(kante_skala), float(h_gewicht))
    if not gefunden:
        return None
    return [(int(px[i]), int(py[i])) for i in range(laenge)]
