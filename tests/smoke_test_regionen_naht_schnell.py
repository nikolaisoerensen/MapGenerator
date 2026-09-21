"""
Path: tests/smoke_test_regionen_naht_schnell.py

SCHNELLER WAECHTER-ERSATZ fuer die Gelaendeverstimmungs-Frage aus
smoke_test_regionen_welt.py (Ticket #46).

WARUM ES DAS GIBT: smoke_test_regionen_welt.py ist der empfindlichste
Waechter fuer Gelaendeform (siehe CLAUDE.md, Abschnitt "Gelaendeaenderungen
verstimmen zuerst die Regionseichung"), braucht aber gemessen 80.8 s in
diesem Lauf (Ticket #45 hatte an einem anderen Tag 46.4 s gemessen - die
Maschine schwankt laut tools/testlauf.py um Faktor 2-3). Allein diese eine
Datei wuerde einen 2-Minuten-Waechter entweder sprengen oder ihn bei einer
einzigen langsamen Ausfuehrung ueber die Grenze schieben. Sie bleibt deshalb
in der Eichung (naechtlich), wo sie ungestoert die vollen fuenf Seeds bei
384 px durch die ECHTE Anzeigepipeline (Redistribution + Fluesse) schickt
und die neun Regionen gegen ihre kalibrierten Hang-/Wasserziele haelt.

DIESE DATEI ERSETZT DIESE KALIBRIERUNG NICHT. Sie beantwortet dieselbe
FRAGE ("hat eine Aenderung an core/terrain_weltkarte.py eine sichtbare
Naht zwischen zwei Regionen aufgerissen?") mit denselben, bereits bewaehrten
Mitteln - der Nahtpruefung aus smoke_test_regionen_welt.py, Schritt 3, Wort
fuer Wort dieselbe Toleranzregel (Grenzstreifen <= 125% des Regionsinneren,
mit derselben Begruendung, siehe dort) -, aber

  * auf ROHEM `weltfeld()` (Rauschen + Kuestenumformung), NICHT durch die
    volle BaseTerrainGenerator-Pipeline (Redistribution/Fluesse). Genau
    `_kuesten_umformen()` innerhalb von `weltfeld()` ist die Stelle, die am
    2026-08-24 den dokumentierten Samarcia/Morobora-Nahtfehler ausloeste -
    die Naht entsteht schon dort, das raue Feld genuegt, um sie zu sehen.
  * bei 256 statt 384/512/1024 Pixeln - eine ECHTE Kartengroesse (2er-Potenz,
    keine ausgedachte Zahl, siehe CLAUDE.md "Gruene Tests koennen eine tote
    Funktion verdecken").
  * mit EINEM Seed statt fuenf.

Gemessen (dieser Rechner, 2026-09-21): weltfeld+regionen bei 256 px rund 6 s,
die Nahtpruefung selbst < 1 s. Die ganze Datei liegt damit weit innerhalb
des 2-Minuten-Waechterbudgets.

WAS DIESE DATEI NICHT FAENGT: die absoluten Hang-/Wasserziele je Region
(Macchia/Thalassia sind zum Zeitpunkt dieses Tickets bereits bekannt
daneben, siehe docs/TESTBESTAND_BEWERTUNG.md) und die Aufloesungs-
unabhaengigkeit (Pruefung 4 der vollen Datei). Beides bleibt der Eichung
vorbehalten - dafuer braucht es die fuenf Seeds und die grossen Groessen.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_regionen_naht_schnell.py
"""

import os
import sys

import numpy as np

_HIER = os.path.dirname(os.path.abspath(__file__))
_WURZEL = os.path.dirname(_HIER)
if _WURZEL not in sys.path:
    sys.path.insert(0, _WURZEL)

_QT_APP = None


def _qt():
    global _QT_APP
    from PyQt6.QtGui import QGuiApplication
    if _QT_APP is None:
        _QT_APP = QGuiApplication.instance() or QGuiApplication([])
    return _QT_APP


SIZE = 256
SEED = 20260804


def lauf():
    _qt()
    import core.terrain_weltkarte as rw
    from managers.shader_manager import ShaderManager
    from scipy import ndimage

    manager = ShaderManager()
    worker = manager._ensure_worker()
    gpu = worker.gpu_available
    fehler = []

    # ---------- 1: Reproduzierbarkeit ----------
    # Billige Zugabe: ein nichtdeterministisches weltfeld() waere selbst ein
    # STILLER_RUECKFALL (z.B. ein nicht geleerter/falsch geleerter Cache) und
    # wuerde die Nahtpruefung unten ohnehin verfaelschen.
    rw._STAPEL_CACHE.clear()
    a, _ = rw.weltfeld(SIZE, SEED, shader_manager=manager if gpu else None)
    rw._STAPEL_CACHE.clear()
    b, _ = rw.weltfeld(SIZE, SEED, shader_manager=manager if gpu else None)
    gleich = np.array_equal(a, b)
    print("1. Reproduzierbar (zweimal gerechnet, bitgleich) ... %s"
          % ("ok" if gleich else "FEHLER"))
    if not gleich:
        fehler.append("gleicher Seed ergibt verschiedene Welten, "
                      "groesste Abweichung %.3g" % np.abs(a - b).max())

    H = a

    # ---------- 2: Nahtpruefung, identische Regel wie in
    #               smoke_test_regionen_welt.py Schritt 3 ----------
    mpp = rw.WELT_KM * 1000.0 / SIZE
    glatt = ndimage.gaussian_filter(H, max(120.0 / mpp, 1.0))
    dy, dx = np.gradient(glatt, mpp)
    steig = np.hypot(dx, dy)
    steig = np.where(H > 0.0, steig, 0.0)
    kante_px = rw.REGION_KM * 1000.0 / mpp
    band = max(int(0.10 * kante_px), 2)

    maske, _sdf = rw.kontinentform(SIZE, SEED, manager if gpu else None)
    gewichte, _zellen = rw.voronoi_regionen(
        maske, SEED, punktzahl=200, shader_manager=manager if gpu else None)
    fuehrend = np.argmax(gewichte, axis=0)

    wechsel = np.zeros_like(maske)
    wechsel[:, :-1] |= fuehrend[:, :-1] != fuehrend[:, 1:]
    wechsel[:-1, :] |= fuehrend[:-1, :] != fuehrend[1:, :]
    naht_band = (ndimage.binary_dilation(wechsel, iterations=band)
                & maske & (H > 0.0))
    innen_band = (maske & (H > 0.0) & (np.max(gewichte, axis=0) > 0.85)
                 & ~naht_band)

    if naht_band.sum() < 50 or innen_band.sum() < 50:
        # Bei 256 px sollte das nicht passieren - schlaegt es doch zu, ist
        # das selbst ein Befund (die Kontinentform hat sich stark veraendert)
        # und wird als Fehler gemeldet statt stillschweigend uebersprungen.
        fehler.append("zu wenig Flaeche fuer die Nahtpruefung bei %dpx "
                      "(Naht %d px, Innen %d px) - Kontinentform hat sich "
                      "vermutlich veraendert" % (SIZE, naht_band.sum(),
                                                 innen_band.sum()))
        print("2. Nahtpruefung ... FEHLER (zu wenig Flaeche)")
    else:
        schlimmste = float(np.percentile(steig[naht_band], 99.5))
        innen_max = float(np.percentile(steig[innen_band], 99.5))
        ok = schlimmste <= 1.25 * innen_max
        print("2. Nahtpruefung: Grenzen %.3f gegen %.3f im Inneren "
              "(+25 %% erlaubt) ... %s"
              % (schlimmste, innen_max, "ok" if ok else "FEHLER"))
        if not ok:
            fehler.append("Regionsgrenzen sind deutlich steiler als die "
                          "Regionen selbst (%.3f gegen %.3f) - voller Befund "
                          "und Zahlen in smoke_test_regionen_welt.py (Eichung)"
                          % (schlimmste, innen_max))

    print()
    if fehler:
        print("NICHT IN ORDNUNG - %d Befunde:" % len(fehler))
        for eintrag in fehler:
            print("   %s" % eintrag)
        return 1
    print("Keine Naht auffaellig - fuer die volle Kalibrierung siehe "
          "die naechtliche smoke_test_regionen_welt.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(lauf())
