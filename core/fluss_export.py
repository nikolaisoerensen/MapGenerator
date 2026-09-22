"""
Pfad: core/fluss_export.py

Flussnetz als Linienzuege - Ticket #37.

BEFUND (Ticket #37): das Flussnetz wird in flussnetz() als Baum berechnet
(netz["punkte"], netz["eltern"], netz["flaeche"], ...), aber nach der
Rasterisierung in _weltfluesse() weggeworfen - es ueberlebt nur als
10.4 m/Pixel-Raster (fluss_maske). Ein Fluss, der im Spiel nur als Raster
existiert, ist eine Treppe: fuer scharfe Linien braucht Godot/Terrain3D
den Graphen selbst.

ENTSCHEIDUNG (Ticket #20, Frage 20.4): der Baum bleibt erhalten und wird
zusaetzlich als Linienzuege exportiert - NICHT anstelle des Rasters
(das Raster bleibt fuer Anzeige und Talgrabung), sondern zusaetzlich dazu.

WIEDERVERWENDUNG ("keine vierte Wahrheit"):
  - fluss_segmente() (core/fluss_sinuositaet.py, Ticket #33) zerlegt den
    Baum in maximale Ketten gleicher Strahler-Ordnung - genau die Einheit,
    die als "ein Fluss" exportiert werden soll.
  - kantenpunkte() (core/terrain_weltfluesse.py) ist DIESELBE Catmull-Rom-
    Kurve, die auch die Anzeige-Maske, river_water und taeler_eingraben()
    zeichnen. Kein eigener Kurvenalgorithmus hier.

WARUM DIE BREITENFORMEL TROTZDEM DUPLIZIERT IST:
  taeler_eingraben() (core/terrain_weltfluesse.py) berechnet dieselbe
  Formel (Glaettung -> gebiet -> anteil -> breite_feld), gibt aber nur ein
  fertig gemaltes Pixel-Raster zurueck, keine Zwischenwerte je Knoten/Kante.
  taeler_eingraben() selbst NICHT anzufassen (regressionsempfindlich, siehe
  CLAUDE.md "Gelaendeaenderungen verstimmen zuerst die Regionseichung").
  Deshalb rechnet dieses Modul dieselbe Formel eigenstaendig, aber MIT
  den importierten Konstanten (TALBREITE_UNTERGRENZE, TALBREITE_EXPONENT,
  BEZUGSFORM_M) statt eigener Kopien - Drift zwischen Talform und
  exportierter Breite ist damit ausgeschlossen, auch wenn die Konstanten
  sich spaeter aendern.

ABSICHTLICH NICHT verwendet: die Anzeige-Breite aus _weltfluesse()
(FLUSS_BREITE_GRUND_PX/FLUSS_BREITE_JE_DEKADE_PX, log-skaliert auf
netz["flaeche"]) - dort im Code ausdruecklich als "ein reines
Anzeigeprodukt" kommentiert, physikalisch nicht die Talbreite. Fuer den
Export zaehlt die tatsaechliche Talbreite aus der Gelaendeformel.
"""
from typing import Any, Dict, List

import logging

import numpy as np

from core.fluss_sinuositaet import fluss_segmente
from core.terrain_weltfluesse import (
    hauptkinder,
    kantenpunkte,
    TALBREITE_UNTERGRENZE,
    TALBREITE_EXPONENT,
    BEZUGSFORM_M,
)

logger = logging.getLogger(__name__)


def gebiet_je_knoten(eltern: np.ndarray, flaeche: np.ndarray, iterationen: int = 4) -> np.ndarray:
    """
    Funktionsweise: dieselbe 4-Iterationen-Glaettung des Einzugsgebiets wie
        taeler_eingraben() (core/terrain_weltfluesse.py, glatt_fl), hier
        eigenstaendig berechnet, weil taeler_eingraben() keine
        Zwischenwerte herausgibt (siehe Moduldocstring).
    Aufgabe: liefert je Knoten den normierten Flaechenanteil (0..1), aus
        dem breite_m_je_knoten() die Talbreite ableitet.
    Parameter: eltern - Elternindex je Knoten (-1 = Muendung/Wurzel)
    Parameter: flaeche - akkumulierte Wasser-/Einzugsflaeche je Knoten
    Parameter: iterationen - Anzahl Glaettungsschritte (Default 4, wie im Original)
    Returns: np.ndarray - normiertes Einzugsgebiet je Knoten, Werte in [0, 1]
    """
    eltern = np.asarray(eltern)
    glatt = np.asarray(flaeche, dtype=np.float64).copy()
    for _ in range(iterationen):
        neu = glatt.copy()
        for i in range(len(glatt)):
            if eltern[i] >= 0:
                neu[i] = 0.5 * glatt[i] + 0.5 * glatt[eltern[i]]
        glatt = neu
    spitze = float(glatt.max()) if len(glatt) else 0.0
    return glatt / max(spitze, 1.0)


def breite_m_je_knoten(netz: Dict[str, Any], felder: Dict[str, Any], mpp: float,
                        breite_faktor: float, abstand_makro_m: float) -> np.ndarray:
    """
    Funktionsweise: Talbreite in METERN je Knoten - dieselbe Formel wie
        taeler_eingraben() (dort als Pixel-Raster breite_feld, hier direkt
        am Knoten ausgewertet statt an jedem Spline-Abtastpunkt).
    Aufgabe: liefert die Breite, die fluss_linien() pro Segment mittelt und
        als "breite_m" exportiert.
    Parameter: netz - Ausgabe von flussnetz() (braucht punkte, eltern, flaeche)
    Parameter: felder - Regionsfelder-Dict (braucht formgroesse_m als Raster)
    Parameter: mpp - Meter pro Pixel der aktuellen Kartengroesse
    Parameter: breite_faktor - Regler wie in taeler_eingraben()
    Parameter: abstand_makro_m - Makro-Knotenabstand in Metern (STUFEN[0][1])
    Returns: np.ndarray - Talbreite in Metern je Knoten, mindestens 2.5 Pixel breit
    """
    punkte = np.asarray(netz["punkte"], dtype=np.float64)
    eltern = np.asarray(netz["eltern"])
    flaeche = netz["flaeche"]
    formgroesse = np.asarray(felder["formgroesse_m"])
    size = formgroesse.shape[0]
    yi = np.clip(np.round(punkte[:, 0]).astype(int), 0, size - 1)
    xi = np.clip(np.round(punkte[:, 1]).astype(int), 0, size - 1)
    formgroesse_m = formgroesse[yi, xi]

    gebiet = gebiet_je_knoten(eltern, flaeche)
    anteil = TALBREITE_UNTERGRENZE + (1.0 - TALBREITE_UNTERGRENZE) * gebiet ** TALBREITE_EXPONENT
    breite_feld_m = float(breite_faktor) * float(abstand_makro_m) * (formgroesse_m / BEZUGSFORM_M)
    return np.maximum(anteil * breite_feld_m, 2.5 * float(mpp))


def fluss_linien(netz: Dict[str, Any], strahler: np.ndarray, felder: Dict[str, Any],
                  mpp: float, breite_faktor: float = 0.35,
                  abstand_makro_m: float = 1200.0) -> List[Dict[str, Any]]:
    """
    Funktionsweise: zerlegt das Flussnetz ueber fluss_segmente() in Ketten
        gleicher Strahler-Ordnung, zieht je Kante dieselbe Catmull-Rom-Kurve
        wie die Anzeige (kantenpunkte()) und haengt die Kurvenstuecke zu
        einer durchgehenden Punktkette pro Segment zusammen.
    Aufgabe: liefert die Datenstruktur, die DataLODManager unter
        "river_lines" speichert und map_export.vektordaten() in Meter
        umrechnet und als "fluesse" exportiert (Ticket #37).
    Parameter: netz - Ausgabe von flussnetz()
    Parameter: strahler - Strahler-Ordnung je Knoten (terrain_river_network.strahler_order)
    Parameter: felder - Regionsfelder-Dict (fuer die Breitenformel)
    Parameter: mpp - Meter pro Pixel der aktuellen Kartengroesse
    Parameter: breite_faktor - Regler wie in taeler_eingraben() (Default 0.35)
    Parameter: abstand_makro_m - Makro-Knotenabstand in Metern (Default 1200.0, STUFEN[0][1])
    Returns: List[Dict] - je Segment {"punkte": [[x_px,y_px],...] Quelle->Muendung,
        "ordnung": int (Strahler-Ordnung des Segments), "breite_m": float (Mittelwert)}.
        Punkte bleiben in PIXELN (wie "wege"/"seewege" in vektordaten()) - die
        Meter-Umrechnung passiert zentral ueber map_export._pfad_in_meter().
    """
    eltern = np.asarray(netz["eltern"])
    if len(eltern) == 0:
        logger.warning("fluss_linien: leeres Flussnetz (netz['eltern'] ist leer) - kein Export")
        return []

    punkte = np.asarray(netz["punkte"], dtype=np.float64)
    flaeche = netz["flaeche"]
    strahler = np.asarray(strahler)
    kinder = hauptkinder(eltern, flaeche)
    breite_je_knoten = breite_m_je_knoten(netz, felder, mpp, breite_faktor, abstand_makro_m)

    segmente = fluss_segmente(eltern, strahler)
    linien: List[Dict[str, Any]] = []
    uebersprungen = 0
    for kette in segmente:
        if len(kette) < 2:
            uebersprungen += 1
            continue
        weg_px = [punkte[kette[0]]]
        for idx in range(1, len(kette)):
            i = kette[idx - 1]  # stromauf (Kind)
            e = kette[idx]      # stromab (Elter)
            strecke = float(np.linalg.norm(punkte[i] - punkte[e]))
            schritte = max(int(strecke * 2.0), 2)
            # kantenpunkte() liefert pk[e] -> pk[i], also stromab -> stromauf
            # (siehe Moduldocstring von taeler_eingraben) - umgedreht, damit
            # die Kette durchgehend Quelle -> Muendung bleibt wie fluss_segmente().
            segment_px = kantenpunkte(punkte, eltern, kinder, e, i, schritte)[::-1]
            weg_px.extend(segment_px[1:])
        weg_px_arr = np.asarray(weg_px, dtype=np.float64)
        breiten = breite_je_knoten[kette]
        linien.append({
            "punkte": [[float(p[1]), float(p[0])] for p in weg_px_arr],  # [x_px, y_px]
            "ordnung": int(strahler[kette[0]]),
            "breite_m": float(np.mean(breiten)),
        })

    if uebersprungen:
        logger.info(f"fluss_linien: {uebersprungen} Ein-Knoten-Segmente uebersprungen (keine Linie moeglich)")
    return linien
