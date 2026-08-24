"""
Path: managers/teilschritte.py

Zeitmessung INNERHALB eines Calculator-Knotens - fuer Log und Ladebalken.

WARUM DAS NOETIG WAR. Das Pipeline-Log (2026-08-11) misst jeden der 38
Knoten einzeln, und das reichte, um die Fresser zu FINDEN. Es reicht nicht,
um sie zu VERSTEHEN: `terrain.redistribution | GPU | Dauer 61.117s` sagt
nur, dass irgendwo in einem Aufrufbaum aus einem Dutzend Funktionen eine
Minute verschwindet. Welche, steht nirgends.

Vier Knoten machen 84 % der Ladezeit aus (tools/pipeline_kritischer_pfad.py).
Ohne Aufschluesselung ist jede Optimierung an ihnen geraten.

BAUART. Ein Kontextmanager je Teilschritt. Beim Betreten geht der
Ladebalken weiter, beim Verlassen wird die Dauer notiert; am Ende schreibt
`bericht()` eine Tabelle in DASSELBE Log wie die Knotenzeilen, damit beides
im selben Konsolenlauf untereinander steht.

AUSGESCHALTET IST DER NORMALFALL. `Teilschritte(None)` und der Aufruf ohne
Objekt liefern ein No-Op - Tools und Smoke-Tests rufen dieselben Funktionen
auf und sollen davon nichts merken. Deshalb nimmt jede gemessene Funktion
den Parameter als `schritte=None` entgegen und ruft ihn ueber `_schritt()`,
das mit None umgehen kann.

GEWICHTE. Der Ladebalken braucht eine Vorstellung davon, wie lang ein
Schritt dauert, sonst springt er ungleichmaessig. Die Gewichte im `plan`
sind die GEMESSENEN Anteile aus dem Lauf vom 2026-08-22 - kein Ratewerk,
und beim naechsten groesseren Umbau nachzufuehren.
"""

import logging
import time
from contextlib import contextmanager

_LOGGER = logging.getLogger("Pipeline")


class Teilschritte:
    """
    Misst die Teilschritte eines Knotens.

    knoten       - Name fuer die Logzeilen, z.B. "terrain.redistribution"
    fortschritt  - Callback (phase, prozent, nachricht), die Signatur aller
                   core/*_generator.py `_update_progress`. None schaltet den
                   Ladebalkenanteil ab, die Messung laeuft trotzdem.
    von, bis     - der Prozentbereich, den dieser Knoten im Balken belegt
    plan         - [(name, gewicht), ...] fuer die Verteilung im Balken
    """

    def __init__(self, knoten, fortschritt=None, von=0, bis=100, plan=None):
        self.knoten = knoten
        self._fortschritt = fortschritt
        self._von, self._bis = float(von), float(bis)
        self._plan = list(plan or [])
        self._gewicht_gesamt = sum(g for _n, g in self._plan) or 1.0
        self._erledigt = 0.0
        self._zeiten = []
        self._t_start = time.perf_counter()

    def _anteil(self, name):
        for n, g in self._plan:
            if n == name:
                return g
        # Nicht im Plan: als ein Zwanzigstel des Gesamtgewichts fuehren,
        # damit ein vergessener Schritt den Balken nicht einfriert.
        return self._gewicht_gesamt * 0.05

    @contextmanager
    def schritt(self, name, text=None):
        gewicht = self._anteil(name)
        if self._fortschritt is not None:
            prozent = self._von + (self._bis - self._von) * (
                self._erledigt / self._gewicht_gesamt)
            try:
                self._fortschritt(self.knoten, int(round(prozent)),
                                  text or name)
            except Exception:                                   # noqa: BLE001
                pass
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._zeiten.append((name, time.perf_counter() - t0))
            self._erledigt += gewicht

    def bericht(self):
        """Eine Tabelle je Knoten, absteigend nach Dauer sortiert."""
        if not self._zeiten:
            return
        gesamt = time.perf_counter() - self._t_start
        gemessen = sum(d for _n, d in self._zeiten)
        _LOGGER.info("--- %s: %d Teilschritte, %.3fs gemessen von %.3fs ---",
                     self.knoten, len(self._zeiten), gemessen, gesamt)
        for name, dauer in sorted(self._zeiten, key=lambda x: -x[1]):
            _LOGGER.info("      %-38s %8.3fs  %5.1f%%",
                         name, dauer, 100.0 * dauer / max(gesamt, 1e-9))
        rest = gesamt - gemessen
        if rest > 0.05:
            _LOGGER.info("      %-38s %8.3fs  %5.1f%%  (nicht zugeordnet)",
                         "[Rest]", rest, 100.0 * rest / max(gesamt, 1e-9))

    # Die gemessenen Paare, fuer Auswertung im Test statt im Log.
    @property
    def zeiten(self):
        return list(self._zeiten)


@contextmanager
def _leer():
    yield


def schritt(teilschritte, name, text=None):
    """
    Ein Teilschritt, der auch ohne Messobjekt funktioniert.

    So kann jede gemessene Funktion `with schritt(schritte, "name"):`
    schreiben, ohne jedes Mal auf None zu pruefen - und ein Tool, das
    dieselbe Funktion ohne Messung aufruft, zahlt nichts dafuer.
    """
    if teilschritte is None:
        return _leer()
    return teilschritte.schritt(name, text)
