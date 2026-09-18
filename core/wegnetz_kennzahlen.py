"""
Path: core/wegnetz_kennzahlen.py

Kennzahlen fuer das aus core/settlement_generator.py.calculate_road_network()
entstandene Wegenetz (Ticket #41, Abnahmekriterium 3: Gesamtlaenge des Netzes,
Anteil je Wegart, mittlerer Umwegfaktor gegen die Luftlinie, Anzahl Knoten).

Reine Funktionen auf den Rueckgabewerten von calculate_road_network() (die
Liste `roads`, das Set `gebaut`) plus den seit Ticket #41 auf der
SettlementGenerator-Instanz abgelegten Attributen `letztes_nutzungsfeld`
(Nutzungs-Feld) und `wegarten` (die geladene Tabelle aus
core/daten/wegarten.toml) - kein eigener Zustand, kein Seiteneffekt, damit
sie sich sowohl in tests/smoke_test_wegarten_kalibrierung.py als auch spaeter
in einer Anzeige verwenden lassen.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from core.daten.wegarten_laden import lade_wegarten, klassifiziere


def _segmente(pfad):
    """Liefert (kanonischer_key, laenge, mittelpunkt_xy) je Teilstueck eines
    Pfades.

    Der Key rundet auf ganze Pixel und ist ungerichtet (frozenset der beiden
    gerundeten Endpunkte) - das ist die Deduplizierung fuer sich
    ueberlappende Wege: baut eine zweite Route denselben Streckenabschnitt
    (Buendelung ueber ein gemeinsames Wegstueck, siehe Ticket #41
    "Wegersparnis"-Effekt), zaehlt dieser Abschnitt fuer die Gesamtlaenge nur
    EINMAL, egal von wie vielen Pfaden er durchlaufen wird.
    """
    for (x0, y0), (x1, y1) in zip(pfad[:-1], pfad[1:]):
        p0 = (round(float(x0)), round(float(y0)))
        p1 = (round(float(x1)), round(float(y1)))
        if p0 == p1:
            continue
        key = frozenset((p0, p1))
        laenge = math.hypot(x1 - x0, y1 - y0)
        mitte = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        yield key, laenge, mitte


def netzlaenge_und_anteile(roads, nutzung, wegarten=None):
    """Gesamtlaenge des Wegenetzes und ihr Anteil je Wegart.

    `roads` ist die von calculate_road_network() gebaute Liste geglaetteter
    Pfade (je eine Liste/ein Array von (x, y)-Punkten). `nutzung` ist das
    Nutzungsfeld derselben Instanz nach demselben Aufruf
    (`gen.letztes_nutzungsfeld`).

    Rueckgabe: {"gesamt": float, "je_wegart": {wegart_id: float},
    "ohne_wegart": float} - Laengen in denselben Einheiten wie die
    Pfadkoordinaten (Pixel). "ohne_wegart" faengt Segmente ab, deren
    Nutzungswert unter der niedrigsten Schwelle liegt (z.B. Randrundung);
    bei einer sinnvollen Kalibrierung sollte das nahe 0 sein, denn jedes
    Segment wurde ja mindestens einmal befahren, um in `roads` zu stehen.
    """
    wegarten = wegarten if wegarten is not None else lade_wegarten()
    gesehen = {}
    for pfad in roads:
        for key, laenge, (mx, my) in _segmente(pfad):
            if key in gesehen:
                continue
            xi, yi = int(round(mx)), int(round(my))
            wert = 0.0
            if 0 <= yi < nutzung.shape[0] and 0 <= xi < nutzung.shape[1]:
                wert = float(nutzung[yi, xi])
            gesehen[key] = (laenge, wert)

    gesamt = 0.0
    je_wegart = defaultdict(float)
    ohne_wegart = 0.0
    for laenge, wert in gesehen.values():
        gesamt += laenge
        wegart_id = klassifiziere(wert, wegarten)
        if wegart_id is None:
            ohne_wegart += laenge
        else:
            je_wegart[wegart_id] += laenge
    return {"gesamt": gesamt, "je_wegart": dict(je_wegart), "ohne_wegart": ohne_wegart}


def mittlerer_umwegfaktor(roads):
    """Mittelwert aus (tatsaechliche Pfadlaenge / Luftlinie) ueber alle
    `roads`.

    Ein Wert nahe 1.0 heisst kaum Umweg; deutlich groesser als 1 heisst, die
    Routen nehmen lieber ein bestehendes Wegbuendel als die Direktverbindung
    - genau der im Ticket beschriebene Effekt, der ein PLAUSIBLES Mass
    braucht, keine unbemerkte Ueberraschung.

    Pfade ohne messbare Luftlinie (Start und Ziel fallen zusammen) werden
    uebersprungen statt mit Faktor 1.0 gezaehlt - das wuerde den Mittelwert
    unbemerkt verwaessern.
    """
    faktoren = []
    for pfad in roads:
        if len(pfad) < 2:
            continue
        laenge = sum(math.hypot(x1 - x0, y1 - y0)
                     for (x0, y0), (x1, y1) in zip(pfad[:-1], pfad[1:]))
        x0, y0 = pfad[0]
        x1, y1 = pfad[-1]
        luftlinie = math.hypot(x1 - x0, y1 - y0)
        if luftlinie <= 1e-9:
            continue
        faktoren.append(laenge / luftlinie)
    return float(np.mean(faktoren)) if faktoren else float("nan")


def knotenzahl(settlements, gebaut):
    """Gesamtzahl der Siedlungen und Anzahl davon, die im fertigen
    Wegenetz an mindestens einer gebauten Verbindung haengen.

    `gebaut` ist das {frozenset({id_a, id_b})}-Set aus
    calculate_road_network() - es waechst sowohl beim paarweisen Netzbau als
    auch beim bedarfsgetriebenen Ausbau (_netz_nach_bedarf_ausbauen), umfasst
    also das fertige Netz vollstaendig.
    """
    verbunden_ids = set()
    for paar in gebaut:
        verbunden_ids.update(paar)
    gesamt = len(settlements)
    verbunden = sum(1 for s in settlements if s.location_id in verbunden_ids)
    return {"gesamt": gesamt, "verbunden": verbunden}
