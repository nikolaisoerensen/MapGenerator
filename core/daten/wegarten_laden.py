"""
Path: core/daten/wegarten_laden.py

Laedt core/daten/wegarten.toml (Ticket #41) und stellt die Wegarten-Tabelle
core/settlement_generator.py.calculate_road_network() zur Verfuegung. Muster
wie core/daten/regionen_laden.py (Ticket #29): tomllib aus der
Standardbibliothek, keine neue Abhaengigkeit, kein stiller Rueckfall - fehlt
oder verstoesst etwas gegen die Reihenfolge-Regel, gibt es einen Fehler statt
eines plausibel aussehenden Default.

Die zentrale Funktion fuer den Aufrufer ist `wegart_faktor_feld(nutzung)`:
sie macht aus einem Nutzungs-Feld (wie oft pro Pixel ein Weg mitbenutzt
wurde) ein Kostenfaktor-Feld derselben Form, das auf das Grundkostenfeld aus
bau_kostenfeld() multipliziert wird.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_TOML_PFAD = Path(__file__).parent / "wegarten.toml"

_BEKANNTE_FELDER = {"id", "name", "reihenfolge", "schwelle_nutzung", "kostenfaktor"}


@dataclass(frozen=True)
class Wegart:
    """Eine Stufe der Wegarten-Staffelung (Trampelpfad/Karrenweg/Strasse/...).

    `zusatz` faengt alle TOML-Schluessel auf, die core/daten/wegarten.toml
    ausser den hier bekannten Feldern noch traegt - Platz fuer Ticket #42
    (Bruecken/Uferwege), ohne dass diese Klasse dafuer schon etwas wissen muss.
    """

    id: str
    name: str
    reihenfolge: int
    schwelle_nutzung: float
    kostenfaktor: float
    zusatz: dict = field(default_factory=dict)


def _pruefe_reihenfolge(wegarten: list[Wegart]) -> None:
    """Erzwingt, dass Nutzungsschwellen steigen und Kostenfaktoren fallen.

    Das ist keine Formalitaet: die Rueckkopplung des Tickets funktioniert nur,
    wenn jede hoehere Wegart wirklich billiger ist als die vorige. Eine
    Verletzung hier ist ein Kalibrierungsfehler in wegarten.toml, kein
    Randfall, der still uebergangen werden darf.
    """
    for voriger, aktueller in zip(wegarten, wegarten[1:]):
        if aktueller.schwelle_nutzung <= voriger.schwelle_nutzung:
            raise ValueError(
                f"wegarten.toml: schwelle_nutzung muss steigen, aber "
                f"'{aktueller.id}' ({aktueller.schwelle_nutzung}) folgt auf "
                f"'{voriger.id}' ({voriger.schwelle_nutzung})"
            )
        if aktueller.kostenfaktor >= voriger.kostenfaktor:
            raise ValueError(
                f"wegarten.toml: kostenfaktor muss fallen, aber "
                f"'{aktueller.id}' ({aktueller.kostenfaktor}) ist nicht "
                f"billiger als '{voriger.id}' ({voriger.kostenfaktor})"
            )


def lade_wegarten(pfad: Path | None = None) -> list[Wegart]:
    """Laedt und validiert die Wegarten-Tabelle, aufsteigend nach `reihenfolge`.

    Wirft KeyError/ValueError statt still weiterzumachen - dieselbe Haltung
    wie core/daten/regionen_laden.py. Kein Cache: die Datei ist klein, und ein
    Cache wuerde eine Aenderung an wegarten.toml waehrend eines Testlaufs
    verschleiern.
    """
    datei = pfad or _TOML_PFAD
    with open(datei, "rb") as f:
        rohdaten = tomllib.load(f)

    eintraege = rohdaten["wegart"]
    if not eintraege:
        raise ValueError(f"{datei}: keine [[wegart]]-Eintraege gefunden")

    wegarten = []
    for eintrag in eintraege:
        zusatz = {k: v for k, v in eintrag.items() if k not in _BEKANNTE_FELDER}
        wegarten.append(Wegart(
            id=eintrag["id"],
            name=eintrag["name"],
            reihenfolge=eintrag["reihenfolge"],
            schwelle_nutzung=float(eintrag["schwelle_nutzung"]),
            kostenfaktor=float(eintrag["kostenfaktor"]),
            zusatz=zusatz,
        ))
    wegarten.sort(key=lambda w: w.reihenfolge)

    ids = [w.id for w in wegarten]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{datei}: doppelte id in [[wegart]]-Eintraegen: {ids}")

    _pruefe_reihenfolge(wegarten)
    return wegarten


def wegart_faktor_feld(nutzung: np.ndarray, wegarten: list[Wegart] | None = None) -> np.ndarray:
    """Kostenfaktor je Pixel, aus dem Nutzungs-Feld ueber die Wegarten-Stufen.

    `nutzung` ist ein Feld gleicher Form wie das Kostenfeld: wie oft eine
    Route genau diesen Pixel bereits mitbenutzt hat (0 = kein Weg). Pixel
    ohne Weg (nutzung <= 0) behalten den neutralen Faktor 1.0 - sie werden
    NICHT hier, sondern vom Aufrufer per `np.where` gegen das unveraenderte
    Grundkostenfeld ausgetauscht (siehe calculate_road_network()), damit ein
    Faktor > 1.0 (gibt es aktuell nicht, waere aber zulaessig) nicht versehentlich
    Gelaende ohne Weg verteuert.

    Bei mehreren erreichten Schwellen gewinnt die hoechste (die Wegarten sind
    nach `reihenfolge` aufsteigend sortiert, `schwelle_nutzung` steigt damit
    zwingend mit).
    """
    wegarten = wegarten if wegarten is not None else lade_wegarten()
    faktor = np.ones_like(nutzung, dtype=np.float64)
    for w in wegarten:
        faktor = np.where(nutzung >= w.schwelle_nutzung, w.kostenfaktor, faktor)
    return faktor


def klassifiziere(nutzungswert: float, wegarten: list[Wegart] | None = None) -> str | None:
    """Wegart-id fuer einen einzelnen Nutzungswert, oder None ohne Weg.

    Fuer die Kennzahlen-Auswertung (core/wegnetz_kennzahlen.py): dort wird
    jeder Wegabschnitt einzeln klassifiziert, um die Laenge je Wegart zu
    summieren.
    """
    wegarten = wegarten if wegarten is not None else lade_wegarten()
    if nutzungswert <= 0:
        return None
    ergebnis = None
    for w in wegarten:
        if nutzungswert >= w.schwelle_nutzung:
            ergebnis = w.id
    return ergebnis
