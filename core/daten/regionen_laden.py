"""
Path: core/daten/regionen_laden.py

Laedt core/daten/regionen_welt.toml und baut daraus REGIONEN,
KUESTEN_ARCHETYPEN, NIEDERSCHLAG_ZIEL und KLIMA_ZIEL - dieselben Namen, die
core/terrain_weltkarte.py vor diesem Umzug als literale Konstanten enthielt.
NIEDERSCHLAG_ZIEL/KLIMA_ZIEL stehen nicht mehr eigens in der TOML-Datei,
sondern werden hier aus den Feldern niederschlag_mm/temp_mittel_m0/
temp_spanne abgeleitet, die schon je Region drinstehen (siehe Kommentar am
Kopf der TOML-Datei: beide Dicts waren bit-identisch mit diesen Feldern).

Kein stiller Rueckfall: fehlt ein Feld oder ein Gitterplatz, wirft der
direkte Dict-/Listenzugriff KeyError statt einen Default einzusetzen (siehe
tests/smoke_test_regionen_datei_vollstaendig.py).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DATEI = Path(__file__).parent / "regionen_welt.toml"

_REGION_FELDER = (
    "farbe", "volk", "bemerkung", "hoehe_m", "relief_m", "formgroesse_m",
    "rauheit", "potenz", "wasser_soll", "flaeche_soll", "kuestenform",
    "temp_mittel_m0", "temp_spanne", "niederschlag_mm", "wind_mittel_ms",
    "hang_trockenheit", "talform",
)

_ARCHETYP_FELDER = (
    "name", "hoehe_faktor", "winkel_grad", "kantig", "strand_anteil",
    "max_anteil", "reichweite_km",
)


def _rohdaten() -> Dict[str, Any]:
    with open(_DATEI, "rb") as datei:
        return tomllib.load(datei)


def _regionen_bauen(roh: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    tabellen = roh["regionen"]
    gitter: List[List[Optional[Dict[str, Any]]]] = [[None, None, None] for _ in range(3)]
    for name, eintrag in tabellen.items():
        zeile = eintrag["zeile"]
        spalte = eintrag["spalte"]
        region: Dict[str, Any] = {"name": name}
        for feld in _REGION_FELDER:
            region[feld] = eintrag[feld]
        gitter[zeile][spalte] = region
    for zeile in gitter:
        for platz in zeile:
            if platz is None:
                raise KeyError(
                    "regionen_welt.toml deckt nicht alle neun Gitterplaetze "
                    "(zeile/spalte 0..2) ab"
                )
    return gitter  # type: ignore[return-value]


def _archetypen_bauen(roh: Dict[str, Any]) -> Dict[str, Tuple[Dict[str, Any], ...]]:
    tabellen = roh["kuesten_archetypen"]
    ergebnis: Dict[str, Tuple[Dict[str, Any], ...]] = {}
    for name, eintraege in tabellen.items():
        archetypen = tuple(
            {feld: eintrag[feld] for feld in _ARCHETYP_FELDER} for eintrag in eintraege
        )
        ergebnis[name] = archetypen
    return ergebnis


def laden() -> Tuple[
    List[List[Dict[str, Any]]],
    Dict[str, Tuple[Dict[str, Any], ...]],
    Dict[str, float],
    Dict[str, Tuple[float, float]],
]:
    """Liest regionen_welt.toml und liefert REGIONEN, KUESTEN_ARCHETYPEN,
    NIEDERSCHLAG_ZIEL, KLIMA_ZIEL in dieser Reihenfolge."""
    roh = _rohdaten()
    regionen = _regionen_bauen(roh)
    kuesten_archetypen = _archetypen_bauen(roh)

    niederschlag_ziel: Dict[str, float] = {}
    klima_ziel: Dict[str, Tuple[float, float]] = {}
    for zeile in regionen:
        for region in zeile:
            name = region["name"]
            niederschlag_ziel[name] = region["niederschlag_mm"]
            klima_ziel[name] = (region["temp_mittel_m0"], region["temp_spanne"])

    return regionen, kuesten_archetypen, niederschlag_ziel, klima_ziel
