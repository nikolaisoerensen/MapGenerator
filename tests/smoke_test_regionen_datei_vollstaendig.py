"""
Path: tests/smoke_test_regionen_datei_vollstaendig.py

Deckt core/daten/regionen_welt.toml wirklich alle neun Regionen und alle
ihre Felder ab? (docs/OFFENE_PUNKTE.md #29, Abnahmekriterium 4)

Der Loader (core/daten/regionen_laden.py) hat bewusst KEINEN stillen
Rueckfall: fehlt ein Gitterplatz oder ein Feld, wirft er KeyError statt
einen Default einzusetzen. Dieser Test prueft direkt gegen die TOML-Datei
(nicht nur ueber den Loader), damit ein Test-Fehlschlag genau die fehlende
Region/das fehlende Feld benennt, statt nur "KeyError irgendwo".

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_regionen_datei_vollstaendig.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.daten.regionen_laden import _ARCHETYP_FELDER, _REGION_FELDER, _rohdaten, laden

ERWARTETE_REGIONEN = {
    "Clonagh", "Skerrheim", "Morobora",
    "Estrande", "Nevadin", "Nebelrode",
    "Samarcia", "Macchia", "Thalassia",
}
ERWARTETE_GITTERPLAETZE = {(z, s) for z in range(3) for s in range(3)}
ARCHETYPEN_JE_REGION = 3


def check(label, bedingung, zusatz=""):
    status = "OK" if bedingung else "FAIL"
    print(f"[{status}] {label}{(' - ' + zusatz) if zusatz else ''}")
    return bool(bedingung)


def run_regionen_vollstaendig():
    roh = _rohdaten()
    tabellen = roh["regionen"]

    ok = check("alle neun Regionsnamen vorhanden",
               set(tabellen.keys()) == ERWARTETE_REGIONEN,
               f"vorhanden: {sorted(tabellen.keys())}")

    fehlende_felder = []
    for name, eintrag in tabellen.items():
        for feld in _REGION_FELDER + ("zeile", "spalte"):
            if feld not in eintrag:
                fehlende_felder.append(f"{name}.{feld}")
    ok &= check("jede Region hat alle Felder", not fehlende_felder,
                ", ".join(fehlende_felder) if fehlende_felder else
                f"{len(_REGION_FELDER) + 2} Felder je Region")

    gitterplaetze = {(e["zeile"], e["spalte"]) for e in tabellen.values()
                      if "zeile" in e and "spalte" in e}
    ok &= check("jeder Gitterplatz 0..2 x 0..2 genau einmal belegt",
                gitterplaetze == ERWARTETE_GITTERPLAETZE,
                f"belegt: {sorted(gitterplaetze)}")
    return ok


def run_archetypen_vollstaendig():
    roh = _rohdaten()
    tabellen = roh["kuesten_archetypen"]

    ok = check("Kuestenarchetypen fuer alle neun Regionen vorhanden",
               set(tabellen.keys()) == ERWARTETE_REGIONEN,
               f"vorhanden: {sorted(tabellen.keys())}")

    falsche_anzahl = [name for name, e in tabellen.items()
                       if len(e) != ARCHETYPEN_JE_REGION]
    ok &= check(f"je Region genau {ARCHETYPEN_JE_REGION} Archetypen",
                not falsche_anzahl,
                ", ".join(f"{n} hat {len(tabellen[n])}" for n in falsche_anzahl)
                if falsche_anzahl else "3/3 je Region")

    fehlende_felder = []
    for name, eintraege in tabellen.items():
        for i, eintrag in enumerate(eintraege):
            for feld in _ARCHETYP_FELDER:
                if feld not in eintrag:
                    fehlende_felder.append(f"{name}[{i}].{feld}")
    ok &= check("jeder Archetyp hat alle Felder", not fehlende_felder,
                ", ".join(fehlende_felder) if fehlende_felder else
                f"{len(_ARCHETYP_FELDER)} Felder je Archetyp")
    return ok


def run_loader_laeuft_durch():
    """laden() selbst ist der scharfste Test: er wirft KeyError bei jeder
    Luecke, die die beiden Pruefungen oben eventuell nicht abdecken."""
    try:
        regionen, kuesten_archetypen, niederschlag_ziel, klima_ziel = laden()
    except KeyError as exc:
        return check("laden() wirft keinen KeyError", False, str(exc))

    ok = check("laden() liefert 3x3-Gitter", len(regionen) == 3
               and all(len(z) == 3 for z in regionen))
    ok &= check("laden() liefert alle neun Namen",
                {r["name"] for z in regionen for r in z} == ERWARTETE_REGIONEN)
    ok &= check("NIEDERSCHLAG_ZIEL/KLIMA_ZIEL fuer alle neun Regionen",
                set(niederschlag_ziel.keys()) == ERWARTETE_REGIONEN
                and set(klima_ziel.keys()) == ERWARTETE_REGIONEN)
    return ok


def main():
    print("=" * 78)
    print("REGIONEN-DATEI VOLLSTAENDIGKEIT (core/daten/regionen_welt.toml)")
    print("=" * 78)
    print()

    ergebnis = {}
    for titel, fn in (
        ("Regionen", run_regionen_vollstaendig),
        ("Kuestenarchetypen", run_archetypen_vollstaendig),
        ("Loader", run_loader_laeuft_durch),
    ):
        print(f"--- {titel} ---")
        ergebnis[titel] = fn()
        print()

    print("=" * 78)
    gut = sum(1 for v in ergebnis.values() if v)
    print(f"{gut}/{len(ergebnis)} Gruppen gruen")
    return 0 if gut == len(ergebnis) else 1


if __name__ == "__main__":
    sys.exit(main())
