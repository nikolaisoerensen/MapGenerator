"""
Path: tests/smoke_test_parameter_eindeutig.py

Prueft, dass kein Parameterschluessel unbemerkt in zwei Konfigurationsklassen
steht (docs/OFFENE_PUNKTE.md 12.3) und dass die Sperrliste die unerreichbaren
Regler wirklich benennt (12.4).

WARUM ES DIESEN TEST GIBT

`erosion_strength` war ZWEIMAL definiert: in class EROSION (0.0-2.0, Vorgabe
0.5) und in class WATER (0.1-5.0, Vorgabe 2.5). Beide schreiben nach
`parameters['erosion_strength']`; welcher gewinnt, haengt an der Reihenfolge
der Zusammenstellung. Nichts daran war zu sehen - kein Fehler, keine Warnung,
nur zwei Spannen fuer denselben Namen. Genau die Sorte Befund, die spaeter
als unerklaerliches Verhalten zurueckkommt.

Der Eintrag selbst BLEIBT (core/water_generator.py liest die Schluessel
weiter ueber `parameters.get(...)`, ein Loeschen wuerde nur die dokumentierte
Spanne entfernen). Was sich aendert: er ist jetzt in `DOPPELTE_SCHLUESSEL`
angemeldet. Dieser Test schlaegt fehl, sobald ein NEUER, nicht angemeldeter
Doppelschluessel dazukommt.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_parameter_eindeutig.py
"""
import sys

sys.path.insert(0, ".")

from gui.config import value_default as vd

# Die Klassen, aus denen die Reiter ihre Regler zusammenstellen.
KONFIG_KLASSEN = ("TERRAIN", "GEOLOGY", "SETTLEMENT", "WEATHER",
                  "RIVER_NETWORK", "EROSION_FILTER", "EROSION", "WATER",
                  "BIOME")

# Schluessel, die laut 12.4 definiert sind, aber in keinem Reiter stehen.
# Sie MUESSEN in der Sperrliste eine Begruendung tragen - sonst steht der
# Nutzer vor einem Regler, der nichts tut, und kann Absicht nicht von
# Fehler unterscheiden.
UNERREICHBAR = ("erosion_passes", "sediment_capacity_factor",
                "settling_velocity", "thermal_erosion_strength",
                "frequency")


def check(label, bedingung, zusatz=""):
    status = "OK" if bedingung else "FAIL"
    print(f"[{status}] {label}{(' - ' + zusatz) if zusatz else ''}")
    return bool(bedingung)


def parameter_je_klasse():
    """
    {schluessel: {klasse: konfig}} - ein Parameter ist ein Klassenattribut in
    GROSSSCHREIBUNG, dessen Wert ein dict mit 'default' ist.
    """
    gefunden = {}
    for name in KONFIG_KLASSEN:
        klasse = getattr(vd, name, None)
        if klasse is None:
            continue
        for attribut in vars(klasse):
            if attribut.startswith("_") or not attribut.isupper():
                continue
            wert = getattr(klasse, attribut)
            if isinstance(wert, dict) and "default" in wert:
                gefunden.setdefault(attribut.lower(), {})[name] = wert
    return gefunden


def registrierte_schluessel():
    """
    {parameterschluessel: {"KLASSE.ATTRIBUT", ...}} aus den Reitern.

    Das ist die EIGENTLICHE Wahrheit: ein Reiter registriert seine Regler als
    ("schluessel", "Beschriftung", KLASSE.ATTRIBUT). Der Attributname allein
    sagt nichts - `EROSION_FILTER.OCTAVES` laeuft unter dem Schluessel
    `erosion_filter_octaves`, nicht unter `octaves`. Die erste Fassung dieses
    Tests hat genau das verwechselt und `octaves` faelschlich als Kollision
    gemeldet.
    """
    import ast
    import os
    gefunden = {}
    ordner = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "gui", "tabs")
    for datei in sorted(os.listdir(ordner)):
        if not datei.endswith(".py"):
            continue
        pfad = os.path.join(ordner, datei)
        with open(pfad, encoding="utf-8") as f:
            baum = ast.parse(f.read(), filename=pfad)
        for knoten in ast.walk(baum):
            if not isinstance(knoten, ast.Tuple) or len(knoten.elts) < 3:
                continue
            schluessel, _label, konfig = knoten.elts[0], knoten.elts[1], knoten.elts[2]
            if not (isinstance(schluessel, ast.Constant)
                    and isinstance(schluessel.value, str)):
                continue
            if isinstance(konfig, ast.Attribute) and isinstance(konfig.value, ast.Name):
                gefunden.setdefault(schluessel.value, set()).add(
                    f"{konfig.value.id}.{konfig.attr}")
    return gefunden


def run_kein_schluessel_zeigt_auf_zwei_konfigs():
    """
    Die harte Zusicherung: derselbe Parameterschluessel darf nicht auf zwei
    verschiedene Konfigurationen zeigen. Genau das waere im Reiter nicht zu
    sehen - der Regler traegt seine Spanne aus der zuletzt gewinnenden Quelle.
    """
    reg = registrierte_schluessel()
    print(f"       {len(reg)} Schluessel in den Reitern registriert")
    ok = True
    for schluessel, quellen in sorted(reg.items()):
        if len(quellen) > 1:
            ok &= check(f"'{schluessel}' zeigt auf genau eine Konfiguration",
                        False, ", ".join(sorted(quellen)))
    if ok:
        print("[OK] kein Schluessel zeigt auf zwei Konfigurationen")
    return ok


def run_keine_unangemeldeten_doppel():
    alle = parameter_je_klasse()
    doppelt = {k: v for k, v in alle.items() if len(v) > 1}
    print(f"       {len(alle)} Parameter in {len(KONFIG_KLASSEN)} Klassen, "
          f"{len(doppelt)} davon mehrfach")

    ok = True
    for schluessel, vorkommen in sorted(doppelt.items()):
        klassen = tuple(sorted(vorkommen))
        angemeldet = vd.DOPPELTE_SCHLUESSEL.get(schluessel)
        if angemeldet is None:
            ok &= check(f"'{schluessel}' ist angemeldet", False,
                        f"steht in {klassen} - entweder entfernen oder in "
                        f"DOPPELTE_SCHLUESSEL eintragen")
            continue
        erwartet = tuple(sorted(angemeldet[0]))
        ok &= check(f"'{schluessel}' angemeldet, Klassen stimmen",
                    klassen == erwartet, f"{klassen}")
        # Der Sinn des Registers ist die BEGRUENDUNG, nicht der Eintrag.
        ok &= check(f"'{schluessel}' hat eine Begruendung",
                    isinstance(angemeldet[1], str) and len(angemeldet[1]) > 40)

    # Gegenprobe: ein Register-Eintrag ohne echten Doppelschluessel ist
    # veraltet und gehoert raus, sonst waechst hier stille Altlast an.
    for schluessel in vd.DOPPELTE_SCHLUESSEL:
        ok &= check(f"'{schluessel}' im Register ist noch aktuell",
                    schluessel in doppelt,
                    "steht im Register, ist aber nicht mehr doppelt")
    return ok


def run_doppel_werden_wirklich_gefunden():
    """
    GEGENPROBE - ohne sie prueft der Test oben moeglicherweise nichts.

    Ein Wachhund, der nie anschlaegt, ist von einem schlafenden nicht zu
    unterscheiden. Hier wird ein kuenstlicher Doppelschluessel eingesetzt und
    geprueft, dass die Suche ihn findet.
    """
    vorher = parameter_je_klasse()
    ok = check("Suche findet den bekannten Fall",
               len(vorher.get("erosion_strength", {})) == 2,
               f"erosion_strength in {sorted(vorher.get('erosion_strength', {}))}")

    vd.BIOME.SEA_LEVEL_PROBE = {"min": 0, "max": 1, "default": 0, "step": 1}
    vd.TERRAIN.SEA_LEVEL_PROBE = {"min": 0, "max": 1, "default": 0, "step": 1}
    try:
        nachher = parameter_je_klasse()
        ok &= check("kuenstlicher Doppelschluessel wird erkannt",
                    len(nachher.get("sea_level_probe", {})) == 2,
                    f"{sorted(nachher.get('sea_level_probe', {}))}")
        ok &= check("und er ist NICHT angemeldet, wuerde also melden",
                    "sea_level_probe" not in vd.DOPPELTE_SCHLUESSEL)
    finally:
        del vd.BIOME.SEA_LEVEL_PROBE
        del vd.TERRAIN.SEA_LEVEL_PROBE
    ok &= check("Probe wieder entfernt",
                "sea_level_probe" not in parameter_je_klasse())
    return ok


def run_unerreichbare_sind_benannt():
    """12.4 - definiert, aber in keinem Reiter: das muss begruendet sein."""
    gesperrt = vd.stillgelegte_regler()
    alle = parameter_je_klasse()
    ok = True
    for schluessel in UNERREICHBAR:
        if schluessel not in alle:
            print(f"[--] '{schluessel}' gibt es nicht mehr - Eintrag hier "
                  f"kann weg")
            continue
        grund = gesperrt.get(schluessel)
        ok &= check(f"'{schluessel}' traegt eine Begruendung",
                    isinstance(grund, str) and len(grund) > 30,
                    (grund or "keine")[:60])
    return ok


def main():
    print("=" * 70)
    print("Parameterschluessel eindeutig (12.3) und unerreichbare benannt (12.4)")
    print("=" * 70)
    ergebnisse = []
    for name, funktion in [
            ("kein Schluessel auf zwei Konfigurationen", run_kein_schluessel_zeigt_auf_zwei_konfigs),
            ("keine unangemeldeten Doppelschluessel", run_keine_unangemeldeten_doppel),
            ("Gegenprobe: Doppel werden gefunden", run_doppel_werden_wirklich_gefunden),
            ("unerreichbare Regler sind benannt", run_unerreichbare_sind_benannt)]:
        print(f"\n--- {name} ---")
        ergebnisse.append(funktion())
    print("\n" + "=" * 70)
    fehlend = ergebnisse.count(False)
    print(f"{len(ergebnisse) - fehlend}/{len(ergebnisse)} Gruppen gruen")
    return 0 if fehlend == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
