"""
Path: tests/smoke_test_flussstufen.py

Gibt es auf der Karte alle vier Wasserstufen - Creek, River, Grand River,
Lake?

ANLASS (gemessen 2026-08-24): `water_biomes_map` enthielt ausschliesslich
Creeks und Seen. Die Stufen `river` (2) und `grand_river` (3) kamen NIE
vor, auf keiner Karte - und damit auch die Biome 17/18 nicht. Der Nutzer
hatte danach gefragt: *"die grossen fluesse koennen als overlay bei biome
drin sein."* Es gab keine grossen Fluesse.

DIE URSACHE war eine Mischung zweier Masstaebe. Die Creek-Schwelle ist
ein PERZENTIL der wasserfuehrenden Zellen, passt sich also der Verteilung
an. Die Faktoren fuer River und Grand River waren dagegen ABSOLUTE
Vielfache davon:

    creek-Schwelle                        1716.77
    river braucht  4x                     6867.08
    grand  braucht 20x                   34335.41
    groesster Abfluss der ganzen Karte     6408.09

Zwischen einem hohen Perzentil und dem Maximum liegt in einer
Abflussverteilung systematisch weniger als eine Groessenordnung - hier
Faktor 3.73. Ein absoluter Faktor 20 darauf ist nicht streng, sondern
unerfuellbar.

WARUM ES NIEMAND GEMERKT HAT: die Klassifikation lief fehlerfrei durch
und lieferte eine plausible Karte voller Baeche. Nichts stuerzte ab,
nichts warnte. Genau das Muster, vor dem CLAUDE.md warnt - ein Ergebnis,
das von einem richtigen nicht zu unterscheiden ist, solange niemand
nachzaehlt.

Aufruf (faehrt die volle Pipeline, rund 2 Minuten):
    .venv/Scripts/python.exe tests/smoke_test_flussstufen.py
"""

import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests.smoke_test_pipeline_outputs as P

# Stufen in `water_biomes_map`.
STUFEN = {1: "creek", 2: "river", 3: "grand_river", 4: "lake"}

# Biom-Nummern (super_biome_offset = 15).
BIOME = {16: "lake", 17: "grand_river", 18: "river", 19: "creek",
         21: "beach"}

# Wieviele Pixel eine Stufe mindestens haben muss, um als "kommt vor" zu
# gelten. Ein einzelnes Pixel waere Rauschen; ab hier ist es ein Lauf.
MINDEST_PIXEL = 10


def check(label, bedingung, zusatz=""):
    print(f"[{'OK' if bedingung else 'FAIL'}] {label}"
          + (f" - {zusatz}" if zusatz else ""))
    return [] if bedingung else [f"{label}{' - ' + zusatz if zusatz else ''}"]


def _pipeline():
    """Die volle Kette fahren und die interessanten Ausgaben einsammeln."""
    P._qt()
    from managers.data_lod_manager import DataLODManager
    from managers.calculator_graph import CALCULATOR_GRAPH

    manager = DataLODManager()
    manager.set_map_distance_km(P.KM)
    manager.set_map_seed(P.SEED)
    generatoren = P._generatoren(manager, None)
    parameter = P._parameter()
    for knoten in CALCULATOR_GRAPH:
        manager.set_calculator_target_lod(knoten, P.LOD)
    for generator in generatoren.values():
        if hasattr(generator, "set_active_parameters"):
            generator.set_active_parameters(parameter)

    for knoten in P._reihenfolge():
        spec = CALCULATOR_GRAPH[knoten]
        generator = generatoren.get(spec.generator)
        methode = getattr(generator, "_calc_" + knoten.split(".", 1)[1], None)
        if methode is None:
            continue
        try:
            methode(knoten, P.LOD)
        except Exception:                                    # noqa: BLE001
            pass

    hole = lambda k, s: manager.get_calculator_output(k, s, P.LOD)  # noqa: E731
    return {
        "zentral": hole("water.flow_network", "water_biomes_map"),
        "gemalt": hole("water.manning_flow", "water_biomes_map"),
        "biome": hole("biome.integrate_layers", "biome_map"),
        "super": hole("biome.supersampling", "biome_map_super"),
    }


def alle_stufen_kommen_vor(daten):
    """1. DIE EIGENTLICHE ZUSICHERUNG: keine Stufe fehlt."""
    fehler = []
    for name in ("zentral", "gemalt"):
        feld = daten[name]
        if feld is None:
            fehler += check(f"water_biomes_map ({name}) vorhanden", False)
            continue
        zaehler = collections.Counter(np.asarray(feld).ravel().tolist())
        gefunden = {STUFEN[w]: zaehler.get(w, 0) for w in sorted(STUFEN)}
        fehlend = [n for n, a in gefunden.items() if a < MINDEST_PIXEL]
        fehler += check(f"alle vier Stufen in water_biomes_map ({name})",
                        not fehlend,
                        ", ".join(f"{n} {a}" for n, a in gefunden.items())
                        + (f"   FEHLT: {', '.join(fehlend)}" if fehlend else ""))
    return fehler


def haeufigkeit_stimmt(daten):
    """
    2. Die Faktoren bedeuten HAEUFIGKEIT, nicht Wassermenge.

    RIVER_THRESHOLD_FACTOR = 4 heisst: ein River ist rund vier mal so
    selten wie ein Creek. Genau das war vorher nicht der Fall - es hiess
    "vier mal so viel Wasser", und das war unerfuellbar.

    Die Toleranz ist weit: die Perzentilfunktion klemmt auf 0.5..99.5,
    und Seen nehmen der Zaehlung Pixel weg. Geprueft wird die ORDNUNG und
    die Groessenordnung, nicht der genaue Wert.
    """
    from core.water_generator import FlowNetworkBuilder as F

    fehler = []
    feld = daten["zentral"]
    if feld is None:
        return check("Haeufigkeit pruefbar", False)
    zaehler = collections.Counter(np.asarray(feld).ravel().tolist())
    creek = zaehler.get(1, 0)
    river = zaehler.get(2, 0)
    grand = zaehler.get(3, 0)
    laeufe = creek + river + grand
    if laeufe < 50:
        return check("genug Wasserlaeufe zum Zaehlen", False, f"{laeufe}")

    # Ein River ist alles ab der River-Schwelle, also river + grand.
    anteil_river = (river + grand) / laeufe
    anteil_grand = grand / laeufe
    soll_river = 1.0 / F.RIVER_THRESHOLD_FACTOR
    soll_grand = 1.0 / F.GRAND_RIVER_THRESHOLD_FACTOR

    print(f"       {laeufe} Wasserlaeufe: {creek} creek, {river} river, "
          f"{grand} grand_river")
    fehler += check(
        f"mindestens River ist rund 1/{F.RIVER_THRESHOLD_FACTOR:.0f} der Laeufe",
        0.4 * soll_river <= anteil_river <= 2.5 * soll_river,
        f"{anteil_river:.1%} (soll rund {soll_river:.1%})")
    fehler += check(
        f"Grand River ist rund 1/{F.GRAND_RIVER_THRESHOLD_FACTOR:.0f} der Laeufe",
        0.3 * soll_grand <= anteil_grand <= 3.0 * soll_grand,
        f"{anteil_grand:.1%} (soll rund {soll_grand:.1%})")
    fehler += check("strenge Ordnung creek > river > grand_river",
                    creek > river > grand,
                    f"{creek} > {river} > {grand}")
    return fehler


def biome_kommen_an(daten):
    """
    3. Die Stufen erreichen auch die BIOMKARTE.

    Zwei Karten, mit unterschiedlicher Aufgabe - das ist kein Fehler,
    sondern Bauart, und deshalb hier festgehalten:

      `biome_map`        die grobe Karte. Enthaelt die Wasserstufen, aber
                         KEINE Wahrscheinlichkeits-Biome (beach, cliff,
                         lake_edge, river_bank, snow/alpine_level).
      `biome_map_super`  die feine. Erst das Supersampling setzt die
                         Wahrscheinlichkeiten in Pixel um.

    Wer Straende sehen will, muss also `biome_map_super` anzeigen. Am
    2026-08-24 war die Vermutung, Straende fehlten - sie sind da, nur
    nicht in der groben Karte.
    """
    fehler = []
    for name, schluessel in (("biome_map", "biome"),
                             ("biome_map_super", "super")):
        feld = daten[schluessel]
        if feld is None:
            fehler += check(f"{name} vorhanden", False)
            continue
        zaehler = collections.Counter(np.asarray(feld).ravel().tolist())
        gefunden = {BIOME[w]: zaehler.get(w, 0) for w in sorted(BIOME)}
        print(f"       {name:18s} " + ", ".join(
            f"{n} {a}" for n, a in gefunden.items()))
        fluesse = [n for n in ("creek", "river", "grand_river")
                   if gefunden[n] < MINDEST_PIXEL]
        fehler += check(f"alle Flussstufen in {name}", not fluesse,
                        f"FEHLT: {', '.join(fluesse)}" if fluesse else "")

    # Straende NUR in der feinen Karte - und dort MUESSEN sie sein.
    super_feld = daten["super"]
    if super_feld is not None:
        anzahl = int((np.asarray(super_feld) == 21).sum())
        fehler += check("Straende in biome_map_super", anzahl >= MINDEST_PIXEL,
                        f"{anzahl} Pixel")
    return fehler


def lauf():
    print("Fahre die volle Pipeline - das dauert.")
    daten = _pipeline()
    gruppen = [
        ("alle_stufen_kommen_vor", lambda: alle_stufen_kommen_vor(daten)),
        ("haeufigkeit_stimmt", lambda: haeufigkeit_stimmt(daten)),
        ("biome_kommen_an", lambda: biome_kommen_an(daten)),
    ]
    ergebnis, alle = {}, []
    for name, fn in gruppen:
        print(f"\n--- {name} ---")
        f = fn()
        ergebnis[name] = not f
        alle.extend(f)

    print("\n" + "=" * 78)
    for name, ok in ergebnis.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print(f"{sum(ergebnis.values())}/{len(ergebnis)} Gruppen gruen")
    if alle:
        print(f"\nNICHT IN ORDNUNG - {len(alle)} Befunde:")
        for f in alle:
            print(f"   {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(lauf())
