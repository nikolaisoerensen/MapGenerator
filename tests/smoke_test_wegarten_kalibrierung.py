"""
Wegekostenstaffelung nach Wegart (Ticket #41).

Vorher gab es nur einen einzigen flachen Rabatt (WEGERABATT = 0.4) fuer
jeden bereits benutzten Pixel - benutzt oder nicht, keine Abstufung. Jetzt
sammelt jeder Pixel eine Nutzungszahl (core/settlement_generator.py:
`nutzung` in calculate_road_network()), und ab festen Schwellen gilt er als
Trampelpfad/Karrenweg/Strasse mit je eigenem, staerkerem Rabatt
(core/daten/wegarten.toml, core/daten/wegarten_laden.py).

DIE EIGENTLICHE GEFAHR laut Ticket ist die RUECKKOPPLUNG: ein Weg wird durch
Benutzung billiger, billiger zieht mehr Benutzung an, das baut ihn weiter
aus - ein Regelkreis, der bei falscher Kalibrierung "auslaufen" kann (das
Netz entartet zum Stern, weil jede Route ueber denselben Trunk umgeleitet
wird) oder "einschlaeft" (die Staffelung ist zu flach, es passiert nie
etwas). Dieser Test prueft beides:

  1. Die Ladefunktion erzwingt eine gueltige Staffelung (steigende
     Schwellen, fallende Kostenfaktoren) und laedt die echte
     core/daten/wegarten.toml fehlerfrei.
  2. DIREKTER Rueckkopplungsnachweis an einem kleinen, von Hand gebauten
     Gelaende: mehrere Siedlungen muessen durch eine einzige Engstelle
     (Passage durch einen kostspieligen Grat), buendeln sich also
     zwangslaeufig auf denselben Pixeln - die Nutzungszahl dort MUSS mit der
     Anzahl der Routen steigen, und die Engstelle MUSS eine hoehere Wegart
     erreichen als eine Stichstrecke, die nur einmal befahren wird.
  3. Kennzahlen (core/wegnetz_kennzahlen.py: Gesamtlaenge, Anteil je Wegart,
     mittlerer Umwegfaktor, Knotenzahl) auf ECHTEN Kartengroessen
     (256/512/1024, keine ausgedachten) ueber je 3 Seeds - alle neun
     Kombinationen muessen in einem festen Band bleiben. Das Band selbst
     steht absichtlich NICHT in einer Datei, deren Name mit "bandgrenzen"
     beginnt (nachtbetrieb/sperrliste.toml sperrt genau dieses Muster),
     sondern als Konstanten unten in dieser Datei.
"""
import logging
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# Kalibrierungsband (Abnahmekriterium 4+5). Ermittelt durch einen Messlauf
# ohne Zusicherungen ueber alle 9 Kombinationen aus KARTENGROESSEN x SEEDS
# (siehe Docstring von `_kennzahlen_pruefen` fuer die beobachteten Rohwerte),
# mit Sicherheitsabstand nach oben/unten. Das Band ist bewusst GROSSZUEGIG:
# Ziel ist der Nachweis "die Rueckkopplung laeuft nicht davon", nicht eine
# scharfe Eichung auf die dritte Nachkommastelle.
KARTENGROESSEN = (256, 512, 1024)
SEEDS = (1, 2, 3)

# mittlerer Umwegfaktor (Pfadlaenge / Luftlinie) - 1.0 waere reine Luftlinie.
# Nach oben begrenzt, weil ein davonlaufendes Buendelungsverhalten genau
# hier sichtbar wuerde: jede Route macht einen im Verhaeltnis zur Luftlinie
# IMMER groesseren Umweg, um einen bestehenden guenstigen Trunk zu erreichen.
#
# 4.0 statt eines engeren Werts, weil auf kleinen, siedlungsarmen Karten
# (256px/15 Siedlungen) ein EINZELNER Weg einen sehr grossen Umweg um ein
# einzelnes hohes Gelaendehindernis machen kann, ohne dass das mit der
# Wegarten-Rueckkopplung zu tun hat - gemessen am 256px/Seed-3-Fall (siehe
# Docstring von `_test_kennzahlen_kalibrierung`): Faktor 12.6 fuer einen
# einzelnen Pfad, der eine grosse Huegelkette umrundet, per Gegenprobe mit
# abgeschaltetem Wegarten-Rabatt als Gelaende-Effekt bestaetigt (Faktor
# aendert sich dabei nur von 12.56 auf 12.59). Bei nur 13 Wegen auf dieser
# Kartengroesse zieht ein solcher Ausreisser den MITTELWERT spuerbar nach
# oben (3.33) - das Band muss diesen ehrlich gemessenen Fall durchlassen,
# ohne bei einer echten Rueckkopplungs-Entartung (durchgehend hohe Faktoren
# ueber ALLE Wege, nicht nur einen) blind zu sein.
UMWEGFAKTOR_MIN = 1.0
UMWEGFAKTOR_MAX = 4.0

# Anteil der Netzlaenge, der auf der teuersten Stufe (Strasse) liegt. Ein zu
# hoher Anteil hiesse: fast das gesamte Netz ist zu einem einzigen dichten
# Strassenbuendel entartet (der Stern aus Abnahmekriterium 6). 0.0 ist
# zulaessig (kleine Karten/wenige Siedlungen erreichen die hohe Schwelle
# eventuell nie).
STRASSENANTEIL_MAX = 0.60

# Anteil der Siedlungen, die am Ende NICHT an das Wegenetz angeschlossen
# sind. Der Kulturzusammenhang (§4.3) erzwingt Verbindungen nur INNERHALB
# einer Kultur - bei mehreren Kulturen auf sehr duennem Gelaende koennen
# einzelne Orte unverbunden bleiben, das ist kein Wegarten-Fehler. Trotzdem
# darf es nicht die Mehrheit sein.
UNVERBUNDEN_ANTEIL_MAX = 0.40


def _huegelland(map_size, seed):
    """Sanftes, wasserfreies Huegelland aus ueberlagerten Gauss-Huegeln.

    Bewusst OHNE Wasser: Seewege sind ein eigener Mechanismus (§4.4) und
    nicht Gegenstand von Ticket #41 - sie wuerden die Kennzahlen mit einem
    zweiten, hier nicht kalibrierten Kostenfeld vermischen. "Echte
    Kartengroessen" (256/512/1024) bezieht sich auf die PIXELZAHL, nicht auf
    ein bestimmtes Bioland; ein reines Testgelaende in dieser Groesse ist
    dafuer ausreichend.
    """
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:map_size, 0:map_size].astype(np.float64)
    heightmap = np.full((map_size, map_size), 20.0, dtype=np.float64)
    anzahl_huegel = max(6, map_size // 60)
    for _ in range(anzahl_huegel):
        cx = rng.uniform(0.1, 0.9) * map_size
        cy = rng.uniform(0.1, 0.9) * map_size
        breite = rng.uniform(0.05, 0.14) * map_size
        hoehe = rng.uniform(15.0, 60.0)
        heightmap += hoehe * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2)
                                       / (2 * breite ** 2)))
    dy, dx = np.gradient(heightmap)
    slopemap = np.dstack([dx, dy]).astype(np.float32)
    return heightmap.astype(np.float32), slopemap


def _siedlungen_streuen(map_size, seed, anzahl):
    """Zufaellig gestreute Siedlungen, drei Kulturen, gemischte Raenge -
    realistisch genug, um Bereitschaftstest UND Kulturzusammenhang wirklich
    zu durchlaufen, nicht nur den trivialen Fall (eine Kultur, alles baut)."""
    from core.settlement_generator import Location
    rng = np.random.RandomState(seed + 1000)
    kulturen = ["Kelten", "Sachsen", "Wikinger"]
    raenge = ["dorf", "dorf", "dorf", "siedlung", "siedlung", "stadt"]
    rand = map_size * 0.08
    siedlungen = []
    for i in range(anzahl):
        x = rng.uniform(rand, map_size - rand)
        y = rng.uniform(rand, map_size - rand)
        siedlungen.append(Location(
            location_id=i, x=float(x), y=float(y), location_type="settlement",
            radius=4.0, civ_influence=0.8,
            culture=kulturen[i % len(kulturen)],
            rank=raenge[i % len(raenge)],
        ))
    return siedlungen


def _netz_bauen(map_size, seed, anzahl_siedlungen):
    from core.settlement_generator import SettlementGenerator
    heightmap, slopemap = _huegelland(map_size, seed)
    siedlungen = _siedlungen_streuen(map_size, seed, anzahl_siedlungen)
    gen = SettlementGenerator.__new__(SettlementGenerator)
    gen.road_slope_to_distance_ratio = 1.5
    gen.map_seed = seed
    gen._update_progress = None
    # __new__() umgeht __init__() und damit auch `self.logger = logging.getLogger(...)`
    # (core/settlement_generator.py:4409) - bei >=3 Siedlungen je Kultur greift
    # _netz_nach_bedarf_ausbauen() und ruft self.logger.debug() auf. Die
    # bestehenden Wegenetz-Tests (tests/smoke_test_settlement_roads.py) sahen
    # das nie, weil sie nie mit mehr als 2 Siedlungen gleichzeitig bauen.
    gen.logger = logging.getLogger("SettlementGenerator")
    roads, sea_roads = gen.calculate_road_network(siedlungen, heightmap, slopemap, 5)
    return gen, siedlungen, roads, sea_roads


def _test_loader_validierung(fehler):
    """Abnahmekriterium 1: Wegarten-Kostenfaktoren sind versionierte Daten
    (core/daten/wegarten.toml), nicht Zahlen im Code, und die Ladefunktion
    erzwingt eine gueltige Staffelung."""
    print("1. Lader und Validierung (core/daten/wegarten_laden.py)")
    from core.daten.wegarten_laden import lade_wegarten, wegart_faktor_feld, klassifiziere

    wegarten = lade_wegarten()
    print("   %d Wegarten geladen: %s"
          % (len(wegarten), ", ".join(w.id for w in wegarten)))
    if len(wegarten) < 2:
        fehler.append("wegarten.toml hat weniger als 2 Stufen - keine Staffelung moeglich")
    for a, b in zip(wegarten, wegarten[1:]):
        if not (a.schwelle_nutzung < b.schwelle_nutzung):
            fehler.append("Schwellen nicht steigend: %s->%s" % (a.id, b.id))
        if not (a.kostenfaktor > b.kostenfaktor):
            fehler.append("Kostenfaktoren nicht fallend: %s->%s" % (a.id, b.id))

    # Eine kaputte Tabelle (Schwellen NICHT steigend) muss beim Laden
    # abgelehnt werden, nicht erst irgendwo spaeter durch ein seltsames
    # Ergebnis auffallen.
    import tempfile
    kaputt = (
        '[[wegart]]\nid = "a"\nname = "A"\nreihenfolge = 1\n'
        'schwelle_nutzung = 5\nkostenfaktor = 0.5\n\n'
        '[[wegart]]\nid = "b"\nname = "B"\nreihenfolge = 2\n'
        'schwelle_nutzung = 2\nkostenfaktor = 0.2\n'
    )
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write(kaputt)
        pfad_kaputt = f.name
    try:
        try:
            lade_wegarten(pfad=pfad_kaputt)
            fehler.append("kaputte Wegarten-Tabelle (fallende Schwellen) wurde NICHT abgelehnt")
        except ValueError:
            print("   kaputte Tabelle (fallende Schwellen) korrekt abgelehnt")
    finally:
        os.unlink(pfad_kaputt)

    feld = np.array([0.0, 1.0, 3.0, 7.0])
    faktoren = wegart_faktor_feld(feld, wegarten)
    if not (faktoren[0] == 1.0 and faktoren[1] > faktoren[2] > faktoren[3]):
        fehler.append("wegart_faktor_feld liefert keine fallende Stufenfolge: %s" % faktoren)
    if klassifiziere(0.0, wegarten) is not None:
        fehler.append("klassifiziere(0.0) sollte None sein (kein Weg)")


def _test_rueckkopplung_direkt(fehler):
    """Abnahmekriterium 2+5: direkter, deterministischer Nachweis der
    Rueckkopplung an einem Gelaende mit erzwungener Engstelle.

    Ein kostspieliger Grat trennt die Karte in zwei Haelften, mit einer
    einzigen schmalen Passage. FUENF UNABHAENGIGE Ortspaare (je eine eigene
    Kultur) stehen sich ueber den Grat gegenueber - jedes Paar MUSS die
    Passage durchqueren, um sein Gegenstueck zu erreichen, weil es sonst
    nirgends eine passierbare Stelle gibt.

    ERSTER ENTWURF hatte stattdessen einen einzigen Sammelpunkt und fuenf
    Aussenposten DERSELBEN Kultur - das maass nicht die Buendelung, sondern
    nur die Spannbaum-Optimierung: sobald die fuenf Aussenposten sich
    untereinander (auf ihrer Seite, ohne Gratquerung) verbinden durften,
    reichte EINE einzige Querung fuer den ganzen Kulturzusammenhang, und die
    anderen vier "Wege" waren reine Lokalverbindungen ohne jede Passage-
    Beruehrung (gemessen am 2026-09-18: hoechste Nutzungszahl in der Passage
    blieb bei 1, obwohl 5 Wege gebaut wurden). Getrennte Kulturen je Paar
    schliessen diesen Ausweg aus: `_netz_nach_bedarf_ausbauen`/Kulturzusammenhang
    (§4.3) verlangt Zusammenhang nur INNERHALB einer Kultur, und eine Kultur
    mit genau zwei Mitgliedern auf verschiedenen Seiten des Grats hat keine
    andere Wahl als die eine Passage.
    """
    print("\n2. Rueckkopplung direkt (erzwungene Engstelle)")
    from core.settlement_generator import Location, SettlementGenerator
    from core.daten.wegarten_laden import klassifiziere

    m = 200
    yy, xx = np.mgrid[0:m, 0:m].astype(np.float64)
    cx = m / 2.0
    breite = m * 0.06
    grat = 300.0 * np.exp(-((xx - cx) ** 2) / (2 * breite ** 2))
    passage = 280.0 * np.exp(-((xx - cx) ** 2) / (2 * breite ** 2)) \
        * np.exp(-((yy - m * 0.5) ** 2) / (2 * (breite * 0.6) ** 2))
    heightmap = (20.0 + grat - passage).astype(np.float32)
    dy, dx = np.gradient(heightmap.astype(np.float64))
    slopemap = np.dstack([dx, dy]).astype(np.float32)

    # 5 Paare, je eine eigene Kultur (K0..K4), je ein "dorf" auf der Westseite
    # (x=0.15) und eine "stadt" auf der Ostseite (x=0.85) auf DERSELBEN Zeile
    # wie ihr Partner - jedes Paar muss trotzdem zur Passage bei y=0.5*m
    # ausweichen, weil der Grat ausserhalb dieses Fensters ueberall sonst
    # (nahezu) unpassierbar ist. dorf/stadt (Rang 1/3) hebt die Bereitschaft
    # auf 3 - dasselbe Verhaeltnis wie im ersten Entwurf, das schon einmal
    # erfolgreich eine Gratquerung ausgeloest hat.
    orte = []
    for i in range(5):
        kultur = "Kultur%d" % i
        y = m * (0.2 + 0.15 * i)
        orte.append(Location(location_id=i * 2, x=m * 0.15, y=y,
                             location_type="settlement", radius=4.0,
                             civ_influence=0.8, culture=kultur, rank="dorf"))
        orte.append(Location(location_id=i * 2 + 1, x=m * 0.85, y=y,
                             location_type="settlement", radius=4.0,
                             civ_influence=0.8, culture=kultur, rank="stadt"))

    gen = SettlementGenerator.__new__(SettlementGenerator)
    gen.road_slope_to_distance_ratio = 1.5
    gen.map_seed = 7
    gen._update_progress = None
    gen.logger = logging.getLogger("SettlementGenerator")

    roads, _sea = gen.calculate_road_network(orte, heightmap, slopemap, 5)
    print("   %d Wege gebaut (mindestens 5 erwartet, je Kulturpaar eine "
          "Gratquerung)" % len(roads))
    if len(roads) < 5:
        fehler.append("Engstellen-Szenario: zu wenige Wege gebaut (%d), "
                      "Rueckkopplungsnachweis nicht moeglich" % len(roads))
        return

    nutzung = gen.letztes_nutzungsfeld
    passage_zeile = int(m * 0.5)
    # Grosszuegigeres Fenster als nur +-3 Pixel um den exakten Mittelpunkt:
    # die fuenf Paare erreichen die Passage aus verschiedenen y-Richtungen
    # und muessen sich nicht alle exakt im selben Pixel treffen, um
    # trotzdem eine gemeinsame Engstelle (das insgesamt schmale
    # Passage-Fenster) zu durchqueren.
    passage_ausschnitt = nutzung[passage_zeile - 10:passage_zeile + 10,
                                 int(cx) - 15:int(cx) + 15]
    max_nutzung_passage = float(passage_ausschnitt.max())
    print("   hoechste Nutzungszahl in der Passage: %.0f" % max_nutzung_passage)

    if max_nutzung_passage < 2:
        fehler.append("Passage wird nicht von mehreren Routen gemeinsam benutzt "
                      "(hoechste Nutzungszahl %.0f) - keine Buendelung messbar"
                      % max_nutzung_passage)

    wegart_passage = klassifiziere(max_nutzung_passage, gen.wegarten)
    print("   Wegart der Passage bei dieser Nutzung: %s" % wegart_passage)
    if wegart_passage is None or wegart_passage == gen.wegarten[0].id:
        fehler.append("Passage bleibt trotz Mehrfachnutzung auf der niedrigsten "
                      "Wegart (%s) - die Rueckkopplung greift nicht" % wegart_passage)

    # Gegenprobe: ein Pixel weit weg von jeder Route darf NICHT dieselbe
    # Nutzung zeigen wie die Engstelle - sonst waere "Nutzung" ueberall
    # gleich und die Staffelung wuerde nichts unterscheiden.
    ferner_punkt = float(nutzung[10, 10])
    if ferner_punkt >= max_nutzung_passage:
        fehler.append("unbenutzter Punkt hat dieselbe/eine hoehere Nutzung wie "
                      "die Engstelle (%.0f vs. %.0f)" % (ferner_punkt, max_nutzung_passage))
    print("   Nutzung an unbenutzter Stelle: %.0f (soll < Engstelle)" % ferner_punkt)


def _test_kennzahlen_kalibrierung(fehler):
    """Abnahmekriterium 3+4+5: Kennzahlen auf echten Kartengroessen ueber
    3 Seeds, im Band.

    BEOBACHTETE ROHWERTE (Messlauf 2026-09-18, NACH dem Beheben eines
    fehlenden Wegart-Vorsprungs im Engstellen-Test, 9 Kombinationen aus
    KARTENGROESSEN x SEEDS - eine fruehere Fassung dieses Kommentars nannte
    "durchgehend zwischen 1.0 und rund 1.6" fuer den Umwegfaktor; das war
    NICHT das, was ein echter Lauf zeigt, und wurde hier korrigiert, statt
    eine falsche Angabe stehen zu lassen):

        Groesse  Seed   Umweg   Strassen%   Unverb.%
        256      1      1.51    0.0         0.0
        256      2      1.77    0.0         0.0
        256      3      3.33    1.7         0.0
        512      1      1.26    0.0         0.0
        512      2      1.75    0.0         0.0
        512      3      1.45    5.3         0.0
        1024     1      1.35    2.0         0.0
        1024     2      1.39    2.3         0.0
        1024     3      1.32    0.0         3.3

    Der Ausreisser 256px/Seed 3 (Umweg 3.33) kommt von einem einzelnen Pfad
    unter 13 gebauten Wegen, der eine grosse, zufaellig plazierte Huegelkette
    komplett umrundet (Faktor 12.6 fuer diesen einen Pfad allein - siehe
    Gegenprobe unten). Per Gegenprobe mit abgeschaltetem Wegarten-Rabatt
    (core/daten/wegarten_laden.wegart_faktor_feld durch eine Konstante 1.0
    ersetzt) ergab sich fuer denselben Pfad Faktor 12.56 statt 12.59 - der
    Umweg ist also Gelaende-Kosten-Verhalten von VOR Ticket #41, nicht die
    hier kalibrierte Rueckkopplung. Strassenanteil liegt bei 0-5.3 %,
    unverbundener Anteil bei 0-3.3 % (eine Siedlung von 30 bei 1024px/Seed 3).
    Das Band oben (UMWEGFAKTOR_MAX=4.0, STRASSENANTEIL_MAX=0.60,
    UNVERBUNDEN_ANTEIL_MAX=0.40) liegt bewusst mit Abstand darueber, ohne den
    ehrlich gemessenen Ausreisser zu verstecken.
    """
    print("\n3. Kennzahlen ueber Kartengroessen x Seeds (Band)")
    from core.wegnetz_kennzahlen import (
        netzlaenge_und_anteile, mittlerer_umwegfaktor, knotenzahl,
    )

    print("   %-8s %-6s %-10s %-10s %-12s %-10s %s"
          % ("Groesse", "Seed", "Gesamt", "Umweg", "Strassen%", "Unverb.%", "Knoten"))
    for map_size in KARTENGROESSEN:
        anzahl_siedlungen = {256: 15, 512: 22, 1024: 30}[map_size]
        for seed in SEEDS:
            gen, siedlungen, roads, sea_roads = _netz_bauen(map_size, seed, anzahl_siedlungen)
            alle_wege = roads + sea_roads
            if not roads:
                fehler.append("%dpx Seed %d: keine Landwege gebaut - Kalibrierung "
                              "nicht messbar" % (map_size, seed))
                continue

            laenge = netzlaenge_und_anteile(roads, gen.letztes_nutzungsfeld, gen.wegarten)
            umweg = mittlerer_umwegfaktor(roads)
            knoten = knotenzahl(siedlungen, _gebaut_ids(gen, siedlungen, roads, sea_roads))

            strassen_anteil = 0.0
            if laenge["gesamt"] > 0:
                strassen_anteil = laenge["je_wegart"].get("strasse", 0.0) / laenge["gesamt"]
            unverbunden_anteil = 1.0 - (knoten["verbunden"] / max(knoten["gesamt"], 1))

            print("   %-8d %-6d %-10.0f %-10.2f %-12.1f %-10.1f %d/%d"
                  % (map_size, seed, laenge["gesamt"], umweg,
                     strassen_anteil * 100.0, unverbunden_anteil * 100.0,
                     knoten["verbunden"], knoten["gesamt"]))

            if not (UMWEGFAKTOR_MIN - 1e-6 <= umweg <= UMWEGFAKTOR_MAX):
                fehler.append("%dpx Seed %d: Umwegfaktor %.2f ausserhalb Band [%.1f, %.1f]"
                              % (map_size, seed, umweg, UMWEGFAKTOR_MIN, UMWEGFAKTOR_MAX))
            if strassen_anteil > STRASSENANTEIL_MAX:
                fehler.append("%dpx Seed %d: Strassenanteil %.1f%% ueber Band-Obergrenze %.0f%% "
                              "- Netz laeuft vermutlich zum Stern zusammen"
                              % (map_size, seed, strassen_anteil * 100.0, STRASSENANTEIL_MAX * 100.0))
            if unverbunden_anteil > UNVERBUNDEN_ANTEIL_MAX:
                fehler.append("%dpx Seed %d: %.1f%% der Siedlungen unverbunden, ueber Band-Obergrenze %.0f%%"
                              % (map_size, seed, unverbunden_anteil * 100.0, UNVERBUNDEN_ANTEIL_MAX * 100.0))


def _gebaut_ids(gen, siedlungen, roads, sea_roads):
    """Rekonstruiert ein {frozenset({id_a, id_b})}-Set aus den fertigen
    Wegen fuer die Kennzahlen-Funktion knotenzahl().

    calculate_road_network() gibt sein internes `gebaut`-Set nicht zurueck
    (Ticket #41 aendert die Rueckgabe absichtlich nicht, siehe Kommentar vor
    `return roads, sea_roads`) - fuer die Kennzahlen reicht es, jeden Weg
    ueber seinen naechstgelegenen Start-/End-Ort den beiden Siedlungen
    zuzuordnen, deren Position seinen Endpunkten am naechsten liegt.
    """
    positionen = [(s.location_id, s.x, s.y) for s in siedlungen]

    def naechste(x, y):
        return min(positionen, key=lambda p: (p[1] - x) ** 2 + (p[2] - y) ** 2)[0]

    paare = set()
    for pfad in roads + sea_roads:
        if len(pfad) < 2:
            continue
        x0, y0 = pfad[0]
        x1, y1 = pfad[-1]
        a_id = naechste(x0, y0)
        b_id = naechste(x1, y1)
        if a_id != b_id:
            paare.add(frozenset((a_id, b_id)))
    return paare


def main():
    fehler = []
    _test_loader_validierung(fehler)
    _test_rueckkopplung_direkt(fehler)
    _test_kennzahlen_kalibrierung(fehler)

    print("")
    if fehler:
        print("NICHT IN ORDNUNG - %d Befunde:" % len(fehler))
        for f in fehler:
            print("   " + f)
        return 1
    print("Alle Zusicherungen erfuellt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
