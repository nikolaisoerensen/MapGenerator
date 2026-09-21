"""
Ticket #42 ("Bruecken und Uferwege in die Wegekosten").

Bisher kannte bau_kostenfeld() nur Wasser aus der HOEHE (Meeresspiegel).
Ein Fluss schneidet sich aber nicht in die heightmap ein (core/water_
generator.py laesst sie unveraendert) - er liegt auf Land mit h > 0 und
wurde deshalb wie gewoehnliches Gelaende behandelt: kein Uferbonus, keine
Furtkosten, keine Bruecken. Dieser Test prueft die Luecke ist geschlossen,
gegen die Abnahmekriterien des Tickets:

  1. Uferwege (dem Fluss folgend) sind guenstiger als Querfeldein.
  2. Eine Flussquerung ohne Bruecke ist teuer; mit Bruecke billig - beide
     Werte liegen im selben, versionierten Datensatz (Modulkonstanten hier).
  3. Bruecken entstehen dort, wo sich die Querung lohnt - NICHT ueberall und
     NICHT zufaellig (platziere_bruecken() verlangt gebuendelten Verkehr).
  4. Messbar: Anzahl Bruecken je Karte, Anteil der Wege im Uferweg-Bereich.

Backward-Kompatibilitaet zu tests/smoke_test_settlement_roads.py bleibt
Pflicht: bau_kostenfeld() ohne water_map/bruecken_maske muss exakt wie vorher
rechnen, calculate_road_network() bleibt ein 2-Tupel (roads, sea_roads).
"""
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main():
    fehler = []
    from core.settlement_generator import (
        Location, bau_kostenfeld, platziere_bruecken, SettlementGenerator,
        UFERWEG_RABATT, UFERWEG_RADIUS_PX, FURT_KOSTEN_JE_TYP, BRUECKE_KOSTEN,
        BRUECKEN_MIN_VERKEHR,
    )

    # ---------- 1+2: Uferweg guenstiger, Furt teuer, Bruecke billig ----------
    print("1+2. Uferweg / Furt / Bruecke im Kostenfeld")
    n = 40
    heightmap = np.full((n, n), 20.0, dtype=np.float32)   # ueberall flaches Land
    slopemap = np.zeros((n, n, 2), dtype=np.float32)      # kein Hang irgendwo

    water_map = np.zeros((n, n), dtype=np.int32)
    water_map[:, 20] = 2   # ein Fluss (Typ "Fluss") als senkrechte Spalte

    feld_ohne_fluss = bau_kostenfeld(heightmap, slopemap, 1.5)
    feld_mit_fluss = bau_kostenfeld(heightmap, slopemap, 1.5, water_map=water_map)

    # Ohne Fluss unveraendert (Ruecksicherung der Rueckwaertskompatibilitaet).
    if not np.allclose(feld_ohne_fluss, 1.0):
        fehler.append("Kostenfeld ohne Fluss ist nicht ueberall 1.0")

    # Furt: auf dem Fluss selbst, Typ 2 -> FURT_KOSTEN_JE_TYP[2].
    furt_kosten = feld_mit_fluss[15, 20]
    print("   Furt (Typ Fluss)      %.3f (soll %.1f)" % (furt_kosten, FURT_KOSTEN_JE_TYP[2]))
    if abs(furt_kosten - FURT_KOSTEN_JE_TYP[2]) > 1e-6:
        fehler.append("Furtkosten falsch: %.3f statt %.1f" % (furt_kosten, FURT_KOSTEN_JE_TYP[2]))

    # Uferweg: direkt neben dem Fluss (innerhalb UFERWEG_RADIUS_PX), aber
    # NICHT auf dem Fluss selbst - muss GUENSTIGER als ebenes Land sein.
    ufer_kosten = feld_mit_fluss[15, 20 - UFERWEG_RADIUS_PX]
    print("   Uferweg (%d px Abstand) %.3f (soll %.3f, < 1.0)"
          % (UFERWEG_RADIUS_PX, ufer_kosten, UFERWEG_RABATT))
    if abs(ufer_kosten - UFERWEG_RABATT) > 1e-6:
        fehler.append("Uferweg-Rabatt falsch: %.3f statt %.3f" % (ufer_kosten, UFERWEG_RABATT))
    if not (ufer_kosten < 1.0):
        fehler.append("Uferweg ist nicht guenstiger als Querfeldein")

    # Querfeldein: weit weg vom Fluss - normale 1.0.
    quer_kosten = feld_mit_fluss[15, 5]
    print("   Querfeldein (fern)    %.3f (soll 1.0)" % quer_kosten)
    if abs(quer_kosten - 1.0) > 1e-6:
        fehler.append("Querfeldein-Kosten fern vom Fluss veraendert: %.3f" % quer_kosten)
    if not (ufer_kosten < quer_kosten):
        fehler.append("Uferweg ist nicht guenstiger als Querfeldein (Vergleich)")

    # Bruecke: dieselbe Furtstelle, aber mit bruecken_maske -> BRUECKE_KOSTEN,
    # deutlich billiger als die Furt ohne Bruecke.
    bruecken_maske = np.zeros((n, n), dtype=bool)
    bruecken_maske[15, 20] = True
    feld_mit_bruecke = bau_kostenfeld(heightmap, slopemap, 1.5, water_map=water_map,
                                      bruecken_maske=bruecken_maske)
    bruecken_kosten = feld_mit_bruecke[15, 20]
    print("   mit Bruecke           %.3f (soll %.1f)" % (bruecken_kosten, BRUECKE_KOSTEN))
    if abs(bruecken_kosten - BRUECKE_KOSTEN) > 1e-6:
        fehler.append("Bruecken-Kosten falsch: %.3f statt %.1f" % (bruecken_kosten, BRUECKE_KOSTEN))
    if not (bruecken_kosten < furt_kosten):
        fehler.append("Bruecke ist nicht billiger als die Furt ohne Bruecke")
    # Eine ANDERE Furtstelle auf demselben Fluss bleibt teuer - die Bruecke
    # wirkt NUR an ihrer eigenen Stelle, kein globaler Gelaendewert.
    andere_furt = feld_mit_bruecke[5, 20]
    if abs(andere_furt - FURT_KOSTEN_JE_TYP[2]) > 1e-6:
        fehler.append("Bruecke wirkt faelschlich auch an einer anderen Furtstelle: %.3f"
                      % andere_furt)

    # ---------- 3: platziere_bruecken - nicht ueberall, nicht zufaellig ----------
    print("\n3. platziere_bruecken: Schwelle fuer gebuendelten Verkehr")
    weg_maske = np.zeros((n, n), dtype=bool)
    # Zwei VERSCHIEDENE Wege queren an (fast) derselben Stelle (Zeile 15/16,
    # Spalte 19-21) - das ist die gebuendelte Furt, die eine Bruecke verdient.
    weg_a = [(x, 15) for x in range(5, 21)]
    weg_b = [(x, 16) for x in range(35, 19, -1)]
    for x, y in weg_a + weg_b:
        weg_maske[y, x] = True
    # Ein DRITTER Weg quert allein, weit entfernt - keine Buendelung, keine
    # Bruecke dort.
    water_map_einzeln = water_map.copy()
    water_map_einzeln[:, 5] = 1   # zweiter Fluss (Typ Bach) weiter links
    weg_c = [(5, y) for y in range(0, 10)]
    weg_maske_c = weg_maske.copy()
    for x, y in weg_c:
        weg_maske_c[y, x] = True

    fluss_land = (water_map_einzeln > 0) & (heightmap > 0)
    roads = [weg_a, weg_b, weg_c]
    bruecken_maske, bruecken_liste = platziere_bruecken(
        weg_maske_c, roads, fluss_land, water_map_einzeln)

    print("   %d Bruecke(n) gebaut (soll genau 1: die gebuendelte Furt)"
          % len(bruecken_liste))
    if len(bruecken_liste) != 1:
        fehler.append("platziere_bruecken baut nicht genau an der gebuendelten "
                      "Furt: %d Bruecken statt 1" % len(bruecken_liste))
    else:
        cx, cy, typ, verkehr = bruecken_liste[0]
        print("   Lage (%d,%d) Typ %d Verkehr %d (soll Spalte ~20, Verkehr >= %d)"
              % (cx, cy, typ, verkehr, BRUECKEN_MIN_VERKEHR))
        if not (18 <= cx <= 22):
            fehler.append("Bruecke liegt nicht an der gebuendelten Furtstelle: x=%d" % cx)
        if verkehr < BRUECKEN_MIN_VERKEHR:
            fehler.append("Bruecke unterschreitet die Mindestverkehr-Schwelle: %d" % verkehr)
    # Die einsame Furt bei Spalte 5 darf NICHT als Bruecke erscheinen.
    if np.any(bruecken_maske[:, 4:7]):
        fehler.append("platziere_bruecken baut faelschlich an der einsamen Furt "
                      "(nur 1 Weg quert dort)")

    # Keine Furt ueberhaupt -> keine Bruecken, nichts stuerzt ab.
    leere_maske, leere_liste = platziere_bruecken(
        np.zeros((n, n), dtype=bool), [], np.zeros((n, n), dtype=bool), water_map)
    if leere_liste:
        fehler.append("platziere_bruecken findet Bruecken ohne jeden Wegepixel")

    # ---------- 4: Ende-zu-Ende ueber calculate_road_network ----------
    print("\n4. calculate_road_network mit Fluss: Messung")
    m = 70
    heightmap_e2e = np.full((m, m), 20.0, dtype=np.float32)
    slopemap_e2e = np.zeros((m, m, 2), dtype=np.float32)
    water_map_e2e = np.zeros((m, m), dtype=np.int32)
    # Fluss als 3 Pixel breite Spalte, ueberwiegend teurer Grossfluss (Typ 3),
    # mit einer schmalen, GUENSTIGEN Furt (Typ 1) in der Mitte - simuliert
    # eine natuerliche Engstelle, an der sich mehrere Wege buendeln sollten.
    water_map_e2e[:, 29:32] = 3
    water_map_e2e[30:40, 29:32] = 1

    def ort(id_, x, y, culture, rank):
        return Location(location_id=id_, x=float(x), y=float(y), location_type='settlement',
                        radius=4.0, civ_influence=0.8, culture=culture, rank=rank)

    siedlungen = [
        ort(0, 5, 10, "Test", "stadt"),
        ort(1, 5, 35, "Test", "stadt"),
        ort(2, 5, 60, "Test", "stadt"),
        ort(3, 65, 10, "Test", "stadt"),
        ort(4, 65, 35, "Test", "stadt"),
        ort(5, 65, 60, "Test", "stadt"),
    ]

    gen = SettlementGenerator.__new__(SettlementGenerator)
    gen.road_slope_to_distance_ratio = 1.5
    gen.map_seed = 1
    gen._update_progress = None
    gen.logger = logging.getLogger("smoke_test_wege_wasser_kosten")

    roads, sea_roads = gen.calculate_road_network(
        siedlungen, heightmap_e2e, slopemap_e2e, 5, water_map=water_map_e2e)

    print("   %d Landweg(e), %d Bruecke(n), Uferweg-Anteil %.1f %%"
          % (len(roads), len(gen.letzte_bruecken), 100.0 * gen.letzter_uferweg_anteil))
    for cx, cy, typ, verkehr in gen.letzte_bruecken:
        print("      Bruecke bei (%d,%d), Typ %d, Verkehr %d" % (cx, cy, typ, verkehr))

    if len(roads) == 0:
        fehler.append("Ende-zu-Ende: keine Wege gebaut - Szenario taugt nicht als Test")
    # Kriterium 3 nochmal auf Ende-zu-Ende-Ebene: es duerfen NICHT so viele
    # Bruecken entstehen wie es Wege gibt (sonst waere "ueberall", nicht
    # "wo es sich lohnt") - bei einer einzelnen guenstigen Engstelle sollten
    # mehrere Wege sich dort buendeln und EINE (oder wenige) Bruecke(n) teilen.
    if len(gen.letzte_bruecken) >= len(roads) and len(roads) > 1:
        fehler.append("Bruecken wirken wie 'eine je Weg' statt gebuendelt: "
                      "%d Bruecken bei %d Wegen" % (len(gen.letzte_bruecken), len(roads)))
    # Jede gebaute Bruecke muss tatsaechlich auf der guenstigen Furt-Zeile
    # liegen (30-39), nicht irgendwo auf dem teuren Grossfluss-Abschnitt -
    # sonst waere sie zufaellig platziert statt dort, wo es sich lohnt.
    for cx, cy, typ, verkehr in gen.letzte_bruecken:
        if not (30 <= cy < 40):
            fehler.append("Bruecke liegt ausserhalb der guenstigen Engstelle: y=%d" % cy)

    # Ohne water_map bleibt calculate_road_network unveraendert - 0 Bruecken,
    # 0.0 Uferweg-Anteil, keine Regression fuer bestehende Aufrufer.
    gen2 = SettlementGenerator.__new__(SettlementGenerator)
    gen2.road_slope_to_distance_ratio = 1.5
    gen2.map_seed = 1
    gen2._update_progress = None
    gen2.logger = logging.getLogger("smoke_test_wege_wasser_kosten")
    roads2, _sea2 = gen2.calculate_road_network(
        siedlungen, heightmap_e2e, slopemap_e2e, 5)
    print("   ohne water_map: %d Bruecke(n) (soll 0), Uferweg-Anteil %.3f (soll 0.0)"
          % (len(gen2.letzte_bruecken), gen2.letzter_uferweg_anteil))
    if gen2.letzte_bruecken:
        fehler.append("Ohne water_map werden trotzdem Bruecken gemeldet")
    if gen2.letzter_uferweg_anteil != 0.0:
        fehler.append("Ohne water_map ist der Uferweg-Anteil nicht 0.0")

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
