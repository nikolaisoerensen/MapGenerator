"""
Path: tests/smoke_test_kuestengebiete.py

Die Kuestengebiete im Hinterland (docs/archiv/2026-08-25_AUFRAEUMPLAN.md 4.8).

ANLASS (Nutzerbefund 2026-08-25 am 3D-Bild): *"siehst du die kuestenformen
(farbig) wie sie uebergehen in die noiseregionen (gruen). es sieht nicht so
gut aus, weil wir das problem haben, dass wir das inland hinter den
kuestentypen nicht mit aehnlicher topologie betrachten, wie die kuesten
selber."*

Beziffert: ein Kuestenarchetyp bestimmt die ersten 178-208 m voll (p1) und
blendet bis hoechstens 700 m aus (p2). Die Region ist rund 7100 m breit -
der Archetyp regierte also 7 % des Wegs zur Regionsmitte, dahinter uebernahm
ein Rauschfeld, das von ihm nichts weiss.

DER ENTWURF STAMMT VOM NUTZER: entlang der Kueste tragen die Voronoi-Zellen
den Wert ihres Archetyps, von dort waechst eine Breitensuche ueber den
Zellnachbarschaftsgraphen ins Land, Ziel drei Gebiete je Region mit je
15-50 % der Flaeche, Werte an den Grenzen ineinander uebergehend.

ERSTER AUSBAU: NUR DIE MITTLERE HOEHE (Nutzervorgabe 2026-08-26). Alles
andere erbt unveraendert von der Region.

WAS HIER GEPRUEFT WIRD, und warum gerade das:

  1. Die Wasserlinie bleibt liegen. Der erste Entwurf war ADDITIV und
     drueckte Land unter Null - an der Wasserlinie standen noch 68 m
     Delta. Multiplikativ kann das nicht passieren (H = 0 mal irgendwas
     bleibt 0), aber nur eine Messung zeigt, dass es auch so umgesetzt ist.
  2. Die Regionsmittel bleiben stehen. `hoehe_m`, `relief_m`, `wasser_soll`
     und `flaeche_soll` sind REGIONSMITTEL, und die ganze Eichung haengt
     daran (smoke_test_regionen_welt).
  3. Die Rangfolge stimmt: ein Gebiet mit hoeherem GEMESSENEN Hinterland
     liegt hoeher. Ein Vorzeichenfehler waere sonst unsichtbar - das
     Gelaende saehe nur anders aus, nicht falsch.

     GEPRUEFT WIRD SEIT 2026-08-26 GEGEN `GEMESSENE_HINTERLANDHOEHE`, nicht
     mehr gegen `hoehe_faktor`. Der Katalogfaktor beschreibt das UFER; das
     Gebietssystem setzt aber das HINTERLAND, und in vier von neun Regionen
     ist die Reihenfolge der beiden umgekehrt (Fjordbucht 0.30 gegen 195 m
     gemessen, Schaerenkueste 0.50 gegen 15 m). Der Test hat die Umstellung
     korrekt als Fehlschlag gemeldet - die Zusicherung hat sich geaendert,
     nicht die Anforderung an ihre Genauigkeit: die Grenze bleibt +0.50.
  4. Die Flaechenquote. Sie hat DREI Fehler gehabt, die alle plausible
     Karten lieferten: ein saatloser Archetyp mit Abstand 1e3 (unerreichbar
     fuer die Log-Regelung), ein ueberall GLEICHER Ersatzabstand (der Typ
     gewinnt alles oder nichts), und der letzte statt des besten
     Regeldurchgangs (die Regelung schwingt, weil die Graphdistanz
     ganzzahlig ist). Gemessen: Samarcia 0/100/0 bei drei vorhandenen Saaten.
  5. Das Nevadin bleibt ueberwiegend "alpin" - Nutzervorgabe.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_kuestengebiete.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from scipy import ndimage

import core.terrain_weltkarte as rw
from core.vektor_kueste import GEMESSENE_HINTERLANDHOEHE as HINTERLAND

SIZE = 384
SEEDS = (20260804, 20261817, 20262830)

# Die Wasserlinie darf sich um Bruchteile eines Meters bewegen (Rundung),
# nicht um Meter. Gemessen liegt sie bei 0.23 m.
WASSERLINIE_MAX_M = 2.0
# Regionsmittel: gemessen unter 0.1 m. 1.5 m Grenze faengt eine echte
# Verschiebung, ohne bei Rundung anzuschlagen.
REGIONSMITTEL_MAX_M = 1.5
# Anteil je Gebiet, Nutzervorgabe. Eine Region darf danebenliegen: bei 15-25
# Zellen je Region ist eine Zelle 4-7 Prozentpunkte, das Fenster ist damit
# nicht immer erreichbar.
ANTEIL_MIN, ANTEIL_MAX = 0.15, 0.50
FENSTER_AUSNAHMEN_MAX = 2
# Rangkorrelation Gebietshoehe gegen hoehe_faktor.
RANG_MIN = 0.5


def check(label, bedingung, zusatz=""):
    print(f"[{'OK' if bedingung else 'FAIL'}] {label}"
          + (f" - {zusatz}" if zusatz else ""))
    return [] if bedingung else [f"{label}{' - ' + zusatz if zusatz else ''}"]


def lauf():
    fehler = []
    namen = {i: r["name"] for i, (_z, _s, r) in enumerate(rw.alle_regionen())}

    wasserlinie, mittelabweichung, raenge = [], [], []
    fenster_verstoesse, alpin_anteile, ohne_gebiete = [], [], 0

    for seed in SEEDS:
        H, felder = rw.weltfeld(SIZE, seed)
        H = np.asarray(H, dtype=np.float64)
        delta = felder.get("gebiets_delta_m")
        gebiet = felder.get("gebiet")
        if delta is None or gebiet is None:
            fehler += check(f"Seed {seed}: Gebietsfelder vorhanden", False)
            continue
        land = H > 0
        H_ohne = H - delta

        # 1. Wasserlinie
        kueste = land & ~ndimage.binary_erosion(land)
        wasserlinie.append(float(np.abs(delta[kueste]).max())
                           if kueste.any() else 0.0)
        if int((H_ohne > 0).sum()) != int(land.sum()):
            fehler += check(f"Seed {seed}: Landanteil unveraendert", False,
                            f"{int((H_ohne > 0).sum())} statt {int(land.sum())}")

        for i, name in namen.items():
            m = land & (felder["regionen"] == i)
            if m.sum() < 200:
                continue
            # 2. Regionsmittel
            mittelabweichung.append(abs(float(H[m].mean() - H_ohne[m].mean())))

            typen = rw.KUESTEN_ARCHETYPEN.get(name) or []
            hf, mittel, anteile = [], [], []
            for j, t in enumerate(typen):
                mm = m & (gebiet == j)
                anteile.append(float(mm.sum()) / float(m.sum()))
                if mm.sum() < 30:
                    continue
                hf.append(float(HINTERLAND.get(t["name"],
                                               t["hoehe_faktor"])))
                mittel.append(float(delta[mm].mean()))

            if name == "Nevadin":
                # 5. ueberwiegend "alpin" (Gebiet -1)
                alpin_anteile.append(float((m & (gebiet < 0)).mean()
                                           / max(m.mean(), 1e-9)))
                continue
            if sum(anteile) < 0.5:
                ohne_gebiete += 1
                continue
            # 3. Rangfolge
            if len(hf) > 1 and np.std(hf) > 1e-9 and np.std(mittel) > 1e-9:
                raenge.append(float(np.corrcoef(hf, mittel)[0, 1]))
            # 4. Fenster
            if any(a > 0.02 and not (ANTEIL_MIN <= a <= ANTEIL_MAX)
                   for a in anteile):
                fenster_verstoesse.append(
                    f"{name} {'/'.join(f'{a:.0%}' for a in anteile)}")

    n_reg = len(SEEDS) * 8       # ohne Nevadin
    fehler += check("Wasserlinie bleibt liegen",
                    max(wasserlinie) < WASSERLINIE_MAX_M,
                    f"groesstes |Delta| an der Kueste {max(wasserlinie):.2f} m "
                    f"(Grenze {WASSERLINIE_MAX_M})")
    fehler += check("Regionsmittel bleiben stehen",
                    max(mittelabweichung) < REGIONSMITTEL_MAX_M,
                    f"groesste Verschiebung {max(mittelabweichung):.2f} m "
                    f"(Grenze {REGIONSMITTEL_MAX_M})")
    fehler += check("hoeheres gemessenes Hinterland -> hoeheres Gebiet",
                    len(raenge) >= 12 and min(raenge) > RANG_MIN,
                    f"{len(raenge)} Regionen, schlechteste Rangkorrelation "
                    f"{min(raenge):+.2f} (Grenze {RANG_MIN:+.2f})")
    fehler += check(f"Gebiete im Fenster {ANTEIL_MIN:.0%}-{ANTEIL_MAX:.0%}",
                    len(fenster_verstoesse) <= FENSTER_AUSNAHMEN_MAX * len(SEEDS),
                    (", ".join(fenster_verstoesse[:6])
                     if fenster_verstoesse else
                     f"alle {n_reg} Regionen") +
                    f" (bis zu {FENSTER_AUSNAHMEN_MAX} je Karte geduldet)")
    fehler += check("jede Region bekommt ueberhaupt Gebiete",
                    ohne_gebiete == 0,
                    f"{ohne_gebiete} von {n_reg} ohne")
    if alpin_anteile:
        fehler += check("Nevadin bleibt ueberwiegend alpin",
                        min(alpin_anteile) > 0.5,
                        f"kleinster alpiner Anteil {min(alpin_anteile):.0%}")

    print()
    print("=" * 78)
    if fehler:
        print(f"NICHT IN ORDNUNG - {len(fehler)} Befunde:")
        for f in fehler:
            print(f"   {f}")
        return 1
    print(f"Kuestengebiete in Ordnung ({len(SEEDS)} Karten)")
    return 0


if __name__ == "__main__":
    sys.exit(lauf())
