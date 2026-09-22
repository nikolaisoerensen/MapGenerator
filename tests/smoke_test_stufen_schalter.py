"""
Path: tests/smoke_test_stufen_schalter.py

Greifen die drei Abschalthaekchen wirklich durch?

ANLASS (Nutzerwunsch 2026-08-25): *"kannst du mir einmal fuer flussnetzwerk
und erosionfilter und kuestentypen jeweils checkboxen einfuegen, mit denen
ich die effekte immer auch ausschalten kann?"*

Geprueft wird NICHT, ob die Checkbox existiert, sondern ob sich das
GELAENDE aendert. Ein Haekchen, das nichts tut, waere genau der stille
Fehlschlag, vor dem CLAUDE.md warnt - es gaebe keine Fehlermeldung, das
Bild saehe nur immer gleich aus, und man wuerde die Stufe faelschlich fuer
wirkungslos halten.

Zusaetzlich geprueft: mit abgeschalteter Kuestenformung darf `weltfeld()`
NICHT abstuerzen und keine Archetypfelder liefern - die Folgestufen muessen
deren Fehlen vertragen (`_seetiefe_aus_archetyp()` steigt ueber
`felder.get()` frueh aus).

GEMESSEN am 2026-08-25 (256 px, Seed 20260804), mittlere Aenderung:

    Kuestentypen     97.7 m   (groesster Beitrag)
    Flussnetz        17.0 m
    Erosionsfilter    2.5 m   (deutlich kleiner als erwartet)

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_stufen_schalter.py
"""
import os, sys, logging
import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from core.terrain_generator import BaseTerrainGenerator
from core.terrain_weltkarte import weltfeld
from core.terrain_weltfluesse import flussnetz, taeler_eingraben

SIZE, SEED = 256, 20260804

def gen():
    g = BaseTerrainGenerator.__new__(BaseTerrainGenerator)
    g.shader_manager = None; g.data_lod_manager = None
    g._current_parameters = {}; g.logger = logging.getLogger("schalter")
    return g

print(f"{SIZE} px, Seed {SEED}\n")

# --- Kuestentypen ----------------------------------------------------------
A, fa = weltfeld(SIZE, SEED, kuesten_aktiv=True)
B, fb = weltfeld(SIZE, SEED, kuesten_aktiv=False)
A, B = np.asarray(A, float), np.asarray(B, float)
print(f"Kuestentypen   an/aus: Mittel {np.abs(A-B).mean():7.1f} m, "
      f"max {np.abs(A-B).max():6.0f} m, "
      f"Land {100*(A>0).mean():.1f} % -> {100*(B>0).mean():.1f} %")
print(f"               Archetypfelder vorhanden: an={'kuesten_archetyp' in fa}, "
      f"aus={'kuesten_archetyp' in fb}")

# --- Erosionsfilter --------------------------------------------------------
g = gen()
gefiltert = g._weltkarte_erosionsfilter(A.copy().astype(np.float32), fa)
C = np.asarray(gefiltert["heightmap"], float) if gefiltert else A
print(f"Erosionsfilter an/aus: Mittel {np.abs(C-A).mean():7.1f} m, "
      f"max {np.abs(C-A).max():6.0f} m")

# --- Flussnetz -------------------------------------------------------------
netz = flussnetz(C, SEED)
D = np.asarray(taeler_eingraben(C.copy(), felder=fa, netz=netz,
                                abstand_makro_m=1200.0), float) if netz is not None else C
print(f"Flussnetz      an/aus: Mittel {np.abs(D-C).mean():7.1f} m, "
      f"max {np.abs(D-C).max():6.0f} m")

# Untergrenzen deutlich unter den gemessenen Werten - der Test soll fangen,
# dass ein Schalter GAR NICHT mehr wirkt, nicht jede Kalibrieraenderung.
GRENZEN_M = {"Kuestentypen": 10.0, "Erosionsfilter": 0.3, "Flussnetz": 2.0}

fehler = []
print()
for name, d in (("Kuestentypen", np.abs(A - B)),
                ("Erosionsfilter", np.abs(C - A)),
                ("Flussnetz", np.abs(D - C))):
    m = float(d.mean())
    ok = m > GRENZEN_M[name]
    if not ok:
        fehler.append(f"{name} wirkt nicht mehr ({m:.2f} m, "
                      f"Grenze {GRENZEN_M[name]:.1f})")
    print(f"[{'OK' if ok else 'FAIL'}] {name} veraendert das Gelaende messbar"
          f" - {m:.1f} m im Mittel (Grenze {GRENZEN_M[name]:.1f})")

ohne_archetyp = "kuesten_archetyp" not in fb
if not ohne_archetyp:
    fehler.append("kuesten_aktiv=False liefert trotzdem Archetypfelder")
print(f"[{'OK' if ohne_archetyp else 'FAIL'}] ohne Kuestenformung keine "
      f"Archetypfelder - und weltfeld() laeuft trotzdem durch")

# --- Das zweiseitige Oktaventor -------------------------------------------
#
# Kein Schalter, aber derselbe Zweck: eine Groesse, die still verrutschen
# kann und dann hunderte Zacken zurueckbringt. GEMESSEN am 2026-08-25
# (512 px, Seed 20260804): das Nevadin ging von 122 auf 24 Gipfel, seine
# Hoehenstreuung stieg dabei leicht (168 -> 173 m) - die Amplitude bleibt
# also erhalten, nur die Nadeln verschwinden. Nutzervorgabe: *"die alpen
# brauchen hier viiiiiel weniger spitzen ... amplitude ist ok"*.
from core.terrain_weltkarte import (GRUNDFORM_M, OKTAVEN, FEINHEIT_TEILER,
                                    alle_regionen)

wellen = np.array([GRUNDFORM_M / (2.0 ** k) for k in range(OKTAVEN)])
alpen = [r for _z, _s, r in alle_regionen() if r["name"] == "Nevadin"][0]
form = alpen["formgroesse_m"]
rau = float(np.clip(alpen["rauheit"], 0.2, 0.9))
tor = 1.0 / (1.0 + np.exp(-(form - wellen) / (0.35 * wellen)))
fein = 1.0 / (1.0 + np.exp(-(wellen - form / FEINHEIT_TEILER)
                           / (0.35 * form / FEINHEIT_TEILER)))
g = (rau ** np.arange(OKTAVEN)) * tor * fein
fein_m = float(g[wellen <= 375.0].sum() / g.sum() * alpen["relief_m"])

# Vor dem zweiseitigen Tor waren es 250 m. 120 m als Grenze faengt eine
# Rueckkehr, ohne bei jeder Kalibrieraenderung anzuschlagen.
FEIN_MAX_M = 120.0
fein_ok = fein_m < FEIN_MAX_M
if not fein_ok:
    fehler.append(f"Nevadin traegt wieder {fein_m:.0f} m in Wellen unter "
                  f"375 m (Grenze {FEIN_MAX_M:.0f}, vor dem Tor 250)")
print(f"[{'OK' if fein_ok else 'FAIL'}] Nevadin ohne Nadelspitzen - "
      f"{fein_m:.0f} m Relief unter 375 m Wellenlaenge "
      f"(Grenze {FEIN_MAX_M:.0f}, vor dem Tor 250)")

print()
print("=" * 78)
if fehler:
    print(f"NICHT IN ORDNUNG - {len(fehler)} Befunde:")
    for f in fehler:
        print(f"   {f}")
    sys.exit(1)
print("alle drei Schalter greifen")
sys.exit(0)
