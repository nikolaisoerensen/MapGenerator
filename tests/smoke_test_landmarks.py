"""
Path: tests/smoke_test_landmarks.py

Prueft die Landmark-Platzierung nach dem Umbau vom 2026-08-13
(docs/OFFENE_PUNKTE.md 5.18, Nutzerbefund "Die Strassen und Roadsites und
Landmarks sind schlecht").

DER KERN DES UMBAUS: vorher waren die vier Kategorien BINAERE Masken, aus
denen zufaellig gezogen wurde - ein "Gipfel" landete auf irgendeinem Pixel
oberhalb 60 % der Hoehenspanne statt auf dem Gipfel. Jetzt liefert
`landmark_eignungen()` je Kategorie eine kontinuierliche Guete und es wird
die BESTE Stelle gewaehlt.

Die Zusicherungen hier pruefen genau das: dass ein Gipfel-Landmark wirklich
oben liegt, ein Kueste-Landmark wirklich am Meer - und dass nicht eine
einzige Kategorie alles belegt.
"""
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, ".")

import logging
from scipy.ndimage import gaussian_filter, distance_transform_edt

from core.terrain_weltkarte import weltfeld, WELT_KM, alle_regionen
from core.settlement_generator import (
    SettlementGenerator, Location, ABGELEGEN_DECKEL, KATEGORIE_WIEDERHOLUNG)


def check(label, bedingung, zusatz=""):
    status = "OK" if bedingung else "FAIL"
    print(f"[{status}] {label}{(' - ' + zusatz) if zusatz else ''}")
    return bool(bedingung)


def _welt(size, seed):
    """Gelaende plus ein REALISTISCHES Zivilisations- und Wasserfeld.

    Mit leeren Feldern zu testen waere irrefuehrend: `civ_map` ganz auf 0
    macht die Kategorie "abgelegen" ueberall maximal, und genau daran waere
    beim Bauen um ein Haar ein Fehler unbemerkt geblieben (siehe
    ABGELEGEN_DECKEL).
    """
    H, felder = weltfeld(size, seed)
    gy, gx = np.gradient(H.astype(np.float32), WELT_KM * 1000.0 / size)
    slope = np.stack([gx, gy], axis=-1).astype(np.float32)
    land = H > 0

    rng = np.random.RandomState(1)
    civ = np.zeros_like(H, dtype=np.float32)
    landpunkte = np.argwhere(land)
    voelker = [r["volk"] for _z, _s, r in alle_regionen()]
    staedte = []
    for i in range(9):
        y, x = landpunkte[rng.randint(len(landpunkte))]
        civ[max(0, y - 25):y + 25, max(0, x - 25):x + 25] = 1.0
        s = Location(location_id=i, x=float(x), y=float(y),
                     location_type="settlement", radius=3.0, civ_influence=0.7,
                     properties={}, culture=voelker[i % len(voelker)])
        s.rank = "stadt"
        staedte.append(s)
    civ = gaussian_filter(civ, sigma=12)
    civ /= max(float(civ.max()), 1e-6)

    fluss = felder.get("river_mask")
    water = ((np.asarray(fluss) > 0).astype(np.float32) if fluss is not None
             else np.zeros_like(H, dtype=np.float32))
    return H, slope, civ, water, staedte, felder


def _generator():
    g = SettlementGenerator.__new__(SettlementGenerator)
    g.logger = logging.getLogger("smoke_landmarks")
    g.landmarks = 3
    g.landmark_wilderness = 0.5
    g.scale_factor = 1.0
    g.next_location_id = 1000
    g._update_progress = None
    # WICHTIG: `random.Random`, NICHT `np.random.RandomState` - genau das,
    # was die echte `_knoten_zufall()` liefert. Ein numpy-Generator hier war
    # der erste Anlauf und hat einen echten Fehler VERDECKT: der Code rief
    # `zufall.uniform(..., size=...)`, was `random.Random` nicht kennt. Der
    # Landmark-Test blieb gruen, `smoke_test_settlement_sites.py` fiel mit
    # `TypeError` um. Ein Mock, der mehr kann als das Original, prueft nichts.
    g.map_seed = 42
    return g


def run_eignungskarten():
    H, slope, civ, water, _staedte, _f = _welt(384, 20260804)
    g = _generator()
    eig = g.landmark_eignungen(civ, H, slope, water)
    land = H > 0
    ok = True

    ok &= check("vier Kategorien geliefert",
                set(eig) == {"gipfel", "kueste", "quelle", "abgelegen"})
    for kat, karte in eig.items():
        ok &= check(f"  {kat}: Form/Bereich/kein Wert im Meer",
                    karte.shape == H.shape and float(karte.min()) >= 0.0
                    and float(karte.max()) <= 1.0
                    and bool(np.all(karte[~land] == 0.0)))

    # "abgelegen" darf NICHT alles dominieren (dafuer der Deckel)
    namen = list(eig)
    gewinner = np.argmax(np.stack([eig[k] for k in namen]), axis=0)
    anteil_abgelegen = float((gewinner[land] == namen.index("abgelegen")).mean())
    ok &= check(f"'abgelegen' dominiert nicht alles (Deckel {ABGELEGEN_DECKEL})",
                anteil_abgelegen < 0.90, f"{anteil_abgelegen:.1%} der Landflaeche")

    # Die beste Gipfelstelle muss wirklich hoch sein
    yi, xi = np.unravel_index(np.argmax(np.where(land, eig["gipfel"], -1.0)), H.shape)
    perzentil = float((H[land] < H[yi, xi]).mean())
    ok &= check("beste 'gipfel'-Stelle liegt im obersten Zehntel der Landhoehen",
                perzentil > 0.90, f"{perzentil:.1%}-Perzentil, {H[yi, xi]:.0f} m")

    # Die beste Kuestenstelle muss wirklich am Meer sein
    dist = distance_transform_edt(land)
    yi, xi = np.unravel_index(np.argmax(np.where(land, eig["kueste"], -1.0)), H.shape)
    ok &= check("beste 'kueste'-Stelle liegt am Meer",
                dist[yi, xi] <= 4.0, f"{dist[yi, xi]:.1f} px")
    return ok


def run_platzierung():
    ok = True
    for seed in (20260804, 12345):
        H, slope, civ, water, staedte, felder = _welt(384, seed)
        g = _generator()
        landmarks = g.calculate_landmarks(civ, H, slope, water, staedte, "FINAL",
                                           region_map=felder["regionen"])
        land = H > 0
        dist = distance_transform_edt(land)

        ok &= check(f"Seed {seed}: Landmarks platziert", len(landmarks) > 0,
                    f"{len(landmarks)}")
        ok &= check("  keines im Meer",
                    all(H[int(l.y), int(l.x)] > 0 for l in landmarks))

        kategorien = Counter(l.properties["kategorie"] for l in landmarks)
        staerkste = max(kategorien.values()) / max(len(landmarks), 1)
        # Ohne die Kategorie-Daempfung lagen 70 % auf "kueste" (gemessen).
        ok &= check(f"  keine Kategorie ueber 65 % (Daempfung {KATEGORIE_WIEDERHOLUNG})",
                    staerkste <= 0.65, f"{staerkste:.0%}, {dict(kategorien)}")

        gipfel = [l for l in landmarks if l.properties["kategorie"] == "gipfel"]
        if gipfel:
            perz = [float((H[land] < H[int(l.y), int(l.x)]).mean()) for l in gipfel]
            ok &= check("  'gipfel' liegen wirklich hoch",
                        float(np.mean(perz)) > 0.85,
                        f"Mittel {np.mean(perz):.1%}-Perzentil")
        kueste = [l for l in landmarks if l.properties["kategorie"] == "kueste"]
        if kueste:
            dd = [float(dist[int(l.y), int(l.x)]) for l in kueste]
            ok &= check("  'kueste' liegen wirklich am Meer",
                        float(np.mean(dd)) < 8.0, f"Mittel {np.mean(dd):.1f} px")
    return ok


def run_determinismus():
    H, slope, civ, water, staedte, felder = _welt(256, 4242)
    a = _generator().calculate_landmarks(civ, H, slope, water, staedte, "FINAL",
                                          region_map=felder["regionen"])
    b = _generator().calculate_landmarks(civ, H, slope, water, staedte, "FINAL",
                                          region_map=felder["regionen"])
    return check("gleicher Seed -> gleiche Landmarks",
                 [(l.x, l.y, l.properties["kategorie"]) for l in a]
                 == [(l.x, l.y, l.properties["kategorie"]) for l in b],
                 f"{len(a)} Stueck")


if __name__ == "__main__":
    ergebnisse = {
        "eignungskarten": run_eignungskarten(),
        "platzierung": run_platzierung(),
        "determinismus": run_determinismus(),
    }
    print("\n=== SUMMARY ===")
    for name, bestanden in ergebnisse.items():
        print(f"{name}: {'PASS' if bestanden else 'FAIL'}")
    sys.exit(0 if all(ergebnisse.values()) else 1)
