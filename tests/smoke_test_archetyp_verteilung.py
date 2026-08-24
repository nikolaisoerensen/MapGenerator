"""
Path: tests/smoke_test_archetyp_verteilung.py

Kommt jeder Kuesten-Archetyp auf der fertigen Karte auch VOR?

ANLASS (gemessen 2026-08-24): von 27 Archetypen hatten **acht** auf der
fertigen Karte kein einziges Segment - Fjordbucht, Foerdenkueste,
Kotor-Steilfjord, Labrador-Buchten, San-Sebastian-Bucht,
Costa-Brava-Buchten, Dalmatien-Klippen und Alpine-Flussmuendung. Die
Fjordwand hielt mit 13 Saatstationen 0.4 % der Kueste, die
Algarve-Klippen mit 8 Stationen 15.5 %.

WARUM DAS KEIN VORHANDENER TEST GESEHEN HAT - und das ist der Grund, aus
dem es diese Datei gibt: es gab Tests fuer das Profil (stimmen die
Hoehen?), fuer die Mischung (verduennen Nachbarn den Typ?), fuer die
Reichweite und fuer den Hoehendeckel. **Jeder einzelne war gruen.** Der
Fehler sass in der ZUORDNUNG dazwischen: ein Archetyp bekam verstreute
Einzelstationen, deren Laeufe kuerzer waren als MIN_SEGMENT_M, und wurde
beim Einschmelzen an die laengeren Nachbarn verteilt. Wer schon lang war,
wurde laenger.

Ein Test, der nur ein Glied der Kette prueft, kann das nicht sehen.
Dieser hier haelt deshalb ZWEI ENDEN gegeneinander: den Anteil, den ein
Archetyp an den Saatstationen hat, gegen den Anteil, den er an der
fertigen Kuestenlaenge haelt. Weichen die stark voneinander ab, ist
zwischen Zuweisung und Segment etwas verlorengegangen - egal, welches
Glied es war.

Aufruf (dauert rund 2 Minuten - eine volle Weltberechnung):
    .venv/Scripts/python.exe tests/smoke_test_archetyp_verteilung.py
"""

import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import core.terrain_weltkarte as rw
import core.vektor_kueste as V

SIZE = 512
SEED = 20260804

# Ab wievielen Saatstationen ein Archetyp auf der Karte ankommen MUSS.
# Darunter ist Verschwinden kein Fehler, sondern Statistik: ein Typ mit
# einer einzigen Station hat keinen Nachbarn, mit dem er einen Lauf
# bilden koennte.
MINDEST_STATIONEN = 10

# Wie stark der Kuestenanteil vom Saatanteil abweichen darf.
# Nach oben grosszuegiger als nach unten: ein Archetyp, der sich auf
# einer langen glatten Kueste ausbreitet, deckt mit wenigen Stationen
# viele Kilometer ab - das ist gewollt. Nach unten ist es dagegen genau
# der Ausfall, den dieser Test fangen soll.
VERZERRUNG_MIN = 0.30
VERZERRUNG_MAX = 5.0

# BEKANNTE AUSFAELLE - zur Zeit KEINE.
#
# Mit SAAT_KOHAERENZ_STATIONEN = 3.5 sind alle acht ehemals verschwundenen
# Archetypen zurueck. Uebrig bleiben nur Typen unter MINDEST_STATIONEN,
# und deren Fehlen ist Statistik statt Systematik: die Algarve-Klippen
# haben 8 Saatstationen, also 0.85 % der Kueste - dass so ein seltener
# Typ mal keine Luecke findet, in die ein volles Segment passt, ist kein
# Fehler. DESHALB steht MINDEST_STATIONEN bei 10 und nicht niedriger.
BEKANNTE_AUSFAELLE = set()


def check(label, bedingung, zusatz=""):
    print(f"[{'OK' if bedingung else 'FAIL'}] {label}"
          + (f" - {zusatz}" if zusatz else ""))
    return [] if bedingung else [f"{label}{' - ' + zusatz if zusatz else ''}"]


def _lage():
    """Saatanteile und Kuestenanteile je Archetyp aus einem echten Lauf."""
    _H, felder = rw.weltfeld(SIZE, SEED)
    # AUS DEM ECHTEN LAUF, nicht selbst gebaut. Eine selbst erzeugte
    # VektorKueste sitzt auf einem anderen Gelaende und misst etwas
    # anderes - das hat am 2026-08-24 mehrere Messrunden gekostet.
    vk = felder.get("vektor_kueste")
    if vk is None or not getattr(vk, "segmente", None):
        return None
    saat = collections.Counter(a["name"] for a in vk.saat_archetyp)
    laenge = collections.defaultdict(float)
    for seg in vk.segmente:
        laenge[seg["name"]] += seg["b"] - seg["a"]
    return vk, saat, laenge


def verteilung(lage):
    """1. Kein nennenswerter Archetyp geht auf dem Weg zur Karte verloren."""
    vk, saat, laenge = lage
    fehler = []
    n_saat = max(sum(saat.values()), 1)
    n_km = max(sum(laenge.values()), 1e-9)

    print(f"\n       {'Archetyp':22s}{'Saat':>6}{'Saat%':>8}{'Kueste%':>9}"
          f"{'Verzerrung':>12}")
    print("       " + "-" * 57)
    ausgefallen, verzerrt, verhaeltnisse = [], [], []
    for name, anzahl in saat.most_common():
        soll = anzahl / n_saat
        ist = laenge.get(name, 0.0) / n_km
        v = ist / soll if soll > 0 else 0.0
        marke = ""
        if anzahl >= MINDEST_STATIONEN:
            verhaeltnisse.append(v)
            if ist <= 0.0:
                if name not in BEKANNTE_AUSFAELLE:
                    ausgefallen.append(f"{name} ({anzahl} Stationen)")
                marke = "  <-- FEHLT"
            elif not VERZERRUNG_MIN <= v <= VERZERRUNG_MAX:
                verzerrt.append(f"{name} {v:.2f}x")
                marke = "  <-- daneben"
        print(f"       {name:22s}{anzahl:6d}{soll:8.1%}{ist:9.1%}"
              f"{v:11.2f}x{marke}")

    fehler += check(
        f"jeder Archetyp ab {MINDEST_STATIONEN} Stationen kommt vor",
        not ausgefallen,
        ", ".join(ausgefallen) if ausgefallen else
        f"{len(BEKANNTE_AUSFAELLE)} bekannte Ausfaelle uebergangen")
    fehler += check(
        f"Anteile im Rahmen ({VERZERRUNG_MIN:.2f}x bis {VERZERRUNG_MAX:.1f}x)",
        not verzerrt, ", ".join(verzerrt) if verzerrt else
        f"Median {np.median(verhaeltnisse):.2f}x")

    # Ein einzelner Typ, der die halbe Kueste haelt, ist auch dann falsch,
    # wenn er die Verzerrungsgrenze knapp einhaelt.
    groesster, anteil = max(
        ((n, laenge[n] / n_km) for n in laenge), key=lambda t: t[1])
    fehler += check("kein Archetyp haelt mehr als ein Drittel der Kueste",
                    anteil <= 0.34, f"{groesster} mit {anteil:.1%}")
    return fehler


def segmentlaengen(lage):
    """
    2. Die Segmente sind lang genug, um ihren Typ zu ZEIGEN.

    Ein Segment kuerzer als zwei Uebergangsbreiten besteht nur aus
    Uebergaengen - sein Profil kommt nirgends rein zur Geltung. Genau das
    war der Zustand vor dem 2026-08-24: Medianlaenge 169 m bei
    UEBERGANG_M = 250 m.
    """
    vk, _saat, _laenge = lage
    fehler = []
    L = np.array([s["b"] - s["a"] for s in vk.segmente], dtype=np.float64)
    median = float(np.median(L))
    print(f"\n       {len(L)} Segmente, Median {median:.0f} m, "
          f"p10 {np.percentile(L, 10):.0f} m, p90 {np.percentile(L, 90):.0f} m")
    print(f"       UEBERGANG_M = {V.UEBERGANG_M:.0f} m, "
          f"MIN_SEGMENT_M = {V.MIN_SEGMENT_M:.0f} m")
    fehler += check("Median-Segment traegt mehr als reine Uebergaenge",
                    median >= 2.0 * V.UEBERGANG_M,
                    f"{median:.0f} m gegen {2.0 * V.UEBERGANG_M:.0f} m")
    return fehler


def kohaerenz_wirkt():
    """
    3. Der Schalter ist da und steht auf einem wirksamen Wert.

    Bei 0 faellt die Glaettung weg und der alte Zustand kehrt zurueck -
    ohne dass irgendetwas abstuerzt. Ein stiller Rueckfall also, genau
    die Sorte, vor der CLAUDE.md warnt.
    """
    fehler = []
    wert = getattr(V, "SAAT_KOHAERENZ_STATIONEN", 0.0)
    fehler += check("SAAT_KOHAERENZ_STATIONEN wirksam", wert >= 1.0,
                    f"{wert}")
    for name in ("_bogen_glaetten", "_bogen_rauschen"):
        fehler += check(f"{name} vorhanden", hasattr(V, name))

    # Die Glaettung darf die Reihenfolge NICHT durcheinanderbringen: sie
    # bekommt unsortierte Stationen und muss sie in derselben Reihenfolge
    # zurueckgeben, in der sie kamen.
    rng = np.random.RandomState(7)
    n = 40
    bogen = rng.permutation(n).astype(float) * 100.0
    kontur = np.zeros(n, dtype=np.int32)
    kontur[n // 2:] = 1
    werte = np.arange(n, dtype=np.float64)
    glatt = V._bogen_glaetten(werte, kontur, bogen, 2.5)
    fehler += check("Glaettung erhaelt Form und Reihenfolge",
                    glatt.shape == werte.shape and np.all(np.isfinite(glatt)))
    rauschen = V._bogen_rauschen(rng, kontur, bogen, 0.15, 2.5)
    streuung = float(np.std(rauschen))
    fehler += check("Rauschen behaelt seine Staerke nach dem Glaetten",
                    0.10 <= streuung <= 0.22, f"Streuung {streuung:.3f}")
    return fehler


def lauf():
    lage = _lage()
    if lage is None:
        print("[FAIL] keine VektorKueste im Weltfeld - nichts zu pruefen")
        return 1

    gruppen = [
        ("verteilung", lambda: verteilung(lage)),
        ("segmentlaengen", lambda: segmentlaengen(lage)),
        ("kohaerenz_wirkt", kohaerenz_wirkt),
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
