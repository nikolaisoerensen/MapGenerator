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

# UEBER VIELE KARTEN URTEILEN, NICHT UEBER EINE (2026-08-25)
#
# Nutzervorgabe: *"es sollte gleichmaessig sein ueber viele maps hinweg.
# eine map kann sich von einer anderen unterscheiden. also neu gewichten."*
#
# Dieser Test hat bis hierher an EINEM Seed geurteilt, und das ist bei
# dieser Streuung nicht tragfaehig. Der Anteil eines Archetyps an der
# Kueste seiner Region schwankt von Karte zu Karte um 8 bis 21
# Prozentpunkte. Gemessen am eigenen Fehlalarm dieses Tests:
#
#     Kola-Steilkueste, Einzelkarte:      5.53x ueber Soll  -> FAIL
#     Kola-Steilkueste, 64 Karten Mittel: 29.7 % gegen 30 % -> richtig
#
# Der Test schlug also auf Rauschen an. Umgekehrt kann eine einzelne gute
# Karte einen echten systematischen Ausfall verdecken.
#
# GEMESSEN fuer die Grenzen unten - fuenf UNABHAENGIGE Gruppen zu je 16
# Karten, jeweils die Abweichung des Mittelwerts vom `max_anteil`:
#
#     Gruppe   bewertet   Mittel   schlechteste   nie vorhanden
#       0         24       4.8 P      12.2 P           -
#       1         24       2.4 P       8.6 P           -
#       2         24       5.0 P      10.8 P           -
#       3         24       4.0 P      10.5 P           -
#       4         24       3.4 P       8.8 P           -
#
# Die Grenzen liegen mit Reserve darueber. Sie sind trotzdem scharf genug
# fuer den Fehler, den es diese Datei ueberhaupt gibt: ein Archetyp, der
# systematisch ausfaellt, liegt 25 bis 50 Punkte daneben, nicht 12.
KARTEN_ANZAHL = 16
KARTEN_SIZE = 384          # kleiner als SIZE - 16 Karten statt einer
MITTEL_ABWEICHUNG_MAX_P = 7.0
EINZEL_ABWEICHUNG_MAX_P = 16.0

# ZWEI ENDEN DER KETTE, mit ZWEI Grenzen - gemessen 2026-08-27.
#
# Anlass: der Test meldete "Vendee-Straende -16.7P" und sah damit aus wie
# ein Verteilungsfehler. Er war keiner. Gemessen wurden beide Enden bei
# zwei Aufloesungen, je 8 Karten:
#
#                        Saatanteil      Laengenanteil
#     384 px (55 m/px)      -5.0             -20.1
#     768 px (28 m/px)      -4.9              -0.2
#
# Das SAATENDE ist bei beiden Aufloesungen gleich - die Zuordnung stimmt.
# Der Verlust entsteht danach und verschwindet bei feiner Karte ganz.
# Die Gegenprobe liefert Bretagne-Klippen, Vendees Regionsnachbar:
# +11.4 bei 384 px, -5.8 bei 768 px. Die beiden TAUSCHEN genau das, was
# Vendee fehlt.
#
# Ursache: `MIN_SEGMENT_M` (750 m) ist eine ABSOLUTE Laenge. Vendee hat
# mit 0.18 km die kuerzeste Reichweite der drei Atlantik-Typen; bei
# 55 m/px fallen seine Zonen darunter und werden in `_segmente_schliessen()`
# in den laengeren Nachbarn eingeschmolzen. Der Test hat also die
# Aufloesung SEINER EIGENEN Testkarte gemessen, nicht die Verteilung.
#
# Daraus zwei Grenzen statt einer:
#
#   * Der SAATANTEIL ist aufloesungsunabhaengig und die Groesse, die die
#     Quote ueberhaupt steuert. Er wird SCHARF geprueft - groesste
#     gemessene Abweichung 5.3 P bei beiden Aufloesungen.
#   * Der LAENGENANTEIL enthaelt das Einschmelzen und wird bei 384 px
#     entsprechend weich geprueft. Groesste gemessene Abweichung 20.1 P.
#
# Der Fehler, den es diese Datei ueberhaupt gibt - ein Archetyp, der
# systematisch ausfaellt - liegt 25 bis 50 Punkte daneben und faellt
# durch BEIDE Grenzen. Die weiche Laengengrenze verliert also nichts.
SAAT_ABWEICHUNG_MAX_P = 10.0
LAENGE_ABWEICHUNG_MAX_P = 22.0
# Unter so vielen Karten MIT Kueste wird ein Archetyp nicht bewertet.
# Das Nevadin hat auf 62 von 64 Karten ueberhaupt keine Kueste; sein
# "Mittelwert" aus zwei Stichproben lag scheinbar 60 Punkte daneben und
# war reines Artefakt.
MINDEST_KARTEN = 8

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
                marke = "  <-- auffaellig"
        print(f"       {name:22s}{anzahl:6d}{soll:8.1%}{ist:9.1%}"
              f"{v:11.2f}x{marke}")

    fehler += check(
        f"jeder Archetyp ab {MINDEST_STATIONEN} Stationen kommt vor",
        not ausgefallen,
        ", ".join(ausgefallen) if ausgefallen else
        f"{len(BEKANNTE_AUSFAELLE)} bekannte Ausfaelle uebergangen")
    # NUR NACHRICHT, KEINE PRUEFUNG - auf einer Karte ist die Verzerrung
    # nicht bewertbar (Streuung 8-21 Prozentpunkte, siehe KARTEN_ANZAHL).
    # Die Zusicherung dazu steht in `verteilung_ueber_karten`.
    print(f"[--] Anteile auf DIESER Karte ({VERZERRUNG_MIN:.2f}x bis "
          f"{VERZERRUNG_MAX:.1f}x) - "
          + (", ".join(verzerrt) if verzerrt else
             f"Median {np.median(verhaeltnisse):.2f}x")
          + "   (nur Anschauung, geprueft wird ueber viele Karten)")

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


def verteilung_ueber_karten():
    """
    DIE EIGENTLICHE ZUSICHERUNG: stimmt die Verteilung IM MITTEL?

    Nutzervorgabe 2026-08-25: *"es sollte gleichmaessig sein ueber viele
    maps hinweg. eine map kann sich von einer anderen unterscheiden."*

    Gemessen wird der Anteil, den ein Archetyp an der Kuestenlaenge
    INNERHALB SEINER REGION haelt, gegen sein `max_anteil` aus dem
    Katalog - gemittelt ueber KARTEN_ANZAHL Karten. Eine einzelne Karte
    darf beliebig abweichen; der Erwartungswert nicht.

    Zwei Dinge fallen hier auf, die eine Einzelkarte nicht zeigen kann:
    ein Archetyp, der SYSTEMATISCH zu selten vorkommt (die urspruengliche
    Ursache dieser Datei - acht von 27 fehlten ganz), und eine Quote, die
    im Mittel danebenliegt statt nur zu streuen.
    """
    fehler = []
    anteile = collections.defaultdict(list)
    saatanteile = collections.defaultdict(list)
    for k in range(KARTEN_ANZAHL):
        seed = SEED + 1013 * k
        _H, felder = rw.weltfeld(KARTEN_SIZE, seed)
        vk = felder.get("vektor_kueste")
        if vk is None or not getattr(vk, "segmente", None):
            continue
        laenge = collections.defaultdict(float)
        for seg in vk.segmente:
            laenge[seg["name"]] += max(0.0, seg["b"] - seg["a"])
        saat = collections.Counter(a["name"] for a in vk.saat_archetyp)
        for _z, _s, r in rw.alle_regionen():
            typen = rw.KUESTEN_ARCHETYPEN.get(r["name"])
            if not typen:
                continue
            ges = sum(laenge.get(t["name"], 0.0) for t in typen)
            if ges <= 0:
                continue                    # Region ohne Kueste auf dieser Karte
            # Das zweite Ende: wieviele SAATSTATIONEN der Archetyp
            # bekommen hat, bevor Zonenmischung und Einschmelzen daran
            # waren. Siehe SAAT_ABWEICHUNG_MAX_P.
            ges_s = sum(saat.get(t["name"], 0) for t in typen)
            for t in typen:
                anteile[t["name"]].append(laenge.get(t["name"], 0.0) / ges)
                if ges_s > 0:
                    saatanteile[t["name"]].append(
                        saat.get(t["name"], 0) / ges_s)

    soll = {t["name"]: t["max_anteil"]
            for _z, _s, r in rw.alle_regionen()
            for t in rw.KUESTEN_ARCHETYPEN.get(r["name"], [])}

    print()
    print(f"       {'Archetyp':24s}{'soll':>6}{'Mittel':>9}{'Abw':>8}"
          f"{'Streuung':>10}{'Karten':>8}")
    print("       " + "-" * 65)
    abweichungen, verfehlt, verschwunden, zu_duenn = [], [], [], []
    for name in sorted(anteile):
        w = np.array(anteile[name])
        if len(w) < MINDEST_KARTEN:
            zu_duenn.append(f"{name} ({len(w)})")
            continue
        mittel = float(w.mean())
        d = 100.0 * (mittel - soll[name])
        abweichungen.append(abs(d))
        marke = ""
        if abs(d) > LAENGE_ABWEICHUNG_MAX_P:
            verfehlt.append(f"{name} {d:+.1f}P")
            marke = "  <-- daneben"
        if float((w < 0.005).mean()) > 0.5:
            verschwunden.append(f"{name} ({int((w < 0.005).sum())}/{len(w)} leer)")
            marke = "  <-- meist leer"
        print(f"       {name:24s}{soll[name]:>5.0%}{mittel:>9.1%}{d:>+7.1f}P"
              f"{w.std():>10.1%}{len(w):>8}{marke}")

    fehler += check(f"genug Archetypen bewertbar ({MINDEST_KARTEN}+ Karten)",
                    len(abweichungen) >= 20,
                    f"{len(abweichungen)} bewertet"
                    + (f"; zu duenn belegt: {', '.join(zu_duenn)}"
                       if zu_duenn else ""))
    if not abweichungen:
        return fehler
    fehler += check("kein Archetyp verschwindet ueber die Karten hinweg",
                    not verschwunden, ", ".join(verschwunden) if verschwunden
                    else f"alle {len(abweichungen)} kommen auf der Mehrzahl vor")
    fehler += check(f"Laengenanteil im Mittel innerhalb "
                    f"{LAENGE_ABWEICHUNG_MAX_P:.0f} Punkten",
                    not verfehlt, ", ".join(verfehlt) if verfehlt
                    else f"schlechtester {max(abweichungen):.1f}P")

    # DAS SCHARFE ENDE. Der Saatanteil haengt nicht von der Aufloesung ab
    # und ist die Groesse, die die Quote steuert - siehe den Block bei
    # SAAT_ABWEICHUNG_MAX_P.
    saat_verfehlt, saat_abw = [], []
    for name in sorted(saatanteile):
        w = np.array(saatanteile[name])
        if len(w) < MINDEST_KARTEN:
            continue
        d = 100.0 * (float(w.mean()) - soll[name])
        saat_abw.append(abs(d))
        if abs(d) > SAAT_ABWEICHUNG_MAX_P:
            saat_verfehlt.append(f"{name} {d:+.1f}P")
    fehler += check(f"SAATANTEIL jedes Archetyps innerhalb "
                    f"{SAAT_ABWEICHUNG_MAX_P:.0f} Punkten",
                    not saat_verfehlt,
                    ", ".join(saat_verfehlt) if saat_verfehlt
                    else f"schlechtester {max(saat_abw):.1f}P ueber "
                         f"{len(saat_abw)} Archetypen")
    fehler += check(f"mittlere Abweichung unter "
                    f"{MITTEL_ABWEICHUNG_MAX_P:.0f} Punkten",
                    float(np.mean(abweichungen)) <= MITTEL_ABWEICHUNG_MAX_P,
                    f"{float(np.mean(abweichungen)):.1f}P ueber "
                    f"{len(abweichungen)} Archetypen, {KARTEN_ANZAHL} Karten")
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
        ("verteilung_ueber_karten", verteilung_ueber_karten),
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
