"""
Path: tests/smoke_test_stadttypen.py

Prueft die Stadttypen und Handelsgewichte (Nutzer-Vorgabe 2026-08-13,
docs/OFFENE_PUNKTE.md 5.16):

  Bergdorf   "In den Bergen", klein bis mittel
  Marktstadt "am Wasser oder erreicht viele Staedte", NUR EINE JE REGION,
             mittel bis gross
  Agrarstadt "viel flaches Land und fruchtbare Biome in der Naehe", mittel
             bis gross
  sonstige   alles uebrige

Handelsgewicht einer Kante = SUMME beider einseitiger Interessen
(Nutzerentscheidung; Alternativen waeren Maximum oder Mittel gewesen).

Laeuft mit echtem Weltgelaende und echten Kartengroessen.
"""
import sys

import numpy as np

sys.path.insert(0, ".")

from core.terrain_weltkarte import weltfeld, WELT_KM
from core.settlement_generator import (
    SettlementGenerator, TerrainSuitabilityAnalyzer, Location,
    STADTTYPEN, TYP_GRUNDGUETE, handelsgewicht, RANG_ZAHL)


def check(label, bedingung, zusatz=""):
    status = "OK" if bedingung else "FAIL"
    print(f"[{status}] {label}{(' - ' + zusatz) if zusatz else ''}")
    return bool(bedingung)


def _gelaende(size, seed):
    H, felder = weltfeld(size, seed)
    gy, gx = np.gradient(H.astype(np.float32), WELT_KM * 1000.0 / size)
    slope = np.stack([gx, gy], axis=-1).astype(np.float32)
    return H, slope, felder


def _ort(i, x, y, kultur="A"):
    return Location(location_id=i, x=float(x), y=float(y),
                    location_type="settlement", radius=3.0, civ_influence=0.7,
                    properties={}, culture=kultur)


def run_eignungskarten():
    """Form, Wertebereich und Landmaske je Typ - plus die Verteilung, an der
    ein frueherer Fehler sichtbar wurde."""
    ok = True
    for seed in (20260804, 12345):
        H, slope, _ = _gelaende(384, seed)
        an = TerrainSuitabilityAnalyzer(1.0, 384)
        eig = an.stadttyp_eignungen(H, slope, np.zeros_like(H, dtype=np.float32))
        land = H > 0

        ok &= check(f"Seed {seed}: alle vier Typen vorhanden",
                    set(eig) == set(STADTTYPEN))
        for typ, karte in eig.items():
            ok &= check(f"  {typ}: Form/Bereich/Land-Maske",
                        karte.shape == H.shape
                        and float(karte.min()) >= 0.0 and float(karte.max()) <= 1.0
                        and bool(np.all(karte[~land] == 0.0)))

        # BERGDORF DARF NICHT VERSCHWINDEN. Der erste Anlauf leitete die
        # Bergigkeit aus `1 - hoehe_suit` ab - das ist eine EIGNUNG, keine
        # Hoehe, ihr Median lag bei 0.94 und nur 2.6 % der Landflaeche kamen
        # ueberhaupt als Bergdorf in Frage, obwohl 34.6 % des Landes ueber
        # 200 m liegen. Diese Zusicherung fangt genau diesen Rueckfall.
        anteil_berg = float((eig["bergdorf"][land] > TYP_GRUNDGUETE).mean())
        ok &= check(f"  Bergdorf-Eignung auf mind. 5 % des Landes ueber der Grundguete",
                    anteil_berg >= 0.05, f"{anteil_berg:.1%}")
    return ok


def run_zuweisung_trifft_die_lage():
    """An der besten Bergposition muss ein Bergdorf entstehen, an der besten
    Kuestenposition eine Marktstadt, am besten Ackerland eine Agrarstadt."""
    H, slope, _ = _gelaende(384, 20260804)
    an = TerrainSuitabilityAnalyzer(1.0, 384)
    eig = an.stadttyp_eignungen(H, slope, np.zeros_like(H, dtype=np.float32))
    land = H > 0

    def beste_position(karte):
        maskiert = np.where(land, karte, -1.0)
        yi, xi = np.unravel_index(np.argmax(maskiert), maskiert.shape)
        return float(xi), float(yi)

    erwartet = [("bergdorf", beste_position(eig["bergdorf"])),
                ("marktstadt", beste_position(eig["marktstadt"])),
                ("agrarstadt", beste_position(eig["agrarstadt"]))]
    orte = [_ort(i, x, y) for i, (_typ, (x, y)) in enumerate(erwartet)]

    g = SettlementGenerator.__new__(SettlementGenerator)
    raenge = g._typen_zuweisen(orte, eig, ["stadt", "siedlung", "dorf"])

    ok = True
    for (soll_typ, _pos), ort in zip(erwartet, orte):
        ok &= check(f"beste {soll_typ}-Position wird '{soll_typ}'",
                    ort.settlement_type == soll_typ, ort.settlement_type)
    for ort, rang in zip(orte, raenge):
        erlaubt = STADTTYPEN[ort.settlement_type]["rang_erlaubt"]
        ok &= check(f"  {ort.settlement_type}: Rang '{rang}' erlaubt {erlaubt}",
                    rang in erlaubt)
    return ok


def run_nur_eine_marktstadt():
    """Auch wenn viele Orte an guter Wasserlage stehen: hoechstens eine
    Marktstadt je Kultur/Region."""
    H, slope, _ = _gelaende(384, 12345)
    an = TerrainSuitabilityAnalyzer(1.0, 384)
    eig = an.stadttyp_eignungen(H, slope, np.zeros_like(H, dtype=np.float32))
    land = H > 0

    # die zwoelf besten Marktstadt-Positionen - alle waeren fuer sich geeignet
    maskiert = np.where(land, eig["marktstadt"], -1.0)
    flach = np.argsort(maskiert, axis=None)[::-1][:12]
    ys, xs = np.unravel_index(flach, maskiert.shape)
    orte = [_ort(i, x, y) for i, (x, y) in enumerate(zip(xs, ys))]

    g = SettlementGenerator.__new__(SettlementGenerator)
    g._typen_zuweisen(orte, eig, ["stadt"] * len(orte))
    anzahl = sum(1 for o in orte if o.settlement_type == "marktstadt")
    return check("zwoelf Orte an bester Wasserlage -> genau eine Marktstadt",
                 anzahl == 1, f"{anzahl}")


def run_handelsgewichte():
    """Die Zahlen aus der Vorgabe, als Summe beider Sichten."""
    ok = True

    def paar(typ_a, rang_a, typ_b, rang_b, kultur_b="A"):
        a, b = _ort(0, 0, 0, "A"), _ort(1, 1, 1, kultur_b)
        a.settlement_type, a.rank = typ_a, rang_a
        b.settlement_type, b.rank = typ_b, rang_b
        return handelsgewicht(a, b)

    # Bergdorf <-> eigene Marktstadt: 5 (Bergdorf-Sicht) + 3 (Markt-Sicht) = 8
    ok &= check("Bergdorf <-> eigene Marktstadt = 5+3 = 8",
                paar("bergdorf", "dorf", "marktstadt", "stadt") == 8.0,
                f"{paar('bergdorf','dorf','marktstadt','stadt')}")
    # Marktstadt <-> Marktstadt: 10 + 10 = 20, der hoechste Wert ueberhaupt
    ok &= check("Marktstadt <-> Marktstadt = 10+10 = 20",
                paar("marktstadt", "stadt", "marktstadt", "stadt") == 20.0)
    # fremde Kultur muss stets schwaecher sein als dieselbe Kultur
    gleich = paar("agrarstadt", "stadt", "sonstige", "stadt", kultur_b="A")
    fremd = paar("agrarstadt", "stadt", "sonstige", "stadt", kultur_b="B")
    ok &= check("gleiche Kultur handelt staerker als fremde",
                gleich > fremd, f"{gleich:.1f} > {fremd:.1f}")
    # groesserer Partner = mehr Handel (bei den groessenabhaengigen Typen)
    klein = paar("agrarstadt", "stadt", "sonstige", "dorf")
    gross = paar("agrarstadt", "stadt", "sonstige", "stadt")
    ok &= check("groesserer Handelspartner erzeugt mehr Handel",
                gross > klein, f"{gross:.1f} > {klein:.1f}")
    # Symmetrie: die Kante ist richtungsunabhaengig
    a, b = _ort(0, 0, 0, "A"), _ort(1, 1, 1, "B")
    a.settlement_type, a.rank = "bergdorf", "dorf"
    b.settlement_type, b.rank = "marktstadt", "stadt"
    ok &= check("Kantengewicht ist symmetrisch",
                handelsgewicht(a, b) == handelsgewicht(b, a))
    return ok


def run_erreichbarkeit():
    """Wegkosten-Matrix und Zentralitaet (docs/OFFENE_PUNKTE.md 5.17) - die
    Grundlage fuer "Marktstadt ... kann viele Staedte gut erreichen"."""
    import time
    from core.settlement_generator import (bau_kostenfeld, erreichbarkeits_matrix,
                                            zentralitaet)

    H, slope, _ = _gelaende(512, 20260804)
    feld = bau_kostenfeld(H, slope, 1.0)
    land = np.argwhere(H > 30)
    rng = np.random.RandomState(7)
    punkte = [land[rng.randint(len(land))] for _ in range(6)]
    positionen = [(float(p[1]), float(p[0])) for p in punkte]

    t0 = time.time()
    matrix = erreichbarkeits_matrix(feld, positionen)
    dauer = time.time() - t0
    z = zentralitaet(matrix)

    ok = True
    ok &= check("Matrix hat die richtige Form", matrix.shape == (6, 6))
    ok &= check("Diagonale ist 0 (Weg zu sich selbst)",
                bool(np.all(np.diag(matrix) == 0.0)))
    # Symmetrie nur naeherungsweise: das grobe Gitter rundet Start- und
    # Zielpixel, Hin- und Rueckweg koennen dadurch minimal abweichen.
    endlich = np.isfinite(matrix)
    ok &= check("Matrix ist naeherungsweise symmetrisch",
                bool(np.allclose(matrix[endlich], matrix.T[endlich], rtol=0.05)))
    ok &= check("Zentralitaet ist auf 0..1 normiert",
                float(z.min()) >= -1e-6 and float(z.max()) <= 1.0 + 1e-6,
                f"{z.min():.2f}..{z.max():.2f}")
    # Der zentralste Ort muss die kleinste Kostensumme haben - sonst ist
    # irgendwo ein Vorzeichen verdreht (kleine Kosten = gut erreichbar).
    summen = np.where(endlich, matrix, 0).sum(axis=1)
    ok &= check("hoechste Zentralitaet = kleinste Kostensumme",
                int(np.argmax(z)) == int(np.argmin(summen)))
    ok &= check("laeuft schnell genug fuer den Platzierungsschritt",
                dauer < 2.0, f"{dauer:.2f}s")

    # Die Erreichbarkeit muss die Marktstadt-Wahl auch WIRKLICH beeinflussen
    # koennen - sonst waere der ganze Aufwand wirkungslos.
    an = TerrainSuitabilityAnalyzer(1.0, 512)
    eig = an.stadttyp_eignungen(H, slope, np.zeros_like(H, dtype=np.float32))
    g = SettlementGenerator.__new__(SettlementGenerator)
    import logging
    g.logger = logging.getLogger("smoke")
    raenge = ["stadt", "siedlung", "siedlung", "dorf", "dorf", "dorf"]

    def markt_index(kostenfeld):
        orte = [_ort(i, x, y) for i, (x, y) in enumerate(positionen)]
        g._typen_zuweisen(orte, eig, list(raenge), kostenfeld=kostenfeld)
        treffer = [i for i, o in enumerate(orte) if o.settlement_type == "marktstadt"]
        return treffer[0] if treffer else None

    mit = markt_index(feld)
    ohne = markt_index(None)
    ok &= check("mit Erreichbarkeit wird eine Marktstadt gewaehlt", mit is not None)
    ok &= check("ohne Kostenfeld faellt es sauber auf die Wasserlage zurueck",
                ohne is not None)
    print(f"     Marktstadt mit Erreichbarkeit: Ort {mit} (Zentralitaet "
          f"{z[mit]:.2f}), ohne: Ort {ohne}")
    return ok


def run_steigungskosten():
    """Exponentielle Steigungskosten (docs/OFFENE_PUNKTE.md 5.19, Nutzerbefund
    "die hoehenkosten sind zu niedrig. es gibt strassen die ueber hohe berge
    gehen")."""
    from skimage.graph import route_through_array, MCP_Geometric
    from core.settlement_generator import (bau_kostenfeld, STEIGUNG_SKALA_GRAD,
                                            MAX_WEG_STEIGUNG_GRAD, WEGEBAU_UNMOEGLICH,
                                            WASSER_SPERRE_M, WASSERKOSTEN_FLACH,
                                            WASSERKOSTEN_TIEF)
    ok = True

    # Die Kostenkurve selbst
    def neu(w):
        k = 1.0 + 1.5 * np.expm1(w / STEIGUNG_SKALA_GRAD)
        return max(k, WEGEBAU_UNMOEGLICH) if w >= MAX_WEG_STEIGUNG_GRAD else k

    ok &= check("ebener Grund kostet 1.0", abs(neu(0.0) - 1.0) < 1e-9)
    ok &= check("Kosten wachsen streng monoton mit der Steigung",
                all(neu(a) < neu(b) for a, b in zip(range(0, 29), range(1, 30))))
    # Die eigentliche Nutzer-Vorgabe: "5 Grad weniger ist wesentlich besser"
    faktor = neu(20.0) / neu(15.0)
    ok &= check("5 Grad Unterschied kosten deutlich mehr (>1.5x)",
                faktor > 1.5, f"15->20 Grad: Faktor {faktor:.2f}")
    ok &= check(f"ab {MAX_WEG_STEIGUNG_GRAD} Grad praktisch unpassierbar",
                neu(MAX_WEG_STEIGUNG_GRAD) >= WEGEBAU_UNMOEGLICH)
    # ENDLICH, nicht inf - sonst koennen Landesteile unerreichbar werden
    ok &= check("Sperre ist endlich (sonst waeren Orte abschneidbar)",
                np.isfinite(neu(60.0)))

    # Wirkung am echten Gelaende: flacher als mit der alten Formel
    H, slope, _ = _gelaende(384, 20260804)
    winkel = np.degrees(np.arctan(np.hypot(slope[..., 0], slope[..., 1])))
    land = H > 0
    feld_neu = bau_kostenfeld(H, slope, 1.5)

    hang = np.hypot(slope[..., 0], slope[..., 1]).astype(np.float64)
    feld_alt = 1.0 + 1.5 * hang ** 2
    h = H.astype(np.float64)
    feld_alt = np.where((h <= 0) & (h > -5), WASSERKOSTEN_FLACH, feld_alt)
    feld_alt = np.where((h <= -5) & (h > WASSER_SPERRE_M), WASSERKOSTEN_TIEF, feld_alt)
    feld_alt = np.where(h <= WASSER_SPERRE_M, np.inf, feld_alt)

    punkte = np.argwhere(H > 80)
    rng = np.random.RandomState(5)
    paare = [(punkte[rng.randint(len(punkte))], punkte[rng.randint(len(punkte))])
             for _ in range(8)]

    def steigungen(feld):
        f = np.where(np.isfinite(feld), feld, 1e7)
        werte = []
        for a, b in paare:
            try:
                idx, _ = route_through_array(f, tuple(a), tuple(b),
                                              fully_connected=True, geometric=True)
                werte.extend(winkel[y, x] for y, x in idx)
            except Exception:
                pass
        return np.array(werte)

    alt = steigungen(feld_alt)
    neu_w = steigungen(feld_neu)
    if len(alt) and len(neu_w):
        ok &= check("Wege werden flacher als mit der alten Formel",
                    float(np.median(neu_w)) < float(np.median(alt)),
                    f"Median {np.median(alt):.1f} -> {np.median(neu_w):.1f} Grad")
        ok &= check("deutlich weniger sehr steile Wegstuecke (>20 Grad)",
                    float((neu_w > 20).mean()) < float((alt > 20).mean()) * 0.75,
                    f"{(alt > 20).mean():.1%} -> {(neu_w > 20).mean():.1%}")

    # Und trotzdem bleibt alles erreichbar
    f = np.where(np.isfinite(feld_neu), feld_neu, 1e7)
    kosten, _ = MCP_Geometric(f, fully_connected=True).find_costs([tuple(punkte[0])])
    anteil = float((np.isfinite(kosten) & land).sum()) / float(land.sum())
    ok &= check("Land bleibt vollstaendig erreichbar (Sperre schneidet nichts ab)",
                anteil > 0.99, f"{anteil:.1%}")
    return ok


def run_kreuzungsgrade():
    """Kreuzungsgrad (docs/OFFENE_PUNKTE.md 5.20) - Grundlage fuer die
    Nutzer-Vorgabe "taverne ... an einer kreuzung mit min. drei wegen".

    Synthetische Wege statt eines echten Netzes: hier wird GEZAEHLT, und ob
    ein Stern vier und eine Beruehrung zwei Wege hat, muss exakt stimmen. An
    einem generierten Netz waere die erwartete Zahl selbst unbekannt - der
    Test koennte dann nur sich selbst bestaetigen.
    """
    from core.settlement_generator import (kreuzungen_finden, kreuzungsgrade,
                                            KREUZUNG_MIN_WEGE)
    shape = (100, 100)
    faelle = {
        "Stern, 4 Wege": ([[(50, y) for y in range(10, 51)],
                            [(50, y) for y in range(50, 91)],
                            [(x, 50) for x in range(10, 51)],
                            [(x, 50) for x in range(50, 91)]], 4),
        "T-Kreuzung, 3 Wege": ([[(20, y) for y in range(5, 21)],
                                 [(x, 20) for x in range(5, 21)],
                                 [(x, 20) for x in range(20, 36)]], 3),
        "blosse Beruehrung, 2 Wege": ([[(80, y) for y in range(70, 81)],
                                        [(x, 80) for x in range(80, 91)]], 2),
    }
    ok = True
    for name, (wege, erwartet) in faelle.items():
        kreuzungen = kreuzungen_finden(wege, [], [], shape)
        grade = kreuzungsgrade(wege, [], kreuzungen, shape)
        ok &= check(f"{name}: erkannt", len(kreuzungen) >= 1)
        if grade:
            ok &= check(f"  Grad = {erwartet}", max(grade) == erwartet, f"{max(grade)}")
            ist_wegscheide = max(grade) >= KREUZUNG_MIN_WEGE
            soll_wegscheide = erwartet >= KREUZUNG_MIN_WEGE
            ok &= check(f"  gilt als Wegscheide: {soll_wegscheide}",
                        ist_wegscheide == soll_wegscheide)

    # Gemeinsam: Grade und Kreuzungen duerfen nicht auseinanderlaufen
    alle = [w for wege, _e in faelle.values() for w in wege]
    kreuzungen = kreuzungen_finden(alle, [], [], shape)
    grade = kreuzungsgrade(alle, [], kreuzungen, shape)
    ok &= check("Grade und Kreuzungen bleiben deckungsgleich",
                len(grade) == len(kreuzungen), f"{len(grade)} zu {len(kreuzungen)}")
    ok &= check("keine 'Kreuzung' mit weniger als zwei Wegen",
                all(g >= 2 for g in grade), f"{sorted(grade)}")
    wegscheiden = sum(1 for g in grade if g >= KREUZUNG_MIN_WEGE)
    ok &= check("die beiden echten Wegscheiden werden als solche erkannt",
                wegscheiden == 2, f"{wegscheiden}")
    return ok


def run_netzausbau_nach_bedarf():
    """Bedarfsgetriebener Netzausbau (docs/OFFENE_PUNKTE.md 5.21).

    Der Kernfall ist die Nutzerbeobachtung am Kartenbild: zwei Ortsgruppen
    beiderseits eines Berges, verbunden nur ueber einen langen Umweg. Ein
    Pass lohnt sich fuer KEIN einzelnes Paar, fuer alle zusammen aber sehr.
    Genau das wird hier als kleines, exakt nachrechenbares Modell geprueft -
    an einem generierten Netz waere das erwartete Ergebnis unbekannt.
    """
    from core.settlement_generator import (netzdistanzen, umwegfaktoren,
                                            kanten_nach_bedarf)
    ok = True

    # Zwei Ketten a vier Orte, nur unten herum verbunden (Kante 3-4, teuer).
    # Oben waere ein kurzer Pass 0-4 moeglich.
    n = 8
    C = np.full((n, n), np.inf)
    def setz(i, j, k):
        C[i, j] = C[j, i] = k
    for i in range(3):
        setz(i, i + 1, 10)
    for i in range(4, 7):
        setz(i, i + 1, 10)
    setz(3, 4, 90)     # der lange Weg unten herum
    setz(0, 4, 30)     # der kurze Pass oben
    np.fill_diagonal(C, 0)

    W = np.ones((n, n))
    np.fill_diagonal(W, 0)

    bestehend = {(i, i + 1): 10 for i in range(3)}
    bestehend.update({(i, i + 1): 10 for i in range(4, 7)})
    bestehend[(3, 4)] = 90

    D = netzdistanzen(bestehend, n)
    ok &= check("Netzdistanz laeuft ueber die gebauten Kanten, nicht Luftlinie",
                D[0, 4] == 120.0, f"{D[0, 4]}")
    ok &= check("Netzdistanz ist symmetrisch", bool(np.allclose(D, D.T)))
    ok &= check("Weg zu sich selbst kostet nichts",
                bool(np.all(np.diag(D) == 0.0)))

    U = umwegfaktoren(D, C)
    ok &= check("Umwegfaktor erkennt die fehlende Verbindung",
                U[0, 4] > 3.0, f"{U[0, 4]:.1f}x")

    gewaehlt = kanten_nach_bedarf(C, W, bestehend, [(0, 4), (1, 5), (2, 6)],
                                   hoechstens=2)
    ok &= check("der Pass wird gebaut, obwohl kein Einzelpaar ihn erzwingt",
                (0, 4) in gewaehlt, f"{gewaehlt}")

    netz2 = dict(bestehend)
    for i, j in gewaehlt:
        netz2[(i, j)] = C[i, j]
    D2 = netzdistanzen(netz2, n)
    vorher = float(np.sum(np.where(np.isfinite(D), D, 0) * W))
    nachher = float(np.sum(np.where(np.isfinite(D2), D2, 0) * W))
    ok &= check("die gewichteten Gesamtwegkosten sinken deutlich",
                nachher < vorher * 0.8, f"{vorher:.0f} -> {nachher:.0f}")

    # Ohne Bedarf darf NICHTS gebaut werden - sonst waechst jedes Netz
    # unbegrenzt, egal wie sinnlos die Kante ist.
    W_leer = np.zeros((n, n))
    ok &= check("ohne Handelsinteresse wird keine Kante gebaut",
                kanten_nach_bedarf(C, W_leer, bestehend, [(0, 4)]) == [])
    # Eine absurd teure Kante darf den Test nicht bestehen
    C_teuer = C.copy()
    C_teuer[0, 4] = C_teuer[4, 0] = 1e6
    ok &= check("eine viel zu teure Kante wird nicht gebaut",
                kanten_nach_bedarf(C_teuer, W, bestehend, [(0, 4)]) == [])
    ok &= check("die Obergrenze wird eingehalten",
                len(kanten_nach_bedarf(C, W, bestehend,
                                        [(0, 4), (1, 5), (2, 6)], hoechstens=1)) <= 1)
    return ok


def run_seehandel_und_fischersiedlung():
    """Hafen-Umsteigekosten und Fischersiedlung (docs/OFFENE_PUNKTE.md 5.22).

    Geprueft wird die MECHANIK - dass der Regler in die richtige Richtung
    wirkt und aufloesungsunabhaengig ist. Der konkrete 35-%-Wert ist ueber
    vier Karten gemittelt geeicht und je Einzelkarte naturgemaess anders;
    ihn hier hart zu fordern hiesse, eine Zahl zu pruefen, die es je Karte
    gar nicht gibt.
    """
    from core.settlement_generator import (hafenkosten, HAFEN_UMSTEIGEKOSTEN_KM,
        SEEHANDEL_ZIEL, FISCHER_DECKEL, kanten_traffic, seehandel_anteil,
        STADTTYPEN, TerrainSuitabilityAnalyzer)
    ok = True

    # --- Umsteigekosten sind aufloesungsUNabhaengig in km, nicht in Punkten
    ok &= check("Fischersiedlung ist ein eigener Typ",
                "fischersiedlung" in STADTTYPEN)
    ok &= check("  und ausdruecklich klein (nie 'stadt')",
                "stadt" not in STADTTYPEN["fischersiedlung"]["rang_erlaubt"],
                str(STADTTYPEN["fischersiedlung"]["rang_erlaubt"]))

    welt_m = 21300.0
    punkte = {size: hafenkosten(welt_m / size) for size in (320, 512, 1024)}
    ok &= check("Hafenkosten wachsen mit der Aufloesung (gleiche km)",
                punkte[320] < punkte[512] < punkte[1024],
                " < ".join(f"{v:.0f}" for v in punkte.values()))
    # In KILOMETERN muss es bei jeder Groesse dasselbe sein - das ist der Sinn
    km = {size: punkte[size] * (welt_m / size) / 1000.0 for size in punkte}
    ok &= check("in Kilometern gerechnet ist es bei jeder Groesse gleich",
                max(km.values()) - min(km.values()) < 1e-6,
                f"{list(km.values())[0]:.1f} km")

    # --- Der Regler wirkt monoton: mehr Hafenkosten = weniger Seehandel
    n = 6
    L = np.full((n, n), np.inf)
    S = np.full((n, n), np.inf)
    for i in range(2):
        L[i, i+1] = L[i+1, i] = 40
        L[i+3, i+4] = L[i+4, i+3] = 40
    for a in range(3):
        for b in range(3, 6):
            L[a, b] = L[b, a] = 400 + 30 * abs(a - (b - 3))
            S[a, b] = S[b, a] = 60 + 40 * (a + b)      # bewusst STREUEND
    np.fill_diagonal(L, 0)
    W = np.ones((n, n))
    np.fill_diagonal(W, 0)

    def anteil_bei(hk):
        C = np.full((n, n), np.inf)
        seek = set()
        for a in range(n):
            for b in range(a + 1, n):
                see = S[a, b] + 2 * hk
                if see < L[a, b]:
                    C[a, b] = C[b, a] = see
                    seek.add((a, b))
                else:
                    C[a, b] = C[b, a] = L[a, b]
        np.fill_diagonal(C, 0)
        kanten = {(a, b): float(C[a, b]) for a in range(n) for b in range(a + 1, n)
                  if np.isfinite(C[a, b])}
        return seehandel_anteil(kanten, W, n, seek)

    reihe = [anteil_bei(hk) for hk in (0, 50, 100, 200, 400)]
    ok &= check("mehr Umsteigekosten -> nie MEHR Seehandel (monoton)",
                all(a >= b - 1e-9 for a, b in zip(reihe, reihe[1:])),
                " -> ".join(f"{v:.0%}" for v in reihe))
    ok &= check("ohne Umsteigekosten faehrt viel ueber See", reihe[0] > 0.5,
                f"{reihe[0]:.0%}")
    ok &= check("mit hohen Umsteigekosten faehrt wenig ueber See", reihe[-1] < 0.2,
                f"{reihe[-1]:.0%}")

    # --- Traffic: die Grundlage der Messung selbst
    kanten = {(0, 1): 40.0, (1, 2): 40.0}
    traffic = kanten_traffic(kanten, W, 3)
    ok &= check("Traffic zaehlt durchlaufenden Handel",
                traffic[(0, 1)] > W[0, 1],
                f"{traffic[(0,1)]:.0f} (Kante 0-1 traegt auch 0->2)")

    # --- Fischersiedlung: am Wasser, aber unter der Marktstadt gedeckelt
    H, slope, _ = _gelaende(384, 20260804)
    an = TerrainSuitabilityAnalyzer(1.0, 384)
    eig = an.stadttyp_eignungen(H, slope, np.zeros_like(H, dtype=np.float32))
    land = H > 0
    ok &= check("Fischersiedlung hat eine Eignungskarte", "fischersiedlung" in eig)
    ok &= check("  nie ueber ihrem Deckel",
                float(eig["fischersiedlung"].max()) <= FISCHER_DECKEL + 1e-6,
                f"{eig['fischersiedlung'].max():.2f}")
    besser_markt = float((eig["marktstadt"][land] > eig["fischersiedlung"][land]).mean())
    ok &= check("  wo beide moeglich sind, gewinnt meist die Marktstadt",
                besser_markt > 0.3, f"{besser_markt:.0%} der Landflaeche")
    ok &= check(f"Zielanteil dokumentiert ({SEEHANDEL_ZIEL:.0%})",
                0.0 < SEEHANDEL_ZIEL < 1.0)
    return ok


if __name__ == "__main__":
    ergebnisse = {
        "eignungskarten": run_eignungskarten(),
        "zuweisung_trifft_die_lage": run_zuweisung_trifft_die_lage(),
        "nur_eine_marktstadt": run_nur_eine_marktstadt(),
        "handelsgewichte": run_handelsgewichte(),
        "erreichbarkeit": run_erreichbarkeit(),
        "steigungskosten": run_steigungskosten(),
        "kreuzungsgrade": run_kreuzungsgrade(),
        "netzausbau_nach_bedarf": run_netzausbau_nach_bedarf(),
        "seehandel_und_fischersiedlung": run_seehandel_und_fischersiedlung(),
    }
    print("\n=== SUMMARY ===")
    for name, bestanden in ergebnisse.items():
        print(f"{name}: {'PASS' if bestanden else 'FAIL'}")
    sys.exit(0 if all(ergebnisse.values()) else 1)
