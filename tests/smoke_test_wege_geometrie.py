"""
Path: tests/smoke_test_wege_geometrie.py

Prueft gui/widgets/wege_geometrie.py - Wege als echte Bandgeometrie statt als
Textur (docs/OFFENE_PUNKTE.md 6.28, Nutzerwunsch 2026-08-13: "es sieht nicht
so schoen aus mit den strassen als textur ... ich will fuer den editor der nur
in python ist ein bisschen schoenere optik").

Die beiden Zusicherungen, an denen beim Bauen tatsaechlich etwas schieflief,
stehen hier zuerst:

  * **Nichts darf im Gelaende versinken.** Der erste Entwurf gab jeder
    Bandkante ihre eigene Gelaendehoehe; am Querhang lag der Rand dadurch
    UNTER dem Terrain (gemessen 0.002 Welteinheiten, bei 0.006 Schwebehoehe).
  * **Die Breite muss auch auf kleinen Karten sichtbar bleiben.** Bei 256 px
    (83 m/px) waere ein 60-m-Weg 0.72 Pixel breit gewesen.

Laeuft mit den ECHTEN Kartengroessen dieses Projekts (256/512/1024).
"""
import sys

import numpy as np

sys.path.insert(0, ".")

from gui.widgets.wege_geometrie import (
    baue_wegbaender, band_aus_pfad, WEG_BREITE_M, SEEWEG_BREITE_M,
    MINDEST_BREITE_PX, SCHWEBE_ANTEIL, PROFIL_T, WOELBUNG_ANTEIL,
    SCHULTER_ANTEIL, QUER_STUETZSTELLEN)

# Wie tief das geglaettete Laengsprofil hoechstens in eine Kuppe
# schneiden darf. NICHT null: seit 2026-08-24 haelt `glPolygonOffset`
# im Renderer das Band im TIEFENPUFFER vorn, nicht mehr ein Versatz in
# der Welt - und die vom Nutzer gewuenschte Laengsglaettung ('glaetten
# den weg etwas') schneidet an Kuppen zwangslaeufig ein, so wie eine
# echte Trasse. Gemessen sind es 0.7 m bei 256 px. Die Grenze prueft,
# dass daraus kein sichtbares Versinken wird.
EINSCHNITT_MAX_M = 2.0

WELT_KM = 21.3


def check(label, bedingung, zusatz=""):
    status = "OK" if bedingung else "FAIL"
    print(f"[{status}] {label}{(' - ' + zusatz) if zusatz else ''}")
    return bool(bedingung)


def _welt(size):
    """Welliges Testgelaende - eben WAERE der einfachere, aber nutzlose Fall:
    genau am Querhang trat der Versink-Fehler auf."""
    yy, xx = np.mgrid[0:size, 0:size]
    H = (200 * np.sin(xx / (size / 6.4)) * np.cos(yy / (size / 7.3)) + 300)
    return H.astype(np.float32), 10.0 / size, 10.0 / (WELT_KM * 1000.0)


def _pixel_zurueck(v, size, tsf):
    """Weltkoordinate zurueck in Kartenpixel - fuer die Versink-Pruefung."""
    px = (v[0] / tsf / size + 0.5) * (size - 1)
    pz = (v[2] / tsf / size + 0.5) * (size - 1)
    return (int(np.clip(round(px), 0, size - 1)),
            int(np.clip(round(pz), 0, size - 1)))


def run_nichts_versinkt():
    ok = True
    for size in (256, 512, 1024):
        H, tsf, ths = _welt(size)
        s = size / 256.0
        wege = [[(20 * s + i * s, 30 * s + i * 0.7 * s) for i in range(60)],
                [(200 * s - i * 1.5 * s, 40 * s + i * s) for i in range(50)]]
        v, idx, _b = baue_wegbaender(wege, H, WELT_KM, tsf, ths)
        abstand = np.empty(len(v))
        for k in range(len(v)):
            xi, yi = _pixel_zurueck(v[k], size, tsf)
            abstand[k] = v[k, 1] - H[yi, xi] * ths
        tiefster_m = float(abstand.min()) / ths
        ok &= check(f"{size} px: Band schneidet hoechstens {EINSCHNITT_MAX_M:.0f} m ein",
                    tiefster_m > -EINSCHNITT_MAX_M, f"tiefster {tiefster_m:+.1f} m")
        ok &= check(f"{size} px: liegt im Mittel ueber dem Gelaende",
                    float(np.median(abstand)) > 0,
                    f"Median {float(np.median(abstand)) / ths:+.1f} m")
    return ok


def run_querprofil():
    """
    Das Querprofil (2026-08-24): fuenf Bahnen, gewoelbt, mit weichem Rand.

    HIER STAND FRUEHER "Band ist an jedem Segment quer eben" mit
    Nulltoleranz. Das galt fuer das flache Zwei-Vertex-Band und war die
    Absicherung gegen den Versink-Fehler am Querhang. Mit der Woelbung ist
    "eben" nicht mehr wahr - die Absicherung bleibt aber dieselbe: die
    Fahrbahn darf nicht seitlich mitkippen. Geprueft wird deshalb jetzt
    SYMMETRIE (beide Haelften gleich hoch) statt Gleichheit aller Punkte.
    """
    size = 512
    H, tsf, ths = _welt(size)
    wege = [[(40 + i * 2, 60 + i * 1.3) for i in range(70)]]
    v, _i, _b = baue_wegbaender(wege, H, WELT_KM, tsf, ths)
    bahnen = len(PROFIL_T)
    ok = check("fuenf Bahnen je Wegpunkt", len(v) % bahnen == 0,
               f"{len(v)} Vertices")

    hoehen = v[:, 1].reshape(-1, bahnen)
    ok &= check("Band kippt nicht seitlich - Kanten gleich hoch",
                bool(np.allclose(hoehen[:, 0], hoehen[:, 4], atol=1e-9)))
    ok &= check("Band kippt nicht seitlich - Schultern gleich hoch",
                bool(np.allclose(hoehen[:, 1], hoehen[:, 3], atol=1e-9)))
    ok &= check("Scheitel liegt am hoechsten",
                bool(np.all(hoehen[:, 2] >= hoehen[:, 1] - 1e-9)
                     and np.all(hoehen[:, 1] >= hoehen[:, 0] - 1e-9)))

    breite_px = max(MINDEST_BREITE_PX, WEG_BREITE_M / (WELT_KM * 1000.0 / size))
    soll = WOELBUNG_ANTEIL * breite_px * tsf
    ist = float(np.max(hoehen[:, 2] - hoehen[:, 0]))
    ok &= check("Woelbung trifft die Vorgabe", abs(ist - soll) < soll * 0.02 + 1e-9,
                f"{ist / ths:.1f} m ueber der Kante")

    deckung = v[:, 6].reshape(-1, bahnen)
    ok &= check("Deckung: Kanten durchsichtig, Fahrbahn deckend",
                bool(np.allclose(deckung[:, [0, 4]], 0.0)
                     and np.allclose(deckung[:, [1, 2, 3]], 1.0)),
                "sonst gaebe es keinen weichen Rand")
    return ok


def run_breite():
    ok = True
    for size in (256, 512, 1024):
        H, tsf, ths = _welt(size)
        wege = [[(20 + i, 30 + i * 0.7) for i in range(40)]]
        v, _i, _b = baue_wegbaender(wege, H, WELT_KM, tsf, ths)
        mpp = WELT_KM * 1000.0 / size
        breite_px = max(MINDEST_BREITE_PX, WEG_BREITE_M / mpp)
        soll = breite_px / (size - 1) * size * tsf
        # Volle Breite = linke Kante (Bahn 0) bis rechte Kante (Bahn 4)
        # desselben Wegpunkts. Nur die Position vergleichen, nicht Normale
        # und Deckung - deshalb [:3].
        bahnen = len(PROFIL_T)
        k = 10 * bahnen
        ist = float(np.linalg.norm(v[k, :3] - v[k + bahnen - 1, :3]))
        ok &= check(f"{size} px: Bandbreite trifft die Vorgabe",
                    abs(ist - soll) < soll * 0.02,
                    f"{breite_px * mpp:.0f} m ({breite_px:.1f} px)")
    ok &= check("auf kleinen Karten greift die Pixel-Untergrenze",
                MINDEST_BREITE_PX > WEG_BREITE_M / (WELT_KM * 1000.0 / 256),
                f"{WEG_BREITE_M / (WELT_KM * 1000.0 / 256):.2f} px ohne sie")
    return ok


def run_indizes_und_bereiche():
    """Die Bereichsliste ist die Grundlage fuers spaetere Anklicken einzelner
    Strassen - Luecken oder Ueberlappungen darin waeren dort still falsche
    Treffer."""
    size = 512
    H, tsf, ths = _welt(size)
    wege = [[(20 + i, 30 + i) for i in range(30)],
            [(300, 100)],                      # zu kurz - wird uebersprungen
            [(400 - i, 200 + i * 0.5) for i in range(25)]]
    v, idx, bereiche = baue_wegbaender(wege, H, WELT_KM, tsf, ths)
    ok = True
    ok &= check("zu kurze Wege werden uebersprungen", len(bereiche) == 2,
                f"{len(bereiche)} Baender aus {len(wege)} Wegen")
    ok &= check("jeder Index zeigt in das Vertex-Array",
                len(idx) > 0 and int(idx.max()) < len(v))
    ok &= check("Indexzahl ist durch 3 teilbar (Dreiecke)", len(idx) % 3 == 0)
    summe = 0
    lueckenlos = True
    for _wi, start, anzahl in bereiche:
        if start != summe:
            lueckenlos = False
        summe += anzahl
    ok &= check("Bereiche decken die Indexliste lueckenlos",
                lueckenlos and summe == len(idx), f"{summe} von {len(idx)}")
    return ok


def run_randfaelle():
    size = 256
    H, tsf, ths = _welt(size)
    ok = True
    for name, eingabe in (("keine Wege", []), ("None als Wegliste", None)):
        v, idx, b = baue_wegbaender(eingabe, H, WELT_KM, tsf, ths)
        ok &= check(f"{name}: leeres Ergebnis statt Absturz",
                    len(v) == 0 and len(idx) == 0 and b == [])
    v, idx, b = baue_wegbaender([[(1, 1), (2, 2)]], None, WELT_KM, tsf, ths)
    ok &= check("ohne Heightmap: leeres Ergebnis statt Absturz", len(v) == 0)
    # Ein Weg, der aus dem Kartenrand laeuft, darf keine NaN erzeugen
    v, idx, b = baue_wegbaender([[(-20 + i, -10 + i) for i in range(40)]],
                                 H, WELT_KM, tsf, ths)
    ok &= check("Weg ueber den Kartenrand: alle Werte endlich",
                len(v) > 0 and bool(np.all(np.isfinite(v))))
    return ok


def run_auswahl():
    """Anklicken von Orten und Wegen (docs/OFFENE_PUNKTE.md 6.29).

    Reine Projektionsmathematik - deshalb hier vollstaendig pruefbar, im
    Gegensatz zu Color-Picking, das ohne GL-Kontext gar nicht liefe. Genau
    das war der Grund, die urspruengliche Color-Picking-Empfehlung aus 6.27
    zu revidieren.
    """
    from gui.widgets.karten_auswahl import (welt_zu_bildschirm, treffer_suchen,
                                             weglaenge_km)
    ok = True
    model = view = proj = np.eye(4)
    B, H_PX = 800, 600

    p = welt_zu_bildschirm([[0, 0, 0]], model, view, proj, B, H_PX)
    ok &= check("Weltursprung landet in der Bildmitte",
                abs(p[0][0] - B / 2) < 1e-6 and abs(p[0][1] - H_PX / 2) < 1e-6,
                f"{p[0][:2]}")
    p = welt_zu_bildschirm([[0.5, 0, 0], [0, 0.5, 0]], model, view, proj, B, H_PX)
    ok &= check("+x nach rechts, +y nach OBEN (Bildschirm-y invertiert)",
                p[0][0] > B / 2 and p[1][1] < H_PX / 2)

    # Hinter der Kamera darf NIE ein Treffer entstehen
    proj_p = np.array([[1.5, 0, 0, 0], [0, 2, 0, 0],
                       [0, 0, -1.002, -0.2], [0, 0, -1, 0]], dtype=float)
    p = welt_zu_bildschirm([[0, 0, 1.0]], model, view, proj_p, B, H_PX)
    ok &= check("hinter der Kamera -> NaN statt Falschtreffer",
                bool(np.all(np.isnan(p[0]))))

    orte = np.array([[0, 0, 0], [0.5, 0, 0]], dtype=float)
    kennungen = ["A", "B"]
    tr = treffer_suchen(B / 2, H_PX / 2, orte, kennungen, [], [],
                        model, view, proj, B, H_PX)
    ok &= check("Klick auf einen Ort trifft ihn", tr and tr["kennung"] == "A")
    tr = treffer_suchen(10, 10, orte, kennungen, [], [], model, view, proj, B, H_PX)
    ok &= check("Klick ins Leere liefert None (kein Fehler)", tr is None)

    weg = np.array([[-0.5, 0.3, 0], [0.5, 0.3, 0]], dtype=float)
    y_weg = welt_zu_bildschirm([[0, 0.3, 0]], model, view, proj, B, H_PX)[0][1]
    tr = treffer_suchen(B / 2, y_weg, np.zeros((0, 3)), [], [weg], ["W"],
                        model, view, proj, B, H_PX)
    ok &= check("Klick auf einen Weg trifft ihn", tr and tr["kennung"] == "W")

    # DER INDEX MUSS MIT (2026-08-24). Ohne ihn weiss der Aufrufer zwar, WAS
    # getroffen wurde, kann es aber nicht wiederfinden - und damit auch nicht
    # einfaerben. Genau daran lag der Nutzerbefund "nicht markierbar": der
    # Klick funktionierte, im Bild passierte nur nichts.
    zwei_wege = [np.array([[-0.5, 0.3, 0], [0.5, 0.3, 0]], dtype=float),
                 np.array([[-0.5, -0.3, 0], [0.5, -0.3, 0]], dtype=float)]
    y_zweiter = welt_zu_bildschirm([[0, -0.3, 0]], model, view, proj, B, H_PX)[0][1]
    tr2 = treffer_suchen(B / 2, y_zweiter, np.zeros((0, 3)), [], zwei_wege,
                         ["W0", "W1"], model, view, proj, B, H_PX)
    ok &= check("Treffer nennt den Index des Weges",
                bool(tr2) and tr2.get("index") == 1,
                f"index={tr2.get('index') if tr2 else None}, "
                f"kennung={tr2.get('kennung') if tr2 else None}")

    # Der Vorrang ist keine Kosmetik: an einem Ort enden fast immer Wege
    weg_mitte = np.array([[-0.5, 0, 0], [0.5, 0, 0]], dtype=float)
    tr = treffer_suchen(B / 2, H_PX / 2, orte, kennungen, [weg_mitte], ["W"],
                        model, view, proj, B, H_PX)
    ok &= check("bei Ort UND Weg gewinnt der Ort",
                tr and tr["art"] == "ort", f"{tr['art'] if tr else None}")

    km = weglaenge_km([(0, 0), (100, 0), (100, 100)], 21.3, 512)
    ok &= check("Weglaenge in km stimmt",
                abs(km - 200 * (21300 / 512) / 1000) < 1e-6, f"{km:.2f} km")
    ok &= check("Weglaenge eines Ein-Punkt-Pfads ist 0",
                weglaenge_km([(5, 5)], 21.3, 512) == 0.0)
    return ok


if __name__ == "__main__":
    ergebnisse = {
        "nichts_versinkt": run_nichts_versinkt(),
        "querprofil": run_querprofil(),
        "breite": run_breite(),
        "indizes_und_bereiche": run_indizes_und_bereiche(),
        "randfaelle": run_randfaelle(),
        "auswahl": run_auswahl(),
    }
    print("\n=== SUMMARY ===")
    for name, bestanden in ergebnisse.items():
        print(f"{name}: {'PASS' if bestanden else 'FAIL'}")
    sys.exit(0 if all(ergebnisse.values()) else 1)
