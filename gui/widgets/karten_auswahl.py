"""
Path: gui/widgets/karten_auswahl.py

Anklicken von Orten und Wegen in der 3D-Ansicht (Nutzerwunsch 2026-08-13:
"man soll dann spaeter staedte und landmarks selecten koennen um etwas ueber
die einzelnen staedte landmarks und roadsites, vielleicht auch wege,
herauszufinden (laenge der strasse) oder traffic oder so").

WARUM PROJEKTION UND NICHT COLOR-PICKING

In docs/OFFENE_PUNKTE.md 6.27 hatte ich Color-Picking empfohlen - die Szene
ein zweites Mal in einen unsichtbaren Puffer rendern, jedes Objekt in seiner
ID-Farbe, dann das Pixel unter der Maus auslesen. Das ist das robustere
Verfahren, wenn viele, teils verdeckte Objekte pixelgenau zu treffen sind.

Fuer diesen Fall ist es aber der falsche Aufwand, und ich revidiere die
Empfehlung:

  * Es braucht einen zweiten Renderpass samt Framebuffer-Verwaltung.
  * Es braucht fuer die Punktobjekte ueberhaupt erst Geometrie (Billboards) -
    heute sind Staedte/Landmarks/Roadsites nur Teil einer Textur.
  * Es ist headless nicht pruefbar, weil es ohne GL-Kontext nicht laeuft.

Die Projektionsvariante hier rechnet stattdessen jede Weltposition ueber
dieselben Matrizen in Bildschirmkoordinaten, die auch der Shader benutzt, und
sucht das naechstliegende Objekt zum Mauszeiger. Das ist:

  * reine Mathematik, also **vollstaendig headless testbar**,
  * ausreichend fuer einige Dutzend Objekte,
  * sofort verfuegbar, ohne dass Billboards existieren muessen.

EHRLICHE GRENZE: ein Objekt hinter einem Berg wird mitgetroffen - die Tiefe
wird zwar berechnet und mitgeliefert (`tiefe` im Treffer), aber es gibt
keinen Sichtbarkeitstest gegen das Gelaende. Fuer einen Editor ist das
vertretbar; wer es genauer braucht, kommt an Color-Picking nicht vorbei.
"""

import numpy as np


# Fangradius in Bildschirmpixeln. Grosszuegig genug, dass man nicht
# pixelgenau zielen muss, klein genug, dass zwei benachbarte Orte
# unterscheidbar bleiben.
FANGRADIUS_PX = 18.0

# Wege sind lang und schmal - fuer sie zaehlt der Abstand zur LINIE, nicht zu
# einem Punkt. Etwas enger, weil ein Weg sonst jeden Klick in seiner Naehe
# abfaengt und die Orte darauf nicht mehr treffbar waeren.
WEG_FANGRADIUS_PX = 10.0


def welt_zu_bildschirm(punkte_welt, model, view, projection, breite, hoehe):
    """
    (N,3)-Weltpositionen -> (N,3) mit (x_px, y_px, tiefe).

    Dieselbe Kette wie im Vertex-Shader (projection * view * model), danach
    die uebliche Perspektivdivision und die Abbildung auf den Ansichtsbereich.
    `tiefe` ist die z-Komponente nach der Division: kleiner = naeher an der
    Kamera. Punkte hinter der Kamera bekommen NaN, damit sie nicht
    versehentlich als Treffer gelten.
    """
    punkte = np.asarray(punkte_welt, dtype=np.float64).reshape(-1, 3)
    if len(punkte) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    homogen = np.hstack([punkte, np.ones((len(punkte), 1))])
    mvp = np.asarray(projection) @ np.asarray(view) @ np.asarray(model)
    geclippt = homogen @ np.asarray(mvp).T

    w = geclippt[:, 3]
    gueltig = np.abs(w) > 1e-9
    ndc = np.full((len(punkte), 3), np.nan)
    ndc[gueltig] = geclippt[gueltig, :3] / w[gueltig, None]
    # Hinter der Kamera: in OpenGL-Konvention ist w dort negativ.
    ndc[w <= 0] = np.nan

    aus = np.empty((len(punkte), 3))
    aus[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * breite
    aus[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * hoehe   # y zeigt nach unten
    aus[:, 2] = ndc[:, 2]
    return aus


def _abstand_zu_strecke(punkt, a, b):
    """Kuerzester Abstand eines Punktes zur Strecke a-b (alle 2D)."""
    ab = b - a
    laenge2 = float(ab @ ab)
    if laenge2 < 1e-12:
        return float(np.linalg.norm(punkt - a)), 0.0
    t = float(np.clip((punkt - a) @ ab / laenge2, 0.0, 1.0))
    naechster = a + t * ab
    return float(np.linalg.norm(punkt - naechster)), t


def treffer_suchen(maus_x, maus_y, orte_welt, orte_kennung,
                   wege_welt, wege_kennung, model, view, projection,
                   breite, hoehe, fangradius=FANGRADIUS_PX,
                   weg_fangradius=WEG_FANGRADIUS_PX):
    """
    Was liegt unter dem Mauszeiger?

    Parameter:
        orte_welt    (N,3) Weltpositionen der Punktobjekte
        orte_kennung Liste beliebiger Kennungen, gleiche Reihenfolge
        wege_welt    Liste von (M,3)-Arrays, je ein Weg
        wege_kennung Liste von Kennungen, gleiche Reihenfolge wie wege_welt

    Rueckgabe: dict mit `art` ("ort"/"weg"), `kennung`, `abstand` (Pixel) und
    `tiefe` - oder None, wenn nichts in Reichweite liegt.

    ORTE HABEN VORRANG vor Wegen, wenn beide in Reichweite sind: ein Ort ist
    das kleinere, gezieltere Ziel, und an einem Ort enden fast immer mehrere
    Wege - ohne Vorrang waere eine Stadt am Wegende praktisch nie anklickbar.
    """
    maus = np.array([float(maus_x), float(maus_y)])

    bester_ort = None
    if len(orte_welt):
        bildschirm = welt_zu_bildschirm(orte_welt, model, view, projection,
                                        breite, hoehe)
        for i, punkt in enumerate(bildschirm):
            if not np.all(np.isfinite(punkt)):
                continue
            abstand = float(np.linalg.norm(punkt[:2] - maus))
            if abstand > fangradius:
                continue
            if bester_ort is None or abstand < bester_ort["abstand"]:
                # `index` ist die Position in der uebergebenen Liste. Ohne
                # sie weiss der Aufrufer zwar WAS getroffen wurde, kann es
                # aber nicht wiederfinden - und damit auch nicht einfaerben
                # (Nutzerbefund 2026-08-24: "nicht markierbar").
                bester_ort = {"art": "ort", "index": int(i),
                              "kennung": orte_kennung[i],
                              "abstand": abstand, "tiefe": float(punkt[2])}
    if bester_ort is not None:
        return bester_ort

    bester_weg = None
    for w, pfad in enumerate(wege_welt or []):
        pfad = np.asarray(pfad, dtype=np.float64).reshape(-1, 3)
        if len(pfad) < 2:
            continue
        bildschirm = welt_zu_bildschirm(pfad, model, view, projection,
                                        breite, hoehe)
        for k in range(len(bildschirm) - 1):
            a, b = bildschirm[k], bildschirm[k + 1]
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                continue
            abstand, t = _abstand_zu_strecke(maus, a[:2], b[:2])
            if abstand > weg_fangradius:
                continue
            tiefe = float(a[2] + t * (b[2] - a[2]))
            if bester_weg is None or abstand < bester_weg["abstand"]:
                bester_weg = {"art": "weg", "index": int(w),
                              "kennung": wege_kennung[w],
                              "abstand": abstand, "tiefe": tiefe}
    return bester_weg


def weglaenge_km(pfad_pixel, welt_km, kartengroesse):
    """Laenge eines Weges in Kilometern - eine der Angaben, die der Nutzer
    beim Anklicken sehen will ("laenge der strasse")."""
    punkte = np.asarray(pfad_pixel, dtype=np.float64).reshape(-1, 2)
    if len(punkte) < 2:
        return 0.0
    schritte = np.linalg.norm(np.diff(punkte, axis=0), axis=1)
    meter_pro_pixel = welt_km * 1000.0 / max(int(kartengroesse), 1)
    return float(schritte.sum() * meter_pro_pixel / 1000.0)
