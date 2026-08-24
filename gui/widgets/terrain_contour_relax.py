"""
Path: gui/widgets/terrain_contour_relax.py

KONTUREN-RELAXATION - Hoehenlinien glaetten, Feld daraus neu aufbauen.

Nutzeridee 2026-08-16, nach dem Lesen von Huftier et al. (Eurographics 2026,
"Terrain Synthesis and Authoring based on Iso-Contours", von Nutzer
zugeschickt als PDF): *"ich will das System der Konturen-Relaxation als ein
System implementieren und ausprobieren."*

AUS DEM PAPIER UEBERNOMMEN, ZWEI BAUSTEINE:

* Section 5.1 (Smoothing iso-contours) - die Glaettungsformel selbst: jeder
  Punkt einer Kontur wandert iterativ zum Mittel seiner k Nachbarn auf
  beiden Seiten, gewichtet mit alpha:
      q_i <- (1-alpha)*q_i + alpha/(2k) * (Summe der 2k Nachbarn)
* Section 3.2 (Reconstruction from contours) - die distanzgewichtete
  Interpolation zwischen den zwei naechsten Konturen (Hormann et al. 2003),
  um aus den (jetzt geglaetteten) Konturen wieder ein durchgehendes
  Hoehenfeld zu machen:
      h(p) = (d(p,G_k+1)*h_k + d(p,G_k)*h_k+1) / (d(p,G_k+1) + d(p,G_k))

NICHT UEBERNOMMEN: der generative Teil des Papiers (Open Eden Growth, um
Konturen aus einer Nutzerskizze zu ERZEUGEN). Wir haben schon ein fertiges
Hoehenfeld (core.terrain_weltkarte.weltfeld() + Erosionsfilter + Fluesse) -
wir EXTRAHIEREN Konturen daraus (skimage.measure.find_contours, marching
squares), glaetten sie, und rechnen ein neues Feld zurueck. Die
Distanztransformation je Kontur-Level ersetzt das direkte d(p,Gamma) aus dem
Papier - schneller, weil rasterbasiert statt Punkt-zu-Punkt.

WARUM DAS DIE RASTERTREPPE ANDERS ANGEHT ALS QEM/Delatin/Quadtree:
alle bisherigen Verfahren versuchten, Vertex-POSITIONEN vom Raster zu loesen.
Hier bleiben die Vertices exakt auf dem Pixelraster - stattdessen wird die
HOEHE selbst so umgerechnet, dass jede Hoehenlinie (Kueste bei 0 m, jede
andere Stufe) einer geglaetteten Kurve folgt statt der rohen Pixelkante. Kein
Meer/Land-Split noetig - eine einzige durchgehende Rekonstruktion ueber die
ganze Karte, also auch keine Naht wie bei der QEM-Trennung (siehe die
"Stufen am Meer"-Rueckmeldung vom 2026-08-16 zu terrain_remesh.py).

EHRLICH BENANNTER VORBEHALT AUS DEM PAPIER SELBST (Abschnitt 6.5): genau
diese Rekonstruktion frisst lokale Maxima/Minima, die zwischen zwei
Kontur-Stufen liegen - bei den Autoren verschwanden mit 12 Konturen 2-10x so
viele Gipfel wie im Referenzgelaende. Zu wenige Stufen koennten also unsere
Klippen genauso wegglaetten wie die bisherigen Verfahren. Deshalb ist die
Konturenzahl ein Regler, nicht fest verdrahtet.

Baut absichtlich ein VOLLES Gitter (keine Dezimierung) - der Punkt dieses
Prototyps ist, das rekonstruierte Hoehenfeld selbst zu beurteilen, entkoppelt
von der Frage, wie stark man es hinterher noch dezimieren wuerde.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import measure


def _kontur_relaxieren(punkte, geschlossen, alpha, k, iterationen):
    """Section 5.1: q_i <- (1-alpha) q_i + alpha/(2k) * Summe der 2k Nachbarn."""
    p = punkte.astype(np.float64).copy()
    n = len(p)
    if n < 3 or iterationen <= 0:
        return p
    k = min(k, (n - 1) // 2) if not geschlossen else min(k, n // 2)
    if k < 1:
        return p
    idx = np.arange(n)
    for _ in range(iterationen):
        summe = np.zeros_like(p)
        for j in range(1, k + 1):
            if geschlossen:
                summe += np.roll(p, j, axis=0) + np.roll(p, -j, axis=0)
            else:
                # Offene Kontur (endet am Kartenrand): Indizes klemmen statt
                # umlaufen, sonst wandert die Kueste am Bildrand weg.
                summe += p[np.clip(idx - j, 0, n - 1)]
                summe += p[np.clip(idx + j, 0, n - 1)]
        p_neu = (1 - alpha) * p + (alpha / (2 * k)) * summe
        if not geschlossen:
            # Raender festhalten - sie sitzen auf dem Kartenrand und muessen
            # dort bleiben, sonst reisst die Kontur vom Nachbarpixel ab.
            p_neu[0], p_neu[-1] = p[0], p[-1]
        p = p_neu
    return p


def _kontur_maske(punkte_liste, shape):
    """Alle Teilkonturen EINES Levels als duenne Linienmaske rastern."""
    maske = np.ones(shape, dtype=bool)
    for punkte in punkte_liste:
        zeilen = np.clip(np.round(punkte[:, 0]).astype(int), 0, shape[0] - 1)
        spalten = np.clip(np.round(punkte[:, 1]).astype(int), 0, shape[1] - 1)
        maske[zeilen, spalten] = False
    return maske


def relaxiertes_feld(heightmap, anzahl_konturen=30, alpha=0.5, k=3,
                     iterationen=10):
    """
    Baut ein neues Hoehenfeld, dessen Isolinien geglaettete Versionen der
    Original-Konturen sind (Section 3.2 + 5.1, Huftier et al. 2026).

    `iterationen=0` gibt praktisch das (nur leicht rasterquantisierte)
    Original zurueck - nuetzlich als A/B-Nullprobe im Regler.

    Rueckgabe: Hoehenfeld gleicher Form wie `heightmap`, float32.
    """
    H = np.asarray(heightmap, dtype=np.float64)
    h_min, h_max = float(H.min()), float(H.max())
    anzahl_konturen = max(int(anzahl_konturen), 2)
    level_werte = np.linspace(h_min, h_max, anzahl_konturen)

    distanzfelder = [None] * anzahl_konturen
    for i, level in enumerate(level_werte):
        konturen = measure.find_contours(H, level)
        if not konturen:
            continue
        geglaettet = []
        for kontur in konturen:
            geschlossen = bool(np.allclose(kontur[0], kontur[-1]))
            geglaettet.append(_kontur_relaxieren(kontur, geschlossen, alpha,
                                                 k, iterationen))
        maske = _kontur_maske(geglaettet, H.shape)
        distanzfelder[i] = distance_transform_edt(maske)

    # Level ohne eigene Kontur (z.B. ueber dem hoechsten Gipfel oder unter
    # der tiefsten Rinne, wo `find_contours` nichts findet) durch das
    # naechste tatsaechlich vorhandene Distanzfeld ersetzen - sonst bricht
    # die Interpolation dort mit None ab.
    letzte = None
    for i in range(anzahl_konturen):
        if distanzfelder[i] is None:
            distanzfelder[i] = letzte
        else:
            letzte = distanzfelder[i]
    letzte = None
    for i in range(anzahl_konturen - 1, -1, -1):
        if distanzfelder[i] is None:
            distanzfelder[i] = letzte
        else:
            letzte = distanzfelder[i]
    if any(d is None for d in distanzfelder):
        # Kein einziges Level hatte eine Kontur (voellig flache Karte) -
        # dann gibt es nichts zu rekonstruieren.
        return H.astype(np.float32)

    schritt = (h_max - h_min) / max(anzahl_konturen - 1, 1)
    band = np.clip(((H - h_min) / max(schritt, 1e-9)).astype(np.int32),
                   0, anzahl_konturen - 2)

    H_neu = np.empty_like(H)
    for k_idx in range(anzahl_konturen - 1):
        maske = band == k_idx
        if not maske.any():
            continue
        dk = distanzfelder[k_idx][maske]
        dk1 = distanzfelder[k_idx + 1][maske]
        summe = dk + dk1
        summe_sicher = np.where(summe > 1e-9, summe, 1.0)
        wert = (dk1 * level_werte[k_idx] + dk * level_werte[k_idx + 1]) / summe_sicher
        H_neu[maske] = np.where(summe > 1e-9, wert, level_werte[k_idx])

    return H_neu.astype(np.float32)


def baue_konturen_relax(heightmap, terrain_scale_factor, terrain_height_scale,
                        anzahl_konturen=30, glaettung_iterationen=10,
                        alpha=0.5, k=3):
    """
    Rueckgabe wie `adaptive_terrain_mesh.build_adaptive_mesh()`:
    (vertices float32 interleaved [x,y,z,nx,ny,nz,u,v], indices uint32,
    stats dict) - volles Gitter aus dem konturrelaxierten Hoehenfeld, keine
    Dezimierung.
    """
    from gui.widgets.adaptive_terrain_mesh import _normalen_voll
    from gui.widgets.terrain_remesh import _volles_gitter

    H = relaxiertes_feld(heightmap, anzahl_konturen=anzahl_konturen,
                         alpha=alpha, k=k, iterationen=glaettung_iterationen)
    punkte, dreiecke = _volles_gitter(H, terrain_scale_factor, terrain_height_scale)

    nx_feld, ny_feld, nz_feld = _normalen_voll(H, terrain_height_scale,
                                               terrain_scale_factor)
    hoehe_px, breite_px = H.shape
    ys, xs = np.mgrid[0:hoehe_px, 0:breite_px]

    vertices = np.stack([
        punkte[:, 0], punkte[:, 1], punkte[:, 2],
        nx_feld.ravel(), ny_feld.ravel(), nz_feld.ravel(),
        (xs / (breite_px - 1)).ravel(), (ys / (hoehe_px - 1)).ravel(),
    ], axis=-1).astype(np.float32)

    stats = {
        "vertices": len(punkte), "dreiecke": len(dreiecke),
        "voll_dreiecke": len(dreiecke), "voll_vertices": len(punkte),
        # Bewusst 0.0: dieses Verfahren verschiebt keine Vertex-POSITIONEN,
        # nur die Hoehe - "Abstand zur Pixelecke" ist hier also immer 0 und
        # sagt nichts ueber die Qualitaet des Ansatzes.
        "frei_verschoben": 0.0, "versatz_median_px": 0.0,
        "anzahl_konturen": anzahl_konturen,
        "glaettung_iterationen": glaettung_iterationen,
        "aus_cache": False,
    }
    return vertices.reshape(-1), dreiecke.reshape(-1).astype(np.uint32), stats
