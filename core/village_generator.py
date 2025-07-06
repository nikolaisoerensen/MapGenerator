# --- Generator für Dörfer, Wahrzeichen und Kneipen ---

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math
import random

# --- Konfiguration ---
konfiguration = {
    "bildGroesse": 1000,
    "randAbstand": 100,
    "minEntfernung": 570,
    "maxEntfernung": 720,
    "minRadius": 25,
    "maxRadius": 55,
    "verblassungsStaerke": 0.7,
    "verblassungsMultiplikator": 6,
    "wahrzeichenAnzahl": 2,
    "wahrzeichenRadius": 20,
    "wahrzeichenIntensitaet": 0.5,
    "wahrzeichenVerblassungsEntfernung": 150,
    "kneipenAnzahl": 3,
    "kneipenRadius": 20,
    "kneipenIntensitaet": 0.5,
    "kneipenVerblassungsEntfernung": 150,
    "kneipenMinEntfernung": 200, # Erhöhter Mindestabstand
    "perlinStaerke": 0.25, # Erhöhte Perlin-Stärke, um den Effekt bei mittlerer Intensität sichtbarer zu machen
    "perlinRauschenGroesse1": (10, 10), # Perlin-Rauschgröße 1
    "perlinRauschenGroesse2": (25, 25), # Perlin-Rauschgröße 2
    "perlinRauschenGewicht1": 0.6, # Gewichtung für Perlin-Rauschen 1
    "perlinRauschenGewicht2": 0.4 # Gewichtung für Perlin-Rauschen 2
}

# --- Hilfsfunktionen ---
def erhalte_punkt_auf_linie(p1, p2, verhaeltnis):
    """
    Berechnet einen Punkt auf der Verbindungslinie zwischen zwei Punkten basierend auf einem Verhältnis.

    :param p1: Der erste Punkt als Dictionary {"x": float, "y": float}.
    :param p2: Der zweite Punkt als Dictionary {"x": float, "y": float}.
    :param verhaeltnis: Das Verhältnis, das angibt, wie weit der Punkt zwischen p1 und p2 liegt (0.0 bis 1.0).
    :return: Der berechnete Punkt als Dictionary {"x": float, "y": float}.
    """
    return {
        "x": p1["x"] + (p2["x"] - p1["x"]) * verhaeltnis,
        "y": p1["y"] + (p2["y"] - p1["y"]) * verhaeltnis
    }

def berechne_entfernung(p1, p2):
    """
    Berechnet die euklidische Entfernung zwischen zwei Punkten.

    :param p1: Der erste Punkt als Dictionary {"x": float, "y": float}.
    :param p2: Der zweite Punkt als Dictionary {"x": float, "y": float}.
    :return: Die Entfernung zwischen den beiden Punkten als float.
    """
    return math.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2)

# --- Perlin-Rauschen-Generator (einfach, 2D) ---
def generiere_perlin_rauschen_2d(form, aufloesung):
    """
    Generiert 2D Perlin-Rauschen mit einer bestimmten Form und Auflösung.

    :param form: Die gewünschte Form des Rauschbildes (Höhe, Breite) als Tupel von Integern.
    :param aufloesung: Die Auflösung des Rauschgitters (Anzahl der Gitterpunkte in y, Anzahl der Gitterpunkte in x) als Tupel von Integern.
    :return: Ein numpy-Array mit dem generierten Perlin-Rauschen, normalisiert auf den Bereich [0, 1].
    """
    def f(t):
        # Glättungsfunktion für Interpolation
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    # Berechnung der Gitterpunkte und Gradienten
    delta = (aufloesung[0] / form[0], aufloesung[1] / form[1])
    d = (form[0] // aufloesung[0], form[1] // aufloesung[1])
    gitter = np.stack(np.meshgrid(np.arange(aufloesung[0]+1), np.arange(aufloesung[1]+1)), axis=-1)
    winkel = 2 * np.pi * np.random.rand(aufloesung[0]+1, aufloesung[1]+1)
    gradienten = np.stack((np.cos(winkel), np.sin(winkel)), axis=-1)

    # Definition einer Funktion zur Berechnung des Punkt-Gitter-Gradienten-Produkts
    def punkt_gitter_gradient(ix, iy, x, y):
        # Sicherstellen, dass die Indizes innerhalb der Grenzen liegen
        ix = ix % (aufloesung[0] + 1)
        iy = iy % (aufloesung[1] + 1)
        dx = x - ix
        dy = y - iy
        gradient = gradienten[ix, iy]
        return dx * gradient[0] + dy * gradient[1]

    rauschen = np.zeros(form)
    # Generierung des Rauschens für jedes Pixel
    for i in range(form[0]):
        for j in range(form[1]):
            x = j * delta[0]
            y = i * delta[1]
            x0 = int(x)
            y0 = int(y)
            sx = f(x - x0)
            sy = f(y - y0)

            # Interpolation der Gradienten
            n0 = punkt_gitter_gradient(x0, y0, x, y)
            n1 = punkt_gitter_gradient(x0+1, y0, x, y)
            ix0 = n0 + sx * (n1 - n0)

            n0 = punkt_gitter_gradient(x0, y0+1, x, y)
            n1 = punkt_gitter_gradient(x0+1, y0+1, x, y)
            ix1 = n0 + sx * (n1 - n0)

            wert = ix0 + sy * (ix1 - ix0)
            rauschen[i][j] = wert
    # Normalisierung des Rauschens auf den Bereich [0, 1]
    return (rauschen - rauschen.min()) / (rauschen.max() - rauschen.min())

# --- Dorf-, Wahrzeichen- und Kneipengenerierung ---
def generiere_doerfer():
    """
    Generiert eine Liste von Dörfern basierend auf den Konfigurationsparametern.
    Versucht, 3 Dörfer innerhalb der definierten Abstände voneinander und vom Rand zu platzieren.

    :return: Eine Liste von Dorf-Objekten (Dictionaries) oder eine leere Liste, wenn die Generierung fehlschlägt.
    """
    doerfer = []
    rand = konfiguration["randAbstand"] + konfiguration["maxRadius"]
    min_entfernung_zwischen_entitaeten = konfiguration["kneipenMinEntfernung"] # Mindestabstand für alle Entitäten verwenden

    versuche = 0
    # Versucht, solange Dörfer zu generieren, bis 3 gefunden wurden oder die maximale Anzahl an Versuchen erreicht ist
    while len(doerfer) < 3 and versuche < 5000:
        kandidat = {
            "x": random.uniform(rand, konfiguration["bildGroesse"] - rand),
            "y": random.uniform(rand, konfiguration["bildGroesse"] - rand),
            "radius": random.uniform(konfiguration["minRadius"], konfiguration["maxRadius"])
        }
        # Prüft, ob der Kandidat gültig ist (innerhalb der Mindest-/Maximalentfernung zu bestehenden Dörfern und dem Mindestabstand zu allen Entitäten)
        gueltig = all(
            konfiguration["minEntfernung"] <= berechne_entfernung(kandidat, v) <= konfiguration["maxEntfernung"] and
            berechne_entfernung(kandidat, v) >= min_entfernung_zwischen_entitaeten # Mindestabstand prüfen
            for v in doerfer
        )
        if gueltig:
            doerfer.append(kandidat)
        versuche += 1

    # Prüfen, ob 3 Dörfer generiert wurden, falls nicht, leere Liste zurückgeben, um Fehler zu signalisieren
    if len(doerfer) == 3:
        return doerfer
    else:
        return []

def generiere_wahrzeichen(doerfer):
    """
    Generiert eine Liste von Wahrzeichen entlang der Verbindungslinien zwischen den Dörfern.

    :param doerfer: Eine Liste von Dorf-Objekten (Dictionaries).
    :return: Eine Liste von Wahrzeichen-Objekten (Dictionaries) oder eine leere Liste, wenn die Generierung fehlschlägt.
    """
    wahrzeichen = []
    if len(doerfer) < 3: # Sicherstellen, dass genügend Dörfer vorhanden sind, um Seiten zu bilden
        return [] # Leere Liste zurückgeben, um Fehler zu signalisieren

    # Definiert die Verbindungslinien zwischen den Dörfern
    seiten = [
        (doerfer[0], doerfer[1]),
        (doerfer[1], doerfer[2]),
        (doerfer[2], doerfer[0])
    ]
    min_entfernung_zwischen_entitaeten = konfiguration["kneipenMinEntfernung"]

    versuche = 0
    # Versucht, solange Wahrzeichen zu generieren, bis die gewünschte Anzahl erreicht ist oder die maximale Anzahl an Versuchen erreicht ist
    while len(wahrzeichen) < konfiguration["wahrzeichenAnzahl"] and versuche < 5000:
        seite = random.choice(seiten)
        verhaeltnis = 0.2 + random.random() * 0.6 # Wählt einen Punkt entlang der Linie
        punkt = erhalte_punkt_auf_linie(seite[0], seite[1], verhaeltnis)
        punkt["radius"] = konfiguration["wahrzeichenRadius"]

        # Prüft, ob der Kandidat gültig ist (Mindestabstand zu bestehenden Dörfern und Wahrzeichen)
        gueltig = all(berechne_entfernung(punkt, v) >= min_entfernung_zwischen_entitaeten for v in doerfer + wahrzeichen)
        if gueltig:
            wahrzeichen.append(punkt)
        versuche += 1
    return wahrzeichen

def generiere_kneipen(doerfer, wahrzeichen):
    """
    Generiert eine Liste von Kneipen in der Nähe des Bildrandes.

    :param doerfer: Eine Liste von Dorf-Objekten (Dictionaries).
    :param wahrzeichen: Eine Liste von Wahrzeichen-Objekten (Dictionaries).
    :return: Eine Liste von Kneipen-Objekten (Dictionaries) oder eine leere Liste, wenn die Generierung fehlschlägt.
    """
    kneipen = []
    min_entfernung_zwischen_entitaeten = konfiguration["kneipenMinEntfernung"]
    versuche = 0
    # Versucht, solange Kneipen zu generieren, bis die gewünschte Anzahl erreicht ist oder die maximale Anzahl an Versuchen erreicht ist
    while len(kneipen) < konfiguration["kneipenAnzahl"] and versuche < 5000:
        # Wählt zufällig eine Seite des Bildes für die Platzierung
        seite = random.randint(0, 3)
        if seite == 0: # Oberer Rand
            x = random.uniform(0, konfiguration["bildGroesse"])
            y = random.uniform(100, 200)
        elif seite == 1: # Rechter Rand
            x = random.uniform(konfiguration["bildGroesse"] - 200, konfiguration["bildGroesse"] - 100)
            y = random.uniform(0, konfiguration["bildGroesse"])
        elif seite == 2: # Unterer Rand
            x = random.uniform(0, konfiguration["bildGroesse"])
            y = random.uniform(konfiguration["bildGroesse"] - 200, konfiguration["bildGroesse"] - 100)
        else: # Linker Rand
            x = random.uniform(100, 200)
            y = random.uniform(0, konfiguration["bildGroesse"])

        kandidat = {"x": x, "y": y, "radius": konfiguration["kneipenRadius"]}
        # Prüft, ob der Kandidat gültig ist (Mindestabstand zu bestehenden Dörfern, Wahrzeichen und Kneipen)
        gueltig = all(berechne_entfernung(kandidat, v) >= min_entfernung_zwischen_entitaeten for v in doerfer + wahrzeichen + kneipen) # Abstand auch zu Kneipen prüfen
        if gueltig:
            kneipen.append(kandidat)
        versuche += 1
    return kneipen

# --- Bildgenerierung ---
def generiere_bild():
    """
    Generiert das endgültige Bild, das Dörfer, Wahrzeichen und Kneipen mit Perlin-Rauschen kombiniert.

    :return: Ein PIL Image Objekt im Graustufenformat ('L') oder None, wenn die Generierung der Entitäten fehlschlägt.
    """
    groesse = konfiguration["bildGroesse"]
    bild = Image.new("L", (groesse, groesse))
    pixel = bild.load()

    # Wiederholter Versuch der Generierung von Dörfern, Wahrzeichen und Kneipen, falls anfängliche Versuche fehlschlagen
    doerfer = []
    wahrzeichen = []
    kneipen = []
    max_wiederholungen = 1000
    for _ in range(max_wiederholungen):
        doerfer = generiere_doerfer()
        if doerfer: # Prüfen, ob Dörfer erfolgreich generiert wurden
            wahrzeichen = generiere_wahrzeichen(doerfer)
            if len(wahrzeichen) == konfiguration["wahrzeichenAnzahl"]: # Prüfen, ob genügend Wahrzeichen generiert wurden
                kneipen = generiere_kneipen(doerfer, wahrzeichen)
                if len(kneipen) == konfiguration["kneipenAnzahl"]: # Prüfen, ob genügend Kneipen generiert wurden
                    break # Schleife verlassen, wenn alle Entitäten generiert wurden
        # Listen für den erneuten Versuch leeren
        doerfer = []
        wahrzeichen = []
        kneipen = []

    # Überprüfen, ob die Generierung erfolgreich war
    if not doerfer or len(wahrzeichen) < konfiguration["wahrzeichenAnzahl"] or len(kneipen) < konfiguration["kneipenAnzahl"]:
        print("Konnte nach mehreren Versuchen nicht genügend Dörfer, Wahrzeichen oder Kneipen generieren. Bitte passen Sie die Parameter an.")
        return None # None zurückgeben oder Fehler auslösen, wenn die Generierung fehlschlägt

    # Generiert und kombiniert Perlin-Rauschen
    perlin1 = generiere_perlin_rauschen_2d((groesse, groesse), konfiguration["perlinRauschenGroesse1"])
    perlin2 = generiere_perlin_rauschen_2d((groesse, groesse), konfiguration["perlinRauschenGroesse2"])
    kombiniertes_perlin = (perlin1 * konfiguration["perlinRauschenGewicht1"] + perlin2 * konfiguration["perlinRauschenGewicht2"]) / (konfiguration["perlinRauschenGewicht1"] + konfiguration["perlinRauschenGewicht2"])


    # Berechnet die Intensität für jedes Pixel basierend auf der Nähe zu Entitäten und Perlin-Rauschen
    for y in range(groesse):
        for x in range(groesse):
            intensitaet = 0.0

            # Intensität basierend auf Dörfern
            for v in doerfer:
                d = berechne_entfernung(v, {"x": x, "y": y})
                if d <= v["radius"]:
                    intensitaet = max(intensitaet, 1.0)
                # Anpassung der Ausblendungsberechnung für weicheres Übergehen
                elif d <= v["radius"] * konfiguration["verblassungsMultiplikator"]:
                    # Eine weichere Ausblendungsfunktion verwenden (z.B. umgekehrtes Quadrat oder ähnliches)
                    # Derzeit wird eine lineare Ausblendung verwendet, versuchen wir etwas anderes oder passen nur den Bereich an
                    # Die dunkle Trennung könnte auf den abrupten Übergang und die lineare Ausblendung zurückzuführen sein.
                    # Wir behalten einfach die aktuelle lineare Ausblendung bei, aber die Erhöhung des Fade-Multiplikators wird beim Zusammenführen helfen
                    verblassungs_verhaeltnis = 1 - (d - v["radius"]) / (v["radius"] * (konfiguration["verblassungsMultiplikator"] - 1))
                    intensitaet = max(intensitaet, konfiguration["verblassungsStaerke"] * verblassungs_verhaeltnis)
                else:
                    # Füge auch einen kleinen Beitrag für Entfernungen jenseits des Haupt-Fades zum Zusammenführen hinzu
                    zusaetzliche_verblassungs_entfernung = 200 # Zusätzliche Entfernung, die zum Zusammenführen beiträgt
                    if d <= v["radius"] * konfiguration["verblassungsMultiplikator"] + zusaetzliche_verblassungs_entfernung:
                         zusaetzliches_verblassungs_verhaeltnis = 1 - (d - v["radius"] * konfiguration["verblassungsMultiplikator"]) / zusaetzliche_verblassungs_entfernung
                         intensitaet = max(intensitaet, 0.1 * zusaetzliches_verblassungs_verhaeltnis) # Kleiner Beitrag zum Zusammenführen


            # Intensität basierend auf Wahrzeichen
            for l in wahrzeichen:
                d = berechne_entfernung(l, {"x": x, "y": y})
                if d <= l["radius"]:
                    intensitaet = max(intensitaet, konfiguration["wahrzeichenIntensitaet"])
                elif d <= konfiguration["wahrzeichenVerblassungsEntfernung"]:
                    verblassungs_verhaeltnis = 1 - (d - l["radius"]) / (konfiguration["wahrzeichenVerblassungsEntfernung"] - l["radius"])
                    intensitaet = max(intensitaet, konfiguration["wahrzeichenIntensitaet"] * verblassungs_verhaeltnis)

            # Intensität basierend auf Kneipen
            for p in kneipen:
                d = berechne_entfernung(p, {"x": x, "y": y})
                if d <= p["radius"]:
                    intensitaet = max(intensitaet, konfiguration["kneipenIntensitaet"])
                elif d <= konfiguration["kneipenVerblassungsEntfernung"]:
                    verblassungs_verhaeltnis = 1 - (d - p["radius"]) / (konfiguration["kneipenVerblassungsEntfernung"] - p["radius"])
                    intensitaet = max(intensitaet, konfiguration["kneipenIntensitaet"] * verblassungs_verhaeltnis)


            # Perlin-Rauschen mit einer Gewichtung anwenden, die um Intensität 0,5 stärker ist
            # Wir verwenden eine quadratische Funktion wie -4*(Intensität-0,5)^2 + 1, die bei 0 und 1 0 ist und bei 0,5 1
            perlin_gewicht = (-4 * (intensitaet - 0.5)**2 + 1)
            rausch_wert = kombiniertes_perlin[y][x] * konfiguration["perlinStaerke"] * perlin_gewicht

            # Endgültiger Pixelwert, begrenzt auf 0-255
            wert = min(255, int((intensitaet + rausch_wert) * 255))
            pixel[x, y] = wert

    return bild

# --- Anzeige und Export ---
bild = generiere_bild()

if bild: # Nur anzeigen, wenn die Bildgenerierung erfolgreich war
    plt.figure(figsize=(10, 10))
    plt.imshow(bild, cmap='gray')
    plt.axis('off')
    plt.show()

# --- Download-Link ---
# bild.save("dorfkarte.png")
# from google.colab import files
# files.download("dorfkarte.png")