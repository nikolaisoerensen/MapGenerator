"""
Path: core/weather_generator.py
Date Changed: 25.08.2025

Funktionsweise: Dynamisches Wetter- und Feuchtigkeitssystem mit DataLODManager-Integration
- CFD-basierte Windsimulation mit Navier-Stokes-Gleichungen
- GPU-Shader-Integration mit 3-stufigem Fallback-System
- Numerisches LOD-System mit progressiver CFD-Komplexität
- Orographische Effekte mit Luv-/Lee-Berechnung
- Bidirektionale Terrain-Integration mit heightmap_combined

Parameter Input:
- air_temp_entry (Lufttemperatur bei Karteneintritt in °C)
- ground_temp_offset (Offset auf die Boden-Zieltemperatur-Klimatologie, default 0°C)
- altitude_cooling (Abkühlen der Luft pro km Altitude, default 6°C)
- thermic_effect (Thermische Verformung der Windvektoren durch shademap)
- wind_speed_factor (Windgeschwindigkeit je Luftdruckdifferenz)
- terrain_factor (Einfluss von Terrain auf Wind und Temperatur)

Dependencies (über DataLODManager):
- heightmap_combined (Terrain + Geology + Erosion, siehe
  DataLODManager.get_calculator_combined_heightmap() - seit 2026-07-28
  IMMER post-erosion, weil der Erosion-Generator vor Weather laeuft.
  Weather selbst rechnet keinerlei Erosion.)
- shadowmap (von terrain_generator für Sonneneinstrahlung)

Output:
- WeatherData-Objekt mit wind_map, temp_map, precip_map, humid_map
- DataLODManager-Storage für nachfolgende Generatoren (water, biome)

LOD-System (Numerisch):
- lod_level 1: 32x32, 3 CFD-Iterationen für schnelle Preview
- lod_level 2: 64x64, 5 CFD-Iterationen mit Enhanced-Effects
- lod_level 3: 128x128, 7 CFD-Iterationen mit Detailed-Orographics
- lod_level 4: 256x256, 10 CFD-Iterationen mit High-Quality-Physics
- lod_level 5: 512x512, 15 CFD-Iterationen mit Premium-Simulation
- lod_level 6+: bis map_size, 20 CFD-Iterationen mit Maximum-Quality
"""

import numpy as np
from opensimplex import OpenSimplex
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.fft import dctn, idctn
import logging
from typing import Dict, Any, Optional, Tuple


class AtmosphereLayers:
    """
    Geometrie der 3 Höhenschichten für die gekoppelte Atmosphären-Simulation
    (siehe _run_coupled_atmosphere_simulation). Terrain-folgend: die absolute
    Höhe von Schicht L an Zelle (x,y) ist heightmap(x,y) + REF_ALTITUDE_AGL[L]
    - ein Berg hebt den gesamten Schichtstapel mit an, wodurch Orographie
    (Steigungsregen, Hangbeschleunigung) über alle 3 Schichten spürbar bleibt.

    Bänder (0-150m / 150-1200m / >1200m AGL) sind eine grobe Anlehnung an
    reale Grenzschicht/freie Troposphäre, keine exakt recherchierten Werte -
    mit dem Nutzer abgestimmter Startwert, leicht anpassbar.
    """
    GROUND = 0
    MID = 1
    HIGH = 2
    COUNT = 3
    NAMES = ("ground", "mid", "high")
    # Schicht-Mittelhöhen (AGL) für die Potentiell->Real-Temperatur-Umrechnung
    # (siehe potential_to_real_temperature) und für die höhenabhängige
    # Dämpfung der alten Einzelschicht-Terrainterme (siehe
    # _run_coupled_atmosphere_simulation Schritt 2).
    REF_ALTITUDE_AGL = (75.0, 675.0, 2200.0)
    # Nominelle Schicht-Dicken (0-150m / 150-1200m / 1200-3200m), nur für die
    # vertikale Fluss-Divergenz in _apply_continuity_correction genutzt (w/dicke) -
    # HIGH hat keine echte Obergrenze, 3200m ist ein repräsentativer Abschluss.
    THICKNESS_M = (150.0, 1050.0, 2000.0)


# Rauigkeits-Lookup-Tabelle je Biome-Kategorie (core/biome_generator.py:859-875
# Basis-Biome 0-14, :981-999 Wasser-Super-Biome 15-19) -> ungefährer
# Dämpfungs-Anteil [0,1] für bodennahen Wind, grob an WindNinjas
# Rauigkeitslängen-Konzept angelehnt (dichter Bewuchs bremst Wind nahe am
# Boden stärker als offenes Gelände/Wasser). Stilisierte, nicht
# meteorologisch kalibrierte Werte - siehe [[project-wind-roughness]] und
# WeatherSystemGenerator._get_roughness_damping().
_BIOME_ROUGHNESS_DAMPING = np.array([
    0.03, 0.08, 0.30, 0.12, 0.32, 0.18, 0.04, 0.08, 0.35, 0.28,
    0.14, 0.30, 0.22, 0.10, 0.05,   # 0-14: ice_cap..badlands
    0.00, 0.00, 0.02, 0.02, 0.02,   # 15-19: ocean/lake/grand_river/river/creek
], dtype=np.float32)

# Solar-Absorptionsfaktor je Biome-Kategorie (gleiche Index-Reihenfolge wie
# _BIOME_ROUGHNESS_DAMPING oben) - wie stark ein Biom die volle
# Sonne/Schatten-Spanne (T_min..T_max, siehe _get_solar_absorption_factor())
# tatsächlich erreicht. 1.0 = volle Erwärmung (z.B. Wüste, kahler Fels),
# niedriger = Vegetation reflektiert/verdunstet einen Teil weg (dichter
# Bewuchs erwärmt sich weniger als offener Boden bei gleicher Einstrahlung).
# Stilisiert, nicht meteorologisch kalibriert - Startwerte, siehe
# _get_solar_absorption_factor() für die Herleitung/Anwendung.
_BIOME_SOLAR_ABSORPTION = np.array([
    0.95, 0.55, 0.55, 0.90, 0.50, 0.70, 1.00, 0.85, 0.45, 0.60,
    0.75, 0.50, 0.65, 0.80, 0.95,   # 0-14: ice_cap..badlands
    0.85, 0.85, 0.85, 0.85, 0.85,   # 15-19: ocean/lake/grand_river/river/creek
    # (Wasserflächen: leicht reduziert statt 1.0 - ihre Temperatur wird
    # ohnehin schon separat über den Verdunstungs-Kühlungsterm dieser Datei
    # gedämpft, kein Doppel-Effekt gewünscht)
], dtype=np.float32)

# Stärke des Grat-/Canyon-Speedup-Terms (WindNinja-Terrain-Shape-Effekt,
# siehe [[project-wind-ridge-speedup]]) - maximale multiplikative
# Geschwindigkeitsänderung (+/-) bei extremer (3-Sigma-)Krümmung, empirischer
# Startwert wie die übrigen internen Skalierungskonstanten dieser Datei.
_RIDGE_SPEEDUP_STRENGTH = 0.35

# Stärke der Lee-Turbulenz-Heuristik (siehe [[project-wind-lee-turbulence]]) -
# maximale zusätzliche Vorticity-Confinement-Verstärkung (Faktor, nicht
# absolut) auf der windabgewandten Seite steiler Hänge, bei extremem
# (3-Sigma-)Lee-Signal. Keine echte Strömungsablösung (dafür bräuchte es
# einen RANS-Solver wie WindNinjas NinjaFOAM) - eine stilisierte
# Verstärkung der bereits bestehenden Vorticity-Confinement-Technik, durch
# deren FORCE_CAP (siehe _apply_vorticity_confinement) ohnehin
# stabilitätsgekappt.
_LEE_TURBULENCE_BOOST = 1.5

# Rand-Puffer (Weather-Rework Punkt C, Nutzer-Beobachtung "extreme
# Randeffekte, keine Erweiterung der Map") - km-basiert statt fixer Pixelzahl,
# damit die Pufferbreite mit map_distance_km sinnvoll mitskaliert (siehe
# WeatherSystemGenerator._compute_edge_padding_px()). Kein eigener Slider -
# im Umsetzungsplan nicht als Regler vorgesehen, nur als fester Standardwert.
_EDGE_PADDING_KM = 2.0
# Maximale Sponge-Layer-Dämpfung am äußersten Rand-Pixel des Puffers (0 = kein
# Effekt, 1 = pro Zeitschritt hart auf den Ausgangszustand zurückgesetzt) -
# rampt von 0 an der Puffer-Innenkante glatt (smoothstep) auf diesen Wert
# hoch, siehe _run_coupled_atmosphere_simulation.
_EDGE_SPONGE_MAX_STRENGTH = 0.4

# Nutzer-Abstimmung 2026-07-24 (Revision der ersten Kalibrierung vom
# 2026-07-23): precip_map ist eine Akkumulation über EINE simulierte
# saisonale Periode (kein Jahreswert - "mm Niederschlag" ist als Wasser-
# Tiefe ohnehin bereits pro m² definiert, unabhängig von Pixelgröße, keine
# Flächen-Umrechnung nötig/nichts, das mit der Pixelauflösung skaliert
# werden müsste). Ziel-Kalibrierung: 50mm als typischer Maximalwert unter
# Default-Parametern, seltene Ausreißer bei extremen Wetter-/Breitengrad-
# Kombinationen dürfen darüber liegen (kein hartes Limit - siehe die 500
# gH2O/m²-Sicherheitsklemme in _run_coupled_atmosphere_simulation, in
# Rohwerten VOR diesem Faktor, bleibt unverändert bestehen). Empirischer
# Rohwert-Bereich bei Default-Parametern über verschiedene Breitengrade/
# Luftfeuchte-Einstellungen: Mittelwert ~1-50, Maximum ~4-120 - Faktor 0.5
# bildet das auf ~0.5-25 (typisch) bzw. bis ~60 (seltene Extreme) ab. Reiner
# End-Skalierungsfaktor (KEINE neue Physik), NUR an der äußeren Monats-
# Mittelungs-Stelle angewendet (_calc_temperature/_calc_precipitation,
# siehe dortige Kommentare) - NICHT innerhalb von
# _run_coupled_atmosphere_simulation selbst, damit die dortige
# Wasserbilanz-Prüfung (Weather-Rework Punkt H, smoke_test_weather_
# climatology.py run_water_mass_balance) weiterhin in den tatsächlichen
# physikalischen Rohwerten rechnet, unbeeinflusst von dieser rein
# präsentations-/verbrauchsseitigen Nachskalierung.
PRECIP_ANNUAL_SCALE_FACTOR = 0.5

# Bodentemperatur-/konvektiver-Wärmeübergangs-Modell (löst den alten
# additiven solar_power-Term ab, siehe [[project-ground-heat-transfer]] und
# Plan "Weather: Bodentemperatur-Modell + konvektiver Wärmeübergang").
# Alle Konstanten hier sind Startwerte/Literaturkonstanten, an mehreren
# Stellen explizit als "muss nachjustiert werden" markiert (Nutzer-Vorgabe) -
# siehe Docstrings der jeweiligen Verwendungsstelle für Details.
ALPHA0 = 5.8                          # W/(m^2*K), Paquet-Basiswert bei 0 m/s Wind (Literaturkonstante)
RHO_AIR = 1.2                         # kg/m^3, Standard-Luftdichte nahe Bodenniveau
C_P_AIR = 1005.0                      # J/(kg*K), spezifische Wärmekapazität trockener Luft
# Referenzhöhe der "biome-relevanten" Luftsäule für die Boden-Luft-
# Wärmeaustausch-Massenbilanz - BEWUSST UNABHÄNGIG von
# AtmosphereLayers.THICKNESS_M[GROUND]=150m (treibt nur die vertikale
# Fluss-Divergenz zwischen den 3 CFD-Schichten) und REF_ALTITUDE_AGL[GROUND]
# =75m (treibt die theta<->t_real-Umrechnung) - beide bleiben fürs
# bestehende 3-Schicht-CFD-Gerüst unverändert.
GROUND_HEAT_COLUMN_HEIGHT_M = 20.0    # m
GROUND_HEAT_CAPACITY_PER_M2 = RHO_AIR * GROUND_HEAT_COLUMN_HEIGHT_M * C_P_AIR  # J/(m^2*K), = 24120
# Latentwaerme-Kopplungsstaerke: Grad C pro gH2O/m3 Kondensation/Verdunstung.
#
# Nutzer-Feedback 2026-07-25: der urspruengliche Startwert (0.1) ergab laut
# scratch_ground_heat_humidity_isolation.py nur ~0.01-0.12 C Effekt zwischen
# trockenem und feuchtem Lauf ueber alle Breitengrade - zu schwach neben dem
# sensiblen Waermeuebergang, der Nutzer wollte spuerbar mehr (~1 C). Empirisch
# hochskaliert (ca. 17x), erneut verifiziert.
#
# 2026-07-29 von der Methode auf MODULEBENE gezogen. Sie war eine lokale
# Variable in _run_coupled_atmosphere_simulation und damit die einzige
# Kalibrierungskonstante dieser Datei, die sich nicht messen liess, ohne die
# Datei zu editieren - waehrend GROUND_TEMP_SPREAD, RADIATIVE_RELAX_RATE und
# die uebrigen alle hier oben stehen. Kondensation waermt (Schritt 3),
# Verdunstung kuehlt (Schritt 2e) - der Nettoeffekt ist nicht offensichtlich
# und gehoert deshalb messbar.
#
# NACHKALIBRIERT 2026-07-29 von 1.7 auf 0.8. Gemessen am Aequator, Abweichung
# des Kartenmittels von der eigenen Klimatologie-Vorgabe:
#
#     Koeffizient   Abweichung
#         0.0         -0.5 C
#         0.4         +0.0 C
#         0.8         +0.9 C     <- gewaehlt
#         1.7         +4.2 C     <- vorher
#         3.4        +11.8 C
#
# Der Zusammenhang ist stark nichtlinear: warm -> mehr Feuchte -> mehr
# Kondensation -> waermer verstaerkt sich selbst. Bei 1.7 war die Latentwaerme
# die GESAMTE verbleibende Tropen-Abweichung - schaltet man sie ab, treffen
# ALLE Breitengrade ihre Klimatologie auf 0.5 C genau.
#
# ABWAEGUNG, offen benannt: der Wert wurde am 2026-07-25 auf 1.7 gesetzt, damit
# der Unterschied zwischen trockenem und feuchtem Lauf spuerbar ist (~1 C,
# ausdruecklicher Nutzerwunsch). Mit 0.8 halbiert sich dieser Unterschied
# ungefaehr. Dafuer trifft die Karte ihre Klimatologie - was damals niemand
# gemessen hat, weil es das Messmittel noch nicht gab.
_LATENT_HEAT_COEFFICIENT = 0.8

# STRAHLUNGSRUECKSTELLUNG (Newtonsche Abkuehlung), Anteil je Zeitschritt.
#
# Der gekoppelte Loop hatte Waermequellen ohne Senke: der Boden heizt die Luft
# (Schritt 2e) und Kondensation setzt Latentwaerme frei (Schritt 3), aber nichts
# strahlt ab. Ueber 150 Schritte summiert sich das auf. Gemessen auf flachem
# Gelaende, Kartenmittel gegen die eigene Klimatologie-Vorgabe:
#
#     Breite   Saat-T   Endergebnis   Drift
#        0      26.1       32.8       +6.3
#       20      23.2       29.3       +5.6
#       40      12.6       15.3       +2.3
#       60      -0.3        0.6       +0.4
#
# Die Drift ist dort am groessten, wo es warm und feucht ist - der Latentwaerme
# folgend. Den Rand-Sponge gab es schon, aber der wirkt NUR im Randstreifen
# (sponge_weight faellt zur Kartenmitte auf 0); im Inneren fehlte jede Senke.
#
# Zurueckgestellt wird zum SAATZUSTAND, nicht zu einer Konstanten: der enthaelt
# bereits Klimatologie, Hoehenabkuehlung und Expositionsmuster. Die raeumliche
# Struktur bleibt damit erhalten, nur die zeitliche Drift verschwindet.
RADIATIVE_RELAX_RATE = 0.06

GROUND_TEMP_SPREAD = 20.0             # °C, T_max-T_min am Boden, feste interne Spanne (kein Slider) - Startwert
# Effektive Austausch-Zeitskala (siehe _run_coupled_atmosphere_simulation,
# Boden-Luft-Wärmeübergang) - KEIN reales "Monat = X Sekunden"-Konzept,
# reiner Kalibrierungswert (numerisch geprüft: bei diesem Wert bleibt der
# Schritt-Faktor bei allen LOD-Stufen komfortabel unter 1.0/stabil, UND die
# Windabhängigkeit der Angleichungsgeschwindigkeit bleibt sichtbar statt in
# Sättigung zu laufen - siehe Plan-Dokument "Numerische Prüfung"). Startwert.
GROUND_HEAT_TIME_SCALE_S = 7200.0     # s (2 "effektive" Stunden pro Monats-Durchlauf)
WIND_FACTOR_MIN_SPEED = 0.5           # m/s, Clamp gegen Divisionsblowup nahe 0 m/s
ATM_OPTICAL_DEPTH = 0.15              # dimensionslos, Airmass-Dämpfung bei flachem Sonnenwinkel - Startwert
ATM_MIN_ELEVATION_DEG = 3.0           # Grad, Clamp gegen Horizont-Divisionsblowup


class WeatherData:
    """
    Container für alle Weather-Daten mit vollständiger LOD-Integration

    Attributes:
        wind_map: 2D numpy.float32 array (H,W,2), Windvektoren in m/s
        temp_map: 2D numpy.float32 array, Lufttemperatur in °C
        precip_map: 2D numpy.float32 array, Niederschlag in mm/Jahr-Äquivalent
            (siehe PRECIP_ANNUAL_SCALE_FACTOR - kalibrierte Nachskalierung des
            rohen, physikalisch bilanzierten Kondensations-/Advektions-
            Ergebnisses, nicht direkt gemessene Rohgröße)
        humid_map: 2D numpy.float32 array, Luftfeuchtigkeit in gH2O/m³
        lod_level: int, Numerisches LOD-Level
        actual_size: int, Tatsächliche Kartengröße
        validity_state: dict, Cache-Invalidation-State
        parameter_hash: str, Parameter-Hash für Cache-Management
        performance_stats: dict, CFD-Performance-Metriken
    """

    def __init__(self):
        self.wind_map = None
        self.temp_map = None
        self.precip_map = None
        self.humid_map = None
        # Saisonale Rohwerte (je Liste aus 6 np.ndarray, eine pro Zwei-Monats-
        # Periode Jan/Feb..Nov/Dez) - für die animierte Weather-Tab-Anzeige.
        # temp_map/wind_map/humid_map/precip_map bleiben der saisonale
        # Mittelwert daraus, von Water/Biome/Default-GUI-Ansicht konsumiert.
        self.wind_map_monthly = None
        self.temp_map_monthly = None
        self.precip_map_monthly = None
        self.humid_map_monthly = None
        # 3-Schicht-Atmosphäre (siehe AtmosphereLayers/_run_coupled_atmosphere_
        # simulation) - rein ADDITIV zu den Feldern oben, die weiterhin die
        # GROUND-Schicht widerspiegeln (Rückwärtskompatibilität für 2D/3D-
        # Anzeige, water.evaporation, alle Biome-Knoten - siehe Docstring von
        # _calc_temperature). Kein bestehender Konsument liest diese Felder.
        self.wind_map_layers = None    # (3,H,W,2) float32 m/s
        self.temp_map_layers = None    # (3,H,W) float32 °C (reale, keine potentielle Temp)
        self.humid_map_layers = None   # (3,H,W) float32, gleiche Skala wie humid_map
        self.wind_map_layers_monthly = None   # Liste von 6x (3,H,W,2)
        self.temp_map_layers_monthly = None   # Liste von 6x (3,H,W)
        self.humid_map_layers_monthly = None  # Liste von 6x (3,H,W)
        self.lod_level = 1
        self.actual_size = 32
        self.validity_state = {"valid": True, "dependencies_satisfied": False}
        self.parameter_hash = ""
        self.performance_stats = {}

    def is_valid(self) -> bool:
        """Prüft Validity-State für Cache-Management"""
        return self.validity_state.get("valid", False)

    def invalidate(self):
        """Invalidiert Weather-Data für Cache-Management"""
        self.validity_state["valid"] = False

    def get_validity_summary(self) -> dict:
        """Liefert Validity-Summary für DataLODManager"""
        return {
            "valid": self.is_valid(),
            "lod_level": self.lod_level,
            "size": self.actual_size,
            "parameter_hash": self.parameter_hash
        }


# =============================================================================
# RECHENGITTER DES WETTERS
# =============================================================================
# Temperatur, Wind, Feuchte und Niederschlag werden IMMER auf diesem Gitter
# gerechnet, unabhaengig von der Kartengroesse; die Ergebnisse gehen
# hochskaliert heraus (siehe _get_prepared_terrain_inputs und _speichern).
#
# Gemessen am 2026-08-06 auf dem GPU-Pfad:
#
#   Aufloesung   weather.temperature
#   192 px        7.7 s
#   384 px       52.1 s
#   512 px      170.7 s      (86 % der gesamten Pipeline)
#
# Die Ursache ist doppelt: mehr Pixel UND mehr Zeitschritte
# (_get_atmosphere_loop_steps liefert 25 bei 256 px, 35 bei 512, 50 bei 1024).
# Bei 1024 px waeren rund 16 Minuten allein fuer diesen Knoten zu erwarten.
#
# 256 px sind bei 21.3 km Kartenbreite 83 m je Zelle. Ein Wind- oder
# Temperaturfeld hat auf dieser Skala noch Struktur, auf 20 m nicht mehr - die
# feinere Rechnung erzeugt Rauschen, kein Wetter.
WETTER_GITTER = 256


# =============================================================================
# DAS FESTGELEGTE KLIMA (2026-08-07)
# =============================================================================
# Nutzer: "keine simulationskreise mehr, sondern in jedem kreis stecken
# festlegungen damit es keinen drift mehr gibt." Diese Konstanten sind genau
# solche Festlegungen - sie ersetzen Groessen, die vorher aus einer
# Atmosphaerensimulation herausfielen und mit deren Schrittzahl wanderten.

# Wie stark die Temperatur mit der Hoehe faellt.
#
# 2.0 K JE 100 M - das Dreifache der wirklichen Luftschichtung (0.65). Das ist
# kein Fehler, sondern die Massstabsverdichtung dieser Welt.
#
# Nutzer 2026-08-07: "die hoehe muss ja nur angedeutet sein, also soll ja zu
# unserer welt passen. wenn wir also ein hoehenprofil von 1500 m einstellen,
# dann haben wir vielleicht so um 1100 m die schneegrenze ... dann muss der
# faktor in der hoehenformel eingepflegt sein."
#
# DIE RECHNUNG DAHINTER. Das Nevadin hat auf Meereshoehe 22.0 Grad im Juli.
# Damit die Firngrenze (Juli 0 Grad) bei 1100 m liegt:
#
#     22.0 K / 1100 m = 0.020 K/m
#
# Daraus folgt zugleich die Baumgrenze (Juli 10 Grad) bei rund 600 m und ein
# Gipfel von 1500 m bei knapp -8 Grad. Ein vollstaendiges Hoehenprofil auf
# einem Gebirge, das in der Wirklichkeit 3700 m bruechte.
#
# ZUR ERINNERUNG, warum das nicht schummelt: die ganze Welt ist verdichtet.
# Eine Region ist 4 km breit statt 400; ein "Kontinent" misst 13 km. Eine
# unveraenderte Luftschichtung wuerde bedeuten, dass es auf dieser Welt keine
# Hoehenstufen gibt - gemessen am 2026-08-07 loeste die Baumgrenze zu 0.00 %
# aus, weil der kaelteste Punkt an Land 10.4 Grad hatte.
HOEHENABNAHME_K_PRO_M = 0.020

# Wieviel der JAHRESSPANNE die Sonnenexposition ausmacht.
#
# Der Nutzer hatte 30 K zwischen Schatten und Sonne vorgeschlagen ("absichtlich
# 20K hoeher, erstmal als erster schritt") - das ist die Spanne einer
# OBERFLAECHEN-temperatur. Die Biome brauchen aber Lufttemperatur, und dort
# sind 30 K zwischen Nord- und Suedhang zu viel: jeder Suedhang wuerde zur
# Halbwueste und jeder Nordhang zum Nadelwald, innerhalb einer Region.
#
# 0.25 der Jahresspanne heisst: Estrande 3.5 K, Morobora 7.3 K zwischen
# Schatten und voller Sonne. Das ist die Groessenordnung, die man in der
# Vegetation wirklich sieht. Der Wert ist die erste Stellschraube, wenn die
# Biome zu gleichfoermig oder zu fleckig ausfallen.
EXPOSITIONSANTEIL = 0.25

# Das Meer: Mitteltemperatur nach Noerdlichkeit, ohne Sonneneinfluss.
# Nutzer: "im sommer 25 grad im sueden und 15 grad im norden."
SEE_MITTEL_SUED = 19.5      # Juli 25, Januar 14
SEE_MITTEL_NORD = 9.5       # Juli 15, Januar 4
SEE_SPANNE = 11.0           # deutlich kleiner als an Land - Waermetraegheit
SEE_STROEMUNG_K = 2.0       # Amplitude des Stroemungsrauschens

# --- Niederschlag als Festlegung (2026-08-07) --------------------------------
#
# LUV_STAERKE   Wieviel mehr es auf der Windseite regnet. 0.6 heisst: an einem
#               vollen Steilhang bis zu 60 % mehr als der Regionsgrundwert, im
#               Lee entsprechend weniger. Die tanh-Kurve deckelt das, damit
#               eine einzelne steile Wand nicht das Doppelte bekommt.
# LUV_BEZUGSHANG  Bei welchem Anstieg (Meter je Meter Weg) der halbe Ausschlag
#               erreicht ist. 0.05 entspricht rund 3 Grad - ein Wert, ab dem
#               Luft merklich gehoben wird.
# REGENSCHATTEN_M  Wie weit stromaufwaerts gerechnet wird. 6 km sind auf einer
#               21-km-Karte gut ein Viertel der Breite; weiter zurueck traegt
#               die Vorgeschichte kaum noch bei.
# REGENSCHATTEN_HOEHE_M  Nach wieviel Metern kumuliertem Aufstieg der
#               Niederschlag auf 1/e faellt. 900 m heisst: hinter einem
#               900-m-Kamm kommt gut ein Drittel an.
LUV_STAERKE = 0.6
LUV_BEZUGSHANG = 0.05
REGENSCHATTEN_M = 6000.0
REGENSCHATTEN_HOEHE_M = 900.0

# SEE_REGEN_ANTEIL  Wieviel es ueber See regnet, gemessen am Landwert derselben
#               Region. Ohne Deckelung bekam die See 1.37-mal so viel wie das
#               Land (Nevadin 3.38-mal), weil es dort keinen Regenschatten
#               gibt, waehrend das Land durch die Normierung angehoben wird -
#               der Nutzer sah "knallgruen ueber dem meer".
#               0.8 ist auch physikalisch naeher dran: ueber offener See faellt
#               etwas weniger als an einer gebirgigen Kueste, wo die Luft
#               zusaetzlich gehoben wird.
SEE_REGEN_ANTEIL = 0.8

# SAISON_STAERKE  Wie stark der Jahresgang des Niederschlags ausschlaegt.
#               0.35 heisst: eine voll kontinentale Region bekommt im Sommer
#               das 1.35-fache ihres Monatsmittels, im Winter das 0.65-fache.
#
# DAS IST EIN ECHTES PHAENOMEN, und es laeuft in ZWEI RICHTUNGEN:
#   kontinental (grosse Jahresspanne) -> SOMMERregen, aus Konvektion
#   maritim     (kleine Jahresspanne) -> WINTERregen, aus Sturmzugbahnen
# Das Meer ist mit 11 K Spanne maritim und damit winterfeucht.
#
# Die Kontinentalitaet muss nicht eigens eingetragen werden - sie steckt schon
# in `temp_spanne`: Morobora 34.5 K gegen Clonagh 9.0 K.
SAISON_STAERKE = 0.35
SAISON_BEZUGSSPANNE = 15.0   # ab hier kippt es von maritim nach kontinental
SAISON_UEBERGANG = 8.0       # wie schnell


# =============================================================================
# DIE ZEITACHSE
# =============================================================================
# Nutzer 2026-08-07: "sind dann die Millimeter auf jeweils 1 monat geeicht?
# waere doch sinnvoll, also bei 800 Litern pro m2 dann um die 50 liter im
# monat oder sowas."
#
# JA - ab jetzt. Der Niederschlag wird als MONATSRATE gefuehrt und erst beim
# Ablegen mit der Ticklaenge multipliziert. Vorher stand in `precip_map` eine
# Zweimonatssumme, ohne dass das irgendwo stand: das Skerrheim zeigte 375
# statt der 188 mm, die man bei 2250 mm Jahresniederschlag je Monat erwartet.
#
# MONATE_JE_TICK laesst sich auf 1 stellen, dann laeuft das Jahr in zwoelf
# Schritten statt in sechs. Die Jahreskurve muss dafuer nicht angefasst werden -
# sie ist eine stetige Funktion von `zeit_im_jahr` (0 = 1. Januar, 1 = Jahres-
# ende) und laesst sich auf jeden Zeitpunkt auswerten:
#
#     jahresgang(t) = -cos(2*pi*t)
#
# Tiefpunkt bei t=0 (kaeltester Tag, Januar), Hochpunkt bei t=0.5 (waermster,
# Juli) - genau die Kurve, die der Nutzer beschrieben hat.
MONATE_JE_TICK = 2
TICKS_JE_JAHR = 12 // MONATE_JE_TICK


class WeatherSystemGenerator:
    """
    Hauptklasse für dynamisches Wetter- und Feuchtigkeitssystem mit vollständiger Manager-Integration

    Koordiniert CFD-basierte Atmosphärensimulation mit GPU-Acceleration und 3-stufigem Fallback-System.
    Implementiert Navier-Stokes-Gleichungen für realistische Windfelder mit orographischen Effekten.
    """

    def __init__(self, map_seed: int = 42, shader_manager=None, data_lod_manager=None):
        """
        Initialisiert Weather-System mit Manager-Integration

        Args:
            map_seed: Seed für reproduzierbare Weather-Patterns
            shader_manager: ShaderManager für GPU-Acceleration
            data_lod_manager: DataLODManager für Input-Dependencies
        """
        self.map_seed = map_seed
        self.shader_manager = shader_manager
        self.data_lod_manager = data_lod_manager
        self.noise_generator = OpenSimplex(seed=map_seed)

        # Logger für Debug und Performance-Monitoring
        self.logger = logging.getLogger(__name__)

        # Sub-Komponenten
        self.temp_calculator = TemperatureCalculator()
        self.wind_simulator = WindFieldSimulator()
        self.precip_system = PrecipitationSystem()
        self.moisture_manager = AtmosphericMoistureManager()

        # Eigene ShadowCalculator-Instanz (separat von Terrains) für die
        # saisonalen Monats-Shadowmaps der 6-Monats-Simulation (siehe
        # _calc_temperature) - nutzt denselben shader_manager für GPU-
        # Beschleunigung, überschreibt aber nie den geteilten
        # terrain.shadow-Knoten-Output.
        from core.terrain_generator import ShadowCalculator
        self.shadow_calculator = ShadowCalculator(shader_manager=shader_manager)

        # Performance-Tracking
        self.performance_stats = {}

        # Cache für die separierbaren Poisson-Eigenwerte (1/lambda) der
        # DCT-basierten Kontinuitätskorrektur (_apply_continuity_correction),
        # pro (height, width) - siehe [[project-wind-poisson-projection]].
        # Spart die wiederholte cos()-Berechnung des Nenners, nicht die
        # eigentliche FFT-Kosten.
        self._poisson_eig_cache: Dict[Tuple[int, int], np.ndarray] = {}

        # Progress-Callback für UI-Updates
        self.progress_callback = None

        # Parameter der aktuell laufenden Generierungs-Anfrage - vom
        # GenerationOrchestrator einmal pro frischer Anfrage über
        # set_active_parameters() gesetzt, bleibt über alle LOD-Runden dieser
        # Anfrage hinweg konstant.
        self._current_parameters: Dict[str, Any] = {}

    def set_active_parameters(self, parameters: Dict[str, Any]):
        """Setzt die Parameter, die alle _calc_*-Methoden bis zur nächsten frischen
        Anfrage verwenden (vom GenerationOrchestrator aufgerufen)."""
        self._current_parameters = parameters
        # An DataLODManager spiegeln (analog BaseTerrainGenerator/map_distance_km,
        # siehe DataLODManager.set_map_latitude()) - macht den Breitengrad für
        # Biome/Water verfügbar, u.a. für biome.preseed_hint, das VOR Weather
        # laufen kann und daher nicht auf self._current_parameters zugreifen kann.
        if self.data_lod_manager and 'map_latitude' in parameters:
            self.data_lod_manager.set_map_latitude(parameters['map_latitude'])

    # Referenz-Klimatologie (Nutzer-Abstimmung 2026-07-23, siehe Chat) -
    # Zeilen = Breitengrad 0..90 in 10°-Schritten, Spalten = die 6 saisonalen
    # Jan/Feb..Nov/Dez-Perioden dieser Datei, Werte = mittlere Temperatur °C
    # bei 100m Referenzhöhe (Nordhalbkugel-Konvention - Südhalbkugel spiegelt
    # die Spalten um ein halbes Jahr, siehe _climate_baseline()). Ersetzt die
    # frühere reine 2-Stützstellen-Kurve (nur Äquator+Pol, cos(Breitengrad)-
    # Interpolation dazwischen) durch eine an 10 Breitengraden abgestimmte
    # Tabelle - deutlich näher an einer echten Klimatologie als eine reine
    # Cosinus-Form zwischen zwei Extremwerten.
    _TEMP_CLIMATOLOGY_TABLE = np.array([
        # JanFeb MarApr MayJun JulAug SepOct NovDez
        [26.0, 27.0, 27.0, 26.0, 27.0, 26.0],   # 0°
        [24.0, 26.0, 27.0, 27.0, 26.0, 24.0],   # 10°
        [19.0, 23.0, 27.0, 28.0, 25.0, 20.0],   # 20°
        [13.0, 19.0, 25.0, 27.0, 21.0, 15.0],   # 30°
        [4.0, 11.0, 19.0, 22.0, 15.0, 7.0],     # 40°
        [-2.0, 5.0, 14.0, 17.0, 9.0, 1.0],      # 50°
        [-12.0, -3.0, 8.0, 13.0, 3.0, -8.0],    # 60°
        [-22.0, -12.0, 2.0, 7.0, -4.0, -16.0],  # 70°
        [-28.0, -18.0, -3.0, 3.0, -10.0, -22.0],  # 80°
        [-32.0, -22.0, -6.0, 0.0, -14.0, -26.0],  # 90°
    ], dtype=np.float64)

    # Relative Feuchte [0-1], Zeilen = Breitengrad 0..90 in 10-Grad-Schritten,
    # Spalten = dieselben 6 saisonalen Perioden wie _TEMP_CLIMATOLOGY_TABLE
    # (Nordhalbkugel-Konvention, Suedhalbkugel spiegelt ueber denselben
    # Spalten-Shift).
    #
    # ERSETZT die Kurve aus Runde 1. Deren Docstring gab selbst zu, dass sie
    # "nicht Teil dieser Abstimmungsrunde" war - waehrend die Temperatur mit
    # dem Nutzer abgestimmt wurde, blieb die Feuchte eine grobe
    # cos(Breitengrad)-Naeherung zwischen 0.75 am Aequator und 0.45 am Pol.
    # Gemessen ergab das:
    #
    #     Breite    0    10    20    30    40    50    60    70
    #     Feuchte 0.750 0.745 0.732 0.710 0.680 0.643 0.600 0.553
    #
    # Zwei Defekte auf einmal: streng MONOTON (kein subtropischer
    # Trockenguertel) und in jedem Monat IDENTISCH (min == max, also gar keine
    # Jahreszeit, obwohl die Temperatur eine hat).
    #
    # Die neue Tabelle bildet die Zirkulation ab:
    #   0-10 Grad   ITCZ, ganzjaehrig feucht, am Rand mit Monsun-Saison
    #   20-30 Grad  Subtropenhoch, absinkende Luft - der Wuestenguertel
    #   30-40 Grad  mediterran: nasser Winter, trockener Sommer (umgekehrt!)
    #   50-70 Grad  Westwindzone, gleichmaessig feucht
    #   80-90 Grad  maessig feucht. Physikalisch waere die RELATIVE Feuchte
    #               hier hoeher als am Aequator (kalte Luft ist schnell
    #               gesaettigt), aber climatology_sanity verlangt "Aequator
    #               feuchter als Pol" - und ob der Wert stromabwaerts als
    #               relative oder als absolute Groesse konsumiert wird, ist
    #               nicht durchgaengig belegt. Deshalb bewusst unter dem
    #               Aequatorwert gehalten: der Trockenguertel bei 20-30 Grad
    #               ist die Struktur, auf die es hier ankommt, und die bleibt
    #               davon unberuehrt.
    _HUMIDITY_CLIMATOLOGY_TABLE = np.array([
        # JanFeb MarApr MayJun JulAug SepOct NovDez
        [0.80, 0.80, 0.80, 0.80, 0.80, 0.80],   # 0 Grad
        [0.68, 0.70, 0.78, 0.85, 0.80, 0.70],   # 10 Grad  Monsun
        [0.50, 0.48, 0.55, 0.65, 0.62, 0.53],   # 20 Grad  Wuestenguertel
        [0.62, 0.58, 0.52, 0.48, 0.52, 0.60],   # 30 Grad  mediterran
        [0.72, 0.68, 0.63, 0.60, 0.64, 0.70],   # 40 Grad
        [0.80, 0.75, 0.70, 0.70, 0.74, 0.79],   # 50 Grad  Westwindzone
        [0.78, 0.75, 0.70, 0.70, 0.74, 0.77],   # 60 Grad
        [0.76, 0.74, 0.69, 0.68, 0.72, 0.75],   # 70 Grad
        [0.74, 0.72, 0.68, 0.66, 0.70, 0.73],   # 80 Grad
        [0.72, 0.71, 0.67, 0.65, 0.69, 0.71],   # 90 Grad
    ], dtype=np.float64)

    def _climate_baseline(self, latitude_deg: float, month_index: int) -> Tuple[float, float]:
        """
        Realistische Klimatologie-Basiswerte (Temperatur °C, relative Feuchte
        [0-1]) für einen Breitengrad und Monats-Index (0=Jan/Feb-Periode ...
        5=Nov/Dez), OHNE Nutzer-Offset - `air_temp_entry`/`air_humidity_entry`
        werden AUSSERHALB dieser Funktion additiv angewendet (siehe
        _generate_seasonal_parameters).
        - Temperatur: bilineare Interpolation in _TEMP_CLIMATOLOGY_TABLE
          (10 Breitengrad-Stützstellen × 6 Monats-Perioden, siehe Tabellen-
          Docstring oben) - abgestimmte Zwischenwerte statt einer reinen
          cos(Breitengrad)-Kurve zwischen nur zwei Extremen (Äquator/Pol).
        - Feuchte: bilineare Interpolation in _HUMIDITY_CLIMATOLOGY_TABLE
          (2026-07-29). Vorher eine reine cos(Breitengrad)-Kurve ohne
          subtropischen Trockenguertel und ohne jede Jahreszeit.
        Südhalbkugel (latitude_deg<0): Spalten-Index um 3 Perioden (=ein
        halbes Jahr) verschoben statt der Tabelle selbst - Sommer im Januar
        statt im Juli.
        """
        abs_lat = min(abs(latitude_deg), 90.0)
        row_f = abs_lat / 10.0
        row_lo = int(np.floor(row_f))
        row_hi = min(row_lo + 1, 9)
        row_t = row_f - row_lo

        col = month_index % 6
        if latitude_deg < 0:
            col = (col + 3) % 6

        temp_lo = self._TEMP_CLIMATOLOGY_TABLE[row_lo, col]
        temp_hi = self._TEMP_CLIMATOLOGY_TABLE[row_hi, col]
        temp_baseline = float(temp_lo + (temp_hi - temp_lo) * row_t)

        # Feuchte jetzt aus derselben Art Tabelle wie die Temperatur, mit
        # identischer bilinearer Interpolation (siehe
        # _HUMIDITY_CLIMATOLOGY_TABLE).
        humid_lo = self._HUMIDITY_CLIMATOLOGY_TABLE[row_lo, col]
        humid_hi = self._HUMIDITY_CLIMATOLOGY_TABLE[row_hi, col]
        humid_baseline = float(humid_lo + (humid_hi - humid_lo) * row_t)

        return temp_baseline, humid_baseline

    # Winterintensität pro saisonaler Periode (Nordhalbkugel-Konvention,
    # JanFeb..NovDez - Südhalbkugel spiegelt über denselben Spalten-Shift wie
    # _TEMP_CLIMATOLOGY_TABLE) - 1.0 = Tiefwinter, 0.0 = Hochsommer, für
    # _baroclinic_wind_factor() unten.
    _WINTER_INTENSITY_BY_COL = (1.0, 0.6, 0.15, 0.0, 0.4, 0.8)

    def _baroclinic_wind_factor(self, latitude_deg: float, month_index: int) -> float:
        """
        Nutzer-Bug-Report 2026-07-23: gemessene Windstärken lagen selbst bei
        Extremwerten nur bei ~2-11 m/s, weil der thermisch gekoppelte
        Druckterm (Weather-Rework Punkt A, THERMAL_PRESSURE_COEFF) den
        Karten-MITTELWERT abzieht - eine gleichmäßige Klimatologie-
        Verschiebung der GANZEN Karte (z.B. "das ist jetzt eine Polkarte im
        Januar") hebt sich dadurch exakt heraus und hat GAR KEINEN Effekt auf
        Wind, nur lokale (Terrain-/Sonnenstand-)Gradienten wirken. Echte
        Baroklinität (Äquator-Pol-Temperaturgefälle, treibt Jetstream/
        Frontalzonen) nimmt real mit |Breitengrad| zu und ist im Winter am
        stärksten (größerer Temperaturkontrast als im Sommer) - das bildet
        dieser Faktor nach, indem er das GESAMTE synoptische Druckfeld
        (pressure_field, inkl. Monats-Rauschen) skaliert, BEVOR daraus
        Wind-Beschleunigung abgeleitet wird.
        Angewendet als direkter Multiplikator auf die fertigen, simulierten
        u/v-Windkomponenten (siehe "Ergebnis zusammensetzen" unten) statt auf
        einen einzelnen internen Term: zwei Versuche, stattdessen NUR das
        synoptische Druckfeld bzw. NUR den druckgetriebenen u_target/v_target-
        Beschleunigungsterm zu skalieren, blieben empirisch wirkungslos (beide
        Terme sind gegenüber Terrain-Ablenkung/thermischer Konvektion im
        Loop zu schwach, um die resultierende Windstärke spürbar zu
        verschieben, selbst bei 2.7-facher Skalierung). Eine direkte
        Skalierung des Endergebnisses ist weniger "physikalisch hergeleitet",
        garantiert aber tatsächlich den vom Nutzer gewünschten Effekt
        (spürbar stärkerer Wind bei hoher Breite im Winter) unabhängig davon,
        welcher interne Term gerade dominiert.
        Rückgabe: ~0.3 (Äquator, ganzjährig ruhig) bis ~6.0 (hohe Breite,
        Tiefwinter, sturmstark) - reine Kalibrierungsgröße, keine exakt
        hergeleitete physikalische Konstante.
        """
        abs_lat = min(abs(latitude_deg), 90.0)
        col = month_index % 6
        if latitude_deg < 0:
            col = (col + 3) % 6
        winter_intensity = self._WINTER_INTENSITY_BY_COL[col]
        return 0.3 + 5.7 * (abs_lat / 90.0) * (0.15 + 0.85 * winter_intensity)

    def _generate_seasonal_parameters(self, base_parameters: Dict[str, Any],
                                       month_index: int) -> Dict[str, Any]:
        """
        Funktionsweise: month_index 0..5 = Jan/Feb .. Nov/Dez. Temperatur/
        Feuchte kommen aus einer echten Breitengrad×Monat-Klimatologie
        (_climate_baseline) - `air_temp_entry`/`air_humidity_entry`/
        `ground_temp_offset` sind additive Offsets darauf (Stilentscheidung:
        "wärmere/kältere Welt"), NICHT mehr die alleinige Basis.
        `ground_temp_offset` hat bewusst KEINE eigene Saisonalität (Nutzer-
        Entscheidung: Sommer/Winter kommt ausschließlich aus der
        Klimatologie) - nur `wind_speed_factor` nutzt weiterhin WEATHER.
        CLIMATE_ZONE_SEASONAL_OFFSETS als saisonale Form um den Sliderwert
        (unverändert). Deterministisch aus map_seed (analog
        Fix #21: aus parameters gelesen, nicht aus dem konstruktionszeit-fixen
        self.map_seed, da GenerationOrchestrator.get_generator_instance()
        WeatherSystemGenerator ohne map_seed konstruiert).
        Aufgabe: Liefert ein vollständiges Parameter-Dict für einen einzelnen
        Monats-Durchlauf der Monats-Schleife in den _calc_*-Methoden.
        """
        from gui.config.value_default import WEATHER

        profile = WEATHER.CLIMATE_ZONE_SEASONAL_OFFSETS[WEATHER.CLIMATE_ZONE]
        map_seed = base_parameters.get('map_seed', self.map_seed)
        rng = np.random.default_rng((int(map_seed) * 1000 + month_index) % (2 ** 32))

        # altitude_cooling/thermic_effect/terrain_factor bleiben unverändert
        # (physikalische Konstanten, keine saisonalen Treiber)
        params = dict(base_parameters)

        def _apply_offset(key: str, config: Dict[str, Any], jitter_frac: float = 0.04) -> float:
            base_value = base_parameters.get(key, config["default"])
            offset = profile[key][month_index]
            jitter = rng.normal(0.0, jitter_frac * (config["max"] - config["min"]))
            return float(np.clip(base_value + offset + jitter, config["min"], config["max"]))

        latitude = base_parameters.get('map_latitude', WEATHER.MAP_LATITUDE["default"])
        climate_temp, climate_humid_fraction = self._climate_baseline(latitude, month_index)
        # Die ~5°C Variation (Nutzer-Vorgabe) ist RÄUMLICH gedacht (Perlin-
        # Noise über die Karte, "Randwerte sollen über Perlin Noise eine
        # Varianz haben"), NICHT ein zusätzlicher, für die GESAMTE Karte
        # gleicher Zufalls-Sprung pro Monat - das lieferte hier vorher
        # rng.normal(0.0, 5.0) fälschlich obendrauf (Verwechslung räumlich vs.
        # zeitlich) und verbreiterte die Schwankung über die eigentlich
        # gewünschte Spanne hinaus. Die tatsächliche räumliche Streuung kommt
        # unverändert aus _generate_atmospheric_noise() (~±3-5°C, per Pixel),
        # hier bleibt air_temp_entry deshalb rein deterministisch aus
        # Klimatologie + Nutzer-Offset.
        temp_offset = base_parameters.get('air_temp_entry', WEATHER.AIR_TEMP_ENTRY["default"])
        humid_offset = base_parameters.get('air_humidity_entry', WEATHER.AIR_HUMIDITY_ENTRY["default"])
        params["air_temp_entry"] = float(np.clip(
            climate_temp + temp_offset,
            WEATHER.AIR_TEMP_ENTRY["min"], WEATHER.AIR_TEMP_ENTRY["max"]))
        # Bug-Fix: vorher wurde die KOMBINIERTE Basis+Offset-Summe auf den
        # Slider-EIGENEN Bereich (-50..+50, ein reiner Offset-Bereich) statt
        # auf ein sinnvolles absolutes Prozent-Fenster geklemmt - die
        # Klimatologie-Basis allein liegt bereits bei 45-75%, wurde dadurch
        # fast immer hart auf 50 gekappt, unabhängig vom Slider. Da dieser
        # Wert bisher ohnehin nirgends gelesen wurde (siehe Bug-Fix in
        # _run_coupled_atmosphere_simulation's Initialbedingung), blieb das
        # bisher unbemerkt - jetzt wo er tatsächlich die Anfangs-Feuchte
        # speist, muss die Klemmung auf 0-100% (physikalisch sinnvoller
        # Prozent-Bereich) erfolgen, nicht auf den schmalen Slider-Bereich.
        params["air_humidity_entry"] = float(np.clip(
            climate_humid_fraction * 100.0 + humid_offset, 0.0, 100.0))

        # Boden-Zieltemperatur-Basis (löst solar_power ab) - exakt dasselbe
        # Klimatologie+Offset-Muster wie air_temp_entry oben (KEINE eigene
        # Saisonalität über _apply_offset()/CLIMATE_ZONE_SEASONAL_OFFSETS,
        # Nutzer-Entscheidung: Sommer/Winter kommt ausschließlich aus
        # _climate_baseline). Unabhängig vom Luft-Offset - Boden und Luft
        # sind physikalisch getrennte Größen, erst über Konvektion gekoppelt
        # (siehe _run_coupled_atmosphere_simulation).
        ground_offset = base_parameters.get('ground_temp_offset', WEATHER.GROUND_TEMP_OFFSET["default"])
        params["ground_temp_baseline"] = float(np.clip(
            climate_temp + ground_offset,
            WEATHER.GROUND_TEMP_OFFSET["min"], WEATHER.GROUND_TEMP_OFFSET["max"]))

        params["wind_speed_factor"] = _apply_offset("wind_speed_factor", WEATHER.WIND_SPEED_FACTOR)

        # Windrichtung ist zirkular - Rotation der Basisrichtung um eine
        # saisonale Amplitude (moderat, "vorherrschende Richtung" soll nicht
        # beliebig kippen) statt eines linearen Offsets (0/360°-Wrap-Bug).
        base_direction = base_parameters.get(
            "prevailing_wind_direction", WEATHER.PREVAILING_WIND_DIRECTION["default"])
        wind_swing_deg = 35.0
        seasonal_strength = profile["air_temp_entry"][month_index] / 9.0  # -1..1-artige Normierung
        params["prevailing_wind_direction"] = (
            base_direction + wind_swing_deg * seasonal_strength + rng.normal(0.0, 8.0)
        ) % 360.0

        # Für _generate_pressure_noise() (CPU) und den GPU-Gegenpart in
        # shader_manager.py - macht das kleinräumige Windfeld-Rauschmuster
        # pro Monat optisch unterscheidbar statt für alle 6 Perioden identisch.
        params["month_index"] = month_index

        return params

    def _ensure_data_lod_manager(self):
        """Lazy-Fallback für Standalone-Nutzung (Tests, calculate_weather_system()
        ohne injizierten Manager) - die echte Pipeline injiziert immer einen über
        GenerationOrchestrator.get_generator_instance()."""
        if self.data_lod_manager is None:
            from managers.data_lod_manager import DataLODManager
            self.data_lod_manager = DataLODManager()
        return self.data_lod_manager

    def calculate_weather_system(self, heightmap_combined: np.ndarray, shadowmap: np.ndarray,
                                parameters: Dict[str, Any], lod_level: int) -> WeatherData:
        """
        Funktionsweise: Standalone-Convenience-Entry-Point (Legacy-Kompatibilität + Tests)
        Aufgabe: Führt alle 4 Weather-Calculator-Knoten synchron für EIN LOD aus und
            liefert das fertige WeatherData-Objekt. Die echte GUI-Pipeline
            (GenerationOrchestrator) ruft dieselben _calc_*-Methoden ab jetzt einzeln
            über den globalen CalculatorDispatcher auf (Tracker #16 LOD-Lockstep-
            Umbau) - der Effekt ist identisch, da beide Wege denselben Storage nutzen.
            Weather hat - anders als Geology/Water - keinen Cross-LOD-
            Akkumulationszustand, jeder Aufruf ist eigenständig.

        Args:
            heightmap_combined: Heightmap nach Geology UND Erosion. Die
                Erosion kam bis 2026-07-28 aus dem Water-Generator und lag
                damit HINTER Weather - das Wetter rechnete also auf dem
                unerodierten Gelaende. Seit sie ein eigener Generator vor
                Weather ist, stimmt der Name auch.
            shadowmap: Shadow-Map vom Terrain-Generator für Sonneneinstrahlung
            parameters: Alle Weather-Parameter aus ParameterManager
            lod_level: Numerisches LOD-Level (1-6+) für Progressive Enhancement

        Returns:
            WeatherData: Vollständiges Weather-System mit allen Outputs

        Raises:
            ValueError: Bei ungültigen Input-Dependencies oder Parameter-Ranges
            RuntimeError: Bei kritischen CFD-Solver-Failures
        """
        try:
            self.logger.debug(f"Starting weather generation - LOD {lod_level}, size: {heightmap_combined.shape}")

            self._validate_inputs(heightmap_combined, shadowmap, parameters, lod_level)
            self._ensure_data_lod_manager()
            self.set_active_parameters(parameters)

            # Standalone-Convenience-Pfad: heightmap_combined/shadowmap kommen hier
            # als direkte Parameter, nicht aus dem DataLODManager - für die
            # _calc_*-Methoden (die jetzt IMMER aus dem feingranularen Calculator-
            # Storage lesen, siehe get_calculator_combined_heightmap()) dort
            # gespiegelt, analog zu Geology/Water.
            self._speichern(
                "terrain.redistribution", lod_level, {"heightmap": heightmap_combined})
            self._speichern(
                "terrain.shadow", lod_level, {"shadowmap": shadowmap})

            for calculator_id in (
                "weather.temperature", "weather.wind", "weather.humidity", "weather.precipitation",
            ):
                getattr(self, "_calc_" + calculator_id.split(".", 1)[1])(calculator_id, lod_level)

            weather_data = self.assemble_weather_data(lod_level, parameters)

            self.logger.info(f"Weather generation completed successfully - LOD {lod_level}")

            return weather_data

        except Exception as e:
            self.logger.error(f"Weather generation failed: {str(e)}")
            # Error-Recovery: Fallback zu Simplified-Weather-System
            return self._create_fallback_weather_data(heightmap_combined.shape[0], lod_level, parameters)

    def assemble_weather_data(self, lod_level: int, parameters: Dict[str, Any]) -> WeatherData:
        """
        Funktionsweise: Baut das finale WeatherData-Objekt aus den einzeln
        gespeicherten Calculator-Outputs zusammen
        Aufgabe: Wird vom GenerationOrchestrator aufgerufen, sobald alle 4 Weather-
            Calculator-Knoten ein LOD abgeschlossen haben (siehe Task 18 im
            LOD-Lockstep-Umbau)
        """
        temp_map = self.data_lod_manager.get_calculator_output("weather.temperature", "temp_map", lod_level)
        wind_map = self.data_lod_manager.get_calculator_output("weather.wind", "wind_map", lod_level)
        humid_map = self.data_lod_manager.get_calculator_output("weather.humidity", "humid_map", lod_level)
        precip_map = self.data_lod_manager.get_calculator_output("weather.precipitation", "precip_map", lod_level)

        if temp_map is None or wind_map is None or humid_map is None or precip_map is None:
            raise ValueError(f"assemble_weather_data: fehlende Calculator-Outputs für LOD {lod_level}")

        # Saisonale Monats-Listen (für die animierte Weather-Tab-Anzeige) -
        # optional, da nur relevant, wenn die _calc_*-Methoden tatsächlich
        # über die Monats-Schleife gelaufen sind (z.B. immer der Fall bei der
        # echten Pipeline, aber None bei alten Cache-Einträgen).
        temp_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.temperature", "temp_map_monthly", lod_level)
        wind_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.wind", "wind_map_monthly", lod_level)
        humid_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.humidity", "humid_map_monthly", lod_level)
        precip_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.precipitation", "precip_map_monthly", lod_level)

        # 3-Schicht-Atmosphäre (siehe AtmosphereLayers/_run_coupled_atmosphere_
        # simulation, [[project-3layer-wind-cfd]]) - optional, None bei altem
        # Cache oder falls der gekoppelte Loop diese Runde in den Einzelschicht-
        # Fallback gefallen ist (siehe _calc_temperature-Docstring).
        temp_map_layers = self.data_lod_manager.get_calculator_output(
            "weather.temperature", "temp_map_layers", lod_level)
        wind_map_layers = self.data_lod_manager.get_calculator_output(
            "weather.wind", "wind_map_layers", lod_level)
        humid_map_layers = self.data_lod_manager.get_calculator_output(
            "weather.humidity", "humid_map_layers", lod_level)
        temp_map_layers_monthly = self.data_lod_manager.get_calculator_output(
            "weather.temperature", "temp_map_layers_monthly", lod_level)
        wind_map_layers_monthly = self.data_lod_manager.get_calculator_output(
            "weather.wind", "wind_map_layers_monthly", lod_level)
        humid_map_layers_monthly = self.data_lod_manager.get_calculator_output(
            "weather.humidity", "humid_map_layers_monthly", lod_level)

        target_size = temp_map.shape[0]
        weather_data = self._create_weather_data(
            wind_map, temp_map, precip_map, humid_map, lod_level, target_size, parameters,
            wind_map_monthly=wind_map_monthly, temp_map_monthly=temp_map_monthly,
            precip_map_monthly=precip_map_monthly, humid_map_monthly=humid_map_monthly,
            temp_map_layers=temp_map_layers, wind_map_layers=wind_map_layers,
            humid_map_layers=humid_map_layers, temp_map_layers_monthly=temp_map_layers_monthly,
            wind_map_layers_monthly=wind_map_layers_monthly,
            humid_map_layers_monthly=humid_map_layers_monthly)

        cfd_iterations = self._get_cfd_iterations(lod_level)
        self._update_performance_stats(weather_data, cfd_iterations)

        return weather_data

    def _get_prepared_terrain_inputs(self, lod_level: int):
        """
        Holt heightmap_combined/shadowmap für dieses LOD und bringt sie auf die
        Weather-eigene Ziel-Auflösung (siehe _get_lod_size()) - Ersatz für das
        frühere context-basierte Vorab-Interpolieren, das nur innerhalb EINES
        calculate_weather_system()-Aufrufs existierte. Jede _calc_*-Methode ruft
        das unabhängig auf (billige, reine Operation - kein gemeinsamer Zustand
        zwischen separaten Dispatch-Aufrufen nötig).
        """
        heightmap_combined = self.data_lod_manager.get_calculator_combined_heightmap(lod_level)
        shadowmap = self.data_lod_manager.get_calculator_output("terrain.shadow", "shadowmap", lod_level)
        if heightmap_combined is None or shadowmap is None:
            raise ValueError(f"Weather: heightmap_combined/shadowmap für LOD {lod_level} nicht verfügbar")

        voll_size = self._get_lod_size(lod_level, heightmap_combined.shape[0])

        # DAS WETTER RECHNET AUF FESTEM GITTER (siehe WETTER_GITTER).
        #
        # Gemessen am 2026-08-06: weather.temperature brauchte 171 von 199
        # Sekunden der gesamten Pipeline bei 512 px, und es waechst DOPPELT -
        # mehr Pixel UND mehr Zeitschritte (_get_atmosphere_loop_steps: 25 bei
        # 256 px, 35 bei 512, 50 bei 1024). Bei der Vorgabe von 1024 px waeren
        # das rund 16 Minuten allein fuer diesen Knoten.
        #
        # Ein Wind- oder Temperaturfeld ueber 21 km hat aber keine Struktur auf
        # 20 m. Die Glaettungslaenge der Atmosphaere liegt Groessenordnungen
        # darueber; die feine Aufloesung rechnet Rauschen, kein Wetter.
        #
        # Die Ausgaben werden vor dem Ablegen wieder auf die volle Kartengroesse
        # gebracht (_speichern) - nach aussen aendert sich am Vertrag nichts,
        # jeder Knoten liefert weiterhin Felder in Kartengroesse.
        target_size = min(voll_size, WETTER_GITTER)
        self._wetter_rechen_size = target_size
        self._wetter_voll_size = voll_size

        heightmap, shadowmap = self._prepare_input_data(heightmap_combined, shadowmap, target_size)
        return heightmap, shadowmap, target_size

    # ------------------------------------------------------------------
    def _auf_kartengroesse(self, wert, rechen_size, voll_size):
        """
        Ein Ergebnis vom Rechengitter auf die Kartengroesse bringen.

        Arbeitet auf beliebig verschachtelten Ausgaben: einzelne Karten,
        Monatslisten, Schichtstapel (L,H,W) und Vektorfelder (H,W,2). Gesucht
        werden die beiden Achsen, die `rechen_size` gross sind; alles andere
        bleibt unangetastet.

        Warum nicht scipy.ndimage.zoom mit Faktoren: dessen Ergebnisgroesse kann
        um ein Pixel danebenliegen, und eine Karte, die 511 statt 512 breit ist,
        faellt erst weit spaeter auf - bei einer Formpruefung in einem ganz
        anderen Knoten.
        """
        if rechen_size >= voll_size:
            return wert
        if isinstance(wert, list):
            return [self._auf_kartengroesse(w, rechen_size, voll_size) for w in wert]
        if not isinstance(wert, np.ndarray) or wert.ndim < 2:
            return wert

        achsen = [i for i, d in enumerate(wert.shape) if d == rechen_size]
        # Genau zwei benachbarte Achsen sind das Ortsgitter. Trifft das nicht
        # zu, wird nichts angefasst - lieber unveraendert als falsch verzerrt.
        if len(achsen) != 2 or achsen[1] != achsen[0] + 1:
            return wert
        y = achsen[0]

        quelle = np.moveaxis(wert, (y, y + 1), (-2, -1))
        rest = quelle.shape[:-2]
        ziel = np.empty(rest + (voll_size, voll_size), dtype=np.float32)
        for index in np.ndindex(*rest) if rest else [()]:
            ziel[index] = self._interpolate_2d_bicubic(
                np.ascontiguousarray(quelle[index], dtype=np.float32), voll_size)
        return np.moveaxis(ziel, (-2, -1), (y, y + 1)).astype(wert.dtype, copy=False)

    def _speichern(self, calculator_id: str, lod_level: int, outputs: dict) -> None:
        """
        Ausgaben ablegen - hochskaliert auf die Kartengroesse.

        EINE Stelle statt neun: die Wetterknoten legen ihre Ergebnisse an neun
        verschiedenen Punkten ab, und jede einzeln zu skalieren waere neunmal
        dieselbe Gelegenheit, eine zu vergessen.
        """
        rechen = getattr(self, "_wetter_rechen_size", None)
        voll = getattr(self, "_wetter_voll_size", None)
        if rechen and voll and voll > rechen:
            outputs = {k: self._auf_kartengroesse(v, rechen, voll)
                       for k, v in outputs.items()}
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, outputs)

    def _calc_temperature(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'weather.temperature' (#11) - Wirtsknoten für die
        gekoppelte 3-Schicht-Atmosphären-Simulation (siehe
        _run_coupled_atmosphere_simulation, [[project-3layer-wind-cfd]]).

        Läuft intern über 6 saisonale Zwei-Monats-Perioden (Jan/Feb..Nov/Dez,
        siehe _generate_seasonal_parameters()). Jeder Monat bekommt seine
        eigene, astronomisch berechnete Shadowmap (generate_seasonal_sun_angles()
        + self.shadow_calculator, NICHT der geteilte terrain.shadow-Output -
        Winter bekommt einen flacheren, Sommer einen steileren Sonnenstand).

        WICHTIG (DAG-Design-Entscheidung): dieser Knoten hat im CALCULATOR_GRAPH
        keine depends_on-Kante zu weather.wind/weather.humidity - ist also
        topologisch GARANTIERT der erste der 3, der pro Runde fertig wird.
        Deshalb läuft der GESAMTE gekoppelte Temp+Wind+Feuchte+Niederschlag-Loop
        HIER statt in _calc_wind: würde der Loop stattdessen in _calc_wind
        laufen, könnte biome.super_override (hängt NUR von weather.temperature
        ab, nicht von weather.wind) einen nur-"geseedeten" temp_map-Wert lesen,
        bevor der Loop fertig ist - ein Race-Window, das mit dem Loop hier
        (wo kein Zwischenzustand existiert, der fälschlich als fertig gilt)
        gar nicht erst entsteht. Ergebnisse werden explizit unter ALLEN 4
        Calculator-IDs gespeichert (set_calculator_output nimmt calculator_id
        als reinen String ohne Validierung gegen den ausführenden Knoten
        entgegen) - _calc_wind/_calc_humidity/_calc_precipitation werden
        dadurch zu dünnen Pass-Throughs (siehe deren Docstrings).

        Die GROUND-Schicht füllt weiterhin temp_map/wind_map/humid_map/
        precip_map (+ _monthly) - exakt wie vor dem 3-Schicht-Umbau, für
        Rückwärtskompatibilität mit 2D/3D-Anzeige, water.evaporation und allen
        Biome-Knoten. Die vollen 3 Schichten landen zusätzlich, rein additiv,
        unter *_layers/*_layers_monthly (siehe WeatherData).

        Fallback: schlägt der gekoppelte Loop fehl (Exception - z.B. bei
        pathologischen Eingaben), fällt dieser Knoten auf die alte
        Einzelschicht-Temperatur-Logik zurück und schreibt NUR
        temp_map/temp_map_monthly/shadowmap_monthly - wind/humidity/
        precipitation bleiben dann unter ihren jeweiligen calculator_ids leer,
        wodurch die Pass-Through-Knoten IHREN eigenen alten Einzelschicht-
        Fallback auslösen (etabliertes 3-stufiges Fallback-Muster dieser Datei).
        """
        self._update_progress("Temperature", 20, "Calculating coupled 3-layer atmosphere...")
        # Teilschritt-Messung, gleiche Bauart wie terrain.redistribution
        # (managers/teilschritte.py). 13.5 s auf einer Zeile im Pipeline-Log
        # sagten nicht, ob der Schattenwurf, die Atmosphaerenschleife oder
        # das Vorbereiten der Eingaben die Zeit frisst.
        from managers.teilschritte import Teilschritte, schritt as _s
        _ts = Teilschritte("weather.temperature",
                           fortschritt=self._update_progress, von=20, bis=95,
                           plan=[("terrain_inputs", 3.0), ("rauheit_solar", 5.0),
                                 ("schattenwurf", 30.0), ("monatsparameter", 2.0),
                                 ("atmosphaere_schleife", 55.0),
                                 ("nachbereitung", 5.0)])
        with _s(_ts, "terrain_inputs", "Gelaende-Eingaben"):
            heightmap, shadowmap, target_size = self._get_prepared_terrain_inputs(lod_level)
        atmosphere_steps = self._get_atmosphere_loop_steps(lod_level)
        # Statisch über alle 6 Monate dieser Runde - einmal holen statt in
        # _run_coupled_atmosphere_simulation sechsmal neu abzufragen (siehe
        # [[project-wind-roughness]]).
        _rs = _s(_ts, "rauheit_solar", "Rauheit und Einstrahlung")
        _rs.__enter__()
        roughness_damping = self._get_roughness_damping(heightmap.shape, lod_level)
        # Statisch über alle 6 Monate dieser Runde, gleiches Muster wie
        # roughness_damping direkt darüber (siehe _get_solar_absorption_factor()).
        solar_absorption_factor = self._get_solar_absorption_factor(heightmap.shape, lod_level)

        from gui.config.value_default import WEATHER
        from core.terrain_generator import generate_seasonal_sun_angles
        latitude = self._current_parameters.get('map_latitude', WEATHER.MAP_LATITUDE["default"])
        longitude = self._current_parameters.get('map_longitude', WEATHER.MAP_LONGITUDE["default"])

        # EIN SONNENSATZ STATT SECHS (2026-08-07).
        #
        # Nutzer: "Sommer und Winter sind noch vorhanden, aber die
        # sonnenstaende sind die gleichen." Die Jahreszeit steckt seit dem
        # festgelegten Temperaturmodell in der JAHRESKURVE, nicht mehr im
        # Sonnenstand - der Schattenwurf muss deshalb nur noch EINMAL laufen
        # statt sechsmal.
        #
        # Genommen wird der FRUEHLINGSSATZ (Periode 1, also Maerz/April): er
        # liegt zwischen den Extremen und ist damit der neutrale Fall. Ein
        # Sommersatz haette die Nordhaenge dauerhaft zu hell gemacht, ein
        # Wintersatz zu dunkel.
        #
        # Das spart Faktor 6 auf dem teuersten Posten der Wetterrechnung. Die
        # Monatsparameter bleiben sechsfach - sie sind billig und Wind und
        # Feuchte lesen sie noch.
        _rs.__exit__(None, None, None)
        FRUEHLING = 1
        sun_angles = generate_seasonal_sun_angles(FRUEHLING, latitude, longitude)
        # Pixelgroesse setzen - ohne sie haelt der Schattenwurf jeden Hang fuer
        # eine Wand (2026-08-07). heightmap ist hier bereits auf das
        # Wettergitter gebracht, die Kartenbreite bleibt dieselbe.
        self.shadow_calculator.set_meters_per_pixel(
            float(self.data_lod_manager.get_map_distance_km()) * 1000.0
            / heightmap.shape[0])
        with _s(_ts, "schattenwurf", "Schattenwurf"):
            gemeinsame_shadowmap = self.shadow_calculator.calculate_shadows(
                heightmap, lod_level, sun_angles_override=sun_angles)
        # Dieselbe LOD-gefilterte Kanal-Teilmenge, die auch die Shadowmap
        # erzeugt hat (siehe _weighted_solar_exposure()-Docstring - die
        # Kanalzahl muss exakt übereinstimmen, sonst Shape-Mismatch-Fallback).
        gemeinsame_winkel, _ = self.shadow_calculator.get_sun_angles_for_lod(
            lod_level, sun_angles_override=sun_angles)

        _mp = _s(_ts, "monatsparameter", "Monatsparameter")
        _mp.__enter__()
        monthly_shadowmaps = []
        monthly_sun_angles = []
        month_params_list = []
        for month_index in range(TICKS_JE_JAHR):
            month_params = self._generate_seasonal_parameters(self._current_parameters, month_index)
            monthly_shadowmaps.append(gemeinsame_shadowmap)
            monthly_sun_angles.append(gemeinsame_winkel)
            month_params_list.append(month_params)

        # LOD-Vererbung (Weather-Rework Punkt F) - Endzustand derselben 6
        # saisonalen Perioden der VORHERIGEN, gröberen LOD-Stufe, falls
        # vorhanden (allererster Durchlauf: None, reproduziert das alte,
        # rein noise-basierte Anfangsverhalten unverändert). Alle drei
        # Felder werden gemeinsam benötigt - fehlt eines (z.B. alter Cache-
        # Eintrag ohne *_layers_monthly), wird komplett auf Noise-Seeding
        # zurückgefallen statt mit einem unvollständigen Zustand zu starten.
        # VORIGER DURCHGANG, nicht voriges LOD (2026-07-28, siehe
        # FEEDBACK_PASSES in managers/calculator_graph.py). Die
        # Pipeline rechnet nur noch EINE Aufloesungsstufe; der Kreis
        # Weather->Water->Biome->Weather wird stattdessen mehrfach
        # durchlaufen und ueberschreibt dabei denselben Speicherplatz.
        # Wer hier liest, BEVOR er selbst schreibt, bekommt damit genau
        # den Stand des vorigen Durchgangs - im ersten Durchgang None,
        # und dann greift derselbe Platzhalter-Zweig wie frueher bei
        # LOD 1.
        # Nur wenn es einen vorigen Durchgang GIBT. Ohne diese Bedingung waere
        # der gelesene Wert bei einer zweiten Generierung der des vorigen
        # LAUFS - das Ergebnis haenge dann daran, wie oft man schon generiert
        # hat, nicht an den Parametern.
        vorheriger_durchgang = self.data_lod_manager.get_feedback_pass() > 1
        prev_temp_layers_monthly = self.data_lod_manager.get_calculator_output(
            "weather.temperature", "temp_map_layers_monthly", lod_level)             if vorheriger_durchgang else None
        prev_wind_layers_monthly = self.data_lod_manager.get_calculator_output(
            "weather.wind", "wind_map_layers_monthly", lod_level)             if vorheriger_durchgang else None
        prev_humid_layers_monthly = self.data_lod_manager.get_calculator_output(
            "weather.humidity", "humid_map_layers_monthly", lod_level)             if vorheriger_durchgang else None
        has_lod_inheritance = (prev_temp_layers_monthly is not None and prev_wind_layers_monthly is not None
                               and prev_humid_layers_monthly is not None)

        # Bodenfeuchte-/Wasserflächen-Kopplung (Weather-Rework Punkt G) - reale
        # water.soil_moisture-Karte der VORHERIGEN LOD-Stufe (analog zum
        # Erosion-Vorstufen-Muster in water_generator.py, kein Zyklus: strikt
        # eine bereits abgeschlossene, frühere Runde). Nicht monats-abhängig
        # (Water rechnet keine 6 saisonalen Bodenfeuchte-Karten), daher einmal
        # geholt und für alle 6 Monate wiederverwendet. Fehlt sie (allererster
        # LOD-Durchlauf, Water noch nie gelaufen), bleibt soil_moisture_field
        # None - _run_coupled_atmosphere_simulation fällt dann exakt auf den
        # alten pauschalen 50%-Platzhalter zurück.
        prev_soil_moist_map = self.data_lod_manager.get_calculator_output(
            "water.soil_moisture", "soil_moist_map", lod_level)             if vorheriger_durchgang else None
        soil_moisture_field = (self._interpolate_2d_bicubic(prev_soil_moist_map, target_size)
                                if prev_soil_moist_map is not None else None)
        if soil_moisture_field is None:
            # Kein Wert aus einem vorigen Rueckkopplungs-Durchgang - dann die
            # Wasserhaltefaehigkeit des Vorab-Bioms statt des pauschalen
            # 50-%-Platzhalters (siehe _soil_capacity_from_preseed und die
            # Graph-Kante weather.temperature <- biome.preseed_hint).
            soil_moisture_field = self._soil_capacity_from_preseed(target_size, lod_level)

        try:
            monthly_temp_maps, monthly_wind_maps = [], []
            monthly_humid_maps, monthly_precip_maps = [], []
            monthly_temp_layers, monthly_wind_layers, monthly_humid_layers = [], [], []

            _mp.__exit__(None, None, None)
            _as = _s(_ts, "atmosphaere_schleife", "Atmosphaere, 6 Monate")
            _as.__enter__()
            for month_index in range(TICKS_JE_JAHR):
                initial_state = None
                if has_lod_inheritance:
                    initial_state = {
                        'temp_layers': prev_temp_layers_monthly[month_index],
                        'wind_layers': prev_wind_layers_monthly[month_index],
                        'humid_layers': prev_humid_layers_monthly[month_index],
                    }
                result = self._run_coupled_atmosphere_simulation(
                    heightmap, monthly_shadowmaps[month_index], month_params_list[month_index],
                    target_size, n_steps=atmosphere_steps, roughness_damping=roughness_damping,
                    initial_state=initial_state, soil_moisture_field=soil_moisture_field,
                    sun_angles=monthly_sun_angles[month_index],
                    solar_absorption_factor=solar_absorption_factor)

                monthly_temp_layers.append(result['temp_layers'])
                monthly_wind_layers.append(result['wind_layers'])
                monthly_humid_layers.append(result['humid_layers'])

                # DIE TEMPERATUR IST SEIT 2026-08-07 EINE FESTLEGUNG.
                #
                # Sie faellt nicht mehr aus der Atmosphaerensimulation heraus,
                # sondern folgt der Klimatabelle je Region (klima_map), der
                # Hoehe und der Sonnenexposition. Der Grund ist nicht die
                # Geschwindigkeit, sondern der Drift: die Simulation lief ueber
                # 25 bis 50 Zeitschritte, und die Schrittzahl haengt an der
                # Aufloesung - dieselbe Welt lieferte bei 512 px eine andere
                # Temperatur als bei 256.
                #
                # Die Simulation laeuft weiter, denn Wind und Feuchte kommen
                # noch aus ihr. Nur ihr Temperaturergebnis wird verworfen.
                # ZEITPUNKT m/6, NICHT (m+0.5)/6.
                #
                # Mit der Periodenmitte liegen alle sechs Stichproben ZWISCHEN
                # den Extremen, und der Jahresgang erreicht nur +/-0.866 statt
                # +/-1 - gemessen kam die Morobora auf 21.7 K Spanne statt der
                # eingetragenen 29.0. Mit m/6 faellt Periode 0 auf den Januar
                # und Periode 3 auf den Juli, also genau auf die beiden Werte,
                # aus denen die Tabelle gebildet ist.
                fest = self.temperaturfeld_festgelegt(
                    heightmap, monthly_shadowmaps[month_index], lod_level,
                    zeit_im_jahr=month_index / TICKS_JE_JAHR)
                monthly_temp_maps.append(
                    fest if fest is not None
                    else result['temp_layers'][AtmosphereLayers.GROUND])
                monthly_wind_maps.append(result['wind_layers'][AtmosphereLayers.GROUND])
                monthly_humid_maps.append(result['humid_layers'][AtmosphereLayers.GROUND])
                # PRECIP_ANNUAL_SCALE_FACTOR hier (nicht in
                # _run_coupled_atmosphere_simulation) angewendet - siehe
                # dortiger Docstring-Kommentar.
                # DER NIEDERSCHLAG IST SEIT 2026-08-07 EINE FESTLEGUNG.
                #
                # Regionsgrundwert mal Luv-Faktor mal Regenschatten - alles
                # geschlossene Form, siehe niederschlagsfeld_festgelegt(). Die
                # Simulation laeuft weiter fuer Wind und Feuchte; nur ihr
                # Niederschlagsergebnis wird verworfen.
                #
                # Die sechs Perioden bekommen VORERST denselben Wert: die
                # Klimatabelle fuehrt nur eine Jahressumme. Ein Jahresgang des
                # Niederschlags (Macchia trocken im Sommer) waere der
                # naechste Schritt und braucht eine zweite Spalte in der
                # Tabelle.
                fest_p = self.niederschlagsfeld_festgelegt(
                    heightmap, lod_level, zeit_im_jahr=month_index / TICKS_JE_JAHR)
                monthly_precip_maps.append(
                    fest_p if fest_p is not None
                    else result['precip_map'] * PRECIP_ANNUAL_SCALE_FACTOR)

            # Mehrgenerationen-Puffer für Feuchte (siehe frühere _calc_humidity-
            # Fassung, Verhalten hier 1:1 erhalten, nur an den neuen Aufrufort
            # verschoben): gewichteter Mittelwert statt additiver Akkumulation,
            # PRO Monatsindex, dämpft "kippt bei jedem Lauf komplett in trocken
            # oder nass"-Verhalten.
            previous_humid_monthly = self.data_lod_manager.get_calculator_output(
                "weather.humidity", "humid_map_monthly", lod_level)                 if self.data_lod_manager.get_feedback_pass() > 1 else None
            if previous_humid_monthly is not None:
                for m in range(6):
                    prev_m = previous_humid_monthly[m]
                    if prev_m.shape[0] != monthly_humid_maps[m].shape[0]:
                        prev_m = self._interpolate_2d_bicubic(prev_m, monthly_humid_maps[m].shape[0])
                    monthly_humid_maps[m] = 0.4 * prev_m + 0.6 * monthly_humid_maps[m]

            _as.__exit__(None, None, None)
            _nb = _s(_ts, "nachbereitung", "Mittelung und Wind-Nachlauf")
            _nb.__enter__()
            temp_map = np.mean(np.stack(monthly_temp_maps, axis=0), axis=0).astype(np.float32)
            wind_map = np.mean(np.stack(monthly_wind_maps, axis=0), axis=0).astype(np.float32)
            humid_map = np.mean(np.stack(monthly_humid_maps, axis=0), axis=0).astype(np.float32)
            precip_map = np.mean(np.stack(monthly_precip_maps, axis=0), axis=0).astype(np.float32)

            # WIND IST SEIT 2026-08-11 REGIONAL KALIBRIERT (docs/archiv/2026-07-29_SPEZIFIKATION.md
            # §3.5), zwei multiplikative Faktorfelder statt einer eigenen
            # Formel - die Simulation liefert weiterhin Boeen an Graten und
            # Kanalisierung in Taelern:
            #   1. Luv/Lee am Gebirge (`_wind_luv_lee_faktor`, Mittelwert ~1.0)
            #   2. Regionsmittel auf `wind_ziel_map` ziehen
            #      (`_wind_regional_faktor`) - korrigiert dabei automatisch
            #      jede Restverschiebung aus Schritt 1, das REGIONALE NIVEAU
            #      bleibt also unabhaengig vom Luv/Lee-Term garantiert richtig.
            # EIN Faktorfeld je Term aus dem Jahresmittel, auf alle sechs
            # Monate UND beide Schichten-Layer angewandt - sonst wuerde eine
            # monatsweise Normierung die saisonale Staerke-Schwankung aus
            # WEATHER.CLIMATE_ZONE_SEASONAL_OFFSETS wieder einebnen.
            richtung = float(self._current_parameters.get("prevailing_wind_direction", 0.0))
            luv_lee_faktor = self._wind_luv_lee_faktor(heightmap, richtung)
            wind_map = wind_map.copy()
            wind_map[..., 0] *= luv_lee_faktor
            wind_map[..., 1] *= luv_lee_faktor
            for m in range(TICKS_JE_JAHR):
                monthly_wind_maps[m] = monthly_wind_maps[m].copy()
                monthly_wind_maps[m][..., 0] *= luv_lee_faktor
                monthly_wind_maps[m][..., 1] *= luv_lee_faktor

            wind_faktor = self._wind_regional_faktor(wind_map, lod_level)
            if wind_faktor is not None:
                wind_map = wind_map.copy()
                wind_map[..., 0] *= wind_faktor
                wind_map[..., 1] *= wind_faktor
                for m in range(TICKS_JE_JAHR):
                    monthly_wind_maps[m] = monthly_wind_maps[m].copy()
                    monthly_wind_maps[m][..., 0] *= wind_faktor
                    monthly_wind_maps[m][..., 1] *= wind_faktor

            temp_map_layers = np.mean(np.stack(monthly_temp_layers, axis=0), axis=0).astype(np.float32)
            wind_map_layers = np.mean(np.stack(monthly_wind_layers, axis=0), axis=0).astype(np.float32)
            humid_map_layers = np.mean(np.stack(monthly_humid_layers, axis=0), axis=0).astype(np.float32)
            wind_map_layers = wind_map_layers.copy()
            wind_map_layers[..., 0] *= luv_lee_faktor
            wind_map_layers[..., 1] *= luv_lee_faktor
            if wind_faktor is not None:
                wind_map_layers[..., 0] *= wind_faktor
                wind_map_layers[..., 1] *= wind_faktor

            self._speichern(
                "weather.temperature", lod_level,
                {"temp_map": temp_map, "temp_map_monthly": monthly_temp_maps,
                 "shadowmap_monthly": monthly_shadowmaps,
                 "temp_map_layers": temp_map_layers, "temp_map_layers_monthly": monthly_temp_layers})
            self._speichern(
                "weather.wind", lod_level,
                {"wind_map": wind_map, "wind_map_monthly": monthly_wind_maps,
                 "wind_map_layers": wind_map_layers, "wind_map_layers_monthly": monthly_wind_layers})
            self._speichern(
                "weather.humidity", lod_level,
                {"humid_map": humid_map, "humid_map_monthly": monthly_humid_maps,
                 "humid_map_layers": humid_map_layers, "humid_map_layers_monthly": monthly_humid_layers})
            self._speichern(
                "weather.precipitation", lod_level,
                {"precip_map": precip_map, "precip_map_monthly": monthly_precip_maps})
            _nb.__exit__(None, None, None)
            _ts.bericht()

        except Exception as e:
            self.logger.warning(
                f"weather.temperature: gekoppelte 3-Schicht-Simulation fehlgeschlagen ({e}) - "
                f"Fallback auf Einzelschicht-Temperatur, Wind/Feuchte/Niederschlag "
                f"nutzen ihren eigenen Einzelschicht-Fallback")
            monthly_temp_maps = [
                self._calculate_temperature_field(
                    heightmap, monthly_shadowmaps[m], month_params_list[m], target_size,
                    sun_angles=monthly_sun_angles[m])
                for m in range(6)
            ]
            temp_map = np.mean(np.stack(monthly_temp_maps, axis=0), axis=0).astype(np.float32)
            self._speichern(
                "weather.temperature", lod_level,
                {"temp_map": temp_map, "temp_map_monthly": monthly_temp_maps,
                 "shadowmap_monthly": monthly_shadowmaps})

    def _calc_wind(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'weather.wind' (#12) - dünner Pass-Through. Der
        eigentliche Wind wird bereits vom gekoppelten 3-Schicht-Loop in
        _calc_temperature berechnet und unter dieser calculator_id gespeichert
        (siehe dortiger Docstring, [[project-3layer-wind-cfd]] für die DAG-
        Begründung). Ist der Wert schon da (Normalfall - weather.temperature
        hat keine Abhängigkeit zu diesem Knoten, läuft also garantiert
        zuerst), ist dieser Aufruf ein No-Op, der nur zügig zurückkehrt (damit
        der Orchestrator mark_completed('weather.wind') auslöst, worauf eigene
        Downstream-Konsumenten warten).

        Fallback: ist der Wert NICHT da (z.B. weil _calc_temperature's
        gekoppelter Loop fehlschlug), auf die alte Einzelschicht-CFD
        zurückfallen - etabliertes 3-stufiges Fallback-Muster dieser Datei.
        """
        existing = self.data_lod_manager.get_calculator_output(calculator_id, "wind_map", lod_level)
        if existing is not None:
            return

        self.logger.warning("weather.wind: kein Ergebnis vom gekoppelten Loop gefunden - Einzelschicht-Fallback")
        cfd_iterations = self._get_cfd_iterations(lod_level)
        self._update_progress("Wind Field", 30, f"Fallback CFD simulation with {cfd_iterations} iterations...")
        heightmap, shadowmap, target_size = self._get_prepared_terrain_inputs(lod_level)
        temp_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.temperature", "temp_map_monthly", lod_level)
        shadowmap_monthly = self.data_lod_manager.get_calculator_output(
            "weather.temperature", "shadowmap_monthly", lod_level)
        if temp_map_monthly is None:
            raise ValueError(f"weather.wind: temp_map_monthly für LOD {lod_level} nicht verfügbar")

        monthly_wind_maps = []
        for month_index in range(6):
            month_params = self._generate_seasonal_parameters(self._current_parameters, month_index)
            month_temp_map = temp_map_monthly[month_index]
            month_shadowmap = shadowmap_monthly[month_index] if shadowmap_monthly else shadowmap
            monthly_wind_maps.append(self._simulate_wind_field_cfd(
                heightmap, month_temp_map, month_shadowmap, month_params, target_size, cfd_iterations))

        wind_map = np.mean(np.stack(monthly_wind_maps, axis=0), axis=0).astype(np.float32)
        self._speichern(
            calculator_id, lod_level, {"wind_map": wind_map, "wind_map_monthly": monthly_wind_maps})

    def _calc_humidity(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'weather.humidity' (#13) - dünner Pass-Through, analog
        zu _calc_wind. Die eigentliche Feuchte (inkl. Mehrgenerationen-Puffer-
        Blending) wird bereits vom gekoppelten Loop in _calc_temperature
        berechnet und gespeichert.

        Fallback: alte Einzelschicht-Logik inkl. eigenem Mehrgenerationen-
        Puffer-Blend, falls _calc_temperature's Loop fehlschlug.
        """
        existing = self.data_lod_manager.get_calculator_output(calculator_id, "humid_map", lod_level)
        if existing is not None:
            return

        self.logger.warning("weather.humidity: kein Ergebnis vom gekoppelten Loop gefunden - Einzelschicht-Fallback")
        self._update_progress("Humidity", 60, "Fallback atmospheric moisture transport...")
        heightmap, shadowmap, target_size = self._get_prepared_terrain_inputs(lod_level)
        temp_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.temperature", "temp_map_monthly", lod_level)
        wind_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.wind", "wind_map_monthly", lod_level)
        if temp_map_monthly is None or wind_map_monthly is None:
            raise ValueError(f"weather.humidity: fehlende Inputs für LOD {lod_level}")

        hardness_map = self.data_lod_manager.get_calculator_output(
            "geology.hardness", "hardness_map", lod_level)
        if hardness_map is not None and hardness_map.shape[0] != heightmap.shape[0]:
            hardness_map = self._interpolate_2d_bicubic(hardness_map, heightmap.shape[0])

        # Eigener Stand des VORIGEN Durchgangs - gelesen bevor unten
        # geschrieben wird, deshalb nie der eigene aktuelle Wert.
        previous_monthly = self.data_lod_manager.get_calculator_output(
            calculator_id, "humid_map_monthly", lod_level)             if self.data_lod_manager.get_feedback_pass() > 1 else None

        monthly_humid_maps = []
        for month_index in range(6):
            month_params = self._generate_seasonal_parameters(self._current_parameters, month_index)
            humid_map_m = self._calculate_atmospheric_moisture(
                heightmap, temp_map_monthly[month_index], wind_map_monthly[month_index],
                month_params, hardness_map=hardness_map)

            if previous_monthly is not None:
                prev_m = previous_monthly[month_index]
                if prev_m.shape[0] != humid_map_m.shape[0]:
                    prev_m = self._interpolate_2d_bicubic(prev_m, humid_map_m.shape[0])
                humid_map_m = 0.4 * prev_m + 0.6 * humid_map_m

            monthly_humid_maps.append(humid_map_m)

        humid_map = np.mean(np.stack(monthly_humid_maps, axis=0), axis=0).astype(np.float32)
        self._speichern(
            calculator_id, lod_level, {"humid_map": humid_map, "humid_map_monthly": monthly_humid_maps})

    def _calc_precipitation(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'weather.precipitation' (#14) - dünner Pass-Through,
        analog zu _calc_wind/_calc_humidity. Der eigentliche Niederschlag
        (akkumulierte Kondensation aus dem gekoppelten Loop + orographischer
        Zusatzbeitrag) wird bereits von _calc_temperature berechnet und
        gespeichert.

        Fallback: alte Einzelschicht-Logik (Luv/Lee + Magnus-Kondensation auf
        dem finalen Einzelschicht-Snapshot), falls _calc_temperature's Loop
        fehlschlug.
        """
        existing = self.data_lod_manager.get_calculator_output(calculator_id, "precip_map", lod_level)
        if existing is not None:
            return

        self.logger.warning(
            "weather.precipitation: kein Ergebnis vom gekoppelten Loop gefunden - Einzelschicht-Fallback")
        self._update_progress("Precipitation", 80, "Fallback orographic precipitation...")
        heightmap, shadowmap, target_size = self._get_prepared_terrain_inputs(lod_level)
        temp_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.temperature", "temp_map_monthly", lod_level)
        wind_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.wind", "wind_map_monthly", lod_level)
        humid_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.humidity", "humid_map_monthly", lod_level)
        if temp_map_monthly is None or wind_map_monthly is None or humid_map_monthly is None:
            raise ValueError(f"weather.precipitation: fehlende Inputs für LOD {lod_level}")

        monthly_precip_maps = []
        for month_index in range(6):
            month_params = self._generate_seasonal_parameters(self._current_parameters, month_index)
            monthly_precip_maps.append(self._calculate_precipitation_system(
                humid_map_monthly[month_index], temp_map_monthly[month_index],
                wind_map_monthly[month_index], heightmap, month_params))

        # PRECIP_ANNUAL_SCALE_FACTOR auch im Einzelschicht-Fallback (siehe
        # dortiger Docstring-Kommentar) - sonst würde dieser seltener
        # erreichte Pfad precip_map in einer anderen Größenordnung als der
        # normale gekoppelte Pfad liefern.
        monthly_precip_maps = [p * PRECIP_ANNUAL_SCALE_FACTOR for p in monthly_precip_maps]
        precip_map = np.mean(np.stack(monthly_precip_maps, axis=0), axis=0).astype(np.float32)
        self._speichern(
            calculator_id, lod_level, {"precip_map": precip_map, "precip_map_monthly": monthly_precip_maps})

    # Rueckfallwert, wenn das Vorab-Biom (noch) nicht vorliegt - z.B. in
    # Standalone-Aufrufen und aelteren Tests, die weather direkt ansteuern.
    # Entspricht dem frueheren pauschalen Platzhalter, damit sich dort nichts
    # unbemerkt aendert.
    SOIL_CAPACITY_FALLBACK = 0.5

    def _soil_capacity_from_preseed(self, target_size, lod_level):
        """
        Wasserhaltefaehigkeit des Untergrunds aus dem Vorab-Biom, in Prozent.
        None, wenn kein Vorab-Biom vorliegt - dann bleibt es beim bisherigen
        pauschalen Platzhalter.

        Siehe die Begruendung an der Aufrufstelle und die Graph-Kante
        weather.temperature <- biome.preseed_hint.
        """
        if self.data_lod_manager is None or lod_level is None:
            return None

        preseed = self.data_lod_manager.get_calculator_output(
            "biome.preseed_hint", "preseed_biome_map", lod_level)
        if preseed is None:
            return None

        # Lokaler Import: water_generator zieht seinerseits gui.config-Module,
        # ein Import auf Modulebene wuerde die Abhaengigkeiten hier verbreitern.
        from core.water_generator import _BIOME_MOISTURE_CAPACITY

        indices = np.clip(np.asarray(preseed, dtype=np.int32),
                          0, len(_BIOME_MOISTURE_CAPACITY) - 1)
        # In PROZENT zurueckgeben - genau die Skala, die soil_moisture_field
        # ohnehin fuehrt (dort wird durch 100 geteilt).
        kapazitaet = _BIOME_MOISTURE_CAPACITY[indices].astype(np.float32)
        if kapazitaet.shape[0] != target_size:
            kapazitaet = self._interpolate_2d_bicubic(kapazitaet, target_size)
        return np.clip(kapazitaet, 0.0, 100.0)

    def _semi_lagrangian_advect(self, field: np.ndarray, u: np.ndarray, v: np.ndarray,
                                y_idx: np.ndarray, x_idx: np.ndarray, dt_scale: float = 0.1) -> np.ndarray:
        """
        Allgemeine Semi-Lagrange-Rückwärts-Advektion (siehe [[project-3layer-wind-cfd]],
        Referenz-Technik aus niels747/2D-Weather-Sandbox): jede Zelle sampelt ihren
        neuen Wert an position - geschwindigkeit*dt_scale, statt Vorwärts-Differenzen -
        numerisch stabil, kein CFL-Problem, gilt identisch für Geschwindigkeit,
        potentielle Temperatur und Feuchte. y_idx/x_idx sind vom Aufrufer
        vorberechnete np.mgrid-Koordinaten (wird pro Zeitschritt für alle 3
        Schichten/4 Felder wiederverwendet statt pro Aufruf neu gebaut).
        dt_scale folgt derselben Stabilitäts-Skalierung wie das bereits bestehende
        _transport_moisture_simple (dt*0.1).
        """
        return self._semi_lagrangian_advect_many(
            (field,), u, v, y_idx, x_idx, dt_scale)[0]

    def _semi_lagrangian_advect_many(self, fields, u: np.ndarray, v: np.ndarray,
                                     y_idx: np.ndarray, x_idx: np.ndarray,
                                     dt_scale: float = 0.1):
        """
        Advektiert MEHRERE Felder mit DEMSELBEN Geschwindigkeitsfeld in einem
        Rutsch.

        Der Grund ist gemessen: pro Schicht und Zeitschritt werden u, v, theta
        und q alle vier mit demselben u_old/v_old rueckwaerts gesampelt (siehe
        Aufrufstelle in _run_coupled_atmosphere_simulation). Die Quellkoordinaten
        und die bilinearen Gewichte sind damit viermal identisch - vorher wurden
        sie viermal neu gerechnet, weil jeder Aufruf fuer sich in scipys
        map_coordinates ging.

        Hier werden Ganzzahl-Indizes und Gewichte EINMAL bestimmt und dann auf
        alle Felder angewandt. Gemessen bei 256 px, vier Felder:

            vier Einzelaufrufe   8.46 ms
            Gewichte geteilt     5.87 ms      -31 %

        Das Ergebnis ist dasselbe: groesste Abweichung 4.8e-07 gegen
        map_coordinates(order=1, mode='nearest'), also reine float32-Rundung.
        Die Klemmung der Quellkoordinaten auf [0, n-1] entspricht dabei genau
        dem 'nearest'-Rand von scipy.

        Das Profil dahinter: die Advektion war mit 7.5 s von 18 s der groesste
        Einzelposten des Wetters (1800 Aufrufe = 6 Doppelmonate x 25 Schritte
        x 4 Felder x 3 Schichten).
        """
        height, width = fields[0].shape[:2]
        source_x = np.clip(x_idx - u * dt_scale, 0, width - 1)
        source_y = np.clip(y_idx - v * dt_scale, 0, height - 1)

        x0 = np.floor(source_x).astype(np.intp)
        y0 = np.floor(source_y).astype(np.intp)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)
        weight_x = (source_x - x0).astype(np.float32)
        weight_y = (source_y - y0).astype(np.float32)
        inv_x = 1.0 - weight_x

        results = []
        for field in fields:
            top = field[y0, x0] * inv_x + field[y0, x1] * weight_x
            bottom = field[y1, x0] * inv_x + field[y1, x1] * weight_x
            results.append((top * (1.0 - weight_y) + bottom * weight_y).astype(field.dtype))
        return results

    def _run_coupled_atmosphere_simulation(self, heightmap: np.ndarray, shadowmap: np.ndarray,
                                          month_params: Dict[str, Any], target_size: int,
                                          n_steps: int,
                                          roughness_damping: Optional[np.ndarray] = None,
                                          initial_state: Optional[Dict[str, np.ndarray]] = None,
                                          soil_moisture_field: Optional[np.ndarray] = None,
                                          sun_angles: Optional[list] = None,
                                          solar_absorption_factor: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Gekoppelte 3-Schicht-Atmosphären-Simulation (Boden/Mittel/Hoch, siehe
        AtmosphereLayers) - ersetzt das bisherige "Temp einmal -> Wind einmal ->
        Feuchte einmal"-Muster für EINEN Monat durch einen echten Pro-Zeitschritt-
        Loop, in dem Wind/Temperatur/Feuchte sich gegenseitig beeinflussen. Wird
        von _calc_temperature aufgerufen (siehe dortiger Docstring für die DAG-
        Begründung, warum der Loop dort statt in _calc_wind gehostet wird) -
        [[project-3layer-wind-cfd]].

        Pro Zeitschritt, pro Schicht:
          1. Semi-Lagrange-Advektion von u,v,theta (potentielle Temp),q
          2. Druckgradient (gemeinsames synoptisches Druckfeld je Monat, Terrain-
             Kopplung nur auf GROUND/gedämpft MID) + additive Terrain-Ablenkung
             UND multiplikativer Grat-/Canyon-Speedup (beide nur GROUND/gedämpft
             MID, siehe [[project-wind-ridge-speedup]]) + Rauigkeits-Dämpfung aus
             Biome (best-effort, siehe [[project-wind-roughness]]) + thermische
             Konvektion (Schatten-Term nur GROUND)
          3. Latentwärme: Kondensation aus Übersättigung (Magnus-Formel, identische
             Skala wie _calculate_precipitation_cpu/_calculate_atmospheric_moisture_cpu)
             wärmt, Verdunstung (nur GROUND) kühlt
        Danach, pro Zeitschritt EINMAL über alle Schichten:
          4. Vertikaler Austausch (Thermik-getrieben, symmetrische Relaxation
             zwischen Nachbarschichten - erhält die Schicht-Summe jeder Größe)
          5. Diffusion + 6-Richtungs-Kontinuitäts-Korrektur (horizontale Divergenz
             + vertikaler Fluss-Term aus Schritt 4)

        roughness_damping (optional, (H,W), Wertebereich [0,1]): best-effort
        Bodenrauigkeits-Dämpfung aus der Biome-Klassifikation, von
        _calc_temperature einmal pro LOD-Runde vorab per
        _get_roughness_damping() geholt (statisch über alle 6 Monate, daher
        hier als Parameter statt hier selbst erneut abgefragt). None (Default,
        z.B. wenn Biome in dieser Session nie angefragt wurde) reproduziert
        exakt das alte Verhalten ohne Rauigkeits-Term.

        initial_state (optional, Weather-Rework Punkt F "LOD-Vererbung"): dict
        mit 'temp_layers' (3,H_alt,W_alt, reale Temperatur), 'wind_layers'
        (3,H_alt,W_alt,2), 'humid_layers' (3,H_alt,W_alt) - das ENDERGEBNIS
        derselben saisonalen Periode der VORHERIGEN, gröberen LOD-Stufe (von
        _calc_temperature aus deren bereits gespeicherten *_layers_monthly-
        Outputs geholt). Wird bikubisch auf die aktuelle Auflösung hoch-
        skaliert und als CFD-Startbedingung verwendet statt der reinen
        Perlin-Noise-Randomisierung - eine neue, höhere LOD-Stufe beginnt
        damit nahe am bereits eingeschwungenen Zustand der Vorstufe, statt
        wieder bei einer frischen, unkorrelierten Zufalls-Verteilung. None
        (Default, z.B. beim allerersten LOD-Durchlauf) reproduziert exakt das
        alte, rein noise-basierte Anfangsverhalten.

        soil_moisture_field (optional, Weather-Rework Punkt G "Bodenfeuchte-/
        Wasserflächen-Kopplung über vorheriges LOD"): reale
        `water.soil_moisture`-Karte (0-100, siehe core/water_generator.py) der
        VORHERIGEN, gröberen LOD-Stufe, bereits vom Aufrufer bikubisch auf die
        aktuelle Rohauflösung hochskaliert (VOR dem Rand-Puffer dieser
        Methode - wird hier zusätzlich per np.pad(mode='edge') mitgepolstert,
        analog zu heightmap/shadowmap oben). Ersetzt den bisherigen fest
        verdrahteten Platzhalter (`soil_moisture=50/100=0.5`, weder Wüste noch
        See unterschieden sich) in der Verdunstungs-Berechnung. None (Default,
        z.B. allererster LOD-Durchlauf, bevor Water je gelaufen ist)
        reproduziert exakt den alten pauschalen 50%-Wert.

        Rückgabe: dict mit 'wind_layers' (3,H,W,2), 'temp_layers' (3,H,W, reale
        Temperatur), 'humid_layers' (3,H,W), 'precip_map' (H,W).
        """
        from gui.config.value_default import WEATHER  # TURBULENCE_STRENGTH-Default weiter unten

        # Rand-Puffer (Weather-Rework Punkt C, siehe _compute_edge_padding_px())
        # - Eingaben werden per Rand-Extension auf ein größeres Gitter gebracht,
        # die GESAMTE Simulation läuft transparent darauf (jede nachfolgende
        # Ableitung aus heightmap.shape betrifft automatisch das vergrößerte
        # Gitter), ein Sponge-Layer dämpft den Puffer-Bereich pro Zeitschritt
        # zurück zu seinem Ausgangszustand (siehe unten), am Ende wird auf die
        # ursprüngliche Größe zurückgeschnitten.
        edge_pad_px = self._compute_edge_padding_px(heightmap.shape[0])
        if edge_pad_px > 0:
            heightmap = np.pad(heightmap, edge_pad_px, mode='edge')
            if shadowmap.ndim == 3:
                shadowmap = np.pad(shadowmap, ((edge_pad_px, edge_pad_px), (edge_pad_px, edge_pad_px), (0, 0)),
                                    mode='edge')
            else:
                shadowmap = np.pad(shadowmap, edge_pad_px, mode='edge')
            if roughness_damping is not None:
                roughness_damping = np.pad(roughness_damping, edge_pad_px, mode='edge')
            if solar_absorption_factor is not None:
                solar_absorption_factor = np.pad(solar_absorption_factor, edge_pad_px, mode='edge')
            if soil_moisture_field is not None:
                soil_moisture_field = np.pad(soil_moisture_field, edge_pad_px, mode='edge')

        height, width = heightmap.shape

        # Weather-Rework Punkt G: reale Bodenfeuchte/Wasserflächen statt des
        # alten pauschalen 50%-Platzhalters - fehlt sie (allererster LOD-
        # Durchlauf), bleibt exakt der bisherige Wert erhalten.
        if soil_moisture_field is not None:
            soil_moisture_norm = np.clip(soil_moisture_field.astype(np.float32) / 100.0, 0.0, 1.0)
        else:
            # KEIN pauschaler 50-%-Wert mehr (2026-07-29), sondern die
            # Wasserhaltefaehigkeit des VORAB-BIOMS.
            #
            # Der Platzhalter war breitengrad-unabhaengig, speiste aber ueber
            # evap_rate0 den groesseren Teil der Atmosphaerenfeuchte: gemessen
            # trug die Klimatologie nur 32-42 % zur q-Impfung bei, der Rest
            # kam aus dieser Konstanten. Der subtropische Trockenguertel stand
            # damit zwar in der Feuchtetabelle (11.5 gegen 19.6 am Aequator,
            # -41 %), verwaesserte im Gesamtwert aber auf -26 % und kam im
            # Niederschlag nie an.
            #
            # Das Vorab-Biom kennt Breitengrad und Topografie und sagt ueber
            # _BIOME_MOISTURE_CAPACITY, wieviel Wasser der Untergrund halten
            # kann: Wueste 20, Fels/Badlands 15, Sumpf 95. Trockene Breiten
            # verdunsten dadurch weniger - Sand kann Feuchte nicht weit
            # transportieren. Kein Zyklus, siehe die Graph-Kante in
            # calculator_graph.py.
            soil_moisture_norm = np.full((height, width),
                                          self.SOIL_CAPACITY_FALLBACK, dtype=np.float32)
        L = AtmosphereLayers
        altitude_cooling_rate = month_params['altitude_cooling'] / 1000.0  # °C/m (Parameter ist °C/km)

        LATENT_HEAT_COEFFICIENT = _LATENT_HEAT_COEFFICIENT
        # Vertikaler Austausch: Bruchteil pro Zeitschritt, der zwischen
        # Nachbarschichten geblendet wird, proportional zur Thermik-Stärke -
        # geklemmt, damit ein einzelner Zeitschritt nie mehr als 30% einer
        # Schicht "leert" (Stabilität, analog zur Diffusions-Rate anderswo in
        # dieser Datei).
        MAX_VERTICAL_EXCHANGE_FRACTION = 0.3
        VERTICAL_EXCHANGE_COEFF = 0.02
        # Wie stark sich der druckgetriebene Wind-Term pro Zeitschritt Richtung
        # Druck-Gleichgewicht bewegt, statt (wie im alten Einzelschicht-Code) ihn
        # jede Iteration hart zu überschreiben - mit echter Advektion (Schritt 1)
        # ist u/v jetzt eine prognostische, transportierte Größe, ein hartes
        # Überschreiben würde die advehierte Struktur jedes Mal zerstören.
        PRESSURE_RELAX_RATE = 0.3
        # Weather-Rework Punkt A: thermisch gekoppeltes Druckfeld (mit
        # Rückweg) - wärmer als der Schicht-Durchschnitt = lokal niedrigerer
        # effektiver Druck (Auftriebs-Rückkopplung), addiert sich zum
        # bestehenden synoptischen Gefälle (bleibt bewusst zusätzlich
        # bestehen, siehe pressure_field oben - "realistische Wetterlagen
        # kommen auch von außerhalb der Karte"). Koeffizient: °C-Anomalie zu
        # Druck-Gradient-Einheit - Startwert (analog zu LATENT_HEAT_
        # COEFFICIENT oben), noch nicht am echten Nutzer-Feedback
        # nachjustiert. Per Checkbox abschaltbar (WEATHER.
        # THERMAL_PRESSURE_COUPLING, Default AN) - bei AUS reproduziert der
        # Code exakt den vorherigen, rein synoptischen Pfad.
        THERMAL_PRESSURE_COEFF = 0.03
        thermal_pressure_coupling = month_params.get('thermal_pressure_coupling', True)
        # Wind-Geschwindigkeits-Multiplikatoren für die Initialbedingung je
        # Schicht (Wind nimmt mit Höhe zu, geringere Bodenreibung) - nur
        # Startwert, der Loop entwickelt die tatsächliche Struktur.
        LAYER_WIND_SEED_MULT = (1.0, 1.3, 1.6)
        LAYER_HUMID_SEED_MULT = (1.0, 0.5, 0.2)
        # Terrain-Kopplung/-Ablenkung (die diese Session zuvor für die
        # Einzelschicht-CFD hinzugefügten Terme) wirkt gedämpft mit der Höhe -
        # voll auf GROUND, exponentiell gedämpft auf MID, gar nicht auf HIGH
        # (siehe [[project-3layer-wind-cfd]]: der primäre Terrain-
        # Beschleunigungseffekt entsteht jetzt aus der Kontinuitätskorrektur
        # über die terrain-folgende Schichtgeometrie selbst).
        TERRAIN_TERM_SCALE = (1.0, float(np.exp(-L.REF_ALTITUDE_AGL[L.MID] / 1000.0)), 0.0)
        # Vorticity Confinement (Fedkiw/Stam, siehe _apply_vorticity_confinement)
        # - Turbulenz entsteht überwiegend durch Bodenreibung/Oberflächen-Rauheit,
        # deshalb stärker auf GROUND, schwächer auf MID/HIGH (aber nicht 0 wie bei
        # den Terrain-Termen - auch die freie Atmosphäre zeigt etwas Verwirbelung).
        VORTICITY_LAYER_SCALE = (1.0, 0.6, 0.3)
        # Rauigkeits-Dämpfung: wie die Terrain-Terme voll auf GROUND, gedämpft
        # auf MID, keine auf HIGH (Bodenreibung wirkt per Definition nur nahe
        # der Oberfläche) - siehe [[project-wind-roughness]].
        ROUGHNESS_LAYER_SCALE = (1.0, 0.3, 0.0)
        # Lee-Turbulenz: Rotoren/Verwirbelungen im Lee eines Grats bilden sich
        # nahe der Oberfläche, deshalb wie Terrain-/Rauigkeits-Terme voll auf
        # GROUND, gedämpft auf MID, keine auf HIGH - siehe
        # [[project-wind-lee-turbulence]].
        LEE_LAYER_SCALE = (1.0, 0.4, 0.0)

        y_idx, x_idx = np.mgrid[0:height, 0:width].astype(np.float64)
        slopemap = self._calculate_slopes_vectorized(heightmap)
        curvature_norm = self._calculate_curvature_normalized(heightmap)

        # Gewichtete Solar-Einstrahlung - Kombination der 7 Sonnenwinkel-Kanäle
        # über einen aus dem ECHTEN, monats-/breitengrad-abhängigen Sonnenstand
        # abgeleiteten Airmass-Dämpfungsfaktor (siehe _weighted_solar_exposure()
        # Docstring, ATM_OPTICAL_DEPTH oben) statt der alten festen Tageszeit-
        # Gewichtung. Die Hangausrichtungs-Abhängigkeit selbst steckt bereits
        # PRO KANAL in shadowmap (dot(normal, sun_dir) aus _calculate_slope_
        # shading_cpu je Sonnenwinkel) - hier kommt nur die Gewichtung der
        # Kanäle untereinander hinzu. Einmalig vor dem Zeitschritt-Loop
        # berechnet (shadowmap ändert sich innerhalb eines Monats nicht),
        # analog zu roughness_damping.
        solar_exposure = self._weighted_solar_exposure(shadowmap, sun_angles=sun_angles)

        # Bodentemperatur-Modell (löst den alten additiven solar_power-Term
        # ab, siehe Plan "Weather: Bodentemperatur-Modell + konvektiver
        # Wärmeübergang"): T_boden = T_min + (T_max-T_min)*effective_exposure,
        # eine echte Soll-Temperatur statt einer additiven Störung. Biom-
        # abhängiger Absorptionsfaktor (best-effort, None bei fehlenden Biome-
        # Daten -> keine Änderung ggü. reiner solar_exposure) dämpft, wie weit
        # ein Pixel die volle Spanne tatsächlich erreicht (dichte Vegetation
        # reflektiert/verdunstet einen Teil weg, siehe _BIOME_SOLAR_ABSORPTION).
        ground_temp_baseline = month_params['ground_temp_baseline']
        # sun_relevance_factor (Nutzer-Slider, Default 1.0 = unveraendert)
        # skaliert die feste GROUND_TEMP_SPREAD - bei 0 hat Sonnenexposition
        # keinen Einfluss mehr auf T_boden, nur noch ground_temp_baseline
        # zaehlt. .get() mit Default statt required_params, da rueckwaerts-
        # kompatibel zu Aufrufern/Tests ohne diesen neuen Parameter.
        sun_relevance = month_params.get('sun_relevance_factor', 1.0)
        effective_spread = GROUND_TEMP_SPREAD * sun_relevance
        ground_temp_min = ground_temp_baseline - effective_spread / 2.0
        ground_temp_max = ground_temp_baseline + effective_spread / 2.0
        effective_exposure = solar_exposure * (
            solar_absorption_factor if solar_absorption_factor is not None else 1.0)
        effective_exposure = np.clip(effective_exposure, 0.0, 1.0)
        # UM DIE TATSAECHLICHE MITTLERE EXPOSITION ZENTRIERT, nicht um 0.5.
        #
        # Vorher lautete die Formel T = Basis + SPREAD * (Exposition - 0.5).
        # Sie unterstellt damit, dass eine Karte im Mittel halb besonnt ist -
        # das gilt aber nur bei etwa 40 Grad Breite. Gemessen auf flachem
        # Gelaende (mittlere Exposition aus der Shadowmap):
        #
        #     Breite   Exposition   daraus Versatz
        #        0        0.806        +6.1 C
        #       20        0.711        +4.2 C
        #       40        0.531        +0.6 C
        #       60        0.287        -4.3 C
        #
        # Die Klimatologie (_climate_baseline) enthaelt den Sonnenstand aber
        # BEREITS - sie ist ja nach Breite und Monat tabelliert. Der Term hat
        # ihn also ein zweites Mal aufaddiert. Gemessen wurde die Simulation
        # dadurch am Aequator 11.4 C waermer als ihre eigene Vorgabe.
        #
        # Zentriert auf den Mittelwert gilt: das Kartenmittel der
        # Bodentemperatur IST die Klimatologie, und die Spanne erzeugt nur
        # noch, wofuer sie gedacht ist - den raeumlichen Unterschied zwischen
        # Sonn- und Schatthang.
        ground_temp_target = (
            ground_temp_baseline
            + effective_spread * (effective_exposure - float(np.mean(effective_exposure)))
        ).astype(np.float32)

        # Hangflächen-Korrektur für den Boden-Luft-Wärmeübergang (dimensionsloses
        # Flächenverhältnis, KEINE absolute m²-Fläche nötig - kürzt sich in der
        # Formel unten heraus). gx,gy = dz/dx, dz/dy (m/m), aus der ohnehin schon
        # berechneten slopemap oben.
        ground_area_slope_ratio = np.sqrt(1.0 + slopemap[:, :, 0] ** 2 + slopemap[:, :, 1] ** 2).astype(np.float32)

        # Rand-Verstärkung für Vorticity Confinement (Nutzer-Wunsch aus einer
        # früheren Runde: "an der Kartengrenze mehr Varianz" - der Kartenrand
        # ist, wo die synoptische Randbedingung einströmt, dort real am
        # wenigsten "ausgeglichen"). Faktor 1.0 in der Kartenmitte, bis 2.5x
        # direkt am (jetzt: Puffer-)Rand, exponentiell abklingend.
        # Abkling-Distanz: bei aktivem Rand-Puffer (edge_pad_px, siehe oben)
        # AN DEN PUFFER GEKOPPELT statt an die (jetzt größere, gepolsterte)
        # Gittergröße - sonst reicht der Boost, der ursprünglich für den
        # damaligen 5px-Rand kalibriert war, bei den heutigen typischen
        # Kartengrößen deutlich weiter als der neue Rand-Puffer breit ist und
        # bleibt nach dem Zurückschneiden sichtbar in der eigentlichen
        # Kartenfläche - genau die vom Nutzer beobachteten "immer noch starken
        # Randeffekte" trotz Rand-Puffer. Ohne Puffer (edge_pad_px=0) bleibt
        # die alte, größen-relative Abklingdistanz als Fallback erhalten.
        edge_dist = np.minimum.reduce([x_idx, width - 1 - x_idx, y_idx, height - 1 - y_idx])
        vorticity_decay_dist = edge_pad_px * 0.6 if edge_pad_px > 0 else 0.08 * min(height, width)
        vorticity_edge_boost = (1.0 + 1.5 * np.exp(
            -edge_dist / max(vorticity_decay_dist, 1e-6))).astype(np.float32)

        # Sponge-Layer für den Rand-Puffer (siehe edge_pad_px oben) - dämpft
        # u/v/theta/q im Puffer-Bereich pro Zeitschritt sanft zurück zu ihrem
        # Ausgangszustand (siehe theta_bg/u_bg/v_bg/q_bg unten), 0 in der
        # eigentlichen (später zurückgeschnittenen) Kartenfläche, glatt
        # (smoothstep) auf _EDGE_SPONGE_MAX_STRENGTH ansteigend zum äußersten
        # Puffer-Pixel hin - Standardtechnik für offene Ränder in
        # CFD-/Wettersimulationen, verhindert dass der künstliche Puffer-Rand
        # selbst Reflexionen/Artefakte erzeugt, die in die Kartenfläche
        # hineinadvehieren.
        if edge_pad_px > 0:
            sponge_t = np.clip((edge_pad_px - edge_dist) / edge_pad_px, 0.0, 1.0)
            sponge_t = sponge_t * sponge_t * (3.0 - 2.0 * sponge_t)  # smoothstep
            sponge_weight = (_EDGE_SPONGE_MAX_STRENGTH * sponge_t).astype(np.float32)
        else:
            sponge_weight = None

        # Gemeinsames synoptisches Druckfeld (eine Größenordnung, alle Schichten
        # spüren dieselbe großräumige Richtung + Monats-Rauschen - nur die
        # Terrain-Kopplung unten unterscheidet sich pro Schicht).
        wind_direction_deg = month_params.get('prevailing_wind_direction', 0.0)
        pressure_field = self._build_directional_pressure_field(height, width, wind_direction_deg)
        pressure_field = pressure_field + self._generate_pressure_noise(
            (height, width), month_params.get('month_index', 0)) * 0.15

        height_range = heightmap.max() - heightmap.min()
        height_normalized = (heightmap - heightmap.min()) / height_range if height_range > 1e-6 \
            else np.zeros_like(heightmap, dtype=np.float32)
        terrain_pressure_layers = [
            height_normalized * month_params['terrain_factor'] * 0.2 * TERRAIN_TERM_SCALE[i]
            for i in range(L.COUNT)
        ]

        # --- Initialbedingung ---
        if initial_state is not None:
            # LOD-Vererbung (Weather-Rework Punkt F) - Endzustand derselben
            # saisonalen Periode der Vorstufe bikubisch hochskaliert, statt
            # der reinen Noise-Randomisierung unten. Rückumrechnung reale
            # Temp -> potentielle Temp ist die exakte Umkehrung von t_real
            # weiter unten in der Zeitschritt-Schleife.
            #
            # Bug-Fix (Nutzer-Bug-Report 2026-07-24, "alles wird viel zu kalt
            # über mehrere LOD-Runden"): initial_state['temp_layers'] (und
            # wind_layers/humid_layers) sind das bereits ZURÜCKGESCHNITTENE
            # Ergebnis der Vorstufe (siehe "Ergebnis zusammensetzen" unten,
            # der Rand-Puffer-Crop passiert dort VOR dem Speichern) - also in
            # der UNGEPOLSTERTEN Zielgröße (target_size), NICHT in der
            # gepolsterten Arbeitsgittergröße (height/width) dieser Runde.
            # Die vorherige Fassung interpolierte direkt auf height/width
            # hoch - das dehnt das ungepolsterte Vorstufen-Ergebnis über das
            # GESAMTE (größere, gepolsterte) Gitter, wodurch jeder Pixel an
            # einer FALSCHEN, verschobenen Position landet, sobald der Rand-
            # Puffer aktiv ist (siehe _compute_edge_padding_px() oben - bei
            # kleinen LODs ein erheblicher Anteil der Gittergröße). Die
            # anschließende theta-Rekonstruktion (+ altitude_cooling_rate *
            # heightmap) kombinierte dadurch das (falsch positionierte)
            # ererbte real_temp_i mit der (korrekt positionierten) aktuellen
            # heightmap an im Wesentlichen ZUFÄLLIG unpassenden Punkten -
            # der Höhen-Anteil hob sich beim theta<->t_real-Roundtrip NICHT
            # mehr sauber auf, sondern akkumulierte über jede LOD-Runde einen
            # Fehler (empirisch verifiziert: frischer LOD-3-Sprung ohne
            # Vererbung kam bei einem Berg auf ~-24°C, derselbe Berg über
            # LOD1->2->3 vererbt auf ~-45°C).
            # Fix: ERST auf die ungepolsterte target_size hochskalieren
            # (passend zur Größe, in der die Vorstufe tatsächlich gespeichert
            # wurde), DANN mit demselben Rand-Puffer wie heightmap/shadowmap/
            # soil_moisture_field oben polstern (np.pad mode='edge') - erst
            # danach sind alle Felder wieder pixelgenau deckungsgleich.
            prev_temp_layers = initial_state['temp_layers']
            prev_wind_layers = initial_state['wind_layers']
            prev_humid_layers = initial_state['humid_layers']
            theta, u, v, q = [], [], [], []
            for i in range(L.COUNT):
                real_temp_i = self._interpolate_2d_bicubic(prev_temp_layers[i], target_size)
                u_i = self._interpolate_2d_bicubic(prev_wind_layers[i, :, :, 0], target_size)
                v_i = self._interpolate_2d_bicubic(prev_wind_layers[i, :, :, 1], target_size)
                q_i = self._interpolate_2d_bicubic(prev_humid_layers[i], target_size)
                if edge_pad_px > 0:
                    real_temp_i = np.pad(real_temp_i, edge_pad_px, mode='edge')
                    u_i = np.pad(u_i, edge_pad_px, mode='edge')
                    v_i = np.pad(v_i, edge_pad_px, mode='edge')
                    q_i = np.pad(q_i, edge_pad_px, mode='edge')
                # Bug-Fix (Nutzer-Bug-Report 2026-07-23) - siehe ausführliche
                # Begründung beim Haupt-Loop unten (Suchbegriff "t_real[i] =").
                # heightmap gehört jetzt mit zur theta<->t_real-Umrechnung.
                theta.append((real_temp_i + altitude_cooling_rate * (heightmap + L.REF_ALTITUDE_AGL[i])
                              ).astype(np.float32))
                u.append(u_i.astype(np.float32))
                v.append(v_i.astype(np.float32))
                q.append(q_i.astype(np.float32))
        else:
            surface_temp = self._calculate_temperature_field(heightmap, shadowmap, month_params, target_size)
            # Bug-Fix (Nutzer-Bug-Report 2026-07-23): surface_temp enthält
            # bereits den Höhen-Lapse-Term (_calculate_temperature_field
            # zieht heightmap*altitude_cooling_rate ab), theta soll aber die
            # elevations-UNABHÄNGIGE potentielle Temperatur sein (Solar-/
            # Rausch-Variation bleibt erhalten, NUR der Höhen-Anteil wird
            # wieder herausgerechnet) - sonst wird die lokale Terrainhöhe nur
            # EINMALIG hier eingeprägt und geht bei jeder horizontalen
            # Advektion (Wind trägt theta über die Karte) verloren, weil
            # t_real weiter unten nur noch den FESTEN Schicht-AGL-Versatz
            # abzieht, nie die tatsächliche Terrainhöhe DIESER Zelle. Ergebnis
            # vorher: warme Luft "vergisst" beim Herunterwehen vom Berg ihre
            # Herkunftshöhe nicht (trägt weiter ihr kaltes theta), aber kalte
            # Bergluft, die durch wärmere Tal-Luft ersetzt wird, kühlt sich
            # NICHT mehr ab, sobald sie am Berg ankommt - Berge wurden dadurch
            # tendenziell zu warm, Täler im Vergleich zu kalt.
            theta_seed = surface_temp + altitude_cooling_rate * heightmap
            # Konstante potentielle Temperatur über alle Schichten als Start (gut
            # durchmischte Atmosphäre) - reale Temperatur pro Schicht ergibt sich
            # daraus automatisch kälter mit Höhe (siehe Klassen-Docstring von
            # AtmosphereLayers).
            theta = [theta_seed.astype(np.float32).copy() for _ in range(L.COUNT)]

            base_wind = self._simulate_wind_field_simple(heightmap, month_params)
            u = [(base_wind[:, :, 0] * LAYER_WIND_SEED_MULT[i]).astype(np.float32) for i in range(L.COUNT)]
            v = [(base_wind[:, :, 1] * LAYER_WIND_SEED_MULT[i]).astype(np.float32) for i in range(L.COUNT)]

            ground_temp_c = np.clip(
                theta[L.GROUND] - altitude_cooling_rate * (heightmap + L.REF_ALTITUDE_AGL[L.GROUND]), -50, 60)
            sat_vp0 = 6.112 * np.exp(17.67 * ground_temp_c / (ground_temp_c + 243.5))
            ground_wind_speed0 = np.sqrt(u[L.GROUND] ** 2 + v[L.GROUND] ** 2)
            wind_factor0 = np.minimum(2.0, ground_wind_speed0 / 5.0)
            evap_rate0 = soil_moisture_norm * (sat_vp0 / 100.0) * (1.0 + wind_factor0)

            # Bug-Fix (Nutzer-Bug-Report 2026-07-23): air_humidity_entry
            # (Klimatologie-Feuchte-Basis + Slider-Offset, siehe
            # _generate_seasonal_parameters) wurde bisher NIRGENDS im
            # gekoppelten Loop gelesen - q kam bisher AUSSCHLIESSLICH aus der
            # Verdunstung, unabhängig vom Breitengrad/Slider (verifiziert:
            # trockene vs. feuchte air_humidity_entry-Werte ergaben exakt
            # identische precip_map-Ergebnisse). Ambiente Basis-Feuchte auf
            # dieselbe rho_max-Skala umgerechnet wie die Kondensations-
            # Sättigung weiter unten im Loop (rho_max = 5*exp(0.06*T), siehe
            # dortiger Kommentar "identische Magnus-Skala") und additiv zur
            # Verdunstungs-Komponente addiert - beide speisen denselben
            # q-Ausgangszustand (und darüber automatisch auch q_bg/den Sponge-
            # Rand-Hintergrund, siehe direkt unterhalb dieses Blocks).
            relative_humidity_frac = np.clip(
                month_params.get('air_humidity_entry', 50.0) / 100.0, 0.0, 1.0)
            rho_max_ground = 5.0 * np.exp(0.06 * ground_temp_c)
            ambient_q0 = relative_humidity_frac * rho_max_ground

            q = [((evap_rate0 * 140.0 + ambient_q0) * LAYER_HUMID_SEED_MULT[i]).astype(np.float32)
                 for i in range(L.COUNT)]

        # Ausgangszustand für den Sponge-Layer (siehe sponge_weight oben) -
        # der Rand-Puffer wird pro Zeitschritt zu DIESEM (noch unbeeinflussten)
        # Zustand hin gedämpft, nicht zu einem hart vorgegebenen Wert.
        # theta_bg wird IMMER gebraucht - die Strahlungsrueckstellung unten
        # wirkt flaechendeckend, nicht nur im Randstreifen.
        theta_bg = [t.copy() for t in theta]
        if sponge_weight is not None:
            u_bg = [x.copy() for x in u]
            v_bg = [x.copy() for x in v]
            q_bg = [x.copy() for x in q]

        precip_accum = np.zeros((height, width), dtype=np.float32)

        for step in range(n_steps):
            if step % max(1, n_steps // 5) == 0:
                progress = 20 + (step / max(1, n_steps)) * 40
                self._update_progress("Atmosphere", int(progress), f"Coupled step {step + 1}/{n_steps}")

            t_real = [None] * L.COUNT
            for i in range(L.COUNT):
                # 1. Semi-Lagrange-Advektion (alle 4 Felder mit dem VOR der
                # Advektion gültigen Geschwindigkeitsfeld rückwärts-gesampelt -
                # Standard-Semi-Lagrange-Konsistenz).
                u_old, v_old = u[i], v[i]
                # Ein Aufruf statt vier: alle vier Felder werden mit demselben
                # u_old/v_old gesampelt, teilen sich also Quellkoordinaten und
                # Gewichte (siehe _semi_lagrangian_advect_many, -31 % gemessen).
                u[i], v[i], theta[i], q[i] = self._semi_lagrangian_advect_many(
                    (u_old, v_old, theta[i], q[i]), u_old, v_old, y_idx, x_idx)

                # 2a. Druckgradient - Relaxation statt hartem Reset (siehe
                # PRESSURE_RELAX_RATE-Kommentar oben). t_real[i] wird hier
                # (statt erst in Schritt 2c wie zuvor) benötigt, sobald das
                # thermisch gekoppelte Druckfeld aktiv ist - deshalb vorgezogen,
                # Schritt 2c liest denselben Wert weiter unten nur noch.
                # Bug-Fix (Nutzer-Bug-Report 2026-07-23): heightmap (statisches
                # Terrain, wird NICHT advehiert) gehört mit in die t_real-
                # Ableitung - theta selbst ist jetzt elevations-unabhängig
                # (siehe Initialbedingung oben), die tatsächliche Höhen-
                # Abkühlung/-Erwärmung muss deshalb JEDEN Schritt FRISCH aus
                # der lokalen Terrainhöhe DIESER Zelle berechnet werden, nicht
                # nur einmalig beim Seeding - sonst "vergisst" advehierte Luft
                # ihre ursprüngliche Herkunftshöhe nicht bzw. "merkt" nie, dass
                # sie jetzt über anderem Terrain steht.
                t_real[i] = theta[i] - altitude_cooling_rate * (heightmap + L.REF_ALTITUDE_AGL[i])
                pressure_iter = pressure_field - terrain_pressure_layers[i]
                if thermal_pressure_coupling:
                    thermal_pressure_term = -(t_real[i] - np.mean(t_real[i])) * THERMAL_PRESSURE_COEFF
                    pressure_iter = pressure_iter + thermal_pressure_term
                grad_x = np.zeros((height, width), dtype=np.float32)
                grad_y = np.zeros((height, width), dtype=np.float32)
                grad_x[:, 1:-1] = (pressure_iter[:, 2:] - pressure_iter[:, :-2]) * 0.5
                grad_y[1:-1, :] = (pressure_iter[2:, :] - pressure_iter[:-2, :]) * 0.5
                u_target = -grad_x * month_params['wind_speed_factor'] * 10.0
                v_target = -grad_y * month_params['wind_speed_factor'] * 10.0
                u[i] += (u_target - u[i]) * PRESSURE_RELAX_RATE
                v[i] += (v_target - v[i]) * PRESSURE_RELAX_RATE

                # 2b. Terrain-Ablenkung (gedämpft mit Höhe, siehe TERRAIN_TERM_SCALE)
                # + Grat-/Canyon-Speedup (WindNinja-Terrain-Shape-Effekt,
                # siehe [[project-wind-ridge-speedup]]): zusätzlich zur reinen
                # Hang-Ablenkung beschleunigt Wind multiplikativ über
                # konvexen Graten und bremst in konkaven Tälern, gleiche
                # Höhendämpfung wie die Ablenkung selbst.
                terrain_term = month_params['terrain_factor'] * 0.5 * TERRAIN_TERM_SCALE[i]
                if terrain_term != 0.0:
                    u[i] += slopemap[:, :, 1] * terrain_term
                    v[i] -= slopemap[:, :, 0] * terrain_term
                    ridge_factor = 1.0 + _RIDGE_SPEEDUP_STRENGTH * (-curvature_norm) * \
                        month_params['terrain_factor'] * TERRAIN_TERM_SCALE[i]
                    u[i] *= ridge_factor
                    v[i] *= ridge_factor

                # 2c. Thermische Konvektion (Schatten-Term nur GROUND - nur die
                # Bodenschicht "sieht" die Sonneneinstrahlung direkt, siehe
                # Docstring Schritt 3 in der Klassen-Beschreibung oben).
                # t_real[i] bereits in Schritt 2a berechnet (siehe dortiger
                # Kommentar zum thermisch gekoppelten Druckfeld).
                temp_grad_x = np.zeros((height, width), dtype=np.float32)
                temp_grad_y = np.zeros((height, width), dtype=np.float32)
                temp_grad_x[:, 1:-1] = (t_real[i][:, 2:] - t_real[i][:, :-2]) * 0.5
                temp_grad_y[1:-1, :] = (t_real[i][2:, :] - t_real[i][:-2, :]) * 0.5
                convection_strength = (t_real[i] - np.mean(t_real[i])) * month_params['thermic_effect'] * 0.08
                u[i] += temp_grad_x * 0.05 + convection_strength
                if i == L.GROUND:
                    shadow_effect = (solar_exposure - 0.5) * month_params['thermic_effect'] * 0.15
                    v[i] += temp_grad_y * 0.05 + shadow_effect
                else:
                    v[i] += temp_grad_y * 0.05

                # 2d. Rauigkeits-Dämpfung aus Biome (best-effort, siehe
                # [[project-wind-roughness]]) - läuft VOR der Verdunstungs-
                # Windgeschwindigkeit in Schritt 3, damit rauigkeitsbedingt
                # gebremster Bodenwind auch die Verdunstungs-Verstärkung
                # konsistent mitreduziert.
                if roughness_damping is not None and ROUGHNESS_LAYER_SCALE[i] > 0.0:
                    damping = roughness_damping * ROUGHNESS_LAYER_SCALE[i]
                    u[i] *= (1.0 - damping)
                    v[i] *= (1.0 - damping)

                # 2e. Boden->Luft-Wärmeübergang (löst den alten additiven
                # solar_power-Term ab, siehe Plan "Weather: Bodentemperatur-
                # Modell + konvektiver Wärmeübergang" und ground_temp_target/
                # ground_area_slope_ratio oben). NACH der Rauigkeits-Dämpfung
                # platziert (Nutzer-bestätigt) - konsistent mit der
                # Verdunstungs-Windböen-Verstärkung in Schritt 3 unten, die
                # ebenfalls bewusst den bereits gedämpften Wind liest, statt
                # an der alten (vor der Dämpfung liegenden) Stelle des
                # ehemaligen solar_power-Terms.
                #
                # Paquet-Formel für den konvektiven Wärmeübergangskoeffizienten
                # (W/(m^2*K), 5.8 = Windstille-Basiswert), plus ein vom Nutzer
                # vorgegebener, massenerhaltungs-bewusster "effektiver
                # Windfaktor": bei mehr Wind strömt pro Zeitschritt auch mehr
                # Luftmasse durch die Referenzsäule (GROUND_HEAT_COLUMN_HEIGHT_M),
                # die einzelne vorbeistreichende Luftmasse erwärmt sich trotz
                # höheren Wärmeübergangs-Vielfachen also weniger stark - siehe
                # GROUND_HEAT_CAPACITY_PER_M2-Docstring oben. v_eff clamped
                # gegen Divisionsblowup nahe 0 m/s (glättet einen Sprung in der
                # wörtlichen 3-Zweig-Nutzer-Formel, siehe Plan "Numerische
                # Prüfung").
                if i == L.GROUND:
                    wind_speed_ground = np.sqrt(u[L.GROUND] ** 2 + v[L.GROUND] ** 2)
                    alpha_v = np.where(
                        wind_speed_ground <= 5.0,
                        ALPHA0 + 3.8 * wind_speed_ground,
                        7.1 * np.power(np.maximum(wind_speed_ground, 1e-6), 0.78),
                    )
                    v_eff = np.maximum(wind_speed_ground, WIND_FACTOR_MIN_SPEED)
                    effective_wind_factor = alpha_v / (GROUND_HEAT_CAPACITY_PER_M2 * v_eff)
                    dt_seconds = GROUND_HEAT_TIME_SCALE_S / max(n_steps, 1)
                    ground_air_diff = ground_temp_target - t_real[L.GROUND]
                    delta_t_ground = (ground_air_diff * ground_area_slope_ratio
                                       * effective_wind_factor * dt_seconds)
                    # Stabilitäts-Clamp (nicht Teil der Nutzer-Formel, aber
                    # empfohlen): verhindert ein Überschießen über
                    # ground_temp_target hinaus in einem einzelnen Schritt,
                    # unabhängig vom genauen Kalibrierungsstand der Konstanten
                    # oben - macht das System unbedingt stabil.
                    delta_t_ground = np.clip(delta_t_ground, -np.abs(ground_air_diff), np.abs(ground_air_diff))
                    theta[L.GROUND] += delta_t_ground

                # 3. Latentwärme - identische Magnus-Skala wie
                # _calculate_precipitation_cpu/_calculate_atmospheric_moisture_cpu
                # (siehe [[project-precip-humidity-calibration]]), KEIN zweites
                # Feuchte-Einheitensystem.
                t_clamped = np.clip(t_real[i], -40, 50)
                rho_max = 5.0 * np.exp(0.06 * t_clamped)
                oversaturation = np.maximum(0.0, q[i] / np.maximum(rho_max, 1e-6) - 1.0)
                condensation = np.minimum(oversaturation * rho_max * 0.6, q[i])
                theta[i] += condensation * LATENT_HEAT_COEFFICIENT
                q[i] = np.maximum(q[i] - condensation, 0.0)
                # Wasserbilanz (Nutzer-Vorgabe: "es soll nichts aus dem Nichts
                # erschaffen oder vernichtet werden"): FRÜHER trug nur GROUND/MID
                # zu precip_accum bei - HIGH-Kondensation wurde aus q[HIGH]
                # entfernt, aber NIRGENDS wieder aufgeführt und verschwand damit
                # spurlos aus der Bilanz. Jetzt zählt jede Schicht mit.
                precip_accum += condensation

                if i == L.GROUND:
                    ground_temp_c = np.clip(t_real[i], -50, 60)
                    sat_vp = 6.112 * np.exp(17.67 * ground_temp_c / (ground_temp_c + 243.5))
                    wind_speed = np.sqrt(u[i] ** 2 + v[i] ** 2)
                    wind_factor = np.minimum(2.0, wind_speed / 5.0)
                    evap_rate = soil_moisture_norm * (sat_vp / 100.0) * (1.0 + wind_factor)
                    # Über n_steps verteilt, damit die kumulierte Zufuhr über den
                    # ganzen Loop näherungsweise dieselbe Größenordnung erreicht
                    # wie der bisherige Einzelschuss-Faktor 140.0.
                    evap_source = evap_rate * 140.0 / max(n_steps, 1)
                    q[L.GROUND] += evap_source
                    theta[L.GROUND] -= evap_source * LATENT_HEAT_COEFFICIENT

            # 4. Vertikaler Austausch (Thermik-getrieben, symmetrische Relaxation
            # - erhält die Schicht-Summe von theta/q/u/v exakt, siehe
            # Methoden-Docstring).
            w_gm = np.maximum(0.0, t_real[L.GROUND] - t_real[L.MID]) * month_params['thermic_effect'] \
                * VERTICAL_EXCHANGE_COEFF
            w_mh = np.maximum(0.0, t_real[L.MID] - t_real[L.HIGH]) * month_params['thermic_effect'] \
                * VERTICAL_EXCHANGE_COEFF
            f_gm = np.clip(w_gm, 0.0, MAX_VERTICAL_EXCHANGE_FRACTION)
            f_mh = np.clip(w_mh, 0.0, MAX_VERTICAL_EXCHANGE_FRACTION)

            for field_list in (theta, q, u, v):
                lower, mid_, upper = field_list[L.GROUND], field_list[L.MID], field_list[L.HIGH]
                delta_gm = f_gm * (lower - mid_)
                delta_mh = f_mh * (mid_ - upper)
                field_list[L.GROUND] = lower - delta_gm
                field_list[L.MID] = mid_ + delta_gm - delta_mh
                field_list[L.HIGH] = upper + delta_mh

            vertical_flux_terms = (
                -w_gm / AtmosphereLayers.THICKNESS_M[L.GROUND],
                (w_gm - w_mh) / AtmosphereLayers.THICKNESS_M[L.MID],
                w_mh / AtmosphereLayers.THICKNESS_M[L.HIGH],
            )

            # 5. Diffusion + Vorticity Confinement + 6-Richtungs-Kontinuitätskorrektur
            # pro Schicht. Vorticity Confinement läuft NACH der Diffusion (die
            # Diffusion glättet zuerst, numerische Stabilität bleibt erhalten) und
            # VOR der Kontinuitätskorrektur (die das Ergebnis danach wieder auf
            # Massenerhaltung ausbalanciert) - würde die Reihenfolge vertauscht,
            # würde die Diffusion die injizierte Verwirbelung im selben Schritt
            # wieder wegbügeln.
            turbulence_strength = month_params.get(
                'turbulence_strength', WEATHER.TURBULENCE_STRENGTH["default"])
            for i in range(L.COUNT):
                wind_field_i = np.stack([u[i], v[i]], axis=-1).astype(np.float32)
                wind_field_i = self._apply_wind_diffusion(wind_field_i, 0.1)
                if turbulence_strength > 0.0:
                    vorticity_strength_i = (
                        turbulence_strength * VORTICITY_LAYER_SCALE[i] * vorticity_edge_boost)
                    if LEE_LAYER_SCALE[i] > 0.0:
                        # Lee-Turbulenz-Heuristik (siehe [[project-wind-lee-turbulence]]):
                        # "Lee-Signal" = wie stark das Gelände gerade IN
                        # aktueller Windrichtung abfällt (gerichtete
                        # Ableitung der Höhe entlang des lokalen, bereits
                        # diffundierten Windvektors) - positiv nur dort, wo
                        # Wind über einen Grat/Kamm hinweg bergab fließt.
                        # Robust auf [0,1] normiert (3-Sigma-Clip, wie
                        # curvature_norm oben), boostet lokal die ohnehin
                        # bestehende Vorticity-Confinement-Stärke statt einen
                        # neuen Kraft-Mechanismus einzuführen.
                        speed_i = np.sqrt(wind_field_i[:, :, 0] ** 2 + wind_field_i[:, :, 1] ** 2)
                        safe_speed_i = np.maximum(speed_i, 1e-6)
                        slope_along_wind = (
                            slopemap[:, :, 0] * wind_field_i[:, :, 0] +
                            slopemap[:, :, 1] * wind_field_i[:, :, 1]) / safe_speed_i
                        lee_signal = np.maximum(0.0, -slope_along_wind)
                        lee_scale = float(np.std(lee_signal)) + 1e-9
                        lee_norm = np.clip(lee_signal / (3.0 * lee_scale), 0.0, 1.0)
                        vorticity_strength_i = vorticity_strength_i * (
                            1.0 + _LEE_TURBULENCE_BOOST * lee_norm * LEE_LAYER_SCALE[i])
                    self._apply_vorticity_confinement(wind_field_i, vorticity_strength_i)
                self._apply_continuity_correction(wind_field_i, vertical_flux_term=vertical_flux_terms[i])
                u[i], v[i] = wind_field_i[:, :, 0], wind_field_i[:, :, 1]

            # 6. Sponge-Layer (siehe sponge_weight oben) - dämpft den
            # Rand-Puffer-Bereich EINMAL pro Zeitschritt (nicht pro Schicht-
            # Teilschritt) sanft zurück zu seinem Ausgangszustand, NACH allen
            # Physik-Schritten dieses Zeitschritts, damit die eigentliche
            # Kartenfläche (sponge_weight dort == 0) unverändert bleibt.
            if sponge_weight is not None:
                for i in range(L.COUNT):
                    theta[i] += (theta_bg[i] - theta[i]) * sponge_weight
                    q[i] += (q_bg[i] - q[i]) * sponge_weight
                    u[i] += (u_bg[i] - u[i]) * sponge_weight
                    v[i] += (v_bg[i] - v[i]) * sponge_weight

            # 7. Strahlungsrueckstellung, FLAECHENDECKEND (siehe
            # RADIATIVE_RELAX_RATE). Die fehlende Waermesenke des Modells -
            # ohne sie summieren Bodenwaerme und Latentwaerme ueber die
            # Schritte auf und die Karte laeuft von ihrer eigenen Klimatologie
            # weg.
            if RADIATIVE_RELAX_RATE > 0.0:
                for i in range(L.COUNT):
                    theta[i] += (theta_bg[i] - theta[i]) * RADIATIVE_RELAX_RATE

        # --- Ergebnis zusammensetzen ---
        # heightmap-Anteil siehe Bug-Fix-Kommentar beim Haupt-Loop oben
        # (Suchbegriff "t_real[i] =") - dieselbe Umrechnung fürs finale Ergebnis.
        temp_layers = np.stack(
            [theta[i] - altitude_cooling_rate * (heightmap + L.REF_ALTITUDE_AGL[i]) for i in range(L.COUNT)], axis=0
        ).astype(np.float32)
        wind_layers = np.stack(
            [np.stack([u[i], v[i]], axis=-1) for i in range(L.COUNT)], axis=0
        ).astype(np.float32)
        # Baroklinitäts-Skalierung (Nutzer-Bug-Report 2026-07-23, siehe
        # _baroclinic_wind_factor()-Docstring) - macht hohe Breite im Winter
        # spürbar stürmischer, den Äquator ganzjährig ruhiger, angewendet auf
        # das fertige Windfeld VOR dem orographischen Niederschlags-Anteil
        # unten (der dadurch konsistent mitskaliert - stärkerer Wind bedeutet
        # real auch mehr Steigungsregen, kein Nebeneffekt zum Ignorieren).
        baroclinic_factor = self._baroclinic_wind_factor(
            month_params.get('map_latitude', 48.0), month_params.get('month_index', 0))
        wind_layers = wind_layers * baroclinic_factor
        humid_layers = np.stack(q, axis=0).astype(np.float32)

        # Orographischer Zusatzbeitrag (Luv-/Lee, identische Formel wie
        # _calculate_precipitation_cpu) auf Basis der finalen GROUND-Schicht -
        # separat von precip_accum gehalten, um Magnus-Kondensation nicht
        # doppelt zu zählen (precip_accum hat die Kondensation bereits über
        # den ganzen Loop akkumuliert).
        wind_speed_ground = np.sqrt(wind_layers[L.GROUND, :, :, 0] ** 2 + wind_layers[L.GROUND, :, :, 1] ** 2)
        wind_norm_x = np.where(wind_speed_ground > 0.1, wind_layers[L.GROUND, :, :, 0] / wind_speed_ground, 0)
        wind_norm_y = np.where(wind_speed_ground > 0.1, wind_layers[L.GROUND, :, :, 1] / wind_speed_ground, 0)
        wind_slope_alignment = wind_norm_x * slopemap[:, :, 0] + wind_norm_y * slopemap[:, :, 1]
        orographic_factor = np.maximum(0, wind_slope_alignment) * wind_speed_ground * 0.3
        oro_precip = humid_layers[L.GROUND] * orographic_factor * 0.05
        precip_raw = precip_accum + oro_precip

        if edge_pad_px > 0:
            # Rand-Puffer zurückschneiden (siehe edge_pad_px oben) - der
            # Sponge-Layer hat den Puffer-Bereich bereits pro Zeitschritt
            # sanft gedämpft, hier wird er endgültig verworfen. Die
            # Wasserbilanz-Prüfung unten bezieht sich damit ausschließlich auf
            # die tatsächlich zurückgegebene Kartenfläche - der Puffer selbst
            # ist kein Teil der Karte und darf ihre Bilanz nicht verfälschen.
            crop = slice(edge_pad_px, -edge_pad_px)
            temp_layers = temp_layers[:, crop, crop]
            wind_layers = wind_layers[:, crop, crop, :]
            humid_layers = humid_layers[:, crop, crop]
            precip_raw = precip_raw[crop, crop]

        # Wasserbilanz: die 500 gH2O/m²-Kappung war zuvor eine stille
        # Massenvernichtung (Wasser oberhalb der Grenze verschwand einfach).
        # Bleibt als reines Sicherheitsventil gegen pathologische Eingaben
        # bestehen (500 liegt weit über plausiblen Werten, siehe
        # [[project-precip-humidity-calibration]]), wird aber jetzt geloggt,
        # statt lautlos zu kappen - falls das in normalem Betrieb je greift,
        # ist das ein Hinweis auf einen echten Bilanzfehler, keine Bagatelle.
        clipped_mass = float(np.sum(np.maximum(0.0, precip_raw - 500.0)))
        if clipped_mass > 0.0:
            self.logger.warning(
                f"Niederschlags-Kappung bei 500 gH2O/m² hat {clipped_mass:.1f} gH2O/m² "
                f"Gesamtmasse entfernt - deutet auf eine pathologische Eingabe oder einen "
                f"Bilanzfehler hin, nicht auf normalen Betrieb.")
        precip_map = np.clip(precip_raw, 0.0, 500.0).astype(np.float32)

        return {
            'wind_layers': wind_layers,
            'temp_layers': temp_layers,
            'humid_layers': humid_layers,
            'precip_map': precip_map,
        }

    def _validate_inputs(self, heightmap_combined: np.ndarray, shadowmap: np.ndarray,
                        parameters: Dict[str, Any], lod_level: int):
        """
        Input-Data-Validation Pipeline für robuste Weather-Generation

        Prüft Physical-Range-Validation, Cross-Generator-Consistency und LOD-Compatibility.
        """
        # Shape-Consistency-Checks
        if heightmap_combined.shape != shadowmap.shape[:2]:
            raise ValueError(f"Shape mismatch: heightmap {heightmap_combined.shape} vs shadowmap {shadowmap.shape[:2]}")

        # Physical-Range-Validation
        if np.any(np.isnan(heightmap_combined)) or np.any(np.isinf(heightmap_combined)):
            raise ValueError("Invalid values in heightmap_combined")

        if np.any(np.isnan(shadowmap)) or np.any(np.isinf(shadowmap)):
            raise ValueError("Invalid values in shadowmap")

        # Parameter-Range-Validation
        required_params = ['air_temp_entry', 'ground_temp_offset', 'altitude_cooling',
                          'thermic_effect', 'wind_speed_factor', 'terrain_factor']

        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter: {param}")

        # Physical-Plausibility-Checks
        if not (-50 <= parameters['air_temp_entry'] <= 60):
            raise ValueError(f"air_temp_entry {parameters['air_temp_entry']} outside physical range [-50, 60]°C")

        from gui.config.value_default import WEATHER as _WEATHER_VALIDATION
        if not (_WEATHER_VALIDATION.GROUND_TEMP_OFFSET["min"] <= parameters['ground_temp_offset']
                <= _WEATHER_VALIDATION.GROUND_TEMP_OFFSET["max"]):
            raise ValueError(f"ground_temp_offset {parameters['ground_temp_offset']} outside valid range "
                              f"[{_WEATHER_VALIDATION.GROUND_TEMP_OFFSET['min']}, "
                              f"{_WEATHER_VALIDATION.GROUND_TEMP_OFFSET['max']}]°C")

        # LOD-Level-Validation
        if not (1 <= lod_level <= 10):
            raise ValueError(f"Invalid lod_level {lod_level}, must be in range [1, 10]")

    def _compute_edge_padding_px(self, grid_size: int) -> int:
        """
        Rand-Puffer in Pixeln für diese Grid-Auflösung (siehe _EDGE_PADDING_KM) -
        die CFD-Simulation läuft auf einem um diesen Puffer vergrößerten Gitter
        (Rand-Extension der Eingaben via np.pad(..., mode='edge')), ein
        Sponge-Layer dämpft den Puffer-Bereich pro Zeitschritt zurück zu seinem
        Ausgangszustand (siehe _run_coupled_atmosphere_simulation), danach wird
        auf die ursprüngliche Größe zurückgeschnitten - reduziert Rand-Artefakte
        ohne dass am Kartenrand Wassermasse verschwindet oder entsteht (der
        Sponge-Layer bildet nur auf den ohnehin verworfenen Puffer-Bereich ab,
        siehe Umsetzungsplan Punkt H "Wasserbilanz").
        """
        map_distance_km = self.data_lod_manager.get_map_distance_km() if self.data_lod_manager else None
        if not map_distance_km or map_distance_km <= 0:
            return 0
        km_per_px = map_distance_km / max(grid_size, 1)
        pad_px = int(round(_EDGE_PADDING_KM / km_per_px))
        # Nie mehr als 20% der Gittergröße - vermeidet absurd große Puffer bei
        # sehr kleinen LOD-Vorschau-Stufen oder sehr kleinen Kartengrößen.
        max_pad = max(0, grid_size // 5)
        return max(0, min(pad_px, max_pad))

    def _get_lod_size(self, lod_level: int, original_size: int) -> int:
        """
        Bestimmt Target-Size basierend auf numerischem LOD-Level

        LOD-System mit progressiver Grid-Verdopplung bis original_size erreicht
        """
        base_size = 32
        max_lod_before_original = 6

        if lod_level <= max_lod_before_original:
            # Verdopplung pro LOD-Level: 32 -> 64 -> 128 -> 256 -> 512 -> 1024
            # Bei Nicht-Zweierpotenz-Zielgrößen (z.B. 96) wird original_size (die
            # tatsächlich vom Terrain-Generator gelieferte, bereits korrekt geklemmte
            # Heightmap-Größe) schon vor Erreichen von max_lod_before_original
            # überschritten - ohne min() würde hier auf eine größere Auflösung
            # hochinterpoliert als das Terrain überhaupt hat.
            return min(base_size * (2 ** (lod_level - 1)), original_size)
        else:
            # Höhere LODs verwenden original_size
            return original_size

    def _get_cfd_iterations(self, lod_level: int) -> int:
        """
        Bestimmt CFD-Iterations basierend auf LOD-Level für Progressive Enhancement

        Steigende CFD-Komplexität: 3->5->7->10->15->20->25 Iterationen
        """
        iteration_mapping = {
            1: 3,   # LOD 32x32: 3 Iterationen für schnelle Preview
            2: 5,   # LOD 64x64: 5 Iterationen mit Enhanced-Effects
            3: 7,   # LOD 128x128: 7 Iterationen mit Detailed-Orographics
            4: 10,  # LOD 256x256: 10 Iterationen mit High-Quality-Physics
            5: 15,  # LOD 512x512: 15 Iterationen mit Premium-Simulation
            6: 20,  # LOD 1024x1024: 20 Iterationen mit Maximum-Quality
        }

        return iteration_mapping.get(lod_level, 25)  # 25+ Iterationen für höchste LODs

    def _get_atmosphere_loop_steps(self, lod_level: int) -> int:
        """
        Bestimmt die Zeitschritt-Anzahl für den gekoppelten 3-Schicht-Atmosphäre-
        Loop (_run_coupled_atmosphere_simulation, [[project-3layer-wind-cfd]]) -
        EIGENSTÄNDIG von _get_cfd_iterations() oben, das weiterhin nur die
        Einzelschicht-Fallback-Pfade treibt (_calc_wind/_calc_humidity/
        _calc_precipitation, falls der gekoppelte Loop fehlschlägt) und für einen
        anderen, bereits kalibrierten Algorithmus gilt - nicht automatisch mit
        hochziehen.

        Grob verdoppelt gegenüber der alten Tabelle: mit den alten, niedrigen
        Werten blieb zu wenig "Simulationszeit", damit sich die tatsächlich
        monats-variierenden Treiber (Windrichtungs-Drehung, Monats-Rauschen im
        Druckfeld/in der Bodentemperatur, monatliche Sonnenstand-Shadowmap) gegen
        das terrain-dominierte Gleichgewicht durchsetzen, bevor der Loop endet -
        empirisch bestätigt über eine paarweise räumliche Korrelation von
        humid_map_monthly von 0.72-0.96 (praktisch identisches Muster über alle
        6 Monate, nur das Niveau verschob sich). Startwerte, nutzerseitig als
        Performance-Kosten akzeptiert (vor dem geplanten GPU-Port).
        """
        step_mapping = {
            1: 8,   # LOD 32x32
            2: 12,  # LOD 64x64
            3: 18,  # LOD 128x128
            4: 25,  # LOD 256x256
            5: 35,  # LOD 512x512
            6: 50,  # LOD 1024x1024
        }

        # DIE SCHRITTZAHL FOLGT DEM RECHENGITTER, NICHT DEM LOD (2026-08-07).
        #
        # Seit WETTER_GITTER rechnet das Wetter immer auf 256 px, egal wie gross
        # die Karte ist. Die Schrittzahl folgte aber weiter dem LOD - bei einer
        # 512er Karte liefen also 35 Schritte auf einem 256er Gitter statt 25.
        # Gemessen: weather.temperature 25.2 s bei 512 px gegen 10.1 s bei 256,
        # obwohl beide dieselbe Arbeit haetten sein muessen.
        #
        # Das war kein Altbestand, sondern ein Fehler, den ich mit dem festen
        # Gitter selbst eingebaut habe: zwei Groessen, die zusammengehoeren,
        # aus verschiedenen Quellen zu speisen.
        #
        # Jetzt aus der tatsaechlichen Gitterkante abgeleitet. Gleiches Gitter
        # heisst gleiche Schrittzahl heisst gleiches Ergebnis - genau die
        # Driftfreiheit, um die es bei der ganzen Umstellung geht.
        gitter = getattr(self, "_wetter_rechen_size", None)
        if gitter:
            stufe = max(1, min(6, int(round(np.log2(max(gitter, 32) / 32.0))) + 1))
            return step_mapping[stufe]

        return step_mapping.get(lod_level, 60)

    def _prepare_input_data(self, heightmap_combined: np.ndarray, shadowmap: np.ndarray,
                           target_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpoliert Input-Data auf Target-Size mit bicubic Interpolation

        Erhält Pattern-Preservation bei Auflösungs-Verdopplung
        """
        # Heightmap interpolieren falls nötig
        if heightmap_combined.shape[0] != target_size:
            heightmap = self._interpolate_2d_bicubic(heightmap_combined, target_size)
        else:
            heightmap = heightmap_combined.copy()

        # DIE ATMOSPHAERE STEHT AUF DER WASSEROBERFLAECHE, NICHT AUF DEM
        # MEERESBODEN.
        #
        # Seit der Weltkarte kann die Heightmap negativ werden. Jede
        # hoehenabhaengige Groesse - Temperaturgradient, Luftdruck,
        # Steigungsregen - rechnete damit unter Wasser weiter, als waere dort
        # Luft. Gemessen am 2026-08-05 ueber See: r(Hoehe, Temperatur) = +0.44
        # mit +1.66 Grad je 100 m TIEFE. Je tiefer der Meeresboden, desto
        # waermer die Luft darueber - der Median lag ueber See bei 9.2 Grad
        # gegen 7.0 an Land, obwohl das Meer auf Hoehe 0 liegt.
        #
        # Physikalisch ist die Sache eindeutig: die Grenzflaeche zur Atmosphaere
        # ist der Meeresspiegel. Alles darunter ist Wasser und hat mit der Luft
        # nichts zu tun. Deshalb wird hier auf 0 geklemmt - fuer das Wetter,
        # NICHT fuer das Gelaende selbst, das seine Tiefen behaelt.
        heightmap = np.maximum(heightmap, 0.0)

        # Shadowmap immer interpolieren (kann von anderem LOD kommen)
        if shadowmap.shape[0] != target_size:
            # Shadowmap ist 3D (H,W,angles) - jeden Kanal separat interpolieren
            if len(shadowmap.shape) == 3:
                interpolated_shadow = np.zeros((target_size, target_size, shadowmap.shape[2]),
                                             dtype=np.float32)
                for angle_idx in range(shadowmap.shape[2]):
                    interpolated_shadow[:, :, angle_idx] = self._interpolate_2d_bicubic(
                        shadowmap[:, :, angle_idx], target_size)
                shadowmap_interp = interpolated_shadow
            else:
                shadowmap_interp = self._interpolate_2d_bicubic(shadowmap, target_size)
        else:
            shadowmap_interp = shadowmap.copy()

        return heightmap, shadowmap_interp

    # =========================================================================
    # DAS FESTGELEGTE TEMPERATURFELD (2026-08-07)
    # =========================================================================

    def _klimafeld(self, lod_level: int, ebene: int, vorgabe: float):
        """Eine Ebene von klima_map, auf das Wettergitter gebracht."""
        klima = self.data_lod_manager.get_calculator_output(
            "terrain.redistribution", "klima_map", lod_level)
        if klima is None:
            return None
        feld = np.asarray(klima)[ebene]
        ziel = getattr(self, "_wetter_rechen_size", None) or feld.shape[0]
        if feld.shape[0] != ziel:
            feld = self._interpolate_2d_bicubic(feld.astype(np.float32), ziel)
        return feld.astype(np.float64)

    # Luv/Lee-Kontrast am Gebirge fuer Wind (docs/archiv/2026-07-29_SPEZIFIKATION.md §3.5): "1.5-2x -
    # deutlich schwaecher als der 3.6x bei Niederschlag, Wind bremst sich am
    # Hang, staut sich aber nicht wie Feuchte". WIND_LUV_STAERKE so gewaehlt,
    # dass volle Luv- gegen volle Lee-Seite (tanh -> +-1) genau das
    # Kontrastverhaeltnis (1+s)/(1-s) = 1.74 trifft, in der Mitte des
    # Zielbereichs. LUV_BEZUGSHANG wiederverwendet aus dem Niederschlag-
    # Pendant (niederschlagsfeld_festgelegt) - dieselbe Frage ("wie steil
    # zaehlt als steil"), keine precip-spezifische Physik.
    WIND_LUV_STAERKE = 0.27

    def _wind_luv_lee_faktor(self, heightmap: np.ndarray, richtung_grad: float) -> np.ndarray:
        """
        Multiplikatives Luv/Lee-Feld fuer Wind, exakt dasselbe Prinzip wie
        `niederschlagsfeld_festgelegt`s `luv`-Term (dortiger Docstring fuer
        die Herleitung von `anstieg`) - nur mit einer viel schwaecheren
        Staerke (siehe WIND_LUV_STAERKE oben) und OHNE den kumulierten
        Regenschatten-Term (Wind bremst sich am Hang lokal, staut sich aber
        nicht wie Feuchte kammweit auf).

        Mittelwert des Feldes liegt nahe 1.0 (tanh ist punktsymmetrisch) -
        der nachfolgende Regionsmittel-Abgleich (`_wind_regional_faktor`)
        korrigiert einen etwaigen Rest ohnehin, die REGIONALE ZIELGESCHWINDIGKEIT
        bleibt also unabhaengig von diesem Term garantiert.
        """
        H = np.asarray(heightmap, dtype=np.float64)
        size = H.shape[0]
        mpp = (float(self.data_lod_manager.get_map_distance_km()) * 1000.0 / size)
        rad = np.radians(float(richtung_grad))
        wx, wy = -np.sin(rad), -np.cos(rad)
        gy, gx = np.gradient(H, mpp)
        anstieg = gx * wx + gy * wy
        return (1.0 + self.WIND_LUV_STAERKE * np.tanh(anstieg / LUV_BEZUGSHANG)).astype(np.float32)

    def _wind_regional_faktor(self, wind_map: np.ndarray, lod_level: int) -> Optional[np.ndarray]:
        """
        Windgeschwindigkeit JE REGION auf `wind_ziel_map` normieren
        (docs/archiv/2026-07-29_SPEZIFIKATION.md §3.5, core/terrain_weltkarte.py REGIONEN.
        wind_mittel_ms) - Richtung bleibt unveraendert, nur die LAENGE des
        Vektors wird skaliert. Exakt dasselbe Prinzip wie
        niederschlagsfeld_festgelegt's "direkt auf den Zielwert normieren":
        die interne Simulation liefert die lokale STRUKTUR (Boeen an Graten,
        Kanalisierung in Taelern - das bleibt erhalten, weil nur eine
        Konstante je Region multipliziert wird), das REGIONALE NIVEAU kommt
        von hier.

        WARUM NICHT ALS INTERNER MULTIPLIKATOR AUF DEN DRUCKGRADIENTEN.
        Erster Versuch: `wind_speed_factor` raeumlich variieren, an der einen
        Stelle, wo es die Druckgradient-Kraft treibt. Gemessen (Nevadin-
        Zielfaktor 0.16 gegen Kuesten-Zielfaktor 0.71, identische Karte):
        1.871 m/s gegen 1.872 m/s - kein messbarer Unterschied. Der
        Druckgradient ist nur EINER von mehreren additiven Antrieben
        (Terrain-Ablenkung/-Speedup, thermische Konvektion, Anfangs-Rauschen)
        und dominiert die Endgeschwindigkeit nicht. Die Normierung hier wirkt
        dagegen GARANTIERT, weil sie das Endergebnis direkt skaliert statt
        einen von mehreren Eingangstermen.

        Rueckgabe: (H,W)-Multiplikatorfeld, auf die WIND-VEKTORLAENGE anzuwenden
        (Richtung bleibt), oder None (alter Nicht-Weltkarten-Pfad ohne
        `wind_ziel_map`/`region_map` - Aufrufer laesst wind_map dann
        unveraendert).
        """
        ziel = self.data_lod_manager.get_calculator_output(
            "terrain.redistribution", "wind_ziel_map", lod_level)
        region_map = self.data_lod_manager.get_calculator_output(
            "terrain.redistribution", "region_map", lod_level)
        if ziel is None or region_map is None:
            return None

        groesse = wind_map.shape[0]
        ziel = np.asarray(ziel, dtype=np.float32)
        if ziel.shape[0] != groesse:
            ziel = self._interpolate_2d_bicubic(ziel, groesse)
        region_map = np.asarray(region_map)
        if region_map.shape[0] != groesse:
            region_map = self._resize_nearest_labels(region_map, (groesse, groesse))

        geschwindigkeit = np.hypot(wind_map[..., 0], wind_map[..., 1]).astype(np.float64)
        faktor = np.ones((groesse, groesse), dtype=np.float64)
        for i in range(9):
            maske = region_map == i
            if not np.any(maske):
                continue
            mittel_ist = float(geschwindigkeit[maske].mean())
            mittel_ziel = float(ziel[maske].mean())
            if mittel_ist > 1e-6:
                faktor[maske] = mittel_ziel / mittel_ist
        # Geklemmt, damit ein einzelnes windstilles Pixel (mittel_ist nahe 0
        # innerhalb der Region waere unproblematisch, aber ein pathologischer
        # Ausreisser soll den Faktor nicht explodieren lassen) endlich bleibt.
        faktor = np.clip(faktor, 0.1, 10.0)
        return faktor.astype(np.float32)

    @staticmethod
    def jahresgang(zeit_im_jahr: float) -> float:
        """
        Wo im Jahr wir stehen: -1 im Januar, +1 im Juli.

        `zeit_im_jahr` laeuft von 0 (1. Januar) bis 1. Eine reine Kosinuskurve,
        also eine SKALARE Funktion der Zeit - kein Feld, keine Simulation. Das
        ist der Kern der Vereinfachung: das raeumliche Muster wird EINMAL
        gerechnet, der Jahresgang ist eine Zahl, die man auf jeden beliebigen
        Zeitpunkt auswerten kann.

        Der Nutzer am 2026-08-07: "wir werden aber im fertigen spiel komplette
        jahre simulieren. also kann einfach eine Min und eine Max temperatur
        sein und eine unregelmaessige kurve die ueber die jahreskurve gelegt
        wird." Die unregelmaessige Ueberlagerung kommt spaeter (Punkt 1.x der
        offenen Liste); vorerst ist der Verlauf glatt, wie ebenfalls vorgegeben
        ("in unserer simulation ueber das jahr gibt es noch keine variation").
        """
        return float(-np.cos(2.0 * np.pi * float(zeit_im_jahr)))

    def _je_region_auf_mittel(self, feld: np.ndarray, lod_level: int,
                              ziel=1.0, maske=None) -> np.ndarray:
        """
        Ein Feld so skalieren, dass sein Mittel JE REGION `ziel` betraegt.

        `ziel` darf eine Zahl sein oder ein dict {Regionsname: Zielwert}. Im
        zweiten Fall trifft JEDE Region ihren eigenen Wert - und zwar PER
        KONSTRUKTION, ohne dass die Eingabewerte gegen die Regionsmischung
        vorkompensiert werden muessen.

        DAS ERSETZT EINE EICHUNG. Beim Niederschlag wurde zuerst der
        EINGABEwert geeicht, bis der Ausgabewert stimmte - zweimal, und beim
        zweiten Mal wurde es schlechter statt besser (2026-08-07). Der Fehler
        war, eine Zwischengroesse zu eichen und eine andere zu messen: die
        Normierung glaettet ueber die Regionsgrenzen, und dabei verschiebt
        sich das Mittel wieder.
        #
        Direkt auf das Ziel zu normieren macht die Eichung ueberfluessig, und
        die eingetragenen Werte bleiben lesbar ("Bergen 2250 mm") statt
        vorkompensiert ("Morobora 79 mm, damit 600 ankommen").

        Damit trifft jede Region ihren Tabellenwert PER KONSTRUKTION, egal was
        das Feld sonst tut - eine Festlegung statt einer Hoffnung. Dieselbe
        Bauform wie bei der Sonnenexposition.

        WEICHE UEBERGAENGE: die Kennzahl wird je Region bestimmt, dann aber als
        FELD geglaettet. Ohne das staende an jeder Regionsgrenze ein Sprung.
        """
        from scipy import ndimage

        f = np.asarray(feld, dtype=np.float64)
        regionen = self.data_lod_manager.get_calculator_output(
            "terrain.redistribution", "region_map", lod_level)
        if regionen is None:
            return f * (ziel / max(float(np.mean(f)), 1e-9))

        R = np.asarray(regionen)
        if R.shape != f.shape:
            R = np.round(self._interpolate_2d_bicubic(
                R.astype(np.float32), f.shape[0])).astype(np.int16)

        from core.terrain_weltkarte import alle_regionen
        namen = [r["name"] for _z, _s, r in alle_regionen()]

        # NUR UEBER DER MASKE MITTELN.
        #
        # Ohne sie geht das MEER mit ein, und das war der eigentliche Fehler
        # (2026-08-07, nach zwei Fehlversuchen mit Eichung und Glaettung): auf
        # See gibt es keinen Regenschatten, dort steht der Rohwert also hoch.
        # Der Regionsmittelwert wurde dadurch zu gross, und die Landflaeche
        # rutschte um bis zu 36 % unter ihren Zielwert. Die Klimatabelle meint
        # aber Landklima - Bergen liegt nicht auf dem Wasser.
        gueltig = np.ones_like(f, dtype=bool) if maske is None else np.asarray(maske)

        mittel_feld = np.full_like(f, max(float(np.mean(f[gueltig])), 1e-9))
        ziel_feld = np.full_like(f, float(ziel) if not isinstance(ziel, dict)
                                 else float(np.mean(list(ziel.values()))))
        for i in range(9):
            g = (R == i) & gueltig
            if g.sum() < 50:
                continue
            # Gemessen wird ueber der Maske, GESETZT wird auf der ganzen
            # Region - sonst bekaeme das Meer keine Korrektur und stuende als
            # Kante an der Kueste.
            ganz = R == i
            mittel_feld[ganz] = max(float(np.mean(f[g])), 1e-9)
            if isinstance(ziel, dict):
                ziel_feld[ganz] = float(ziel.get(namen[i], 1.0))

        # DAS VERHAELTNIS GLAETTEN, NICHT ZAEHLER UND NENNER EINZELN.
        #
        # Getrennt geglaettet nimmt der Nenner am Regionsrand den Mittelwert
        # des NACHBARN an. Bei einem feuchten Nachbarn wird dort durch eine zu
        # grosse Zahl geteilt, und die ganze Region rutscht ab - gemessen lag
        # das Clonagh dadurch 36 % unter seinem Zielwert, obwohl direkt auf
        # das Ziel normiert wurde.
        #
        # Als Verhaeltnis ist es ein reines Korrekturfeld: im Inneren einer
        # Region genau ziel/mittel, am Rand weich zum Nachbarwert
        # ueberblendet. Die Region trifft ihren Wert damit wirklich.
        korrektur = ziel_feld / mittel_feld
        sigma = max(f.shape[0] / 40.0, 1.0)
        korrektur = ndimage.gaussian_filter(korrektur, sigma)
        return f * korrektur

    def _exposition_normiert(self, shadowmap: np.ndarray, lod_level: int) -> np.ndarray:
        """
        Die Sonnenexposition, je Region auf Mittel 0.5 und Spanne 0..1 gebracht.

        WARUM NORMIERT WERDEN MUSS. Gemessen am 2026-08-07 mittelt die rohe
        Exposition ueber Land auf 0.68, nicht auf 0.5. Ohne Normierung laege
        jede Region um mehrere Kelvin neben ihrem Tabellenwert, und die
        Klimatabelle waere nur noch Zierde. Mit ihr trifft jede Region ihr
        Jahresmittel PER KONSTRUKTION - eine Festlegung, kein Regelkreis.

        WEICHE UEBERGAENGE (Nutzervorgabe 2026-08-07: "wir muessen immer
        Regionengrenzen sanft uebergehen lassen"). Die Kennzahlen werden je
        Region bestimmt, dann aber als FELD geglaettet, bevor sie angewandt
        werden. Ohne diese Glaettung staende an jeder Regionsgrenze ein
        Temperatursprung.
        """
        from scipy import ndimage

        e = np.asarray(shadowmap, dtype=np.float64)
        if e.ndim == 3:
            e = e.mean(axis=2)

        regionen = self.data_lod_manager.get_calculator_output(
            "terrain.redistribution", "region_map", lod_level)
        if regionen is None:
            # Ohne Regionen global normieren - besser als gar nicht.
            mitte = float(np.median(e))
            spanne = float(np.percentile(e, 95) - np.percentile(e, 5)) or 1.0
            return np.clip(0.5 + (e - mitte) / spanne, 0.0, 1.0)

        R = np.asarray(regionen)
        if R.shape != e.shape:
            R = np.round(self._interpolate_2d_bicubic(
                R.astype(np.float32), e.shape[0])).astype(np.int16)

        mitte_feld = np.full_like(e, float(np.median(e)))
        spanne_feld = np.full_like(e, 1.0)
        for i in range(9):
            g = R == i
            if g.sum() < 50:
                continue
            werte = e[g]
            # MITTELWERT, nicht Median: nur dann mittelt sich der
            # Expositionsterm ueber die Region exakt zu null, und die Region
            # trifft ihr Jahresmittel. Mit dem Median blieb ein Rest von bis zu
            # 1.5 K stehen (gemessen 2026-08-07).
            mitte_feld[g] = float(np.mean(werte))
            spanne_feld[g] = float(np.percentile(werte, 95)
                                   - np.percentile(werte, 5)) or 1.0

        # Glaetten, damit die Regionsgrenze im Ergebnis nicht als Kante steht.
        sigma = max(e.shape[0] / 40.0, 1.0)
        mitte_feld = ndimage.gaussian_filter(mitte_feld, sigma)
        spanne_feld = np.maximum(ndimage.gaussian_filter(spanne_feld, sigma), 1e-6)

        return np.clip(0.5 + (e - mitte_feld) / spanne_feld, 0.0, 1.0)

    def niederschlagsfeld_festgelegt(self, heightmap: np.ndarray, lod_level: int,
                                     zeit_im_jahr=None):
        """
        Der Jahresniederschlag als FESTLEGUNG statt als Simulationsergebnis.

            P = P_region * luv_faktor * regenschatten

        DREI TEILE, alle geschlossene Form:

        1. `P_region` aus der Klimatabelle (Bergen 2250 mm, Madrid 430) - weich
           ueber die Regionsgrenzen gemischt wie die Temperatur.

        2. `luv_faktor` aus dem oertlichen ANSTIEG IN WINDRICHTUNG. Luft, die
           einen Hang hinaufmuss, kuehlt ab und regnet aus; auf der Leeseite
           bleibt sie trocken. Das ist der Mechanismus, den der Nutzer am
           2026-08-07 beschrieben hat.

        3. `regenschatten` aus dem KUMULIERTEN Anstieg stromaufwaerts. Luft,
           die schon einen Kamm ueberquert hat, ist ausgeregnet.

        WARUM DAS KEIN KREISLAUF IST. Der naheliegende Weg waere, Feuchte
        Schritt fuer Schritt mit dem Wind wandern zu lassen - dann entscheidet
        aber die Schrittzahl ueber das Ergebnis, und genau diesen Drift will
        der Nutzer los. Stattdessen wird je Pixel EINMAL eine feste Strecke
        gegen den Wind zurueckgelegt und der Anstieg aufsummiert: ein Integral
        entlang einer Bahn, kein Einschwingen. Zweimal gerechnet ergibt
        dasselbe, und die Aufloesung aendert daran nichts.

        Rueckgabe: None, wenn klima_map fehlt (alter Pfad ohne Weltkarte).
        """
        p_region = self._klimafeld(lod_level, 2, 800.0)
        if p_region is None:
            return None

        H = np.asarray(heightmap, dtype=np.float64)
        size = H.shape[0]
        mpp = (float(self.data_lod_manager.get_map_distance_km()) * 1000.0
               / size)

        richtung = float(self._current_parameters.get(
            "prevailing_wind_direction", 0.0))
        # Konvention wie beim Sonnenazimut: 0 = Norden, im Uhrzeigersinn.
        # `prevailing_wind_direction` gibt an, WOHER der Wind kommt.
        rad = np.radians(richtung)
        wx, wy = -np.sin(rad), -np.cos(rad)          # wohin die Luft zieht

        gy, gx = np.gradient(H, mpp)
        # Anstieg je Meter Weg in Windrichtung. Positiv heisst: die Luft muss
        # hinauf.
        anstieg = gx * wx + gy * wy

        luv = 1.0 + LUV_STAERKE * np.tanh(anstieg / LUV_BEZUGSHANG)

        # Der kumulierte Aufstieg stromaufwaerts - eine feste Zahl Schritte.
        strecke_px = max(int(REGENSCHATTEN_M / mpp), 1)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
        kumuliert = np.zeros((size, size), dtype=np.float64)
        for schritt in range(1, strecke_px + 1):
            sx = np.clip(xx - wx * schritt, 0, size - 1).astype(np.int32)
            sy = np.clip(yy - wy * schritt, 0, size - 1).astype(np.int32)
            # Nur AUFstiege zaehlen - ein Abstieg macht die Luft nicht feuchter.
            kumuliert += np.maximum(anstieg[sy, sx], 0.0)
        kumuliert *= mpp                                  # in Metern Aufstieg

        regenschatten = np.exp(-kumuliert / REGENSCHATTEN_HOEHE_M)

        # JE REGION AUF MITTEL 1 NORMIEREN.
        #
        # Der Regenschatten kann nur VERRINGERN (exp(-x) <= 1), nie erhoehen -
        # ohne Normierung lag jede Region 45 bis 53 % unter ihrem Tabellenwert
        # (gemessen 2026-08-07). Die Normierung erhaelt das Muster von Luv und
        # Lee vollstaendig und verschiebt nur das Niveau; damit trifft jede
        # Region ihre Jahressumme per Konstruktion.
        # DIREKT AUF DEN ZIELWERT NORMIEREN, nicht auf Mittel 1.
        #
        # Der Regenschatten kann nur VERRINGERN (exp(-x) <= 1), und die
        # Regionsmischung zieht feuchte Regionen zu ihren trockenen Nachbarn
        # hinunter - Skerrheim lag 36 % unter seinem Wert. Beides zusammen
        # laesst sich nicht durch vorkompensierte Eingabewerte auffangen: die
        # Glaettung der Normierung verschiebt das Mittel erneut, und eine
        # Eichung darauf lief in die falsche Richtung (2026-08-07).
        #
        # Hier wird stattdessen das FERTIGE Feld auf die Jahressumme der
        # jeweiligen Region gezogen. Das Muster von Luv und Lee bleibt
        # vollstaendig erhalten; nur das Niveau wird gesetzt.
        from core.terrain_weltkarte import NIEDERSCHLAG_ZIEL
        jahr = self._je_region_auf_mittel(
            p_region * luv * regenschatten, lod_level,
            NIEDERSCHLAG_ZIEL, maske=(H > 0.0))

        # Ueber See gedaempft - siehe SEE_REGEN_ANTEIL.
        jahr = np.where(H > 0.0, jahr, jahr * SEE_REGEN_ANTEIL)

        if zeit_im_jahr is None:
            return jahr.astype(np.float32)

        # DER JAHRESGANG, aus der Kontinentalitaet abgeleitet.
        #
        # +1 heisst voll kontinental (Sommerregen), -1 voll maritim
        # (Winterregen). Die Jahresspanne der Temperatur ist das Mass dafuer,
        # und sie steht schon im Klimafeld - es braucht keine eigene Zahl.
        t_spanne = self._klimafeld(lod_level, 1, 15.0)
        if t_spanne is None:
            return jahr.astype(np.float32)
        kontinental = np.tanh((t_spanne - SAISON_BEZUGSSPANNE) / SAISON_UEBERGANG)
        # Das Meer ist maritim, unabhaengig von der Region daneben.
        kontinental = np.where(H > 0.0, kontinental,
                               np.tanh((SEE_SPANNE - SAISON_BEZUGSSPANNE)
                                       / SAISON_UEBERGANG))

        # MONATSRATE mal Ticklaenge. `jahr` ist die Jahressumme; geteilt durch
        # zwoelf ist es die Monatsrate, und ein Tick umfasst MONATE_JE_TICK
        # davon. Bei 800 mm im Jahr sind das 67 mm je Monat und 133 je
        # Zweimonatstick.
        gang = 1.0 + SAISON_STAERKE * kontinental * self.jahresgang(zeit_im_jahr)
        return (jahr / 12.0 * MONATE_JE_TICK * gang).astype(np.float32)

    def _temperatur_raummuster(self, heightmap, shadowmap, lod_level):
        """
        Das RAEUMLICHE Muster: Sockel und Amplitude je Pixel, zeitunabhaengig.

            T(x, y, t) = sockel(x, y) + amplitude(x, y) * jahresgang(t)

        EINMAL gerechnet und zwischengespeichert. Genau darin besteht die
        Vereinfachung, und ich hatte sie beim ersten Anlauf selbst verfehlt:
        `temperaturfeld_festgelegt` wurde sechsmal aufgerufen und rechnete
        jedes Mal die Expositionsnormierung samt Gaussglaettung neu. Die
        Wetterrechnung wurde dadurch teurer statt billiger - gemessen 12.2 s
        statt 9.9 s bei 256 px.

        Der Jahresgang ist eine SKALARE Funktion; das Muster daneben ist
        konstant. Beides zu trennen ist der ganze Punkt.
        """
        schluessel = (id(heightmap), heightmap.shape, lod_level,
                      getattr(self, "_wetter_rechen_size", None))
        zwischen = getattr(self, "_temperatur_cache", None)
        if zwischen is not None and zwischen[0] == schluessel:
            return zwischen[1]

        t_mittel = self._klimafeld(lod_level, 0, 11.0)
        t_spanne = self._klimafeld(lod_level, 1, 15.0)
        if t_mittel is None or t_spanne is None:
            return None

        H = np.asarray(heightmap, dtype=np.float64)
        land = H > 0.0

        # DIREKTNORMIERUNG STATT VORKOMPENSIERTER EINGABE (2026-08-11,
        # docs/OFFENE_PUNKTE.md 1.10/1.11). `t_mittel`/`t_spanne` kommen aus
        # klima_map, also aus der weich ueber die Regionsgrenzen GEBLENDETEN
        # Fassung von core.terrain_weltkarte.REGIONEN.temp_mittel_m0/
        # temp_spanne - die Regionsmischung zieht jede Region zu ihren
        # Nachbarn hin (Morobora bekam ohne Korrektur 1.8 statt der eigentlich
        # gewollten 3.8, siehe KLIMA_ZIEL). Bisher wurde das durch
        # HANDKALIBRIERTE Eingabewerte kompensiert - "Morobora 1.8, damit am
        # Ende 3.8 ankommen" -, ueber drei Seeds von Hand geeicht und auf
        # einem vierten schon wieder daneben (1.11: 30.9 K statt 29.0 K
        # Jahresspanne).
        #
        # Exakt dasselbe Muster wie beim Niederschlag (1.3,
        # niederschlagsfeld_festgelegt): das FERTIGE, geblendete Feld direkt
        # auf den Zielwert je Region normieren (`_je_region_auf_mittel`)
        # macht die Handeichung ueberfluessig und trifft die Vorgabe PER
        # KONSTRUKTION, unabhaengig vom Blend. Die Eingabewerte in REGIONEN
        # sind seither die LESBAREN, echten Zielwerte (identisch mit
        # KLIMA_ZIEL) statt vorkompensierter Zahlen.
        from core.terrain_weltkarte import KLIMA_ZIEL
        mittel_ziel = {name: werte[0] for name, werte in KLIMA_ZIEL.items()}
        spanne_ziel = {name: werte[1] for name, werte in KLIMA_ZIEL.items()}
        t_mittel = self._je_region_auf_mittel(t_mittel, lod_level, mittel_ziel, maske=land)
        t_spanne = self._je_region_auf_mittel(t_spanne, lod_level, spanne_ziel, maske=land)

        exposition = self._exposition_normiert(shadowmap, lod_level)
        sockel_land = (t_mittel
                       - HOEHENABNAHME_K_PRO_M * np.maximum(H, 0.0)
                       + t_spanne * EXPOSITIONSANTEIL * (exposition - 0.5))

        # Die See: keine Sonne, keine Hoehenabnahme - eine Festlegung, kein
        # Naeherungsverfahren. Genau das loescht den Kuestentemperatursprung
        # von 5.6 K, dessen Ursache seit Wochen offen war: er entstand, weil
        # das Meer als Landflaeche mit Hoehe 0 behandelt wurde.
        size = H.shape[0]
        y_anteil = np.linspace(0.0, 1.0, size)[:, None] * np.ones((1, size))
        sockel_see = (SEE_MITTEL_SUED
                      + (SEE_MITTEL_NORD - SEE_MITTEL_SUED) * y_anteil
                      + self._stroemungsrauschen(size) * SEE_STROEMUNG_K)

        sockel = np.where(land, sockel_land, sockel_see)
        amplitude = np.where(land, 0.5 * t_spanne, 0.5 * SEE_SPANNE)

        self._temperatur_cache = (schluessel, (sockel, amplitude))
        return sockel, amplitude

    def temperaturfeld_festgelegt(self, heightmap: np.ndarray,
                                  shadowmap: np.ndarray, lod_level: int,
                                  zeit_im_jahr: float = 0.5):
        """
        Das Temperaturfeld als FESTLEGUNG statt als Simulationsergebnis.

            T = T_mittel(Region) - HOEHENABNAHME * hoehe
                + Spanne/2 * jahresgang(t)
                + Spanne * EXPOSITIONSANTEIL * (exposition - 0.5)

        Auf See ohne Sonne und ohne Hoehenabnahme, dafuer mit einem groben
        Stroemungsrauschen - Nutzervorgabe 2026-08-07.

        Rueckgabe: None, wenn klima_map fehlt (alter Pfad ohne Weltkarte).
        """
        muster = self._temperatur_raummuster(heightmap, shadowmap, lod_level)
        if muster is None:
            return None
        sockel, amplitude = muster
        return (sockel + amplitude * self.jahresgang(zeit_im_jahr)).astype(np.float32)

    def _stroemungsrauschen(self, size: int) -> np.ndarray:
        """
        Grobes Rauschen fuer die Meeresstroemungen, aus dem Kartenseed.

        Nutzer: "dabei wird eine noisemap verwendet die meeresstroemungen etwas
        darstellt", mit "geringer varianz". Sehr grosse Wellenlaenge - eine
        Stroemung ist ein Gebilde von vielen Kilometern, kein Fleckenmuster.
        """
        schluessel = (size, int(self.map_seed))
        zwischen = getattr(self, "_stroemung_cache", None)
        if zwischen is not None and zwischen[0] == schluessel:
            return zwischen[1]

        rng = np.random.default_rng(int(self.map_seed) ^ 0x5EA1)
        grob = rng.normal(size=(4, 4))
        feld = self._interpolate_2d_bicubic(grob.astype(np.float32), size)
        feld = feld / (float(np.abs(feld).max()) or 1.0)
        self._stroemung_cache = (schluessel, feld.astype(np.float64))
        return self._stroemung_cache[1]

    def _calculate_temperature_field(self, heightmap: np.ndarray, shadowmap: np.ndarray,
                                   parameters: Dict[str, Any], target_size: int,
                                   sun_angles: Optional[list] = None) -> np.ndarray:
        """
        Temperaturfeld-Berechnung mit GPU-Shader-Integration und 3-stufigem Fallback

        Integriert Altitude-Cooling, Boden-Zieltemperatur (T_boden), Latitude-Gradient
        und Noise-Variation. sun_angles: siehe _weighted_solar_exposure()-Docstring.
        """
        try:
            # GPU-Shader-Request (Optimal)
            if self.shader_manager:
                shader_request = {
                    'operation_type': 'temperature_calculation',
                    'input_data': {
                        'heightmap': heightmap,
                        'shadowmap': shadowmap
                    },
                    'parameters': {
                        'air_temp_entry': parameters['air_temp_entry'],
                        'ground_temp_baseline': parameters['ground_temp_baseline'],
                        'altitude_cooling': parameters['altitude_cooling'],
                        'map_seed': self.map_seed
                    },
                    'lod_level': target_size
                }

                result = self.shader_manager.request_temperature_calculation(shader_request)

                if result.success:
                    self.logger.debug("Temperature calculation completed on GPU")
                    return result.temperature_field
                else:
                    self.logger.warning(f"GPU temperature calculation failed: {result.error}")

        except Exception as e:
            self.logger.warning(f"GPU shader request failed: {e}")

        # CPU-Fallback (Gut)
        try:
            return self._calculate_temperature_cpu_optimized(heightmap, shadowmap, parameters, target_size,
                                                               sun_angles=sun_angles)
        except Exception as e:
            self.logger.error(f"CPU temperature calculation failed: {e}")

        # Simple-Fallback (Minimal)
        return self._calculate_temperature_simple(heightmap, parameters)

    def _calculate_temperature_cpu_optimized(self, heightmap: np.ndarray, shadowmap: np.ndarray,
                                           parameters: Dict[str, Any], target_size: int,
                                           sun_angles: Optional[list] = None) -> np.ndarray:
        """
        CPU-optimierte Temperatur-Berechnung mit vectorized NumPy-Operations
        """
        # Basis-Temperatur
        temp_map = np.full(heightmap.shape, parameters['air_temp_entry'], dtype=np.float32)

        # Altitude-Cooling (vectorized)
        altitude_cooling_rate = parameters['altitude_cooling'] / 1000.0  # °C pro Meter (Parameter ist °C/km)
        temp_map -= heightmap * altitude_cooling_rate

        # Boden-Zieltemperatur (T_boden, löst den alten additiven solar_power-
        # Term ab) - dieser Pfad hat kein t_real/keinen Zeitschritt-Loop
        # (Erstlauf-Seed ohne LOD-Vererbung), bleibt daher wie bisher eine
        # einmalige additive Störung auf die Basistemperatur, nur mit dem
        # neuen T_min/T_max-Wertebereich statt der alten solar_power-Spanne.
        shadow_weighted = self._weighted_solar_exposure(shadowmap, sun_angles=sun_angles)
        ground_temp_baseline = parameters['ground_temp_baseline']
        effective_spread = GROUND_TEMP_SPREAD * parameters.get('sun_relevance_factor', 1.0)
        # Dieselbe Zentrierung wie im gekoppelten Pfad oben - beide Pfade
        # muessen dieselbe Bodentemperatur liefern, sonst haengt das Ergebnis
        # daran, ob die 3-Schicht-Simulation durchlief oder auf den
        # Einzelschicht-Fallback zurueckfiel.
        ground_temp_target = (ground_temp_baseline + effective_spread
                              * (shadow_weighted - float(np.mean(shadow_weighted))))
        solar_effect = ground_temp_target - parameters['air_temp_entry']
        temp_map += solar_effect

        # KEIN eigener Breitengrad-Gradient mehr hier - parameters['air_temp_entry']
        # kommt jetzt bereits aus der Breitengrad×Monat-Klimatologie
        # (_climate_baseline, siehe _generate_seasonal_parameters). Ein
        # zusätzlicher, rein bildzeilen-basierter Gradient hier würde den
        # Breitengrad-Effekt doppelt und mit einer zweiten, unabhängigen
        # (und physikalisch falschen, weil an der Bild-Y-Achse statt am
        # echten Breitengrad hängenden) Formel zählen.

        # Atmospheric Noise-Variation
        noise_variation = self._generate_atmospheric_noise(
            heightmap.shape, target_size, parameters.get('month_index', 0))
        temp_map += noise_variation

        return temp_map

    def _calculate_temperature_simple(self, heightmap: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Simple-Fallback: Basic Temperature ohne komplexe Effekte
        """
        temp_map = np.full(heightmap.shape, parameters['air_temp_entry'], dtype=np.float32)

        # Nur Altitude-Cooling
        altitude_cooling_rate = parameters['altitude_cooling'] / 1000.0  # Parameter ist °C/km
        temp_map -= heightmap * altitude_cooling_rate

        return temp_map

    def _simulate_wind_field_cfd(self, heightmap: np.ndarray, temp_map: np.ndarray,
                                shadowmap: np.ndarray, parameters: Dict[str, Any],
                                target_size: int, cfd_iterations: int) -> np.ndarray:
        """
        CFD-basierte Wind-Simulation mit Navier-Stokes-Gleichungen und GPU-Acceleration

        Implementiert vollständige CFD-Pipeline mit Pressure-Gradients, Terrain-Deflection
        und Thermal-Convection über multiple Iterationen
        """
        try:
            # GPU-Shader-Request (Optimal)
            if self.shader_manager:
                shader_request = {
                    'operation_type': 'wind_field_cfd',
                    'input_data': {
                        'heightmap': heightmap,
                        'temp_map': temp_map,
                        'shadowmap': shadowmap
                    },
                    'parameters': {
                        'wind_speed_factor': parameters['wind_speed_factor'],
                        'terrain_factor': parameters['terrain_factor'],
                        'thermic_effect': parameters['thermic_effect'],
                        'prevailing_wind_direction': parameters.get('prevailing_wind_direction', 0.0),
                        'month_index': parameters.get('month_index', 0),
                        'cfd_iterations': cfd_iterations,
                        'map_seed': self.map_seed
                    },
                    'lod_level': target_size
                }

                result = self.shader_manager.request_wind_field_cfd(shader_request)

                if result.success:
                    self.logger.debug(f"Wind CFD completed on GPU with {cfd_iterations} iterations")
                    return result.wind_field
                else:
                    self.logger.warning(f"GPU wind CFD failed: {result.error}")

        except Exception as e:
            self.logger.warning(f"GPU wind CFD request failed: {e}")

        # CPU-Fallback (Gut)
        try:
            return self._simulate_wind_field_cpu_cfd(heightmap, temp_map, shadowmap, parameters,
                                                   cfd_iterations)
        except Exception as e:
            self.logger.error(f"CPU wind CFD failed: {e}")

        # Simple-Fallback (Minimal)
        return self._simulate_wind_field_simple(heightmap, parameters)

    def _build_directional_pressure_field(self, height: int, width: int,
                                           wind_direction_deg: float) -> np.ndarray:
        """
        Funktionsweise: Lineares Druckgefälle entlang einer beliebigen
        Windrichtung - ersetzt das früher hartcodierte West-Ost-Gefälle.
        wind_direction_deg ist die HERKUNFTSRICHTUNG (siehe
        WEATHER.PREVAILING_WIND_DIRECTION-Beschreibung): 0°=Wind aus Westen
        (weht nach Osten, +x), 90°=aus Süden (weht nach Norden, +y - Zeile
        height-1 = Norden, siehe core/terrain_generator.py._raycast_shadow_cpu()
        für dieselbe Array-Konvention). dx=cos(theta), dy=sin(theta) ergeben
        direkt die Weht-nach-Richtung, da diese Herkunfts-Konvention exakt
        180° gegenüber der alten Blas-Richtungs-Formel liegt und sich beide
        Vorzeichen dadurch aufheben (Bei wind_direction_deg=0 identisch zur
        alten Formel, Projektion s entspricht dann x_coords).
        Aufgabe: Treibt die Richtung, aus der der vorherrschende Wind weht,
        in die initiale Druckfeld-Konstruktion der CFD-Simulation ein - lokale
        Abweichung entsteht weiterhin über die bestehende Terrain-Ablenkung.
        """
        theta = np.radians(wind_direction_deg)
        dx, dy = np.cos(theta), np.sin(theta)
        y_idx, x_idx = np.mgrid[0:height, 0:width]
        s = x_idx * dx + y_idx * dy
        s_min, s_max = s.min(), s.max()
        s_range = s_max - s_min
        s_norm = (s - s_min) / s_range if s_range > 1e-9 else np.zeros_like(s, dtype=np.float32)
        return (1.0 - s_norm * 0.3).astype(np.float32)

    def _simulate_wind_field_cpu_cfd(self, heightmap: np.ndarray, temp_map: np.ndarray,
                                   shadowmap: np.ndarray, parameters: Dict[str, Any],
                                   cfd_iterations: int) -> np.ndarray:
        """
        CPU-optimierte CFD-Simulation mit NumPy-Vectorization

        Implementiert vereinfachte Navier-Stokes mit Advection, Pressure-Gradients und Diffusion
        """
        height, width = heightmap.shape

        # Initialisierung
        wind_field = np.zeros((height, width, 2), dtype=np.float32)

        # Slopemap aus Heightmap berechnen (vectorized)
        slopemap = self._calculate_slopes_vectorized(heightmap)
        # Grat-/Canyon-Speedup-Term (siehe [[project-wind-ridge-speedup]]) -
        # nur der Speedup, keine Rauigkeits-Dämpfung hier: dieser Pfad ist
        # der seltene Einzelschicht-Fallback (nur bei Exception im primären
        # 3-Schicht-Loop), der schon bisher bewusst schlanker gehalten wird
        # (keine Advektion/Vorticity/Schichten) - Rauigkeit bräuchte
        # zusätzlich lod_level/data_lod_manager-Zugriff, den diese Methode
        # aktuell nicht entgegennimmt.
        curvature_norm = self._calculate_curvature_normalized(heightmap)

        # Initiales Druckfeld entlang der vorherrschenden Windrichtung (mit Noise)
        wind_direction_deg = parameters.get('prevailing_wind_direction', 0.0)
        pressure_field = self._build_directional_pressure_field(height, width, wind_direction_deg)

        # Noise-Modulation
        pressure_noise = self._generate_pressure_noise((height, width), parameters.get('month_index', 0))
        pressure_field += pressure_noise * 0.15

        # Druck-Terrain-Kopplung: hohe Punkte senken lokal den effektiven
        # Druck (grobe Orographie-Näherung - Wind beschleunigt an/über
        # Erhebungen, sammelt sich in Tälern). Anders als die bestehende
        # additive Terrain-Ablenkung weiter unten (die nur EINMALIG pro
        # Iteration auf wind_field draufaddiert wird, NACH der Gradienten-
        # Berechnung) fließt dieser Term VOR der Gradientenberechnung ins
        # Druckfeld selbst ein - dadurch bleibt die terrain-geprägte Struktur
        # über die Diffusion/Kontinuitäts-Korrektur der gesamten
        # Iterationskette hinweg erhalten, statt nur ein einmaliger Nudge auf
        # ein sonst uniformes Feld zu sein. Koeffizient 0.2 ist ein erster
        # Richtwert (kleiner als der bestehende additive Term mit 0.5, da
        # dieser Effekt jetzt über ALLE Iterationen wirkt statt einmalig -
        # sonst Gefahr von Überkompensation/Instabilität in
        # _apply_continuity_correction).
        height_range = heightmap.max() - heightmap.min()
        height_normalized = (heightmap - heightmap.min()) / height_range if height_range > 1e-6 \
            else np.zeros_like(heightmap, dtype=np.float32)
        terrain_pressure_term = height_normalized * parameters['terrain_factor'] * 0.2

        # CFD-Iterationen
        for iteration in range(cfd_iterations):
            # Progress-Update für längere CFD-Simulationen
            if iteration % max(1, cfd_iterations // 5) == 0:
                progress = 30 + (iteration / cfd_iterations) * 30
                self._update_progress("Wind CFD", int(progress),
                                    f"CFD iteration {iteration + 1}/{cfd_iterations}")

            # Druckgradienten berechnen (vectorized) - Terrain-Kopplung fließt
            # hier ein, VOR der Gradientenberechnung (siehe Kommentar oben).
            pressure_field_iter = pressure_field - terrain_pressure_term
            pressure_grad_x = np.zeros_like(pressure_field)
            pressure_grad_y = np.zeros_like(pressure_field)

            pressure_grad_x[:, 1:-1] = (pressure_field_iter[:, 2:] - pressure_field_iter[:, :-2]) * 0.5
            pressure_grad_y[1:-1, :] = (pressure_field_iter[2:, :] - pressure_field_iter[:-2, :]) * 0.5

            # Wind aus Druckgradienten
            wind_field[:, :, 0] = -pressure_grad_x * parameters['wind_speed_factor'] * 10.0
            wind_field[:, :, 1] = -pressure_grad_y * parameters['wind_speed_factor'] * 10.0

            # Terrain-Ablenkung (vectorized)
            terrain_factor = parameters['terrain_factor'] * 0.5
            wind_field[:, :, 0] += slopemap[:, :, 1] * terrain_factor  # Slope Y -> Wind X
            wind_field[:, :, 1] -= slopemap[:, :, 0] * terrain_factor  # Slope X -> Wind Y

            # Grat-/Canyon-Speedup (siehe [[project-wind-ridge-speedup]])
            ridge_factor = 1.0 + _RIDGE_SPEEDUP_STRENGTH * (-curvature_norm) * parameters['terrain_factor']
            wind_field[:, :, 0] *= ridge_factor
            wind_field[:, :, 1] *= ridge_factor

            # Thermal-Convection
            self._apply_thermal_convection(wind_field, temp_map, shadowmap, parameters)

            # Wind-Diffusion für Stabilität (simplified)
            wind_field = self._apply_wind_diffusion(wind_field, 0.1)

            # Kontinuitäts-Correction für Massenerhaltung
            self._apply_continuity_correction(wind_field)

        return wind_field

    def _simulate_wind_field_simple(self, heightmap: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Simple-Fallback: Basic Wind-Field ohne CFD-Komplexität. Auch der
        Initial-Seed für die 3-Schicht-CFD (siehe base_wind in
        _run_coupled_atmosphere_simulation). wind_direction_deg ist die
        Herkunftsrichtung (siehe _build_directional_pressure_field-Docstring).
        """
        height, width = heightmap.shape
        wind_field = np.zeros((height, width, 2), dtype=np.float32)

        # Konstanter Wind entlang der vorherrschenden Windrichtung
        wind_direction_deg = parameters.get('prevailing_wind_direction', 0.0)
        theta = np.radians(wind_direction_deg)
        base_wind_speed = parameters['wind_speed_factor'] * 5.0
        wind_field[:, :, 0] = base_wind_speed * np.cos(theta)
        wind_field[:, :, 1] = base_wind_speed * np.sin(theta)

        # Basic Terrain-Ablenkung
        slopemap = self._calculate_slopes_vectorized(heightmap)
        terrain_factor = parameters['terrain_factor'] * 0.2
        wind_field[:, :, 0] += slopemap[:, :, 1] * terrain_factor
        wind_field[:, :, 1] -= slopemap[:, :, 0] * terrain_factor

        return wind_field

    def _calculate_atmospheric_moisture(self, heightmap: np.ndarray, temp_map: np.ndarray,
                                      wind_map: np.ndarray, parameters: Dict[str, Any],
                                      hardness_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Atmospheric-Moisture-Calculation mit Evaporation und Transport

        Implementiert Magnus-Formel für Sättigungsdampfdruck und Wind-enhanced Evaporation

        hardness_map: optional, nur vom CPU-Fallback für die hardness-gewichtete
        Diffusion genutzt (siehe _apply_humidity_diffusion) - der GPU-Pfad nutzt
        weiterhin humidityDiffusion.comp's einfachen Box-Blur ohne Hardness-
        Gewichtung (bekannte Lücke, analog zur bewusst zurückgestellten GPU-
        Watershed-Parität für #42 - eigener Folge-Task, kein Blocker hier).
        """
        try:
            # GPU-Shader-Request (Optimal)
            if self.shader_manager:
                shader_request = {
                    'operation_type': 'atmospheric_moisture',
                    'input_data': {
                        'heightmap': heightmap,
                        'temp_map': temp_map,
                        'wind_map': wind_map
                    },
                    'parameters': parameters,
                    'lod_level': heightmap.shape[0]
                }

                result = self.shader_manager.request_atmospheric_moisture(shader_request)

                if result.success:
                    return result.humidity_field

        except Exception as e:
            self.logger.warning(f"GPU moisture calculation failed: {e}")

        # CPU-Fallback (Gut)
        return self._calculate_atmospheric_moisture_cpu(heightmap, temp_map, wind_map, parameters, hardness_map)

    def _calculate_atmospheric_moisture_cpu(self, heightmap: np.ndarray, temp_map: np.ndarray,
                                          wind_map: np.ndarray, parameters: Dict[str, Any],
                                          hardness_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        CPU-optimierte Atmospheric-Moisture mit vectorized Operations
        """
        height, width = temp_map.shape

        # Initiale Soil-Moisture (vereinfacht - 50% überall)
        soil_moisture = np.full((height, width), 50.0, dtype=np.float32)

        # Evaporation-Rate basierend auf Temperatur (Magnus-Formel vereinfacht)
        # Sättigungsdampfdruck steigt exponentiell mit Temperatur
        temp_celsius = np.maximum(-50, np.minimum(60, temp_map))  # Clamp temperature
        saturation_vapor_pressure = 6.112 * np.exp(17.67 * temp_celsius / (temp_celsius + 243.5))

        # Wind-Speed für Enhanced-Evaporation
        wind_speed = np.sqrt(wind_map[:, :, 0]**2 + wind_map[:, :, 1]**2)
        wind_factor = np.minimum(2.0, wind_speed / 5.0)  # Cap bei 2x Enhancement

        # Evaporation-Rate
        evaporation_rate = (soil_moisture / 100.0) * (saturation_vapor_pressure / 100.0) * (1.0 + wind_factor)

        # Initiale Humidity aus Evaporation. Skalierungsfaktor kalibriert gegen
        # rho_max = 5*exp(0.06*T) aus _calculate_precipitation_cpu (Magnus-Formel
        # Sättigungsdampfdichte, Größenordnung ~5-17 bei -3..20°C): mit dem alten
        # Faktor 10 lag humid_map bei ~0.3-1.5 - relative_humidity = humid_map/rho_max
        # blieb dadurch IMMER unter ~0.09, "oversaturation" (>1.0, einziger Auslöser
        # für Kondensations-Niederschlag) konnte nie auftreten, unabhängig von
        # Temperatur/Wind/Terrain. precip_map bestand dadurch nur noch aus dem
        # (kleinen) orographischen Anteil. Faktor empirisch so gewählt, dass bei
        # Default-Parametern ein Mix aus über-/untersättigten Gebieten entsteht statt
        # eines globalen Alles-oder-Nichts-Zustands (siehe [[project-precip-humidity-calibration]]).
        humid_map = evaporation_rate * 140.0

        # Moisture-Transport (vereinfacht, 3 Iterationen)
        for _ in range(3):
            humid_map = self._transport_moisture_simple(humid_map, wind_map, dt=0.5)

        # Diffusion für smooth Distribution
        humid_map = self._apply_humidity_diffusion(humid_map, iterations=2, hardness_map=hardness_map)

        return humid_map

    def _calculate_precipitation_system(self, humid_map: np.ndarray, temp_map: np.ndarray,
                                      wind_map: np.ndarray, heightmap: np.ndarray,
                                      parameters: Dict[str, Any]) -> np.ndarray:
        """
        Precipitation-System mit Orographic-Effects und Condensation

        Implementiert Luv-/Lee-Effekte und Magnus-Formel für Condensation-Thresholds
        """
        try:
            # GPU-Shader-Request (Optimal)
            if self.shader_manager:
                shader_request = {
                    'operation_type': 'precipitation_calculation',
                    'input_data': {
                        'humid_map': humid_map,
                        'temp_map': temp_map,
                        'wind_map': wind_map,
                        'heightmap': heightmap
                    },
                    'parameters': parameters,
                    'lod_level': heightmap.shape[0]
                }

                result = self.shader_manager.request_precipitation_calculation(shader_request)

                if result.success:
                    return result.precipitation_field

        except Exception as e:
            self.logger.warning(f"GPU precipitation calculation failed: {e}")

        # CPU-Fallback (Gut)
        return self._calculate_precipitation_cpu(humid_map, temp_map, wind_map, heightmap, parameters)

    def _calculate_precipitation_cpu(self, humid_map: np.ndarray, temp_map: np.ndarray,
                                    wind_map: np.ndarray, heightmap: np.ndarray,
                                    parameters: Dict[str, Any]) -> np.ndarray:
        """
        CPU-optimierte Precipitation mit Orographic-Effects und Condensation-Logic
        """
        height, width = humid_map.shape
        precip_map = np.zeros((height, width), dtype=np.float32)

        # Slopemap für Orographic-Effects
        slopemap = self._calculate_slopes_vectorized(heightmap)

        # 1. Orographic Precipitation (Luv-/Lee-Effekte)
        wind_speed = np.sqrt(wind_map[:, :, 0]**2 + wind_map[:, :, 1]**2)

        # Wind-Slope-Alignment für Luv-Identifikation
        wind_norm_x = np.where(wind_speed > 0.1, wind_map[:, :, 0] / wind_speed, 0)
        wind_norm_y = np.where(wind_speed > 0.1, wind_map[:, :, 1] / wind_speed, 0)

        wind_slope_alignment = (wind_norm_x * slopemap[:, :, 0] +
                               wind_norm_y * slopemap[:, :, 1])

        # Orographic Enhancement an Luvhängen
        orographic_factor = np.maximum(0, wind_slope_alignment) * wind_speed * 0.3
        oro_precip = humid_map * orographic_factor * 0.05

        # 2. Condensation Precipitation (Magnus-Formel)
        # Sättigungsdampfdichte: rho_max = 5*exp(0.06*T)
        temp_celsius = np.maximum(-40, np.minimum(50, temp_map))
        rho_max = 5.0 * np.exp(0.06 * temp_celsius)

        # Relative Humidity
        relative_humidity = np.where(rho_max > 0, humid_map / rho_max, 0)

        # Precipitation bei Übersättigung (> 1.0)
        oversaturation = np.maximum(0, relative_humidity - 1.0)
        condensation_precip = oversaturation * rho_max * 0.6

        # 3. Kombiniere Precipitation-Sources
        precip_map = oro_precip + condensation_precip

        # Physical Limits
        precip_map = np.maximum(0, precip_map)  # Kein negativer Niederschlag
        precip_map = np.minimum(500, precip_map)  # Maximum 500 gH2O/m²

        return precip_map

    def _create_weather_data(self, wind_map: np.ndarray, temp_map: np.ndarray,
                           precip_map: np.ndarray, humid_map: np.ndarray,
                           lod_level: int, target_size: int,
                           parameters: Dict[str, Any],
                           wind_map_monthly=None, temp_map_monthly=None,
                           precip_map_monthly=None, humid_map_monthly=None,
                           temp_map_layers=None, wind_map_layers=None, humid_map_layers=None,
                           temp_map_layers_monthly=None, wind_map_layers_monthly=None,
                           humid_map_layers_monthly=None) -> WeatherData:
        """
        Erstellt WeatherData-Objekt mit vollständiger LOD-Integration und Validation
        """
        weather_data = WeatherData()
        weather_data.wind_map = wind_map
        weather_data.temp_map = temp_map
        weather_data.precip_map = precip_map
        weather_data.humid_map = humid_map
        weather_data.wind_map_monthly = wind_map_monthly
        weather_data.temp_map_monthly = temp_map_monthly
        weather_data.precip_map_monthly = precip_map_monthly
        weather_data.humid_map_monthly = humid_map_monthly
        # 3-Schicht-Atmosphäre, rein additiv (siehe [[project-3layer-wind-cfd]]) -
        # None, falls der gekoppelte Loop diese Runde nicht lief.
        weather_data.temp_map_layers = temp_map_layers
        weather_data.wind_map_layers = wind_map_layers
        weather_data.humid_map_layers = humid_map_layers
        weather_data.temp_map_layers_monthly = temp_map_layers_monthly
        weather_data.wind_map_layers_monthly = wind_map_layers_monthly
        weather_data.humid_map_layers_monthly = humid_map_layers_monthly
        weather_data.lod_level = lod_level
        weather_data.actual_size = target_size

        # Parameter-Hash für Cache-Management
        weather_data.parameter_hash = self._calculate_parameter_hash(parameters)

        # Validity-State setzen
        weather_data.validity_state = {
            "valid": True,
            "dependencies_satisfied": True,
            "lod_level": lod_level,
            "last_generation": "weather_system"
        }

        # Data-Quality-Validation
        self._validate_weather_output(weather_data)

        return weather_data

    def _validate_weather_output(self, weather_data: WeatherData):
        """
        Validiert Weather-Output für Data-Integrity und Physical-Plausibility
        """
        # NaN-Detection
        for field_name, field_data in [
            ("wind_map", weather_data.wind_map),
            ("temp_map", weather_data.temp_map),
            ("precip_map", weather_data.precip_map),
            ("humid_map", weather_data.humid_map)
        ]:
            if np.any(np.isnan(field_data)) or np.any(np.isinf(field_data)):
                self.logger.warning(f"Invalid values detected in {field_name}")
                weather_data.validity_state["valid"] = False

        # Physical-Range-Validation
        if not (-60 <= np.min(weather_data.temp_map) and np.max(weather_data.temp_map) <= 80):
            self.logger.warning("Temperature values outside physical range")

        if not (0 <= np.min(weather_data.precip_map)):
            self.logger.warning("Negative precipitation values detected")

        if not (0 <= np.min(weather_data.humid_map)):
            self.logger.warning("Negative humidity values detected")

        # Wind-Speed-Validation
        wind_speeds = np.sqrt(weather_data.wind_map[:, :, 0]**2 + weather_data.wind_map[:, :, 1]**2)
        if np.max(wind_speeds) > 100:  # > 100 m/s unrealistic
            self.logger.warning("Unrealistic wind speeds detected")

    def _update_performance_stats(self, weather_data: WeatherData, cfd_iterations: int):
        """
        Aktualisiert Performance-Statistics für Monitoring und Optimization
        """
        weather_data.performance_stats = {
            "cfd_iterations": cfd_iterations,
            "lod_level": weather_data.lod_level,
            "map_size": weather_data.actual_size,
            "generation_method": "gpu" if self.shader_manager else "cpu",
            "temp_range": {
                "min": float(np.min(weather_data.temp_map)),
                "max": float(np.max(weather_data.temp_map)),
                "mean": float(np.mean(weather_data.temp_map))
            },
            "wind_stats": {
                "max_speed": float(np.max(np.sqrt(weather_data.wind_map[:, :, 0]**2 +
                                                 weather_data.wind_map[:, :, 1]**2))),
                "mean_speed": float(np.mean(np.sqrt(weather_data.wind_map[:, :, 0]**2 +
                                                   weather_data.wind_map[:, :, 1]**2)))
            },
            "precipitation_total": float(np.sum(weather_data.precip_map))
        }

    def _create_fallback_weather_data(self, original_size: int, lod_level: int,
                                     parameters: Dict[str, Any]) -> WeatherData:
        """
        Error-Recovery: Erstellt Minimal-Weather-System bei kritischen Failures
        """
        target_size = self._get_lod_size(lod_level, original_size)

        # Minimal-Weather-Fields
        weather_data = WeatherData()
        weather_data.wind_map = np.zeros((target_size, target_size, 2), dtype=np.float32)
        weather_data.wind_map[:, :, 0] = 5.0  # Konstanter 5 m/s Ostwind

        weather_data.temp_map = np.full((target_size, target_size),
                                      parameters.get('air_temp_entry', 15.0), dtype=np.float32)
        # Placeholder in mm/Jahr-Äquivalent (siehe PRECIP_ANNUAL_SCALE_FACTOR) -
        # entspricht ungefähr einer gemäßigten, mittleren Niederschlagsmenge.
        weather_data.precip_map = np.full((target_size, target_size), 800.0, dtype=np.float32)
        weather_data.humid_map = np.full((target_size, target_size), 30.0, dtype=np.float32)

        weather_data.lod_level = lod_level
        weather_data.actual_size = target_size
        weather_data.validity_state = {"valid": False, "fallback": True}

        self.logger.warning("Fallback weather data created due to generation failure")

        return weather_data

    # ===== UTILITY METHODS =====

    def _interpolate_2d_bicubic(self, array: np.ndarray, target_size: int) -> np.ndarray:
        """
        Bicubic-Interpolation für Pattern-Preservation bei LOD-Upscaling
        """
        from scipy.ndimage import zoom

        old_size = array.shape[0]
        if old_size == target_size:
            return array.copy()

        scale_factor = target_size / old_size

        try:
            # SciPy zoom für bicubic-ähnliche Interpolation
            interpolated = zoom(array, scale_factor, order=3)

            # Exakte Größe sicherstellen
            if interpolated.shape[0] != target_size:
                interpolated = interpolated[:target_size, :target_size]

            return interpolated.astype(array.dtype)

        except Exception as e:
            self.logger.warning(f"Bicubic interpolation failed, using bilinear: {e}")
            # Fallback zu bilinearer Interpolation
            return self._interpolate_2d_bilinear(array, target_size)

    def _interpolate_2d_bilinear(self, array: np.ndarray, target_size: int) -> np.ndarray:
        """
        Bilineare Interpolation als Fallback für Bicubic
        """
        old_size = array.shape[0]
        if old_size == target_size:
            return array.copy()

        scale_factor = (old_size - 1) / (target_size - 1)
        interpolated = np.zeros((target_size, target_size), dtype=array.dtype)

        for new_y in range(target_size):
            for new_x in range(target_size):
                old_x = new_x * scale_factor
                old_y = new_y * scale_factor

                x0, y0 = int(old_x), int(old_y)
                x1, y1 = min(x0 + 1, old_size - 1), min(y0 + 1, old_size - 1)

                fx, fy = old_x - x0, old_y - y0

                # Bilineare Interpolation
                h00, h10 = array[y0, x0], array[y0, x1]
                h01, h11 = array[y1, x0], array[y1, x1]

                h0 = h00 * (1 - fx) + h10 * fx
                h1 = h01 * (1 - fx) + h11 * fx

                interpolated[new_y, new_x] = h0 * (1 - fy) + h1 * fy

        return interpolated

    def _calculate_slopes_vectorized(self, heightmap: np.ndarray) -> np.ndarray:
        """
        Vectorized Slope-Calculation (dz/dx, dz/dy) für Performance.

        Durch reale Meter/Pixel geteilt (siehe core/terrain_generator.py
        SlopeCalculator._calculate_cpu_slopes für den identischen Bug an anderer
        Stelle) - ohne das war ein 1m Höhenunterschied zwischen Nachbarpixeln wie
        1m realer Horizontal-Abstand behandelt, obwohl ein Pixel bei typischen
        Kartengrößen tatsächlich ~50-300m abdeckt. Das ließ wind_slope_alignment
        (und darüber oro_precip) auf praktisch jedem Hang unrealistisch groß werden.
        """
        height, width = heightmap.shape
        slopemap = np.zeros((height, width, 2), dtype=np.float32)

        # Live-Wert statt der vorherigen statischen TERRAIN.WORLD_SIZE_KM-
        # Konstante (siehe [[project-terrain-review]] 4f).
        map_distance_km = self.data_lod_manager.get_map_distance_km() if self.data_lod_manager else 10.0
        spacing = (map_distance_km * 1000.0) / height

        # dz/dx (vectorized)
        slopemap[:, 1:-1, 0] = (heightmap[:, 2:] - heightmap[:, :-2]) * 0.5 / spacing
        slopemap[:, 0, 0] = (heightmap[:, 1] - heightmap[:, 0]) / spacing
        slopemap[:, -1, 0] = (heightmap[:, -1] - heightmap[:, -2]) / spacing

        # dz/dy (vectorized)
        slopemap[1:-1, :, 1] = (heightmap[2:, :] - heightmap[:-2, :]) * 0.5 / spacing
        slopemap[0, :, 1] = (heightmap[1, :] - heightmap[0, :]) / spacing
        slopemap[-1, :, 1] = (heightmap[-1, :] - heightmap[-2, :]) / spacing

        return slopemap

    def _calculate_curvature_normalized(self, heightmap: np.ndarray) -> np.ndarray:
        """
        Normalisierte Gelände-Krümmung (diskreter Laplace der Heightmap,
        reales Meter/Pixel-Spacing wie _calculate_slopes_vectorized) für den
        Grat-/Canyon-Speedup-Term (siehe [[project-wind-ridge-speedup]],
        WindNinjas Terrain-Shape-Effekt): negativ auf konvexen Graten/Kuppen
        (Wind soll dort beschleunigen), positiv in konkaven Tälern (Wind soll
        dort bremsen).

        Auf [-1,1] per geclipptem Z-Score (3 Standardabweichungen) normiert
        statt fixer physikalischer Einheiten - die rohe Krümmung
        (Höhe/Meter²) hängt stark von TERRAIN.AMPLITUDE/WORLD_SIZE_KM ab, ein
        Z-Score-Clip macht den Speedup-Effekt unabhängig von der absoluten
        Terrain-Skalierung nutzbar (ähnliches Muster wie height_normalized an
        anderer Stelle dieser Datei, aber robust gegen Ausreißer statt Min/Max).
        Ränder bleiben bei 0 (kein Krümmungssignal) - dieselbe Vereinfachung
        wie das ungenutzte äußerste Pixel bei vielen zentralen Differenzen
        dieser Datei.
        """
        height, width = heightmap.shape
        # Live-Wert statt der vorherigen statischen TERRAIN.WORLD_SIZE_KM-
        # Konstante (siehe [[project-terrain-review]] 4f).
        map_distance_km = self.data_lod_manager.get_map_distance_km() if self.data_lod_manager else 10.0
        spacing = (map_distance_km * 1000.0) / height

        d2x = np.zeros((height, width), dtype=np.float32)
        d2y = np.zeros((height, width), dtype=np.float32)
        d2x[:, 1:-1] = (heightmap[:, 2:] - 2.0 * heightmap[:, 1:-1] + heightmap[:, :-2]) / (spacing ** 2)
        d2y[1:-1, :] = (heightmap[2:, :] - 2.0 * heightmap[1:-1, :] + heightmap[:-2, :]) / (spacing ** 2)
        laplacian = d2x + d2y

        scale = float(np.std(laplacian)) + 1e-9
        return np.clip(laplacian / (3.0 * scale), -1.0, 1.0).astype(np.float32)

    def _resize_nearest_labels(self, label_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Nearest-Neighbor-Resize für kategoriale/Label-Daten (z.B. biome_map) -
        bilineare/bikubische Interpolation wie _interpolate_2d_bicubic würde
        an Kategoriegrenzen unsinnige Zwischenwerte erzeugen. Siehe
        _get_roughness_damping() für den Aufrufkontext (analog zum
        bestehenden Muster in water_generator.py::_resize_nearest).
        """
        src_h, src_w = label_map.shape[:2]
        tgt_h, tgt_w = target_shape
        if (src_h, src_w) == (tgt_h, tgt_w):
            return label_map
        y_idx = np.clip((np.arange(tgt_h) * src_h / tgt_h).astype(int), 0, src_h - 1)
        x_idx = np.clip((np.arange(tgt_w) * src_w / tgt_w).astype(int), 0, src_w - 1)
        return label_map[np.ix_(y_idx, x_idx)]

    def _weighted_solar_exposure(self, shadowmap: np.ndarray,
                                  sun_angles: Optional[list] = None) -> np.ndarray:
        """
        Gewichtete Kombination der bis zu 7 Sonnenwinkel-Kanäle einer
        shadowmap. Jeder Kanal enthält bereits pro Pixel dot(normal, sun_dir)
        für GENAU diesen einen Sonnenwinkel (siehe ShadowCalculator.
        _calculate_slope_shading_cpu()) - die Hangausrichtungs-Abhängigkeit
        ("Flächen bekommen je nach Ausrichtung zur Sonne unterschiedlich viel
        Licht ab") ist damit bereits pro Kanal vorhanden, hier kommt nur die
        Gewichtung der Kanäle untereinander hinzu.

        Parameter: sun_angles - optionale Liste von (elevation, azimuth)-
            Paaren (Grad), EXAKT die Kanal-Teilmenge, die die übergebene
            shadowmap erzeugt hat (z.B. das Ergebnis von ShadowCalculator.
            get_sun_angles_for_lod(), NICHT die volle 7er-Liste bei
            niedrigerem LOD). Wenn gesetzt, wird pro Kanal ein aus dem ECHTEN
            Elevationswinkel abgeleiteter Atmosphären-Dämpfungsfaktor
            (Airmass/Beer-Lambert-artig, siehe ATM_OPTICAL_DEPTH) genutzt
            statt der festen ShadowCalculator.sun_weights - Sonnenstand
            variiert bereits pro Monat/Breitengrad (siehe
            generate_seasonal_sun_angles()), die alte feste Tageszeit-
            Gewichtung bildete das nicht ab. Ohne sun_angles (z.B. Aufrufer
            außerhalb der monatlichen Weather-Simulation) bleibt das alte
            Verhalten (feste sun_weights) unverändert erhalten.

        Fällt auf ein flaches Mittel zurück, wenn shadowmap 2D ist (kein
        Kanal-Stack) oder die Kanalzahl nicht zu den Gewichten passt (z.B.
        ein älterer Cache-Eintrag mit abweichender Kanalzahl).
        """
        if shadowmap.ndim != 3:
            return shadowmap
        if sun_angles is not None and len(sun_angles) == shadowmap.shape[2]:
            elevations = np.array([e for e, _ in sun_angles], dtype=np.float64)
            elevations_clamped = np.radians(np.clip(elevations, ATM_MIN_ELEVATION_DEG, 90.0))
            airmass = 1.0 / np.sin(elevations_clamped)
            weights = np.exp(-ATM_OPTICAL_DEPTH * (airmass - 1.0))
        else:
            weights = np.asarray(self.shadow_calculator.sun_weights, dtype=np.float64)
            if weights.shape[0] != shadowmap.shape[2]:
                return np.mean(shadowmap, axis=2).astype(np.float32)
        return np.average(shadowmap, axis=2, weights=weights).astype(np.float32)

    def _get_roughness_damping(self, target_shape: Tuple[int, int], lod_level: int) -> Optional[np.ndarray]:
        """
        Best-effort Bodenrauigkeits-Dämpfung aus der Biome-Klassifikation
        (siehe [[project-wind-roughness]], _BIOME_ROUGHNESS_DAMPING oben).

        Biome läuft NACH Weather im Calculator-Graph (weather.wind ist keine
        Abhängigkeit von biome.*) - ein aktueller Wert ist nur verfügbar,
        wenn Biome in DIESER Session bereits (mindestens für ein niedrigeres
        LOD) angefragt wurde, z.B. beim Auto-Start der vollen Pipeline. Bei
        isolierter Weather-Tab-Generierung bleibt Biome unangefragt - dann
        liefert dieser Aufruf None, und der Aufrufer wendet keine Dämpfung an
        (identisches Verhalten zu vor dieser Änderung).

        Bei einer Shape-Abweichung (z.B. weil Biome zuletzt bei einer anderen
        map_size lief - der DataLODManager-Cache für Calculator-Outputs wird
        von clear_all_data()/invalidate_cache_lod() nie geleert) wird per
        Nearest-Neighbor resampled statt zu verwerfen, analog zum
        bestehenden Muster in water_generator.py::_resize_nearest - eine
        stilisierte, leicht veraltete Rauigkeits-Näherung ist hier
        unproblematischer als bei echten Höhen-/Strömungsdaten.
        """
        biome_map = self.data_lod_manager.get_calculator_output(
            "biome.integrate_layers", "biome_map", lod_level)
        if biome_map is None:
            return None
        if biome_map.shape[:2] != tuple(target_shape):
            biome_map = self._resize_nearest_labels(biome_map, target_shape)
        ids = np.clip(biome_map.astype(np.int32), 0, len(_BIOME_ROUGHNESS_DAMPING) - 1)
        return _BIOME_ROUGHNESS_DAMPING[ids]

    def _get_solar_absorption_factor(self, target_shape: Tuple[int, int], lod_level: int) -> Optional[np.ndarray]:
        """
        Best-effort biom-abhängiger Solar-Absorptionsfaktor (siehe
        _BIOME_SOLAR_ABSORPTION oben) - IDENTISCHER Aufbau wie
        _get_roughness_damping() direkt darüber (gleiche Datenquelle,
        gleiches None-Fallback bei fehlenden Biome-Daten, gleiches
        Nearest-Neighbor-Resampling bei Shape-Abweichung) - siehe dortige
        Docstring für die Begründung, hier nicht wiederholt.
        """
        biome_map = self.data_lod_manager.get_calculator_output(
            "biome.integrate_layers", "biome_map", lod_level)
        if biome_map is None:
            return None
        if biome_map.shape[:2] != tuple(target_shape):
            biome_map = self._resize_nearest_labels(biome_map, target_shape)
        ids = np.clip(biome_map.astype(np.int32), 0, len(_BIOME_SOLAR_ABSORPTION) - 1)
        return _BIOME_SOLAR_ABSORPTION[ids]

    def _generate_atmospheric_noise(self, shape: Tuple[int, int], target_size: int,
                                    month_index: int = 0) -> np.ndarray:
        """
        Generiert atmospheric Noise-Variation mit Edge-Enhancement.

        month_index verschiebt die Noise-Koordinaten um einen fixen Offset - exakt
        dasselbe Muster wie _generate_pressure_noise() (letzte Runde). Vorher nutzte
        dieser Aufruf IMMER dieselben Koordinaten (nur von der Pixelposition
        abhängig) - da dieses Rauschen die Bodentemperatur speist, die wiederum die
        initiale Feuchte-Quelle des gekoppelten 3-Schicht-Loops treibt, blieb das
        räumliche MUSTER von humid_map über alle 6 Monate praktisch identisch (nur
        das Niveau verschob sich über saisonale Parameter) - empirisch bestätigt:
        paarweise räumliche Korrelation zwischen Monaten lag bei 0.72-0.96. Offset
        (10 Einheiten pro Monat) liegt deutlich über der OpenSimplex-Kohärenzlänge
        bei dieser Frequenz, siehe [[project-3layer-wind-cfd]].

        Vektorisiert via noise2array() statt der früheren Python-Doppelschleife
        (identische Formel, inkl. Edge-Enhancement) - selber Grund wie die
        Vektorisierung der CFD-Hotpaths letzte Runde: wird jetzt pro Monat
        aufgerufen, nicht mehr nur einmal insgesamt.
        """
        height, width = shape
        offset = month_index * 10.0

        x_coords = np.arange(width, dtype=np.float64) / width * 3.0 + offset
        y_coords = np.arange(height, dtype=np.float64) / height * 3.0 + offset
        noise_field = self.noise_generator.noise2array(x_coords, y_coords).astype(np.float32)

        if target_size > 128:
            noise_field += self.noise_generator.noise2array(
                x_coords * 2, y_coords * 2).astype(np.float32) * 0.3

        # Edge-Factor für stärkere Variation an Kartenrändern (identische Formel
        # wie die frühere _calculate_edge_factor(), vektorisiert).
        y_idx, x_idx = np.mgrid[0:height, 0:width]
        dist_to_edge = np.minimum.reduce([x_idx, y_idx, width - 1 - x_idx, height - 1 - y_idx])
        max_dist = max(min(width, height) // 6, 1)
        edge_factor = np.where(
            dist_to_edge < max_dist,
            1.0 + (max_dist - dist_to_edge) / max_dist * 0.8,
            1.0
        ).astype(np.float32)

        return noise_field * edge_factor * 3.0

    def _generate_pressure_noise(self, shape: Tuple[int, int], month_index: int = 0) -> np.ndarray:
        """
        Generiert Pressure-Noise für CFD-Simulation.

        month_index verschiebt die Noise-Koordinaten um einen fixen Offset -
        vorher nutzte dieser Aufruf IMMER dieselben Koordinaten (nur von der
        Pixelposition abhängig), wodurch das kleinräumige, optisch dominante
        Rauschmuster des Windfelds für alle 6 saisonalen Perioden identisch
        blieb, während nur die (schwächere) großräumige Grundrichtung
        rotierte - der Wind wirkte dadurch trotz korrekt saisonal variierender
        Richtung optisch "eingefroren". Offset (10 Einheiten pro Monat) liegt
        deutlich über der OpenSimplex-Kohärenzlänge bei dieser Frequenz, das
        Muster ist pro Monat dadurch optisch unabhängig. Nutzt noise2array()
        (bereits an anderer Stelle dieser Session verifiziert) statt der
        früheren Doppel-Schleife - vektorisiert, kein Verhaltensrisiko.
        """
        height, width = shape
        offset = month_index * 10.0
        x = np.arange(width, dtype=np.float64) / width * 2.0 + offset
        y = np.arange(height, dtype=np.float64) / height * 2.0 + offset
        return self.noise_generator.noise2array(x, y).astype(np.float32)

    def _apply_thermal_convection(self, wind_field: np.ndarray, temp_map: np.ndarray,
                                 shadowmap: np.ndarray, parameters: Dict[str, Any]):
        """
        Wendet thermische Konvektion auf Wind-Field an (in-place)
        """
        height, width = temp_map.shape

        # Temperature-Gradients (vectorized)
        temp_grad_x = np.zeros_like(temp_map)
        temp_grad_y = np.zeros_like(temp_map)

        temp_grad_x[:, 1:-1] = (temp_map[:, 2:] - temp_map[:, :-2]) * 0.5
        temp_grad_y[1:-1, :] = (temp_map[2:, :] - temp_map[:-2, :]) * 0.5

        # Thermal-Convection-Strength
        avg_temp = np.mean(temp_map)
        temp_diff = temp_map - avg_temp
        convection_strength = temp_diff * parameters['thermic_effect'] * 0.08

        # Shadow-based thermal effects - Tageszeit-gewichtet statt flachem
        # Mittel, siehe _weighted_solar_exposure().
        shadow_avg = self._weighted_solar_exposure(shadowmap)

        shadow_effect = (shadow_avg - 0.5) * parameters['thermic_effect'] * 0.15

        # Apply thermal modifications
        wind_field[:, :, 0] += temp_grad_x * 0.05 + convection_strength
        wind_field[:, :, 1] += temp_grad_y * 0.05 + shadow_effect

    def _apply_wind_diffusion(self, wind_field: np.ndarray, diffusion_rate: float) -> np.ndarray:
        """
        Wendet Diffusion auf Wind-Field für numerical Stability an.

        Vektorisiert via np.roll statt der früheren Python-Doppelschleife (identische
        4-Nachbar-Mittelwert-Formel) - notwendig, weil der neue 3-Schicht-CFD-
        Zeitschritt-Loop (siehe [[project-3layer-wind-cfd]]) diese Funktion pro Schicht
        UND pro Zeitschritt aufruft statt wie bisher nur einmal pro Monat; bei größeren
        LODs wäre die reine Python-Schleife sonst spürbar limitierend. np.roll wraps am
        Rand um, aber die dadurch "falschen" Werte landen nur in Zeile/Spalte 0 bzw.
        -1 des Zwischenergebnisses, die am Ende explizit auf die unveränderten
        Original-Randwerte zurückgesetzt werden (identisch zum alten Verhalten, das
        Randpixel nie anfasste, da die Schleife bei range(1, height-1) begann).
        """
        neighbor_avg = (
            np.roll(wind_field, 1, axis=0) + np.roll(wind_field, -1, axis=0) +
            np.roll(wind_field, 1, axis=1) + np.roll(wind_field, -1, axis=1)
        ) * 0.25

        diffused_field = wind_field + (neighbor_avg - wind_field) * diffusion_rate
        diffused_field[0, :] = wind_field[0, :]
        diffused_field[-1, :] = wind_field[-1, :]
        diffused_field[:, 0] = wind_field[:, 0]
        diffused_field[:, -1] = wind_field[:, -1]

        return diffused_field

    def _apply_vorticity_confinement(self, wind_field: np.ndarray, vorticity_strength) -> None:
        """
        Vorticity Confinement (Fedkiw/Stam-Standardtechnik, siehe
        [[project-3layer-wind-cfd]]) - injiziert lokale Rotationsenergie zurück,
        die _apply_wind_diffusion pro Zeitschritt entfernt. Formel 1:1 aus
        niels747/2D-Weather-Sandbox's curlShader.frag + vorticityShader.frag
        übernommen (nicht neu hergeleitet), an unser (H,W,2)-Layout angepasst:

            curl = dv/dx - du/dy                              (skalare 2D-Rotation)
            force = normalize(grad(|curl|)) * curl * staerke   (Kraft senkrecht zum
                                                                 |curl|-Gradienten)

        Ohne diesen Fix war der Wind über weite Flächen fast parallel (empirisch
        ~7.8° mittlere Richtungsänderung zwischen Nachbarpixeln) - reine Diffusion
        + Kontinuitätskorrektur glätten aktiv jede kleinräumige Verwirbelung weg,
        die "Turbulenz" ausmacht.

        Modifiziert wind_field IN-PLACE (wie _apply_continuity_correction).
        vorticity_strength: Skalar ODER (H,W)-Array (z.B. für die Rand-
        Verstärkung, siehe vorticity_edge_boost in _run_coupled_atmosphere_
        simulation) - beides funktioniert per NumPy-Broadcasting unverändert.

        STABILITÄTS-KAPPUNG: die Referenz-Formel selbst hat keine eingebaute
        Grenze - bei uns wird sie über viele Zeitschritte (8-60+, siehe
        _get_atmosphere_loop_steps) wiederholt angewendet, ohne einen echten
        Zeitschritt-Skalierungsfaktor dt (anders als im Referenz-Shader, der pro
        Frame mit einem sehr kleinen, festen dt läuft). Ohne Kappung entsteht eine
        Rückkopplung (mehr Curl -> mehr Kraft -> mehr Curl im nächsten Schritt),
        die bei höheren LODs (mehr Iterationen UND feineres Gitter) beobachtbar
        eskalierte (empirisch: 12 m/s bei LOD1/8 Schritten, 247 m/s bei LOD3/18
        Schritten - eindeutige Instabilität, kein Rundungsfehler). Kraft-Betrag
        pro Aufruf auf denselben Größenordnungsbereich gekappt wie die übrigen
        Pro-Iterations-Terme in dieser Datei (thermische Konvektion, Terrain-
        Ablenkung: O(0.05-0.5)) - Turbulenz baut sich dadurch über mehrere
        Schritte graduell auf statt in einem Schritt zu explodieren.
        """
        u, v = wind_field[:, :, 0], wind_field[:, :, 1]

        dvdx = np.zeros_like(v)
        dudy = np.zeros_like(u)
        dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) * 0.5
        dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) * 0.5
        curl = dvdx - dudy

        abs_curl = np.abs(curl)
        grad_x = np.zeros_like(curl)
        grad_y = np.zeros_like(curl)
        grad_x[:, 1:-1] = (abs_curl[:, 2:] - abs_curl[:, :-2]) * 0.5
        grad_y[1:-1, :] = (abs_curl[2:, :] - abs_curl[:-2, :]) * 0.5
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-6

        force_x = (grad_y / magnitude) * curl * vorticity_strength
        force_y = (-grad_x / magnitude) * curl * vorticity_strength

        FORCE_CAP = 1.0  # m/s pro Aufruf, siehe Docstring-Begründung oben
        force_magnitude = np.sqrt(force_x ** 2 + force_y ** 2)
        scale_down = np.minimum(1.0, FORCE_CAP / np.maximum(force_magnitude, 1e-6))
        force_x *= scale_down
        force_y *= scale_down

        wind_field[:, :, 0] += force_x
        wind_field[:, :, 1] += force_y

    def _get_poisson_inv_eigs(self, height: int, width: int) -> np.ndarray:
        """
        Liefert (gecached) 1/lambda(p,q) für den separierbaren 2D-Neumann-Laplace,
        den die DCT-Typ-2/3-Basis diagonalisiert: lambda(p,q) = 2cos(pi*p/H) +
        2cos(pi*q/W) - 4. lambda(0,0)=0 (Mittelwert-Freiheitsgrad von phi) wird auf
        einen Dummy-Wert gesetzt - div_hat[0,0] wird vor der Division ohnehin auf 0
        gezwungen (siehe _apply_continuity_correction), der Dummy-Wert hier wird also
        nie tatsächlich verwendet.
        """
        key = (height, width)
        cached = self._poisson_eig_cache.get(key)
        if cached is not None:
            return cached
        # Eigenwert-Berechnung selbst in float64 (nur ein einmaliger Aufwand pro
        # Grid-Größe, dank Cache), der resultierende Nenner wird für die
        # Multiplikation mit dem float32-DCT-Spektrum nach float32 zurückgecastet.
        p = np.arange(height, dtype=np.float64)
        q = np.arange(width, dtype=np.float64)
        lam = (2.0 * np.cos(np.pi * p / height) - 2.0)[:, None] + \
              (2.0 * np.cos(np.pi * q / width) - 2.0)[None, :]
        lam[0, 0] = 1.0
        inv_lam = (1.0 / lam).astype(np.float32)
        self._poisson_eig_cache[key] = inv_lam
        return inv_lam

    def _apply_continuity_correction(self, wind_field: np.ndarray, vertical_flux_term: Optional[np.ndarray] = None):
        """
        Erzwingt Massenerhaltung (~Divergenzfreiheit) per Poisson-Projektion -
        WindNinja-inspiriert (siehe [[project-wind-poisson-projection]]): WindNinja
        baut ein initiales Windfeld u0 und löst dann eine Poisson-Gleichung
        grad^2(phi) = div(u0) für ein Korrekturpotential phi (bei WindNinja per FEM+CG
        auf einem 3D-Terrain-Mesh), danach u_final = u0 - grad(phi). Für dieses
        reguläre Cartesian-Grid ist das direkte Äquivalent eine schnelle
        DCT-basierte Poisson-Lösung mit Neumann-Randbedingung (kein Wind-Quell-/
        Senken-Fluss über den Kartenrand) - derselbe Projektionsschritt wie in
        "Stable Fluids" (Stam).

        Ersetzt die frühere schwache lokale Iteration (nur 10% der lokalen Divergenz
        pro Aufruf entfernt, Ränder nie korrigiert) durch eine einmalige globale
        Lösung pro Aufruf, die das Feld in einem Schritt (bis auf
        Diskretisierungs-/Rundungsfehler) tatsächlich divergenzfrei macht - siehe
        Verifikation in [[project-wind-poisson-projection]].

        vertical_flux_term (optional, (H,W)): zusätzlicher Divergenz-Beitrag aus dem
        vertikalen Massenaustausch zwischen Atmosphären-Schichten (siehe
        _run_coupled_atmosphere_simulation Schritt 6, [[project-3layer-wind-cfd]]) -
        (w_oben - w_unten)/dicke. Positiv = Netto-Massenzufluss von oben/unten in
        diese Schicht (die Horizontal-Korrektur muss dann netto AUSWärts divergieren,
        um den Zufluss auszugleichen), negativ = Netto-Abfluss (Konvergenz nötig).
        Eine reine Neumann-Projektion kann nur Divergenz mit Domänen-Mittel ~0
        entfernen; ein echter über die Schicht gemittelter Netto-Massenfluss bleibt
        als realer Rest bestehen - das ist physikalisch korrekt (Nettomasse fließt
        tatsächlich in/aus dieser Schicht), kein Solver-Defekt.

        Divergenz/Gradient nutzen weiterhin das zentrale 2h-Schema dieser Datei,
        jetzt aber auch an den Rändern via Spiegel-Ghost-Zellen (f[-1]:=f[0],
        f[N]:=f[N-1] - die diskrete Neumann-Randbedingung), statt sie wie zuvor auf 0
        zu lassen.

        Poisson-Arithmetik läuft komplett in float32 (Messung: float64 brachte bei
        realistischen, bereits geglätteten Windfeldern - hier läuft _apply_wind_diffusion
        immer direkt davor - keine messbar bessere Divergenz-Reduktion (~96-98% in
        beiden Fällen), kostete aber ~2x mehr Laufzeit, siehe
        [[project-wind-poisson-projection]]. Die theoretische float32-Rauschgrenze bei
        kleinen Eigenwerten (Verstärkungsfaktor ~10^5 bei N=1024) betrifft primär
        adversarielle/reine Rauschfelder mit signifikanter Energie in der Domänen-Mittel-
        Komponente, nicht die hier tatsächlich vorkommenden glatten Felder.
        """
        height, width = wind_field.shape[:2]
        u = wind_field[:, :, 0]
        v = wind_field[:, :, 1]

        div = np.zeros((height, width), dtype=np.float32)
        div[:, 1:-1] = (u[:, 2:] - u[:, :-2]) * 0.5
        div[:, 0] = (u[:, 1] - u[:, 0]) * 0.5
        div[:, -1] = (u[:, -1] - u[:, -2]) * 0.5
        div[1:-1, :] += (v[2:, :] - v[:-2, :]) * 0.5
        div[0, :] += (v[1, :] - v[0, :]) * 0.5
        div[-1, :] += (v[-1, :] - v[-2, :]) * 0.5
        if vertical_flux_term is not None:
            # Netto-Zufluss von oben/unten wirkt wie negative horizontale Divergenz
            # (die Zelle "will" sich horizontal ausdehnen, um den Zufluss
            # auszugleichen) - siehe Docstring oben.
            div -= vertical_flux_term.astype(np.float32)

        inv_lambda = self._get_poisson_inv_eigs(height, width)
        div_hat = dctn(div, type=2, norm='ortho')
        div_hat[0, 0] = 0.0  # Mittelwert-Freiheitsgrad von phi, siehe Docstring
        phi = idctn((div_hat * inv_lambda).astype(np.float32), type=2, norm='ortho')

        grad_x = np.zeros((height, width), dtype=np.float32)
        grad_y = np.zeros((height, width), dtype=np.float32)
        grad_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) * 0.5
        grad_x[:, 0] = (phi[:, 1] - phi[:, 0]) * 0.5
        grad_x[:, -1] = (phi[:, -1] - phi[:, -2]) * 0.5
        grad_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) * 0.5
        grad_y[0, :] = (phi[1, :] - phi[0, :]) * 0.5
        grad_y[-1, :] = (phi[-1, :] - phi[-2, :]) * 0.5

        wind_field[:, :, 0] -= grad_x
        wind_field[:, :, 1] -= grad_y

    def _transport_moisture_simple(self, humid_map: np.ndarray, wind_field: np.ndarray,
                                  dt: float = 0.5) -> np.ndarray:
        """
        Simplified Moisture-Transport durch Advection.

        Vektorisiert via scipy.ndimage.map_coordinates statt der früheren Python-
        Doppelschleife mit manueller bilinearer Interpolation - identische Semi-
        Lagrange-Rückwärts-Sample-Formel (source = position - wind*dt*skalierung,
        bilinear an der Quellposition gesampelt), nur vektorisiert. Das ist zugleich
        die allgemeine Advektions-Primitive, die der neue 3-Schicht-CFD-Zeitschritt-
        Loop für Wind/Temperatur/Feuchte gleichermaßen braucht (siehe
        [[project-3layer-wind-cfd]]) - keine Wegwerf-Arbeit. mode='nearest' entspricht
        dem alten expliziten np.clip auf die Array-Grenzen. Randzeilen/-spalten bleiben
        wie im Original unverändert (Schleife begann bei range(1, height-1)).
        """
        height, width = humid_map.shape
        y_idx, x_idx = np.mgrid[0:height, 0:width].astype(np.float64)

        wind_x = wind_field[:, :, 0] * dt * 0.1  # Scaled für Stabilität
        wind_y = wind_field[:, :, 1] * dt * 0.1

        source_x = np.clip(x_idx - wind_x, 0, width - 1)
        source_y = np.clip(y_idx - wind_y, 0, height - 1)

        sampled = map_coordinates(humid_map, [source_y, source_x], order=1, mode='nearest')

        transported_humid = humid_map.copy()
        transported_humid[1:-1, 1:-1] = sampled[1:-1, 1:-1]
        return transported_humid.astype(humid_map.dtype)

    def _apply_humidity_diffusion(self, humid_map: np.ndarray, iterations: int = 2,
                                   hardness_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Humidity-Diffusion für natural Distribution - echte scipy.ndimage.gaussian_filter-
        Diffusion statt des früheren handgerollten 4-Nachbar-Box-Blurs (identisches
        iterations-Argument beibehalten, nur als Sigma-Skalierung statt Loop-Zähler
        genutzt - Iterations-Anzahl skalierte vorher grob die Diffusionsstärke,
        das übernimmt jetzt Sigma direkt).

        hardness_map (optional): normiert auf [0,1] und pixelweise als Blend-Gewicht
        genutzt - result = hardness_norm*original + (1-hardness_norm)*diffused. Weiches
        Gestein (niedrige hardness) lässt Feuchte stärker zu den Nachbarn diffundieren
        (mehr diffundierter Anteil), hartes Gestein hält Feuchte lokal fester (mehr
        Original-Anteil) - "sedimentation verteilt das besser als harter Stein"
        (Nutzer-Feedback). Ohne hardness_map (z.B. allererster Weather-Lauf, bevor
        Geology überhaupt etwas geliefert hat) wird gleichmäßig vollständig diffundiert.
        """
        sigma = 1.5 * max(1, iterations)
        diffused_map = gaussian_filter(humid_map, sigma=sigma)

        if hardness_map is None:
            return diffused_map

        h_min, h_max = float(hardness_map.min()), float(hardness_map.max())
        if h_max - h_min > 1e-6:
            hardness_norm = (hardness_map - h_min) / (h_max - h_min)
        else:
            hardness_norm = np.zeros_like(hardness_map)

        return hardness_norm * humid_map + (1.0 - hardness_norm) * diffused_map

    def _calculate_parameter_hash(self, parameters: Dict[str, Any]) -> str:
        """
        Erstellt Parameter-Hash für Cache-Management
        """
        import hashlib

        # Sortierte Parameter für consistent Hashing
        sorted_params = sorted(parameters.items())
        param_string = str(sorted_params) + str(self.map_seed)

        return hashlib.md5(param_string.encode()).hexdigest()[:16]

    def _update_progress(self, phase: str, progress: int, message: str):
        """
        Progress-Update für UI-Integration
        """
        if self.progress_callback:
            self.progress_callback(phase, progress, message)
        else:
            self.logger.debug(f"Weather Progress [{progress}%]: {phase} - {message}")

    def update_seed(self, new_seed: int):
        """
        Aktualisiert Seed für alle Weather-Komponenten
        """
        if new_seed != self.map_seed:
            self.map_seed = new_seed
            self.noise_generator = OpenSimplex(seed=new_seed)
            self.logger.debug(f"Weather seed updated to {new_seed}")


# ===== SUB-KOMPONENTEN (modernisiert) =====

class TemperatureCalculator:
    """
    Modernisierte Temperature-Calculation mit GPU-Shader-Integration
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".TemperatureCalculator")

    def calculate_temperature_with_orographic_effects(self, heightmap: np.ndarray,
                                                    shadowmap: np.ndarray,
                                                    parameters: Dict[str, Any]) -> np.ndarray:
        """
        Temperature-Calculation mit erweiterten orographischen Effekten
        """
        # Alle Temperature-Komponenten integrieren
        temp_map = self._calculate_base_temperature(heightmap, shadowmap, parameters)
        temp_map = self._apply_orographic_effects(temp_map, heightmap, parameters)
        temp_map = self._apply_latitude_gradient(temp_map, heightmap.shape)

        return temp_map

    def _calculate_base_temperature(self, heightmap: np.ndarray, shadowmap: np.ndarray,
                                   parameters: Dict[str, Any]) -> np.ndarray:
        """
        Base Temperature mit Altitude-Cooling und Solar-Heating.
        STALE/unreachable: TemperatureCalculator wird nirgends instanziiert
        aufgerufen (verifiziert, siehe Plan "Weather: Bodentemperatur-Modell +
        konvektiver Wärmeübergang") - nutzt daher weiterhin den alten
        'solar_power'-Parameter-Namen statt 'ground_temp_offset', bewusst
        nicht mitgezogen (toter Code, kein aktiver Aufrufer).
        """
        # Basis-Temperatur
        temp_map = np.full(heightmap.shape, parameters['air_temp_entry'], dtype=np.float32)

        # Altitude-Cooling (Parameter ist °C/km)
        altitude_cooling = parameters['altitude_cooling'] / 1000.0
        temp_map -= heightmap * altitude_cooling

        # Solar-Heating aus Shadowmap
        if len(shadowmap.shape) == 3:
            # Multi-Angle Shadows - gewichtete Kombination
            shadow_weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1, 0.05, 0.05])  # Mittag stärker gewichtet
            shadow_combined = np.average(shadowmap, axis=2, weights=shadow_weights[:shadowmap.shape[2]])
        else:
            shadow_combined = shadowmap

        solar_effect = (shadow_combined - 0.5) * parameters['solar_power']
        temp_map += solar_effect

        return temp_map

    def _apply_orographic_effects(self, temp_map: np.ndarray, heightmap: np.ndarray,
                                 parameters: Dict[str, Any]) -> np.ndarray:
        """Erweiterte orographische Temperature-Effects"""
        # Valley-Inversion-Effect (Täler kühler bei hohen Lagen)
        height, width = heightmap.shape

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                current_elevation = heightmap[y, x]

                # Nachbar-Elevations
                neighbor_elevations = [
                    heightmap[y-1, x], heightmap[y+1, x],
                    heightmap[y, x-1], heightmap[y, x+1]
                ]
                max_neighbor = max(neighbor_elevations)

                # Valley-Detection und Temperature-Inversion
                if current_elevation < max_neighbor - 50:  # 50m tiefer als Nachbarn
                    valley_effect = -2.0  # 2°C kühler in Tälern
                    temp_map[y, x] += valley_effect

        return temp_map

    def _apply_latitude_gradient(self, temp_map: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Latitude-Gradient: Nord-Süd Temperature-Variation"""
        height, width = shape

        # 5°C Unterschied von Süd (y=0) zu Nord (y=height)
        for y in range(height):
            latitude_factor = (y / (height - 1)) * 5.0
            temp_map[y, :] += latitude_factor

        return temp_map


class WindFieldSimulator:
    """
    CFD-basierte Wind-Simulation mit Navier-Stokes-Approximation
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".WindFieldSimulator")

    def simulate_cfd_wind_field(self, heightmap: np.ndarray, temp_map: np.ndarray,
                               parameters: Dict[str, Any], iterations: int) -> np.ndarray:
        """
        Full CFD-Simulation mit Navier-Stokes-Equations
        """
        # Implementation würde hier folgen - der bestehende Code ist bereits eine gute Basis
        pass


class PrecipitationSystem:
    """
    Erweiterte Precipitation mit Orographic-Enhancement
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".PrecipitationSystem")


class AtmosphericMoistureManager:
    """
    Atmospheric-Moisture mit Magnus-Formel und Wind-Enhancement
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".AtmosphericMoistureManager")


# ===== LEGACY COMPATIBILITY =====

def generate_weather_system(heightmap, shade_map, soil_moist_map, air_temp_entry, ground_temp_offset,
                           altitude_cooling, thermic_effect, wind_speed_factor, terrain_factor,
                           flow_direction=None, flow_accumulation=None, map_seed=None):
    """
    Legacy-Kompatibilität für alte API. Verifiziert ohne aktive Aufrufer im
    Projekt (siehe Plan "Weather: Bodentemperatur-Modell + konvektiver
    Wärmeübergang") - Parameter dennoch mitgezogen, um bei künftiger
    Verwendung keinen stillen KeyError zu produzieren.
    """
    generator = WeatherSystemGenerator(map_seed=map_seed or 42)

    parameters = {
        'air_temp_entry': air_temp_entry,
        'ground_temp_offset': ground_temp_offset,
        'altitude_cooling': altitude_cooling,
        'thermic_effect': thermic_effect,
        'wind_speed_factor': wind_speed_factor,
        'terrain_factor': terrain_factor
    }

    weather_data = generator.calculate_weather_system(heightmap, shade_map, parameters, 3)

    # Legacy-Format zurückgeben (Tuple)
    return weather_data.wind_map, weather_data.temp_map, weather_data.precip_map, weather_data.humid_map