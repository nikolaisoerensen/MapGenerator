"""
Path: core/geology_generator.py
Date Changed: 22.07.2026

Funktionsweise: 3D-Gesteinsstapel-Modell, terrain-gekoppelte Fassung (Teil-2-
Rework, siehe Umsetzungsplan "Geology-Rework Teil 2"). Ersetzt das frühere
2D-RGB-Mischverhältnis durch einen festen, geordneten Schichtstapel
(core/geology_layers.py), der jetzt dem Terrain folgt statt unabhängig
davon zu sein:
- Schichtdicken kombinieren räumliches Rauschen (budget-renormiert auf die
  nominale Gesamttiefe) MIT einer Slope-getriebenen Verdünnung (steile
  Hänge = dünnere Schichten, verstärkt das Relief - siehe
  `LayerThicknessBuilder.build()`).
- ZWEI getrennte Felder statt einem gemeinsamen Δz:
  - `stack_deformation` (intern, NIE additiv in `height_delta`): Terrain-Hub
    (regional geglättete Terrainhöhe, zieht den ganzen Stapel mit an,
    siehe `_build_terrain_hub()`) + Tilt + Fold + Fault-Throw. Verschiebt
    ausschließlich die Schichtgrenzen für die Ausbiss-Berechnung.
  - `height_delta`: IMMER Null - weder Tilt/Fold/Fault/Terrain-Hub NOCH
    Intrusion wirken auf die sichtbare Terrainhöhe (Nutzer-Vorgabe:
    "Störungen greifen nicht in das Terrain ein", später auf Intrusionen
    erweitert - die frühere gekappte Dom-Hebung erzeugte eine Höhenänderung,
    die nicht gewollt war). Intrusionen sind rein ein "Durchbruch durch die
    Schichten" (siehe `_apply_intrusions_to_layer_id()`), in der
    Cross-Section vom realen Terrain abgeschnitten.
- Ausbiss-Berechnung: vergleicht die REALE, ungeglättete Terrain-Höhe gegen
  die deformierten Schicht-Obergrenzen - vollständig vektorisiert über
  N_LAYERS Vergleichsmasken, kein Pixel-Loop. WICHTIG: weil `stack_
  deformation` eine GEGLÄTTETE (nicht die reale) Version der Terrainhöhe
  enthält, kürzt sich der Terrain-Hub beim Vergleich NICHT algebraisch
  heraus (siehe `_compute_outcrop()`-Docstring) - lokale Gipfel/Täler
  weichen von ihrem eigenen geglätteten Mittel ab und schneiden dadurch
  tatsächlich durch den (an Steilhängen ohnehin dünneren) Stapel. Der
  Schicht-Index wird dabei gespiegelt (`N_LAYERS - 1 - Anzahl der
  überschrittenen Grenzen`), damit höheres Terrain zu ÄLTEREM Gestein
  führt (Kristallin an Gipfeln) statt umgekehrt - siehe
  `_compute_outcrop()`-Docstring, Punkt 2.
- Intrusionen (Basalt-Blobs), Sediment-Überlagerung (rein Gesteinstyp,
  KEIN Höhenbeitrag - vermeidet Doppelzählung mit Waters Erosion/
  Sedimentation) und metamorpher Overprint (Störungs-/Intrusionsnähe)
  ergänzen den Ausbiss.
- Alle Rausch-/Distanz-Komponenten sind auf `map_distance_km` (siehe
  DataLODManager.get_map_distance_km()) statt auf reine Pixel-UV-Koordinaten
  bezogen, mit Nyquist-Fade für feine Detail-Komponenten - macht das
  Ergebnis unabhängig von Auflösung UND von der Anzahl durchlaufener
  LOD-Zwischenstufen (siehe `_resolvability_fade`).

Parameter Input (siehe gui/config/value_default.py GEOLOGY):
- sedimentary_hardness, igneous_hardness, metamorphic_hardness [1-100]
- tilt_intensity [m/km], tilt_direction [Grad] - wirkt nur auf den
  Gesteinsstapel/Ausbiss, nicht auf die Geländehöhe
- fold_intensity [m], fold_detail [0-1] - dito
- fault_intensity [m], fault_detail [0-1], fault_edge_softness [km] - dito
- intrusion_density [0-1], intrusion_size [km], intrusion_detail [0-1]
- metamorphic_overprint_intensity [0-1], foliation_detail [0-1]

Dependencies (über DataLODManager):
- heightmap (terrain.redistribution) - siehe Kommentar in _compute_outcrop()
  zur bewussten Verwendung der ROHEN statt der kombinierten Heightmap.
- slopemap (terrain.slope) - steuert jetzt die Slope-Verdünnung der
  Schichtdicke (siehe `LayerThicknessBuilder.build()`).

Output:
- GeologyData-Objekt mit rock_map, hardness_map, layer_id_map,
  height_delta, sowie Diagnose-Feldern (fault_distance_map,
  intrusion_distance_map, metamorphic_grade_map, layer_boundaries,
  delta_components) für die neuen Diagnose-Anzeigemodi/Cross-Section-View.
"""

import math
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from opensimplex import OpenSimplex
from scipy import ndimage
from scipy.ndimage import gaussian_filter, zoom

# Um wieviel gröber eine weiträumige Glättung gerechnet werden darf. Siehe
# grosse_glaettung() - eine Glättung über ein Fünftel der Karte hat unterhalb
# von sigma/4 keine Struktur mehr, das Ergebnis wird nur hochgezogen.
HUB_TEILER = 8

# Kantenlänge der Kacheln, in denen das Störungsfeld gerechnet wird. Klein
# genug, dass je Kachel nur wenige Segmente überhaupt in Frage kommen, groß
# genug, dass der Verwaltungsaufwand je Kachel nicht überwiegt. Siehe
# _build_fault_field_schnell().
FAULT_KACHEL = 128


def grosse_glaettung(feld: np.ndarray, sigma_px: float) -> np.ndarray:
    """
    Weiträumig glätten, ohne dass der Aufwand mit der dritten Potenz wächst.

    WOFÜR. Zwei Stellen in diesem Modul glätten über einen festen BRUCHTEIL der
    Karte: der Terrain-Hub (0.2 der Kantenlänge) und das Sediment-Overlay
    (2 km von 21.3). SciPys `gaussian_filter` arbeitet mit einem Kern von
    4·sigma Radius, der Aufwand ist also Pixelzahl mal sigma - und sigma wächst
    hier selbst mit der Kantenlänge:

        Aufwand ~ Kante² · Kante = Kante³

    Gemessen am Terrain-Hub: 0.03 s bei 256 px, 0.19 bei 512, 2.08 bei 1024,
    16.6 bei 2048. Bei 4096 wären es über zwei Minuten für EINE Glättung. Das
    ist der Einbruch, den der Nutzer bei "stack deformations" sah.

    WIE. Eine Glättung über ein Fünftel der Karte hat unterhalb von etwa
    sigma/4 gar keine Struktur mehr. Sie darf deshalb auf einem gröberen Gitter
    gerechnet und wieder hochgezogen werden - dieselbe Überlegung wie beim
    Schattenwurf (SCHATTEN_TEILER in terrain_generator). Gemessen bei 2048 px:
    16.6 s auf 0.52 s, also 32-fach, bei 0.03 m Abweichung gegen eine
    Feldspanne von 8 m.

    ZWEI FALLSTRICKE, beide beim ersten Versuch getroffen (97 m Fehler bei
    39 m Spanne):
      - Verkleinert werden muss mit dem BLOCKMITTEL, nicht durch Abtasten.
        `zoom` würde genau die Feinstruktur wegwerfen, die die Glättung mitteln
        soll, statt sie einzurechnen.
      - Die Randbehandlung muss `reflect` bleiben wie in der Vorlage. Bei
        einem sigma dieser Größe macht der Rand den Großteil des Ergebnisses
        aus; mit `nearest` lag das Ergebnis um mehr als die Feldspanne daneben.
    """
    height, width = feld.shape
    # Nie gröber als sigma/8. Bei sigma/4 lag die Abweichung bei 256 px noch
    # bei 2 % der Feldspanne - das grobe Gitter war dann so klein, dass das
    # Hochziehen selbst zum Fehler wurde. Bei den Größen, um die es geht
    # (1024 und mehr), greift ohnehin die Obergrenze HUB_TEILER.
    teiler = max(1, min(HUB_TEILER, int(sigma_px / 8)))
    while teiler > 1 and (width % teiler or height % teiler):
        teiler -= 1
    if teiler <= 1:
        return gaussian_filter(feld, sigma=sigma_px)

    klein = feld.reshape(height // teiler, teiler,
                         width // teiler, teiler).mean(axis=(1, 3))
    # Das Blockmittel ist selbst eine Kastenglättung der Breite 1 im groben
    # Gitter (Varianz 1/12); sie wird von der Zielvarianz abgezogen.
    ziel = max((sigma_px / teiler) ** 2 - 1.0 / 12.0, 0.01)
    klein = gaussian_filter(klein, sigma=float(np.sqrt(ziel)), mode="reflect")
    return zoom(klein, teiler, order=3, mode="reflect")[:height, :width]

from core.geology_layers import (
    ALL_ROCK_TYPES,
    TOP_REFERENCE_HEIGHT_M,
    N_LAYERS,
    NOMINAL_TOTAL_DEPTH_M,
    ROCK_LAYERS,
)

# Slope-Normierung: gleiche "maximale Steigung ~2.0"-Konvention wie das
# Vorgänger-Modell (RockTypeClassifier.apply_slope_hardening), damit die
# Slope-Verdünnung der Schichtdicke (siehe LayerThicknessBuilder.build())
# auf denselben Wertebereich reagiert, den slopemap tatsächlich liefert.
SLOPE_REFERENCE_MAGNITUDE = 2.0

# Minimaler Restanteil der Schichtdicke an sehr steilen Hängen - verhindert,
# dass Schichten an Steilhängen komplett auf 0 zusammenschrumpfen (was den
# Ausbiss-Vergleich degenerieren ließe), während der Effekt trotzdem deutlich
# sichtbar bleibt.
MIN_THICKNESS_FRACTION_AT_MAX_SLOPE = 0.15

# Glättungsbreite des Terrain-Hubs in km - groß genug, dass einzelne Gipfel/
# Täler sich sichtbar von ihrem eigenen regionalen Mittel abheben (siehe
# TectonicDisplacementField._build_terrain_hub()). Als fester Bruchteil von
# map_distance_km parametrisiert, kein eigener Slider (Design-Entscheidung,
# analog zur Sediment-Overlay-Glättungsbreite in _apply_sediment_overlay()).
REGIONAL_HUB_SIGMA_FRACTION = 0.2

# layer_id_map-Wert für Intrusionen - kein Stapel-Glied, daher außerhalb von
# [0, N_LAYERS - 1] (dem Index-Bereich von ROCK_LAYERS) angesiedelt.
BASALT_LAYER_ID = N_LAYERS

# Kappung der lokalen Dom-Hebung durch Intrusionen, als Anteil der aktuellen
# Heightmap-Spannweite - gleiche Größenordnung wie die frühere
# Igneous-Flowing-Kappung (3%) im Vorgänger-Modell.
INTRUSION_UPLIFT_CAP_FRACTION = 0.03


def _smoothstep(t):
    """Kubische Smoothstep-Rampe, geklemmt auf [0, 1]."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _resolvability_fade(px_per_cycle, low: float = 2.0, high: float = 8.0):
    """
    Weiche Ein-/Ausblendung feiner Rausch-Komponenten anhand der tatsächlich
    verfügbaren Pixel pro Wellenlänge (Nyquist-Kriterium: <2 px/Zyklus =
    Aliasing statt Detail). Reine Funktion von (Auflösung, Wellenlänge,
    map_distance_km) - KEINE LOD-Nummer, KEINE Historie. Ersetzt das frühere
    `lod_detail_factor = lod_level/5.0`-Hack: ein direkter Sprung auf die
    Zielauflösung ergibt dasselbe Ergebnis wie das schrittweise Durchlaufen
    aller Zwischen-LODs, weil beide Male exakt derselbe px_per_cycle-Wert
    an derselben Weltkoordinate herauskommt.
    """
    return _smoothstep((px_per_cycle - low) / (high - low))


@dataclass
class GeologyData:
    """Container für alle Geology-Daten mit Validity-System und Cache-Management."""
    rock_map: np.ndarray  # (H,W,3) uint8 RGB - Farbe der ausbeißenden Schicht/Intrusion je Pixel
    hardness_map: np.ndarray  # (H,W) float, Gesteinshärte [1-100]
    layer_id_map: np.ndarray  # (H,W) int16, Index in ALL_ROCK_TYPES (0..N_LAYERS = Basalt-Intrusion)
    lod_level: int
    actual_size: Tuple[int, int]
    validity_state: Dict[str, bool]
    parameter_hash: str
    parameters: Dict[str, Any]
    # 2D array, Höhenbeitrag zur Karte in Metern - IMMER Null (Nutzer-Korrektur:
    # die frühere gekappte Intrusions-Dom-Hebung erzeugte eine tatsächliche
    # Terrain-Erhebung, die nicht gewollt war). Tilt/Fold/Fault/Terrain-Hub/
    # Intrusion wirken ALLE nur auf den Gesteinsstapel/Ausbiss (siehe
    # delta_components unten), Geology trägt keinen Höhenbeitrag mehr zur
    # Karte bei. Feld bleibt aus API-Kompatibilität erhalten (DataLODManager.
    # get_terrain_data_combined() addiert es weiterhin, ist aber strukturell
    # ein No-Op) - siehe DataLODManager.get_geology_height_delta().
    height_delta: Optional[np.ndarray] = None
    fault_distance_map: Optional[np.ndarray] = None  # (H,W) km, Abstand zur nächsten Störung
    intrusion_distance_map: Optional[np.ndarray] = None  # (H,W) km, signiert (<0 = innerhalb einer Intrusion)
    metamorphic_grade_map: Optional[np.ndarray] = None  # (H,W) [0-1]
    layer_boundaries: Optional[np.ndarray] = None  # (N_LAYERS,H,W) m, deformierte Schicht-Obergrenzen - Basis für Cross-Section-View
    # {"terrain_hub":.., "tilt":.., "fold":.., "fault":.., "intrusion":..},
    # je (H,W) - für die isolierten Diagnose-Anzeigemodi in geology_tab.py.
    # Reine Gesteinsstapel-Diagnosen, KEINE davon wirkt auf die Kartenhöhe
    # (auch "intrusion" nicht mehr, siehe height_delta oben).
    delta_components: Optional[Dict[str, np.ndarray]] = None

    def is_valid(self) -> bool:
        return all(self.validity_state.values())

    def invalidate(self):
        self.validity_state = {key: False for key in self.validity_state.keys()}

    def validate_against_parameters(self, new_parameters: Dict[str, Any]) -> bool:
        critical_params = [
            'sedimentary_hardness', 'igneous_hardness', 'metamorphic_hardness',
            'tilt_intensity', 'tilt_direction', 'fold_intensity', 'fold_detail',
            'fault_intensity', 'fault_detail', 'fault_edge_softness',
            'intrusion_density', 'intrusion_size', 'intrusion_detail',
            'metamorphic_overprint_intensity', 'foliation_detail',
        ]
        for param in critical_params:
            if abs(self.parameters.get(param, 0) - new_parameters.get(param, 0)) > 0.01:
                return False
        return True

    def get_validity_summary(self) -> Dict[str, str]:
        return {
            'overall_valid': str(self.is_valid()),
            'hardness_range': str(self.validity_state.get('hardness_range', False)),
            'layer_assignment': str(self.validity_state.get('layer_assignment', False)),
        }


# =============================================================================
# SCHICHTDICKEN (Plan Punkt 2)
# =============================================================================

class LayerThicknessBuilder:
    """
    Baut die räumlich variierende, aber budget-begrenzte Schichtdicke jeder
    ROCK_LAYERS-Formation, kombiniert aus zwei Effekten (Teil-2-Rework,
    Klärungsrunde: "Rauschen UND Slope zusammen"):
    1. Milde, niederfrequente Rausch-Variation pro Schicht (Noise,
       UV-normiert - Schichtdicke selbst ist kein tektonischer Effekt, für
       den eine km-Wellenlänge nötig wäre), anschließend proportionale
       Reskalierung ALLER Schichten auf die nominale Gesamttiefe
       (NOMINAL_TOTAL_DEPTH_M) - dasselbe Prinzip wie die frühere
       rock_map-Mass-Conservation (R+G+B=255), jetzt auf Schichtdicken
       angewendet.
    2. Slope-Verdünnung: an steilen Hängen werden ALLE Schichten an diesem
       Punkt gleichermaßen dünner gezeichnet (stärkere Erosion), was das
       Relief optisch verstärkt. Wirkt NACH der Budget-Reskalierung, wird
       NICHT erneut renormiert - die Verdünnung ist gewollt, kein
       Rauschen, das korrigiert werden müsste.
    Vollständig vektorisiert.
    """

    def __init__(self, map_seed: int):
        self.set_seed(map_seed)

    def set_seed(self, map_seed: int):
        """Erneuert alle Schicht-Rauschgeneratoren mit einem neuen Seed - nötig,
        weil GeologySystemGenerator (und damit dieser Builder) vom
        GenerationOrchestrator einmalig lazy instanziiert und für die
        gesamte App-Session wiederverwendet wird (siehe GeologySystem
        Generator.set_active_parameters()); ohne dies bliebe die
        Schichtdicken-Verteilung permanent beim Konstruktions-Seed hängen,
        unabhängig von späteren Map-Seed-Änderungen."""
        self._noise = [OpenSimplex(seed=(map_seed + 100 + i) & 0xFFFFFFFF) for i in range(N_LAYERS)]

    def build(self, shape: Tuple[int, int], map_distance_km: float, slopemap: np.ndarray) -> np.ndarray:
        height, width = shape
        norm_x = np.arange(width, dtype=np.float64) / width
        norm_y = np.arange(height, dtype=np.float64) / height

        thickness = np.empty((N_LAYERS, height, width), dtype=np.float64)
        for i, layer in enumerate(ROCK_LAYERS):
            field = np.clip(self._noise[i].noise2array(norm_x * 3.0, norm_y * 3.0), -1.0, 1.0)
            # Asymmetrische Variation [-0.4, +0.6] um die Basis-Dicke (Plan Punkt 2)
            variation = np.where(field >= 0, field * 0.6, field * 0.4)
            thickness[i] = layer.base_thickness_m * (1.0 + variation)

        total = np.sum(thickness, axis=0)
        total = np.where(total <= 1e-6, 1.0, total)
        scale = NOMINAL_TOTAL_DEPTH_M / total
        thickness *= scale[None, :, :]

        thin_factor = _compute_slope_thin_factor(slopemap)
        thickness *= thin_factor[None, :, :]
        return thickness.astype(np.float32)


def _compute_slope_thin_factor(slopemap: np.ndarray) -> np.ndarray:
    """
    Slope-getriebene Verdünnung der Schichtdicke (Teil-2-Rework, Punkt A4):
    an steilen Hängen dünner (stärkere Erosion), in flachen Bereichen volle
    Mächtigkeit - verstärkt das Relief optisch. `slopemap` ist (H,W,2)
    dz/dx,dz/dy (terrain.slope-Output); dieselbe "maximale Steigung ~2.0"-
    Normierung wie im Vorgänger-Modell (RockTypeClassifier.
    apply_slope_hardening).
    """
    slope_magnitude = np.hypot(slopemap[..., 0], slopemap[..., 1])
    slope_norm = np.clip(slope_magnitude / SLOPE_REFERENCE_MAGNITUDE, 0.0, 1.0)
    return 1.0 - slope_norm * (1.0 - MIN_THICKNESS_FRACTION_AT_MAX_SLOPE)


# =============================================================================
# TEKTONIK-VERSCHIEBUNGSFELD Δz (Plan Punkt 3)
# =============================================================================

def _fault_segments(map_distance_km: float, map_seed: int,
                    fault_intensity: float, fault_detail: float
                    ) -> List[Tuple[float, float, float, float, float, float]]:
    """
    Die Liniengeometrie des Störungsnetzes: (x0, y0, x1, y1, throw0, throw1).

    Herausgezogen, damit die schnelle und die ausführliche Fassung von
    `_build_fault_field` GARANTIERT dieselben Störungen bekommen - sonst
    verglichen die Messungen zwei verschiedene Welten miteinander. Die
    Zufallsfolge bleibt Zeile für Zeile die alte.
    """
    rng = np.random.RandomState((map_seed + 5000) & 0xFFFFFFFF)
    n_seeds = max(1, int(round(2 + fault_detail * 3)))
    max_depth = max(1, int(round(1 + fault_detail * 3)))
    segments: List[Tuple[float, float, float, float, float, float]] = []

    def branch(x, y, angle_deg, length_km, depth, throw_scale):
        if length_km <= 0.05 or depth < 0:
            return
        angle = angle_deg + rng.uniform(-18, 18)
        x2 = x + length_km * math.cos(math.radians(angle))
        y2 = y + length_km * math.sin(math.radians(angle))
        throw_start = fault_intensity * throw_scale
        # Tapering zur Astspitze hin auf 0 (Frage 7 - auslaufender Versatz)
        throw_end = throw_start * (0.55 if depth > 0 else 0.0)
        segments.append((x, y, x2, y2, throw_start, throw_end))
        if depth <= 0:
            return
        n_children = 1 if rng.uniform() < 0.35 else 2
        for i in range(n_children):
            child_angle = angle + rng.uniform(15, 45) * (1 if i == 0 else -1)
            branch(x2, y2, child_angle, length_km * 0.68, depth - 1, throw_scale * 0.68)

    for _ in range(n_seeds):
        sx = rng.uniform(0, map_distance_km)
        sy = rng.uniform(0, map_distance_km)
        a0 = rng.uniform(0, 360)
        branch(sx, sy, a0, map_distance_km * 0.22, max_depth, 1.0)
    return segments


def _build_fault_field(shape: Tuple[int, int], map_distance_km: float,
                       map_seed: int, fault_intensity: float,
                       fault_detail: float, fault_edge_softness: float
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dasselbe Störungsfeld wie `_build_fault_field_referenz`, aber kachelweise -
    und BITGLEICH zu ihr.

    WARUM. Die ausführliche Fassung darunter schleift über alle Segmente und
    rechnet je Segment rund elf volle Kartenarrays, um am Ende nur das
    NÄCHSTGELEGENE zu behalten. Der Aufwand ist Segmentzahl mal Pixelzahl - bei
    fault_detail 1.0 sind das etwa 85 Segmente, bei 2048 px also Gigabytes
    Speicherverkehr auf einem Kern. Gemessen 29.1 s bei 2048 px.

    EIN ERSTER ANLAUF ÜBER EINE ABSTANDSTRANSFORMATION war 23-fach schneller,
    aber falsch: er zeichnete die Segmente ins Raster und las aus dem
    getroffenen Quellpixel ab, welches Segment gemeint war. An den
    Verzweigungsknoten liegen mehrere Segmente fast gleich weit entfernt, das
    Raster entschied sich dort für ein anderes als die exakte Rechnung - und
    weil über jeder Störungslinie das VORZEICHEN kippt, waren das keine
    Rundungsreste, sondern volle Sprünge: 9 m mittlere und 195 m größte
    Abweichung bei 200 m Spanne. Jede bestehende Karte hätte anders ausgesehen.

    WAS STATTDESSEN WIRKT. Nicht jedes Segment kann für jedes Pixel das nächste
    sein. Die Karte wird deshalb in Kacheln zerlegt, und je Kachel bleiben nur
    die Segmente übrig, die überhaupt in Frage kommen:

      - untere Schranke: der Abstand zwischen Kachel- und Segmentrechteck ist
        nie größer als der echte Abstand zum Segment;
      - obere Schranke: der Abstand zu einem Segment ist eine KONVEXE Funktion
        des Punktes, sein Größtwert über eine Kachel sitzt also immer in einer
        Ecke - vier Auswertungen genügen, und der Wert ist exakt.

    Ein Segment fliegt raus, sobald seine untere Schranke über der kleinsten
    oberen aller Segmente liegt. Das ist eine reine Vorauswahl; was übrig
    bleibt, wird mit derselben Formel wie vorher gerechnet. Das Ergebnis ist
    daher identisch, nicht nur ähnlich.
    """
    height, width = shape
    if fault_intensity <= 0.0:
        return (np.zeros(shape, dtype=np.float32),
                np.full(shape, map_distance_km, dtype=np.float32))

    segments = _fault_segments(map_distance_km, map_seed, fault_intensity,
                               fault_detail)
    if not segments:
        return (np.zeros(shape, dtype=np.float32),
                np.full(shape, map_distance_km, dtype=np.float32))

    seg = np.asarray(segments, dtype=np.float64)          # (S, 6)
    sx0, sy0, sx1, sy1, st0, st1 = (seg[:, i] for i in range(6))
    sdx, sdy = sx1 - sx0, sy1 - sy0
    laenge2 = np.maximum(sdx * sdx + sdy * sdy, 1e-12)
    lebt = laenge2 > 1e-9
    # Rechteck je Segment - Grundlage der unteren Schranke.
    rx0, rx1 = np.minimum(sx0, sx1), np.maximum(sx0, sx1)
    ry0, ry1 = np.minimum(sy0, sy1), np.maximum(sy0, sy1)

    def abstand_zu(px, py):
        """Exakter Punkt-zu-Segment-Abstand, Punkte gegen alle Segmente."""
        p = np.asarray(px)[:, None]
        q = np.asarray(py)[:, None]
        t = np.clip(((p - sx0) * sdx + (q - sy0) * sdy) / laenge2, 0.0, 1.0)
        return np.hypot(p - (sx0 + t * sdx), q - (sy0 + t * sdy))

    x_km = (np.arange(width, dtype=np.float64) / width) * map_distance_km
    y_km = (np.arange(height, dtype=np.float64) / height) * map_distance_km

    min_abs_dist = np.full(shape, np.inf, dtype=np.float64)
    best_signed_dist = np.zeros(shape, dtype=np.float64)
    best_throw_mag = np.zeros(shape, dtype=np.float64)

    kachel = FAULT_KACHEL
    for y_a in range(0, height, kachel):
        y_e = min(y_a + kachel, height)
        for x_a in range(0, width, kachel):
            x_e = min(x_a + kachel, width)
            bx0, bx1 = x_km[x_a], x_km[x_e - 1]
            by0, by1 = y_km[y_a], y_km[y_e - 1]

            # Untere Schranke: Rechteck gegen Rechteck.
            luecke_x = np.maximum(np.maximum(bx0 - rx1, rx0 - bx1), 0.0)
            luecke_y = np.maximum(np.maximum(by0 - ry1, ry0 - by1), 0.0)
            unten = np.hypot(luecke_x, luecke_y)
            # Obere Schranke: Größtwert sitzt in einer Ecke (Konvexität).
            ecken = abstand_zu([bx0, bx1, bx0, bx1], [by0, by0, by1, by1])
            oben = ecken.max(axis=0)
            schwelle = float(np.min(np.where(lebt, oben, np.inf)))
            kandidaten = np.nonzero(lebt & (unten <= schwelle))[0]

            X, Y = np.meshgrid(x_km[x_a:x_e], y_km[y_a:y_e])
            nah = min_abs_dist[y_a:y_e, x_a:x_e]
            vz = best_signed_dist[y_a:y_e, x_a:x_e]
            wurf = best_throw_mag[y_a:y_e, x_a:x_e]
            for i in kandidaten:
                t = ((X - sx0[i]) * sdx[i] + (Y - sy0[i]) * sdy[i]) / laenge2[i]
                tc = np.clip(t, 0.0, 1.0)
                dist = np.hypot(X - (sx0[i] + tc * sdx[i]),
                                Y - (sy0[i] + tc * sdy[i]))
                seite = (X - sx0[i]) * sdy[i] - (Y - sy0[i]) * sdx[i]
                naeher = dist < nah
                nah = np.where(naeher, dist, nah)
                vz = np.where(naeher, np.sign(seite) * dist, vz)
                wurf = np.where(naeher, st0[i] + (st1[i] - st0[i]) * tc, wurf)
            min_abs_dist[y_a:y_e, x_a:x_e] = nah
            best_signed_dist[y_a:y_e, x_a:x_e] = vz
            best_throw_mag[y_a:y_e, x_a:x_e] = wurf

    edge_width_km = max(0.02, fault_edge_softness)
    fault_throw = np.tanh(best_signed_dist / edge_width_km) * best_throw_mag
    return fault_throw.astype(np.float32), min_abs_dist.astype(np.float32)


def _build_fault_field_referenz(shape: Tuple[int, int], map_distance_km: float,
                                map_seed: int, fault_intensity: float,
                                fault_detail: float, fault_edge_softness: float
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    DIE VORLAGE, nicht mehr im Betrieb - sie schleift über ALLE Segmente und
    ist damit die einfach nachzulesende Fassung, gegen die
    `_build_fault_field` in tests/smoke_test_geology_speed.py geprüft wird.

    Störungsnetz: wenige Saatpunkte -> L-System-artige rekursive Verzweigung
    (Rekursionstiefe = fault_detail) mit an den Astspitzen auslaufendem
    Versatz -> pro Pixel nächstgelegenes Segment (vektorisierte
    Punkt-zu-Segment-Distanz) bestimmt Vorzeichen-Seite und Versatzgröße.
    fault_edge_softness (km) ersetzt das frühere bevel_warping als Breite der
    weichen tanh-Übergangszone an der Bruchkante statt eines harten Sprungs.
    Segmentwinkel werden bei jedem Verzweigungsschritt zufällig verzerrt -
    ein leichtgewichtiger Ersatz für eine Perlin-Rauschverzerrung entlang der
    Linie, ohne dass Liniengeometrie und Rauschfeld getrennt gepflegt werden
    müssen.

    Rückgabe: (fault_throw (H,W) m, fault_distance_map (H,W) km, unsigniert).
    """
    height, width = shape
    if fault_intensity <= 0.0:
        return (np.zeros(shape, dtype=np.float32),
                np.full(shape, map_distance_km, dtype=np.float32))

    x_km = (np.arange(width, dtype=np.float64) / width) * map_distance_km
    y_km = (np.arange(height, dtype=np.float64) / height) * map_distance_km
    X, Y = np.meshgrid(x_km, y_km)

    segments = _fault_segments(map_distance_km, map_seed, fault_intensity,
                               fault_detail)

    min_abs_dist = np.full(shape, np.inf, dtype=np.float64)
    best_signed_dist = np.zeros(shape, dtype=np.float64)
    best_throw_mag = np.zeros(shape, dtype=np.float64)

    for (x0, y0, x1, y1, t0, t1) in segments:
        dx, dy = x1 - x0, y1 - y0
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-9:
            continue
        t = ((X - x0) * dx + (Y - y0) * dy) / seg_len2
        tc = np.clip(t, 0.0, 1.0)
        proj_x = x0 + tc * dx
        proj_y = y0 + tc * dy
        dist = np.hypot(X - proj_x, Y - proj_y)
        side = (X - x0) * dy - (Y - y0) * dx  # Vorzeichen = Seite der Linie
        signed_dist = np.sign(side) * dist
        throw_mag = t0 + (t1 - t0) * tc  # Tapering entlang des Segments
        closer = dist < min_abs_dist
        min_abs_dist = np.where(closer, dist, min_abs_dist)
        best_signed_dist = np.where(closer, signed_dist, best_signed_dist)
        best_throw_mag = np.where(closer, throw_mag, best_throw_mag)

    edge_width_km = max(0.02, fault_edge_softness)
    fault_throw = np.tanh(best_signed_dist / edge_width_km) * best_throw_mag
    return fault_throw.astype(np.float32), min_abs_dist.astype(np.float32)


def _build_terrain_hub(terrain_height: np.ndarray, map_distance_km: float) -> np.ndarray:
    """
    Terrain-Hub (Teil-2-Rework, Punkt A2): eine regional geglättete Version
    der realen Terrainhöhe - zieht den ganzen Schichtstapel lokal mit an,
    wo das Terrain großräumig höher liegt ("Berge heben die Schichtung mit
    an"). Bewusst GEGLÄTTET statt der exakten Terrainhöhe: würde man den
    Stapel um die exakte Terrainhöhe verschieben UND später mit derselben
    exakten Terrainhöhe schneiden, würde sich die Verschiebung beim
    Vergleich algebraisch herauskürzen (siehe _compute_outcrop()) - der
    Effekt wäre unsichtbar. Die geglättete Version lässt lokale Gipfel/
    Täler von ihrem eigenen regionalen Mittel abweichen, wodurch der
    Ausbiss-Schnitt tatsächlich variiert.

    AUF GROBEM GITTER GERECHNET (2026-08-10)
    ----------------------------------------
    Der Nutzer meldete, dass Geology "bei der Größe sehr langsam geworden" ist
    und die Oberfläche bei `stack_deformation` stehenbleibt. Gemessen war das
    hier: 0.03 s bei 256 px, 0.19 bei 512, 2.08 bei 1024, 16.6 bei 2048 - also
    Wachstum mit der DRITTEN Potenz der Kantenlänge.

    Der Grund steht in der Zeile darunter: `sigma_px` ist ein fester Bruchteil
    der Kantenlänge (0.2), und SciPys `gaussian_filter` arbeitet mit einem Kern
    von 4·sigma Radius. Der Aufwand ist damit Pixelzahl mal sigma, und sigma
    wächst selbst mit der Kantenlänge. Bei 4096 px wären es über zwei Minuten -
    für EINE Glättung.

    Eine Glättung über 0.2 der Karte hat unterhalb von etwa sigma/4 keine
    Struktur mehr. Sie darf deshalb auf einem gröberen Gitter gerechnet und
    wieder hochgezogen werden - dieselbe Überlegung wie beim Schattenwurf
    (SCHATTEN_TEILER in terrain_generator). Gemessen bei 2048 px: 16.6 s auf
    0.52 s, also 32-fach, bei 0.03 m Abweichung gegen eine Feldspanne von 8 m.

    Zwei Fallstricke, beide beim ersten Versuch getroffen (97 m Fehler bei
    39 m Spanne): verkleinert werden muss mit dem BLOCKMITTEL, nicht durch
    Abtasten - `zoom` würde genau die Feinstruktur wegwerfen, die die Glättung
    mitteln soll -, und die Randbehandlung muss dieselbe sein wie vorher
    (`reflect`), denn bei diesem sigma macht der Rand den Großteil aus.
    """
    _height, width = terrain_height.shape
    sigma_km = max(0.5, REGIONAL_HUB_SIGMA_FRACTION * map_distance_km)
    sigma_px = max(1.0, sigma_km * (width / map_distance_km))
    return grosse_glaettung(terrain_height.astype(np.float64), sigma_px)


class TectonicDisplacementField:
    """
    Baut `stack_deformation`(x,y) = terrain_hub + tilt + fold + fault_throw
    (Teil-2-Rework) - verschiebt AUSSCHLIESSLICH den Schichtstapel für die
    Ausbiss-Berechnung (siehe _compute_outcrop()), NIE additiv in
    `height_delta` (Nutzer-Vorgabe: Störungen/Tektonik dürfen die
    sichtbare Terrainhöhe nicht verändern - dasselbe gilt inzwischen auch
    für Intrusionen, siehe GeologySystemGenerator._calc_intrusions()).
    Ersetzt
    die fünf früher unabhängigen Ridge/Bevel/Foliation/Folding/Igneous-
    Flowing-Formeln des Vorgänger-Modells durch benannte Komponenten EINES
    Felds.
    """

    def __init__(self, map_seed: int):
        self.set_seed(map_seed)

    def set_seed(self, map_seed: int):
        """Erneuert Fold-Rauschgeneratoren + den intern für _build_fault_field()
        verwendeten Seed - siehe LayerThicknessBuilder.set_seed()-Docstring
        für den Grund (lazy-instanziierter, session-lang wiederverwendeter
        Generator)."""
        self.map_seed = map_seed
        self._fold_broad_noise = OpenSimplex(seed=(map_seed + 6000) & 0xFFFFFFFF)
        self._fold_fine_noise = OpenSimplex(seed=(map_seed + 6100) & 0xFFFFFFFF)

    def build(self, terrain_height: np.ndarray, map_distance_km: float,
              parameters: Dict[str, Any]) -> Dict[str, np.ndarray]:
        shape = terrain_height.shape
        terrain_hub = _build_terrain_hub(terrain_height, map_distance_km)
        tilt = self._build_tilt(shape, map_distance_km, parameters)
        fold = self._build_fold(shape, map_distance_km, parameters)
        fault_throw, fault_distance_km = _build_fault_field(
            shape, map_distance_km, self.map_seed,
            parameters.get('fault_intensity', 0.0),
            parameters.get('fault_detail', 0.0),
            parameters.get('fault_edge_softness', 0.3))

        stack_deformation = (terrain_hub + tilt + fold + fault_throw).astype(np.float32)
        return {
            "stack_deformation": stack_deformation,
            "terrain_hub_delta": terrain_hub.astype(np.float32),
            "tilt_delta": tilt.astype(np.float32),
            "fold_delta": fold.astype(np.float32),
            "fault_delta": fault_throw.astype(np.float32),
            "fault_distance_map": fault_distance_km.astype(np.float32),
        }

    def _build_tilt(self, shape, map_distance_km, parameters) -> np.ndarray:
        """Exakte Ebenen-Verkippung - geschlossene Form, praktisch kostenlos
        (Frage 4: "so genau wie möglich, wenn's nichts kostet")."""
        height, width = shape
        intensity = parameters.get('tilt_intensity', 0.0)  # m pro km Gradient
        if intensity <= 0.0:
            return np.zeros(shape, dtype=np.float64)
        direction_deg = parameters.get('tilt_direction', 0.0)
        x_km = (np.arange(width, dtype=np.float64) / width) * map_distance_km
        y_km = (np.arange(height, dtype=np.float64) / height) * map_distance_km
        X, Y = np.meshgrid(x_km, y_km)
        gx = intensity * math.cos(math.radians(direction_deg))
        gy = intensity * math.sin(math.radians(direction_deg))
        tilt = gx * X + gy * Y
        return tilt - np.mean(tilt)  # reine Rotation um den Schwerpunkt, kein Netto-Hub

    def _build_fold(self, shape, map_distance_km, parameters) -> np.ndarray:
        """
        EIN Fold-Effekt mit Intensity (Amplitude) + Detail (mischt eine
        höherfrequente Rauheits-Komponente bei) - absorbiert sowohl
        metamorph_folding als auch ridge_warping aus dem Vorgänger-Modell
        (Frage 8: Ridge wird Teil von Folding). Wellenlänge ist ein fester
        Bruchteil von map_distance_km -> Faltung sieht bei jeder Weltgröße
        proportional gleich aus, die absolute km-Wellenlänge skaliert mit
        der Weltgröße mit (Frage 13).
        """
        height, width = shape
        intensity = parameters.get('fold_intensity', 0.0)
        if intensity <= 0.0:
            return np.zeros(shape, dtype=np.float64)
        detail = parameters.get('fold_detail', 0.0)

        norm_x = np.arange(width, dtype=np.float64) / width
        norm_y = np.arange(height, dtype=np.float64) / height
        broad_wavelength_km = max(1.0, map_distance_km * 0.45)
        fine_wavelength_km = max(0.2, broad_wavelength_km / 6.0)

        broad = self._fold_broad_noise.noise2array(
            norm_x * (map_distance_km / broad_wavelength_km),
            norm_y * (map_distance_km / broad_wavelength_km))

        fine_px_per_cycle = (width / map_distance_km) * fine_wavelength_km
        fine_fade = _resolvability_fade(fine_px_per_cycle)
        fine = self._fold_fine_noise.noise2array(
            norm_x * (map_distance_km / fine_wavelength_km),
            norm_y * (map_distance_km / fine_wavelength_km))

        return intensity * (broad + 0.6 * detail * fine_fade * fine)


# =============================================================================
# INTRUSIONEN (Plan Punkt 5)
# =============================================================================

def _build_intrusion_field(shape: Tuple[int, int], map_distance_km: float, map_seed: int,
                            height_range: float, intrusion_density: float,
                            intrusion_size: float, intrusion_detail: float
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    N lokal begrenzte Blob-Körper (Position, Radius, randverzerrt durch
    zwei überlagerte Rausch-Oktaven statt perfektem Kreis - bis zu 90%
    Radius-Auslenkung bei intrusion_detail=1.0 für deutlich amöbenhafte,
    unförmige Konturen statt nur leichter Welligkeit) - überschreiben lokal
    den Ausbiss mit Basalt (siehe _apply_intrusions_to_layer_id) und tragen
    einen kleinen, gekappten Dom-Hebungs-Beitrag zu Δz bei.

    Rückgabe: (intrusion_distance_map (H,W) km, signiert, <0 = innerhalb;
    intrusion_delta (H,W) m).
    """
    height, width = shape
    rng = np.random.RandomState((map_seed + 7000) & 0xFFFFFFFF)
    n_blobs = max(0, int(round(intrusion_density * 6)))
    base_radius_km = max(0.2, intrusion_size)

    x_km = (np.arange(width, dtype=np.float64) / width) * map_distance_km
    y_km = (np.arange(height, dtype=np.float64) / height) * map_distance_km
    min_signed = np.full(shape, np.inf, dtype=np.float64)

    if n_blobs > 0:
        # Zwei überlagerte Rausch-Oktaven (grob+fein, wie beim Fold-Detail-
        # Muster) statt einem einzigen glatten Oktav - ein Einzel-Oktav mit
        # max. 30% Radius-Amplitude ergab nur leichte Welligkeit, keine
        # echte Unförmigkeit (Nutzer-Feedback: "selbst auf 0.9 noch sehr
        # rund"). Amplitude jetzt bis 90% des Radius bei intrusion_detail=1.0,
        # mit einer moderaten Basis-Unregelmäßigkeit (25%) schon bei 0.
        edge_noise_broad = OpenSimplex(seed=(map_seed + 7100) & 0xFFFFFFFF)
        edge_noise_fine = OpenSimplex(seed=(map_seed + 7200) & 0xFFFFFFFF)
        for _ in range(n_blobs):
            cx = rng.uniform(0, map_distance_km)
            cy = rng.uniform(0, map_distance_km)
            radius = base_radius_km * rng.uniform(0.6, 1.4)
            dx_1d = x_km - cx
            dy_1d = y_km - cy
            dist = np.hypot(dx_1d[None, :], dy_1d[:, None])

            wavelength_broad = max(0.1, radius * 0.6)
            wavelength_fine = max(0.05, radius * 0.22)
            fade_broad = _resolvability_fade((width / map_distance_km) * wavelength_broad)
            fade_fine = _resolvability_fade((width / map_distance_km) * wavelength_fine)
            field_broad = edge_noise_broad.noise2array(dx_1d / wavelength_broad, dy_1d / wavelength_broad)
            field_fine = edge_noise_fine.noise2array(dx_1d / wavelength_fine, dy_1d / wavelength_fine)
            wobble = 0.65 * fade_broad * field_broad + 0.35 * fade_fine * field_fine
            wobble_amplitude = radius * (0.25 + 0.65 * intrusion_detail)
            signed = dist - radius + wobble_amplitude * wobble
            min_signed = np.minimum(min_signed, signed)

    transition_km = max(0.05, base_radius_km * 0.15)
    inside_amount = _smoothstep(-min_signed / transition_km)
    intrusion_delta = (INTRUSION_UPLIFT_CAP_FRACTION * height_range * inside_amount).astype(np.float32)
    return min_signed.astype(np.float32), intrusion_delta


def _apply_intrusions_to_layer_id(layer_id_map: np.ndarray, intrusion_distance_map: np.ndarray) -> np.ndarray:
    inside = intrusion_distance_map < 0
    return np.where(inside, BASALT_LAYER_ID, layer_id_map).astype(np.int16)


# =============================================================================
# AUSBISS-BERECHNUNG (Plan Punkt 7)
# =============================================================================

def _compute_layer_boundaries_base(layer_thickness: np.ndarray) -> np.ndarray:
    """
    Unverformte Schicht-Obergrenzen, von OBEN verankert: die oberste/
    jüngste Schicht (Holozän) sitzt bei TOP_REFERENCE_HEIGHT_M (ein
    positiver Headroom-Wert relativ zum Terrain-Hub, siehe core/
    geology_layers.py-Kommentar), ältere Schichten liegen darunter.
    boundary_base[i] = TOP_REFERENCE_HEIGHT_M - sum(thickness[i+1:]) für
    i<N-1, boundary_base[-1] = TOP_REFERENCE_HEIGHT_M.
    """
    cum_top_down = np.cumsum(layer_thickness[::-1], axis=0)[::-1]  # cum_top_down[i] = sum(thickness[i:])
    return TOP_REFERENCE_HEIGHT_M - (cum_top_down - layer_thickness)


def _compute_outcrop(terrain_height: np.ndarray, layer_thickness: np.ndarray,
                      stack_deformation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vergleicht die REALE Terrain-Höhe (terrain.redistribution-Output) gegen
    die deformierten Schichtgrenzen. Vollständig vektorisiert über N_LAYERS
    Vergleichsmasken, kein Pixel-Loop.

    WICHTIG - zwei Design-Punkte, beide notwendig, damit "höheres Terrain
    zeigt älteres Gestein" tatsächlich funktioniert (Nutzer-Vorgabe:
    Berge legen automatisch das tiefer liegende, harte Kristallin frei):

    1. terrain_height darf NICHT bereits stack_deformation enthalten -
       sonst kürzt sich die Verschiebung beim Vergleich mit den (ebenfalls
       verschobenen) Schichtgrenzen algebraisch heraus und hätte GAR KEINE
       Wirkung auf den Ausbiss. Das gilt insbesondere für den Terrain-Hub-
       Anteil von `stack_deformation` (_build_terrain_hub()): der ist zwar
       aus der Terrainhöhe abgeleitet, aber bewusst GEGLÄTTET.
       GeologySystemGenerator._calc_outcrop() liest deshalb bewusst den
       rohen "terrain.redistribution"-Output, NICHT
       get_calculator_combined_heightmap().
    2. Die rohe Vergleichszählung (`sum(exceeded)`) wächst mit der
       Terrainhöhe UND mit dem Schicht-Index (ROCK_LAYERS ist alt->jung
       sortiert, jüngere Schichten liegen weiter oben) - unveränderte
       Zählung würde also "höheres Terrain -> jüngere Schicht" ergeben,
       das GEGENTEIL des gewünschten Verhaltens. Deshalb wird der Index
       am Ende gespiegelt (`N_LAYERS - 1 - Zählung`): höheres Terrain
       exceeded mehr Grenzen -> nach Spiegelung ein NIEDRIGERER Index ->
       älteres Gestein.
    """
    boundaries_base = _compute_layer_boundaries_base(layer_thickness)  # (N_LAYERS,H,W)
    boundaries_deformed = boundaries_base + stack_deformation[None, :, :]
    exceeded_count = np.sum(terrain_height[None, :, :] >= boundaries_deformed, axis=0)
    layer_id_map = np.clip(N_LAYERS - 1 - exceeded_count, 0, N_LAYERS - 1).astype(np.int16)
    return layer_id_map, boundaries_deformed.astype(np.float32)


def _apply_sediment_overlay(layer_id_map: np.ndarray, heightmap_combined: np.ndarray,
                             map_distance_km: float) -> np.ndarray:
    """
    Rein Gesteinstyp-/Farb-Klassifikation in erkannten Senken/Tälern - trägt
    ABSICHTLICH keinen Höhenbeitrag bei (Plan Punkt 8, nach Nutzer-Hinweis
    korrigiert: Water berechnet Erosion/Sedimentation bereits als eigenes
    Höhen-Delta; ein zusätzlicher Höhenbeitrag hier würde sich damit
    verdoppeln). Erkennung über einen Relief-Proxy direkt aus der
    Heightmap, km-skaliert statt fixem Pixel-Radius (Frage 13) - keine
    neue Water-Abhängigkeit nötig.
    """
    height, width = heightmap_combined.shape
    sigma_km = 2.0  # feste Glättungsbreite, kein eigener Slider (Design-Entscheidung)
    sigma_px = max(1.0, sigma_km * (width / map_distance_km))
    # Grobgitter, siehe grosse_glaettung(): auch hier ist sigma ein fester
    # Bruchteil der Karte (2 von 21.3 km) und der Aufwand wüchse sonst kubisch.
    smoothed = grosse_glaettung(heightmap_combined.astype(np.float64), sigma_px)
    relief = heightmap_combined.astype(np.float64) - smoothed
    span = float(np.max(heightmap_combined) - np.min(heightmap_combined))
    threshold = -0.02 * (span if span > 1e-6 else 1.0)
    is_valley = relief < threshold
    holozaen_id = N_LAYERS - 1  # letzter (jüngster) Eintrag in ROCK_LAYERS
    return np.where(is_valley, holozaen_id, layer_id_map).astype(np.int16)


# =============================================================================
# METAMORPHOSE (Plan Punkt 6, eigene Entscheidung zu Fragen 10/11)
# =============================================================================

def _compute_metamorphic_grade(fault_distance_km: np.ndarray, intrusion_distance_km: np.ndarray,
                                overprint_intensity: float) -> np.ndarray:
    """
    Metamorpher Grad [0-1] = Nähe zu Störungen (Regional-/Dynamometamorphose)
    ODER Nähe zu Intrusionen (Kontaktmetamorphose), beide Ursachen kombiniert
    (eigene Entscheidung zu Frage 10 - dient dem vom Nutzer gewünschten
    Realismus für eine spätere Ressourcen-Ableitung).
    """
    if overprint_intensity <= 0.0:
        return np.zeros_like(fault_distance_km, dtype=np.float32)
    fault_scale_km = 1.0 + 4.0 * overprint_intensity
    intrusion_scale_km = 0.5 + 2.0 * overprint_intensity
    regional = np.exp(-np.maximum(fault_distance_km, 0.0) / fault_scale_km)
    contact = np.exp(-np.maximum(intrusion_distance_km, 0.0) / intrusion_scale_km)
    grade = np.clip(np.maximum(regional, contact) * overprint_intensity, 0.0, 1.0)
    return grade.astype(np.float32)


# =============================================================================
# EINFÄRBUNG UND HÄRTE
# =============================================================================

def _build_rock_map(layer_id_map: np.ndarray, metamorphic_grade: np.ndarray, foliation_detail: float,
                     map_distance_km: float, map_seed: int) -> np.ndarray:
    """
    Farbe der ausbeißenden Schicht/Intrusion je Pixel (Lookup, kein Blending
    mehr - jeder Pixel ist genau EIN Gesteinstyp, "Mass Conservation" wie im
    Vorgänger-Modell ist damit gegenstandslos). Foliation ist rein visuell/
    texturell (Frage 11, eigene Entscheidung): feines Streifenmuster,
    moduliert mit dem metamorphen Grad, OHNE jede Höhenwirkung.
    """
    height, width = layer_id_map.shape
    colors = np.array([layer.color for layer in ALL_ROCK_TYPES], dtype=np.float32)  # (N_LAYERS+1, 3)
    rock_map = colors[layer_id_map]  # Fancy-Indexing -> (H,W,3)

    if foliation_detail > 0.0:
        norm_x = np.arange(width, dtype=np.float64) / width
        norm_y = np.arange(height, dtype=np.float64) / height
        wavelength_km = max(0.05, 0.5 / (1.0 + foliation_detail * 9.0))
        px_per_cycle = (width / map_distance_km) * wavelength_km
        fade = _resolvability_fade(px_per_cycle)
        noise = OpenSimplex(seed=(map_seed + 8000) & 0xFFFFFFFF)
        stripe = noise.noise2array(
            norm_x * (map_distance_km / wavelength_km),
            norm_y * (map_distance_km / wavelength_km))
        modulation = (1.0 + 0.15 * foliation_detail * fade * stripe * metamorphic_grade)[:, :, None]
        rock_map = np.clip(rock_map * modulation, 0, 255)

    return rock_map.astype(np.uint8)


def _build_hardness_map(layer_id_map: np.ndarray, metamorphic_grade: np.ndarray, terrain_height: np.ndarray,
                         sedimentary_hardness: float, igneous_hardness: float,
                         metamorphic_hardness: float) -> np.ndarray:
    """
    Härte = Kategorie-Härte der ausbeißenden Schicht (Lookup über die feste
    Kategorie-Zuordnung in ROCK_LAYERS, Frage 19: nur 3 Haupt-Regler) MAL
    einem festen, geologisch motivierten Härte-Faktor je Einzelschicht
    (RockLayer.hardness_factor, siehe core/geology_layers.py) - ohne diesen
    Faktor hätten alle 11 sedimentären Schichten exakt dieselbe Härte
    (Nutzer-Feedback: "zu unvariabel"). Die relativen Stufen zwischen den
    Formationen (z.B. Muschelkalk härter als Keuper) bleiben dadurch immer
    erhalten, während die 3 Slider weiterhin die GESAMTE Bandbreite je
    Kategorie skalieren. Zum metamorphen Grad hin auf metamorphic_hardness
    verschoben, plus milde Höhen-Tendenz wie im Vorgänger-Modell. Die
    frühere, rein noise-basierte 9-Tier-Hardness-Maske entfällt weiterhin -
    die Schicht-Identität liefert die reale räumliche Struktur, die die
    Maske künstlich simulieren musste (Plan Punkt 9).
    """
    hardness_by_category = {
        "sedimentary": sedimentary_hardness,
        "igneous": igneous_hardness,
        "metamorphic": metamorphic_hardness,
    }
    category_hardness_lut = np.array(
        [hardness_by_category[layer.category] * layer.hardness_factor for layer in ALL_ROCK_TYPES],
        dtype=np.float32)
    base_hardness = category_hardness_lut[layer_id_map]

    blended = base_hardness * (1.0 - metamorphic_grade) + metamorphic_hardness * metamorphic_grade

    min_h, max_h = float(np.min(terrain_height)), float(np.max(terrain_height))
    height_range = max_h - min_h if max_h > min_h else 1.0
    norm_height = (terrain_height - min_h) / height_range
    elevation_factor = 0.85 + 0.15 * norm_height

    hardness_map = np.clip(blended * elevation_factor, 1.0, 100.0)
    return hardness_map.astype(np.float32)


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class GeologySystemGenerator:
    """
    Hauptklasse für den 3D-Gesteinsstapel mit vollständiger LOD-Integration.
    """

    def __init__(self, map_seed: int = 42, data_lod_manager=None, shader_manager=None):
        """
        Args:
            map_seed: Globaler Seed für reproduzierbare Geologie
            data_lod_manager: DataLODManager für feingranularen Calculator-Storage
            shader_manager: Slot für einen künftigen GPU-Compute-Pfad (siehe
                Umsetzungsplan Punkt 11) - aktuell mit KEINEN echten
                Shader-Operationen hinterlegt, jede _calc_*-Methode fällt
                deshalb immer auf den CPU-Pfad zurück. Strukturell aber
                bereits GPU-portabel (reine elementweise NumPy-Operationen
                über wenige Schichten/Störungen/Intrusionen).
        """
        self.map_seed = map_seed
        self.logger = logging.getLogger(__name__)
        self.shader_manager = shader_manager

        self.thickness_builder = LayerThicknessBuilder(map_seed)
        self.displacement_builder = TectonicDisplacementField(map_seed)

        self.default_parameters = self._load_default_parameters()
        self.progress_callback = None

        self.data_lod_manager = data_lod_manager
        self._current_parameters: Dict[str, Any] = dict(self.default_parameters)

    def set_active_parameters(self, parameters: Dict[str, Any]):
        """Setzt die (mit Defaults gemergten) Parameter, die alle _calc_*-Methoden
        bis zur nächsten frischen Anfrage verwenden."""
        self._current_parameters = {**self.default_parameters, **parameters}

        # map_seed ist ein Terrain-Tab-Parameter, kein Geology-eigener (siehe
        # _load_default_parameters() oben - taucht dort nicht auf) - über
        # DataLODManager.get_map_seed() gespiegelt (analog zum map_latitude-
        # Muster für Biome/Water). Ohne diesen Refresh bliebe self.map_seed
        # (und die davon abgeleiteten, session-lang gecachten Noise-Builder)
        # permanent beim Konstruktor-Default (42) hängen, unabhängig von
        # tatsächlichen Map-Seed-Änderungen (Nutzer-Bug-Report: Intrusion/
        # Fault-Lines/Tilt sahen bei jeder Map identisch aus).
        if self.data_lod_manager is not None and hasattr(self.data_lod_manager, "get_map_seed"):
            live_seed = int(self.data_lod_manager.get_map_seed())
            if live_seed != self.map_seed:
                self.map_seed = live_seed
                self.thickness_builder.set_seed(live_seed)
                self.displacement_builder.set_seed(live_seed)

    def _ensure_data_lod_manager(self):
        if self.data_lod_manager is None:
            from managers.data_lod_manager import DataLODManager
            self.data_lod_manager = DataLODManager()
        return self.data_lod_manager

    def _get_map_distance_km(self) -> float:
        """Liest die reale Kartenausdehnung analog zum Terrain-4f-Pattern
        (core/terrain_generator.py) - DataLODManager zuerst, sonst
        TERRAIN.WORLD_SIZE_KM als Fallback für Standalone-/Test-Nutzung."""
        try:
            from gui.config.value_default import TERRAIN
            default_km = TERRAIN.WORLD_SIZE_KM
        except ImportError:
            default_km = 10.0
        if self.data_lod_manager is not None and hasattr(self.data_lod_manager, "get_map_distance_km"):
            try:
                return float(self.data_lod_manager.get_map_distance_km())
            except Exception:
                pass
        return float(self._current_parameters.get('map_distance_km', default_km))

    def _load_default_parameters(self) -> Dict[str, Any]:
        try:
            from gui.config.value_default import GEOLOGY
            return {
                'sedimentary_hardness': GEOLOGY.SEDIMENTARY_HARDNESS["default"],
                'igneous_hardness': GEOLOGY.IGNEOUS_HARDNESS["default"],
                'metamorphic_hardness': GEOLOGY.METAMORPHIC_HARDNESS["default"],
                'tilt_intensity': GEOLOGY.TILT_INTENSITY["default"],
                'tilt_direction': GEOLOGY.TILT_DIRECTION["default"],
                'fold_intensity': GEOLOGY.FOLD_INTENSITY["default"],
                'fold_detail': GEOLOGY.FOLD_DETAIL["default"],
                'fault_intensity': GEOLOGY.FAULT_INTENSITY["default"],
                'fault_detail': GEOLOGY.FAULT_DETAIL["default"],
                'fault_edge_softness': GEOLOGY.FAULT_EDGE_SOFTNESS["default"],
                'intrusion_density': GEOLOGY.INTRUSION_DENSITY["default"],
                'intrusion_size': GEOLOGY.INTRUSION_SIZE["default"],
                'intrusion_detail': GEOLOGY.INTRUSION_DETAIL["default"],
                'metamorphic_overprint_intensity': GEOLOGY.METAMORPHIC_OVERPRINT_INTENSITY["default"],
                'foliation_detail': GEOLOGY.FOLIATION_DETAIL["default"],
            }
        except ImportError:
            self.logger.warning("Could not load parameters from value_default.py, using fallback values")
            return {
                'sedimentary_hardness': 30.0, 'igneous_hardness': 80.0, 'metamorphic_hardness': 65.0,
                'tilt_intensity': 15.0, 'tilt_direction': 45.0,
                'fold_intensity': 400.0, 'fold_detail': 0.4,
                'fault_intensity': 100.0, 'fault_detail': 0.5, 'fault_edge_softness': 0.3,
                'intrusion_density': 0.3, 'intrusion_size': 1.0, 'intrusion_detail': 0.5,
                'metamorphic_overprint_intensity': 0.4, 'foliation_detail': 0.5,
            }

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def _update_progress(self, phase: str, progress: int, message: str):
        if self.progress_callback:
            try:
                self.progress_callback(phase, progress, message)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")

    def calculate_geology(self, heightmap_combined: np.ndarray, slopemap: np.ndarray,
                           parameters: Dict[str, Any], lod_level: int,
                           previous_height_delta: Optional[np.ndarray] = None) -> GeologyData:
        """
        Hauptmethode für Geology-Generierung mit vollständiger LOD-Integration.

        Hinweis: previous_height_delta wird nicht mehr verwendet (Parameter
        aus Aufruf-Kompatibilität erhalten) - das neue Δz-Feld wird pro
        LOD-Aufruf vollständig frisch aus stetigen, weltkoordinaten-basierten
        Funktionen berechnet statt wie zuvor über scipy.ndimage.zoom von der
        letzten LOD-Stufe fortgeschrieben (siehe `_resolvability_fade`) -
        macht das Ergebnis unabhängig davon, wie viele LOD-Zwischenstufen
        durchlaufen wurden.
        """
        try:
            self.logger.info(f"Starting geology generation - LOD {lod_level}, Size: {heightmap_combined.shape}")

            self._ensure_data_lod_manager()
            self.set_active_parameters(parameters)
            merged_params = self._current_parameters

            self._validate_inputs(heightmap_combined, slopemap, merged_params)

            # Standalone-Convenience-Pfad (Legacy-Kompatibilität + Tests): siehe
            # Docstring der Vorgänger-Version - heightmap_combined/slopemap
            # werden hier gespiegelt, damit die _calc_*-Methoden (die immer aus
            # dem feingranularen Calculator-Storage lesen) etwas vorfinden.
            self.data_lod_manager.set_calculator_output(
                "terrain.redistribution", lod_level, {"heightmap": heightmap_combined})
            self.data_lod_manager.set_calculator_output(
                "terrain.slope", lod_level, {"slopemap": slopemap})

            for calculator_id in (
                "geology.layer_thickness", "geology.tectonic_displacement", "geology.outcrop",
                "geology.intrusions", "geology.sediment_overlay", "geology.metamorphic_overprint",
                "geology.rock_color", "geology.hardness",
            ):
                getattr(self, "_calc_" + calculator_id.split(".", 1)[1])(calculator_id, lod_level)

            geology_data = self.assemble_geology_data(lod_level, merged_params)

            self._update_progress("Generation Complete", 100, "Geology generation completed successfully")
            self.logger.info(f"Geology generation completed - LOD {lod_level}")

            return geology_data

        except Exception as e:
            self.logger.error(f"Geology generation failed: {e}")
            return self._create_fallback_geology_data(heightmap_combined, self._current_parameters, lod_level)

    def assemble_geology_data(self, lod_level: int, parameters: Dict[str, Any]) -> GeologyData:
        """
        Baut das finale GeologyData-Objekt aus den einzeln gespeicherten
        Calculator-Outputs zusammen, sobald alle 8 Geology-Calculator-Knoten
        ein LOD abgeschlossen haben.
        """
        dlm = self.data_lod_manager
        rock_map = dlm.get_calculator_output("geology.rock_color", "rock_map", lod_level)
        hardness_map = dlm.get_calculator_output("geology.hardness", "hardness_map", lod_level)
        layer_id_map = dlm.get_calculator_output("geology.sediment_overlay", "layer_id_map", lod_level)

        if rock_map is None or hardness_map is None or layer_id_map is None:
            raise ValueError(f"assemble_geology_data: fehlende Calculator-Outputs für LOD {lod_level}")

        height_delta = dlm.get_calculator_output("geology.intrusions", "height_delta", lod_level)
        fault_distance_map = dlm.get_calculator_output(
            "geology.tectonic_displacement", "fault_distance_map", lod_level)
        intrusion_distance_map = dlm.get_calculator_output(
            "geology.intrusions", "intrusion_distance_map", lod_level)
        metamorphic_grade_map = dlm.get_calculator_output(
            "geology.metamorphic_overprint", "metamorphic_grade_map", lod_level)
        layer_boundaries = dlm.get_calculator_output("geology.outcrop", "layer_boundaries", lod_level)
        # Diagnose-Komponenten: NUR "intrusion" ist auch Teil von height_delta
        # (siehe oben) - terrain_hub/tilt/fold/fault wirken ausschließlich auf
        # den Gesteinsstapel/Ausbiss, nie auf die sichtbare Kartenhöhe.
        delta_components = {
            "terrain_hub": dlm.get_calculator_output("geology.tectonic_displacement", "terrain_hub_delta", lod_level),
            "tilt": dlm.get_calculator_output("geology.tectonic_displacement", "tilt_delta", lod_level),
            "fold": dlm.get_calculator_output("geology.tectonic_displacement", "fold_delta", lod_level),
            "fault": dlm.get_calculator_output("geology.tectonic_displacement", "fault_delta", lod_level),
            "intrusion": dlm.get_calculator_output("geology.intrusions", "intrusion_delta", lod_level),
        }

        return self._create_geology_data(
            rock_map, hardness_map, layer_id_map, lod_level, parameters, height_delta,
            fault_distance_map, intrusion_distance_map, metamorphic_grade_map,
            layer_boundaries, delta_components)

    def _create_geology_data(self, rock_map, hardness_map, layer_id_map, lod_level, parameters,
                              height_delta, fault_distance_map, intrusion_distance_map,
                              metamorphic_grade_map, layer_boundaries, delta_components=None) -> GeologyData:
        validity_state = {
            'hardness_range': bool(np.all((hardness_map >= 1.0) & (hardness_map <= 100.0))),
            'layer_assignment': bool(np.all((layer_id_map >= 0) & (layer_id_map <= N_LAYERS))),
        }
        return GeologyData(
            rock_map=rock_map, hardness_map=hardness_map, layer_id_map=layer_id_map,
            lod_level=lod_level, actual_size=tuple(hardness_map.shape[:2]),
            validity_state=validity_state, parameter_hash=self._calculate_parameter_hash(parameters),
            parameters=dict(parameters), height_delta=height_delta,
            fault_distance_map=fault_distance_map, intrusion_distance_map=intrusion_distance_map,
            metamorphic_grade_map=metamorphic_grade_map, layer_boundaries=layer_boundaries,
            delta_components=delta_components,
        )

    @staticmethod
    def _calculate_parameter_hash(parameters: Dict[str, Any]) -> str:
        relevant = sorted((k, v) for k, v in parameters.items() if isinstance(v, (int, float, str)))
        return hashlib.md5(str(relevant).encode()).hexdigest()[:12]

    # -------------------------------------------------------------------
    # Calculator-Knoten (siehe managers/calculator_graph.py)
    # -------------------------------------------------------------------

    def _calc_layer_thickness(self, calculator_id: str, lod_level: int) -> None:
        self._update_progress("Layer Stack", 10, "Building rock layer thickness...")
        heightmap = self.data_lod_manager.get_calculator_output("terrain.redistribution", "heightmap", lod_level)
        slopemap = self.data_lod_manager.get_calculator_output("terrain.slope", "slopemap", lod_level)
        if heightmap is None or slopemap is None:
            raise ValueError(f"geology.layer_thickness: heightmap/slopemap für LOD {lod_level} nicht verfügbar")

        map_distance_km = self._get_map_distance_km()
        thickness = None
        if self.shader_manager:
            try:
                result = self.shader_manager.request_shader_operation(
                    "geology", "layerThickness",
                    {"shape": heightmap.shape, "map_distance_km": map_distance_km}, self._current_parameters)
                if result.get("success"):
                    thickness = result["layer_thickness"]
            except Exception as e:
                logging.warning(f"GPU layer thickness failed: {e}, falling back to CPU")
        if thickness is None:
            thickness = self.thickness_builder.build(heightmap.shape, map_distance_km, slopemap)

        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"layer_thickness": thickness})

    def _calc_tectonic_displacement(self, calculator_id: str, lod_level: int) -> None:
        self._update_progress("Tectonics", 25, "Building tectonic displacement field...")
        heightmap = self.data_lod_manager.get_calculator_output("terrain.redistribution", "heightmap", lod_level)
        if heightmap is None:
            raise ValueError(f"geology.tectonic_displacement: heightmap für LOD {lod_level} nicht verfügbar")

        map_distance_km = self._get_map_distance_km()
        parameters = self._current_parameters
        components = None
        if self.shader_manager:
            try:
                result = self.shader_manager.request_shader_operation(
                    "geology", "tectonicDisplacement",
                    {"shape": heightmap.shape, "map_distance_km": map_distance_km}, parameters)
                if result.get("success"):
                    components = result["components"]
            except Exception as e:
                logging.warning(f"GPU tectonic displacement failed: {e}, falling back to CPU")
        if components is None:
            components = self.displacement_builder.build(heightmap, map_distance_km, parameters)

        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, components)

    def _calc_outcrop(self, calculator_id: str, lod_level: int) -> None:
        self._update_progress("Outcrop", 45, "Intersecting rock layers with terrain...")
        terrain_height = self.data_lod_manager.get_calculator_output("terrain.redistribution", "heightmap", lod_level)
        layer_thickness = self.data_lod_manager.get_calculator_output(
            "geology.layer_thickness", "layer_thickness", lod_level)
        stack_deformation = self.data_lod_manager.get_calculator_output(
            "geology.tectonic_displacement", "stack_deformation", lod_level)
        if terrain_height is None or layer_thickness is None or stack_deformation is None:
            raise ValueError(f"geology.outcrop: fehlende Inputs für LOD {lod_level}")

        layer_id_map, layer_boundaries = _compute_outcrop(terrain_height, layer_thickness, stack_deformation)
        self.data_lod_manager.set_calculator_output(
            calculator_id, lod_level, {"layer_id_map": layer_id_map, "layer_boundaries": layer_boundaries})

    def _calc_intrusions(self, calculator_id: str, lod_level: int) -> None:
        """
        Platziert Intrusionen - wirkt AUSSCHLIESSLICH auf den Gesteinsstapel/
        Ausbiss (layer_id_map) und die Diagnose-Felder, NICHT auf die
        sichtbare Kartenhöhe (Nutzer-Korrektur: die frühere gekappte
        Intrusions-Dom-Hebung erzeugte eine tatsächliche Terrain-Erhebung,
        die der Nutzer nicht wollte - Intrusionen sollen nur ein
        "Durchbruch durch die Schichten" sein, in der Cross-Section vom
        realen Terrain abgeschnitten, siehe map_display_2d.py
        _render_cross_section() das den Basalt-"Wurzel"-Bereich bereits
        gegen terrain_slice kappt). `height_delta` liefert deshalb jetzt
        IMMER Nullen - Tilt/Fold/Fault/Terrain-Hub taten das schon vorher
        (siehe TectonicDisplacementField-Docstring), Geology trägt damit gar
        keinen Höhenbeitrag mehr zur kombinierten Heightmap bei.
        `intrusion_delta` bleibt als reines Diagnose-Feld (Anzeigemodus
        "Intrusion Δz", analog zu terrain_hub/tilt/fold/fault) erhalten -
        zeigt weiterhin die (rein hypothetische) Dom-Stärke, ohne sie
        anzuwenden.
        """
        self._update_progress("Intrusions", 60, "Placing igneous intrusions...")
        layer_id_map = self.data_lod_manager.get_calculator_output("geology.outcrop", "layer_id_map", lod_level)
        terrain_height = self.data_lod_manager.get_calculator_output("terrain.redistribution", "heightmap", lod_level)
        if layer_id_map is None or terrain_height is None:
            raise ValueError(f"geology.intrusions: fehlende Inputs für LOD {lod_level}")

        map_distance_km = self._get_map_distance_km()
        parameters = self._current_parameters
        height_range = float(np.max(terrain_height) - np.min(terrain_height)) or 1.0

        intrusion_distance_map, intrusion_delta = _build_intrusion_field(
            layer_id_map.shape, map_distance_km, self.map_seed, height_range,
            parameters.get('intrusion_density', 0.0), parameters.get('intrusion_size', 1.0),
            parameters.get('intrusion_detail', 0.0))

        updated_layer_id_map = _apply_intrusions_to_layer_id(layer_id_map, intrusion_distance_map)
        height_delta = np.zeros_like(intrusion_delta, dtype=np.float32)

        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {
            "layer_id_map": updated_layer_id_map,
            "intrusion_distance_map": intrusion_distance_map,
            "intrusion_delta": intrusion_delta,
            "height_delta": height_delta,
        })

    def _calc_sediment_overlay(self, calculator_id: str, lod_level: int) -> None:
        self._update_progress("Sediment Overlay", 72, "Classifying valley sediments...")
        layer_id_map = self.data_lod_manager.get_calculator_output("geology.intrusions", "layer_id_map", lod_level)
        heightmap_combined = self.data_lod_manager.get_calculator_combined_heightmap(lod_level)
        if layer_id_map is None or heightmap_combined is None:
            raise ValueError(f"geology.sediment_overlay: fehlende Inputs für LOD {lod_level}")

        map_distance_km = self._get_map_distance_km()
        updated = _apply_sediment_overlay(layer_id_map, heightmap_combined, map_distance_km)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"layer_id_map": updated})

    def _calc_metamorphic_overprint(self, calculator_id: str, lod_level: int) -> None:
        self._update_progress("Metamorphism", 82, "Computing metamorphic overprint...")
        fault_distance_map = self.data_lod_manager.get_calculator_output(
            "geology.tectonic_displacement", "fault_distance_map", lod_level)
        intrusion_distance_map = self.data_lod_manager.get_calculator_output(
            "geology.intrusions", "intrusion_distance_map", lod_level)
        if fault_distance_map is None or intrusion_distance_map is None:
            raise ValueError(f"geology.metamorphic_overprint: fehlende Inputs für LOD {lod_level}")

        overprint_intensity = self._current_parameters.get('metamorphic_overprint_intensity', 0.0)
        grade = _compute_metamorphic_grade(fault_distance_map, intrusion_distance_map, overprint_intensity)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"metamorphic_grade_map": grade})

    def _calc_rock_color(self, calculator_id: str, lod_level: int) -> None:
        self._update_progress("Rock Coloring", 90, "Coloring rock outcrop...")
        layer_id_map = self.data_lod_manager.get_calculator_output("geology.sediment_overlay", "layer_id_map", lod_level)
        metamorphic_grade = self.data_lod_manager.get_calculator_output(
            "geology.metamorphic_overprint", "metamorphic_grade_map", lod_level)
        if layer_id_map is None or metamorphic_grade is None:
            raise ValueError(f"geology.rock_color: fehlende Inputs für LOD {lod_level}")

        map_distance_km = self._get_map_distance_km()
        foliation_detail = self._current_parameters.get('foliation_detail', 0.0)
        rock_map = _build_rock_map(layer_id_map, metamorphic_grade, foliation_detail, map_distance_km, self.map_seed)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"rock_map": rock_map})

    def _calc_hardness(self, calculator_id: str, lod_level: int) -> None:
        self._update_progress("Hardness", 97, "Calculating hardness map...")
        layer_id_map = self.data_lod_manager.get_calculator_output("geology.sediment_overlay", "layer_id_map", lod_level)
        metamorphic_grade = self.data_lod_manager.get_calculator_output(
            "geology.metamorphic_overprint", "metamorphic_grade_map", lod_level)
        terrain_height = self.data_lod_manager.get_calculator_output("terrain.redistribution", "heightmap", lod_level)
        if layer_id_map is None or metamorphic_grade is None or terrain_height is None:
            raise ValueError(f"geology.hardness: fehlende Inputs für LOD {lod_level}")

        parameters = self._current_parameters
        hardness_map = _build_hardness_map(
            layer_id_map, metamorphic_grade, terrain_height,
            parameters['sedimentary_hardness'], parameters['igneous_hardness'], parameters['metamorphic_hardness'])
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"hardness_map": hardness_map})

    # -------------------------------------------------------------------
    # Validierung / Fallback / Info
    # -------------------------------------------------------------------

    def _validate_inputs(self, heightmap_combined: np.ndarray, slopemap: np.ndarray,
                          parameters: Dict[str, Any]):
        if heightmap_combined is None or heightmap_combined.size == 0:
            raise ValueError("Invalid heightmap_combined - empty or None")
        if len(heightmap_combined.shape) != 2:
            raise ValueError("Heightmap must be 2D array")
        if slopemap is None or slopemap.size == 0:
            raise ValueError("Invalid slopemap - empty or None")

        for key in ('sedimentary_hardness', 'igneous_hardness', 'metamorphic_hardness'):
            value = parameters.get(key)
            if value is None or not (0.0 <= value <= 100.0):
                raise ValueError(f"{key} out of range [0-100]: {value}")

    def _create_fallback_geology_data(self, heightmap_combined: Optional[np.ndarray],
                                       parameters: Dict[str, Any], lod_level: int) -> GeologyData:
        self.logger.warning("Using fallback geology data due to generation error")
        if heightmap_combined is not None and heightmap_combined.ndim >= 2:
            height, width = heightmap_combined.shape[:2]
        else:
            height, width = 64, 64

        layer_id_map = np.zeros((height, width), dtype=np.int16)  # überall Kristallin (Index 0)
        rock_map = np.tile(np.array(ROCK_LAYERS[0].color, dtype=np.uint8), (height, width, 1))
        hardness_map = np.full((height, width), 50.0, dtype=np.float32)

        return GeologyData(
            rock_map=rock_map, hardness_map=hardness_map, layer_id_map=layer_id_map,
            lod_level=lod_level, actual_size=(height, width),
            validity_state={'hardness_range': False, 'layer_assignment': False},
            parameter_hash="fallback", parameters=dict(parameters),
            height_delta=np.zeros((height, width), dtype=np.float32),
        )

    def get_generation_info(self) -> Dict[str, Any]:
        """
        Status-Info für Debug/UI. Frühere Versionen behaupteten hier
        fälschlich ein aktives 3-stufiges GPU/CPU/Simple-Fallback-System
        (siehe docs/session_review_2026-07-22_geology.md) - korrigiert:
        der shader_manager-Slot ist strukturell vorbereitet, aber aktuell
        mit keinen echten GPU-Operationen hinterlegt, jede _calc_*-Methode
        läuft deshalb faktisch immer CPU-only.
        """
        return {
            'generator': 'GeologySystemGenerator',
            'model': '3D-Gesteinsstapel (core/geology_layers.py, ROCK_LAYERS)',
            'n_layers': N_LAYERS,
            'shader_manager_configured': self.shader_manager is not None,
            'gpu_operations_implemented': False,
            'fallback_levels': ['cpu'],
        }
