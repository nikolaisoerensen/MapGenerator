"""
Path: core/terrain_erosion_filter.py

CPU-Referenz des Advanced Terrain Erosion Filter (ATEF).

Numpy-Portierung von shaders/terrain/ATEF_Buffer_A.comp und der davon
benutzten Teile von ATEF_common.comp. Zeile fuer Zeile uebersetzt, damit sie
als Vergleichsseite des GPU-Paritaetstests taugt (SPEZIFIKATION §4.1: jeder
Rechenweg mit GPU-Pfad hat einen Paritaetstest gegen die CPU, und eine
Aenderung an einem Pfad ist erst fertig, wenn der andere mitgezogen ist).

WAS DIESER FILTER IST: ein Filter PRO PIXEL, keine Simulation. Er liest
ausschliesslich den Punkt und dessen Hoehe samt Ableitungen - kein
Nachbarzugriff, keine Ping-Pong-Buffer, keine Zeitschritte. Fuenf Oktaven mal
eine 4x4-Zellenschleife, ein Durchgang. Gegen die 6500-8000 Iterationen der
Feld-Erosion (SPEZIFIKATION §7) ist das praktisch kostenlos.

WAS ER NICHT IST: eine Entwaesserung. Er bewegt keine Masse, kennt kein
Routing und keine Konnektivitaet. Die Kennzahlen aus §3.2 (Netzgroesse,
Zusammenfluesse, Randabfluss) erfuellt er nicht, und §4.3 (Massenbilanz) gilt
fuer ihn nicht, weil es nichts zu bilanzieren gibt. Er macht das AUSSEHEN von
Erosion. Die Entwaesserung kommt aus dem Skelett-Ansatz (§8) - die beiden sind
die zwei Haelften, nicht zwei Alternativen.

Nuetzliche Nebenausgabe: die ridge_map, -1 in Kerben und +1 auf Kaemmen. Der
Autor nennt sie ausdruecklich als Eingang fuer Entwaesserung.

-----------------------------------------------------------------------------
Advanced Terrain Erosion Filter and Phacelle Noise
copyright (c) 2025 Rune Skovbo Johansen

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Herkunft: https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html
Diese Datei ist eine Portierung nach numpy und steht damit unter derselben
Lizenz. MPL 2.0 ist datei-bezogen: solange der portierte Code in dieser Datei
mit diesem Hinweis bleibt, beruehrt er die Lizenz des uebrigen Projekts nicht.
-----------------------------------------------------------------------------
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TAU = 6.28318530717959

# Vorgaben aus dem Demonstrationsteil von ATEF_Buffer_A.comp. Namen und Werte
# absichtlich identisch zum Shader, damit ein Vergleich moeglich bleibt; die
# Erklaerungen stehen dort ab Zeile 296 und werden hier nicht verdoppelt.
#
# Alle Werte sind RELATIV (Einheitsquadrat, Hoehe in [0,1]) - keine Meterwerte.
# Nach SPEZIFIKATION §4.4 ist das der wichtigste Punkt an diesem Filter: der
# Fehlertyp "absolute Konstante, wo eine relative hingehoert" entsteht hier
# nicht.
ATEF_DEFAULTS: Dict[str, Any] = {
    "erosion_scale": 0.15,
    "erosion_strength": 0.22,
    "erosion_gully_weight": 0.5,
    "erosion_detail": 1.5,
    # (Rundung Kaemme, Rundung Kerben, Faktor auf die Eingangshoehe,
    #  Faktor je Folgeoktave)
    "erosion_rounding": (0.1, 0.0, 0.1, 2.0),
    # (Onset Eingangshoehe, Onset je Oktave, Onset ridge_map Eingang,
    #  Onset ridge_map je Oktave)
    "erosion_onset": (1.25, 1.25, 2.8, 1.5),
    # (angenommene Neigung, Anteil, mit dem sie die echte ersetzt)
    #
    # Der Anteil steht auf 1.0: der BETRAG der Eingangsneigung wird vollstaendig
    # ersetzt, nur ihre RICHTUNG geht in die Rinnen ein (siehe erosion_filter(),
    # Zeile mit gully_slope). Deshalb reicht fuer die Rinnenrichtung ein
    # Differenzenquotient und es braucht keine analytische Ableitung - der
    # Betrag wirkt nur noch in der Onset-Maske.
    "erosion_assumed_slope": (0.7, 1.0),
    "erosion_cell_scale": 0.7,
    "erosion_normalization": 0.5,
    "erosion_octaves": 5,
    "erosion_lacunarity": 2.0,
    "erosion_gain": 0.5,
    # (Versatz -1..1, Anteil, mit dem der negierte fade_target ihn ersetzt)
    #
    # GEMESSEN 2026-07-30, weil die Erklaerung im Shader in die Irre fuehrt:
    # der zweite Wert wird dort als Relieferhalt beschrieben ("largely
    # preserving the minima and maxima"), und das stimmt fuer die EXTREMWERTE.
    # Er setzt aber den fade_target der LETZTEN Oktave ein, also eine
    # hochfrequente Groesse - das Bild wird dadurch feinkoerniges Gekrissel
    # statt zusammenhaengender Grate. Bei 0.0 ist der Versatz dagegen eine
    # KONSTANTE (magnitude ist ein Skalar) und aendert die Form gar nicht.
    #
    # Hier bleibt der Wert des Originals stehen, damit demo_heightmap() das
    # Demonstrationsgelaende unveraendert nachbaut. Fuer den Einsatz auf
    # unseren Karten setzt filter_heightmap() ihn auf (0.0, 0.0) - siehe dort.
    "terrain_height_offset": (-0.65, 0.0),
}

# Basis-Noise des Demonstrationsteils - nur fuer demo_heightmap() gebraucht,
# nicht fuer den Einsatz auf unserer eigenen Heightmap.
ATEF_HEIGHT_DEFAULTS: Dict[str, Any] = {
    "height_frequency": 3.0,
    "height_amp": 0.125,
    "height_octaves": 3,
    "height_lacunarity": 2.0,
    "height_gain": 0.1,
}


# =============================================================================
# Hilfsfunktionen - GLSL-Semantik, nicht numpy-Semantik
# =============================================================================

def _fract(a: np.ndarray) -> np.ndarray:
    """GLSL fract: a - floor(a), also immer in [0,1) - auch fuer negative a.

    np.mod(a, 1.0) waere fast dasselbe, gibt aber fuer sehr kleine negative
    Werte exakt 1.0 zurueck statt 0.0. Die floor-Form ist die, die der Shader
    rechnet.
    """
    return a - np.floor(a)


def _clamp01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)


def _mix(a, b, t):
    """GLSL mix: linear von a nach b. t wird NICHT geklemmt (wie in GLSL)."""
    return a * (1.0 - t) + b * t


def _pow_inv(t, power):
    """ATEF_Buffer_A.comp: umklappen, potenzieren, zurueckklappen."""
    return 1.0 - np.power(1.0 - _clamp01(t), power)


def _ease_out(t):
    v = 1.0 - _clamp01(t)
    return 1.0 - v * v


def _smooth_start(t, smoothing):
    """
    ATEF_Buffer_A.comp smooth_start() mit dem Zweig als np.where.

    Der Shader schreibt:
        if (t >= smoothing) return t - 0.5*smoothing;
        return 0.5*t*t/smoothing;

    smoothing kann 0 werden (erosion_rounding[1] steht per Vorgabe auf 0.0).
    Dann ist der erste Zweig fuer jedes t >= 0 zustaendig und der zweite wird
    nie erreicht - t ist hier immer eine Neigungslaenge, also >= 0. numpy
    rechnet aber beide Zweige aus, deshalb der Ersatznenner: ohne ihn stehen
    NaN in einem Ergebnis, das der Shader korrekt liefert.
    """
    smoothing = np.asarray(smoothing, dtype=np.float64)
    nenner = np.where(smoothing > 0.0, smoothing, 1.0)
    return np.where(t >= smoothing, t - 0.5 * smoothing, 0.5 * t * t / nenner)


def _safe_normalize(nx: np.ndarray, ny: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Div-durch-0-sichere Normierung, wie safe_normalize() im Shader."""
    laenge = np.sqrt(nx * nx + ny * ny)
    ok = np.abs(laenge) > 1e-10
    teiler = np.where(ok, laenge, 1.0)
    return nx / teiler, ny / teiler


def _hash2(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    ATEF_common.comp hash().

    Der Shader rechnet `x = x * k + k.yx`, also komponentenweise mit
    VERTAUSCHTEM Summanden - x bekommt k.y, y bekommt k.x. Danach ist
    fract(x.x*x.y*(x.x+x.y)) ein SKALAR, der mit dem Vektor 16*k multipliziert
    wird.
    """
    kx, ky = 0.3183099, 0.3678794
    ax = x * kx + ky
    ay = y * ky + kx
    s = _fract(ax * ay * (ax + ay))
    return (-1.0 + 2.0 * _fract(16.0 * kx * s),
            -1.0 + 2.0 * _fract(16.0 * ky * s))


def noised(px: np.ndarray, py: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ATEF_common.comp noised(): Gradient-Noise samt seinen ABLEITUNGEN.

    Returns: (wert, d/dx, d/dy)
    """
    ix, iy = np.floor(px), np.floor(py)
    fx, fy = px - ix, py - iy

    ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)
    uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0)
    dux = 30.0 * fx * fx * (fx * (fx - 2.0) + 1.0)
    duy = 30.0 * fy * fy * (fy * (fy - 2.0) + 1.0)

    gax, gay = _hash2(ix, iy)
    gbx, gby = _hash2(ix + 1.0, iy)
    gcx, gcy = _hash2(ix, iy + 1.0)
    gdx, gdy = _hash2(ix + 1.0, iy + 1.0)

    va = gax * fx + gay * fy
    vb = gbx * (fx - 1.0) + gby * fy
    vc = gcx * fx + gcy * (fy - 1.0)
    vd = gdx * (fx - 1.0) + gdy * (fy - 1.0)

    eck = va - vb - vc + vd
    wert = va + ux * (vb - va) + uy * (vc - va) + ux * uy * eck

    # Der Shader baut die Ableitung als vec2; u.yx dreht die Komponenten, und
    # vec2(vb, vc) liefert x aus vb, y aus vc.
    dx = (gax + ux * (gbx - gax) + uy * (gcx - gax) + ux * uy * (gax - gbx - gcx + gdx)
          + dux * (uy * eck + vb - va))
    dy = (gay + ux * (gby - gay) + uy * (gcy - gay) + ux * uy * (gay - gby - gcy + gdy)
          + duy * (ux * eck + vc - va))
    return wert, dx, dy


def fractal_noise(px, py, frequency, octaves, lacunarity, gain):
    """
    ATEF_Buffer_A.comp FractalNoise(): fBm samt Ableitungen.

    Der Shader multipliziert je Oktave mit vec3(1, nf, nf) - die Ableitungen
    tragen also die Kettenregel des Frequenzfaktors.
    """
    wert = np.zeros_like(px, dtype=np.float64)
    dx = np.zeros_like(px, dtype=np.float64)
    dy = np.zeros_like(px, dtype=np.float64)
    nf, na = float(frequency), 1.0
    for _ in range(int(octaves)):
        w, ddx, ddy = noised(px * nf, py * nf)
        wert += w * na
        dx += ddx * na * nf
        dy += ddy * na * nf
        na *= gain
        nf *= lacunarity
    return wert, dx, dy


# =============================================================================
# Phacelle Noise
# =============================================================================

def phacelle_noise(px, py, norm_dir_x, norm_dir_y, freq, offset, normalization):
    """
    ATEF_Buffer_A.comp PhacelleNoise(): Streifenmuster, das an der uebergebenen
    Richtung ausgerichtet ist. "Phacelle" ist ein Kofferwort aus phase und
    cell.

    Returns: (cos_anteil, sin_anteil, side_dir_x, side_dir_y) - dieselben vier
    Komponenten wie der vec4 des Shaders.
    """
    side_dir_x = -norm_dir_y * freq * TAU
    side_dir_y = norm_dir_x * freq * TAU
    offset = offset * TAU

    p_int_x, p_int_y = np.floor(px), np.floor(py)
    p_frac_x, p_frac_y = px - p_int_x, py - p_int_y

    phase_x = np.zeros_like(px, dtype=np.float64)
    phase_y = np.zeros_like(px, dtype=np.float64)
    gewicht_summe = np.zeros_like(px, dtype=np.float64)

    # 4x4 Zellen. Die Zufallsverschiebung betraegt bis zu 0.5, deshalb kann
    # keine Zelle ausserhalb dieses Fensters naeher als 1.5 liegen.
    for i in (-1, 0, 1, 2):
        for j in (-1, 0, 1, 2):
            gitter_x = p_int_x + i
            gitter_y = p_int_y + j
            zufall_x, zufall_y = _hash2(gitter_x, gitter_y)
            vx = p_frac_x - i - zufall_x * 0.5
            vy = p_frac_y - j - zufall_y * 0.5

            quad_abstand = vx * vx + vy * vy
            gewicht = np.exp(-quad_abstand * 2.0)
            # Die 0.01111 machen die Funktion bei Abstand 1.5 exakt null und
            # vermeiden Gitterlinien-Artefakte.
            gewicht = np.maximum(0.0, gewicht - 0.01111)
            gewicht_summe += gewicht

            wellen_eingang = vx * side_dir_x + vy * side_dir_y + offset
            phase_x += np.cos(wellen_eingang) * gewicht
            phase_y += np.sin(wellen_eingang) * gewicht

    interp_x = phase_x / gewicht_summe
    interp_y = phase_y / gewicht_summe
    betrag = np.sqrt(interp_x * interp_x + interp_y * interp_y)
    betrag = np.maximum(1.0 - normalization, betrag)
    return interp_x / betrag, interp_y / betrag, side_dir_x, side_dir_y


# =============================================================================
# Der Filter
# =============================================================================

def erosion_filter(px, py, height, slope_x, slope_y, fade_target, parameters=None):
    """
    ATEF_Buffer_A.comp ErosionFilter().

    Parameter: px, py - Ortskoordinaten im Einheitsquadrat. height, slope_x,
        slope_y - Eingangshoehe und ihre Ableitungen. fade_target - soll -1 in
        Taelern und +1 auf Gipfeln sein, Ueberschwingen ist erlaubt.
        parameters - Ueberschreibungen von ATEF_DEFAULTS.

    Returns: dict mit
        height_delta, slope_delta_x, slope_delta_y   Aenderung, nicht Ergebnis
        magnitude                                    Summe der Oktavenstaerken
        ridge_map                                    -1 Kerbe .. +1 Kamm

    Es wird ein DELTA zurueckgegeben, kein fertiges Gelaende - dieselbe
    Heightmap-Semantik, die Geology und Water in diesem Projekt schon benutzen
    (get_calculator_combined_heightmap() summiert die Deltas).
    """
    p = dict(ATEF_DEFAULTS)
    p.update(parameters or {})

    scale = float(p["erosion_scale"])
    strength = float(p["erosion_strength"]) * scale
    gully_weight = float(p["erosion_gully_weight"])
    detail = float(p["erosion_detail"])
    rounding = tuple(float(v) for v in p["erosion_rounding"])
    onset = tuple(float(v) for v in p["erosion_onset"])
    assumed_slope = tuple(float(v) for v in p["erosion_assumed_slope"])
    cell_scale = float(p["erosion_cell_scale"])
    normalization = float(p["erosion_normalization"])
    octaves = int(p["erosion_octaves"])
    lacunarity = float(p["erosion_lacunarity"])
    gain = float(p["erosion_gain"])

    fade_target = np.clip(np.asarray(fade_target, dtype=np.float64), -1.0, 1.0)

    hoehe = np.array(height, dtype=np.float64, copy=True)
    neigung_x = np.array(slope_x, dtype=np.float64, copy=True)
    neigung_y = np.array(slope_y, dtype=np.float64, copy=True)
    eingang_hoehe = hoehe.copy()
    eingang_nx = neigung_x.copy()
    eingang_ny = neigung_y.copy()

    freq = 1.0 / (scale * cell_scale)
    neigung_laenge = np.maximum(np.sqrt(neigung_x ** 2 + neigung_y ** 2), 1e-10)
    magnitude = 0.0
    rounding_mult = 1.0

    rounding_eingang = _mix(rounding[1], rounding[0],
                            _clamp01(fade_target + 0.5)) * rounding[2]
    combi_mask = _ease_out(_smooth_start(neigung_laenge * onset[0],
                                        rounding_eingang * onset[0]))

    ridge_combi_mask = _ease_out(neigung_laenge * onset[2])
    ridge_fade_target = fade_target.copy()

    # Nur die RICHTUNG der Eingangsneigung geht ein, ihr Betrag wird bei
    # assumed_slope[1] == 1.0 vollstaendig durch assumed_slope[0] ersetzt.
    gully_x = _mix(neigung_x, neigung_x / neigung_laenge * assumed_slope[0],
                   assumed_slope[1])
    gully_y = _mix(neigung_y, neigung_y / neigung_laenge * assumed_slope[0],
                   assumed_slope[1])

    for _ in range(octaves):
        richtung_x, richtung_y = _safe_normalize(gully_x, gully_y)
        ph_x, ph_y, side_x, side_y = phacelle_noise(
            px * freq, py * freq, richtung_x, richtung_y,
            cell_scale, 0.25, normalization)

        # Mit freq multiplizieren, weil p mit freq multipliziert wurde.
        # Negieren, weil abwaerts zeigende Neigungsrichtungen benutzt werden.
        side_x = side_x * -freq
        side_y = side_y * -freq
        sloping = np.abs(ph_y)

        gully_x = gully_x + np.sign(ph_y) * side_x * strength * gully_weight
        gully_y = gully_y + np.sign(ph_y) * side_y * strength * gully_weight

        # gullies: Hoehenversatz in x, Ableitung in yz
        gullies_h = ph_x
        gullies_dx = ph_y * side_x
        gullies_dy = ph_y * side_y

        faded_h = _mix(fade_target, gullies_h * gully_weight, combi_mask)
        faded_dx = _mix(0.0, gullies_dx * gully_weight, combi_mask)
        faded_dy = _mix(0.0, gullies_dy * gully_weight, combi_mask)

        hoehe = hoehe + faded_h * strength
        neigung_x = neigung_x + faded_dx * strength
        neigung_y = neigung_y + faded_dy * strength
        magnitude += strength

        fade_target = faded_h

        rounding_oktave = _mix(rounding[1], rounding[0],
                               _clamp01(ph_x + 0.5)) * rounding_mult
        neue_maske = _ease_out(_smooth_start(sloping * onset[1],
                                            rounding_oktave * onset[1]))
        combi_mask = _pow_inv(combi_mask, detail) * neue_maske

        ridge_fade_target = _mix(ridge_fade_target, gullies_h, ridge_combi_mask)
        ridge_combi_mask = ridge_combi_mask * _ease_out(sloping * onset[3])

        strength *= gain
        freq *= lacunarity
        rounding_mult *= rounding[3]

    return {
        "height_delta": hoehe - eingang_hoehe,
        "slope_delta_x": neigung_x - eingang_nx,
        "slope_delta_y": neigung_y - eingang_ny,
        "magnitude": magnitude,
        "ridge_map": ridge_fade_target * (1.0 - ridge_combi_mask),
        "fade_target": fade_target,
    }


# =============================================================================
# Demonstration - 1:1 der Heightmap()-Teil von ATEF_Buffer_A.comp
# =============================================================================

def demo_heightmap(size: int = 256, parameters: Optional[Dict[str, Any]] = None,
                   scroll: Tuple[float, float] = (0.0, 0.0)) -> Dict[str, np.ndarray]:
    """
    Baut das Gelaende genau so, wie es der Demonstrationsteil des Shaders baut
    (eigenes fBm mit analytischen Ableitungen, dann der Filter).

    Zweck: das Ergebnis der Portierung gegen das bekannte Bild des Originals
    halten, BEVOR der Filter auf unsere eigene Heightmap angewandt wird. Ohne
    diesen Schritt weiss man bei einem schlechten Ergebnis nicht, ob die
    Portierung falsch ist oder die Anwendung.

    Der animierte Teil (AnimateLoHi/AnimateWaveTo, iTime) ist bewusst NICHT
    portiert - er gehoert zur Vorfuehrung, nicht zum Verfahren.
    """
    p = dict(ATEF_DEFAULTS)
    p.update(ATEF_HEIGHT_DEFAULTS)
    p.update(parameters or {})

    # uv wie im Shader: fragCoord / BUFFER_SIZE, also [0,1)
    achse = (np.arange(size, dtype=np.float64) + 0.5) / float(size)
    px, py = np.meshgrid(achse + scroll[0], achse + scroll[1], indexing="xy")

    wert, dx, dy = fractal_noise(
        px, py, p["height_frequency"], p["height_octaves"],
        p["height_lacunarity"], p["height_gain"])
    amp = float(p["height_amp"])
    wert, dx, dy = wert * amp, dx * amp, dy * amp

    # fade_target VOR der Umskalierung auf [0,1] - so steht es im Shader.
    fade_target = np.clip(wert / (amp * 0.6), -1.0, 1.0)

    # Der Shader schreibt `n = n * 0.5 + vec3(0.5, 0, 0)`. Die Multiplikation
    # trifft alle drei Komponenten, die Addition nur x - die Ableitungen werden
    # also halbiert, aber nicht verschoben.
    hoehe = wert * 0.5 + 0.5
    dx, dy = dx * 0.5, dy * 0.5

    ergebnis = erosion_filter(px, py, hoehe, dx, dy, fade_target, p)

    offset_p = tuple(float(v) for v in p["terrain_height_offset"])
    offset = _mix(offset_p[0], -ergebnis["fade_target"], offset_p[1]) * ergebnis["magnitude"]
    eroded = hoehe + ergebnis["height_delta"] + offset

    return {
        "height_raw": hoehe,
        "height_eroded": eroded,
        "height_delta": ergebnis["height_delta"],
        "ridge_map": ergebnis["ridge_map"],
        "magnitude": ergebnis["magnitude"],
        "slope_x": dx + ergebnis["slope_delta_x"],
        "slope_y": dy + ergebnis["slope_delta_y"],
    }


# =============================================================================
# Anwendung auf unsere eigene Heightmap
# =============================================================================

def filter_heightmap(heightmap: np.ndarray, meters_per_pixel: float,
                     parameters: Optional[Dict[str, Any]] = None
                     ) -> Dict[str, np.ndarray]:
    """
    Wendet den Filter auf eine Heightmap dieses Projekts an (Meter, beliebige
    Spanne) und gibt das Delta IN METERN zurueck.

    Der Filter rechnet im Einheitsquadrat mit Hoehe in [0,1]. Hier wird also
    hin- und zurueckskaliert, und zwar gegen die TATSAECHLICHE Spanne der
    uebergebenen Karte - nicht gegen AMPLITUDE. Dieselbe Entscheidung wie in
    BaseTerrainGenerator._apply_redistribution(), und aus demselben Grund: die
    theoretische Spanne wird real nie erreicht, und gegen sie zu normieren
    verliert Hoehe.

    Die Neigung kommt aus dem Differenzenquotienten, nicht analytisch. Das ist
    zulaessig, weil erosion_assumed_slope[1] = 1.0 den BETRAG der Neigung
    vollstaendig ersetzt und nur ihre RICHTUNG benutzt (siehe ATEF_DEFAULTS).
    Der Betrag wirkt allein in der Onset-Maske.

    Returns: dict mit height_delta (Meter), ridge_map, magnitude.
    """
    heightmap = np.asarray(heightmap, dtype=np.float64)
    if heightmap.ndim != 2 or heightmap.shape[0] != heightmap.shape[1]:
        raise ValueError(
            "filter_heightmap erwartet eine quadratische 2D-Karte, bekam %s"
            % (heightmap.shape,))

    size = heightmap.shape[0]
    p = dict(ATEF_DEFAULTS)
    # Der konstante Hoehenversatz des Demonstrationsteils entfaellt: er
    # verschiebt die Karte als Ganzes, und unsere Pipeline legt die
    # Hoehenspanne ohnehin neu fest (_apply_redistribution). Der zweite Wert
    # bleibt bei 0, weil er sonst die letzte Gully-Oktave als Rauschen
    # einsetzt - gemessen, siehe ATEF_DEFAULTS.
    p["terrain_height_offset"] = (0.0, 0.0)
    p.update(parameters or {})

    tief, hoch = float(heightmap.min()), float(heightmap.max())
    spanne = hoch - tief
    if spanne < 1e-9:
        # Voellig flache Karte - es gibt keine Neigung, an der sich Rinnen
        # ausrichten koennten. Tritt tatsaechlich auf: ist `amplitude` gleich
        # TERRAIN.BASE_ELEVATION_M, ist die Zielspanne der Redistribution
        # 100..100 (siehe test_terrain_generator(), das genau das einstellt).
        #
        # Der Rueckgabewert fuehrt ALLE Schluessel des Normalfalls. Vorher
        # fehlte effective_octaves, und der Aufrufer
        # (BaseTerrainGenerator._apply_erosion_filter) lief in einen KeyError,
        # der die ganze Terrain-Generierung in die Fehlerbehandlung schickte -
        # sichtbar nur daran, dass validity_state auf "error" stand.
        return {
            "height_delta": np.zeros_like(heightmap, dtype=np.float32),
            "ridge_map": np.zeros_like(heightmap, dtype=np.float32),
            "magnitude": 0.0,
            "effective_octaves": 0,
        }

    # Oktaven, die ueber die Nyquist-Grenze gehen, fuegen nur Aliasing hinzu -
    # dieselbe Klemme und dieselbe Begruendung wie
    # BaseTerrainGenerator._max_safe_octaves(). Die feinste Oktave hat
    # 1/(scale*cell_scale) * lacunarity^(n-1) Zyklen ueber die Karte; mehr als
    # size/2 Zyklen sind nicht darstellbar.
    p["erosion_octaves"] = _max_safe_octaves(
        1.0 / (float(p["erosion_scale"]) * float(p["erosion_cell_scale"])),
        float(p["erosion_lacunarity"]), int(p["erosion_octaves"]), size)

    normiert = (heightmap - tief) / spanne

    achse = (np.arange(size, dtype=np.float64) + 0.5) / float(size)
    px, py = np.meshgrid(achse, achse, indexing="xy")

    # Ableitung im Einheitsquadrat: d(normierte Hoehe) / d(normierter Ort).
    # np.gradient liefert (d/dZeile, d/dSpalte); Spalte ist x.
    d_zeile, d_spalte = np.gradient(normiert, 1.0 / float(size))
    slope_x, slope_y = d_spalte, d_zeile

    # fade_target: -1 in Taelern, +1 auf Gipfeln. Aus der normierten Hoehe um
    # den Median, damit eine schiefe Hoehenverteilung nicht alles auf eine
    # Seite legt.
    mitte = float(np.median(normiert))
    streuung = max(float(np.std(normiert)), 1e-6)
    fade_target = np.clip((normiert - mitte) / (2.0 * streuung), -1.0, 1.0)

    ergebnis = erosion_filter(px, py, normiert, slope_x, slope_y, fade_target, p)

    offset_p = tuple(float(v) for v in p["terrain_height_offset"])
    offset = _mix(offset_p[0], -ergebnis["fade_target"],
                  offset_p[1]) * ergebnis["magnitude"]

    delta_normiert = ergebnis["height_delta"] + offset
    return {
        "height_delta": (delta_normiert * spanne).astype(np.float32),
        "ridge_map": ergebnis["ridge_map"].astype(np.float32),
        "magnitude": float(ergebnis["magnitude"]),
        "effective_octaves": int(p["erosion_octaves"]),
    }


def _max_safe_octaves(base_cycles: float, lacunarity: float,
                      requested: int, size: int) -> int:
    """
    Groesste Oktavenzahl, deren feinste Struktur noch unter der Nyquist-Grenze
    der Kartenauflaesung liegt. Geschlossene Form, wie
    BaseTerrainGenerator._max_safe_octaves().
    """
    if requested <= 1 or base_cycles <= 0 or lacunarity <= 1.0:
        return max(1, requested)
    grenze = size / 2.0
    if base_cycles > grenze:
        return 1
    moeglich = 1 + int(np.floor(np.log(grenze / base_cycles) / np.log(lacunarity)))
    return max(1, min(requested, moeglich))
