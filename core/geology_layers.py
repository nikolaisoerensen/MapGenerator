"""
Path: core/geology_layers.py
Date Changed: 22.07.2026

Funktionsweise: Feste Definition des Gesteins-Schichtstapels für das
3D-Gesteinsstapel-Modell in core/geology_generator.py. Ersetzt die
frühere freie RGB-Mischverhältnis-Klassifizierung durch eine geordnete
Liste benannter geologischer Formationen (älteste unten, jüngste oben),
angelehnt an eine reale geologische Übersichtskarte (Oberrheingraben-Typ
Legende). Reine Datendefinition, keine Berechnung.

ROCK_LAYERS ist die geordnete Stapel-Reihenfolge (Index 0 = ältestes/
unterstes Glied). BASALT_INTRUSION ist kein Stapel-Glied, sondern der
Gesteinstyp lokaler Intrusionskörper, die den Stapel durchschlagen
(siehe GeologySystemGenerator._apply_intrusions()).
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RockLayer:
    """Eine benannte geologische Formation im Schichtstapel."""
    name: str
    category: str  # "sedimentary" | "igneous" | "metamorphic" - siehe HARDNESS_CATEGORIES
    color: Tuple[int, int, int]  # RGB 0-255, für Rock-Outcrop-Anzeige
    base_thickness_m: float  # nominale Schichtdicke in Metern, siehe geology_generator._build_layer_thickness()
    # Fester, geologisch motivierter relativer Härte-Multiplikator auf den
    # jeweiligen Kategorie-Slider (sedimentary_hardness/igneous_hardness/
    # metamorphic_hardness) - OHNE eigenen Slider (Nutzer-Entscheidung aus
    # der 3D-Stack-Konzept-Diskussion, Frage 19: nur 3 Härte-Regler). Ohne
    # dieses Feld hätten alle 11 sedimentären Schichten exakt dieselbe
    # Härte (nur die Kategorie zählte) - Nutzer-Feedback: "zu unvariabel".
    # Siehe geology_generator._build_hardness_map().
    hardness_factor: float


# Härte-Kategorien, wie sie von den 3 Rock-Hardness-Slidern
# (sedimentary_hardness/igneous_hardness/metamorphic_hardness,
# gui/config/value_default.py GEOLOGY) bedient werden.
HARDNESS_CATEGORIES = ("sedimentary", "igneous", "metamorphic")

# Geordneter Schichtstapel, Index 0 = ältestes/unterstes Glied (Basis),
# letzter Index = jüngstes/oberstes Glied. Kristallin bildet die Basis
# und erhält bewusst eine deutlich größere Dicke als die übrigen
# Schichten, da es im realen Vorbild das nach unten effektiv unbegrenzte
# Grundgebirge darstellt - hier als sehr dicke, aber endliche Schicht
# approximiert, damit die Stapel-Arithmetik (Punkt 2 des Plans) mit
# einer endlichen Gesamttiefe arbeiten kann.
ROCK_LAYERS: Tuple[RockLayer, ...] = (
    RockLayer("Kristallin", "metamorphic", (150, 30, 30), 2000.0, 1.30),
    RockLayer("Gefaltetes Paläozoikum", "metamorphic", (140, 110, 120), 400.0, 1.15),
    RockLayer("Rotliegend+Oberkarbon", "sedimentary", (200, 80, 140), 300.0, 0.85),
    RockLayer("Zechstein", "sedimentary", (110, 110, 190), 150.0, 0.50),
    RockLayer("Buntsandstein", "sedimentary", (230, 150, 100), 300.0, 0.75),
    RockLayer("Muschelkalk", "sedimentary", (220, 170, 210), 200.0, 0.95),
    RockLayer("Keuper", "sedimentary", (220, 120, 60), 250.0, 0.55),
    RockLayer("Jura", "sedimentary", (150, 200, 230), 300.0, 0.90),
    RockLayer("Kreide", "sedimentary", (170, 200, 90), 250.0, 0.60),
    RockLayer("Alttertiär", "sedimentary", (255, 230, 80), 150.0, 0.50),
    RockLayer("Jungtertiär", "sedimentary", (255, 255, 200), 150.0, 0.35),
    RockLayer("Pleistozän", "sedimentary", (222, 214, 175), 80.0, 0.25),
    RockLayer("Holozän", "sedimentary", (200, 200, 200), 40.0, 0.15),
)

# Lokale Intrusionskörper (siehe GeologySystemGenerator._apply_intrusions())
# durchschlagen den Stapel unabhängig von der Ausbiss-Berechnung - kein
# Stapel-Glied, daher separat definiert statt in ROCK_LAYERS.
BASALT_INTRUSION = RockLayer("Känozoischer Basalt", "igneous", (30, 120, 60), 0.0, 1.00)

# Bequemer Zugriff für Rendering/Legende: alle darstellbaren Gesteinstypen
# inkl. Intrusion, in Anzeige-Reihenfolge (älteste zuerst, Intrusion zuletzt).
ALL_ROCK_TYPES: Tuple[RockLayer, ...] = ROCK_LAYERS + (BASALT_INTRUSION,)

N_LAYERS = len(ROCK_LAYERS)

# Nominale Gesamttiefe des unverformten Stapels (Summe aller
# base_thickness_m) - Referenzwert für die Dicken-Normalisierungs-Toleranz
# in geology_generator._build_layer_thickness().
NOMINAL_TOTAL_DEPTH_M = sum(layer.base_thickness_m for layer in ROCK_LAYERS)

# Z-Höhe (Meter, gleiche Skala wie terrain_height) der Oberkante der
# obersten/jüngsten Schicht (Holozän) im unverformten Stapel, RELATIV zum
# Terrain-Hub (siehe TectonicDisplacementField._build_terrain_hub() in
# core/geology_generator.py) - der Stapel wird von OBEN nach UNTEN
# verankert, ältere Schichten liegen darunter.
#
# WICHTIG, kein reiner Bezugspunkt bei 0: der Stapel braucht "Kopffreiheit"
# oberhalb des Terrain-Hubs, sonst würde JEDE Stelle, die minimal über
# ihrem eigenen regional geglätteten Mittel liegt (praktisch jeder
# Hügel/Gipfel), sofort den gesamten Stapel überragen und pauschal die
# jüngste Schicht zeigen (ausgereizter Clip statt abgestufter Variation).
# Ein positiver Headroom-Wert lässt "durchschnittliches" Terrain (Höhe
# ≈ Hub) mittig im Stapel landen; nur Terrain, das deutlich ÜBER seinem
# eigenen regionalen Mittel liegt (echte Gipfel), erreicht dann Kristallin.
# 25% der Gesamttiefe als Startwert - Feinabstimmung braucht einen echten
# Live-Vergleich (siehe Geology-Review-Konvention für Farbskalen-Kalibrierung).
TOP_REFERENCE_HEIGHT_M = 0.25 * NOMINAL_TOTAL_DEPTH_M


def category_index(category: str) -> int:
    """Index einer Härte-Kategorie in HARDNESS_CATEGORIES, für Vektor-Lookups."""
    return HARDNESS_CATEGORIES.index(category)
