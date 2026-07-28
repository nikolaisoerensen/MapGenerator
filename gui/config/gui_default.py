"""
Path: gui/config/gui_default.py

Funktionsweise: GUI-Layout und Styling-Konfiguration
- Window-Größen und Positionen für alle Tabs
- Button-Styling (Farben, Größen, Fonts)
- Canvas-Konfiguration (Split-Ratios, Render-Settings)
- Color-Schemes und Theme-Definitionen

Struktur:
class WindowSettings:
    MAIN_MENU = {"width": 800, "height": 600}
    MAP_EDITOR = {"width": 1500, "height": 1000}

class ButtonSettings:
    PRIMARY = {"color": "#27ae60", "hover": "#229954"}
    SECONDARY = {"color": "#3498db", "hover": "#2980b9"}

class CanvasSettings:
    SPLIT_RATIO = 0.7  # 70% Canvas, 30% Controls
"""


class WindowSettings:
    """
    Funktionsweise: Definiert Standard-Fenstergrößen und Positionen für alle GUI-Komponenten
    Aufgabe: Zentrale Konfiguration aller Window-Parameter für konsistente Darstellung
    """
    MAIN_MENU = {
        "width": 1200,
        "height": 800,
        "min_width": 600,
        "min_height": 600
    }

    MAP_EDITOR = {
        "width": 1500,
        "height": 1000,
        "min_width": 1200,
        "min_height": 800
    }


class ButtonSettings:
    """
    Funktionsweise: Definiert einheitliche Button-Styles für alle UI-Komponenten
    Aufgabe: Konsistente Farbgebung und Hover-Effekte für alle Buttons
    """
    PRIMARY = {
        "color": "#488852",
        "hover": "#5e8964",
        "font_size": 18,
        "font_weight": "bold"
    }

    SECONDARY = {
        "color": "#487188",
        "hover": "#5e7a89",
        "font_size": 18,
        "font_weight": "normal"
    }

    DANGER = {
        "color": "#884858",
        "hover": "#895e69",
        "font_size": 18,
        "font_weight": "bold"
    }


class CanvasSettings:
    """
    Funktionsweise: Konfiguriert Canvas-Layout und Render-Parameter für Map-Display
    Aufgabe: Einheitliche Split-Ratios und Render-Einstellungen für alle Map-Tabs
    """
    SPLIT_RATIO = 0.7  # 70% Canvas, 30% Controls

    # 2D Canvas Settings
    CANVAS_2D = {
        "background_color": "#2c3e50",
        "grid_color": "#34495e",
        # Kein Blau mehr (sah wie Wasser aus, siehe elevation_vmin/vmax unten) -
        # matplotlib zyklt durch diese Liste je Contour-Level, betraf also nicht
        # nur niedrige Höhen sondern Linien über die ganze Karte verteilt.
        "contour_colors": ["#7f8c8d", "#e74c3c", "#f39c12"],
        "dpi": 100,
        # Feste Höhen-Farbskala (Meter) statt Auto-Skalierung pro Heightmap,
        # damit z.B. ein 200m-Hügel und ein 3500m-Berg nicht dieselbe volle
        # Farbspanne bekommen - entspricht realen topografischen Karten.
        # Werte oberhalb von elevation_vmax clippen auf die höchste Farbe.
        "elevation_vmin": 0.0,
        "elevation_vmax": 4000.0,
        # Feste Farbskalen pro Daten-Layer (colormap_name, vmin, vmax) - ersetzt
        # die bisherige Auto-Skalierung pro Frame (matplotlib nimmt sonst das
        # aktuelle Min/Max der gerade angezeigten Daten), die bei wechselnden
        # Karten (z.B. der saisonalen Monats-Animation im Weather-Tab) zu
        # "flackernden" Farben führte, weil derselbe Absolutwert je nach Frame
        # eine andere Farbe bekam. Dieselbe Tabelle treibt sowohl die 2D-
        # imshow()-Aufrufe (map_display_2d.py) als auch die 3D-Overlay-Textur-
        # Einfärbung (map_display_3d.py _colorize_layer()). Werte sind erste
        # plausible Richtwerte (analog zur RAIN_THRESHOLD-Kalibrierung dieser
        # Session empirisch nachjustierbar) - precip_map orientiert sich an der
        # bereits dokumentierten realen Größenordnung (~0-2.8 bei Default-
        # Parametern). rock_map/biome_map/super_biome_mask sind bewusst NICHT
        # hier gelistet - sie nutzen bereits eigene kategorische Farbtabellen
        # (ColorSchemes.BIOME_COLOR_TABLE etc.), keine kontinuierliche Skala.
        "layer_ranges": {
            "temp_map": ("RdBu_r", -30.0, 40.0),        # Blau=kalt, Rot=warm
            # Nutzer-Abstimmung 2026-07-24 (Revision der Zwischen-Kalibrierung
            # vom 2026-07-23): precip_map ist keine Jahresmenge, sondern eine
            # Perioden-Akkumulation mit ~50 als typischem Maximalwert (siehe
            # PRECIP_ANNUAL_SCALE_FACTOR=0.5 in core/weather_generator.py) -
            # vmax mit etwas Puffer über diesem typischen Maximum, damit
            # seltene, legitime Ausreißer noch sichtbar differenzierbar
            # bleiben statt sofort auf die volle Sättigungsfarbe zu clippen.
            "precip_map": ("Greens", 0.0, 70.0),
            "humid_map": ("Blues", 0.0, 100.0),
            "wind_map": ("plasma", 0.0, 40.0),           # Windstärke m/s - EINE Farbskala für Heatmap-Hintergrund, Stromlinien UND Pfeile in _render_wind_map (siehe dortiger Docstring: vorher hatte die Colorbar keinen Bezug zu den tatsächlichen Pfeil-/Stromlinien-Farben)
            "water_map": ("Blues", 0.0, 10.0),
            "flow_map": ("Blues", 0.0, 50.0),
            # Logarithmisch statt linear: Werteverteilung ist stark
            # rechtsschief (die meisten Pixel exakt 0, wenige Ausreißer
            # deutlich höher) - eine lineare Skala ließ den typischen/
            # repräsentativen Wertebereich kaum vom Hintergrund unterscheiden.
            # Nutzer-Abstimmung 2026-07-25 (Revision der Kalibrierung vom
            # 2026-07-24 - der alte 0.02-0.5-Bereich stammte noch aus der Zeit
            # des festen 0.1m-Erosions-Deckels; seit dem relief-relativen
            # Deckel (EROSION_CAP_RELIEF_FRACTION, core/water_generator.py
            # ErosionSedimentationSystem) reichen akkumulierte Werte je nach
            # Kartenrelief bis in den zweistelligen Meterbereich - der alte
            # Bereich sättigte dadurch fast die gesamte Karte auf die oberste
            # Farbe). Neuer Bereich 0.01-40m deckt vom gerade noch sichtbaren
            # cm-Bereich bis zu einem substantiellen Gebirgs-Szenario ab
            # (empirisch, siehe scratch_erosion_accumulated_depth_check.py -
            # akkumulierter Erosions-Max von ~40m bei 850m Kartenrelief).
            # Erosion/Sedimentation nutzen bewusst DENSELBEN Wertebereich
            # (Nutzer-Vorgabe: direkt vergleichbar, gleiche Skala) - wie bei
            # allen statischen Bereichen in dieser Tabelle KEINE Auto-
            # Skalierung pro Karte, daher bei sehr flachen (kleines Relief)
            # oder sehr extremen (Relief >> 2000m) Karten weiterhin nicht
            # perfekt ausgenutzt - akzeptierter Kompromiss, konsistent mit
            # z.B. temp_map's ebenfalls festem Bereich.
            # NEU KALIBRIERT 2026-07-27. Der Bereich 0.01-40 m darüber stammte
            # aus der Zeit, als Erosion auf JEDER LOD-Stufe einmal lief. Seit
            # dem Umbau auf "nur am finalen LOD, dafür mit 4 Durchgängen und
            # ~5 Partikeln pro Pixel" sind beide Karten kumulierter DURCHSATZ
            # über sehr viel mehr Partikelwege: eine Zelle gibt dieselbe Fracht
            # über den Lauf hinweg mehrfach ab und nimmt sie wieder auf (siehe
            # DropletErosionSystem.simulate_erosion_sedimentation, Abschnitt
            # "BEIDE KARTEN SIND KUMULIERTER DURCHSATZ"). Gemessen, jeweils
            # 5 Partikel/Pixel und 4 Durchgänge:
            #
            #     Kartengröße   Median (>0)   99.9%      Maximum
            #     128 px            ~100 m    ~7 100 m   ~10 000 m
            #     512 px              75 m    17 383 m    41 698 m
            #
            # Mit vmax = 40 m sättigte praktisch die gesamte Karte auf die
            # oberste Farbe, und der Kanalausgang stach als einzelner
            # gesättigter Fleck heraus - genau das vom Nutzer auf der
            # 512er-Karte gemeldete Bild.
            #
            # vmax ist bewusst am 99.9. Perzentil der größten hier gemessenen
            # Karte ausgerichtet, nicht am Maximum: darüber liegen nur die
            # wenigen Zellen des Kanalausgangs, und die sollen sättigen statt
            # den Rest der Verteilung zusammenzudrücken. Der Kompromiss der
            # ganzen Tabelle gilt auch hier - der Durchsatz wächst mit der
            # Kartengröße (mehr Partikel, längere Wege), eine statische Skala
            # kann deshalb nicht jede Größe gleich gut ausnutzen. Die
            # Log-Skala federt genau das ab.
            # NEU KALIBRIERT 2026-07-28 fuer das Feldverfahren
            # (core/erosion_generator.py), Werte GEMESSEN statt geschaetzt.
            #
            # Erster Versuch war 0-400 m LINEAR. Das war zweifach falsch:
            # der Bereich zu weit und die Skala zur Verteilung unpassend.
            # Gemessen (Relief 4000 m, Regen 2.0):
            #
            #   Sim px  Schritte   max Erosion   95. Perzentil
            #      128       300          90 m           24 m
            #      128      2000         245 m           75 m
            #      128      7200         275 m          127 m
            #      256      2000         247 m           88 m
            #
            # Der typische Wert liegt also weit unter dem Maximum - die
            # Verteilung ist stark rechtsschief, weil die meisten Zellen kaum
            # etwas abbekommen und die Kanaele alles. Auf einer linearen
            # 0-400-Skala landete damit praktisch die ganze Karte in der
            # hellsten Farbstufe: der Nutzer sah "leere Karten", obwohl die
            # Daten da waren. Logarithmisch mit 0.5 m als Untergrenze (darunter
            # ist es Rauschen) und 300 m als Obergrenze bildet den gemessenen
            # Bereich ueber alle Laufzeiten ab.
            "erosion_map": ("Reds", 0.5, 300.0, "log"),
            "sedimentation_map": ("Oranges", 0.5, 300.0, "log"),
            # Signierte Netto-Hoehenaenderung - die Karte, an der eine
            # unplausible Spitze sofort auffaellt. Divergierende Skala um 0,
            # deshalb linear (eine Log-Skala kann keine Vorzeichen). Der
            # Bereich ist am 95. Perzentil ausgerichtet und nicht am Maximum:
            # die wenigen Extremzellen sollen saettigen, statt den Rest der
            # Karte in ein einheitliches Grau zu druecken.
            "net_change_map": ("RdBu_r", -150.0, 150.0, "linear"),
            # Die noch im Wasser geloeste Fracht - im Vorbild die cyanfarbenen
            # Frachtspuren, deshalb dieselbe Farbfamilie. Gemessenes Maximum
            # ueber alle Laeufe: 0.6 bis 1.1 m.
            "sediment_load_map": ("GnBu", 0.0, 1.2, "linear"),
            # Fliessgeschwindigkeit. Die Obergrenze ist keine Schaetzung,
            # sondern folgt aus dem Modell: der Netto-Fluss kann hoechstens
            # das Doppelte der Gitter-Geschwindigkeit erreichen (siehe
            # HydraulicFieldSimulator.MAX_FLOW_VELOCITY_M_S) - gemessenes
            # Maximum 15.9 m/s bei einer Bezugsgeschwindigkeit von 8 m/s.
            "flow_velocity_map": ("viridis", 0.0, 16.0, "linear"),
            # Eingeschwungener Wasserstand aus dem Erosionslauf. Eigener
            # Eintrag statt Mitbenutzung von "water_map": dort steht der
            # Wasserstand des Water-Generators mit echtem Regen, hier der des
            # synthetischen Erosionsregens - zwei verschiedene Groessen, die
            # sich nicht dieselbe Skala teilen sollten. Gemessen 1.9 bis 15 m
            # je nach Laufzeit.
            "water_depth_map": ("Blues", 0.0, 12.0, "linear"),
            # Böschungswinkel-Erosion (2026-07-27 als eigene Anzeigemodi
            # ergänzt, siehe gui/tabs/water_tab.py) - derselbe Wertebereich
            # und dieselbe Log-Skala wie die fluviale Erosion/Sedimentation
            # darüber, damit sich beide Prozesse direkt vergleichen lassen.
            # Farbwahl bewusst anders (violett/grün statt rot/orange), damit
            # auf einen Blick erkennbar bleibt, welcher Mechanismus gerade
            # angezeigt wird.
            "thermal_erosion_map": ("Purples", 0.01, 40.0, "log"),
            "thermal_deposition_map": ("Greens", 0.01, 40.0, "log"),
            # Verdunstung in gH2O/m²/Tag - Obergrenze am Lake-Limit der
            # Klassifikationsstufen orientiert (siehe
            # EvaporationCalculator._EVAPORATION_LIMIT_BY_WATER_TYPE:
            # Grand River deckelt bei 200, Seen sind unbegrenzt).
            "evaporation_map": ("YlOrBr", 0.0, 250.0),
            "soil_moist_map": ("Blues", 0.0, 100.0),
            # Zivilisations-Einfluss 0-1 (settlement.civ_influence). Bis
            # 2026-07-27 ohne Eintrag: 2D nutzte fest plasma mit
            # Auto-Skalierung, 3D fiel auf viridis mit eigener
            # Auto-Skalierung zurueck - derselbe Layer sah in beiden
            # Ansichten unterschiedlich aus. Fester Bereich statt
            # Auto-Skalierung, damit sich zwei Karten direkt vergleichen
            # lassen (gleiches Prinzip wie bei allen Eintraegen hier).
            "civ_map": ("plasma", 0.0, 1.0),
            "slopemap": ("viridis", 0.0, 90.0),         # Grad
            "hardness_map": ("viridis", 0.0, 100.0),
            # Geology-Δz-Diagnose-Layer (3D-Gesteinsstapel-Rework, siehe
            # core/geology_generator.py): signierte Höhen-Komponenten in Metern -
            # diverging Colormap (Blau=Absenkung, Rot=Anhebung), vmin/vmax
            # bewusst None (Auto-Skalierung) statt geraten - konkrete Werte
            # müssten gegen einen echten Diagnose-Lauf kalibriert werden, siehe
            # docs/session_review_2026-07-22_geology.md.
            "terrain_hub_delta": ("RdBu_r", None, None),
            "tilt_delta": ("RdBu_r", None, None),
            "fold_delta": ("RdBu_r", None, None),
            "fault_delta": ("RdBu_r", None, None),
            "intrusion_delta": ("RdBu_r", None, None),
        }
    }

    # 3D Canvas Settings
    CANVAS_3D = {
        "background_color": (0.17, 0.24, 0.31, 1.0),  # RGBA
        # Sonne kommt aus Süden (nicht an generate_seasonal_sun_angles()
        # gekoppelt, das ist die separate Terrain-Shadowmap-Berechnung -
        # hier reicht ein fester, plausibler Wert für die 3D-Preview).
        # Koordinaten-Konvention (verifiziert): map_display_2d.py nutzt
        # imshow(..., origin='lower') -> Zeile 0 der Heightmap liegt UNTEN
        # im Bild = Süden, letzte Zeile = Norden. _generate_terrain_mesh()
        # bildet Zeile 0 auf negatives world-Z ab (pos_z = (y_idx/(h-1)-0.5)*...).
        # Also: world -Z = Süden, world +Z = Norden. Süd-Sonne braucht daher
        # negatives Z, mittig in X, ~45 Grad Elevation.
        "light_position": (0.0, 10.6, -10.6),
        # Terrain-Mesh wird immer auf ein 10x10-Einheiten XZ-Footprint normiert
        # (siehe MapDisplay3D.terrain_scale_factor). 5.0 war kleiner als der
        # Mesh-Diagonal-Halbradius (~7.07) selbst - bei 45° FOV landeten
        # Vertices dadurch weit außerhalb von NDC [-1,1] und es wurde nichts
        # sichtbar. ~17 umfasst das Mesh bei 45° FOV mit Rand.
        "camera_distance": 17.0,
        "fov": 45.0
    }


class ColorSchemes:
    """
    Funktionsweise: Definiert Farbpaletten für verschiedene Map-Visualisierungen
    Aufgabe: Konsistente Farbgebung für Heightmaps, Biomes und andere Visualisierungen
    """
    TERRAIN = {
        "low": "#2980b9",  # Deep Blue
        "erosion": "#c0703a",  # Braun-Orange - Gelaendeformung
        "water": "#3498db",  # Blue
        "land": "#27ae60",  # Green
        "mountain": "#95a5a6",  # Gray
        "high": "#ecf0f1"  # White
    }

    # Einzige Quelle für Biome-Index -> (Name, Farbe), genutzt von
    # BiomeLegendDialog (gui/widgets/widgets.py) UND MapDisplay2D._render_biome_map
    # (gui/widgets/map_display_2d.py) - Index entspricht exakt den Werten aus
    # core/biome_generator.py (0-14 Base-Biomes, 15-25 Super-Biomes über
    # SuperBiomeOverrideSystem.super_biome_offset=15). Vorher hatten Legende und
    # Darstellung getrennte, auseinandergelaufene Farbdefinitionen.
    BIOME_COLOR_TABLE = [
        # Base Biomes (Index 0-14)
        ("Ice Cap", "#f8f9fa"),
        ("Tundra", "#e9ecef"),
        ("Taiga", "#228b22"),
        ("Grassland", "#90ee90"),
        ("Temperate Forest", "#006400"),
        ("Mediterranean", "#9acd32"),
        ("Desert", "#daa520"),
        ("Semi Arid", "#d2691e"),
        ("Tropical Rainforest", "#008000"),
        ("Tropical Seasonal", "#32cd32"),
        ("Savanna", "#bdb76b"),
        ("Montane Forest", "#2e8b57"),
        ("Swamp", "#556b2f"),
        ("Coastal Dunes", "#f4a460"),
        ("Badlands", "#a0522d"),
        # Super Biomes (Index 15-25)
        ("Ocean", "#0077be"),
        ("Lake", "#4da6ff"),
        ("Grand River", "#0066cc"),
        ("River", "#3399ff"),
        ("Creek", "#66b3ff"),
        ("Cliff", "#696969"),
        ("Beach", "#f5deb3"),
        ("Lake Edge", "#87ceeb"),
        ("River Bank", "#98fb98"),
        ("Snow Level", "#fffafa"),
        ("Alpine Level", "#d3d3d3"),
    ]


class LayoutSettings:
    """
    Funktionsweise: Definiert Standard-Layout-Parameter für alle GUI-Komponenten
    Aufgabe: Einheitliche Abstände, Margins und Padding-Werte
    """
    PADDING = 10
    MARGIN = 15
    BUTTON_HEIGHT = 50
    SLIDER_HEIGHT = 25
    LABEL_HEIGHT = 20

    CONTROL_PANEL_WIDTH = 300
    STATUS_BAR_HEIGHT = 25

class AppConstants:
    """Application-wide timing and behavior constants"""
    MEMORY_CHECK_INTERVAL_MS = 60000  # Memory monitoring interval
    CLEANUP_DELAY_MS = 2000  # Cleanup protection delay

# Application constants for timing and behavior
class EditorConstants:
    """MapEditor-specific constants for consistent behavior"""
    STATUS_UPDATE_INTERVAL_MS = 5000     # Status bar update frequency
    GENERATION_TIMEOUT_MS = 300000       # 5 minute timeout for generation
    TAB_SWITCH_DELAY_MS = 100           # Delay for smooth tab transitions
    ERROR_DISPLAY_DURATION_MS = 5000    # Error message display time