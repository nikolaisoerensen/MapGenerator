"""
Path: gui/config/value_default.py

Funktionsweise: Zentrale Parameter-Defaults für alle Slider und Controls
- Min/Max/Step/Default Werte für alle Generator-Parameter aktualisiert
- Neue Parameter für alle Core-Module integriert
- Organisiert nach Generator-Typen (TERRAIN, GEOLOGY, SETTLEMENT, WEATHER, WATER, BIOME)
- Validation-Rules und Parameter-Constraints
- Einheitliche Decimal-Precision und Suffix-Definitionen
- "description"-Key pro Parameter: Hover-Text für den Info-Button neben dem
  Reset-Button jedes Sliders (siehe gui/widgets/widgets.py ParameterSlider) -
  erklärt in einfachen Worten, was der Parameter fachlich bewirkt.
"""
import random


class TERRAIN:
    """Parameter für core/terrain_generator.py"""
    MAPSIZEMIN = 32
    MAPSIZEMAX = 1024

    # Reale Weltgröße, die die Karte immer abdeckt, unabhängig von map_size/
    # Pixelauflösung (siehe gui/widgets/map_display_3d.py _calculate_terrain_scaling()
    # und core/terrain_generator.py SlopeCalculator._calculate_cpu_slopes() - beide
    # brauchen dieselbe Meter-pro-Pixel-Annahme, sonst driften 3D-Darstellung und
    # Slope-basierte Physik/Biome-Klassifikation auseinander). Bleibt als reiner
    # Default-/Fallback-Wert bestehen (siehe MAP_DISTANCE_KM["default"] unten) -
    # der tatsächliche, live einstellbare Wert kommt seit
    # [[project-terrain-review]] über DataLODManager.get_map_distance_km(),
    # nicht mehr aus diesem statischen Import.
    WORLD_SIZE_KM = 10.0

    MAPSIZE = {
        "min": MAPSIZEMIN, "max": MAPSIZEMAX, "default": 128, "step": 32,
        "description": "Auflösung der Karte in Pixeln (Breite = Höhe). Größere "
                        "Werte zeigen mehr Detail, verlangsamen aber jede "
                        "nachfolgende Generierungsstufe."
    }
    # Bisher fest als WORLD_SIZE_KM-Konstante codiert, geteilt von 5 Dateien
    # (terrain_generator.py, biome_generator.py, water_generator.py,
    # weather_generator.py, map_display_3d.py) - jetzt ein echter Slider
    # direkt unter Map Size, da Map Size (Pixel-Auflösung) und Map Distance
    # (reale km-Ausdehnung) zwei unabhängige, aber eng verwandte Größen sind.
    MAP_DISTANCE_KM = {
        "min": 1.0, "max": 100.0, "default": WORLD_SIZE_KM, "step": 1.0, "suffix": "km",
        "description": "Reale Ausdehnung der Karte in Kilometern (Breite = "
                        "Höhe), unabhängig von der Pixel-Auflösung (Map "
                        "Size). Bestimmt, wie viele reale Meter ein Pixel "
                        "abdeckt - beeinflusst dadurch Steigungen, "
                        "Schatten und alle geländeabhängigen Berechnungen "
                        "in Geology/Weather/Water/Biome."
    }
    # Default 4000m (statt vorher 2000m), damit die Farbskala (0-4000m, siehe
    # CanvasSettings.CANVAS_2D) beim Default-Seed auch tatsächlich ausgenutzt wird.
    AMPLITUDE = {
        "min": 30, "max": 6000.0, "default": 4000.0, "step": 10, "suffix": "m",
        "description": "Maximale Höhendifferenz der Landschaft in Metern - "
                        "bestimmt, wie hoch die Berge im Vergleich zu den "
                        "Tälern werden."
    }
    # Default 4 (statt vorher 8): die Pixel-Koordinaten-Frequenz ist NICHT auf
    # die Kartengröße normalisiert (siehe SimplexNoiseGenerator._generate_cpu_optimized),
    # bei frequency=0.037 und lacunarity=2.3 übersteigt die Oktaven-Frequenz ab
    # Oktave 5 den Wert 1.0 (Wellenlänge unter einem Pixel) - diese Oktaven fügen
    # nur noch unkorreliertes "Static"-Rauschen statt Landschaftsdetail hinzu.
    # BaseTerrainGenerator._calc_noise() clamped effective_octaves seit
    # [[project-terrain-review]] automatisch auf die Nyquist-Grenze (0.5
    # Zyklen/Pixel) herunter, sodass ein zu hoher Slider-Wert kein kaputtes
    # Ergebnis mehr erzeugt - er hat ab diesem Punkt nur schlicht keine
    # sichtbare Wirkung mehr, siehe Beschreibung unten.
    OCTAVES = {
        "min": 1, "max": 8, "default": 4, "step": 1,
        "description": "Anzahl der übereinandergelegten Rausch-Schichten "
                        "unterschiedlicher Frequenz. Mehr Oktaven fügen "
                        "feinere Detailebenen hinzu, verlangsamen aber die "
                        "Berechnung. Oktaven jenseits der aktuellen "
                        "Frequency/Frequency-Scaling-Kombination werden "
                        "automatisch ignoriert, sobald ihre Frequenz über "
                        "0.5 Zyklen/Pixel liegt (kein sichtbarer Effekt "
                        "mehr möglich) - bei den Standardwerten betrifft "
                        "das bereits Oktave 5 und höher."
    }
    FREQUENCY = {
        "min": 0.001, "max": 0.1, "default": 0.037, "step": 0.001,
        "description": "Grundfrequenz des Rausch-Musters - höhere Werte "
                        "erzeugen kleinere, dichter aufeinanderfolgende "
                        "Hügel/Täler, niedrigere Werte großflächigere "
                        "Formationen."
    }
    # Default 0.4 (statt vorher 0.68): bei 0.68 tragen selbst hochfrequente
    # Oktaven noch spürbar zur Gesamthöhe bei, was zusammen mit den vielen
    # Oktaven den zerklüfteten "Static"-Look verursacht hat.
    PERSISTENCE = {
        "min": 0.1, "max": 1.0, "default": 0.4, "step": 0.01,
        "description": "Wie stark jede zusätzliche Rausch-Oktave zur "
                        "Gesamthöhe beiträgt. Höhere Werte lassen feine "
                        "Details stärker durchschlagen - macht die "
                        "Landschaft zerklüfteter/rauer."
    }
    LACUNARITY = {
        "min": 1.1, "max": 4.0, "default": 2.3, "step": 0.1,
        "description": "Wie stark sich die Frequenz von einer Rausch-Oktave "
                        "zur nächsten erhöht. Höhere Werte spreizen die "
                        "Detailebenen (grob vs. fein) weiter auseinander."
    }
    # Höher als vorher (2.5 -> 3.5): drückt die Masse der Landschaft näher an
    # die Talsohle, seit _apply_redistribution() gegen amplitude statt gegen
    # das Sample-Min/Max normalisiert (siehe core/terrain_generator.py).
    REDISTRIBUTE_POWER = {
        "min": 0.5, "max": 4.0, "default": 3.5, "step": 0.1,
        "description": "Verzerrt die Höhenverteilung nach der Rausch-"
                        "Erzeugung - höhere Werte drücken den Großteil der "
                        "Landmasse näher zur Talsohle (mehr Ebenen, "
                        "spitzere und isoliertere Berggipfel)."
    }
    MAP_SEED = {
        # Zufällig bei JEDEM Programmstart neu gewürfelt (Modul-Import
        # passiert einmal pro Prozessstart) - Nutzer-Vorgabe: die App soll
        # nicht immer mit demselben Seed starten. Der Slider selbst bleibt
        # danach normal einstellbar/reproduzierbar, nur der Startwert ist
        # nicht mehr fest.
        "min": 0, "max": 999999, "default": random.randint(0, 999999), "step": 1,
        "description": "Zufalls-Startwert für die Terrain-Generierung - "
                        "derselbe Seed erzeugt bei sonst gleichen "
                        "Parametern immer exakt dieselbe Karte."
    }

class GEOLOGY:
    """
    Parameter für core/geology_generator.py (3D-Gesteinsstapel-Modell, siehe
    docs/session_design_2026-07-22_geology_3dstack_concept.md und den
    zugehörigen Umsetzungsplan). Die früheren RIDGE_WARPING/BEVEL_WARPING/
    METAMORPH_FOLIATION/METAMORPH_FOLDING/IGNEOUS_FLOWING-Regler sind ersatzlos
    entfernt - ihre Effekte sind in den Reglern unten als benannte Komponenten
    des gemeinsamen Tektonik-Verschiebungsfelds aufgegangen (Ridge -> FOLD_DETAIL,
    Bevel -> FAULT_EDGE_SOFTNESS, Folding -> FOLD_INTENSITY, Foliation ->
    FOLIATION_DETAIL, Igneous Flowing -> INTRUSION_DETAIL).
    """
    SEDIMENTARY_HARDNESS = {
        "min": 1, "max": 100, "default": 30, "step": 1,
        "description": "Widerstandsfähigkeit von Sedimentgestein gegen "
                        "Erosion (0=weich, 100=hart) - beeinflusst, wie "
                        "schnell Wasser dieses Gestein abträgt."
    }
    IGNEOUS_HARDNESS = {
        "min": 1, "max": 100, "default": 80, "step": 1,
        "description": "Widerstandsfähigkeit von Vulkangestein gegen "
                        "Erosion (0=weich, 100=hart) - beeinflusst, wie "
                        "schnell Wasser dieses Gestein abträgt."
    }
    METAMORPHIC_HARDNESS = {
        "min": 1, "max": 100, "default": 65, "step": 1,
        "description": "Widerstandsfähigkeit von metamorphem (umgewandeltem) "
                        "Gestein gegen Erosion (0=weich, 100=hart) - "
                        "beeinflusst, wie schnell Wasser dieses Gestein "
                        "abträgt."
    }
    TILT_INTENSITY = {
        "min": 0.0, "max": 150.0, "default": 15.0, "step": 1.0, "suffix": " m/km",
        "description": "Verkippt den gesamten Gesteinsstapel wie eine schiefe "
                        "Ebene (Meter Höhenunterschied pro km) - bei 0 bleibt "
                        "der Stapel horizontal, höhere Werte kippen ihn "
                        "sichtbar in eine Richtung (siehe Tilt Direction). "
                        "Wirkt NUR auf den Gesteinsstapel/Ausbiss, NICHT auf "
                        "die sichtbare Geländehöhe - deshalb darf der Regler "
                        "deutlich stärker eingestellt werden als bisher."
    }
    TILT_DIRECTION = {
        "min": 0.0, "max": 360.0, "default": 45.0, "step": 5.0, "suffix": "°",
        "description": "Richtung der Verkippung in Grad (0=Ost, 90=Nord) - "
                        "ohne Wirkung, solange Tilt Intensity 0 ist."
    }
    FOLD_INTENSITY = {
        "min": 0.0, "max": 2000.0, "default": 400.0, "step": 20.0, "suffix": " m",
        "description": "Amplitude der großräumigen geologischen Faltung in "
                        "Metern - verschiebt den Gesteinsstapel wellenförmig "
                        "auf und ab (Antiklinalen/Synklinalen) und damit, "
                        "welche Schicht wo ausbeißt. Wirkt NUR auf den "
                        "Gesteinsstapel, NICHT auf die sichtbare Geländehöhe "
                        "- deshalb darf der Regler deutlich stärker "
                        "eingestellt werden als bisher."
    }
    FOLD_DETAIL = {
        "min": 0.0, "max": 1.0, "default": 0.4, "step": 0.05,
        "description": "Mischt eine feinere, unregelmäßigere Rauheits-Komponente "
                        "in die Faltung (entspricht dem früheren Ridge Warping) - "
                        "0 = nur die glatte großräumige Faltung, höhere Werte "
                        "lassen die Faltenzüge weniger geradlinig wirken."
    }
    FAULT_INTENSITY = {
        "min": 0.0, "max": 400.0, "default": 100.0, "step": 10.0, "suffix": " m",
        "description": "Vertikaler Versatz (Sprunghöhe) an Störungslinien in "
                        "Metern - 0 erzeugt kein Störungsnetz, höhere Werte "
                        "lassen Gesteinsblöcke deutlicher gegeneinander "
                        "versetzt erscheinen."
    }
    FAULT_DETAIL = {
        "min": 0.0, "max": 1.0, "default": 0.5, "step": 0.05,
        "description": "Verzweigungstiefe/-dichte des Störungsnetzes - "
                        "niedrige Werte erzeugen wenige, einfache "
                        "Bruchlinien, hohe Werte ein dichteres, stärker "
                        "verästeltes Netz."
    }
    FAULT_EDGE_SOFTNESS = {
        "min": 0.02, "max": 2.0, "default": 0.3, "step": 0.02, "suffix": " km",
        "description": "Breite der weichen Übergangszone an einer Störungskante "
                        "in km (entspricht dem früheren Bevel Warping) - kleine "
                        "Werte ergeben einen scharfen Versatz-Sprung, große "
                        "Werte eine breite, allmähliche Übergangszone."
    }
    INTRUSION_DENSITY = {
        "min": 0.0, "max": 1.0, "default": 0.3, "step": 0.05,
        "description": "Anzahl vulkanischer Intrusionskörper (Basalt), die den "
                        "Gesteinsstapel lokal durchschlagen - 0 erzeugt keine "
                        "Intrusionen."
    }
    INTRUSION_SIZE = {
        "min": 0.2, "max": 5.0, "default": 1.0, "step": 0.1, "suffix": " km",
        "description": "Typischer Radius einer Intrusion in km."
    }
    INTRUSION_DETAIL = {
        "min": 0.0, "max": 1.0, "default": 0.5, "step": 0.05,
        "description": "Randrauschen der Intrusionskörper (entspricht dem "
                        "früheren Igneous Flowing) - 0 ergibt kreisrunde "
                        "Ränder, höhere Werte unregelmäßigere, natürlicher "
                        "wirkende Konturen."
    }
    METAMORPHIC_OVERPRINT_INTENSITY = {
        "min": 0.0, "max": 1.0, "default": 0.4, "step": 0.05,
        "description": "Reichweite/Stärke der Gesteins-Umwandlung (Härte-"
                        "Anhebung Richtung Metamorphic Hardness) in der Nähe "
                        "von Störungen und Intrusionen - 0 deaktiviert jede "
                        "Metamorphose-Wirkung."
    }
    FOLIATION_DETAIL = {
        "min": 0.0, "max": 1.0, "default": 0.5, "step": 0.05,
        "description": "Feinheit/Kontrast der Schieferungs-Textur (parallele "
                        "Streifen) in metamorph überprägten Zonen - rein "
                        "visuell, ohne jede Höhenwirkung."
    }

class SETTLEMENT:
    """Parameter für core/settlement_generator.py"""
    SETTLEMENTS = {
        "min": 1, "max": 5, "default": 3, "step": 1,
        "description": "Anzahl der Hauptsiedlungen (Städte/Dörfer), die auf "
                        "der Karte platziert werden."
    }
    LANDMARKS = {
        "min": 0, "max": 6, "default": 3, "step": 1,
        "description": "Anzahl markanter Landmarken (z.B. Ruinen, besondere "
                        "Orte) abseits der Siedlungen."
    }
    ROADSITES = {
        "min": 0, "max": 6, "default": 3, "step": 1,
        "description": "Anzahl zusätzlicher kleiner Wegpunkte/Raststätten "
                        "entlang der Überlandstraßen."
    }
    PLOTNODES = {
        "min": 50, "max": 5000, "default": 200, "step": 10,
        "description": "Anzahl der Kandidaten-Punkte für die Grundstücks-/"
                        "Bebauungsplanung - mehr Punkte erlauben feinere "
                        "Parzellierung, kosten aber Rechenzeit. Wird NICHT "
                        "mit der Kartengröße mitskaliert (Plot Base Spacing "
                        "übernimmt das bereits) - dieselbe Anzahl ergibt bei "
                        "größerer Karte automatisch größere, aber gleich "
                        "dichte Parzellen."
    }
    CITY_SIZE = {
        "min": 0.0, "max": 1.0, "default": 0.5, "step": 0.05,
        "description": "Grundgröße einer Stadt - leitet city_reach_factor, "
                        "civ_influence_range und den Zwischenstädte-Verkehr "
                        "gemeinsam ab (0=kleine, kompakte Stadt, 1=weit "
                        "ausgreifende Großstadt)."
    }
    CIV_INFLUENCE_DECAY = {
        "min": 0.1, "max": 2.0, "default": 0.8, "step": 0.1,
        "description": "Wie schnell der zivilisatorische Einfluss einer "
                        "Siedlung mit der Entfernung abnimmt - höhere Werte "
                        "lassen Siedlungen isolierter wirken."
    }
    TERRAIN_FACTOR_VILLAGES = {
        "min": 0.0, "max": 2.0, "default": 1.0, "step": 0.1,
        "description": "Wie stark die Geländeform (Steigung etc.) die "
                        "Standortwahl für Siedlungen beeinflusst."
    }
    ROAD_SLOPE_TO_DISTANCE_RATIO = {
        "min": 0.1, "max": 3.0, "default": 1.5, "step": 0.1,
        "description": "Gewichtet beim Straßenbau Steigung gegen Distanz - "
                        "höhere Werte vermeiden steile Straßen auch auf "
                        "Kosten von Umwegen."
    }
    LANDMARK_WILDERNESS = {
        "min": 0.1, "max": 0.8, "default": 0.3, "step": 0.05,
        "description": "Wie weit abgelegen Landmarken bevorzugt platziert "
                        "werden (niedrig=nah an Zivilisation, hoch=tiefe "
                        "Wildnis)."
    }
    CITY_REACH_FACTOR = {
        "min": 1.0, "max": 10.0, "default": 4.0, "step": 0.5,
        "description": "Wie weit sich eine Stadt maximal ausdehnen kann, "
                        "bezogen auf ihre Grundgröße."
    }
    CIV_INFLUENCE_RANGE = {
        "min": 0.05, "max": 0.6, "default": 0.30, "step": 0.01, "suffix": "x diag",
        "description": "Reichweite des zivilisatorischen Einflusses einer "
                        "Siedlung, als Anteil der Kartendiagonale."
    }
    PLOT_BASE_SPACING = {
        "min": 2.0, "max": 60.0, "default": 20.0, "step": 1.0,
        "description": "Grundabstand zwischen einzelnen Grundstücks-"
                        "Parzellen."
    }
    PLOT_CIV_SPACING_FACTOR = {
        "min": 0.0, "max": 10.0, "default": 8.0, "step": 0.1,
        "description": "Wie stark sich der Parzellenabstand mit der Nähe "
                        "zum Stadtzentrum verringert (dichtere Bebauung im "
                        "Zentrum)."
    }
    PLOT_HEIGHT_COST_FACTOR = {
        "min": 0.0, "max": 10.0, "default": 3.0, "step": 0.1,
        "description": "Wie stark Höhenunterschiede die 'Baukosten' einer "
                        "Parzelle erhöhen - steile Grundstücke werden "
                        "dadurch seltener/kleiner bebaut."
    }

    # --- Plot Physics Advanced (aus tools/biome_lab/ 1:1 übernommene
    # Feder-Masse-Konstanten, bisher hardcodiert in PlotPhysicsSystem.__init__
    # ohne UI-Slider, siehe [[project-settlement-physics-lab-parity]]) ---
    CORE_PLOTNODE_SPRING_STIFFNESS = {
        "min": 0.0, "max": 5.0, "default": 1.2, "step": 0.1,
        "description": "Federsteifigkeit zwischen Plotkern und PlotNode "
                        "(Abstands-Feder, hält PlotNodes in Kern-Nähe)."
    }
    PLOTNODE_PLOTNODE_SPRING_STIFFNESS = {
        "min": 0.0, "max": 5.0, "default": 1.0, "step": 0.1,
        "description": "Federsteifigkeit zwischen benachbarten PlotNodes "
                        "entlang stark befahrener Verbindungen."
    }
    PRESSURE_STRENGTH = {
        "min": 0.0, "max": 4.0, "default": 0.8, "step": 0.1,
        "description": "Innendruck je Kern-Zelle (Flächenerhalt) - hält "
                        "PlotNodes gleichmäßig über die Kern-Fläche verteilt."
    }
    CORE_MASS = {
        "min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1,
        "description": "Trägheit der Plotkerne in der Feder-Masse-"
                        "Simulation - höher = trägere, langsamere Bewegung."
    }
    PLOT_NODE_MASS = {
        "min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1,
        "description": "Trägheit der PlotNodes (Voronoi-Kreuzungspunkte) in "
                        "der Feder-Masse-Simulation."
    }
    PLOT_NODE_REPULSION_STRENGTH = {
        "min": 0.0, "max": 20.0, "default": 4.0, "step": 0.5,
        "description": "Kurzreichweitige Abstoßung zwischen nahen "
                        "PlotNodes, unabhängig von direkter Netz-Nachbarschaft."
    }
    DAMPING = {
        "min": 0.0, "max": 1.0, "default": 0.80, "step": 0.05,
        "description": "Geschwindigkeits-Verlust pro Physik-Tick (0=sofort "
                        "still, 1=keine Dämpfung) - steuert, wie schnell die "
                        "Plot-Physik zur Ruhe kommt."
    }
    PLOT_GRAVITY_STRENGTH = {
        "min": 0.0, "max": 0.05, "default": 0.01, "step": 0.001,
        "description": "Rang-distanz-gewichtete Anziehung der Plotkerne zu "
                        "den Siedlungszentren."
    }
    PLOT_CITY_REPULSION_STRENGTH = {
        "min": 0.0, "max": 2.5, "default": 0.5, "step": 0.05,
        "description": "Gegenkraft, die Plotkerne von der Stadtmauer/"
                        "Kartenrand fernhält."
    }
    PLOT_TIER_FACTOR = {
        "min": 0.2, "max": 5.0, "default": 1.0, "step": 0.1,
        "description": "Skaliert die Verkehrs-Schwellen für Pfad/Weg/Straße "
                        "- höhere Werte brauchen mehr Verkehr, um als "
                        "Straße zu gelten."
    }
    POTENTIAL_STRENGTH = {
        "min": 0.0, "max": 5.0, "default": 1.0, "step": 0.1,
        "description": "Gesamtstärke des Potentialfelds, das alle Kräfte "
                        "auf Kerne/PlotNodes überlagert."
    }

class WEATHER:
    """Parameter für core/weather_generator.py"""
    AIR_TEMP_ENTRY = {
        "min": -30, "max": 40, "default": 0, "step": 1, "suffix": "°C",
        "description": "Zusätzlicher Temperatur-Offset AUF eine realistische, "
                        "aus Breitengrad und Jahreszeit berechnete Basis-"
                        "temperatur (0 = reine Klimatologie ohne Verschiebung) "
                        "- damit lässt sich die ganze Welt gleichmäßig wärmer "
                        "oder kälter stellen (z.B. für eine Eiszeit- oder "
                        "Treibhaus-Stimmung)."
    }
    GROUND_TEMP_OFFSET = {
        "min": -30, "max": 40, "default": 0, "step": 1, "suffix": "°C",
        "description": "Zusätzlicher Temperatur-Offset AUF eine realistische, "
                        "aus Breitengrad und Jahreszeit berechnete Boden-"
                        "Basistemperatur (0 = reine Klimatologie ohne "
                        "Verschiebung) - verschiebt Schatten- UND Sonnen-"
                        "temperatur des Bodens GEMEINSAM (die Differenz "
                        "zwischen beiden ist eine feste interne Konstante, "
                        "kein eigener Regler). Unabhängig vom Luft-Offset "
                        "(air_temp_entry) - Boden und Luft sind physikalisch "
                        "getrennte Größen, die erst über Konvektion "
                        "gekoppelt werden."
    }
    SUN_RELEVANCE_FACTOR = {
        "min": 0.0, "max": 10.0, "default": 1.0, "step": 0.1, "suffix": "x",
        "description": "Multiplikator auf die feste Sonne/Schatten-Boden-"
                        "temperatur-Spanne (GROUND_TEMP_SPREAD) - bei 0 hat "
                        "Sonnenexposition KEINEN Einfluss mehr auf die "
                        "Bodentemperatur (nur noch ground_temp_offset zählt), "
                        "bei 1 (Standard) die kalibrierte Spanne, bei 10 das "
                        "Zehnfache. Feiner Schritt (0.1) für Kontrolle im "
                        "unteren Bereich."
    }
    ALTITUDE_COOLING = {
        "min": 2, "max": 100, "default": 6, "step": 1, "suffix": "°C/km",
        "description": "Temperaturabfall pro Kilometer Höhe (realer "
                        "Richtwert: ca. 6°C/km) - höhere Berge werden "
                        "dadurch kälter."
    }
    THERMIC_EFFECT = {
        "min": 0.0, "max": 2.0, "default": 0.8, "step": 0.1,
        "description": "Stärke der thermischen Konvektion (aufsteigende "
                        "warme Luft) auf das Windfeld."
    }
    # Vorticity Confinement (Fedkiw/Stam-Standardtechnik, siehe
    # _apply_vorticity_confinement in core/weather_generator.py,
    # [[project-3layer-wind-cfd]]): injiziert lokale Rotationsenergie zurück,
    # die _apply_wind_diffusion pro Zeitschritt entfernt - ohne das war der
    # Wind über weite Flächen fast parallel (empirisch ~7.8° mittlere
    # Richtungsänderung zwischen Nachbarpixeln). 0.0 schaltet den Effekt
    # komplett ab (identisches Verhalten wie vor diesem Fix).
    TURBULENCE_STRENGTH = {
        "min": 0.0, "max": 2.0, "default": 0.5, "step": 0.05,
        "description": "Stärke lokaler Wind-Verwirbelungen (Turbulenz) - "
                        "0 lässt den Wind glatt/gleichmäßig strömen, "
                        "höhere Werte erzeugen sichtbar chaotischere, "
                        "lokal wechselnde Windrichtungen (besonders am "
                        "Kartenrand verstärkt)."
    }
    # Checkbox, kein Slider (siehe gui/tabs/weather_tab.py _create_wind_parameters()) -
    # Weather-Rework Punkt A: bei AN wird das synoptische Druckgefälle um einen
    # aus der AKTUELL simulierten Temperatur abgeleiteten Term ergänzt (wärmer
    # als der Schicht-Durchschnitt = lokal niedrigerer effektiver Druck, treibt
    # zusätzlichen thermischen Wind). Default AN, aber abschaltbar (Nutzer-
    # Vorgabe: "ich will aber auch wieder zurückgehen können, falls es
    # misslingt") - bei AUS läuft exakt der bisherige, rein synoptische Pfad.
    THERMAL_PRESSURE_COUPLING = {
        "default": True,
        "description": "Druckfeld zusätzlich aus der simulierten Temperatur "
                        "ableiten (wärmere Gebiete erzeugen lokal niedrigeren "
                        "Druck, was zusätzlichen Wind erzeugt) statt nur dem "
                        "vorgegebenen Grundgefälle zu folgen. Bei Bedarf "
                        "abschaltbar, um auf das alte, rein synoptische "
                        "Verhalten zurückzufallen."
    }
    WIND_SPEED_FACTOR = {
        "min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1,
        "description": "Genereller Multiplikator für die Windgeschwindigkeit "
                        "im gesamten Simulationsgebiet."
    }
    TERRAIN_FACTOR = {
        "min": 0.0, "max": 2.0, "default": 1.2, "step": 0.1,
        "description": "Wie stark das Gelände (Hänge, Erhebungen) den Wind "
                        "ablenkt und lokal beschleunigt."
    }
    # Vorherrschende Windrichtung (0°=Ost, 90°=Nord, math. Konvention) - ersetzt
    # das bisher hartcodierte West-Ost-Druckgefälle in der CFD-Windsimulation
    # (core/weather_generator.py _simulate_wind_field_cpu_cfd). Default 225°
    # (Südwest) entspricht den vorherrschenden Westwinden der gemäßigten Zone.
    # Lokale Abweichung entsteht weiterhin über die bestehende
    # Terrain-Ablenkung (terrain_factor), nicht über diesen Parameter.
    PREVAILING_WIND_DIRECTION = {
        "min": 0, "max": 360, "default": 0, "step": 5, "suffix": "°",
        "description": "Herkunftsrichtung des vorherrschenden Windes, z.B. "
                        "0°=Wind aus Westen, 90°=aus Süden (der Wind WEHT "
                        "dann Richtung Osten bzw. Norden) - rotiert leicht "
                        "mit den Jahreszeiten, lokale Abweichungen entstehen "
                        "zusätzlich durchs Gelände."
    }
    # Eintritts-Luftfeuchte (ersetzt die bisher hartcodierte 50%-Baseline in
    # _calculate_atmospheric_moisture_cpu()'s Verdunstungs-Ausgangswert).
    AIR_HUMIDITY_ENTRY = {
        "min": -50, "max": 50, "default": 0, "step": 1, "suffix": "%",
        "description": "Zusätzlicher Feuchte-Offset AUF eine realistische, "
                        "aus dem Breitengrad berechnete Basis-Luftfeuchte "
                        "(0 = reine Klimatologie ohne Verschiebung, feuchter "
                        "am Äquator als an den Polen)."
    }
    # Geografische Breite/Länge der Karte - treibt die echte astronomische
    # Sonnenstandsberechnung für saisonale Shadowmaps (siehe
    # core/terrain_generator.py ShadowCalculator, sun_angles_override).
    # Default 48°N/15°O entspricht einer Mitteleuropa-Referenz (passend zur
    # "gemäßigten Zone" der Klimazonen-Profile). longitude=15 setzt den
    # (15-long)/15-Zeitzonen-Term der Sonnenstandsformel auf 0.
    MAP_LATITUDE = {
        "min": -70, "max": 70, "default": 48, "step": 1, "suffix": "°N",
        "description": "Geografische Breite der Karte - bestimmt den "
                        "echten Sonnenstand (Winkel und Jahreszeiten-"
                        "Schwankung) für die Schattenberechnung."
    }
    # Nutzer-Entscheidung 2026-07-24: kein eigener Slider mehr (gui/tabs/
    # weather_tab.py) - der Effekt (reine Tageszeit-Feinverschiebung der 7
    # Sonnenwinkel-Samples, siehe calculate_solar_position()) ist real, aber
    # zu schwach, um einen eigenen Regler zu rechtfertigen (KEINE Wirkung
    # auf Klimatologie/Jahreszeiten/Temperatur-Basiswert - die hängen nur an
    # Latitude). Bleibt als feste interne Konstante bestehen - core/
    # weather_generator.py und gui/tabs/base_tab.py lesen den Parameter
    # bereits über `.get('map_longitude', MAP_LONGITUDE["default"])`, ein
    # fehlender Slider-Wert fällt also automatisch hierauf zurück.
    MAP_LONGITUDE = {
        "min": -180, "max": 180, "default": 15, "step": 1, "suffix": "°",
        "description": "Geografische Länge der Karte - beeinflusst die "
                        "Zeitzonen-Komponente der Sonnenstandsberechnung."
    }

    # Saisonale Offsets für die 6 Zwei-Monats-Perioden (Jan/Feb, Mär/Apr,
    # Mai/Jun, Jul/Aug, Sep/Okt, Nov/Dez), relativ zum jeweiligen User-
    # Slider-Wert (= "Jahres-Mittel" der gewählten Klimazone) statt absoluter
    # Ersatzwerte - der Slider bleibt wirksam, das Profil gibt nur die
    # saisonale FORM vor. Grob an mitteleuropäisches Klima angelehnt
    # (gemäßigte Zone), keine exakte meteorologische Quelle - empirisch
    # plausibel, wie schon bei RAIN_THRESHOLD/STREAM_THRESHOLD. Nur
    # "temperate" implementiert; weitere Zonen (tropical/arid/arctic/...)
    # folgen später über dieselbe Struktur, sobald ein
    # Klimazonen-Auswahlfeld im UI existiert.
    CLIMATE_ZONE_SEASONAL_OFFSETS = {
        "temperate": {
            "air_temp_entry":     [-9.0, -3.0, 3.0, 6.0, 0.0, -7.0],   # °C
            "wind_speed_factor":  [0.15, 0.05, -0.05, -0.10, 0.0, 0.10],
            # % - Summe Peak-zu-Tal bleibt < 30% des 0-100%-Sliderbereichs
            "air_humidity_entry": [15.0, 5.0, -5.0, 0.0, 8.0, 18.0],
        },
    }
    CLIMATE_ZONE = "temperate"  # vorerst fix, siehe Backlog für spätere Auswahl


class EROSION:
    """
    Parameter für core/erosion_generator.py (Feld-Erosion, eigener Generator
    seit 2026-07-28 - siehe dortigen Modul-Docstring).

    Die drei Charakter-Regler Kc/Ks/Kd stammen aus dem Vorbild
    (LanLou123/Webgl-Erosion); alles andere ergänzt, was ein
    einmal-durchlaufendes Verfahren gegenüber einer endlos interaktiven
    Anwendung zusätzlich braucht (Abbruch, Auflösung, Varianten).

    KALIBRIERUNG 2026-07-28 gegen das Zielbild des Nutzers (dichte verästelte
    Entwässerung, scharfe Kämme, helle Talböden). Werkzeug:
    scratch_erosion_lab.py - es rechnet Varianten auf der GPU, misst vier
    Formkennzahlen und schreibt einen Kontaktabzug, der nebeneinander
    vergleichbar ist.

    Fünf Sweeps, gemessen bei 512 px auf demselben Gelände (Seed 424242,
    Relief 2611 m). Die entscheidende Zeile ist die letzte:

        Variante                       Aniso  Drainage    beta   hypso
        (unerodiert)                   0.053     0.128   -0.148  0.549
        alte Defaults                  0.580     0.430   -0.270  0.433
        ohne Böschung                  0.224     0.265   -0.283  0.400
        ohne Glättung                  0.598     0.432   -0.283  0.434
        OHNE BEIDES                    0.454     0.177   -0.419  0.702

    `beta` ist der Exponent der Hangneigung-über-Einzugsgebiet-Beziehung, in
    echten Landschaften rund -0.5, bei reinem Rauschen rund 0. Er ist die
    einzige der vier Kennzahlen, die die Siegervariante erkannt hat - die
    Drainage-Dichte hat sie sogar als schlechteste eingestuft (0.177). Der
    Kontaktabzug war eindeutig: nur ohne Böschung UND ohne Glättung entsteht
    das feine verzweigte Netz über die ganze Karte.

    WARUM die beiden Passes hier schaden, obwohl das Vorbild sie hat: sie
    arbeiten auf einer festen Längenskala. Die Böschungsschwelle ist
    `Zellbreite * tan(Winkel)` = 20 m * tan(30°) ~ 11 m; eine frisch
    eingeschnittene Rinne ist tiefer und rutscht deshalb im nächsten Schritt
    wieder zu. Gegenprobe mit doppeltem Winkel (Talus Angle Scale 2.0): auch
    bei Stärke 0.05 bleibt beta bei -0.298 statt -0.419, die Rinnen
    verschwinden trotzdem. Das Vorbild rechnet auf einem Einheitsgitter ohne
    Meter, dort stellt sich dieses Verhältnis nicht ein.

    Beide Regler bleiben vollständig erhalten und wirksam - nur ihr Startwert
    ist jetzt 0. Wer weichere, gerundete Formen will, dreht sie auf.
    """

    EROSION_CAPACITY = {
        "min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1,
        "description": "Wie viel Material fließendes Wasser überhaupt tragen "
                        "kann (Kc). Zusammen mit Erosion Strength und "
                        "Deposition Rate der Charakter-Regler: hoch = viel "
                        "Material in Bewegung, tiefe Täler und ausgedehnte "
                        "Schwemmebenen."
    }
    EROSION_STRENGTH = {
        "min": 0.0, "max": 2.0, "default": 0.5, "step": 0.05,
        "description": "Wie schnell untersättigtes Wasser Material löst (Ks). "
                        "Der Hauptregler für die Tiefe der Täler - 0 schaltet "
                        "die fluviale Erosion vollständig ab."
    }
    DEPOSITION_RATE = {
        "min": 0.0, "max": 2.0, "default": 0.05, "step": 0.05,
        "description": "Wie schnell übersättigtes Wasser seine Fracht wieder "
                        "abgibt (Kd). Hohe Werte erzeugen ausgeprägte Ebenen "
                        "und Schwemmfächer dort, wo das Wasser langsamer wird. "
                        "NIEDRIG kalibriert (0.05 statt 0.5): die Fracht bleibt "
                        "dann lange in Schwebe und verlässt die Karte, statt "
                        "das Nachbartal zuzuschütten. Mit 0.5 flachte der Lauf "
                        "das Gelände ein (hypsometrisch 0.249 -> 0.485), statt "
                        "es zu zertalen."
    }
    RAINFALL = {
        "min": 0.0, "max": 5.0, "default": 5.0, "step": 0.1,
        "description": "Gleichmäßiger Regen über die gesamte Karte. Bewusst "
                        "unabhängig vom Weather-Niederschlag: die Erosion "
                        "modelliert geologische Zeit, nicht das heutige "
                        "Wetter - deshalb läuft sie auch VOR Weather."
    }
    EVAPORATION_RATE = {
        "min": 0.0, "max": 0.1, "default": 0.015, "step": 0.001,
        "description": "Gegenspieler des Regens. Bestimmt, wie weit Wasser "
                        "läuft, bevor es versickert - und damit, wie weit die "
                        "Erosion in flache Bereiche hineinreicht."
    }
    CONVERGENCE_THRESHOLD = {
        "min": 1e-8, "max": 1e-5, "default": 1e-6, "step": 1e-7,
        "description": "Abbruchkriterium: mittlere Höhenänderung pro Schritt, "
                        "relativ zum Relief der Karte. Kleiner = länger = "
                        "ausgereiftere Landschaft. Ein Modell mit ständigem "
                        "Regen und ohne Hebung kommt nie ganz zum Stillstand - "
                        "die Schwelle sagt 'es lohnt nicht mehr', nicht 'fertig'."
    }
    MAX_STEPS = {
        "min": 200, "max": 20000, "default": 8000, "step": 100,
        "description": "Obergrenze der Simulationsschritte. Greift, wenn das "
                        "Konvergenzkriterium vorher nicht erreicht wird - "
                        "verhindert einen Lauf ohne absehbares Ende."
    }
    SIMULATION_RESOLUTION = {
        "min": 128, "max": 1024, "default": 512, "step": 128, "suffix": "px",
        "description": "Auflösung der Simulation, unabhängig von Map Size - "
                        "das Ergebnis wird anschließend auf die Kartengröße "
                        "skaliert. Hält Laufzeit und Optik über alle "
                        "Kartengrößen vergleichbar. Ohne GPU wird auf 256 "
                        "begrenzt (siehe Statistik-Anzeige)."
    }
    THERMAL_STRENGTH = {
        "min": 0.0, "max": 2.0, "default": 0.0, "step": 0.05,
        "description": "Stärke der Böschungswinkel-Erosion: wie schnell zu "
                        "steile Hänge nachrutschen. Startwert 0 - siehe "
                        "Klassen-Docstring: die Schwelle liegt bei rund 11 m "
                        "pro Zelle und schüttet frisch eingeschnittene Rinnen "
                        "sofort wieder zu. Aufdrehen ergibt weichere, "
                        "gerundete Hänge auf Kosten der Verästelung."
    }
    TALUS_ANGLE_SCALE = {
        "min": 0.5, "max": 2.0, "default": 1.0, "step": 0.05,
        "description": "Skaliert den kritischen Böschungswinkel. Klein = "
                        "flachere stabile Hänge (alles rutscht ab), groß = "
                        "steile Wände bleiben stehen."
    }
    HARDNESS_INFLUENCE = {
        "min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05,
        "description": "Wie stark die Gesteinshärte aus Geology eingeht - auf "
                        "die Löserate UND den Böschungswinkel. 0 % ist exakt "
                        "das Verhalten des Vorbilds (das kein Gestein kennt), "
                        "100 % volle Kopplung: hartes Gestein löst langsamer "
                        "und hält steilere Wände."
    }
    SMOOTHING = {
        "min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05,
        "description": "Glättet gezielt Ein-Pixel-Grate und -Rinnen, also "
                        "Gitterartefakte - echte Hänge bleiben unangetastet. "
                        "Startwert 0: gemessen kostet der Pass mehr echte "
                        "Rinnen als er Artefakte entfernt (beta -0.419 -> "
                        "-0.243 schon bei 0.1). Aufdrehen, wenn einzelne "
                        "Pixelgrate stören."
    }
    THERMAL_VARIANT = {
        "min": 0, "max": 1, "default": 0, "step": 1,
        "description": "Verfahren der Böschungserosion: 0 = Gather "
                        "(massenexakt, das in diesem Projekt gewachsene "
                        "Verfahren), 1 = Flux (die Variante des Vorbilds). "
                        "Zum direkten optischen Vergleich."
    }


class WATER:
    """Parameter für core/water_generator.py"""
    # total_volume in _classify_lake_basins (core/water_generator.py) ist in
    # "Meter-Pixel" (sum(spill_height - terrain_height) über alle überfluteten
    # Pixel eines Beckens), NICHT m³ und NICHT einfach Meter Wassertiefe - ein
    # realer Smoke-Log-Lauf (smoke_test_water_pipeline.log, LOD3/128x128,
    # Default-Terrain) zeigte bei altem default=0.1 GAR KEINE See-Pixel
    # (water_biomes_map max=3.0, Klasse 4=Lake nie erreicht) - auf typisch
    # zerklüftetem, ridged Terrain sind die meisten Becken flach/klein genug,
    # dass ihr total_volume deutlich unter 0.1 bleibt. min/default gesenkt und
    # max verkleinert, damit der Slider-Weg überwiegend im tatsächlich
    # wirksamen Bereich liegt (vorher lagen ~90% des Sliders oberhalb jedes
    # real vorkommenden Becken-Volumens - das war zugleich die Ursache für
    # "Slider fühlt sich nicht reaktiv an", siehe [[project-water-flood-calibration]]).
    # Neukalibrierung 2026-07-27: `total_volume` in _classify_lake_basins
    # (core/water_generator.py) wird jetzt mit der realen Zellfläche
    # multipliziert und ist damit ein ECHTES Volumen in m³. Vorher war es die
    # blosse Summe der Wassertiefen über alle überfluteten Pixel
    # ("Meter-Pixel") - dieselbe Geländeform ergab damit bei map_size 512
    # rund viermal so viele Seen wie bei 128, und map_distance_km ging gar
    # nicht ein. Der Slider war dadurch auflösungsabhängig und musste nach
    # jeder Map-Size-Änderung neu gefunden werden.
    # Bereich/Default: bei den Default-Einstellungen (128 px auf 10 km,
    # also 78 m/px = 6100 m² pro Zelle) entspricht der Default von 5000 m³
    # etwa einer Senke von 1 m mittlerer Tiefe auf knapp einer Zelle - klein
    # genug für viele kleine Bergseen, gross genug, um einzelne
    # Rausch-Vertiefungen auszusortieren. Das Maximum (5 Mio. m³) entspricht
    # einem grossen Talsee.
    LAKE_VOLUME_THRESHOLD = {
        "min": 100.0, "max": 5_000_000.0, "default": 5000.0, "step": 100.0, "suffix": "m³",
        "description": "Mindest-Wasservolumen eines Geländebeckens, damit "
                        "dort ein See entsteht - niedrigere Werte lassen auch "
                        "kleine/flache Senken zu Seen werden. Echtes Volumen "
                        "in Kubikmetern, also unabhängig von Map Size und Map "
                        "Distance: derselbe Wert erzeugt bei jeder Auflösung "
                        "dieselben Seen."
    }
    # RAIN_THRESHOLD (Mindest-Niederschlag, damit ein Pixel als Wasserquelle
    # zaehlt) ist 2026-07-27 vollstaendig entfernt: der einzige verbliebene
    # Leser war der Simple-Fallback von FlowNetworkBuilder.build_flow_network(),
    # der selbst geloescht wurde (er setzte die Wassertiefe mit der
    # Niederschlagsmenge gleich und lieferte ein Fluss-Netzwerk ohne Fluesse -
    # ein Ergebnis, das nur so aussah, als waere die Simulation gelungen).
    # Im Pipe-Modell speist JEDE Zelle ihren Niederschlag ins System ein; ob
    # daraus ein sichtbarer Wasserlauf wird, entscheidet allein die
    # Durchfluss-Klassifikation (RIVER_ABUNDANCE unten). Ein vorgeschalteter
    # Mengenfilter hat dort keine physikalische Entsprechung mehr.
    # STREAM_THRESHOLD (interner Performance-Schwellwert fuer Mannings
    # frühere Kanalgeometrie-Suche) ist seit dem D8 -> Pipe-Modell-Umbau
    # 2026-07-25 vollständig entfernt - der einzige Zweck war das
    # Überspringen der teuren Tal-Breite-Suche für schwache Zellen
    # (_optimize_channel_geometry), die selbst gelöscht wurde (siehe
    # ManningFlowCalculator-Docstring in core/water_generator.py). Die
    # Fluss-Klassifikation läuft seit langem ohnehin perzentil-basiert
    # (siehe RIVER_ABUNDANCE unten), kein Ersatzwert nötig.
    # Ersetzt das ehemalige STREAM_THRESHOLD als primären Fluss-Dichte-Regler (Nutzer-Report:
    # "extrem viele Flüsse... nicht jeden Wasserzulauf"). Statt eines festen
    # gH2O/m²-Werts wird der tatsächliche Schwellwert live als PERZENTIL der
    # flow_accumulation-Verteilung DIESER Karte berechnet (siehe
    # water_generator.py's _river_flow_percentile_threshold()) - dadurch bleibt
    # die Fluss-DICHTE (Anteil der wasserführenden Pixel, der als Fluss zählt)
    # unabhängig von Kartengröße/Seed/Niederschlagsmenge konstant, statt bei
    # jeder Karte neu kalibriert werden zu müssen. 0.0 = nur die stärksten
    # ~0.5% der wasserführenden Pixel gelten als Fluss (sehr restriktiv), 1.0 =
    # praktisch jedes wasserführende Pixel gilt als Fluss (sehr freizügig).
    # Default 0.10 = nur die oberen ~10% gelten als Fluss (Nutzer-Vorgabe:
    # "lieber zu wenige Flüsse als zu viele").
    RIVER_ABUNDANCE = {
        "min": 0.0, "max": 1.0, "default": 0.10, "step": 0.01,
        "description": "Steuert, wie groß der Anteil der wasserführenden "
                        "Pixel ist, der als Fluss gilt - niedrig = nur die "
                        "stärksten Wasserläufe werden zu Flüssen, hoch = "
                        "praktisch jeder Wasserzulauf gilt als Fluss."
    }
    # MANNING_COEFFICIENT (Rauheits-Koeffizient) ist 2026-07-27 vollstaendig
    # entfernt: seit dem D8 -> Pipe-Modell-Umbau liefert PipeFlowSimulator
    # Fliessgeschwindigkeit und Wassertiefe als echte simulierte Groessen,
    # ManningFlowCalculator loest die Manning-Gleichung nicht mehr, und
    # `manning_n` wurde von keiner Methode mehr gelesen. Der Slider stand
    # trotzdem im Water-Tab und wurde sogar prominent im Statistics-Widget
    # angezeigt - ein Regler ohne jede Wirkung. Eine Rauheits-Modellierung im
    # Pipe-Modell muesste an der Rohr-Querschnittsflaeche
    # (PipeFlowSimulator.PIPE_CROSS_SECTION_AREA) ansetzen, nicht an einem
    # Manning-n; bis dahin gibt es hier bewusst keinen Ersatz-Slider.
    # Abwechselnd Senken fuellen und erodieren (siehe core/water_generator.py
    # DropletErosionSystem.simulate_erosion_sedimentation). Mehr Durchgaenge
    # bedeuten NICHT mehr Erosion - die Partikelzahl wird gleichmaessig
    # aufgeteilt -, sondern oefter wiederhergestellte Entwaesserung zwischen
    # den Durchgaengen. Gemessen bei 128²/40k Partikeln, groesste
    # zusammenhaengende Komponente des Kanalnetzes: 1 Durchgang -> 107 px,
    # 2 -> 175 px, 4 -> 209 px, 6 -> 161 px. 4 ist das Optimum.
    EROSION_PASSES = {
        "min": 1, "max": 8, "default": 4, "step": 1,
        "description": "Wie oft abwechselnd Senken aufgefüllt und erodiert "
                        "wird. Mehr Durchgänge lassen zusammenhängendere "
                        "Bach- und Flussläufe entstehen, weil zwischen den "
                        "Durchgängen jedes Mal ein durchgehender Abfluss "
                        "hergestellt wird - die Gesamtmenge an Erosion bleibt "
                        "dabei gleich."
    }
    EROSION_STRENGTH = {
        "min": 0.1, "max": 5.0, "default": 2.5, "step": 0.1,
        "description": "Genereller Multiplikator dafür, wie stark ein "
                        "Erosions-Partikel das Gelände abträgt (siehe "
                        "core/water_generator.py DropletErosionSystem, "
                        "Droplet-basierte Erosion seit 2026-07-25 - ersetzt "
                        "die frühere Stream-Power-Formel)."
    }
    # Droplet-Erosion-Umbau 2026-07-25 (siehe core/water_generator.py
    # DropletErosionSystem, ersetzt die vorherige Stream-Power+MacCormack-
    # Sedimenttransport-Formel "transport_capacity = sediment_capacity_factor
    # * flow_speed^2.5"): sedimentCapacity = max(-deltaHeight * speed * water
    # * capacityFactor, minCapacity) - eine STRUKTURELL andere Formel, der
    # alte Bereich (0.0-0.02, auf die ^2.5-Sättigung von flow_speed
    # zugeschnitten) ist dafür bedeutungslos. Neuer Bereich/Default folgt dem
    # Referenzprojekt-Default (Sebastian Lague, github.com/SebLague/
    # Hydraulic-Erosion) als Startpunkt - noch NICHT gegen die laufende App
    # kalibriert.
    SEDIMENT_CAPACITY_FACTOR = {
        "min": 0.5, "max": 12.0, "default": 4.0, "step": 0.1,
        "description": "Wie viel Sediment ein Erosions-Partikel bei "
                        "gegebener Fließgeschwindigkeit/Wassermenge maximal "
                        "transportieren kann, bevor er es ablagert - "
                        "niedrigere Werte lassen Partikel schneller/mehr "
                        "Sediment absetzen (mehr sichtbare Sedimentation, "
                        "besonders an flacheren Streckenabschnitten)."
    }
    EVAPORATION_BASE_RATE = {
        "min": 0.0001, "max": 0.01, "default": 0.002, "step": 0.0001, "suffix": "m/Tag",
        "description": "Grundrate, mit der Wasser von Gewässer-Oberflächen "
                        "verdunstet."
    }
    # Steuert nur die "Grundwasser"-Komponente von SoilMoistureCalculator
    # (core/water_generator.py) - die "kapillare" Komponente hat einen
    # separaten, festen Sigma-Wert (siehe SoilMoistureCalculator.__init__,
    # dort auch die volle Begründung für die Absenkung von 5.0 auf 2.0:
    # dichteres Fluss-Netzwerk nach der precip_map-Neukalibrierung ließ Boden-
    # feuchte bei beiden Werten unverändert bei ~64% Mittelwert sättigen -
    # "Soil Moisture überall 100%"-Report).
    DIFFUSION_RADIUS = {
        "min": 1.0, "max": 20.0, "default": 2.0, "step": 0.5, "suffix": "Pixel",
        "description": "Wie weit sich Bodenfeuchtigkeit vom Grundwasser um "
                        "Flüsse/Seen herum ausbreitet - größerer Radius "
                        "lässt den feuchten Streifen entlang des Wassers "
                        "breiter werden."
    }
    # Droplet-Erosion-Umbau 2026-07-25 (siehe core/water_generator.py
    # DropletErosionSystem, ersetzt ErosionSedimentationSystem._transport_sediment_maccormack):
    # settling_velocity ist jetzt DROPLET_DEPOSIT_SPEED - pro Lebenszeit-Schritt
    # eines Partikels der Anteil des über-Kapazität-Sediments, der bei diesem
    # Schritt tatsächlich abgesetzt wird (Rest bleibt im Partikel und wandert
    # weiter mit). Bereich/Default folgen dem Referenzprojekt-Default
    # (Sebastian Lague, github.com/SebLause/Hydraulic-Erosion: depositSpeed=0.3)
    # als Startpunkt - noch NICHT gegen die laufende App kalibriert.
    SETTLING_VELOCITY = {
        "min": 0.01, "max": 1.0, "default": 0.3, "step": 0.01, "suffix": "m/s",
        "description": "Anteil des überschüssigen (nicht mehr "
                        "transportierbaren) Sediments, der pro "
                        "Partikel-Schritt tatsächlich zu Boden sinkt - "
                        "höhere Werte lassen sichtbar mehr Sedimentation "
                        "entstehen."
    }
    # Böschungswinkel-Erosion ("Phase 6", core/water_generator.py
    # ThermalErosionSystem) - Nutzer-Vorgabe 2026-07-25: härteres Gestein
    # widersteht seitlichem Abrutschen besser (steile V-Wände bleiben
    # erhalten), weicheres Material kollabiert zu Schutthalden/U-Form. Nur
    # die STÄRKE ist per Slider einstellbar - der Böschungswinkel-Bereich
    # selbst (15°-60° je nach Härte, siehe ThermalErosionSystem.
    # REPOSE_ANGLE_MIN_DEG/MAX_DEG) ist ein fester interner Wert, um die
    # UI-Fläche klein zu halten. Startwerte, noch NICHT gegen die laufende
    # App kalibriert.
    THERMAL_EROSION_STRENGTH = {
        "min": 0.0, "max": 3.0, "default": 1.0, "step": 0.1,
        "description": "Multiplikator dafür, wie stark Material lateral "
                        "abrutscht, sobald die Hangneigung den (härte-"
                        "abhängigen) Böschungswinkel überschreitet - 0 "
                        "deaktiviert den Effekt, höhere Werte lassen "
                        "eingeschnittene Täler schneller zu breiteren, "
                        "flacheren Formen kollabieren."
    }


class BIOME:
    """Parameter für core/biome_generator.py"""
    BIOME_WETNESS_FACTOR = {
        "min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1,
        "description": "Multiplikator dafür, wie stark Bodenfeuchtigkeit "
                        "die Biom-Klassifikation beeinflusst (z.B. Wüste "
                        "vs. Feuchtgebiet)."
    }
    BIOME_TEMP_FACTOR = {
        "min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1,
        "description": "Multiplikator dafür, wie stark die Temperatur die "
                        "Biom-Klassifikation beeinflusst."
    }
    SEA_LEVEL = {
        "min": 0, "max": 200, "default": 10, "step": 5, "suffix": "m",
        "description": "Höhe des Meeresspiegels - alles darunter wird als "
                        "Wasser/Küste klassifiziert."
    }
    BANK_WIDTH = {
        "min": 1, "max": 20, "default": 3, "step": 1, "suffix": "Pixel",
        "description": "Breite des Uferstreifens um Gewässer, der als "
                        "eigene Übergangszone (z.B. Strand-/Ufer-Biom) "
                        "behandelt wird."
    }
    EDGE_SOFTNESS = {
        "min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1,
        "description": "Wie weich/verlaufend die Übergänge zwischen "
                        "benachbarten Biomen gezeichnet werden, statt "
                        "scharfer Grenzen."
    }
    ALPINE_LEVEL = {
        "min": 500, "max": 3000, "default": 1500, "step": 50, "suffix": "m",
        "description": "Höhe, ab der alpine (baumfreie Gebirgs-)Biome "
                        "statt Wald/Wiese beginnen."
    }
    SNOW_LEVEL = {
        "min": 800, "max": 4000, "default": 2000, "step": 50, "suffix": "m",
        "description": "Höhe, ab der dauerhaft schneebedeckte Biome "
                        "beginnen (muss über dem Alpine Level liegen)."
    }
    CLIFF_SLOPE = {
        "min": 30, "max": 80, "default": 60, "step": 1, "suffix": "°",
        "description": "Mindest-Hangneigung, ab der ein Bereich als "
                        "Klippe/Fels statt als normales Gelände "
                        "klassifiziert wird."
    }


# Validation Rules für Parameter-Abhängigkeiten
class VALIDATION_RULES:
    """
    Funktionsweise: Definiert Parameter-Abhängigkeiten und Validation-Rules
    - Cross-Parameter Validation (z.B. Snow_Level > Alpine_Level)
    - Generator-Dependencies (welche Inputs werden benötigt)
    - Warning-Thresholds für Performance-kritische Parameter
    """

    # Terrain Parameter Validation
    TERRAIN_CONSTRAINTS = {
        "octaves_frequency": "octaves * frequency < 1.0",  # Verhindert zu hochfrequente Noise
        "redistribute_extreme": "redistribute_power != 1.0 or amplitude < 150"  # Warning bei extremen Werten
    }

    # Biome Parameter Validation
    BIOME_CONSTRAINTS = {
        "elevation_order": "alpine_level < snow_level",  # Alpine Zone muss unter Schneegrenze sein
        "sea_level_reasonable": "sea_level <= amplitude * 0.3"  # Meeresspiegel nicht zu hoch
    }

    # Performance Warnings
    PERFORMANCE_WARNINGS = {
        "large_map": "size >= 1024",  # Warnung bei großen Karten
        "high_detail": "octaves >= 10",  # Warnung bei sehr detaillierten Terrains
        "many_settlements": "settlements + landmarks + roadsites > 15"  # Warnung bei vielen Objekten
    }

    # Generator Dependencies
    DEPENDENCIES = {
        "geology": ["heightmap", "slopemap"],
        "settlement": ["heightmap", "slopemap", "water_map"],
        "weather": ["heightmap", "shademap", "soil_moist_map"],
        # Erosion braucht nur Gelaende und Haerte - bewusst KEIN Weather
        # (siehe core/erosion_generator.py: die Erosion laeuft vor Weather).
        "erosion": ["heightmap", "hardness_map"],
        "water": ["heightmap", "hardness_map", "precip_map", "temp_map", "wind_map",
                  "humid_map"],
        "biome": ["heightmap", "slopemap", "temp_map", "soil_moist_map", "water_biomes_map"]
    }


# Utility Functions für Parameter-Handling
def get_parameter_config(generator_type, parameter_name):
    """
    Funktionsweise: Holt Parameter-Konfiguration für spezifischen Generator und Parameter
    Aufgabe: Zentrale Zugriffsfunktion für alle GUI-Komponenten
    Parameter: generator_type (str), parameter_name (str)
    Return: dict mit min/max/default/step/suffix/description
    """
    generator_classes = {
        "terrain": TERRAIN,
        "geology": GEOLOGY,
        "settlement": SETTLEMENT,
        "weather": WEATHER,
        "erosion": EROSION,
        "water": WATER,
        "biome": BIOME
    }

    if generator_type not in generator_classes:
        raise ValueError(f"Unknown generator type: {generator_type}")

    generator_class = generator_classes[generator_type]

    if not hasattr(generator_class, parameter_name.upper()):
        raise ValueError(f"Unknown parameter {parameter_name} for {generator_type}")

    return getattr(generator_class, parameter_name.upper())


def validate_parameter_set(generator_type, parameters):
    """
    Funktionsweise: Validiert kompletten Parameter-Satz für einen Generator
    Aufgabe: Prüft Cross-Parameter Constraints und Dependencies
    Parameter: generator_type (str), parameters (dict)
    Return: (is_valid: bool, warnings: list, errors: list)
    """
    warnings = []
    errors = []

    # Implementation würde hier Parameter-spezifische Validation durchführen
    # Beispiel für Terrain:
    if generator_type == "terrain":
        if parameters.get("octaves", 1) * parameters.get("frequency", 0.01) >= 1.0:
            warnings.append("Hohe Octaves * Frequency kann zu Noise-Artefakten führen")

    if generator_type == "biome":
        alpine = parameters.get("alpine_level", 1500)
        snow = parameters.get("snow_level", 2000)
        if alpine >= snow:
            errors.append("Alpine Level muss unter Snow Level liegen")

    return len(errors) == 0, warnings, errors
