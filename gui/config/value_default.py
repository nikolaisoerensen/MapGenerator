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
    # Höhe der Talsohle in Metern. Zusammen mit AMPLITUDE legt sie die
    # Höhenspanne JEDER Karte fest: der tiefste Punkt liegt exakt hier, der
    # höchste exakt bei AMPLITUDE (siehe
    # BaseTerrainGenerator._apply_redistribution).
    #
    # 2026-07-30 von 100.0 auf 0.0 gesetzt (Entscheidung des Nutzers). Der
    # frühere Wert sollte verhindern, dass die Talsohle mit dem Meeresspiegel
    # zusammenfällt, hatte aber einen schwereren Nebeneffekt: AMPLITUDE ist die
    # GIPFELHÖHE und der Slider erlaubt ab 30 m. Für jedes amplitude < 100 wurde
    # `(amplitude - base_elevation)` negativ und die Landschaft KIPPTE UM -
    # Gipfel wurden Täler. Gemessen bei amplitude 30: Korrelation -0.998 gegen
    # denselben Seed bei 4000 (mit abgeschaltetem Erosionsfilter; eingeschaltet
    # hob dessen zweite Spannen-Abbildung die Inversion zufällig wieder auf und
    # verdeckte den Fehler).
    #
    # Mit 0.0 ist das ausgeschlossen, weil AMPLITUDE bei 30 beginnt und damit
    # immer über der Talsohle liegt. §1 und 02_INVARIANTEN.md 7: kein
    # Reglerstand darf ein unbrauchbares Ergebnis erzeugen.
    #
    # Der Meeresspiegel liegt entsprechend ebenfalls bei 0 m (siehe
    # BIOME.SEA_LEVEL) - die Karte bleibt damit landseitig, ohne dass die
    # Talsohle künstlich angehoben werden muss.
    BASE_ELEVATION_M = 0.0

    # 2026-07-28 von 10.0 auf 15.0 angehoben, zusammen mit der verdoppelten
    # FREQUENCY (siehe dort): die Karte zeigt jetzt die vierfache Fläche, und
    # 15 km trifft die dargestellte Landschaft besser als 10 km.
    #
    # Zur Einordnung: streng "die Welt setzt sich fort" wären 20 km. Mit 15 km
    # ist jede Geländeform real rund ein Viertel kleiner als vorher - eine
    # bewusste Entscheidung nach Augenschein, kein Rechenergebnis.
    # 2026-08-05 von 15.0 auf 21.3: das ist die Kantenlaenge der Regionenwelt
    # (core/terrain_weltkarte.WELT_KM). Bei aktivem WELTKARTE_AKTIV setzt der
    # Generator diesen Wert ohnehin im DataLODManager - stand der Regler
    # daneben, zeigte die Oberflaeche 15 km an, waehrend alles mit 21.3
    # gerechnet wurde.
    WORLD_SIZE_KM = 21.3

    MAPSIZE = {
        # Vorgabe 2026-07-30 von 128 auf 256 angehoben: unter 30 Pixeln je Tal
        # bricht die Entwaesserung des Flussnetzes ein (gemessen 17 % bei
        # 128 px gegen 87 % bei 256 px, SPEZIFIKATION §13). Bei 2500 m
        # Talabstand und 15 km Karte sind 256 px die untere brauchbare Grenze.
        # 2026-08-06 von 256 auf 1024. Gemessen an der 21-km-Weltkarte:
        #
        #   px    m/px   Gesamtzeit   groesster Nachbarsprung (p99.9)
        #   256   83.2      4.7 s     190 m
        #   512   41.6     20   s     142 m
        #   1024  20.8     31   s      37 m
        #   2048  10.4     76   s      21 m
        #
        # Bei 256 px ist ein Tal zwei bis drei Pixel breit - das Querprofil
        # wird nicht mehr aufgeloest, und das schlaegt als Stufe von 190 m je
        # Pixel durch. 1024 loest die Taeler sauber auf; 2048 kostet das
        # Doppelte und bringt nur noch 37 auf 21 m.
        "min": MAPSIZEMIN, "max": MAPSIZEMAX, "default": 1024, "step": 32,
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
        # SCHRITT 1.0 -> 0.1 (2026-08-26). WORLD_SIZE_KM ist 21.3 und lag
        # damit NICHT auf dem Raster: der Regler rastete beim Aufbau des
        # Reiters auf 21 km. Das ist die WELTBREITE, gegen die jede
        # Meterrechnung des Programms geeicht ist - allein das Oeffnen des
        # Terrain-Reiters haette sie um 1.4 % verstellt, ohne Meldung.
        "min": 1.0, "max": 100.0, "default": WORLD_SIZE_KM, "step": 0.1, "suffix": "km",
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
        "min": 30, "max": 6000.0, "default": 1800.0, "step": 10, "suffix": "m",
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
    # Default 2026-07-30 von 4 auf 2 GESENKT, zusammen mit der Einführung des
    # ATEF-Erosionsfilters (siehe EROSION_FILTER und SPEZIFIKATION §9). Der
    # Filter erwartet einen GLATTEN Untergrund und liefert das Detail selbst;
    # gemessen: mit 5 Oktaven Untergrund ist sein Ergebnis feinkörniges
    # Gekrissel, mit 1 Oktave ein zusammenhängendes verästeltes Netz. Ein
    # detailreicher Untergrund macht den Filter also nicht besser, sondern
    # wirkungslos. 02_INVARIANTEN.md 7: abhängige Defaults ziehen mit.
    #
    # Nebenbefund derselben Messung, unabhängig vom Filter: der glatte
    # Untergrund hat auch für sich die besseren Kennzahlen (9 statt 187
    # Senken, größtes Kanalnetz 1801 statt 157 px).
    OCTAVES = {
        "min": 1, "max": 8, "default": 3, "step": 1,
        "description": "Anzahl der übereinandergelegten Rausch-Schichten "
                        "unterschiedlicher Frequenz. Mehr Oktaven fügen "
                        "feinere Detailebenen hinzu, verlangsamen aber die "
                        "Berechnung. Bei eingeschaltetem Erosionsfilter "
                        "gehören hier KLEINE Werte hin (1-2): der Filter "
                        "erzeugt das Detail, und auf einem bereits "
                        "detailreichen Untergrund wirkt er nicht mehr. "
                        "Oktaven jenseits der aktuellen Frequency/"
                        "Frequency-Scaling-Kombination werden automatisch "
                        "ignoriert, sobald ihre Frequenz über 0.5 "
                        "Zyklen/Pixel liegt."
    }
    # Groesse der Grundformen des Gelaendes in METERN - Hügel, Rücken, Becken,
    # bevor der Erosionsfilter seine Rinnen hineinlegt.
    #
    # 2026-07-30 eingefuehrt, weil FREQUENCY denselben Fehler hatte wie die
    # Rinnengroesse: die Formel `frequency * (64 / size)` in _calc_noise haengt
    # nur an der PIXELZAHL, nicht an der realen Ausdehnung. Die Karte zeigte
    # damit bei jedem map_distance_km dieselben 4.74 Zyklen - bei 5 km also
    # 1064 m grosse Formen, bei 50 km 10638 m grosse. Zoomte man heraus, wurde
    # die Landschaft mitvergroessert statt weiter zu gehen.
    #
    # Gemessen in smoke_test_terrain_scale_coupling.py: die Wellenzahl des
    # Grundrauschens (4.7) zog die gemessene Rinnengroesse bei 5 km und 50 km
    # jeweils zu sich hin - der Fehler war an beiden Stellen derselbe.
    #
    # Vorgabe 3150 m entspricht der bisherigen FREQUENCY von 0.074 bei den
    # vorgegebenen 15 km (15000 / (0.074 * 64) = 3167 m) auf 0.5 % genau.
    FEATURE_SIZE_M = {
        "min": 200.0, "max": 30000.0, "default": 3150.0, "step": 50.0,
        "suffix": "m",
        "description": "Groesse der Grundformen des Gelaendes in Metern - "
                        "Huegel, Ruecken und Becken, in die der Erosionsfilter "
                        "danach seine Rinnen legt. Unabhaengig von Aufloesung "
                        "und Kartenausschnitt: ein groesserer Ausschnitt zeigt "
                        "MEHR Formen, nicht groessere."
    }
    # Default 2026-07-28 von 0.037 auf 0.074 VERDOPPELT: die Karte zeigt jetzt
    # den doppelten Weltausschnitt in x UND y, also die vierfache Fläche.
    #
    # Warum genau das Verdoppeln der Frequenz das leistet: die Rausch-
    # Koordinaten laufen über `pixel_index * frequency * 64 / size`
    # (terrain_generator.py, _calc_noise_generation). Das Weltfenster ist
    # damit `[0, 64 * frequency)` - unabhängig von der Auflösung, das ist der
    # Trick, der die LOD-Stufen deckungsgleich hält. Doppelte Frequenz =
    # doppelt so breites Fenster AB DEMSELBEN URSPRUNG, die Welt setzt sich
    # also nach rechts und unten fort, statt sich zu verändern.
    #
    #     alt   Fenster [0, 2.3680)
    #     neu   Fenster [0, 4.7360)
    #
    # Gegenprobe gerechnet: das alte 256er Bild ist BIT-IDENTISCH mit dem
    # linken oberen Viertel des neuen 512er Bildes (max. Abweichung 0.000e+00).
    # Es ist wirklich dieselbe Welt, nur weiter herausgezoomt - kein neues
    # Muster mit ähnlichem Charakter.
    #
    # Unberührt bleibt die Nyquist-Klemme in _max_safe_octaves(): bei 4
    # Oktaven und Lacunarity 2.3 erlaubt die verdoppelte Frequenz weiterhin
    # 5 Oktaven, es fällt also keine weg.
    FREQUENCY = {
        "min": 0.001, "max": 0.1, "default": 0.074, "step": 0.001,
        "description": "Grundfrequenz des Rausch-Musters - höhere Werte "
                        "erzeugen kleinere, dichter aufeinanderfolgende "
                        "Hügel/Täler, niedrigere Werte großflächigere "
                        "Formationen. Verdoppeln zeigt den doppelten "
                        "Weltausschnitt in beide Richtungen (vierfache "
                        "Fläche), ab demselben Ursprung."
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
        "min": 0.5, "max": 4.0, "default": 2.0, "step": 0.1,
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
    # 2026-08-10 NEUE BEDEUTUNG: seit docs/spezifikation/14_SIEDLUNGEN.md wird die Zahl
    # der Siedlungen NICHT mehr direkt vorgegeben, sondern je Kultur aus der
    # Eignungssumme ihrer Region abgeleitet (2 bis 5, verglichen mit den
    # anderen acht Regionen - core/settlement_generator.py.calculate_settlements()).
    # Damit dieser Regler weiterhin etwas bewirkt, wirkt er jetzt als
    # MULTIPLIKATOR auf diese Ableitung, neutral bei seiner Vorgabe 3 - kleiner
    # gibt insgesamt weniger, groesser mehr Siedlungen, aber die Verteilung
    # zwischen den Kulturen (wer mehr, wer weniger bekommt) bleibt von der
    # Eignung bestimmt, nicht vom Regler.
    SETTLEMENTS = {
        "min": 1, "max": 5, "default": 3, "step": 1,
        "description": "Multiplikator auf die Siedlungszahl je Kultur (2-5, "
                        "aus der Eignung der jeweiligen Region abgeleitet). "
                        "Vorgabe 3 ist neutral; kleiner/groesser gibt "
                        "insgesamt weniger/mehr Siedlungen."
    }
    # 2026-08-10: wie SETTLEMENTS ein MULTIPLIKATOR statt einer absoluten
    # Gesamtzahl (Nutzer-Vorgabe: "pro region ein paar, so 1-4 jeweils").
    # Ziel je der neun Regionen ist 1-4, Vorgabe 3 ist wieder neutral
    # (core/settlement_generator.py.calculate_landmarks()). Vorher war der
    # Regler eine GESAMTzahl fuer die ganze Karte - bei Vorgabe 3 kam am Ende
    # nur eine Handvoll Landmarks auf der gesamten Weltkarte zusammen.
    LANDMARKS = {
        "min": 0, "max": 6, "default": 3, "step": 1,
        "description": "Multiplikator auf die Landmark-Zahl je Region (Ziel "
                        "1-4 je der neun Regionen). Vorgabe 3 ist neutral."
    }
    ROADSITES = {
        "min": 0, "max": 6, "default": 3, "step": 1,
        "description": "Multiplikator auf die Roadsite-Zahl je Region (Ziel "
                        "1-4 je der neun Regionen). Vorgabe 3 ist neutral."
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


# =============================================================================
# HAUPTSCHALTER Flussnetz
# =============================================================================
# Legt ein Flussnetz-Skelett in das Gelände und blendet zwischen Talsohle und
# umgebender Fläche (core/terrain_river_network.py, SPEZIFIKATION §12).
# Läuft in BaseTerrainGenerator._calc_redistribution() NACH dem Erosionsfilter,
# weil dessen Ergebnis die Fläche P ist, in die eingeschnitten wird.
#
# False lässt die Heightmap genau das, was der Erosionsfilter liefert.
FLUSSNETZ_AKTIV = True


def flussnetz_auslaesse(map_distance_km: float) -> int:
    """
    Wieviele getrennte Flussnetze eine Karte dieser Groesse traegt.

    Auf 15 x 15 km liegen keine drei unabhaengigen Flusssysteme - die feste
    Drei bis 2026-07-30 ergab ein zerstueckeltes Bild. Die Zahl waechst jetzt
    mit der Flaeche:

        bis 25 km    1 Netz
        25 - 50 km   2 Netze
        ab 50 km     3 Netze

    Mehrere Auslaesse bleiben wichtig, wo es sie gibt: mit nur einem muss JEDER
    Punkt der Karte dorthin entwaessern, das Netz ueberquert also jeden Ruecken
    dazwischen (§16). Auf kleinen Karten gibt es solche Ruecken selten.
    """
    if map_distance_km >= 50.0:
        return 3
    if map_distance_km >= 25.0:
        return 2
    return 1


class RIVER_NETWORK:
    """
    Regler des Flussnetzes (SPEZIFIKATION §12).

    Drei davon tragen den Charakter einer Landschaft:
        SPACING_M        Abstand der Täler
        INCISION_SHARE   wie tief sie schneiden, als Anteil der Höhenspanne
        PLATEAU_FLATTEN  wie stark die Fläche zwischen den Tälern eingeebnet wird
    """

    # 2026-08-06 auf 1200 m: das ist der MAKRO-Knotenabstand des
    # Weltflussnetzes (core/terrain_weltfluesse.STUFEN). Meso und Mikro folgen
    # im festen Verhaeltnis 1 : 1/2.86 : 1/8, ergeben also 420 und 150 m - die
    # Werte, mit denen das Netz eingemessen wurde. Die Verhaeltnisse sind
    # bewusst NICHT einzeln einstellbar: sie tragen die Schachtelung, und eine
    # Mesostufe groeber als Makro machte die Vererbung sinnlos.
    SPACING_M = {
        "min": 300.0, "max": 12000.0, "default": 1200.0, "step": 100.0,
        "suffix": "m",
        "description": "Abstand benachbarter Talsohlen in Metern. Kleine Werte "
                        "ergeben ein dichtes, feinverästeltes Talnetz, große "
                        "wenige große Täler. Bestimmt zusammen mit der "
                        "Talbreite, wieviel Hochfläche zwischen den Tälern "
                        "übrig bleibt."
    }
    # 2026-07-30 von Metern auf einen ANTEIL DER HÖHENSPANNE umgestellt.
    #
    # Der Nutzer dazu: "wir wollen eigentlich nicht diese tiefe einstellen
    # muessen. das wird doch automatisch durch die terrain-hoehe bestimmt."
    # Genau richtig - 400 m Eintiefung bedeuten in einer 4000-m-Landschaft
    # etwas anderes als in einer 200-m-Landschaft, und der Wert musste bei
    # jeder Änderung der Amplitude nachgezogen werden. 02_INVARIANTEN.md 4:
    # eine absolute Größe, wo eine relative hingehört.
    # 2026-08-06 auf 0.30: der Wert, mit dem taeler_eingraben() bisher fest
    # rechnete. Der alte Vorgabewert 0.55 gehoerte zum abgeloesten Netz.
    # 2026-08-25 von 0.30 auf 0.18 - die zweite Haelfte von "groesser, aber
    # weniger tief". Gemessen sinkt die mediane Abtragung auf Land von 44 auf
    # rund 34 m (384 px), der Anteil ueber 20 m von 68.8 auf 62.1 %.
    INCISION_SHARE = {
        # SCHRITT 0.05 -> 0.01 (2026-08-26). Die Vorgabe 0.18 lag NICHT
        # auf dem Raster: der Regler rastete beim Oeffnen des Reiters
        # auf 0.20, das blosse Anzeigen verstellte also den geeichten
        # Wert. Aufgefallen beim Umzug in den Flussreiter.
        "min": 0.0, "max": 0.6, "default": 0.18, "step": 0.01,
        "description": "Wie tief das größte Tal einschneidet, als Anteil der "
                        "Höhenspanne. Passt sich damit von selbst an die "
                        "Height Amplitude an. 0 lässt das Gelände unberührt; "
                        "kleinere Nebentäler schneiden entsprechend flacher."
    }
    # 0.0  Fläche zwischen den Tälern behält ihr volles Relief (Bergland)
    # 0.9  Fläche zwischen den Tälern ist eingeebnet (Hochebene, Skerrheim)
    #
    # Vorgabe 0.0: Hochebenen sind zurückgestellt, und für die Beurteilung des
    # Übergangs Tal -> Noise-Gelände muss das Noise-Gelände da sein.
    PLATEAU_FLATTEN = {
        "min": 0.0, "max": 0.9, "default": 0.0, "step": 0.05,
        "description": "Wie stark die Fläche ZWISCHEN den Tälern eingeebnet "
                        "wird. 0 lässt ihr das volle Relief (Bergland wie in "
                        "den Alpen), hohe Werte machen daraus eine Hochebene "
                        "mit tief eingeschnittenen Trögen (Skerrheim). Nach "
                        "oben bei 0.9 begrenzt - eine exakt ebene Fläche hätte "
                        "kein Gefälle mehr, dem ein Fluss folgen kann."
    }
    # 2026-08-06 NEUE BEDEUTUNG UND NEUE SPANNE: Faktor auf die Formgroesse der
    # Region, mit der die Talbreite gebildet wird (taeler_eingraben
    # breite_faktor). Die Vorgabe 1.1 ist der bisher fest verdrahtete Wert. Die
    # alte Spanne 0.15..1.0 haette ihn nicht mehr hergegeben.
    #
    # 2026-08-10 VORGABE 1.1 -> 0.35. Die Beschreibung unten sagte schon immer
    # "Anteil des Talabstands" - das stimmte aber nicht: bezogen wurde auf
    # `formgroesse_m`, also auf den Massstab des Rauschens. Damit waren die
    # Taeler 1200 bis 5500 m breit, waehrend die Laeufe im Mittel 111 m
    # auseinanderstanden, und das Eingraben wirkte als flaechige Glaettung: es
    # nahm dem Land 4.9 Grad mittleren Hang ab (Mittelmeerkueste 8.9 von 14.8).
    # `taeler_eingraben` bezieht jetzt auf SPACING_M; bei 0.35 bleibt der
    # mittlere Hang bis auf 0.6 Grad stehen. Die Beschreibung gilt damit
    # woertlich - 0.5 heisst wirklich "bis zur Mitte zwischen zwei Laeufen".
    # 2026-08-25 KURZ AUF 0.60 GESETZT UND WIEDER ZURUECK - der Versuch war
    # falsch, und die Beschreibung dieses Reglers sagte es bereits: "0.5
    # heisst wirklich bis zur Mitte zwischen zwei Laeufen". 0.60 liegt
    # DARUEBER, es bleibt also gar keine Hochflaeche mehr stehen.
    #
    # Gemessen brachte die Verbreiterung ohnehin fast nichts: der Anteil der
    # Landflaeche mit ueber 20 m Abtragung aendert sich zwischen
    # breite_faktor 0.12 und 0.60 nur von 52.4 auf 54.8 %. Der Grund steht
    # bei TALTIEFE_MINDESTWASSER in core/terrain_weltfluesse.py - die Laeufe
    # liegen im Median 1 Pixel auseinander, da spielt die Talbreite keine
    # Rolle mehr.
    #
    # Breitere Taeler fuer die GROSSEN Fluesse kommen stattdessen aus dem
    # Kontrast (TALBREITE_UNTERGRENZE/TALBREITE_EXPONENT): der Hauptstrom
    # bekommt 688 statt 402 m, die feinsten Baeche 104 statt 116 m.
    VALLEY_WIDTH = {
        "min": 0.10, "max": 3.0, "default": 0.35, "step": 0.05,
        "description": "Talbreite als Anteil des Talabstands. Bei 0.5 reicht "
                        "das Tal genau bis zur Mitte zwischen zwei Läufen und "
                        "es bleibt keine Hochfläche übrig; kleinere Werte "
                        "lassen eine stehen."
    }
    VALLEY_FORM = {
        "min": 0.3, "max": 3.0, "default": 1.3, "step": 0.1,
        "description": "Querschnitt des Tales. Unter 1 eine Schlucht mit Wand "
                        "direkt am Fluss, 1 ein V-Tal, über 1 ein U-Tal mit "
                        "flacher Sohle und steilen Flanken (glazial)."
    }
    # VALLEY_STEPS (Klippenbänder) 2026-07-30 ENTFERNT. Der Nutzer: "cliff
    # bands allgemein loeschen. das funktioniert nicht wie ich es haben will."
    # Die Treppenfunktion erzeugte ebene Absätze, die zusätzlich die
    # Entwässerung brachen (§17).
    MEANDER = {
        "min": 0.0, "max": 0.5, "default": 0.18, "step": 0.02,
        "description": "Seitliche Auslenkung der Flussläufe zwischen zwei "
                        "Knoten. Seit die Wege dem Gelände folgen, entsteht "
                        "der Mäander weitgehend von selbst - dieser Regler "
                        "wirkt daher nur noch schwach."
    }
    DIVIDE_BLEND = {
        "min": 0.0, "max": 1.0, "default": 0.75, "step": 0.05,
        "description": "Wie weich das Tal in die Umgebung übergeht. 0 ergibt "
                        "eine harte Kante an der Wasserscheide, 1 einen "
                        "glatten Übergang."
    }
    # ZWEI NEUE REGLER (2026-08-06). Beide waren Konstanten in
    # core/terrain_weltfluesse.py und gehoeren zu den Groessen, an denen beim
    # Bau des Netzes am meisten gedreht wurde - sie gehoeren an die Oberflaeche.
    MOUTH_DEPTH_M = {
        "min": 0.0, "max": 200.0, "default": 50.0, "step": 5.0,
        "description": "Wie tief unter den Meeresspiegel die Laeufe "
                       "weiterlaufen, bevor sie abgeschnitten werden. Ohne "
                       "diese Tiefe enden Fluesse sichtbar VOR der Kueste; "
                       "gezeichnet wird nur, was ueber 0 m liegt."
    }

    INHERIT_COST = {
        "min": 0.02, "max": 1.0, "default": 0.12, "step": 0.02,
        "description": "Was eine von der groberen Stufe geerbte Flussstrecke "
                       "kostet, verglichen mit einer neuen. Klein heisst: ein "
                       "Strom bleibt Strom. Bei 1.0 sucht sich jede Stufe "
                       "ihren eigenen Weg, und die Laeufe reissen ab."
    }

    COST_STRENGTH = {
        "min": 0.0, "max": 12.0, "default": 6.0, "step": 0.5,
        "description": "Wie stark die Flüsse hohes Gelände meiden. 0 lässt sie "
                        "den kürzesten Weg nehmen, hohe Werte zwingen sie um "
                        "die Berge herum statt hindurch."
    }
    # 2026-08-04 ergänzt. Der Nutzer im Bild: alle Flüsse liefen am Kartenrand
    # entlang bis zum einen Auslass und stiegen dafür sogar über die Rücken -
    # "aber nicht in dieser kreisrunden form". Der Rand ist eine durchgehende
    # billige Kette von Delaunay-Kanten; ohne einen Ausweg ist der Weg darauf
    # kürzer als der Weg quer durchs Gebirge.
    #
    # Gemessen als LÄNGSTE Kette, die den Randsaum nie verlässt, in Prozent
    # einer Kantenlänge (256 px, 625 Punkte):
    #     Regler     4.0    3.0    2.0    1.5    1.0    0.5
    #     Randlauf   ...   158%   107%    93%    61%    30%
    # Sehr hohe Werte entsprechen dem Verhalten davor.
    BORDER_OUTFLOW = {
        "min": 0.0, "max": 4.0, "default": 1.0, "step": 0.1,
        "description": "Wie teuer es ist, die Karte an einer beliebigen "
                        "Randstelle zu verlassen statt am Hauptauslass - "
                        "gemessen an einem Lauf über die halbe Karte. Kleine "
                        "Werte lassen viele kurze Flüsse direkt zum nächsten "
                        "Rand laufen, große zwingen alles zum Hauptauslass und "
                        "legen dabei einen Fluss um den Kartenrand herum."
    }


# =============================================================================
# HAUPTSCHALTER ATEF-Erosionsfilter
# =============================================================================
# Der Filter aus shaders/terrain/ATEF_*.comp, portiert in
# core/terrain_erosion_filter.py, angewandt in
# BaseTerrainGenerator._calc_redistribution(). Ein Durchgang pro Pixel, keine
# Iteration - siehe SPEZIFIKATION §9.
#
# False laesst die Heightmap genau das, was die Power-Redistribution liefert.
# =============================================================================
# HAUPTSCHALTER Weltkarte
# =============================================================================
# Steht er, kommt die Heightmap aus core/terrain_weltkarte.py: neun Regionen
# als Parameterfeld auf einer unregelmaessigen Kontinentform, mit Meer
# (docs/archiv/2026-08-04_INTEGRATIONSPLAN.md, Teil II). Der alte Pfad - Noise, Potenzkurve,
# ATEF-Filter, Flussnetz - bleibt vollstaendig erhalten und laeuft, sobald der
# Schalter aus ist.
#
# WAS SICH DAMIT AENDERT, und zwar sichtbar:
#
#   * Die Hoehe steht in ECHTEN METERN und darf NEGATIV sein. Unter 0 ist Meer.
#     Bisher war die Heightmap immer 0..AMPLITUDE.
#   * map_distance_km wird auf die Weltgroesse gesetzt (21.3 km). Der Regler
#     wirkt nicht mehr - die Regionsgroessen haengen daran.
#   * AMPLITUDE, FEATURE_SIZE_M und REDISTRIBUTE_POWER wirken nicht. Jede
#     Region bringt ihre eigenen Werte mit.
#
# Noch NICHT enthalten: Fluesse und Taeler (Stufe P2). Die Heightmap ist also
# das reine Regionengelaende.
WELTKARTE_AKTIV = True

# Kuestenformung ueber die VEKTORBESCHREIBUNG statt ueber Pixelmasken
# (core/vektor_kueste.py, docs/spezifikation/11_GELAENDE.md).
#
# Aus bei True bleibt `_kuesten_umformen()` in core/terrain_weltkarte.py
# unbenutzt. Der Vektorweg ist aufloesungsunabhaengig (Stationen in Metern
# statt per Pixelindex gezogen) und bezieht Profilform, Zielhoehe und
# Reichweite aus Messungen an 19 realen Vorbildkuesten statt aus geschaetzten
# Katalogwerten - der Katalog lag stellenweise um Faktor 16 daneben (Morobora
# 495 m gegen gemessene 30 m).
#
# UMSCHALTBAR, weil es die Gelaendeform aendert und die Regionseichung daran
# haengt (docs/OFFENE_PUNKTE.md 3.9, CLAUDE.md "Gelaendeaenderungen
# verstimmen zuerst die Regionseichung").
VEKTOR_KUESTE_AKTIV = True

# Das 3D-Netz entlang der Kuestenlinie SCHNEIDEN, damit sie eine echte
# Dreieckskante wird statt einer Rastertreppe
# (gui/widgets/kuesten_schnitt.py, docs/spezifikation/11_GELAENDE.md Abschnitt 10).
#
# Gemessen bei 384 px: 99.7 % der Konturvertices liegen NICHT auf einer
# Pixelecke (Quadtree: 0 %). Naht dicht, Konturhoehen exakt 0.
#
# PREIS: der Schnitt geht vom VOLLEN Gitter aus. Das adaptive Quadtree
# braucht nur rund ein Fuenftel davon - bei 1024 px also grob 2.1 Mio
# Dreiecke statt 0.41 Mio. Wer die Treppe hinnimmt, spart das.
KUESTEN_SCHNITT_AKTIV = True

# Fluesse und Taeler der Weltkarte (Stufe P2). Getrennt schaltbar, damit sich
# das reine Regionengelaende auch ohne sie ansehen laesst - und damit bei einem
# Fehler klar ist, welcher der beiden Schritte ihn verursacht.
#
# Kosten gemessen am 2026-08-05: bei 256 px 3.6 s fuer das Netz und 1.3 s fuer
# das Eingraben, bei 512 px 14.0 und 7.3 s.
WELTFLUESSE_AKTIV = True


EROSION_FILTER_AKTIV = True


class EROSION_FILTER:
    """
    Regler des ATEF-Erosionsfilters (SPEZIFIKATION §9).

    Alle Werte sind RELATIV zur Karte und zu ihrer Hoehenspanne, keine
    Meterwerte - nach 02_INVARIANTEN.md 4 der wichtigste Punkt an diesem
    Filter.

    Kein Reglerstand kann die Hoehenspanne verlassen: _calc_redistribution()
    bildet das Ergebnis nach dem Filter wieder auf
    BASE_ELEVATION_M .. AMPLITUDE ab. Die Regler formen also die VERTEILUNG,
    nicht die erreichte Hoehe - dieselbe Zusicherung wie bei
    REDISTRIBUTE_POWER.
    """

    STRENGTH = {
        "min": 0.0, "max": 0.6, "default": 0.22, "step": 0.01,
        "description": "Wie stark die Erosion das Gelaende umformt. 0 laesst "
                        "es unberuehrt. Wirkt auf alle Rinnen-Oktaven "
                        "gleichzeitig und beeinflusst dadurch auch die "
                        "Richtung der Rinnen."
    }
    # 2026-07-30 von einem Anteil der Kartenbreite auf METER umgestellt.
    #
    # Vorher hiess der Regler "Anteil der Kartenbreite", und damit hing die
    # Groesse der Rinnen an der Kartenausdehnung: gemessen ergaben 5 / 15 / 50 km
    # Ausdehnung Rinnen von 631 / 1893 / 6310 m - exakt proportional, Faktor 10
    # ueber den Bereich (smoke_test_terrain_scale_coupling.py). Beim Herauszoomen
    # wurden die Rinnen groesser statt zahlreicher, die Landschaft war also bei
    # jedem Ausschnitt eine andere.
    #
    # Eine Rinne ist gegen die WIRKLICHKEIT bemessen, nicht gegen den
    # Bildausschnitt - 02_INVARIANTEN.md 4, "jede neue Konstante mit Einheit
    # muss beantworten, gegen was sie bemessen ist". Die Umrechnung in den
    # kartenrelativen Wert, den der Filter selbst braucht, macht
    # BaseTerrainGenerator._erosion_filter_parameters().
    #
    # Vorgabe 2250 m = der frueherer Anteil 0.15 bei den vorgegebenen 15 km,
    # damit sich die Standardkarte nicht mit dieser Umstellung veraendert.
    GULLY_SIZE_M = {
        "min": 100.0, "max": 20000.0, "default": 2250.0, "step": 50.0,
        "suffix": "m",
        "description": "Groesse der Erosionsrinnen in Metern, unabhaengig von "
                        "Aufloesung und Kartenausschnitt. Kleine Werte ergeben "
                        "viele feine Rinnen, grosse wenige breite Taeler. "
                        "Wirkt waagerecht UND senkrecht. Werte unterhalb "
                        "weniger Pixel werden automatisch angehoben, weil sie "
                        "sonst nicht darstellbar waeren."
    }
    # Ticket #61 (2026-09-18): der Reglername war nach dem VERFAHREN benannt
    # (ATEF-intern "detail"), nicht nach der WIRKUNG. `GULLY_REACH` sagt, was
    # man sieht: wie weit die Rinne die Flanke hinablaeuft. Der Parameter-
    # schluessel bleibt bewusst "erosion_filter_detail" (siehe
    # docs/AUFRAEUMPLAN.md, Abschnitt "Der Anspruch: ATEF von Rune Johansen")
    # - das ist der Name der ATEF-Quelle und wird NICHT angefasst.
    GULLY_REACH = {
        "min": 0.3, "max": 3.5, "default": 1.5, "step": 0.1,
        "description": "Wie weit die feinen Rinnen von den Steilhaengen auf "
                        "flacheres Gelaende hinauslaufen. Kleine Werte halten "
                        "sie auf den steilen Flanken."
    }
    # Ticket #61: vorher GULLY_WEIGHT - der Name verriet nicht, WOGEGEN
    # gewichtet wird. Parameterschluessel bleibt "erosion_filter_gully_weight"
    # (ATEF-Quelle, siehe oben).
    GULLY_VS_SHARPNESS = {
        "min": 0.0, "max": 1.0, "default": 0.5, "step": 0.05,
        "description": "Rinnen gegen Kantenschaerfe. Bei 0 entstehen kaum "
                        "Rinnen, dafuer werden Gipfel und Talsohlen "
                        "geschaerft. Bei 1 volle Rinnen, dafuer bleiben "
                        "Gipfel und Sohlen runder."
    }
    RIDGE_ROUNDING = {
        "min": 0.0, "max": 1.0, "default": 0.1, "step": 0.05,
        "description": "Rundung der Kaemme. 0 ergibt scharfe Grate, hohe "
                        "Werte abgerundete Ruecken."
    }
    # Ticket #61: vorher CREASE_ROUNDING - "Crease" ist der ATEF-interne
    # Fachbegriff (Falte/Knick), "Talsohle" ist, was man im Bild sieht.
    # Parameterschluessel bleibt "erosion_filter_crease_rounding" (ATEF-Quelle,
    # siehe oben).
    VALLEY_ROUNDING = {
        "min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05,
        "description": "Rundung der Talsohlen. 0 ergibt scharf eingeschnittene "
                        "Kerben, hohe Werte weiche Mulden."
    }
    OCTAVES = {
        "min": 1, "max": 7, "default": 5, "step": 1,
        "description": "Anzahl der uebereinandergelegten Rinnen-Groessen. "
                        "Oktaven, deren Rinnen feiner als zwei Pixel "
                        "wuerden, werden automatisch weggelassen - bei "
                        "kleiner Map Size wirkt der Regler deshalb nach oben "
                        "nicht mehr."
    }


# =============================================================================
# HAUPTSCHALTER Erosion
# =============================================================================
# False schaltet den gesamten Erosionslauf ab: core/erosion_generator.py
# _calc_hydraulic() schreibt dann für JEDES LOD Nullkarten, statt nur für die
# Zwischenrunden. Das Gelände bleibt damit exakt das unerodierte -
# get_calculator_combined_heightmap() zieht erosion_map ab und addiert
# sedimentation_map, beide null.
#
# Bewusst KEIN Slider und bewusst kein Entfernen des Knotens aus dem
# CALCULATOR_GRAPH: der Knoten liefert weiter seine sieben Karten in der
# richtigen Form und Größe, alle Kanten und alle Verbraucher (water.*, biome.*,
# erosion.slope, die Anzeige-Layer) laufen unverändert. Fünf handgepflegte
# Generatorlisten haben in diesem Projekt je einen Deadlock oder eine fehlende
# Invalidierung verursacht (02_INVARIANTEN.md 5) - ein Knoten, der Nullen
# liefert, ist der Weg, der das nicht wieder auslöst.
#
# Absichtlich AUS seit 2026-07-30: die Erosion wird durch den Skelett-Ansatz
# ersetzt (Struktur vor Noise, SPEZIFIKATION §8) und soll später nur noch als
# Feinschliff auf einem bereits entwässerten Gelände laufen. Bis dahin ist ihr
# Beitrag laut §7 negativ - sie ERZEUGT die Becken, die sie auflösen soll.
#
# 2026-07-30 wieder auf False (Nutzer-Entscheidung): die Erosion ist pausiert
# und reicht Nullkarten durch, alles DAHINTER muss weiterlaufen.
#
# Zwischenzeitlich auf True gesetzt, weil der Erosion- und der Water-Tab leer
# blieben. Das war eine Fehldiagnose: die leeren Water- und Geology-Anzeigen
# kamen daher, dass SHADERS_ROOT nach dem Ordnerumzug ins Leere zeigte und
# JEDE GPU-Operation still auf den CPU-Pfad zurueckfiel (siehe
# tests/smoke_test_shader_paths.py). Nur der Erosion-Tab selbst ist von diesem
# Schalter betroffen, und das ist beabsichtigt.
# WIEDER AN seit 2026-08-27, nach Messung. Die Begruendung von 2026-07-30
# ("sie ERZEUGT die Becken, die sie aufloesen soll") war richtig - fuer das
# Gelaende von damals. Sie traegt nicht mehr, seit das Flussnetz ein
# entwaessertes Gelaende vorlegt. Gemessen bei 256 px, Seed 20260804, mit den
# fuenf Kennzahlen aus tests/smoke_test_erosion_quality.py:
#
#     Gelaende                       Top5    Netz  Krater  Ebenen  Schritte
#     synthetisch (so misst der Test) 0.397   105     12     4.0%    875
#     echt, ohne Flussnetz            0.899   546      1    23.3%    125
#     echt, MIT Flussnetz             0.909   273      3    23.7%    100
#
# Auf dem echten Gelaende bleiben 3 Krater von 65536 Pixeln - die Erosion
# erzeugt dort keine Becken mehr. Die drei roten Befunde von
# smoke_test_erosion_quality sind Eigenschaften seines SYNTHETISCHEN
# Testgelaendes, nicht der Erosion.
#
# Vorher behoben werden musste ein zweiter Punkt: der GPU-Pfad prueft die
# Konvergenz jetzt im selben Takt wie die CPU (CONVERGENCE_CHECK_INTERVAL
# statt PROGRESS_REPORT_INTERVAL, siehe core/erosion_generator.py). Ohne das
# haette die Produktionskarte auf der GPU zwanzigmal laenger erodiert als
# noetig - 500 statt 25 Schritte.
EROSION_AKTIV = True


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
    tools/erosion_lab.py - es rechnet Varianten auf der GPU, misst vier
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
        "min": 0.0, "max": 1.0, "default": 0.05, "step": 0.05,
        "description": "Wie stark die Gesteinshärte aus Geology eingeht - auf "
                        "die Löserate UND den Böschungswinkel. 0 % ist exakt "
                        "das Verhalten des Vorbilds (das kein Gestein kennt), "
                        "100 % volle Kopplung: hartes Gestein löst langsamer "
                        "und hält steilere Wände."
    }
    # ACHTUNG bei alten Werten: dieser Regler ist am 2026-07-28 UMGEDREHT
    # worden. Vorher steuerte er die Schwelle direkt, also genau verkehrt -
    # kleine Werte bedeuteten AGGRESSIVE Glättung. Ein gespeichertes Preset
    # mit 0.05 hat damals das Gegenteil von heute bewirkt. Die Messreihe
    # dazu steht in HydraulicFieldSimulator.smoothing_threshold().
    SMOOTHING = {
        "min": 0.0, "max": 1.0, "default": 0.0, "step": 0.05,
        "description": "Wie stark Ein-Pixel-Grate und -Rinnen geglättet "
                        "werden, also Gitterartefakte - echte Hänge bleiben "
                        "unangetastet. 0 = aus, 1 = maximal. Startwert "
                        "bewusst sehr niedrig: ab etwa 0.3 kostet der Pass "
                        "mehr echte Rinnen als er Artefakte entfernt, und "
                        "die Schwemmebenen in den Talböden bleiben aus."
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
    # ACHTUNG, DOPPELTER SCHLUESSEL (docs/OFFENE_PUNKTE.md 12.3).
    #
    # `erosion_strength` gibt es ZWEIMAL: hier (Droplet-Altbestand, 0.1-5.0,
    # Vorgabe 2.5) und in class EROSION (Feld-Erosion, 0.0-2.0, Vorgabe 0.5).
    # Beide schreiben nach parameters['erosion_strength']; welcher gewinnt,
    # haengt an der Reihenfolge der Zusammenstellung. Das ist ausdruecklich
    # KEIN Zustand, den man uebersehen soll - `DOPPELTE_SCHLUESSEL` unten
    # fuehrt ihn, und tests/smoke_test_parameter_eindeutig.py schlaegt fehl,
    # sobald ein NEUER unangemeldeter Doppelschluessel dazukommt.
    EROSION_STRENGTH = {
        "min": 0.1, "max": 5.0, "default": 2.5, "step": 0.1,
        "description": "[ALTBESTAND DROPLET-EROSION - siehe Kommentar oben; "
                        "der wirksame Regler gleichen Namens steht in class "
                        "EROSION] Genereller Multiplikator dafür, wie stark ein "
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
    # Default 2026-07-30 von 10 auf 0 gesenkt, zusammen mit
    # TERRAIN.BASE_ELEVATION_M = 0.0 (siehe dort). Die Talsohle liegt jetzt
    # exakt bei 0 m; bliebe der Meeresspiegel bei 10 m, würde der untere Teil
    # jeder Karte als Ozean klassifiziert - SPEZIFIKATION §1 führt die Karten
    # aber ausdrücklich OHNE Meer. 02_INVARIANTEN.md 7: abhängige Defaults
    # ziehen mit.
    SEA_LEVEL = {
        "min": 0, "max": 200, "default": 0, "step": 5, "suffix": "m",
        "description": "Höhe des Meeresspiegels - alles darunter wird als "
                        "Wasser/Küste klassifiziert. Bei 0 m fällt er mit der "
                        "Talsohle zusammen, die Karte bleibt also vollständig "
                        "Land."
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
    # 2026-08-25 von 60 auf 45 gesenkt (Nutzervorgabe nach der Messung).
    #
    # BEI 60 GRAD GAB ES AUF DER GANZEN KARTE EINE EINZIGE KLIPPE - genau ein
    # Pixel in `biome_map_super`. Kein Fehler in der Rechnung, sondern eine
    # Schwelle, die das fertige Gelaende nicht erreicht: eine Heightmap kann
    # nur so steil sein, wie ihre Aufloesung zulaesst (dieselbe Grenze wie in
    # docs/OFFENE_PUNKTE.md 6.19).
    #
    # Gemessen auf der Heightmap, mit der die Biom-Stufe wirklich rechnet
    # (128 px, 117.2 m/px, 5332 Landpixel) - NICHT auf dem Rohgelaende, das
    # deutlich steiler ist:
    #
    #     Schwelle   Anteil des Landes   noetig je Pixel
    #        30 Grad        18.68 %             67.7 m
    #        40 Grad         8.16 %             98.3 m
    #        45 Grad         4.58 %            117.2 m   <- neu
    #        50 Grad         1.89 %            139.7 m
    #        60 Grad         0.15 %            203.0 m   <- alt
    #
    # 45 Grad liegt damit in derselben Groessenordnung wie die uebrigen
    # Wahrscheinlichkeits-Biome (`beach` 0.47 %, `alpine_level` 1.84 % der
    # Karte), statt praktisch leer zu sein.
    #
    # EHRLICHE EINSCHRAENKUNG: der Anteil haengt an der Aufloesung. Steilere
    # Haenge werden erst bei feinerem Raster ueberhaupt darstellbar - bei
    # 1024 px ueberschreiten im Rohgelaende 9.4 % der Landflaeche 45 Grad
    # gegen 4.4 % bei 256 px. Dieselbe Schwelle gibt auf einer feineren Karte
    # also mehr Klippe. Wer das nicht will, muss die Schwelle an die
    # Kartengroesse koppeln - das ist bewusst NICHT gemacht, weil es dann drei
    # Groessen zu eichen gaebe statt einer.
    CLIFF_SLOPE = {
        "min": 30, "max": 80, "default": 45, "step": 1, "suffix": "°",
        "description": "Mindest-Hangneigung, ab der ein Bereich als "
                        "Klippe/Fels statt als normales Gelände "
                        "klassifiziert wird. Bei 60° kam auf der ganzen "
                        "Karte praktisch keine Klippe vor - so steil wird "
                        "das gerasterte Gelände nicht (gemessen 2026-08-25)."
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
# BEKANNTE, ABSICHTLICH DOPPELTE PARAMETERSCHLUESSEL.
#
# Schluessel -> (Klassen, Begruendung). Alles, was hier NICHT steht und
# trotzdem in zwei Klassen auftaucht, ist ein Versehen und wird von
# tests/smoke_test_parameter_eindeutig.py gemeldet. Ohne dieses Register
# waere ein neuer Doppeleintrag genau das, was 12.3 beschreibt: eine stille
# Mehrdeutigkeit, die spaeter als unerklaerliches Verhalten zurueckkommt.
DOPPELTE_SCHLUESSEL = {
    "erosion_strength": (
        ("EROSION", "WATER"),
        "ECHTE Mehrdeutigkeit: beide schreiben nach "
        "parameters['erosion_strength']. WATER ist Altbestand der "
        "Droplet-Erosion (stillgelegt 2026-07-28), wirksam ist EROSION "
        "(0.0-2.0, Vorgabe 0.5) gegen WATER (0.1-5.0, Vorgabe 2.5)."),
    "octaves": (
        ("EROSION_FILTER", "TERRAIN"),
        "HARMLOS - nur der Attributname ist gleich, der Parameterschluessel "
        "nicht: TERRAIN.OCTAVES laeuft als 'octaves', EROSION_FILTER.OCTAVES "
        "als 'erosion_filter_octaves' (gui/tabs/terrain_tab.py:117 und :141). "
        "Steht hier, damit die Pruefung nicht bei jedem Lauf darueber "
        "stolpert - und damit sichtbar bleibt, dass es geprueft wurde."),
}


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


# =============================================================================
# STILLGELEGTE REGLER
# =============================================================================

def stillgelegte_regler():
    """
    Welche Regler bewirken im aktuellen Programmstand nichts - und warum.

    ANLASS. Am 2026-08-06 wurde gemessen (nicht geschaetzt): ein Grundlauf, dann
    je Regler ein deutlich anderer Wert, nur die Knoten seines eigenen
    Generators neu gerechnet, jeder Output verglichen. Ergebnis: 49 von 109
    Reglern aenderten nichts. Vier von fuenf Ursachen waren dieselbe Sache - die
    Umstellung auf die Weltkarte hat 24 Regler stillgelegt, ohne sie aus der
    Oberflaeche zu nehmen. Die Oberflaeche zeigte einen Programmstand von vor
    dem Umbau.

    DIESE FUNKTION HAENGT AN DEN SCHALTERN, nicht an einer festen Liste. Wird
    WELTKARTE_AKTIV wieder ausgeschaltet, sind die Terrain-, Fluss- und
    Filterregler sofort wieder frei; dasselbe gilt fuer EROSION_AKTIV. Eine
    fest verdrahtete Sperre waere beim naechsten Umschalten falsch.

    NICHT ENTHALTEN sind die Regler, die aus einem anderen Grund nichts tun -
    `octaves`, `frequency`, `river_meander`, `erosion_filter_octaves` wirken in
    KEINEM der beiden Modi. Das ist kein Nebeneffekt eines Schalters, sondern
    ein eigener Befund; sie werden hier trotzdem gefuehrt, aber mit eigenem
    Grund, damit die Untersuchung nicht in Vergessenheit geraet.

    Return: dict {parameter_schluessel: begruendung}
    """
    gesperrt = {}

    if WELTKARTE_AKTIV:
        grund = ("Die Weltkarte ist aktiv. Die neun Regionen bringen ihre "
                 "eigenen Gelaendewerte mit (core/terrain_weltkarte.py), der "
                 "Rauschaufbau dieses Reglers wird nicht mehr durchlaufen.")
        # `octaves` steht seit dem 2026-08-06 HIER und nicht mehr bei den
        # angeblich wirkungslosen: nachgemessen am alten Pfad aendert er das
        # Gelaende sehr wohl (Regler 1 gegen 3: 1434 m Hoehenunterschied). Er
        # ist nur bei aktiver Weltkarte still, wie die anderen vier auch.
        for schluessel in ("amplitude", "feature_size_m", "redistribute_power",
                           "persistence", "lacunarity", "octaves"):
            gesperrt[schluessel] = grund

        gesperrt["map_distance_km"] = (
            "Die Weltkarte setzt die Kartenbreite selbst auf %.1f km - ein "
            "anderer Wert wuerde Regionsgroessen und Talabstaende "
            "gegeneinander verschieben." % TERRAIN.WORLD_SIZE_KM)

        # NUR NOCH VIER. Am 2026-08-06 wurden fuenf der neun Flussregler an
        # core/terrain_weltfluesse.py angeschlossen (river_spacing_m,
        # river_cost_strength, river_valley_width, river_valley_form,
        # river_incision_share) und zwei neue kamen dazu (river_mouth_depth_m,
        # river_inherit_cost). Die vier hier beschreiben Dinge, die es im
        # Weltflussnetz nicht gibt.
        gesperrt["river_border_outflow"] = (
            "Die Welt ist eine Insel im offenen Meer - sie entwaessert ins "
            "Meer und nicht ueber den Kartenrand. Dieser Preis hat kein "
            "Gegenstueck mehr.")
        gesperrt["river_divide_blend"] = (
            "Das Weltflussnetz baut seine Laeufe aus einem Knotengraphen und "
            "kennt keine weich ueberblendeten Wasserscheiden.")
        gesperrt["river_plateau_flatten"] = (
            "Hochflaechen entstehen in der Weltkarte aus der `potenz` der "
            "jeweiligen Region, nicht aus einer Nachbehandlung des Flussnetzes.")
        gesperrt["river_meander"] = (
            "Gehoert zum abgeloesten Flussnetz (terrain_river_network.py); das "
            "Weltflussnetz legt seine Laeufe ueber einen Knotengraphen und "
            "maeandriert sie nicht nachtraeglich. Er wirkte ausserdem schon im "
            "alten Pfad nicht - eigener, ungeklaerter Befund.")

        # DIE FILTERREGLER SIND SEIT DEM 2026-08-07 WIEDER FREI.
        #
        # Der Erosionsfilter laeuft jetzt auch auf der Weltkarte mit - nach dem
        # Weltfeld, vor dem Flussnetz (INTEGRATIONSPLAN S4), mit eigener Weiche
        # ohne Hoehennormierung und mit Regionsgewichtung. Gemessen formt er in
        # den Bergen 64.5 m um und in den Niederungen 17.6, also Faktor 3.7 -
        # genau die Vorgabe "in den bergen wo mehr masse ist haben wir mehr
        # features, und in den niederungen weniger".
        #
        # `erosion_filter_octaves` bleibt gesperrt, aber aus einem anderen
        # Grund: er ist ab 5 durch die Nyquist-Grenze geklemmt (siehe unten).

    if not EROSION_AKTIV:
        grund_erosion = ("Die Erosionskette ist abgeschaltet (EROSION_AKTIV). "
                         "Alle erosion.*-Knoten liefern Nullkarten.")
        for schluessel in ("convergence_threshold", "deposition_rate",
                           "erosion_capacity", "erosion_strength",
                           "evaporation_rate", "hardness_influence",
                           "max_steps", "rainfall", "simulation_resolution",
                           "smoothing", "talus_angle_scale",
                           "thermal_strength", "thermal_variant"):
            gesperrt[schluessel] = grund_erosion

    # ALTBESTAND DROPLET-EROSION (docs/OFFENE_PUNKTE.md 12.3/12.4).
    #
    # Diese Schluessel gehoeren zu `water.erosion_sedimentation` und
    # `water.thermal_erosion` - im Berechnungsgraphen als "ALTBESTAND
    # DROPLET-EROSION (stillgelegt 2026-07-28)" gefuehrt und ausdruecklich
    # "ersetzt durch erosion.hydraulic". Sie stehen in keinem Reiter, werden
    # von `core/water_generator.py` aber weiterhin ueber
    # `parameters.get(...)` mit fest eingebauten Vorgaben gelesen - deshalb
    # sind sie hier NICHT geloescht, sondern benannt. Ein Loeschen wuerde die
    # dokumentierte Spanne entfernen und den stillen Rueckfall auf die
    # Literalwerte im Generator zuruecklassen.
    grund_droplet = (
        "Gehoert zur abgeloesten Droplet-Erosion (water.erosion_sedimentation "
        "/ water.thermal_erosion, stillgelegt 2026-07-28, ersetzt durch "
        "erosion.hydraulic). Steht in keinem Reiter und wirkt nicht.")
    for schluessel in ("erosion_passes", "sediment_capacity_factor",
                       "settling_velocity", "thermal_erosion_strength",
                       "evaporation_base_rate", "diffusion_radius"):
        gesperrt.setdefault(schluessel, grund_droplet)

    # `frequency` ist ABGELOEST, nicht kaputt.
    #
    # _calc_noise benutzt ihn nur, wenn `feature_size_m` NICHT gesetzt ist -
    # und der hat eine Vorgabe, ist also immer gesetzt. Das ist Absicht:
    # feature_size_m gibt die Formgroesse in METERN an und haengt damit an der
    # Wirklichkeit, waehrend `frequency` nur an der Pixelzahl hing. Der alte
    # Weg blieb fuer Labore und Altbestand stehen.
    #
    # Nachgemessen am 2026-08-06: mit feature_size_m = 0 wirkt `frequency`
    # sofort wieder (Rauschstreuung 0.057 bei 0.001 gegen 0.263 bei 0.1).
    # Er steht ohnehin in keinem Reiter.
    gesperrt["frequency"] = (
        "Abgeloest durch `feature_size_m`, der die Formgroesse in Metern angibt "
        "statt in Zyklen je Pixel. Wirkt nur noch, wenn feature_size_m auf 0 "
        "steht.")

    # `alpine_level`/`snow_level` (Biome-Reiter) sind KEIN Schalterfall wie
    # oben - der Umbau, der sie stillgelegt hat, ist nicht umschaltbar,
    # deshalb stehen sie hier unconditional und nicht in einem `if`-Block.
    #
    # Seit dem Umbau vom 2026-08-07 (core/biome_generator.py) lesen
    # _calculate_alpine_level_probabilities() und
    # _calculate_snow_level_probabilities() die festen Modulkonstanten
    # BAUMGRENZE_JULI_C = 10.0 und FIRN_JULI_C = 0.0 (Julitemperatur statt
    # Hoehe) - Begruendung dort: die alte Hoehenregel loeste beim damaligen
    # Gelaende nie aus (hoechster Punkt 728 m, Schwelle ab 2750 m).
    # `self.alpine_level` wird seither nur noch von
    # _alt_alpine_level_probabilities() gelesen, einer nie aufgerufenen
    # Vergleichsmethode (eigener Docstring: "steht nur noch zum Vergleich
    # hier"). `self.snow_level` wird NIRGENDS mehr gelesen.
    #
    # Das gilt unabhaengig vom Aufrufpfad: der GPU-Handler
    # shader_manager.request_biome_classification existiert im Repo gar
    # nicht, jeder Pfad faellt also auf denselben toten CPU-Code zurueck.
    grund_biome_juli = (
        "Seit dem Julitemperatur-Umbau vom 2026-08-07 lesen "
        "_calculate_alpine_level_probabilities()/_calculate_snow_level_"
        "probabilities() (core/biome_generator.py) die festen Konstanten "
        "BAUMGRENZE_JULI_C/FIRN_JULI_C statt dieses Reglers. `alpine_level` "
        "wird nur noch von der toten Vergleichsmethode "
        "_alt_alpine_level_probabilities() gelesen, `snow_level` von "
        "keinem Code mehr.")
    gesperrt["alpine_level"] = grund_biome_juli
    gesperrt["snow_level"] = grund_biome_juli

    return gesperrt
