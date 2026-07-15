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


class TERRAIN:
    """Parameter für core/terrain_generator.py"""
    MAPSIZEMIN = 32
    MAPSIZEMAX = 1024

    # Reale Weltgröße, die die Karte immer abdeckt, unabhängig von map_size/
    # Pixelauflösung (siehe gui/widgets/map_display_3d.py _calculate_terrain_scaling()
    # und core/terrain_generator.py SlopeCalculator._calculate_cpu_slopes() - beide
    # brauchen dieselbe Meter-pro-Pixel-Annahme, sonst driften 3D-Darstellung und
    # Slope-basierte Physik/Biome-Klassifikation auseinander)
    WORLD_SIZE_KM = 10.0

    MAPSIZE = {
        "min": MAPSIZEMIN, "max": MAPSIZEMAX, "default": 128, "step": 32,
        "description": "Auflösung der Karte in Pixeln (Breite = Höhe). Größere "
                        "Werte zeigen mehr Detail, verlangsamen aber jede "
                        "nachfolgende Generierungsstufe."
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
    OCTAVES = {
        "min": 1, "max": 8, "default": 4, "step": 1,
        "description": "Anzahl der übereinandergelegten Rausch-Schichten "
                        "unterschiedlicher Frequenz. Mehr Oktaven fügen "
                        "feinere Detailebenen hinzu, verlangsamen aber die "
                        "Berechnung."
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
        "min": 0, "max": 999999, "default": 542595, "step": 1,
        "description": "Zufalls-Startwert für die Terrain-Generierung - "
                        "derselbe Seed erzeugt bei sonst gleichen "
                        "Parametern immer exakt dieselbe Karte."
    }

class GEOLOGY:
    """Parameter für core/geology_generator.py"""
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
    RIDGE_WARPING = {
        "min": 0.0, "max": 2.0, "default": 0.5, "step": 0.1,
        "description": "Verzerrt Gebirgskämme mit zusätzlichem Rauschen - "
                        "höhere Werte lassen Bergketten weniger geradlinig "
                        "und gleichmäßig wirken."
    }
    BEVEL_WARPING = {
        "min": 0.0, "max": 2.0, "default": 0.3, "step": 0.1,
        "description": "Verzerrt die Übergänge/Kanten zwischen "
                        "Gesteinstypen mit zusätzlichem Rauschen für "
                        "weniger scharfe, natürlicher wirkende Grenzen."
    }
    METAMORPH_FOLIATION = {
        "min": 0.0, "max": 1.0, "default": 0.4, "step": 0.1,
        "description": "Stärke der Schieferung (parallele Streifenstruktur) "
                        "in metamorphem Gestein."
    }
    METAMORPH_FOLDING = {
        "min": 0.0, "max": 1.0, "default": 0.6, "step": 0.1,
        "description": "Stärke der Faltung (wellenförmige Verformung) in "
                        "metamorphem Gestein."
    }
    IGNEOUS_FLOWING = {
        "min": 0.0, "max": 1.0, "default": 0.7, "step": 0.1,
        "description": "Wie stark vulkanisches Gestein Fließmuster zeigt "
                        "(z.B. erkennbare Lavaströme statt gleichmäßiger "
                        "Verteilung)."
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
        "min": 50, "max": 5000, "default": 1000, "step": 10,
        "description": "Anzahl der Kandidaten-Punkte für die Grundstücks-/"
                        "Bebauungsplanung - mehr Punkte erlauben feinere "
                        "Parzellierung, kosten aber Rechenzeit."
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
        "min": 2.0, "max": 40.0, "default": 10.0, "step": 1.0,
        "description": "Grundabstand zwischen einzelnen Grundstücks-"
                        "Parzellen."
    }
    PLOT_CIV_SPACING_FACTOR = {
        "min": 0.0, "max": 10.0, "default": 3.0, "step": 0.1,
        "description": "Wie stark sich der Parzellenabstand mit der Nähe "
                        "zum Stadtzentrum verringert (dichtere Bebauung im "
                        "Zentrum)."
    }
    PLOT_HEIGHT_COST_FACTOR = {
        "min": 0.0, "max": 10.0, "default": 2.0, "step": 0.1,
        "description": "Wie stark Höhenunterschiede die 'Baukosten' einer "
                        "Parzelle erhöhen - steile Grundstücke werden "
                        "dadurch seltener/kleiner bebaut."
    }

class WEATHER:
    """Parameter für core/weather_generator.py"""
    AIR_TEMP_ENTRY = {
        "min": -30, "max": 40, "default": 15, "step": 1, "suffix": "°C",
        "description": "Lufttemperatur am Kartenrand, wo die Luft "
                        "'eintritt' - Grundlage für die gesamte "
                        "Temperaturberechnung auf der Karte."
    }
    SOLAR_POWER = {
        "min": 0, "max": 50, "default": 20, "step": 1, "suffix": "°C",
        "description": "Stärke der Sonnenerwärmung - wie viel wärmer "
                        "sonnenbeschienene Flächen gegenüber beschatteten "
                        "Flächen werden."
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
        "min": 0, "max": 360, "default": 225, "step": 5, "suffix": "°",
        "description": "Grundrichtung des vorherrschenden Windes (0°=Ost, "
                        "90°=Nord) - rotiert leicht mit den Jahreszeiten, "
                        "lokale Abweichungen entstehen zusätzlich durchs "
                        "Gelände."
    }
    # Eintritts-Luftfeuchte (ersetzt die bisher hartcodierte 50%-Baseline in
    # _calculate_atmospheric_moisture_cpu()'s Verdunstungs-Ausgangswert).
    AIR_HUMIDITY_ENTRY = {
        "min": 0, "max": 100, "default": 50, "step": 1, "suffix": "%",
        "description": "Luftfeuchtigkeit am Kartenrand beim Lufteintritt - "
                        "Ausgangswert für die Verdunstungs-/"
                        "Feuchtigkeitsberechnung."
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
            "solar_power":        [-8.0, -2.0, 4.0, 6.0, -1.0, -7.0],  # °C-Skala
            "wind_speed_factor":  [0.15, 0.05, -0.05, -0.10, 0.0, 0.10],
            # % - Summe Peak-zu-Tal bleibt < 30% des 0-100%-Sliderbereichs
            "air_humidity_entry": [15.0, 5.0, -5.0, 0.0, 8.0, 18.0],
        },
    }
    CLIMATE_ZONE = "temperate"  # vorerst fix, siehe Backlog für spätere Auswahl


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
    LAKE_VOLUME_THRESHOLD = {
        "min": 0.001, "max": 0.3, "default": 0.02, "step": 0.005, "suffix": "m",
        "description": "Mindest-'Volumen' eines Geländebeckens, damit dort "
                        "ein See entsteht - niedrigere Werte lassen auch "
                        "kleine/flache Senken zu Seen werden."
    }
    # ZWEITE Neukalibrierung (siehe [[project-3layer-wind-cfd]]): der neue
    # gekoppelte 3-Schicht-Atmosphären-Loop (core/weather_generator.py
    # _run_coupled_atmosphere_simulation) akkumuliert Kondensation über den
    # gesamten Zeitschritt-Loop statt sie aus einem Einzelschuss-Magnus-Snapshot
    # abzuleiten - precip_map liegt dadurch jetzt bei Default-Parametern um
    # Größenordnungen höher (empirisch min~4, Mittel~13, max~31 gH2O/m² statt
    # vorher ~0-2.8, Mittel~0.08) als zum Zeitpunkt der VORHERIGEN Kalibrierung
    # (Kommentar unten) angenommen. Der alte 0.02-1.0-Bereich lag dadurch
    # KOMPLETT unter jedem real vorkommenden precip_map-Wert (100% der Pixel
    # zählten unabhängig vom Slider-Stand als Regen-Quelle) - Bereich neu an die
    # tatsächliche Verteilung angepasst, Default knapp unter dem beobachteten
    # Minimum (bleibt bewusst permissiv, siehe Docstring-Kommentar unten zur
    # eigentlichen Funktion dieses Parameters).
    RAIN_THRESHOLD = {
        "min": 0.5, "max": 20.0, "default": 3.0, "step": 0.1, "suffix": "gH2O/m²",
        "description": "Mindest-Niederschlag, damit ein Pixel überhaupt als "
                        "Wasserquelle für Flüsse zählt (reiner Filter, "
                        "keine Mengenangabe für die Flussgröße selbst)."
    }
    # Separat von RAIN_THRESHOLD: rain_threshold entscheidet nur noch, ob ein
    # Pixel überhaupt Regen-Quelle ist (Beitrag zur Akkumulation), STREAM_THRESHOLD
    # entscheidet, ob AKKUMULIERTER Durchfluss als sichtbarer Fluss gilt. Vorher
    # war Creek = 1x rain_threshold - identisch mit der Quellen-Schwelle selbst,
    # wodurch jedes einzelne Regen-Pixel sofort als Fluss galt, ganz ohne echten
    # Zufluss von Nachbar-Zellen ("Fluss überall wo Regen fällt").
    #
    # ZWEITE Neukalibrierung (wie RAIN_THRESHOLD oben, gleicher Grund): mit dem
    # alten default=2.0 lag SELBST der Minimalwert von flow_accumulation (jetzt
    # ~4, da schon ein einzelnes Regen-Pixel durch den neuen precip_map-Bereich
    # allein über der alten Creek-Schwelle 1x2.0 lag) über der Creek-Schwelle -
    # 0% der Karte hatte noch die Klasse "kein Wasser", ~94% waren River/Grand
    # River. Neuer Default (35.0) empirisch gegen einen frischen Smoke-Test der
    # tatsächlichen flow_accumulation-Verteilung mit dem neuen precip_map-Bereich
    # kalibriert - ergibt wieder eine absteigende Verteilung (mehr Creeks als
    # Rivers, ein plausibler Anteil ganz ohne Wasser) statt eines dominanten
    # Grand-River-Bands. Diese Kalibrierung ist an eine kleine Test-Karte
    # (64x64, 5 Flow-Iterationen) gebunden - bei sehr großen Karten/hohen LODs
    # kann flow_accumulation deutlich höher werden (mehr akkumulierende
    # Upstream-Zellen), ggf. weitere Nachjustierung nötig.
    STREAM_THRESHOLD = {
        "min": 5.0, "max": 150.0, "default": 35.0, "step": 1.0, "suffix": "gH2O/m²",
        "description": "Mindest-Wassermenge, die sich bergab angesammelt "
                        "haben muss, damit aus einzelnen Regenpixeln ein "
                        "sichtbarer Bach/Fluss wird - steuert, wie dicht "
                        "das Flussnetz insgesamt wirkt."
    }
    MANNING_COEFFICIENT = {
        "min": 0.01, "max": 0.1, "default": 0.03, "step": 0.005,
        "description": "Rauheits-Koeffizient nach der Manning-Gleichung "
                        "(Standard-Formel für Fließgewässer). Höhere Werte "
                        "(raueres Flussbett, z.B. felsig oder bewachsen) "
                        "bremsen die Fließgeschwindigkeit, niedrigere "
                        "Werte (glattes Bett) lassen Wasser schneller "
                        "fließen."
    }
    EROSION_STRENGTH = {
        "min": 0.1, "max": 5.0, "default": 2.5, "step": 0.1,
        "description": "Genereller Multiplikator dafür, wie stark "
                        "fließendes Wasser das Gelände abträgt (Erosion)."
    }
    # transport_capacity = sediment_capacity_factor * flow_speed^2.5
    # (ErosionSedimentationSystem._transport_sediment_optimized) - flow_speed
    # liegt seit der precip_map-Neukalibrierung ([[project-3layer-wind-cfd]])
    # empirisch deutlich höher (Median ~3, Ausreißer bis ~30 m/s statt vorher
    # geringerer Werte). Bei altem default=0.1 überstieg die Transport-
    # kapazität wegen der ^2.5-Potenz praktisch überall bei weitem jede
    # realistische Sediment-Fracht - Sedimentation blieb dadurch fast
    # ausschließlich auf die allerletzten, langsamsten Zellen eines Fließpfads
    # konzentriert (empirisch: nur 0.1-0.7% der Pixel überhaupt ungleich 0,
    # "Sedimentation überall 0"-Report). Bei den tatsächlichen LOD-
    # Iterationszahlen (3-10, siehe _get_lod_iterations) reichte das nicht,
    # damit Material entlang des Fließwegs verteilt abgesetzt wird statt nur
    # am Ende. default auf 0.001 gesenkt (verifiziert gegen einen echten
    # Weather→Water-Testlauf: nach _distribute_sediment_floodplain steigt der
    # Anteil ungleich-Null-Pixel von ~1-4% auf ~34-44%, Maximalwerte bleiben
    # klein/plausibel), min/max entsprechend nach unten verschoben.
    # Live-Test (2026-07-15, echte Karte): selbst default=0.001 ließ Sedimentation
    # nur bei gleichzeitig sehr hoher erosion_strength/settling_velocity sichtbar
    # werden - erst der bereits am unteren Rand des Sliders liegende Wert 0.0001
    # (=min) ergab ein realistisches Bild ("wenn ich den auf 0 habe sieht es
    # realistisch aus"). default direkt auf min gesenkt statt weiter in der Mitte
    # zu kalibrieren.
    SEDIMENT_CAPACITY_FACTOR = {
        "min": 0.0001, "max": 0.1, "default": 0.0001, "step": 0.0001,
        "description": "Wie viel Sediment ein Fluss bei gegebener "
                        "Fließgeschwindigkeit maximal transportieren kann, "
                        "bevor er es ablagert - niedrigere Werte lassen "
                        "Flüsse schneller/mehr Sediment absetzen (mehr "
                        "sichtbare Sedimentation, besonders in "
                        "langsameren Flussabschnitten)."
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
    # settling_velocity ist pro Sediment-Iteration nur der Anteil des
    # ÜBER-Kapazität-Sediments, der tatsächlich absetzt (Rest bleibt in
    # Transport und wandert weiter) - bei 0.01 setzen sich empirisch nur ~0.8%
    # des transportierten Sediments über das gesamte LOD-Iterationsbudget ab
    # (core/water_generator.py ErosionSedimentationSystem._transport_sediment_optimized),
    # der Rest verlässt die Karte praktisch unverändert. 0.08 (statt 0.01)
    # ergibt empirisch ~7x mehr abgesetztes Sediment bei identischem
    # Iterationsbudget, bleibt aber innerhalb des bestehenden Slider-Bereichs.
    SETTLING_VELOCITY = {
        "min": 0.001, "max": 0.1, "default": 0.1, "step": 0.001, "suffix": "m/s",
        "description": "Anteil des überschüssigen (nicht mehr "
                        "transportierbaren) Sediments, der pro "
                        "Berechnungsschritt tatsächlich zu Boden sinkt - "
                        "höhere Werte lassen sichtbar mehr Sedimentation "
                        "entstehen."
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
        "water": ["heightmap", "slopemap", "hardness_map", "rock_map", "precip_map", "temp_map", "wind_map",
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
