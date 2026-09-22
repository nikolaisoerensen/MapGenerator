"""
Path: gui/tabs/terrain_tab.py
Date changed: 24.08.2025

TerrainTab implementiert die Terrain-Generator UI mit vollständiger BaseMapTab-Integration
und direkter Anbindung an den TerrainGenerator aus core/terrain_generator.py. Als Basis-Generator
ohne Dependencies liefert er heightmap, slopemap und shadowmap für alle nachgelagerten Systeme.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton,
    QButtonGroup, QLabel, QCheckBox
)
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtGui import QFont
import logging
import numpy as np
from typing import Dict, Any, Optional

from gui.tabs.base_tab import BaseMapTab
from gui.widgets.widgets import (
    ParameterSlider, RandomSeedButton
)
from gui.config.value_default import (
    TERRAIN, EROSION_FILTER, RIVER_NETWORK, get_parameter_config,
    validate_parameter_set)

class TerrainTab(BaseMapTab):
    """
    Terrain-Generator Tab mit vollständiger BaseMapTab-Integration.
    Implementiert heightmap, slopemap und shadowmap Generation als Basis für alle anderen Generatoren.
    """

    def __init__(self, data_lod_manager, parameter_manager, navigation_manager, shader_manager, generation_orchestrator):

        # Generator-Konfiguration vor BaseMapTab.__init__()
        self.generator_type = "terrain"
        self.required_dependencies = []  # Terrain hat keine Dependencies

        # Terrain-spezifische Attribute (vor super(), da create_parameter_controls
        # während BaseMapTab.setup_ui() darauf zugreift und sie befüllt)
        self.parameter_sliders = {}
        # Abschalthaekchen je Erzeugungsstufe (2026-08-25) - Schluessel
        # ist der Parametername, Wert die QCheckBox. Siehe
        # _build_parameter_group(schalter=...).
        self.stufen_schalter = {}
        self.generation_button = None
        self.progress_bar = None
        self.system_status = None
        self.statistics_display = None
        self.display_mode_group = None
        self.current_display_mode = "height"

        # LOD-System Tracking
        self.current_lod = 1
        self.max_lod = 6

        self.logger = logging.getLogger("TerrainTab")

        # Manager-Integration
        super().__init__(
            data_lod_manager=data_lod_manager,
            parameter_manager=parameter_manager,
            navigation_manager=navigation_manager,
            shader_manager=shader_manager,
            generation_orchestrator=generation_orchestrator
        )

        # Registrierung beim ParameterManager: parameter_sliders sind durch
        # create_parameter_controls() (innerhalb von super().__init__()) bereits
        # befüllt, get_current_parameters() liefert damit sofort die Default-Werte
        # als Startwert des Caches.
        if self.parameter_manager:
            self.parameter_manager.register_tab(self.generator_type, self)

        self.logger.info("TerrainTab initialized")

    def create_parameter_controls(self):
        """
        Erstellt alle Parameter-Controls für Terrain-Generation.
        Implementiert Required-Method von BaseMapTab.
        Stellt bei abgetrenntem Panel-Layout ein neues Layout wieder her,
        damit die Parameter-Erstellung nie leer abbricht.
        """
        if not self.control_panel:
            self.logger.error("Parameter creation skipped: control_panel is None")
            return

        if self.control_panel.layout() is None:
            repaired_layout = QVBoxLayout()
            repaired_layout.setContentsMargins(5, 5, 5, 5)
            repaired_layout.setSpacing(10)
            self.control_panel.setLayout(repaired_layout)
            self.control_panel_content_layout = repaired_layout
            self.logger.info("Control panel layout was detached - reinstalled")

        try:
            # Terrain Parameters GroupBox
            self._create_terrain_parameters()

            self.logger.debug("Parameter controls created successfully")

        except Exception as e:
            self.logger.error(f"Parameter control creation failed: {e}")

    def _create_terrain_parameters(self):
        """
        Erstellt Terrain Parameter Controls, unterteilt in "Shape" (Kartengröße/
        -ausdehnung, Höhe, Redistribution, Seed) und "Noise Detail" (die 4
        Rausch-Parameter, die miteinander interagieren - siehe
        [[project-terrain-review]] 5.3) statt einer einzigen flachen Liste.
        """
        # ENTFERNT (nicht nur gesperrt), Stand 2026-09-22: `map_distance_km`,
        # `amplitude`, `redistribute_power` sowie die komplette "Noise
        # Detail"-Gruppe (`octaves`, `feature_size_m`, `persistence`,
        # `lacunarity`) wirken bei aktiver Weltkarte nicht - siehe
        # gui/config/value_default.stillgelegte_regler() und
        # core/terrain_generator.py._calc_redistribution(): bei aktivem
        # WELTKARTE_AKTIV liefert _weltkarte_heightmap() die Heightmap direkt
        # und die gesamte Noise->Amplitude->Redistribution-Kette (die diese
        # Regler lesen wuerde) wird gar nicht mehr durchlaufen. Vorher waren
        # sie per ParameterSlider.stilllegen() nur deaktiviert+ausgegraut,
        # jetzt fehlen die Widgets ganz. Schaltet WELTKARTE_AKTIV wieder auf
        # False, muessten sie hier wieder eingefuegt werden.
        shape_configs = [
            ("map_size", "Map Size", TERRAIN.MAPSIZE),
            ("map_seed", "Map Seed", TERRAIN.MAP_SEED),
        ]
        noise_detail_configs = []
        # ATEF-Erosionsfilter (SPEZIFIKATION §9). Die Parameter-Keys tragen den
        # Praefix erosion_filter_, damit sie in _apply_erosion_filter() eindeutig
        # von den Reglern der Feld-Erosion (class EROSION) zu unterscheiden sind.
        #
        # Wichtig fuer die Bedienung: dieser Filter erzeugt das Detail. Steht
        # "Detail Octaves" oben hoch, ist der Untergrund schon detailreich und
        # der Filter wirkt kaum noch - gemessen, siehe §9. 1-2 Oktaven sind hier
        # richtig.
        erosion_filter_configs = [
            ("erosion_filter_strength", "Erosion Strength", EROSION_FILTER.STRENGTH),
            ("erosion_filter_gully_size_m", "Gully Size (m)",
             EROSION_FILTER.GULLY_SIZE_M),
            ("erosion_filter_detail", "Gully Reach", EROSION_FILTER.GULLY_REACH),
            ("erosion_filter_gully_weight", "Gullies vs Sharpness",
             EROSION_FILTER.GULLY_VS_SHARPNESS),
            ("erosion_filter_ridge_rounding", "Ridge Rounding",
             EROSION_FILTER.RIDGE_ROUNDING),
            ("erosion_filter_crease_rounding", "Valley Rounding",
             EROSION_FILTER.VALLEY_ROUNDING),
            ("erosion_filter_octaves", "Gully Octaves", EROSION_FILTER.OCTAVES),
        ]

        # Flussnetz-Skelett (SPEZIFIKATION §12). Laeuft NACH dem
        # Erosionsfilter - dessen Ergebnis ist die Flaeche, in die die Taeler
        # geschnitten werden.
        # DIE FUENF WICHTIGSTEN STEHEN SEIT 2026-08-26 IM FLUSSREITER.
        #
        # Nutzervorgabe: *"kannst du die 5 wichtigsten parameter fuer die
        # fluesse herausfinden und mir diese auf die Flussnetzwerktab seite
        # packen?"* Sie stehen dort und NICHT zusaetzlich hier - zwei
        # Widgets fuer denselben Schluessel waeren zwei Wahrheiten
        # (tests/smoke_test_parameter_eindeutig.py wacht darueber). Die
        # Auswahl samt Messreihe steht bei `FLUSS_REGLER` in
        # gui/tabs/river_tab.py.
        #
        # Hier bleiben die zwei, die den LAUF verschieben statt die
        # Landschaft zu formen - einmal einzustellen, nicht zum Gestalten.
        #
        # ENTFERNT (nicht nur gesperrt), Stand 2026-09-22: `river_plateau_
        # flatten`, `river_meander`, `river_divide_blend`, `river_border_
        # outflow` beschreiben Dinge, die es im Weltflussnetz nicht gibt
        # (core/terrain_weltfluesse.py) - sie wurden nur vom alten,
        # abgeloesten `_apply_river_network()`/carve_river_network()-Pfad
        # gelesen, der bei aktivem WELTKARTE_AKTIV nie mehr aufgerufen wird
        # (siehe gui/config/value_default.stillgelegte_regler()). Die
        # verbleibenden zwei (`river_mouth_depth_m`, `river_inherit_cost`)
        # wirken weiterhin - sie gehen in core/terrain_weltfluesse.flussnetz().
        river_configs = [
            ("river_mouth_depth_m", "Mouth Depth (m)",
             RIVER_NETWORK.MOUTH_DEPTH_M),
            ("river_inherit_cost", "Trunk Continuity",
             RIVER_NETWORK.INHERIT_COST),
        ]

        shape_group = self._build_parameter_group("Shape", shape_configs)
        self.control_panel.layout().addWidget(shape_group)

        # "Noise Detail" faellt komplett weg (siehe Kommentar oben bei
        # noise_detail_configs) - eine Gruppe ohne Regler waere eine leere
        # Box mit Titel, kein Nutzen.
        if noise_detail_configs:
            noise_detail_group = self._build_parameter_group(
                "Noise Detail", noise_detail_configs)
            self.control_panel.layout().addWidget(noise_detail_group)

        river_group = self._build_parameter_group(
            "River Network", river_configs,
            schalter=("river_network_aktiv", "Flussnetz und Täler berechnen"))
        self.control_panel.layout().addWidget(river_group)

        erosion_filter_group = self._build_parameter_group(
            "Erosion Filter", erosion_filter_configs,
            schalter=("erosion_filter_aktiv", "Erosionsfilter anwenden"))
        self.control_panel.layout().addWidget(erosion_filter_group)

        # KUESTENTYPEN. Eigene Gruppe ohne Slider - die 27 Archetypen sind
        # ein gemessener Katalog (core/vektor_kueste.py), kein Regler. Was
        # der Nutzer davon einstellen koennen wollte, ist genau das eine:
        # ganz aus, um zu sehen, was sie beitragen.
        kueste_group = self._build_parameter_group(
            "Coastline", [],
            schalter=("kuesten_archetypen_aktiv",
                      "Küstentypen formen (27 Archetypen)"))
        self.control_panel.layout().addWidget(kueste_group)

    def _build_parameter_group(self, title: str, parameter_configs,
                               schalter=None) -> QGroupBox:
        """
        Baut eine QGroupBox mit Slidern für die (key, label, config)-Tupel.

        `schalter` ist ein optionales (key, label)-Paar und erzeugt ganz oben
        in der Gruppe ein Häkchen, mit dem sich die ganze Stufe abschalten
        lässt. Nutzerwunsch 2026-08-25: *"kannst du mir einmal für
        flussnetzwerk und erosionfilter und küstentypen jeweils checkboxen
        einfügen, mit denen ich die effekte immer auch ausschalten kann?"*

        Das Häkchen ist ein ganz normaler Parameter (1.0 an, 0.0 aus) und
        läuft über denselben Weg wie jeder Slider. Es greift damit in die
        ERZEUGUNG ein, nicht in die Anzeige - 2D und 3D zeigen also
        zwangsläufig dasselbe, ohne dass es dafür zwei Wege bräuchte
        (stehende Regel in CLAUDE.md).
        """
        group = QGroupBox(title)
        group.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout = QVBoxLayout()

        if schalter is not None:
            schalter_key, schalter_label = schalter
            box = QCheckBox(schalter_label)
            box.setChecked(True)
            box.setToolTip("Aus: diese Stufe wird bei der Erzeugung ganz "
                           "übersprungen. Zum Vergleichen, wieviel sie zum "
                           "Bild beiträgt.")
            box.toggled.connect(
                lambda an, key=schalter_key: self._on_parameter_changed(
                    key, 1.0 if an else 0.0))
            self.stufen_schalter[schalter_key] = box
            layout.addWidget(box)

        for param_key, label, config in parameter_configs:
            if param_key == "map_seed":
                # Seed Parameter mit RandomSeedButton
                seed_layout = self._create_seed_parameter(param_key, label, config)
                layout.addLayout(seed_layout)
            else:
                # Standard Parameter Slider
                slider = ParameterSlider(
                    label=label,
                    min_val=config["min"],
                    max_val=config["max"],
                    default_val=config["default"],
                    step=config["step"],
                    suffix=config.get("suffix", ""),
                    description=config.get("description", "")
                )

                # Parameter-Change Handler
                slider.valueChanged.connect(
                    lambda value, key=param_key: self._on_parameter_changed(key, value)
                )

                self.parameter_sliders[param_key] = slider
                layout.addWidget(slider)

        group.setLayout(layout)
        return group

    def _create_seed_parameter(self, param_key: str, label: str, config: Dict):
        """Erstellt Seed Parameter mit RandomSeedButton"""
        seed_layout = QHBoxLayout()

        # Seed Slider
        seed_slider = ParameterSlider(
            label=label,
            min_val=config["min"],
            max_val=config["max"],
            default_val=config["default"],
            step=config["step"],
            description=config.get("description", "")
        )

        seed_slider.valueChanged.connect(
            lambda value: self._on_parameter_changed(param_key, value)
        )

        # Random Seed Button
        random_button = RandomSeedButton()
        random_button.seed_generated.connect(
            lambda seed: self._set_random_seed(param_key, seed)
        )

        self.parameter_sliders[param_key] = seed_slider

        seed_layout.addWidget(seed_slider)
        seed_layout.addWidget(random_button)

        return seed_layout

    def create_statistics_controls(self, layout: QVBoxLayout):
        """
        Überschreibt BaseMapTab: befüllt das Statistics-Tab (Spalte 3) mit den
        Terrain-Statistics. Generation Control (Berechnen-Button, Ladebalken,
        System-Status) entfällt hier bewusst - das übernehmen jetzt der globale
        [GENERIEREN]-Button und die Pipeline-Status-Spalte im Shell-Layout.
        """
        stats_group = QGroupBox("Terrain Statistics")
        stats_group.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        stats_layout = QVBoxLayout()

        # Statistics Labels
        self.height_range_label = QLabel("Height Range: No data")
        self.slope_stats_label = QLabel("Slope Statistics: No data")
        self.shadow_coverage_label = QLabel("Shadow Coverage: No data")
        self.performance_label = QLabel("Performance: No data")

        # Styling für Statistics
        for label in [self.height_range_label, self.slope_stats_label,
                     self.shadow_coverage_label, self.performance_label]:
            label.setStyleSheet("font-size: 10px; color: #2c3e50; padding: 2px;")

        stats_layout.addWidget(self.height_range_label)
        stats_layout.addWidget(self.slope_stats_label)
        stats_layout.addWidget(self.shadow_coverage_label)
        stats_layout.addWidget(self.performance_label)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

    def create_visualization_controls(self):
        """
        Erstellt Terrain-spezifische Visualization Controls.
        Überschreibt Optional-Method von BaseMapTab.
        """
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # Display Mode Controls (Height/Slope)
        display_mode_layout = self._create_display_mode_controls()
        controls_layout.addLayout(display_mode_layout)

        controls_widget.setLayout(controls_layout)
        return controls_widget

    def _create_display_mode_controls(self):
        """Erstellt Terrain-Heightmap/Heightmap-Combined/Slope Display Mode Controls"""
        layout = QHBoxLayout()

        # Display Mode Button Group
        self.display_mode_group = QButtonGroup()

        # "Height" zeigte bisher fälschlich get_terrain_data_combined()
        # (das Ergebnis NACH Geology-Tektonik + Water-Erosion/Sedimentation),
        # nicht Terrains eigene, unveränderte Heightmap - siehe
        # [[project-terrain-review]]. Jetzt zwei getrennte Optionen: die reine
        # Terrain-Rohform (Default) und explizit das kombinierte Endergebnis.
        # Beschriftungen 2026-08-13 auf Nutzerwunsch geschärft ("Terrain
        # Heightmap"/"Heightmap Combined" klangen wie zwei Varianten
        # desselben Dings): "Terrain Rohform" macht deutlich, dass hier NUR
        # Terrains eigener Anteil steht (KEIN Perlin-Rauschen - bei aktivem
        # WELTKARTE_AKTIV kommt die Form aus weltfeld(), das Rauschen aus
        # terrain.noise wird dabei gar nicht verwendet), "Heightmap" ohne
        # Zusatz ist das eigentliche Endergebnis, das alle anderen Reiter
        # auch sehen.
        height_radio = QRadioButton("Terrain Rohform")
        height_radio.setChecked(True)
        height_radio.toggled.connect(lambda checked: self._on_display_mode_changed("height", checked))
        self.display_mode_group.addButton(height_radio, 0)

        combined_radio = QRadioButton("Heightmap")
        combined_radio.toggled.connect(lambda checked: self._on_display_mode_changed("combined", checked))
        self.display_mode_group.addButton(combined_radio, 2)

        slope_radio = QRadioButton("Slope")
        slope_radio.toggled.connect(lambda checked: self._on_display_mode_changed("slope", checked))
        self.display_mode_group.addButton(slope_radio, 1)

        # Die neun Kulturregionen, halbtransparent ueber dem Gelaende. Sie
        # gehoeren HIERHER und nicht in einen eigenen Reiter: die Regionen SIND
        # das Gelaende - jede bringt ihre eigene Hoehe, Formgroesse und Rauheit
        # mit, und man muss sehen koennen, ob eine Grenze einem Kamm folgt oder
        # quer durch ein Tal laeuft.
        region_radio = QRadioButton("Regionen")
        region_radio.toggled.connect(lambda checked: self._on_display_mode_changed("regions", checked))
        self.display_mode_group.addButton(region_radio, 3)

        # Kuesten-Archetypen (2026-08-12, Nutzer-Vorgabe: "kann man die
        # Kuestentypen auf der 2D-Karte darstellen? jede Region hat eine
        # Farbe und die Helligkeit von flach (hell) zu steil (dunkel) sind
        # die Kuestentypen") - Regionsfarbe wie beim "Regionen"-Modus,
        # Helligkeit kodiert den hoehe_faktor des zugeordneten Archetyps.
        kuesten_radio = QRadioButton("Kuestentypen")
        kuesten_radio.toggled.connect(lambda checked: self._on_display_mode_changed("kuestentypen", checked))
        self.display_mode_group.addButton(kuesten_radio, 4)

        # Spielkarten-Zerlegung (2026-08-13, docs/OFFENE_PUNKTE.md 5.15) - die
        # neun Vielecke, in die die Welt fuer Regionalansicht und spaeteren
        # Export zerschnitten wird. Gehoert in DIESEN Reiter, weil der
        # Zuschnitt aus dem Gelaende folgt (Landmasse, Kuestensee), nicht aus
        # den Siedlungen.
        spielkarten_radio = QRadioButton("Spielkarten")
        spielkarten_radio.toggled.connect(
            lambda checked: self._on_display_mode_changed("spielkarten", checked))
        self.display_mode_group.addButton(spielkarten_radio, 5)

        # HOEHENFAKTOR UND VORONOI (2026-08-26, Nutzerwunsch: *"kannst du mir
        # die voronoiansicht als erstes bauen? ich will den hoehenfaktor sehen
        # koennen (3d und 2D)"*).
        #
        # "Hoehenfaktor" zeigt je Pixel die gemessene Hoehe des Hinterlands
        # seines Kuestengebiets in METERN - die Groesse, aus der
        # `kuestengebiete()` sein Hoehendelta bildet. Sie stammt aus der
        # Zweipunktmethode des Nutzers (Mittel des gemessenen Kuestenprofils
        # im Band 400-700 m), NICHT aus dem Katalogfaktor `hoehe_faktor`:
        # der beschreibt das Ufer, und in vier von neun Regionen dreht sich
        # die Reihenfolge dadurch um.
        #
        # "Voronoi" zeigt die Zellen, aus denen die Gebiete gewachsen sind -
        # dieselben, die auch die Regionen bilden.
        hoehenfaktor_radio = QRadioButton("Hoehenfaktor")
        hoehenfaktor_radio.setToolTip(
            "Gemessene Hinterlandhoehe je Kuestengebiet, in Metern "
            "(Profilmittel 400-700 m landeinwaerts). Grau = alpiner "
            "Sonderfall ohne Kuestensaat.")
        hoehenfaktor_radio.toggled.connect(
            lambda an: self._on_display_mode_changed("hinterland_height", an))
        self.display_mode_group.addButton(hoehenfaktor_radio, 6)

        voronoi_radio = QRadioButton("Voronoi")
        voronoi_radio.setToolTip(
            "Die Zellen, aus denen Regionen und Kuestengebiete wachsen. "
            "Die Farbe ist nur zur Unterscheidung, der Zahlenwert bedeutet "
            "nichts.")
        voronoi_radio.toggled.connect(
            lambda an: self._on_display_mode_changed("voronoi_map", an))
        self.display_mode_group.addButton(voronoi_radio, 7)

        layout.addWidget(height_radio)
        layout.addWidget(combined_radio)
        layout.addWidget(slope_radio)
        layout.addWidget(region_radio)
        layout.addWidget(kuesten_radio)
        layout.addWidget(spielkarten_radio)
        # BUGFIX (Issue #31, 2026-09-17): hoehenfaktor_radio/voronoi_radio waren
        # oben vollstaendig verdrahtet (Titel, Tooltip, Signal, Button-Group),
        # fehlten hier aber im sichtbaren Layout - dadurch fuer den Nutzer
        # unerreichbar, obwohl Daten (core/terrain_generator.py:
        # "hinterland_height"/"voronoi_map"), 2D-Farbskala
        # (gui/config/gui_default.py layer_ranges) und 3D-Overlay-Registrierung
        # (base_tab.py _LAYER_NAME_MAP_3D/_LAYER_SELECTION_KEYS_3D,
        # map_display_3d.py _render_overlay) bereits vorhanden waren.
        layout.addWidget(hoehenfaktor_radio)
        layout.addWidget(voronoi_radio)

        return layout

    # =============================================================================
    # EVENT HANDLERS
    # =============================================================================

    def _on_parameter_changed(self, param_name: str, value: float):
        """Handler für Parameter-Änderungen"""
        try:
            # Parameter an ParameterManager weiterleiten
            if self.parameter_manager:
                self.parameter_ui_changed.emit(self.generator_type, param_name, value)

            # Cross-Parameter Validation
            self._validate_parameter_constraints()

            self.logger.debug(f"Parameter changed: {param_name} = {value}")

        except Exception as e:
            self.logger.error(f"Parameter change handling failed: {e}")

    def _set_random_seed(self, param_key: str, seed_value: int):
        """Setzt zufälligen Seed-Wert"""
        try:
            if param_key in self.parameter_sliders:
                self.parameter_sliders[param_key].setValue(seed_value)
                self._on_parameter_changed(param_key, seed_value)

        except Exception as e:
            self.logger.error(f"Random seed setting failed: {e}")

    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Sammelt die aktuellen Werte aller Terrain-Parameter-Slider.
        Wird vom ParameterManager als zentrale Quelle für die Terrain-Parameter
        genutzt (register_tab()/get_tab_parameters() rufen diese Methode auf,
        siehe managers/parameter_manager.py).
        """
        parameters = {}
        for param_name, slider in self.parameter_sliders.items():
            parameters[param_name] = slider.getValue()

        # DIE EINSTELLUNGEN DES REGIONSREITERS MITNEHMEN (2026-08-26).
        #
        # Die Generierung holt sich `get_tab_parameters(self.generator_type)`,
        # also NUR den eigenen Satz. Ohne diese Zeilen kaeme aus dem
        # Regionsreiter nichts an: man stellt eine Region ein, drueckt
        # "Generieren" - und bekommt dieselbe Karte. Genau das war bis heute
        # der Fall, ohne Fehlermeldung.
        if self.parameter_manager is not None:
            for beitragend in ("region", "kontinent"):
                try:
                    teil = self.parameter_manager.get_tab_parameters(beitragend)
                    if teil:
                        parameters.update(teil)
                except Exception as fehler:              # pragma: no cover
                    self.logger.debug("Parameter von %s nicht verfuegbar: %s",
                                      beitragend, fehler)

        # Die Abschalthaekchen als 1.0/0.0 - derselbe Weg wie jeder Slider,
        # damit der ParameterManager nichts Neues lernen muss.
        for param_name, box in self.stufen_schalter.items():
            parameters[param_name] = 1.0 if box.isChecked() else 0.0
        return parameters

    def _validate_parameter_constraints(self):
        """Validiert Cross-Parameter Constraints"""
        try:
            parameters = self.get_current_parameters()

            # Validation über value_default.py
            is_valid, warnings, errors = validate_parameter_set("terrain", parameters)

            # UI-Status Updates
            if errors:
                if self.system_status:
                    self.system_status.set_error(f"Parameter errors: {', '.join(errors)}")
            elif warnings:
                if self.system_status:
                    self.system_status.set_warning(f"Warnings: {', '.join(warnings)}")
            else:
                if self.system_status:
                    self.system_status.set_success("Parameters valid")

        except Exception as e:
            self.logger.error(f"Parameter validation failed: {e}")

    def generate(self):
        """
        Überschreibt BaseMapTab: Dependency-Check + Parameter-Validation vor
        der eigentlichen Generation. Wird jetzt ausschließlich über den
        globalen [GENERIEREN]-Button im Shell-Footer ausgelöst (kein eigener
        Berechnen-Button mehr im Parameter-Panel).
        """
        try:
            # Dependency Check (Terrain hat keine Dependencies)
            if not self.check_input_dependencies():
                self.logger.warning("Input dependencies not met")
                return

            # Parameter Validation
            self._validate_parameter_constraints()

            super().generate()

            self.logger.info("Terrain generation requested")

        except Exception as e:
            self.logger.error(f"Generation request failed: {e}")
            if self.system_status:
                self.system_status.set_error(f"Generation failed: {e}")

    def _on_display_mode_changed(self, mode: str, checked: bool):
        """Handler für Display Mode Changes"""
        if checked:
            self.current_display_mode = mode
            self.update_display_mode()
            self.logger.debug(f"Display mode changed to: {mode}")

    # =============================================================================
    # DISPLAY UPDATE SYSTEM
    # =============================================================================

    def update_display_mode(self):
        """
        Überschreibt BaseMapTab Display-Update für Terrain-spezifische Modi.
        Implementiert Height/Slope Display-Switching.
        """
        try:
            if not self.data_lod_manager:
                return

            current_display = self.get_current_display()
            if not current_display:
                return

            # Daten basierend auf Display-Mode holen
            if self.current_display_mode == "height":
                # Terrains eigene, unveränderte Heightmap - wird von Geology/
                # Water nie mutiert (siehe get_terrain_data_combined() für
                # das kombinierte Endergebnis, separat unter "combined").
                # data_type "heightmap" (reine Terrain-Rohform) vs.
                # "heightmap_combined" (Endergebnis) ist das Fast-Path-Signal
                # für base_tab.py's 3D-Mesh-Aufbau: nur "heightmap_combined"
                # wird direkt fürs Mesh verwendet, "heightmap" löst dort immer
                # einen Re-Fetch der kombinierten Karte aus, damit die rohe
                # Terrain-Form nicht fälschlich das 3D-Mesh verformt.
                data = self.data_lod_manager.get_terrain_data("heightmap")
                data_type = "heightmap"
                display_data = data
            elif self.current_display_mode == "combined":
                # Finales Ergebnis NACH Geology-Tektonik + Water-Erosion/
                # -Sedimentation - siehe DataLODManager.get_terrain_data_combined().
                data = self.data_lod_manager.get_terrain_data_combined("heightmap")
                data_type = "heightmap_combined"
                display_data = data
            elif self.current_display_mode == "slope":
                data = self.data_lod_manager.get_terrain_data("slopemap")
                data_type = "slopemap"
                # slopemap ist (H,W,2) dx/dy-Gradient - MapDisplay2D._render_
                # slopemap() faerbt es jetzt direkt als Kompass-Farbrad
                # (Hangausrichtung=Hue, Steilheit=Saettigung, siehe
                # compute_slope_compass_rgb()), braucht also die rohen
                # Vektor-Daten unveraendert statt einer vorab auf Grad
                # reduzierten Magnitude (die die Richtungsinformation verwarf).
                display_data = data if data is not None and hasattr(data, 'shape') and len(data.shape) == 3 \
                    else None
            elif self.current_display_mode == "regions":
                # region_map UND heightmap zusammen: der Renderer faerbt nur
                # Land ein und braucht dafuer die Hoehen. Beide in einem
                # payload statt in zwei Zugriffen - sonst haenge der Renderer
                # still an der Reihenfolge der Display-Updates.
                data = self.data_lod_manager.get_terrain_data("region_map")
                hoehe = self.data_lod_manager.get_terrain_data("heightmap")
                data_type = "region_map"
                display_data = ({"regionen": data, "heightmap": hoehe}
                                if data is not None and hoehe is not None else None)
            elif self.current_display_mode in ("hinterland_height", "voronoi_map"):
                # Gewoehnliche Skalarkarten - derselbe Weg wie "slope", also
                # in 2D UND 3D ohne Sonderbehandlung (die Registereintraege
                # dafuer stehen in base_tab._LAYER_NAME_MAP_3D).
                data = self.data_lod_manager.get_terrain_data(
                    self.current_display_mode)
                data_type = self.current_display_mode
                display_data = data
            elif self.current_display_mode == "spielkarten":
                # Rohdaten-Payload wie bei "Regionen"/"Kuestentypen" - der
                # Renderer braucht die Hoehen, um Land von Meer zu trennen.
                data = self.data_lod_manager.get_terrain_data("spielkarte")
                hoehe = self.data_lod_manager.get_terrain_data("heightmap")
                data_type = "spielkarte"
                display_data = ({"spielkarte": data, "heightmap": hoehe}
                                if data is not None and hoehe is not None else None)
            elif self.current_display_mode == "kuestentypen":
                # Regionsfarbe (wie "Regionen") x Helligkeit nach Archetyp-
                # Steilheit (flach=hell, steil=dunkel) - siehe
                # MapDisplay2D._render_kuesten_archetypen().
                data = self.data_lod_manager.get_terrain_data("region_map")
                hoehe = self.data_lod_manager.get_terrain_data("heightmap")
                archetyp = self.data_lod_manager.get_terrain_data("kuesten_archetyp")
                staerke = self.data_lod_manager.get_terrain_data("kuesten_staerke")
                data_type = "kuesten_archetyp"
                display_data = ({"regionen": data, "heightmap": hoehe,
                                 "kuesten_archetyp": archetyp, "kuesten_staerke": staerke}
                                if data is not None and hoehe is not None and archetyp is not None
                                else None)
            else:
                return

            # Display Update mit Change-Detection
            if data is not None and display_data is not None and hasattr(current_display, 'update_display'):
                display_id = f"TerrainTab_{self.current_view}_{data_type}"

                if hasattr(self.data_lod_manager, 'display_update_manager'):
                    needs_update = self.data_lod_manager.display_update_manager.needs_update(
                        display_id, data, data_type
                    )

                    if needs_update:
                        self._push_data_to_current_display(display_data, data_type)
                        self.data_lod_manager.display_update_manager.mark_updated(
                            display_id, data, data_type
                        )

                        # Statistics Update
                        self._update_statistics(data, data_type)
                else:
                    # Fallback ohne Change-Detection
                    self._push_data_to_current_display(display_data, data_type)
                    self._update_statistics(data, data_type)

        except Exception as e:
            self.logger.debug(f"Display mode update failed: {e}")

    def _update_statistics(self, data, data_type: str):
        """Aktualisiert Terrain Statistics basierend auf aktuellen Daten"""
        try:
            if data_type in ("heightmap", "heightmap_combined") and hasattr(data, 'shape'):
                # Height Statistics
                height_min = float(np.min(data))
                height_max = float(np.max(data))
                height_mean = float(np.mean(data))
                height_std = float(np.std(data))

                self.height_range_label.setText(
                    f"Height Range: {height_min:.1f}m - {height_max:.1f}m "
                    f"(Mean: {height_mean:.1f}m ± {height_std:.1f}m)"
                )

                # Performance Metrics
                data_size_mb = (data.nbytes / (1024 * 1024))
                self.performance_label.setText(
                    f"Performance: {data.shape[0]}×{data.shape[1]} "
                    f"({data_size_mb:.1f}MB)"
                )

            elif data_type == "slopemap" and hasattr(data, 'shape') and len(data.shape) == 3:
                # Slope Statistics (data ist (H,W,2) für dx/dy Gradienten)
                slope_magnitude = np.sqrt(data[:,:,0]**2 + data[:,:,1]**2)
                max_slope = float(np.max(slope_magnitude))
                mean_slope = float(np.mean(slope_magnitude))

                # Konvertierung zu Degrees
                max_slope_deg = np.degrees(np.arctan(max_slope))
                mean_slope_deg = np.degrees(np.arctan(mean_slope))

                self.slope_stats_label.setText(
                    f"Slope Statistics: Max {max_slope_deg:.1f}°, "
                    f"Mean {mean_slope_deg:.1f}°"
                )

            elif data_type == "region_map" and hasattr(data, 'shape'):
                # LANDflaeche je Region, nicht Gesamtflaeche. Die Zuordnung gilt
                # auch auf offener See - Clonagh kaeme sonst auf 66 km2, von
                # denen 51 Ozean sind. Nur die Landzahl sagt etwas darueber aus,
                # wieviele Siedlungen eine Kultur tragen kann.
                from core.terrain_weltkarte import alle_regionen
                heightmap = self.data_lod_manager.get_terrain_data("heightmap")
                if heightmap is not None:
                    km_pro_pixel = (self.data_lod_manager.get_map_distance_km()
                                    / float(data.shape[0]))
                    land = np.asarray(heightmap) > 0.0
                    zeilen = []
                    for i, (_z, _s, region) in enumerate(alle_regionen()):
                        flaeche = float((land & (np.asarray(data) == i)).sum()) \
                            * km_pro_pixel * km_pro_pixel
                        zeilen.append("%s %.1f km²" % (region["volk"], flaeche))
                    self.height_range_label.setText("Landflaeche: "
                                                    + ", ".join(zeilen))

            # Shadow Coverage (falls verfügbar)
            shadow_data = self.data_lod_manager.get_terrain_data("shadowmap")
            if shadow_data is not None and hasattr(shadow_data, 'shape'):
                if len(shadow_data.shape) == 3:  # (H,W,7) für 7 Sonnenwinkel
                    shadow_min = float(np.min(shadow_data))
                    shadow_max = float(np.max(shadow_data))
                    shadow_mean = float(np.mean(shadow_data))

                    self.shadow_coverage_label.setText(
                        f"Shadow Coverage: {shadow_min:.2f} - {shadow_max:.2f} "
                        f"(Mean: {shadow_mean:.2f})"
                    )

        except Exception as e:
            self.logger.debug(f"Statistics update failed: {e}")

    # =============================================================================
    # GENERATION PROGRESS TRACKING
    # =============================================================================

    @pyqtSlot(int, str)
    def on_generation_progress(self, progress: int, message: str):
        """
        Überschreibt BaseMapTab Progress Handler für LOD-spezifisches Progress.
        """
        if not self.generation_active:
            return

        try:
            # LOD-Level aus Progress ableiten (0-100 Progress → LOD 1-6)
            lod_level = max(1, min(6, int((progress / 100) * self.max_lod) + 1))

            if self.progress_bar:
                # LOD-spezifischer Progress Text
                phase_text = "Heightmap Generation"
                if progress > 60:
                    phase_text = "Shadow Calculation"
                elif progress > 30:
                    phase_text = "Slope Calculation"

                self.progress_bar.set_lod_progress(lod_level, self.max_lod, phase_text)

            # Generation Button Status
            if self.generation_button:
                self.generation_button.set_loading(True)

            # System Status Update
            if self.system_status:
                self.system_status.set_pending(f"Generating LOD {lod_level}/{self.max_lod} ({progress}%)")

            self.current_lod = lod_level

        except Exception as e:
            self.logger.error(f"Progress tracking failed: {e}")

    @pyqtSlot(str, dict)
    def on_generation_completed(self, result_id: str, result_data: dict):
        """
        Überschreibt BaseMapTab Completion Handler für Terrain-spezifische Completion.
        """
        generator_type = result_data.get("generator_type", "")
        success = result_data.get("success", False)

        if generator_type != self.generator_type:
            return

        try:
            # Progress Bar Reset
            if self.progress_bar:
                if success:
                    self.progress_bar.set_progress(100, "Completed", "")
                else:
                    self.progress_bar.reset()

            # Generation Button aktivieren
            if self.generation_button:
                self.generation_button.set_loading(False)

            # System Status
            if self.system_status:
                if success:
                    self.system_status.set_success(f"Generation completed (LOD {self.max_lod})")
                else:
                    self.system_status.set_error("Generation failed")

            # Display Update nach Completion
            if success:
                self.update_display_mode()

                # Map-Size Sync zu anderen Tabs
                if self.data_lod_manager and hasattr(self.data_lod_manager, 'sync_map_size'):
                    current_map_size = self.parameter_sliders.get('map_size', {}).getValue()
                    if current_map_size:
                        self.data_lod_manager.sync_map_size(int(current_map_size))

            # Parent-Class Completion Handler
            super().on_generation_completed(result_id, result_data)

        except Exception as e:
            self.logger.error(f"Generation completion handling failed: {e}")

    # =============================================================================
    # PARAMETER SYNCHRONISATION
    # =============================================================================

    def update_parameter_ui(self, param_name: str, value):
        """
        Überschreibt BaseMapTab Parameter-UI Update für Terrain-Parameter.
        Synchronisiert UI-Controls mit ParameterManager-Updates.
        """
        try:
            if param_name in self.parameter_sliders:
                # Update ohne Signal-Emission (verhindert Loop)
                slider = self.parameter_sliders[param_name]
                slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(False)

                self.logger.debug(f"Parameter UI updated: {param_name} = {value}")

            # Cross-Parameter Validation nach Update
            self._validate_parameter_constraints()

        except Exception as e:
            self.logger.error(f"Parameter UI update failed: {e}")

    # =============================================================================
    # DEPENDENCY SYSTEM (überschrieben, da Terrain keine Dependencies hat)
    # =============================================================================

    def check_input_dependencies(self) -> bool:
        """
        Überschreibt BaseMapTab Dependency Check.
        Terrain hat keine Input-Dependencies, gibt immer True zurück.
        """
        return True  # Terrain ist Basis-Generator ohne Dependencies

    # =============================================================================
    # RESOURCE MANAGEMENT
    # =============================================================================

    def cleanup_resources(self):
        """
        Erweitert BaseMapTab Cleanup für Terrain-spezifische Resources.
        """
        try:
            self.logger.debug("Cleaning up terrain-specific resources")

            # Terrain-spezifische Cleanup
            self.parameter_sliders.clear()
            self.current_lod = 1

            # Progress Reset
            if self.progress_bar:
                self.progress_bar.reset()

            # Status Reset
            if self.system_status:
                self.system_status.set_success("Ready")

            # Parent Cleanup
            super().cleanup_resources()

        except Exception as e:
            self.logger.error(f"Terrain cleanup failed: {e}")


def terrain_tab():
    """
    Factory-Funktion für TerrainTab-Erstellung.
    Wird von der Main-Application für Tab-Initialisierung verwendet.
    """
    return TerrainTab