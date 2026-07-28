"""
Path: gui/tabs/weather_tab.py

WeatherTab implementiert die Weather-Generator UI mit vollständiger BaseMapTab-Integration
und direkter Anbindung an den WeatherSystemGenerator aus core/weather_generator.py. Als von
Terrain abhängiger Generator (heightmap_combined, shadowmap) liefert er wind_map, temp_map,
precip_map und humid_map für Water und alle nachgelagerten Systeme.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton,
    QButtonGroup, QLabel, QCheckBox
)
from PyQt6.QtCore import pyqtSlot, QTimer, Qt
from PyQt6.QtGui import QFont
import logging
import numpy as np
from typing import Dict, Any

from gui.tabs.base_tab import BaseMapTab
from gui.widgets.widgets import ParameterSlider, StatusIndicator
from gui.config.value_default import get_parameter_config
from core.weather_generator import WeatherSystemGenerator


class WeatherTab(BaseMapTab):
    """
    Weather-Generator Tab mit vollständiger BaseMapTab-Integration.
    Implementiert wind_map/temp_map/precip_map/humid_map Generation auf Basis der
    Terrain-Daten (heightmap_combined, shadowmap).
    """

    def __init__(self, data_lod_manager, parameter_manager, navigation_manager, shader_manager, generation_orchestrator):

        # Generator-Konfiguration vor BaseMapTab.__init__()
        self.generator_type = "weather"
        self.required_dependencies = ["heightmap", "shadowmap"]

        # Weather-spezifische Attribute (vor super(), da create_parameter_controls
        # und create_visualization_controls während BaseMapTab.setup_ui() darauf
        # zugreifen und sie befüllen)
        self.parameter_sliders = {}
        self.parameter_checkboxes = {}
        self.climate_stats = None
        self.dependency_status = None
        self.gpu_status = None
        self.display_mode_group = None
        self.current_display_mode = "height"
        # Schicht-Diagnose (Boden/Mittel/Hoch, siehe AtmosphereLayers in
        # core/weather_generator.py) - Nutzer-Wunsch aus der Weather-Rework-
        # Diskussion: die Mittel-/Hochschicht-Daten werden bereits berechnet,
        # hatten aber bisher KEINEN Konsumenten irgendwo im Code.
        self.layer_group = None
        self.current_layer_index = 0  # 0=GROUND, 1=MID, 2=HIGH

        # Saisonale Monats-Animation (siehe update_display_mode()) - Platzhalter
        # vor super().__init__(), da QTimer(self) einen bereits konstruierten
        # QWidget-Unterbau braucht (self ist an dieser Stelle noch keine gültige
        # QObject-Instanz).
        self._current_month_index = 0
        self._month_cycle_timer = None

        # Live-Klimatologie-Vorschau (Nutzer-Wunsch 2026-07-24) - zeigt die
        # tatsächlichen 6 saisonalen Temperatur-/Feuchte-Werte für den
        # aktuellen Latitude/Offset-Slider-Stand, noch VOR jeder Generierung.
        # Nutzt dieselbe _climate_baseline()-Formel wie die echte Simulation
        # (core/weather_generator.py) - eine leichte, seedlose Instanz reicht,
        # da _climate_baseline() nur die Klasse-Tabelle + Breitengrad/Monat
        # braucht, keinen echten Generierungslauf.
        self._climatology_preview_generator = WeatherSystemGenerator()
        self.climatology_preview_label = None

        self.logger = logging.getLogger("WeatherTab")

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

        # Initialer Dependency-Check für die Status-Anzeige
        self.check_input_dependencies()

        # Timer für die animierte Monats-Umschaltung (1x/Sekunde) - Start/Stop-
        # Gating analog map_display_3d.py's animation_timer-Muster (aktiv nur
        # solange die ausgewählte Karte Monatsdaten hat), siehe
        # _update_month_cycle_timer_state()/_on_month_cycle_tick().
        self._month_cycle_timer = QTimer(self)
        self._month_cycle_timer.setInterval(1000)
        self._month_cycle_timer.timeout.connect(self._on_month_cycle_tick)

        self.logger.info("WeatherTab initialized")

    def create_parameter_controls(self):
        """
        Erstellt alle Parameter-Controls für Weather-Generation.
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
            self._create_temperature_parameters()
            self._create_wind_parameters()
            self._create_humidity_parameters()
            self._create_dependency_status()
            self._create_gpu_status()

            self.logger.debug("Parameter controls created successfully")

        except Exception as e:
            self.logger.error(f"Parameter control creation failed: {e}")

    def _create_temperature_parameters(self):
        """Erstellt Temperature-System Parameter Controls"""
        temp_group = QGroupBox("Temperature System")
        temp_group.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        temp_layout = QVBoxLayout()

        for param_key in ("ground_temp_offset", "sun_relevance_factor", "altitude_cooling"):
            config = get_parameter_config("weather", param_key)

            slider = ParameterSlider(
                label=param_key.replace("_", " ").title(),
                min_val=config["min"],
                max_val=config["max"],
                default_val=config["default"],
                step=config.get("step", 1),
                suffix=config.get("suffix", ""),
                description=config.get("description", "")
            )
            slider.valueChanged.connect(
                lambda value, key=param_key: self._on_parameter_changed(key, value)
            )
            self.parameter_sliders[param_key] = slider
            temp_layout.addWidget(slider)

        temp_group.setLayout(temp_layout)
        self.control_panel.layout().addWidget(temp_group)

    def _create_wind_parameters(self):
        """Erstellt Wind-System Parameter Controls"""
        wind_group = QGroupBox("Wind System")
        wind_group.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        wind_layout = QVBoxLayout()

        for param_key in ("thermic_effect", "wind_speed_factor", "terrain_factor",
                          "prevailing_wind_direction", "turbulence_strength"):
            config = get_parameter_config("weather", param_key)

            slider = ParameterSlider(
                label=param_key.replace("_", " ").title(),
                min_val=config["min"],
                max_val=config["max"],
                default_val=config["default"],
                step=config.get("step", 0.1),
                suffix=config.get("suffix", ""),
                description=config.get("description", "")
            )
            slider.valueChanged.connect(
                lambda value, key=param_key: self._on_parameter_changed(key, value)
            )
            self.parameter_sliders[param_key] = slider
            wind_layout.addWidget(slider)

        # Checkbox statt Slider (Weather-Rework Punkt A) - Toggle statt reinem
        # Rewrite, damit sich der thermisch gekoppelte Druck-Pfad bei Bedarf
        # ohne Code-Änderung wieder abschalten lässt (siehe
        # WEATHER.THERMAL_PRESSURE_COUPLING-Beschreibung).
        pressure_config = get_parameter_config("weather", "thermal_pressure_coupling")
        pressure_checkbox = QCheckBox("Thermal Pressure Coupling (Experimental)")
        pressure_checkbox.setChecked(bool(pressure_config.get("default", True)))
        pressure_checkbox.setToolTip(pressure_config.get("description", ""))
        pressure_checkbox.toggled.connect(
            lambda checked: self._on_parameter_changed("thermal_pressure_coupling", checked)
        )
        self.parameter_checkboxes["thermal_pressure_coupling"] = pressure_checkbox
        wind_layout.addWidget(pressure_checkbox)

        wind_group.setLayout(wind_layout)
        self.control_panel.layout().addWidget(wind_group)

    def _create_humidity_parameters(self):
        """Erstellt Location/Climate-Offset Parameter Controls (Breitengrad,
        Temperatur-/Feuchte-Eintritts-Offsets - Reihenfolge Latitude zuerst,
        dann die beiden additiven Offset-Regler, Nutzer-Vorgabe 2026-07-24).
        Longitude-Regler entfernt (Nutzer-Entscheidung: der Effekt ist real,
        aber so schwach - nur Tageszeit-Feinverschiebung der Sonnenwinkel-
        Samples, siehe core/terrain_generator.py calculate_solar_position() -
        dass er keinen eigenen Slider braucht. map_longitude bleibt intern
        als fester Konstante bestehen (WEATHER.MAP_LONGITUDE["default"]=15,
        der "neutrale" Zeitzonen-Wert) - core/weather_generator.py und
        gui/tabs/base_tab.py lesen den Parameter ohnehin über
        `.get('map_longitude', WEATHER.MAP_LONGITUDE["default"])`, ein
        fehlender Slider fällt also automatisch auf genau diese Konstante
        zurück, ohne Code-Änderung an der Leseseite."""
        humidity_group = QGroupBox("Location & Climate Offset")
        humidity_group.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        humidity_layout = QVBoxLayout()

        for param_key in ("map_latitude", "air_temp_entry", "air_humidity_entry"):
            config = get_parameter_config("weather", param_key)

            slider = ParameterSlider(
                label=param_key.replace("_", " ").title(),
                min_val=config["min"],
                max_val=config["max"],
                default_val=config["default"],
                step=config.get("step", 1),
                suffix=config.get("suffix", ""),
                description=config.get("description", "")
            )
            slider.valueChanged.connect(
                lambda value, key=param_key: self._on_parameter_changed(key, value)
            )
            self.parameter_sliders[param_key] = slider
            humidity_layout.addWidget(slider)

        # Live-Klimatologie-Vorschau (Nutzer-Vorgabe 2026-07-24) - zeigt fuer
        # den aktuellen Latitude-/Offset-Stand die 6 saisonalen Temperatur-/
        # Feuchte-Werte (S1=Jan/Feb .. S6=Nov/Dez), aktualisiert sich live bei
        # jeder Aenderung an einem der 3 Slider oben (siehe
        # _on_parameter_changed()/_update_climatology_preview()).
        self.climatology_preview_label = QLabel("")
        self.climatology_preview_label.setWordWrap(True)
        self.climatology_preview_label.setTextFormat(Qt.TextFormat.RichText)
        humidity_layout.addWidget(self.climatology_preview_label)

        humidity_group.setLayout(humidity_layout)
        self.control_panel.layout().addWidget(humidity_group)
        self._update_climatology_preview()

    def _create_dependency_status(self):
        """Erstellt Dependency-Status-Anzeige (Verfügbarkeit der Terrain-Inputs)"""
        self.dependency_status = StatusIndicator("Weather Dependencies")
        self.control_panel.layout().addWidget(self.dependency_status)

    def _create_gpu_status(self):
        """
        Erstellt GPU-Status-Anzeige. Korrektur (Weather-Review-Runde,
        docs/session_review_2026-07-22_weather.md): der ShaderManager bietet
        durchaus 4 Weather-GPU-Shader-Operationen an (Dispatch-Tabelle in
        gui/OldManagers/shader_manager.py) - nur 3 davon werden im
        Normalbetrieb praktisch nie erreicht, weil der gekoppelte 3-Schicht-
        Loop (_run_coupled_atmosphere_simulation) sie nur im Exception-
        Fallback-Pfad aufruft. Die Anzeige hier zeigt trotzdem ehrlich, ob
        GPU-Beschleunigung grundsätzlich verfügbar ist.
        """
        self.gpu_status = StatusIndicator("GPU Status")
        if self.shader_manager and getattr(self.shader_manager, 'gpu_available', False):
            self.gpu_status.set_success("GPU available")
        else:
            self.gpu_status.set_warning("Using CPU fallback")
        self.control_panel.layout().addWidget(self.gpu_status)

    def create_statistics_controls(self, layout: QVBoxLayout):
        """
        Überschreibt BaseMapTab: befüllt das Statistics-Tab (Spalte 3) mit den
        Climate-Statistics (Parameter-Preview + Climate-Classification + Ergebnissen).
        """
        self.climate_stats = ClimateStatisticsWidget()
        layout.addWidget(self.climate_stats)

    def create_visualization_controls(self):
        """
        Erstellt Weather-spezifische Visualization Controls.
        Überschreibt Optional-Method von BaseMapTab.
        """
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)

        display_mode_layout = self._create_display_mode_controls()
        controls_layout.addLayout(display_mode_layout)
        controls_layout.addWidget(self._create_vertical_separator())
        self.layer_controls_widget = QWidget()
        self.layer_controls_widget.setLayout(self._create_layer_controls())
        # "Height" ist der initial gewählte Modus (siehe _create_display_mode_controls()) -
        # dessen toggled-Signal feuerte bereits VOR diesem Zeitpunkt (das Widget
        # existierte damals noch nicht, siehe hasattr-Guard in
        # _on_display_mode_changed()), deshalb hier explizit den Ausgangszustand setzen.
        self.layer_controls_widget.setVisible(self.current_display_mode != "height")
        controls_layout.addWidget(self.layer_controls_widget)

        controls_widget.setLayout(controls_layout)
        return controls_widget

    def _create_display_mode_controls(self):
        """Erstellt Height/Temperature/Precipitation/Humidity/Wind Display Mode Controls"""
        layout = QHBoxLayout()

        self.display_mode_group = QButtonGroup()

        modes = [
            ("height", "Height", 0),
            ("temp_map", "Temperature", 1),
            ("precip_map", "Precipitation", 2),
            ("humid_map", "Humidity", 3),
            ("wind_map", "Wind", 4),
        ]

        for mode_key, label, button_id in modes:
            radio = QRadioButton(label)
            if button_id == 0:
                radio.setChecked(True)
            radio.toggled.connect(lambda checked, key=mode_key: self._on_display_mode_changed(key, checked))
            self.display_mode_group.addButton(radio, button_id)
            layout.addWidget(radio)

        return layout

    def _create_layer_controls(self):
        """
        Schicht-Umschalter (Boden/Mittel/Hoch) für Temperature/Humidity/Wind -
        macht die bisher ungenutzten Mittel-/Hochschicht-Daten der 3-Schicht-
        Atmosphäre-Simulation sichtbar. Wirkt NICHT auf "Height" oder
        "Precipitation" (Niederschlag ist eine über alle Schichten
        akkumulierte Größe, keine Pro-Schicht-Größe).
        """
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Layer:"))

        self.layer_group = QButtonGroup()
        for index, label in enumerate(("Ground", "Mid", "High")):
            radio = QRadioButton(label)
            if index == 0:
                radio.setChecked(True)
            radio.toggled.connect(lambda checked, i=index: self._on_layer_changed(i, checked))
            self.layer_group.addButton(radio, index)
            layout.addWidget(radio)

        return layout

    def _create_vertical_separator(self):
        """Erstellt vertikalen Separator für UI-Layout"""
        separator = QWidget()
        separator.setFixedWidth(1)
        separator.setStyleSheet("background-color: #bdc3c7;")
        return separator

    # =============================================================================
    # EVENT HANDLERS
    # =============================================================================

    def _on_parameter_changed(self, param_name: str, value: float):
        """Handler für Parameter-Änderungen"""
        try:
            if self.parameter_manager:
                self.parameter_ui_changed.emit(self.generator_type, param_name, value)

            if self.climate_stats:
                self.climate_stats.update_parameter_preview(self.get_current_parameters())

            if param_name in ("map_latitude", "air_temp_entry", "air_humidity_entry"):
                self._update_climatology_preview()

            self.logger.debug(f"Parameter changed: {param_name} = {value}")

        except Exception as e:
            self.logger.error(f"Parameter change handling failed: {e}")

    # Nutzer-Vorgabe 2026-07-24: Temperatur-/Feuchte-Werte im Vorschau-Text
    # farblich unterscheidbar machen - von der grauen Standardschrift aus
    # leicht Richtung Rot (Temperatur) bzw. Richtung Blau (Feuchte) verschoben,
    # kein knalliges Voll-Rot/Blau (soll dezent bleiben, kein Warn-Farbton).
    _CLIMATOLOGY_TEMP_COLOR = "#c98a8a"
    _CLIMATOLOGY_HUMID_COLOR = "#8a9ec9"
    _CLIMATOLOGY_LABEL_COLOR = "#a0a0a0"

    def _update_climatology_preview(self):
        """
        Live-Klimatologie-Vorschau (Nutzer-Vorgabe 2026-07-24): berechnet für
        den aktuellen Latitude-/Offset-Slider-Stand die 6 saisonalen
        Temperatur-/Feuchte-Basiswerte (S1=Jan/Feb .. S6=Nov/Dez) über
        dieselbe _climate_baseline()-Formel wie die echte Simulation
        (core/weather_generator.py) - garantiert identische Werte zur
        tatsächlichen Generierung statt einer separat gepflegten Kopie der
        Formel, die aus dem Tritt geraten könnte. air_temp_entry ist ein
        additiver Offset (kein Limit über den Slider-Bereich hinaus nötig,
        siehe WEATHER.AIR_TEMP_ENTRY), air_humidity_entry ebenso additiv,
        aber IMMER auf 0-100% geklemmt (physikalisch sinnvoller Prozent-
        Bereich, siehe derselbe Clip in _generate_seasonal_parameters()).
        3 Werte pro Zeile (S1-S3 / S4-S6) statt aller 6 in einer Zeile, Werte
        farblich abgesetzt (Rot=Temperatur, Blau=Feuchte) via Rich-Text/HTML.
        """
        if self.climatology_preview_label is None:
            return
        try:
            latitude = self.parameter_sliders["map_latitude"].getValue()
            temp_offset = self.parameter_sliders["air_temp_entry"].getValue()
            humid_offset = self.parameter_sliders["air_humidity_entry"].getValue()

            entries = []
            for month_index in range(6):
                base_temp, base_humid_frac = self._climatology_preview_generator._climate_baseline(
                    latitude, month_index)
                temp = base_temp + temp_offset
                humid = max(0.0, min(100.0, base_humid_frac * 100.0 + humid_offset))
                entries.append(
                    f'<span style="color:{self._CLIMATOLOGY_LABEL_COLOR}">S{month_index + 1}: '
                    f'<span style="color:{self._CLIMATOLOGY_TEMP_COLOR}">{temp:.0f}°C</span>/'
                    f'<span style="color:{self._CLIMATOLOGY_HUMID_COLOR}">{humid:.0f}%</span></span>'
                )

            line1 = "&nbsp;&nbsp;".join(entries[0:3])
            line2 = "&nbsp;&nbsp;".join(entries[3:6])
            self.climatology_preview_label.setText(
                f'<div style="font-size:10px;">{line1}<br>{line2}</div>')
        except Exception as e:
            self.logger.debug(f"Climatology preview update failed: {e}")

    def _on_display_mode_changed(self, mode: str, checked: bool):
        """Handler für Display Mode Changes"""
        if checked:
            self.current_display_mode = mode
            # Layer-Umschalter (Ground/Mid/High) ergibt nur für Temperature/
            # Humidity/Precipitation/Wind einen Sinn, nicht für Height (reine
            # Geländeform, keine Atmosphären-Schicht) - siehe
            # _create_layer_controls()-Docstring.
            if hasattr(self, "layer_controls_widget"):
                self.layer_controls_widget.setVisible(mode != "height")
            self.update_display_mode()
            self._update_month_cycle_timer_state()
            self.logger.debug(f"Display mode changed to: {mode}")

    def _on_layer_changed(self, index: int, checked: bool):
        """Handler für Schicht-Umschalter (Ground/Mid/High), siehe _create_layer_controls()."""
        if checked:
            self.current_layer_index = index
            self.update_display_mode()
            self._update_month_cycle_timer_state()
            self.logger.debug(f"Layer changed to index: {index}")

    # =============================================================================
    # DISPLAY UPDATE SYSTEM
    # =============================================================================

    def _current_monthly_key(self) -> str:
        """
        Monats-Listen-Schlüssel für den aktuellen Anzeige-/Schicht-Zustand -
        Ground (current_layer_index==0) nutzt die bestehenden *_monthly-Listen
        (H,W-Arrays), Mid/High nutzt die *_layers_monthly-Listen (3,H,W-Arrays
        pro Monat, siehe DataLODManager.set_weather_data_complete_lod()).
        precip_map hat KEINE Pro-Schicht-Aufschlüsselung (akkumulierte Größe
        über alle Schichten, siehe update_display_mode()) und bleibt daher
        unabhängig von current_layer_index bei der Ground-Monatsliste.
        """
        use_ground = self.current_layer_index == 0 or self.current_display_mode == "precip_map"
        suffix = "_monthly" if use_ground else "_layers_monthly"
        return f"{self.current_display_mode}{suffix}"

    def _update_month_cycle_timer_state(self):
        """
        Funktionsweise: Start/Stop-Gating für den Monats-Animations-Timer,
        analog zu map_display_3d.py's animation_timer-Muster - läuft nur,
        solange die aktuell ausgewählte Karte/Schicht tatsächlich saisonale
        Monatsdaten hat (temp_map/humid_map/wind_map für alle 3 Schichten,
        precip_map nur für Ground - siehe _current_monthly_key() - NIE
        "height"). Wird bei jedem Moduswechsel und nach erfolgreicher
        Generierung neu geprüft.
        """
        has_monthly_data = (
            self.current_display_mode in ("temp_map", "precip_map", "humid_map", "wind_map")
            and self.data_lod_manager is not None
            and self.data_lod_manager.get_weather_data(self._current_monthly_key())
        )
        if has_monthly_data and not self._month_cycle_timer.isActive():
            self._current_month_index = 0
            self._month_cycle_timer.start()
        elif not has_monthly_data and self._month_cycle_timer.isActive():
            self._month_cycle_timer.stop()

    def _on_month_cycle_tick(self):
        """Wird jede Sekunde vom _month_cycle_timer aufgerufen - schaltet auf
        die nächste der 6 saisonalen Monats-Karten um und stößt ein Redraw an."""
        monthly_list = self.data_lod_manager.get_weather_data(self._current_monthly_key())
        if not monthly_list:
            self._month_cycle_timer.stop()
            return
        self._current_month_index = (self._current_month_index + 1) % len(monthly_list)
        self.update_display_mode()

    def update_display_mode(self):
        """
        Überschreibt BaseMapTab Display-Update für Weather-spezifische Modi.
        Implementiert Height/Temperature/Precipitation/Humidity/Wind Display-Switching.
        """
        try:
            if not self.data_lod_manager:
                return

            current_display = self.get_current_display()
            if not current_display:
                return

            if self.current_display_mode == "height":
                # Kombiniert, nicht die unbearbeitete Terrain-Rohausgabe - siehe
                # DataLODManager.get_terrain_data_combined()
                data = self.data_lod_manager.get_terrain_data_combined("heightmap")
                data_type = "heightmap_combined"
                display_data = data
            elif (self.current_layer_index != 0
                  and self.current_display_mode in ("temp_map", "humid_map", "wind_map")):
                # Schicht-Diagnose (Mid/High statt Ground, siehe
                # _create_layer_controls()) - Precipitation hat keine
                # Pro-Schicht-Aufschlüsselung (akkumulierte Größe über alle
                # Schichten) und bleibt daher außen vor. Animiert wie Ground
                # über die 6 Monats-3-Schicht-Arrays (*_layers_monthly, siehe
                # _on_month_cycle_tick()); ohne Monatsdaten (z.B. direkt nach
                # der Generierung, bevor der Timer tickt, oder alte
                # Cache-Einträge) Fallback auf den saisonalen Mittelwert
                # (*_layers, je 3,H,W bzw. 3,H,W,2).
                monthly_list = self.data_lod_manager.get_weather_data(self._current_monthly_key())
                if monthly_list:
                    month_layers = monthly_list[self._current_month_index % len(monthly_list)]
                    data = month_layers[self.current_layer_index]
                else:
                    layers = self.data_lod_manager.get_weather_data(f"{self.current_display_mode}_layers")
                    if layers is not None and layers.shape[0] > self.current_layer_index:
                        data = layers[self.current_layer_index]
                    else:
                        data = self.data_lod_manager.get_weather_data(self.current_display_mode)
                data_type = self.current_display_mode
                display_data = data
            elif self.current_display_mode in ("temp_map", "precip_map", "humid_map", "wind_map"):
                # Bei vorhandenen saisonalen Monatsdaten wird die aktuell
                # animierte Monats-Karte gezeigt (siehe _on_month_cycle_tick()),
                # sonst Fallback auf den saisonalen Mittelwert (z.B. direkt nach
                # der Generierung, bevor der Timer den ersten Tick gemacht hat,
                # oder für alte Cache-Einträge ohne Monatsdaten).
                monthly_list = self.data_lod_manager.get_weather_data(f"{self.current_display_mode}_monthly")
                if monthly_list:
                    data = monthly_list[self._current_month_index % len(monthly_list)]
                else:
                    data = self.data_lod_manager.get_weather_data(self.current_display_mode)
                data_type = self.current_display_mode
                # wind_map bleibt roh (H,W,2) u/v-Windkomponenten - MapDisplay2D
                # rendert es über _render_wind_map() als Pfeil-Feld + Stärke-
                # Heatmap, keine Magnitude-Reduktion mehr nötig.
                display_data = data
            else:
                return

            if data is not None and display_data is not None and hasattr(current_display, 'update_display'):
                display_id = f"WeatherTab_{self.current_view}_{data_type}"

                if hasattr(self.data_lod_manager, 'display_update_manager'):
                    needs_update = self.data_lod_manager.display_update_manager.needs_update(
                        display_id, data, data_type
                    )

                    if needs_update:
                        self._push_data_to_current_display(display_data, data_type)
                        self.data_lod_manager.display_update_manager.mark_updated(
                            display_id, data, data_type
                        )
                else:
                    self._push_data_to_current_display(display_data, data_type)

        except Exception as e:
            self.logger.debug(f"Weather display mode update failed: {e}")

    # =============================================================================
    # GENERATION
    # =============================================================================

    def generate(self):
        """
        Überschreibt BaseMapTab: Dependency-Check vor der eigentlichen Generation.
        Wird ausschließlich über den globalen [GENERIEREN]-Button im Shell-Footer
        ausgelöst (kein eigener Berechnen-Button mehr im Parameter-Panel).
        """
        try:
            if not self.check_input_dependencies():
                self.logger.warning("Input dependencies (heightmap/shadowmap) not met")
                return

            super().generate()

            self.logger.info("Weather generation requested")

        except Exception as e:
            self.logger.error(f"Generation request failed: {e}")

    @pyqtSlot(str, dict)
    def on_generation_completed(self, result_id: str, result_data: dict):
        """
        Überschreibt BaseMapTab Completion Handler für Weather-spezifische Completion.
        Aktualisiert Climate-Statistics nach erfolgreicher Generation.
        """
        generator_type = result_data.get("generator_type", "")
        success = result_data.get("success", False)

        if generator_type != self.generator_type:
            return

        try:
            if success:
                self.update_display_mode()
                self._update_month_cycle_timer_state()

                if self.climate_stats:
                    results = {
                        "temp_map": self.data_lod_manager.get_weather_data("temp_map"),
                        "wind_map": self.data_lod_manager.get_weather_data("wind_map"),
                        "humid_map": self.data_lod_manager.get_weather_data("humid_map"),
                        "precip_map": self.data_lod_manager.get_weather_data("precip_map"),
                    }
                    if any(value is not None for value in results.values()):
                        self.climate_stats.update_generation_statistics(results)

            super().on_generation_completed(result_id, result_data)

        except Exception as e:
            self.logger.error(f"Generation completion handling failed: {e}")

    # =============================================================================
    # PARAMETER SYNCHRONISATION
    # =============================================================================

    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Sammelt die aktuellen Werte aller Weather-Parameter-Slider.
        Wird vom ParameterManager als zentrale Quelle für die Weather-Parameter
        genutzt (register_tab()/get_tab_parameters() rufen diese Methode auf,
        siehe gui/OldManagers/parameter_manager.py).
        """
        parameters = {}
        for param_name, slider in self.parameter_sliders.items():
            parameters[param_name] = slider.getValue()
        for param_name, checkbox in self.parameter_checkboxes.items():
            parameters[param_name] = checkbox.isChecked()
        return parameters

    def update_parameter_ui(self, param_name: str, value):
        """
        Überschreibt BaseMapTab Parameter-UI Update für Weather-Parameter.
        Synchronisiert UI-Controls mit ParameterManager-Updates.
        """
        try:
            if param_name in self.parameter_sliders:
                slider = self.parameter_sliders[param_name]
                slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(False)

                # blockSignals unterdrückt valueChanged - _on_parameter_changed()
                # (das die Vorschau normalerweise aktualisiert) läuft hier nicht,
                # deshalb explizit nachziehen (Nutzer-Vorgabe 2026-07-24).
                if param_name in ("map_latitude", "air_temp_entry", "air_humidity_entry"):
                    self._update_climatology_preview()

                self.logger.debug(f"Parameter UI updated: {param_name} = {value}")
            elif param_name in self.parameter_checkboxes:
                checkbox = self.parameter_checkboxes[param_name]
                checkbox.blockSignals(True)
                checkbox.setChecked(bool(value))
                checkbox.blockSignals(False)

                self.logger.debug(f"Parameter UI updated: {param_name} = {value}")

        except Exception as e:
            self.logger.error(f"Parameter UI update failed: {e}")

    # =============================================================================
    # DEPENDENCY SYSTEM
    # =============================================================================

    def check_input_dependencies(self) -> bool:
        """
        Überschreibt BaseMapTab Dependency Check.
        Prüft ob die Terrain-Inputs (heightmap, shadowmap) verfügbar sind.
        """
        try:
            heightmap = self.data_lod_manager.get_terrain_data("heightmap")
            shadowmap = self.data_lod_manager.get_terrain_data("shadowmap")

            dependencies_met = heightmap is not None and shadowmap is not None

            if self.dependency_status:
                if dependencies_met:
                    self.dependency_status.set_success("Terrain inputs available")
                else:
                    missing = []
                    if heightmap is None:
                        missing.append("heightmap")
                    if shadowmap is None:
                        missing.append("shadowmap")
                    self.dependency_status.set_warning(f"Missing terrain data: {', '.join(missing)}")

            return dependencies_met

        except Exception as e:
            self.logger.error(f"Dependency check failed: {e}")
            return False

    # =============================================================================
    # RESOURCE MANAGEMENT
    # =============================================================================

    def cleanup_resources(self):
        """
        Erweitert BaseMapTab Cleanup für Weather-spezifische Resources.
        """
        try:
            self.logger.debug("Cleaning up weather-specific resources")

            if self._month_cycle_timer is not None and self._month_cycle_timer.isActive():
                self._month_cycle_timer.stop()

            self.parameter_sliders.clear()
            self.parameter_checkboxes.clear()

            super().cleanup_resources()

        except Exception as e:
            self.logger.error(f"Weather cleanup failed: {e}")


class ClimateStatisticsWidget(QGroupBox):
    """
    Widget für Climate-Statistiken und Parameter-Preview.
    Zeigt Weather-Parameter, Climate-Classification und Generation-Results.
    """

    def __init__(self):
        super().__init__("Climate Statistics")
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Climate-Statistiken"""
        layout = QVBoxLayout()

        preview_group = QGroupBox("Climate Parameters")
        preview_layout = QVBoxLayout()

        self.base_temp_label = QLabel("Base Temperature: 15°C")
        self.ground_temp_offset_label = QLabel("Ground Temp Offset: 0°C")
        self.altitude_cooling_label = QLabel("Altitude Cooling: 6°C/km")
        self.wind_factor_label = QLabel("Wind Factor: 1.0")

        preview_layout.addWidget(self.base_temp_label)
        preview_layout.addWidget(self.ground_temp_offset_label)
        preview_layout.addWidget(self.altitude_cooling_label)
        preview_layout.addWidget(self.wind_factor_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        classification_group = QGroupBox("Climate Classification")
        classification_layout = QVBoxLayout()

        self.climate_type_label = QLabel("Dominant Climate: Not calculated")
        self.temp_range_label = QLabel("Temperature Range: -")
        self.precip_total_label = QLabel("Total Precipitation: -")

        classification_layout.addWidget(self.climate_type_label)
        classification_layout.addWidget(self.temp_range_label)
        classification_layout.addWidget(self.precip_total_label)

        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group)

        results_group = QGroupBox("Generation Results")
        results_layout = QVBoxLayout()

        self.orographic_effect_label = QLabel("Orographic Effect: -")
        self.wind_strength_label = QLabel("Avg Wind Strength: -")
        self.humidity_level_label = QLabel("Avg Humidity: -")

        results_layout.addWidget(self.orographic_effect_label)
        results_layout.addWidget(self.wind_strength_label)
        results_layout.addWidget(self.humidity_level_label)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        self.setLayout(layout)

    def update_parameter_preview(self, parameters: dict):
        """Aktualisiert Parameter-Preview"""
        base_temp = parameters.get("air_temp_entry", 15)
        ground_offset = parameters.get("ground_temp_offset", 0)
        altitude = parameters.get("altitude_cooling", 6)
        wind = parameters.get("wind_speed_factor", 1.0)

        self.base_temp_label.setText(f"Base Temperature: {base_temp}°C")
        self.ground_temp_offset_label.setText(f"Ground Temp Offset: {ground_offset}°C")
        self.altitude_cooling_label.setText(f"Altitude Cooling: {altitude}°C/km")
        self.wind_factor_label.setText(f"Wind Factor: {wind:.1f}")

    def update_generation_statistics(self, results: dict):
        """
        Aktualisiert Statistiken nach abgeschlossener Generation.
        Parameter: results (dict mit temp_map/wind_map/humid_map/precip_map)
        """
        temp_map = results.get("temp_map")
        wind_map = results.get("wind_map")
        humid_map = results.get("humid_map")
        precip_map = results.get("precip_map")

        if temp_map is not None:
            temp_min, temp_max = np.min(temp_map), np.max(temp_map)
            self.temp_range_label.setText(f"Temperature Range: {temp_min:.1f}°C - {temp_max:.1f}°C")

            avg_temp = np.mean(temp_map)
            if avg_temp < 0:
                climate_type = "Arctic"
            elif avg_temp < 10:
                climate_type = "Subarctic"
            elif avg_temp < 20:
                climate_type = "Temperate"
            else:
                climate_type = "Subtropical"

            self.climate_type_label.setText(f"Dominant Climate: {climate_type}")

        if precip_map is not None:
            total_precip = np.sum(precip_map)
            self.precip_total_label.setText(f"Total Precipitation: {total_precip:.1f} gH2O/m²")

        if wind_map is not None:
            if len(wind_map.shape) == 3:
                wind_strength = np.sqrt(wind_map[:, :, 0] ** 2 + wind_map[:, :, 1] ** 2)
                avg_wind = np.mean(wind_strength)
                self.wind_strength_label.setText(f"Avg Wind Strength: {avg_wind:.2f} m/s")

        if humid_map is not None:
            avg_humidity = np.mean(humid_map)
            self.humidity_level_label.setText(f"Avg Humidity: {avg_humidity:.1f} gH2O/m³")

        if temp_map is not None and wind_map is not None:
            temp_grad = np.gradient(temp_map)
            orographic_strength = np.mean(np.sqrt(temp_grad[0] ** 2 + temp_grad[1] ** 2))
            self.orographic_effect_label.setText(f"Orographic Effect: {orographic_strength:.3f}°C/pixel")
