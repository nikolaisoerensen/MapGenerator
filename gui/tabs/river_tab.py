"""
Path: gui/tabs/river_tab.py

REITER "FLUSSNETZWERK" - das Flussnetz der Weltkarte.

Er steht direkt hinter Terrain, weil das Netz dessen Heightmap FORMT: die
Taeler werden eingegraben, bevor Geologie, Wetter und alles Weitere darauf
rechnen (core/terrain_weltfluesse.py, docs/INTEGRATIONSPLAN.md Stufe P2).

Eigene Regler hat er nicht - das Netz haengt an denselben Groessen wie das
Gelaende, und eine zweite Stelle mit denselben Reglern waere eine zweite
Wahrheit.

    Flussnetz           das Gelaende mit den Laeufen darueber, nach
                        GENERATION gefaerbt: Makro rot, Meso gruen
    Gelaende            dasselbe ohne Laeufe, zum Vergleich
    Ordnung (Strahler)  wie gross ein Lauf gemessen an seinen Zufluessen ist

DIE FARBE IST DAS EIGENTLICHE WERKZEUG. An ihr sieht man auf einen Blick, ob
ein Strom durchgehend Strom bleibt oder unterwegs zum Bach wird - der Nutzer
hat genau daran am 2026-08-04 einen echten Fehler entdeckt, als ein roter Lauf
mitten in der Karte abriss.

Die Mikrostufe ist zuschaltbar und standardmaessig AUS: auf 21 km Kartenbreite
sind das Rinnsale von wenigen hundert Metern, die das Bild fuellen, ohne etwas
auszusagen.
"""

import logging

import numpy as np

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton, QButtonGroup, QLabel,
    QCheckBox)

from gui.tabs.base_tab import BaseMapTab, Overlay


# LIVE-VORSCHAU DES FLUSSNETZES (docs/AUFRAEUMPLAN.md 4.10)
#
# Nutzerentwurf 2026-08-26: *"dann kommt flussnetzwerke und auch hier sollte
# eine live sicht moeglich sein."*
#
# GEMESSEN, warum 128 px und nicht mehr:
#
#     px    weltfeld (einmal)   Fluesse (je Reglerzug)
#    128         1.30 s               0.65 s
#    192         1.57 s               1.63 s
#    256         2.25 s               2.10 s
#
# Aufgeschluesselt kostet bei 128 px `flussnetz()` allein 0.50 s, das
# Taeler-Eingraben 0.02 s und die Rasterschleife 0.05 s. **Der Aufwand
# steckt im Netz, nicht im Zeichnen** - eine Vermutung, die ausdruecklich
# geprueft wurde, weil am selben Tag die Linienstaerke in genau diese
# Schleife eingebaut worden war.
#
# DAS GRUNDGELAENDE WIRD EINMAL GERECHNET UND BEHALTEN. Die fuenf
# Flussregler aendern es nicht - nur das Netz und die Taeler. Ohne diesen
# Zwischenspeicher kostete jeder Reglerzug 1.95 statt 0.65 s.
VORSCHAU_PX = 128

# GEZEIGT WIRD DAS GELAENDE MIT DEN TAELERN, nicht eine eigene Flusskarte.
# Alle fuenf Regler dieses Reiters formen Taeler (Abstand, Breite, Tiefe,
# Form, Lauflage) - das eingeschnittene Gelaende IST also ihr Ergebnis. Es
# geht ausserdem als "heightmap" durch den gewoehnlichen Anzeigeweg und
# damit ohne Sonderbehandlung durch 2D UND 3D (stehende Regel in CLAUDE.md).


class RiverTab(BaseMapTab):
    """Anzeige des Flussnetzes. Liest terrain.redistribution."""

    def __init__(self, data_lod_manager, parameter_manager, navigation_manager,
                 shader_manager, generation_orchestrator):

        # Das Netz entsteht im Terrain-Generator (terrain.redistribution), es
        # gibt keinen eigenen. Der Reiter zeigt dessen Ausgaben.
        self.generator_type = "terrain"
        self.required_dependencies = ["heightmap"]

        self.parameter_sliders = {}
        # Zwischenspeicher der Live-Vorschau, siehe VORSCHAU_PX.
        self._vorschau_basis = None
        self._vorschau_seed = None
        self.display_mode_group = None
        self.current_display_mode = "height"
        self._display_modes_by_id = {}
        self.river_stats = None
        self.mikro_checkbox = None

        self.logger = logging.getLogger("RiverTab")

        super().__init__(
            data_lod_manager=data_lod_manager,
            parameter_manager=parameter_manager,
            navigation_manager=navigation_manager,
            shader_manager=shader_manager,
            generation_orchestrator=generation_orchestrator)

        # DER TAKTGEBER ERST NACH super().__init__() - ein QTimer(self)
        # braucht ein fertig gebautes QObject als Elternteil. Davor gibt es
        # `RuntimeError: super-class __init__() of type RiverTab was never
        # called`, und zwar erst beim Bauen des Reiters, nicht beim Import.
        from PyQt6.QtCore import QTimer as _QTimer
        self._vorschau_takt = _QTimer(self)
        self._vorschau_takt.setSingleShot(True)
        self._vorschau_takt.setInterval(250)
        self._vorschau_takt.timeout.connect(self._vorschau_rechnen)

        self.logger.info("RiverTab initialized")

    # ------------------------------------------------------------------
    # DIE FUENF WICHTIGSTEN FLUSSREGLER - GEMESSEN AUSGEWAEHLT (2026-08-26).
    #
    # Nutzervorgabe: *"was davon wird noch benoetigt, das ist a) schon viel
    # zu viel. kannst du die 5 wichtigsten parameter fuer die fluesse
    # herausfinden und mir diese auf die Flussnetzwerktab seite packen?"*
    #
    # Jeder der elf Regler wurde von seinem Minimum zu seinem Maximum
    # gefahren, alles andere auf Vorgabe (384 px, Seed 20260804). Gemessen
    # wurden ZWEI Groessen, weil ein Flussregler auf zwei Arten wirken kann:
    # die Hoehenaenderung (die Taeler) und der Anteil der Flusspixel, die
    # ihren Ort wechseln (der Lauf selbst).
    #
    #   Regler            Hoehe Mittel   groesste   Netz wechselt
    #   SPACING_M              21.58 m    494 m        97.4 %
    #   VALLEY_WIDTH           19.26 m    467 m         0.3 %
    #   INHERIT_COST           10.72 m    690 m        52.6 %
    #   MOUTH_DEPTH_M          10.26 m    451 m        52.2 %
    #   COST_STRENGTH          10.21 m    557 m        72.2 %
    #   INCISION_SHARE          3.22 m    189 m         0.0 %
    #   VALLEY_FORM             0.00 m      0 m         0.0 %  <- war stumm
    #   MEANDER, DIVIDE_BLEND, PLATEAU_FLATTEN, BORDER_OUTFLOW: alle 0.00 m
    #   (die vier stehen ohnehin in stillgelegte_regler)
    #
    # AUSGEWAEHLT WURDE NICHT STRENG NACH DIESER LISTE, und das ist eine
    # Entscheidung, keine Nachlaessigkeit: `INHERIT_COST` und
    # `MOUTH_DEPTH_M` schneiden hoch ab, weil sie den LAUF verschieben - das
    # Netz sieht danach anders aus, die Landschaft aber nicht. Sie sind
    # einmal einzustellen, nicht zum Formen da, und bleiben im
    # Terrain-Reiter.
    #
    # `INCISION_SHARE` steht dagegen HIER, obwohl sein Mittelwert klein ist:
    # 3.22 m im Mittel bei 189 m Maximum heisst, die Wirkung ist auf die
    # Taeler KONZENTRIERT statt ueber die Karte verteilt. Ein Mittelwert
    # allein waere fuer diese Art Regler blind - genau darum steht die
    # Maximalspalte mit in der Tabelle.
    FLUSS_REGLER = (
        ("river_spacing_m", "Talabstand (m)", "SPACING_M"),
        ("river_valley_width", "Talbreite", "VALLEY_WIDTH"),
        ("river_incision_share", "Taltiefe", "INCISION_SHARE"),
        ("river_valley_form", "Talform (V bis U)", "VALLEY_FORM"),
        ("river_cost_strength", "Fluesse folgen dem Tiefland", "COST_STRENGTH"),
    )

    def create_parameter_controls(self):
        """
        Die fuenf wichtigsten Flussregler - siehe FLUSS_REGLER oben.

        Sie stehen NUR hier, nicht zusaetzlich im Terrain-Reiter: zwei
        Widgets fuer denselben Parameterschluessel waeren zwei Wahrheiten
        (SPEZIFIKATION 4.1, und tests/smoke_test_parameter_eindeutig.py
        wacht darueber).
        """
        if not self.control_panel:
            return
        if self.control_panel.layout() is None:
            layout = QVBoxLayout()
            layout.setContentsMargins(5, 5, 5, 5)
            self.control_panel.setLayout(layout)
            self.control_panel_content_layout = layout

        kasten = QGroupBox("Flussnetzwerk")
        innen = QVBoxLayout(kasten)
        hinweis = QLabel(
            "Das Netz entsteht zusammen mit dem Gelaende.\n"
            "Seine Regler stehen im Reiter Terrain.\n\n"
            "Drei Rechenstufen, jede erbt die vorige:\n"
            "  Makro  ~1200 m Knotenabstand - die Stroeme\n"
            "  Meso    ~420 m - die Nebenfluesse\n"
            "  Mikro   ~150 m - die Baeche\n\n"
            "Die Laeufe reichen bis 50 m unter den Meeres-\n"
            "spiegel, damit sie sichtbar muenden; gezeichnet\n"
            "wird nur, was ueber Wasser liegt.")
        hinweis.setWordWrap(True)
        innen.addWidget(hinweis)
        self.control_panel_content_layout.addWidget(kasten)

        from gui.config.value_default import RIVER_NETWORK
        from gui.widgets.widgets import ParameterSlider
        gruppe = QGroupBox("Taeler und Laeufe")
        innen2 = QVBoxLayout(gruppe)
        for schluessel, beschriftung, attr in self.FLUSS_REGLER:
            c = getattr(RIVER_NETWORK, attr)
            regler = ParameterSlider(
                label=beschriftung, min_val=c["min"], max_val=c["max"],
                default_val=c["default"], step=c["step"],
                suffix=c.get("suffix", ""),
                description=c.get("description", ""))
            regler.valueChanged.connect(
                lambda wert, k=schluessel: self._on_parameter_changed(k, wert))
            self.parameter_sliders[schluessel] = regler
            innen2.addWidget(regler)

        from PyQt6.QtWidgets import QCheckBox as _QCheckBox
        self.vorschau_an = _QCheckBox("Live-Vorschau (128 px)")
        self.vorschau_an.setToolTip(
            "Rechnet Flussnetz und Taeler bei 128 px neu, sobald ein Regler "
            "steht - rund 0.65 s. Das Grundgelaende wird dabei behalten, "
            "die Regler aendern es nicht. Aus: es wird gezeigt, was die "
            "Pipeline zuletzt gerechnet hat.")
        self.vorschau_an.toggled.connect(self._vorschau_umschalten)
        innen2.addWidget(self.vorschau_an)
        self.control_panel_content_layout.addWidget(gruppe)

    def _on_parameter_changed(self, name, wert):
        if getattr(self, "parameter_manager", None) is not None:
            try:
                self.parameter_manager.set_parameter(name, wert)
            except Exception as fehler:                   # pragma: no cover
                self.logger.debug("Parameter %s: %s", name, fehler)
        if getattr(self, "vorschau_an", None) is not None                 and self.vorschau_an.isChecked():
            self._vorschau_takt.start()

    def _vorschau_umschalten(self, an):
        if an:
            self._vorschau_takt.start()
        else:
            # Zurueck auf das, was die Pipeline zuletzt gerechnet hat.
            self._vorschau_basis = None
            self.update_display_mode()

    def _vorschau_rechnen(self):
        """
        Flussnetz und Taeler bei VORSCHAU_PX neu - siehe VORSCHAU_PX oben.

        Das Grundgelaende (`weltfeld`) wird beim ersten Mal gerechnet und
        behalten; die fuenf Regler dieses Reiters aendern es nicht.
        """
        import time
        import numpy as _np
        import core.terrain_weltkarte as _rw
        from core.terrain_generator import BaseTerrainGenerator

        t0 = time.time()
        seed = 20260804
        if self.parameter_manager is not None:
            try:
                seed = int(self.parameter_manager.get_tab_parameters(
                    "terrain").get("map_seed", seed))
            except Exception:                             # pragma: no cover
                pass
        if self._vorschau_basis is None or self._vorschau_seed != seed:
            H, felder = _rw.weltfeld(VORSCHAU_PX, seed)
            self._vorschau_basis = (_np.asarray(H, _np.float32), felder)
            self._vorschau_seed = seed
        basis, felder = self._vorschau_basis

        gen = BaseTerrainGenerator.__new__(BaseTerrainGenerator)
        gen.shader_manager = None
        gen.data_lod_manager = None
        gen.logger = self.logger
        gen._current_parameters = dict(self.get_current_parameters())
        try:
            H, _m, _o, _g, _w, _fg = gen._weltfluesse(
                basis.copy(), felder, VORSCHAU_PX, seed)
        except Exception as fehler:                       # pragma: no cover
            self.logger.error("Flussvorschau fehlgeschlagen: %s", fehler)
            return
        self.logger.info("Flussvorschau %d px in %.2f s",
                         VORSCHAU_PX, time.time() - t0)
        # Als "heightmap" durch den gewoehnlichen Weg - damit gilt sie in 2D
        # UND 3D ohne Sonderbehandlung.
        self._show_data(_np.asarray(H, _np.float32), "heightmap")

    def get_current_parameters(self):
        """Die fuenf Flussregler dieses Reiters."""
        return {name: regler.getValue()
                for name, regler in self.parameter_sliders.items()}

    # ------------------------------------------------------------------
    def create_visualization_controls(self):
        """
        Die Knopfleiste ueber der Karte.

        Die Basisklasse ruft GENAU DIESE Methode - `_create_display_mode_controls`
        allein genuegt nicht. Ohne sie wurden die Radio-Knoepfe nie gebaut, und
        `mikro_checkbox` blieb None; aufgefallen erst beim Aufbau des Fensters,
        nicht beim Import.
        """
        from PyQt6.QtWidgets import QWidget
        behaelter = QWidget()
        aussen = QHBoxLayout()
        aussen.setContentsMargins(0, 0, 0, 0)
        aussen.addLayout(self._create_display_mode_controls())
        behaelter.setLayout(aussen)
        return behaelter

    def _create_display_mode_controls(self):
        layout = QHBoxLayout()
        self.display_mode_group = QButtonGroup()

        # DREI ANSICHTEN, NICHT VIER.
        #
        # "Flussnetz" ist die Zweitansicht: Gelaende mit den Laeufen darueber,
        # nach Generation gefaerbt - Makro rot, Meso gruen. Die Mikrostufe ist
        # abschaltbar und standardmaessig AUS, weil sie auf 21 km Kartenbreite
        # nur Rinnsale zeigt.
        #
        # "Ordnung" und "Generation" als rohe Zahlenkarten sind entfallen: sie
        # zeigten dieselbe Information als graue Flecken, aus denen sich nichts
        # ablesen liess.
        #
        # "Gelaende" (reine Heightmap, ohne Laeufe) STEHT ZUERST (2026-08-11,
        # Nutzer-Vorgabe): der erste Radioknopf eines Reiters soll die
        # Grundkarte zeigen, nicht schon eine Ueberlagerung - "Flussnetz" baut
        # als Vergleichsansicht darauf auf, nicht umgekehrt.
        # 2026-08-26: "Wassermenge" ist die neue LEITANSICHT.
        #
        # Nutzervorgabe: *"ich verstehe noch immer nicht die mehrteilung mit
        # roten und gruenen fluessen, jetzt wo wir quasi wassermengen und so
        # haben. koennen wir nur eine karte haben die darstellt wie viel
        # wasser fuer die fluesse berechnet wurde? und dann soll alles gut
        # darstellbar auf der karte zu sehen sein."*
        #
        # `river_water` ist genau das: `netz["flaeche"]`, also Niederschlag
        # mal Flaeche flussabwaerts akkumuliert, aufs Raster gelegt. Sie
        # ersetzt die Rot/Gruen-Faerbung nach Generation als Leitbild.
        #
        # NEBENBEI EINE 3D-SCHULD GETILGT: "Flussnetz" laeuft ueber
        # `overlay_river_generations()`, und die gibt es nur auf
        # MapDisplay2D - in der 3D-Ansicht traf die hasattr-Weiche nie zu und
        # es passierte lautlos nichts (CLAUDE.md, stehende Regel). Die
        # Wassermenge ist eine gewoehnliche Skalarkarte und geht denselben
        # Weg wie "Gelaende", also durch beide Ansichten.
        modi = [
            ("height", "Gelaende"),
            ("river_water", "Wassermenge"),
            ("rivers", "Flussnetz (Generationen)"),
            ("river_order", "Ordnung (Strahler)"),
        ]
        for nummer, (schluessel, beschriftung) in enumerate(modi):
            knopf = QRadioButton(beschriftung)
            if nummer == 0:
                knopf.setChecked(True)
            self.display_mode_group.addButton(knopf, nummer)
            layout.addWidget(knopf)

        # Die Mikrostufe zuschaltbar - abgeschaltet ist die Vorgabe.
        self.mikro_checkbox = QCheckBox("Baeche (Mikro)")
        self.mikro_checkbox.setChecked(False)
        self.mikro_checkbox.toggled.connect(lambda _an: self.update_display_mode())
        layout.addWidget(self.mikro_checkbox)

        self._display_modes_by_id = {n: k for n, (k, _b) in enumerate(modi)}
        # idClicked statt toggled: toggled feuert beim Umschalten zweimal.
        self.display_mode_group.idClicked.connect(self._on_display_mode_selected)
        return layout

    def _on_display_mode_selected(self, nummer: int):
        self.current_display_mode = self._display_modes_by_id.get(nummer, "rivers")
        self.update_display_mode()

    # ------------------------------------------------------------------
    def create_statistics_controls(self, layout: QVBoxLayout):
        kasten = QGroupBox("Netz")
        innen = QVBoxLayout(kasten)
        self.river_stats = QLabel("noch nicht berechnet")
        self.river_stats.setWordWrap(True)
        innen.addWidget(self.river_stats)
        layout.addWidget(kasten)

    def _statistik_auffrischen(self):
        """Was am Netz messbar ist, in Zahlen - nicht nur als Bild."""
        try:
            import numpy as np
            maske = self.data_lod_manager.get_terrain_data("river_mask")
            generation = self.data_lod_manager.get_terrain_data("river_generation")
            hoehe = self.data_lod_manager.get_terrain_data("heightmap")
            if maske is None or hoehe is None:
                self.river_stats.setText("noch nicht berechnet")
                return
            land = np.asarray(hoehe) > 0
            anteil = 100.0 * float(np.asarray(maske)[land].mean()) if land.any() else 0.0
            zeilen = ["Flusspixel: %.1f %% der Landflaeche" % anteil]
            if generation is not None:
                g = np.asarray(generation)
                for wert, name in ((3.0, "Makro"), (2.0, "Meso"), (1.0, "Mikro")):
                    zeilen.append("  %-6s %d Pixel" % (name, int((g == wert).sum())))
            self.river_stats.setText("\n".join(zeilen))
        except Exception as fehler:                      # pragma: no cover
            self.logger.debug("Flussstatistik nicht verfuegbar: %s", fehler)

    # ------------------------------------------------------------------
    def update_display_mode(self):
        """Die gewaehlte Ansicht aufbauen."""
        try:
            if self.current_display_mode == "rivers":
                hoehe = self.data_lod_manager.get_terrain_data("heightmap")
                generation = self.data_lod_manager.get_terrain_data("river_generation")
                if hoehe is None:
                    return
                self._show_data(hoehe, "heightmap")
                # UEBER DAS OVERLAY-REGISTER (Ticket #11,
                # docs/SPEC_OVERLAYS.md): frueher lief das per
                # `ziel.overlay_river_generations(...)` NUR auf dem gerade
                # sichtbaren Display - `overlay_river_generations` gibt es
                # nur auf MapDisplay2D UND (seit 2026-08-24) auf
                # MapDisplay3DWidget, aber der alte Code fragte per hasattr
                # nur DAS AKTUELLE Ziel ab und liess die jeweils andere
                # Ansicht komplett aus. In 3D fehlten die Fluesse deshalb
                # unbemerkt (nur eine laute, aber wirkungslose Warnung).
                # `_push_overlays()` bedient IMMER beide Anzeigen aus
                # demselben Register-Eintrag "fluesse" (siehe
                # BiomeTab.apply_overlays()) und behebt das damit wirklich,
                # statt es nur zu protokollieren.
                zeige_mikro = bool(self.mikro_checkbox and self.mikro_checkbox.isChecked())
                self._push_overlays([
                    Overlay("fluesse", sichtbar=generation is not None,
                            daten=(generation, zeige_mikro) if generation is not None else None),
                ])
            else:
                # DAS FLUSSNETZ ABSCHALTEN, sonst liegt es ueber jeder
                # anderen Ansicht dieses Reiters (Nutzerbefund 2026-08-24:
                # *"wenn man auf Ordnung geht dann aendert sich nichts und
                # wenn man wieder auf gelaende geht aendert sich auch
                # nichts"*). `_push_overlays()` raeumt jetzt BEIDE Anzeigen
                # ab, nicht nur die gerade sichtbare (siehe Kommentar oben).
                self._push_overlays([Overlay("fluesse", sichtbar=False)])
                art = ("heightmap" if self.current_display_mode == "height"
                       else self.current_display_mode)
                daten = self.data_lod_manager.get_terrain_data(art)
                if daten is None:
                    return
                self._show_data(daten, art)
            self._statistik_auffrischen()
        except Exception as fehler:                      # pragma: no cover
            self.logger.error("Anzeige fehlgeschlagen: %s", fehler)

    def _show_data(self, daten, art):
        """
        An die vorhandene Anzeigelogik der Basisklasse weiterreichen.

        HIER LAG DER GRUND, WESHALB IM FLUSS-REITER NICHTS ZU SEHEN WAR
        (Nutzermeldung 2026-08-10: "verschiedene darstellungen die alle nicht
        gehen, weder 2d noch 3d").

        Die Methode fragte nach `self.current_display`. Ein solches Attribut
        gibt es in `BaseMapTab` NICHT - dort heissen sie `map_display_2d` und
        `map_display_3d`, und die richtige Auswahl trifft
        `get_current_display()`. `getattr` lieferte damit immer None, die
        naechste Zeile kehrte zurueck, und zwar lautlos: kein Fehler, keine
        Warnung, nur eine leere Flaeche. Alle drei Ansichten des Reiters waren
        davon betroffen, weil alle drei hierdurch laufen.

        Statt den Namen nur zu berichtigen geht es jetzt ueber
        `_push_data_to_current_display()` - denselben Weg wie in jedem anderen
        Reiter. Der kuemmert sich zusaetzlich um das 3D-Netz, um die
        Weltgroesse und um die Hoehenlinien-Referenz; die Handarbeit hier
        haette das alles uebergehen muessen.
        """
        self._push_data_to_current_display(daten, art)
