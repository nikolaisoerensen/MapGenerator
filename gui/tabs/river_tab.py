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

from gui.tabs.base_tab import BaseMapTab


class RiverTab(BaseMapTab):
    """Anzeige des Flussnetzes. Liest terrain.redistribution."""

    def __init__(self, data_lod_manager, parameter_manager, navigation_manager,
                 shader_manager, generation_orchestrator):

        # Das Netz entsteht im Terrain-Generator (terrain.redistribution), es
        # gibt keinen eigenen. Der Reiter zeigt dessen Ausgaben.
        self.generator_type = "terrain"
        self.required_dependencies = ["heightmap"]

        self.parameter_sliders = {}
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

        self.logger.info("RiverTab initialized")

    # ------------------------------------------------------------------
    def create_parameter_controls(self):
        """
        Keine eigenen Regler - bewusst.

        Das Flussnetz haengt an Seed und Weltparametern, die im Terrain-Reiter
        stehen. Eine zweite Stelle mit denselben Reglern waere eine zweite
        Wahrheit; §4.1 verlangt das Gegenteil.
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
        modi = [
            ("height", "Gelaende"),
            ("rivers", "Flussnetz"),
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
                if generation is not None:
                    ziel = self._anzeigeziel()
                    if ziel is not None and hasattr(ziel, "overlay_river_generations"):
                        ziel.overlay_river_generations(
                            np.asarray(generation),
                            zeige_mikro=bool(self.mikro_checkbox
                                             and self.mikro_checkbox.isChecked()))
            else:
                # DAS FLUSSNETZ ABSCHALTEN, sonst liegt es ueber jeder
                # anderen Ansicht dieses Reiters (Nutzerbefund 2026-08-24:
                # *"wenn man auf Ordnung geht dann aendert sich nichts und
                # wenn man wieder auf gelaende geht aendert sich auch
                # nichts"*). Die 2D-Ansicht zeichnet bei jedem Wechsel
                # ohnehin neu; im 3D bleibt eine einmal gesetzte Textur
                # liegen, bis sie ausgeschaltet wird.
                ziel = self._anzeigeziel()
                if ziel is not None and hasattr(ziel, "clear_river_overlay"):
                    ziel.clear_river_overlay()
                art = ("heightmap" if self.current_display_mode == "height"
                       else self.current_display_mode)
                daten = self.data_lod_manager.get_terrain_data(art)
                if daten is None:
                    return
                self._show_data(daten, art)
            self._statistik_auffrischen()
        except Exception as fehler:                      # pragma: no cover
            self.logger.error("Anzeige fehlgeschlagen: %s", fehler)

    def _anzeigeziel(self):
        anzeige = (self.get_current_display() if hasattr(self, "get_current_display")
                   else getattr(self, "current_display", None))
        return getattr(anzeige, "display", None) if anzeige is not None else None

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
