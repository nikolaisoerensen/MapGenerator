"""
Path: gui/tabs/settlement_regional_tab.py

REITER "SIEDLUNGEN (REGIONAL)" - eine Region herangezoomt.

Der globale Siedlungsreiter zeigt die ganze Welt; auf 21 km Kartenbreite ist
ein Weg zwischen zwei Doerfern aber nur ein paar Pixel lang. Der Nutzer dazu am
2026-08-05: "Regional weil wir fuer die Siedlungen die 9 Regionalmaps anzeigen
(heranzoomen) und dort dann die Wege haben."

Dieser Reiter ZOOMT deshalb auf EINE Region. Er rechnet nichts eigenes - er
zeigt dieselben Daten, nur naeher heran. Zugeschnitten werden die Daten
ausdruecklich NICHT: Siedlungen, Wege und Plot-Kanten tragen Koordinaten der
vollen Karte, ein Zuschnitt haette sie alle verschoben.

Damit ist zugleich ausgeschlossen, dass global und regional verschiedene
Ergebnisse zeigen - bei zwei getrennten Rechenwegen waere das unvermeidlich.

Die Regionsgrenzen kommen aus core/terrain_weltkarte.voronoi_regionen() - also
aus derselben Quelle, die auch das Gelaende formt. Ein zweites Regionsgitter
waere eine zweite Wahrheit.
"""

import logging

import numpy as np
from PyQt6.QtWidgets import (
    QVBoxLayout, QGridLayout, QGroupBox, QRadioButton, QButtonGroup, QLabel)

from gui.tabs.base_tab import BaseMapTab


class SettlementRegionalTab(BaseMapTab):
    """Siedlungen und Wege, auf eine der neun Regionen herangezoomt."""

    def __init__(self, data_lod_manager, parameter_manager, navigation_manager,
                 shader_manager, generation_orchestrator):

        self.generator_type = "settlement"
        self.required_dependencies = ["heightmap"]

        self.parameter_sliders = {}
        self.region_group = None
        self.aktuelle_region = 0
        self.region_stats = None
        self._gewichte_cache = None
        self._gewichte_schluessel = None

        self.logger = logging.getLogger("SettlementRegionalTab")

        super().__init__(
            data_lod_manager=data_lod_manager,
            parameter_manager=parameter_manager,
            navigation_manager=navigation_manager,
            shader_manager=shader_manager,
            generation_orchestrator=generation_orchestrator)

        self.logger.info("SettlementRegionalTab initialized")

    # ------------------------------------------------------------------
    def _regionen(self):
        from core.terrain_weltkarte import alle_regionen
        return [r for _z, _s, r in alle_regionen()]

    def _regionsgewichte(self, size, seed):
        """
        Die Zugehoerigkeit je Pixel - GELESEN, nicht gerechnet.

        Bis 2026-08-06 rief dieser Reiter kontinentform() und voronoi_regionen()
        selbst auf. Das war eine zweite Wahrheit (SPEZIFIKATION §4.5): dieselbe
        Rechnung an zwei Stellen, die bei jeder Aenderung an einer davon
        auseinanderlaufen musste - und nebenbei ein zweiter, teurer Durchlauf
        bei jedem Regionswechsel.

        Seitdem liefert terrain.redistribution `region_map` als eigenen Output.
        `size` und `seed` bleiben in der Signatur, weil der Aufrufer sie ohnehin
        hat und sie den Fall abfangen, dass die gelieferte Karte nicht zur
        angezeigten Heightmap passt.
        """
        karte = self.data_lod_manager.get_terrain_data("region_map")
        if karte is None:
            return None
        karte = np.asarray(karte)
        if karte.shape[0] != size:
            self.logger.debug(
                "region_map ist %d px, Heightmap %d px - Regionsansicht "
                "uebersprungen", karte.shape[0], size)
            return None
        hoehe = self.data_lod_manager.get_terrain_data("heightmap")
        maske = np.asarray(hoehe) > 0.0 if hoehe is not None else np.ones_like(
            karte, dtype=bool)
        return maske, karte

    # ------------------------------------------------------------------
    def create_parameter_controls(self):
        if not self.control_panel:
            return
        if self.control_panel.layout() is None:
            layout = QVBoxLayout()
            layout.setContentsMargins(5, 5, 5, 5)
            self.control_panel.setLayout(layout)
            self.control_panel_content_layout = layout

        kasten = QGroupBox("Region")
        gitter = QGridLayout(kasten)
        self.region_group = QButtonGroup()
        # Im 3x3 angeordnet wie auf der Karte - Nord oben, West links. Eine
        # Liste waere kuerzer, aber man muesste sich merken, welche Region wo
        # liegt.
        for nummer, region in enumerate(self._regionen()):
            knopf = QRadioButton(region["name"])
            if nummer == 0:
                knopf.setChecked(True)
            self.region_group.addButton(knopf, nummer)
            gitter.addWidget(knopf, nummer // 3, nummer % 3)
        self.region_group.idClicked.connect(self._on_region_gewaehlt)
        self.control_panel_content_layout.addWidget(kasten)

        hinweis = QLabel(
            "Zeigt dieselben Siedlungsdaten wie der globale\n"
            "Reiter, nur auf eine Region zugeschnitten.\n"
            "Gerechnet wird nichts eigenes - sonst koennten\n"
            "beide Ansichten verschiedene Ergebnisse zeigen.")
        hinweis.setWordWrap(True)
        self.control_panel_content_layout.addWidget(hinweis)

    def _on_region_gewaehlt(self, nummer: int):
        self.aktuelle_region = nummer
        self.update_display_mode()

    # ------------------------------------------------------------------
    def create_statistics_controls(self, layout: QVBoxLayout):
        kasten = QGroupBox("Region")
        innen = QVBoxLayout(kasten)
        self.region_stats = QLabel("noch nicht berechnet")
        self.region_stats.setWordWrap(True)
        innen.addWidget(self.region_stats)
        layout.addWidget(kasten)

    # ------------------------------------------------------------------
    def update_display_mode(self):
        """
        Die volle Karte zeichnen, dann die ACHSEN auf die Region zoomen.

        NICHT die Daten zuschneiden - das war der erste Entwurf und waere
        falsch gewesen: Siedlungen, Wege und Plot-Kanten tragen Koordinaten der
        VOLLEN Karte. Ein Zuschnitt haette sie alle um den Ursprung des
        Ausschnitts verschoben, und jede Ueberlagerung haette danebengelegen.
        Beim Zoomen der Achsen bleibt jede Koordinate gueltig, und samtliche
        vorhandenen Overlays funktionieren unveraendert weiter.
        """
        try:
            hoehe = self.data_lod_manager.get_terrain_data("heightmap")
            if hoehe is None:
                return
            hoehe = np.asarray(hoehe)
            size = hoehe.shape[0]
            seed = int(self.data_lod_manager.get_map_seed())

            zuordnung = self._regionsgewichte(size, seed)
            if zuordnung is None:
                if self.region_stats:
                    self.region_stats.setText(
                        "Regionskarte noch nicht berechnet.")
                return
            maske, fuehrend = zuordnung
            gebiet = maske & (fuehrend == self.aktuelle_region)
            if gebiet.sum() < 16:
                if self.region_stats:
                    self.region_stats.setText(
                        "Diese Region liegt vollstaendig unter Wasser.")
                return

            anzeige = self.get_current_display() if hasattr(
                self, "get_current_display") else getattr(self, "current_display", None)
            if anzeige is None or not hasattr(anzeige, "display"):
                return
            ziel = anzeige.display

            if hasattr(ziel, "update_map_data"):
                ziel.update_map_data(hoehe, "heightmap")
            elif hasattr(ziel, "update_heightmap"):
                ziel.update_heightmap(hoehe, self.generator_type)

            self._overlays_zeichnen(ziel)
            self._auf_region_zoomen(ziel, gebiet, size)
            self._statistik_auffrischen(gebiet, hoehe, size)
            self.apply_3d_overlays()
        except Exception as fehler:                      # pragma: no cover
            self.logger.error("Regionsanzeige fehlgeschlagen: %s", fehler)

    def _overlays_zeichnen(self, ziel):
        """
        Siedlungen/Landmarken/Roadsites als Punkte wie global, PLUS - anders
        als global (2026-08-10, Nutzer-Vorgabe: "stadtgrenzen und die nodes
        und alles sollten nur regional erscheinen") - die Stadtgrenze und das
        volle PlotPhysicsSystem-Feingewebe (Kerne/Nodes/Kanten/Wildnisgrenze).
        Auf einer einzelnen herangezoomten Region zeigt das tatsaechlich
        etwas; auf der Weltkarte waeren es nur tausende Pixel Gewimmel.
        """
        try:
            if hasattr(ziel, "overlay_region_grid"):
                hoehe = self.data_lod_manager.get_terrain_data("heightmap")
                if hoehe is not None:
                    ziel.overlay_region_grid(np.asarray(hoehe).shape[0])

            if hasattr(ziel, "overlay_settlements"):
                ziel.overlay_settlements(
                    self.data_lod_manager.get_settlement_data("settlement_list") or [],
                    self.data_lod_manager.get_settlement_data("landmark_list") or [],
                    self.data_lod_manager.get_settlement_data("roadsite_list") or [])

            if hasattr(ziel, "overlay_city_boundary_contour"):
                city_mask = self.data_lod_manager.get_settlement_data("city_mask")
                if city_mask is not None:
                    ziel.overlay_city_boundary_contour(city_mask)

            # HIER STAND EIN FEHLER: `overlay_plot_boundaries(kanten)` gab
            # plot_edges (ein Dict[int, PlotEdge]) als ERSTES Argument, also
            # in den `plot_nodes`-Parameter - die Methode iteriert dann ueber
            # die Dict-SCHLUESSEL (int) und ruft `.node_id` darauf auf, was
            # immer einen AttributeError warf. Der `except` unten fing ihn
            # lautlos ab (nur logger.debug) - das Plot-Feingewebe hat auf
            # diesem Reiter deshalb NIE gezeichnet, seit es hier eingefuehrt
            # wurde. Alle vier Argumente in der richtigen Reihenfolge:
            plot_nodes = self.data_lod_manager.get_settlement_data("plot_nodes")
            if plot_nodes and hasattr(ziel, "overlay_plot_boundaries"):
                plot_edges = self.data_lod_manager.get_settlement_data("plot_edges")
                plot_cores = self.data_lod_manager.get_settlement_data("plot_cores")
                wilderness_polygons = self.data_lod_manager.get_settlement_data("wilderness_polygons")
                ziel.overlay_plot_boundaries(plot_nodes, plot_edges, plot_cores, wilderness_polygons)
        except Exception as fehler:                      # pragma: no cover
            self.logger.debug("Overlay nicht verfuegbar: %s", fehler)

    def apply_3d_overlays(self):
        """
        PlotPhysicsSystem als texturierter 3D-Skin - von SettlementTab
        (Global) hierher verschoben (2026-08-10, siehe _overlays_zeichnen()-
        Docstring). `self.map_display_3d` gehoert diesem Tab exklusiv (jeder
        BaseMapTab bekommt sein eigenes 3D-Widget), unabhaengig vom aktuell
        sichtbaren 2D/3D-Modus.
        """
        if not self.map_display_3d or not hasattr(self.map_display_3d.display, 'update_overlay_data'):
            return

        display_3d = self.map_display_3d.display
        plot_nodes = self.data_lod_manager.get_settlement_data("plot_nodes")
        heightmap = self.data_lod_manager.get_terrain_data("heightmap")
        has_plots = bool(plot_nodes) and heightmap is not None
        if has_plots:
            plot_edges = self.data_lod_manager.get_settlement_data("plot_edges")
            plot_cores = self.data_lod_manager.get_settlement_data("plot_cores")
            wilderness_polygons = self.data_lod_manager.get_settlement_data("wilderness_polygons")
            from gui.widgets.map_display_2d import rasterize_plot_boundaries_rgba
            rgba = rasterize_plot_boundaries_rgba(
                plot_nodes, plot_edges, plot_cores, wilderness_polygons,
                map_size=heightmap.shape[0], resolution=heightmap.shape[0])
            display_3d.update_overlay_data("settlement", "plots", rgba)

        if hasattr(display_3d, 'set_layer_visibility'):
            display_3d.set_layer_visibility("settlement", "plots", has_plots)

    def _auf_region_zoomen(self, ziel, gebiet, size):
        """
        Die Achsen auf den FESTEN Regionskasten setzen, mit 25 % ueberlappendem
        Rand je Seite (docs/OFFENE_PUNKTE.md 5.9, Nutzer-Vorgabe 2026-08-10: "die
        siedlungen und landmarks und road sites sollten dabei etwas von der
        grenze entfernt sein, damit diese mit umland auf die karte passen").

        NICHT mehr die Bounding-Box der weichen, verzogenen Regionszugehoerigkeit
        (`gebiet`, aus `region_map`) - die waere organisch verformt und je
        Region unterschiedlich gross, ungeeignet fuer neun gleich grosse,
        durchblaetterbare "Regionalkarten". Der feste Kasten aus
        `regionsbox_px()` ist derselbe, den auch das gelbe Gitter zeichnet
        (`overlay_region_grid`) und gegen den die Platzierung einen weichen
        Randabstand haelt (`settlement_generator._randfaktor`) - eine Wahrheit
        fuer Anzeige, Zuschnitt und Platzierung.
        """
        achse = getattr(ziel, "ax", None)
        if achse is None:
            return
        from core.terrain_weltkarte import regionsbox_px
        zeile, spalte = divmod(self.aktuelle_region, 3)
        x0, x1, y0, y1 = regionsbox_px(zeile, spalte, size, rand_anteil=0.25)
        x0, x1 = max(x0, 0), min(x1, size)
        y0, y1 = max(y0, 0), min(y1, size)
        achse.set_xlim(x0, x1)
        # origin='lower' in _render_heightmap - also nicht umdrehen.
        achse.set_ylim(y0, y1)
        leinwand = getattr(ziel, "canvas", None)
        if leinwand is not None:
            leinwand.draw_idle()

    def _statistik_auffrischen(self, gebiet, hoehe, size):
        try:
            from core.terrain_weltkarte import WELT_KM
            region = self._regionen()[self.aktuelle_region]
            mpp = WELT_KM * 1000.0 / size
            flaeche = float(gebiet.sum()) * mpp * mpp / 1e6
            werte = hoehe[gebiet]
            wasser = 100.0 * float((werte <= 0).mean())

            # Wieviele Orte liegen wirklich IN dieser Region? Das ist die Zahl,
            # wegen der es diesen Reiter gibt.
            drin = 0
            orte = self.data_lod_manager.get_settlement_data("settlement_list") or []
            for ort in orte:
                x = int(round(getattr(ort, "x", -1)))
                y = int(round(getattr(ort, "y", -1)))
                if 0 <= y < size and 0 <= x < size and gebiet[y, x]:
                    drin += 1

            self.region_stats.setText(
                "\n".join([
                    "%s (%s)" % (region["name"], region.get("volk", "-")),
                    region.get("bemerkung", ""),
                    "",
                    "Grundflaeche %.1f km2" % flaeche,
                    "Wasseranteil %.0f %%" % wasser,
                    "Hoehe %.0f bis %.0f m" % (float(werte.min()),
                                               float(werte.max())),
                    "Siedlungen in der Region: %d" % drin,
                ]))
        except Exception as fehler:                      # pragma: no cover
            self.logger.debug("Regionsstatistik nicht verfuegbar: %s", fehler)
