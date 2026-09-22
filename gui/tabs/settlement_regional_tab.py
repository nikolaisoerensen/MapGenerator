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
    QVBoxLayout, QGridLayout, QGroupBox, QRadioButton, QButtonGroup, QLabel,
    QCheckBox)

from gui.tabs.base_tab import BaseMapTab, Overlay


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

        kasten = QGroupBox("Spielkarte")
        gitter = QGridLayout(kasten)
        self.region_group = QButtonGroup()
        # BESCHRIFTUNG "Karte N" STATT DER REGIONSNAMEN (2026-08-13): die
        # Auswahl waehlt seit der Spielkarten-Zerlegung (docs/OFFENE_PUNKTE
        # 5.15) eine der neun SPIELKARTEN aus, nicht mehr einen festen Kasten
        # des 3x3-Rasters. Spielkarten und Kulturregionen sind beide neun an
        # der Zahl, decken sich aber NICHT - eine Spielkarte enthaelt im Mittel
        # 3.6 Regionen (gemessen). Die alten Regionsnamen hier stehen zu
        # lassen waere deshalb schlicht falsch beschriftet gewesen; welche
        # Regionen tatsaechlich drin liegen, sagt jetzt die Statistik rechts.
        self._knoepfe = []
        for nummer in range(9):
            knopf = QRadioButton(f"Karte {nummer + 1}")
            if nummer == 0:
                knopf.setChecked(True)
            self.region_group.addButton(knopf, nummer)
            gitter.addWidget(knopf, nummer // 3, nummer % 3)
            self._knoepfe.append(knopf)
        self.region_group.idClicked.connect(self._on_region_gewaehlt)
        self.control_panel_content_layout.addWidget(kasten)
        # Rasterposition -> Kartenindex. Bis die Zerlegung vorliegt die
        # Identitaet; danach geografisch geordnet (siehe _raster_ordnen()).
        self._karten_reihenfolge = list(range(9))

        hinweis = QLabel(
            "Zeigt dieselben Siedlungsdaten wie der globale\n"
            "Reiter, nur auf eine Spielkarte zugeschnitten.\n"
            "Gerechnet wird nichts eigenes - sonst koennten\n"
            "beide Ansichten verschiedene Ergebnisse zeigen.\n\n"
            "Die neun Karten sind konvexe Vielecke mit etwa\n"
            "gleicher Landmasse (Kuestensee zaehlt halb),\n"
            "nicht das alte starre 3x3-Raster - siehe\n"
            "core/spielkarten.py.")
        hinweis.setWordWrap(True)
        self.control_panel_content_layout.addWidget(hinweis)

        # Regionsfaerbung + weisse Grenzen (docs/OFFENE_PUNKTE.md 6.1) - im
        # Global-Reiter schon vorhanden (SettlementTab), hier bisher nicht:
        # "Regionen in Siedlungen Regional bleibt offen". AUS per Default wie
        # dort, gleiche Deckkraft (0.40).
        self.regions_overlay_cb = QCheckBox("Regionen")
        self.regions_overlay_cb.toggled.connect(self.update_display_mode)
        self.control_panel_content_layout.addWidget(self.regions_overlay_cb)

    def _on_region_gewaehlt(self, nummer: int):
        # `nummer` ist die RASTERPOSITION des Knopfes, nicht der Kartenindex -
        # die Knoepfe sind geografisch angeordnet (siehe _raster_ordnen()).
        if nummer < len(self._karten_reihenfolge):
            self.aktuelle_region = self._karten_reihenfolge[nummer]
        else:
            self.aktuelle_region = nummer
        self.update_display_mode()

    def _raster_ordnen(self):
        """
        Ordnet die neun Auswahlknoepfe so, wie die Karten auf der Welt LIEGEN
        (Nutzer-Vorgabe 2026-08-13: "die nordwestlichste karte vom schwerpunkt
        sollte dann oben links liegen, etc. so das es logisch ist").

        Ohne das ist die Reihenfolge die des k-Means und damit willkuerlich:
        der Knopf oben links kann eine Karte im Suedosten waehlen.

        Die Beschriftung folgt der RASTERPOSITION ("Karte 1" ist immer oben
        links), der dahinterliegende Kartenindex wird in
        `_karten_reihenfolge` gemerkt. Andersherum - Beschriftung nach dem
        internen Index - waere das Raster zwar geografisch richtig, aber die
        Nummern sprangen wild, und genau das soll ja weg.
        """
        karten = self.data_lod_manager.get_terrain_data("spielkarte")
        if karten is None:
            return
        try:
            from core import spielkarten
            hoehe = self.data_lod_manager.get_terrain_data("heightmap")
            gewicht = (np.asarray(hoehe) > 0) if hoehe is not None else None
            reihenfolge = spielkarten.geografische_reihenfolge(
                np.asarray(karten), gewicht)
        except Exception as fehler:                      # pragma: no cover
            self.logger.warning(
                "Geografische Anordnung der Kartenknoepfe fehlgeschlagen (%s) - "
                "Raster bleibt in Rechenreihenfolge", fehler)
            return
        if len(reihenfolge) != len(self._karten_reihenfolge):
            return
        if reihenfolge == self._karten_reihenfolge:
            return
        self._karten_reihenfolge = reihenfolge
        # Die aktuell gewaehlte Karte behalten, auch wenn ihr Knopf jetzt
        # woanders sitzt - sonst springt die Ansicht beim Neuordnen.
        if self.aktuelle_region in reihenfolge:
            neue_position = reihenfolge.index(self.aktuelle_region)
            knopf = self.region_group.button(neue_position)
            if knopf is not None:
                knopf.setChecked(True)

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

            # DAS AUSGEWERTETE GEBIET IST SEIT 2026-08-13 DIE SPIELKARTE
            # (docs/OFFENE_PUNKTE 5.15), nicht mehr die Kulturregion aus
            # `region_map`: die Radioknoepfe waehlen jetzt eine der neun
            # Spielkarten, also muss auch die Statistik daneben ueber
            # dieselbe Flaeche laufen. Ohne Zerlegung (alter Nicht-
            # Weltkarten-Pfad) bleibt es beim bisherigen Verhalten.
            self._raster_ordnen()
            karten = self.data_lod_manager.get_terrain_data("spielkarte")
            zuordnung = self._regionsgewichte(size, seed)
            if karten is not None and np.asarray(karten).shape[0] == size:
                karten = np.asarray(karten)
                gebiet = (hoehe > 0.0) & (karten == self.aktuelle_region)
                leer_text = "Diese Spielkarte enthaelt kein Land."
            elif zuordnung is not None:
                maske, fuehrend = zuordnung
                gebiet = maske & (fuehrend == self.aktuelle_region)
                leer_text = "Diese Region liegt vollstaendig unter Wasser."
            else:
                if self.region_stats:
                    self.region_stats.setText(
                        "Weder Spielkarten- noch Regionskarte berechnet.")
                return

            if gebiet.sum() < 16:
                if self.region_stats:
                    self.region_stats.setText(leer_text)
                return

            anzeige = self.get_current_display() if hasattr(
                self, "get_current_display") else getattr(self, "current_display", None)
            if anzeige is None or not hasattr(anzeige, "display"):
                return
            ziel = anzeige.display

            # HIER STAND DER GRUND, WARUM DIESER REITER IN 2D NIE ETWAS ZEIGTE
            # (Nutzerbefund 2026-08-13: "Settlement Regional ist weiterhin
            # nicht funktional. die Map zeigt nichts an.", docs/OFFENE_PUNKTE
            # 5.15). Der Code pruefte per hasattr auf `update_map_data` bzw.
            # `update_heightmap` - MapDisplay2D hat WEDER das eine NOCH das
            # andere, es heisst dort `update_display(data, layer_type)`.
            # In 2D traf also keine der beiden Weichen zu, die Basiskarte
            # wurde nie gezeichnet, und weil samtliche Overlay-Methoden mit
            # `if self.current_data is None: return` beginnen, kehrten danach
            # auch alle Overlays sofort zurueck - ein vollstaendig leerer
            # Reiter, ohne eine einzige Fehlermeldung. (In 3D lief es, weil
            # MapDisplay3D tatsaechlich `update_heightmap` hat - deshalb war
            # der Fehler nur in einer der beiden Ansichten sichtbar.)
            #
            # Jetzt ueber denselben Weg wie JEDER andere Reiter:
            # `_push_data_to_current_display()` kennt beide Anzeigearten,
            # setzt nebenbei Schattenkarte/Sonnenstand/3D-Layer-Sichtbarkeit
            # und kann nicht still danebengreifen - eine eigene hasattr-Weiche
            # hier war von Anfang an eine zweite, schlechter gepflegte Kopie.
            self._push_data_to_current_display(hoehe, "heightmap")

            self._overlays_zeichnen(ziel)
            self._auf_region_zoomen(ziel, gebiet, size)
            self._statistik_auffrischen(gebiet, hoehe, size)
            self.apply_3d_overlays()

            # Siedlungspunkte (Staedte/Landmarken/Roadsites) UEBER DAS REGISTER
            # (Ticket #11, docs/spezifikation/15_ANZEIGE.md): frueher lief das ueber
            # `ziel.overlay_settlements(...)` in _overlays_zeichnen(), eine
            # Methode, die es NUR auf MapDisplay2D gibt. Genau wie bei den
            # anderen vier hasattr-Weichen dort griff das in der 3D-Ansicht
            # nie - `_push_overlays()` bedient beide Anzeigen aus demselben
            # Register-Eintrag "siedlungen" (siehe BiomeTab/SettlementTab).
            settlements = self.data_lod_manager.get_settlement_data("settlement_list") or []
            landmarks = self.data_lod_manager.get_settlement_data("landmark_list") or []
            roadsites = self.data_lod_manager.get_settlement_data("roadsite_list") or []
            siedlungen_sichtbar = bool(settlements or landmarks or roadsites)
            self._push_overlays([
                Overlay("siedlungen", sichtbar=siedlungen_sichtbar,
                        daten=(settlements, landmarks, roadsites)),
            ])
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
                    # Vieleck-Grenzen statt des geraden Rasters, sobald die
                    # Zerlegung vorliegt (docs/OFFENE_PUNKTE 5.15) - dieser
                    # Reiter zoomt seit demselben Datum auf genau diese
                    # Vielecke, ein gerades Gitter daneben waere widerspruechlich.
                    ziel.overlay_region_grid(
                        np.asarray(hoehe).shape[0],
                        spielkarte=self.data_lod_manager.get_terrain_data("spielkarte"))

            # Regionsfaerbung (docs/OFFENE_PUNKTE.md 6.1), gleiche Deckkraft
            # wie im Global-Reiter (0.40).
            if hasattr(ziel, "overlay_regions") and getattr(
                    self, "regions_overlay_cb", None) is not None \
                    and self.regions_overlay_cb.isChecked():
                hoehe = self.data_lod_manager.get_terrain_data("heightmap")
                region_map = self.data_lod_manager.get_terrain_data("region_map")
                if hoehe is not None and region_map is not None:
                    ziel.overlay_regions(region_map, hoehe, alpha=0.40)

            # Siedlungspunkte laufen seit Ticket #11 NICHT mehr hier ueber
            # `ziel.overlay_settlements(...)`, sondern in update_display_mode()
            # ueber das Overlay-Register (`_push_overlays()`) - siehe dort.

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
            # Vorher logger.debug() - lief damit in der Konsole nie auf,
            # obwohl genau dieser Block jahrelang den overlay_plot_boundaries-
            # Argumentfehler (siehe Kommentar oben) lautlos verschluckt hat.
            # STEHENDE REGEL (CLAUDE.md): "jeder stille Rueckfall auf einen
            # Ersatzpfad braucht eine laute Logzeile" - warning statt debug.
            self.logger.warning("Overlay nicht verfuegbar: %s", fehler)

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
            from gui.widgets.overlay_rasterizer import rasterize_plot_boundaries_rgba
            rgba = rasterize_plot_boundaries_rgba(
                plot_nodes, plot_edges, plot_cores, wilderness_polygons,
                map_size=heightmap.shape[0], resolution=heightmap.shape[0])
            display_3d.update_overlay_data("settlement", "plots", rgba)

        if hasattr(display_3d, 'set_layer_visibility'):
            display_3d.set_layer_visibility("settlement", "plots", has_plots)

    def _spielkarten_kasten(self, size, rand_anteil=0.08):
        """
        Bounding-Box der aktuell gewaehlten Spielkarte in Pixeln (x0,x1,y0,y1),
        quadratisch aufgezogen und mit etwas Rand - oder None, wenn die
        Zerlegung nicht vorliegt.

        QUADRATISCH, weil der Anzeigebereich quadratisch ist (Nutzer-Vorgabe
        2026-08-13: "optimiert um auf einem quadrat gezeigt werden zu
        koennen"). Wuerde man die rohe Bounding-Box nehmen, waere das Bild bei
        einem laenglichen Vieleck verzerrt oder mit ungleichem Rand versehen.
        Das gemessene groesste Seitenverhaeltnis der Vielecke liegt bei 2.11,
        das Aufziehen kostet also im schlechtesten Fall etwa die Haelfte der
        Breite an zusaetzlichem Umland - vertretbar, dafuer ist der Massstab
        auf allen neun Karten gleich.
        """
        karten = self.data_lod_manager.get_terrain_data("spielkarte")
        if karten is None:
            return None
        karten = np.asarray(karten)
        if karten.shape[0] != size:
            self.logger.debug(
                "spielkarte ist %d px, Heightmap %d px - Spielkarten-Zuschnitt "
                "uebersprungen", karten.shape[0], size)
            return None
        treffer = karten == self.aktuelle_region
        if not treffer.any():
            return None

        ys, xs = np.nonzero(treffer)
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        seite = max(x1 - x0, y1 - y0) * (1.0 + 2.0 * rand_anteil)
        mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        return mx - seite / 2, mx + seite / 2, my - seite / 2, my + seite / 2

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

        # SPIELKARTEN-ZUSCHNITT, wenn vorhanden (2026-08-13, docs/OFFENE_PUNKTE
        # 5.15): statt des starren 3x3-Kastens die Bounding-Box des tatsaech-
        # lichen Vielecks. Das behebt den eigentlichen Fehler des Rasters -
        # gemessen lagen 28.9 % des Landes in KEINEM der neun festen Kaesten.
        # Faellt auf den alten Kasten zurueck, wenn die Zerlegung fehlt (alter
        # Nicht-Weltkarten-Pfad) - und sagt das dann auch, statt still zu
        # greifen.
        kasten = self._spielkarten_kasten(size)
        if kasten is None:
            from core.terrain_weltkarte import regionsbox_px
            zeile, spalte = divmod(self.aktuelle_region, 3)
            x0, x1, y0, y1 = regionsbox_px(zeile, spalte, size, rand_anteil=0.25)
        else:
            x0, x1, y0, y1 = kasten
        x0, x1 = max(x0, 0), min(x1, size)
        y0, y1 = max(y0, 0), min(y1, size)
        achse.set_xlim(x0, x1)
        # origin='lower' in _render_heightmap - also nicht umdrehen.
        achse.set_ylim(y0, y1)
        leinwand = getattr(ziel, "canvas", None)
        if leinwand is not None:
            leinwand.draw_idle()

    def _dominante_regionen(self, gebiet, hoechstens=3):
        """Namen der Regionen, die im Gebiet den groessten Flaechenanteil
        haben, absteigend - fuer die Kopfzeile der Statistik. Eine Spielkarte
        deckt sich NICHT mit einer Kulturregion (gemessen im Mittel 3.6
        Regionen je Karte), einen einzelnen Namen anzuzeigen waere falsch."""
        karte = self.data_lod_manager.get_terrain_data("region_map")
        if karte is None:
            return []
        karte = np.asarray(karte)
        if karte.shape != gebiet.shape:
            return []
        werte, anzahl = np.unique(karte[gebiet], return_counts=True)
        gebiete = self._regionen()
        gesamt = float(anzahl.sum()) or 1.0
        paare = sorted(zip(werte.tolist(), anzahl.tolist()),
                       key=lambda p: -p[1])[:hoechstens]
        namen = []
        for index, n in paare:
            if 0 <= index < len(gebiete) and n / gesamt >= 0.05:
                namen.append(f"{gebiete[index]['name']} {n / gesamt:.0%}")
        return namen

    def _statistik_auffrischen(self, gebiet, hoehe, size):
        try:
            from core.terrain_weltkarte import WELT_KM
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

            namen = self._dominante_regionen(gebiet)
            self.region_stats.setText(
                "\n".join([
                    "Spielkarte %d" % (self.aktuelle_region + 1),
                    ("Regionen: " + ", ".join(namen)) if namen
                    else "Regionen: (noch nicht berechnet)",
                    "",
                    "Landflaeche %.1f km2" % flaeche,
                    "Wasseranteil %.0f %%" % wasser,
                    "Hoehe %.0f bis %.0f m" % (float(werte.min()),
                                               float(werte.max())),
                    "Siedlungen auf dieser Karte: %d" % drin,
                ]))
        except Exception as fehler:                      # pragma: no cover
            self.logger.debug("Regionsstatistik nicht verfuegbar: %s", fehler)
