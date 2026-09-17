"""
Path: gui/tabs/region_tab.py

Der Regionsreiter - hier wird die Optik jeder Region eingestellt.

NUTZERENTWURF 2026-08-26 (docs/AUFRAEUMPLAN.md 4.10/4.11):

    *"als ziel wuerde ich vorschlagen, dass wir am anfang eine
    regionen-ansicht haben. das sind dann einfach maps von 128-256 px
    groesse, rechteckig mit unseren drop-downs fuer die regionen. dort
    stellt man die optik jeder region ein."*

WARUM DIESER REITER ANDERS RECHNET ALS DIE UEBRIGEN. Alle anderen haengen am
Rechengraphen: sie fordern Daten beim `DataLODManager` an und zeigen, was
die Pipeline gerechnet hat. Dieser hier rechnet SELBST, direkt ueber
`core.terrain_weltkarte.regionsfeld()` - kein Kontinent, keine
Voronoi-Mischung, kein Flussnetz. Genau deshalb ist er live:

    ohne Kueste   0.10 - 0.29 s
    mit Kueste    0.31 - 1.19 s      (gemessen 2026-08-26, 256x160 px)

Die Kueste ist deshalb ein HAEKCHEN, nicht Pflicht: waehrend man einen
Regler zieht, bleibt sie aus und die Vorschau bleibt unter 0.3 s.

DREI WIDGETS, NICHT EINES - so verlangt es die Shell. Siehe `_aufbauen()`.

WAS DIE VORSCHAU NICHT ZEIGT, und das muss klar sein: auf der fertigen
Karte wird dieselbe Region ueber die Voronoi-Gewichte mit ihren Nachbarn
verschmolzen und vom Gebietssystem in der Hoehe verschoben. **Die Vorschau
ist der CHARAKTER der Region, nicht ihr Aussehen auf der Karte.**

2D UND 3D, wie es die stehende Regel verlangt (CLAUDE.md). Beide zeigen
DASSELBE Feld; die 3D-Ansicht bekommt es ueber
`MapDisplay3DWidget.update_heightmap()`.
"""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QHBoxLayout,
                             QLabel, QPushButton, QSizePolicy, QStackedWidget,
                             QVBoxLayout, QWidget)

import core.terrain_weltkarte as rw
from gui.config.value_default import EROSION_FILTER
from gui.widgets.widgets import ParameterSlider

# Vorschauaufloesung. 256 px auf eine 7.1 km breite Region sind 27.7 m/px -
# ein Drittel groeber als die Kontinentalansicht bei 1024 px (20.8 m/px),
# aber die GANZE Region im Bild.
#
# Ein Fenster mit GLEICHEM Massstab zeigte bei 128 px nur 2.66 km, also 37 %
# der Regionsbreite; im Nevadin waere darin weniger als eine Bergform
# gewesen (gemessen 2026-08-26). Deshalb die ganze Region, etwas groeber.
VORSCHAU_BREITE = 256
VORSCHAU_HOEHE = 160

# Die fuenf Regler, die eine Region ausmachen. Sie duerfen NUR fuer die
# aktuelle Karte ueberschrieben werden (Nutzerentscheidung 2026-08-25) - der
# Katalog in core/terrain_weltkarte.py bleibt die Vorgabe, und die Tests der
# Pipeline messen weiter gegen ihn.
#
# SCHRITTWEITE FEIN GENUG FUER DIE KATALOGWERTE. Ein erster Entwurf nahm
# Schritt 10 fuer Hoehe und Relief. Die Katalogwerte liegen aber auf keiner
# solchen Stufe (Clonagh 165.3 m, Skerrheim 484.9 m) - die Regler
# rasteten beim Aufbau auf 170 bzw. 480, und der Reiter meldete SOFORT eine
# Ueberschreibung, ohne dass jemand etwas angefasst hatte. Genau derselbe
# Fehler wie bei `MAP_DISTANCE_KM` am selben Tag (dort verstellte er die
# Weltbreite um 1.4 %).
REGIONSREGLER = (
    ("hoehe_m", "Mittlere Hoehe (m)", -300.0, 1600.0, 0.1),
    ("relief_m", "Relief (m)", 20.0, 1400.0, 0.1),
    ("formgroesse_m", "Formgroesse (m)", 400.0, 6000.0, 100.0),
    ("rauheit", "Rauheit", 0.20, 0.90, 0.01),
    ("potenz", "Hoehenverteilung", 0.30, 3.00, 0.05),
)

# Die Erosionsfilterregler, nach GEMESSENER Wirkung ausgewaehlt (mittlere
# Hoehenaenderung ueber den vollen Reglerweg, 2026-08-26):
#   GULLY_SIZE_M 47.04 m, GULLY_WEIGHT 4.27 m, STRENGTH 3.44 m,
#   OCTAVES 1.17 m, RIDGE_ROUNDING 0.45 m, CREASE_ROUNDING 0.42 m
# `DETAIL` (0.18 m) fehlt bewusst - ueber seinen ganzen Bereich im Bild
# nicht unterscheidbar.
EROSIONSREGLER = (
    ("erosion_scale", "Rinnengroesse", "GULLY_SIZE_M"),
    ("erosion_gully_weight", "Rinnen gegen Kanten", "GULLY_WEIGHT"),
    ("erosion_strength", "Erosionsstaerke", "STRENGTH"),
    ("erosion_octaves", "Rinnen-Oktaven", "OCTAVES"),
)

# Alle Rohfelder eines Regionskatalog-Eintrags, in Anzeigereihenfolge -
# dieselben Namen wie _REGION_FELDER in core/daten/regionen_laden.py, nur
# um "name" ergaenzt. Fuer den read-only Katalogblock im Statistik-Bereich
# (Ticket #29): zeigt die TOML-Werte unveraendert, unabhaengig von den fuenf
# REGIONSREGLER oben, die nur einen Ausschnitt davon bedienen.
REGIONEN_FELDER = (
    "name", "farbe", "volk", "bemerkung", "hoehe_m", "relief_m",
    "formgroesse_m", "rauheit", "potenz", "wasser_soll", "flaeche_soll",
    "kuestenform", "temp_mittel_m0", "temp_spanne", "niederschlag_mm",
    "wind_mittel_ms", "hang_trockenheit", "talform",
)


class _Vorschauflaeche(QWidget):
    """Traegt die Vorschau und meldet jede Groessenaenderung."""

    def __init__(self, beim_wachsen):
        super().__init__()
        self._beim_wachsen = beim_wachsen

    def resizeEvent(self, ereignis):
        super().resizeEvent(ereignis)
        self._beim_wachsen()


class RegionTab(QWidget):
    """Regionsvorschau mit Dropdown, Reglern und Live-Neuzeichnung."""

    def __init__(self, data_lod_manager=None, parameter_manager=None,
                 navigation_manager=None, shader_manager=None,
                 generation_orchestrator=None, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("RegionTab")
        # Die Manager werden entgegengenommen, damit der Reiter denselben
        # Vertrag erfuellt wie alle anderen (siehe
        # MapEditor._create_tab_instance) - gebraucht wird hier keiner.
        self.shader_manager = shader_manager
        self.parameter_manager = parameter_manager

        self.regler: Dict[str, ParameterSlider] = {}
        self.erosionsregler: Dict[str, ParameterSlider] = {}
        # Ueberschreibungen JE REGION, damit ein Wechsel im Dropdown die
        # Einstellungen nicht verwirft.
        self.ueberschreibungen: Dict[str, Dict[str, float]] = {}
        self._letztes_feld: Optional[np.ndarray] = None
        self._sperre = False

        # Sammelt schnelle Reglerbewegungen ein: waehrend man zieht, feuert
        # valueChanged vielfach. Ohne diesen Puffer wuerde fuer jede
        # Zwischenstellung gerechnet und die Oberflaeche haenge.
        self._takt = QTimer(self)
        self._takt.setSingleShot(True)
        self._takt.setInterval(90)
        self._takt.timeout.connect(self._neu_zeichnen)

        self._aufbauen()
        self._region_gewechselt(0)
        # ANMELDEN, damit die Einstellungen die KARTE erreichen und nicht nur
        # die Vorschau. Der Terrain-Reiter holt sie sich von hier ab; siehe
        # get_current_parameters() unten.
        if self.parameter_manager is not None:
            try:
                self.parameter_manager.register_tab("region", self)
            except Exception as fehler:                  # pragma: no cover
                self.logger.warning("Anmeldung beim ParameterManager "
                                    "fehlgeschlagen: %s", fehler)
        self.logger.info("RegionTab initialized")

    # ------------------------------------------------------------------
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Was dieser Reiter zur ERZEUGUNG beitraegt.

        BIS ZUM 2026-08-26 TRUG ER GAR NICHTS BEI. Die Einstellungen lebten
        allein in `self.ueberschreibungen` und wirkten nur auf die Vorschau:
        man konnte eine Region einstellen, "Generieren" druecken - und bekam
        dieselbe Karte wie vorher. Nichts stuerzte ab, nichts meldete sich.
        Gefunden nur, weil nachgesehen wurde, WER die Ueberschreibungen
        liest (niemand).

        Als GESCHACHTELTES Dict unter einem Schluessel, nicht als flache
        `region_<Name>_<Regler>`-Schluessel: die Regionsnamen enthalten
        Leerzeichen, und eine Namensverstuemmelung waere eine zweite
        Kodierung, die irgendwann von der ersten abweicht.
        """
        werte: Dict[str, Any] = {}
        if self.ueberschreibungen:
            werte["regionen_ueberschreibung"] = {
                name: dict(regler)
                for name, regler in self.ueberschreibungen.items()}
        return werte

    # ------------------------------------------------------------------
    def _aufbauen(self):
        """
        DREI WIDGETS, NICHT EINES - so verlangt es die Shell.

        `MapEditor._add_successful_tab()` verteilt jeden Reiter auf drei
        Bereiche:

            self.viewport_stack.addWidget(tab.viewport_widget)
            self.parameter_stack.addWidget(tab.parameter_widget)
            self.statistics_stack.addWidget(tab.statistics_widget)

        Ein erster Entwurf baute stattdessen EIN Widget mit Splitter und
        legte alles hinein. Im laufenden Programm gab das

            AttributeError: 'RegionTab' object has no attribute
                            'viewport_widget'

        und weil `addTab(tab_name)` VOR dieser Zeile laeuft, blieb ein
        Eintrag in der Reiterleiste ohne Inhalt stehen: **jeder folgende
        Reiter war um eins verschoben**, und die Fehlerbehandlung trug den
        Namen ein zweites Mal ein.

        HEADLESS WAR DAS NICHT ZU SEHEN. `RegionTab()` liess sich bauen, und
        alle zwoelf Pruefungen liefen gruen - der Vertrag lebt in der Shell,
        nicht im Reiter. `tests/smoke_test_reiter_vertrag.py` prueft ihn
        jetzt fuer ALLE Reiter.
        """
        self.parameter_widget = self._parameter_bauen()
        self.viewport_widget = self._viewport_bauen()
        self.statistics_widget = self._statistik_bauen()

    def _parameter_bauen(self) -> QWidget:
        seite = QWidget()
        spalte = QVBoxLayout(seite)
        spalte.setContentsMargins(4, 4, 4, 4)

        self.auswahl = QComboBox()
        for _z, _s, r in rw.alle_regionen():
            self.auswahl.addItem(r["name"])
        self.auswahl.currentIndexChanged.connect(self._region_gewechselt)
        kasten = QGroupBox("Region")
        innen = QVBoxLayout(kasten)
        innen.addWidget(self.auswahl)
        hinweis = QLabel(
            "Die Vorschau zeigt den CHARAKTER der Region. Auf der Karte "
            "wird sie zusaetzlich mit ihren Nachbarn verschmolzen und vom "
            "Gebietssystem in der Hoehe verschoben.")
        hinweis.setWordWrap(True)
        innen.addWidget(hinweis)
        spalte.addWidget(kasten)

        kasten2 = QGroupBox("Gelaende")
        innen2 = QVBoxLayout(kasten2)
        for name, beschriftung, mn, mx, schritt in REGIONSREGLER:
            regler = ParameterSlider(label=beschriftung, min_val=mn,
                                     max_val=mx, default_val=mn, step=schritt)
            regler.valueChanged.connect(self._regler_bewegt)
            self.regler[name] = regler
            innen2.addWidget(regler)
        spalte.addWidget(kasten2)

        kasten3 = QGroupBox("Erosionsfilter")
        innen3 = QVBoxLayout(kasten3)
        for name, beschriftung, attr in EROSIONSREGLER:
            c = getattr(EROSION_FILTER, attr)
            regler = ParameterSlider(label=beschriftung, min_val=c["min"],
                                     max_val=c["max"],
                                     default_val=c["default"], step=c["step"],
                                     description=c.get("description", ""))
            regler.valueChanged.connect(self._regler_bewegt)
            self.erosionsregler[name] = regler
            innen3.addWidget(regler)
        spalte.addWidget(kasten3)

        self.kueste_an = QCheckBox("Kuestentypen zeigen")
        self.kueste_an.setToolTip(
            "Legt die Vektorkueste mit den drei Archetypen der Region auf. "
            "Kostet 0.2 bis 1.0 s zusaetzlich - beim Ziehen eines Reglers "
            "besser aus.")
        self.kueste_an.toggled.connect(self._regler_bewegt)
        spalte.addWidget(self.kueste_an)

        zuruecksetzen = QPushButton("Auf Katalogwerte zuruecksetzen")
        zuruecksetzen.clicked.connect(self._zuruecksetzen)
        spalte.addWidget(zuruecksetzen)
        spalte.addStretch(1)
        return seite

    def _viewport_bauen(self) -> QWidget:
        seite = _Vorschauflaeche(self._anzeigen)
        spalte = QVBoxLayout(seite)
        spalte.setContentsMargins(0, 0, 0, 0)

        leiste = QHBoxLayout()
        self.knopf_2d = QPushButton("2D")
        self.knopf_3d = QPushButton("3D")
        for knopf, nummer in ((self.knopf_2d, 0), (self.knopf_3d, 1)):
            knopf.setCheckable(True)
            knopf.setMaximumWidth(60)
            knopf.clicked.connect(
                lambda _a, i=nummer: self._ansicht_wechseln(i))
            leiste.addWidget(knopf)
        self.knopf_2d.setChecked(True)
        leiste.addStretch(1)
        spalte.addLayout(leiste)

        self.stapel = QStackedWidget()
        self.bild = QLabel()
        self.bild.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bild.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Expanding)
        self.stapel.addWidget(self.bild)

        # DIE 3D-ANSICHT, in derselben Aenderung wie die 2D - stehende Regel
        # in CLAUDE.md. Sie bekommt genau dasselbe Hoehenfeld.
        try:
            from gui.widgets.map_display_3d import MapDisplay3DWidget
            self.anzeige_3d = MapDisplay3DWidget()
        except Exception as fehler:                      # pragma: no cover
            self.logger.warning("3D-Ansicht nicht verfuegbar: %s", fehler)
            self.anzeige_3d = QLabel("3D-Ansicht nicht verfuegbar")
        self.stapel.addWidget(self.anzeige_3d)
        spalte.addWidget(self.stapel, 1)
        return seite

    def _statistik_bauen(self) -> QWidget:
        seite = QWidget()
        spalte = QVBoxLayout(seite)
        spalte.setContentsMargins(6, 6, 6, 6)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setAlignment(Qt.AlignmentFlag.AlignTop)
        spalte.addWidget(self.status)

        # ROHER KATALOGDATENSATZ (Ticket #29): zeigt genau die Werte, die
        # jetzt aus core/daten/regionen_welt.toml kommen, statt aus
        # literalem Code in core/terrain_weltkarte.py - read-only, keine
        # Regler. Wer die TOML-Datei aendert und die Regionsauswahl neu
        # trifft, sieht die neuen Zahlen hier ungerechnet.
        katalog_kasten = QGroupBox("Regionskatalog (roh, aus regionen_welt.toml)")
        katalog_spalte = QVBoxLayout(katalog_kasten)
        self.katalog_anzeige = QLabel("")
        self.katalog_anzeige.setWordWrap(True)
        self.katalog_anzeige.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.katalog_anzeige.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        schrift = self.katalog_anzeige.font()
        schrift.setFamily("Consolas")
        self.katalog_anzeige.setFont(schrift)
        katalog_spalte.addWidget(self.katalog_anzeige)
        spalte.addWidget(katalog_kasten)

        spalte.addStretch(1)
        return seite

    # ------------------------------------------------------------------
    def _ansicht_wechseln(self, index: int):
        self.knopf_2d.setChecked(index == 0)
        self.knopf_3d.setChecked(index == 1)
        self.stapel.setCurrentIndex(index)
        self._anzeigen()

    def _regionsname(self) -> str:
        return self.auswahl.currentText()

    def _katalog(self, name: str) -> Dict[str, Any]:
        for _z, _s, r in rw.alle_regionen():
            if r["name"] == name:
                return r
        return {}

    def _region_gewechselt(self, _index: int):
        """Die Regler auf die Werte der gewaehlten Region setzen."""
        name = self._regionsname()
        katalog = self._katalog(name)
        eigene = self.ueberschreibungen.get(name, {})
        self._sperre = True
        try:
            for schluessel, _b, _mn, _mx, _st in REGIONSREGLER:
                wert = eigene.get(schluessel, katalog.get(schluessel))
                if wert is not None:
                    self.regler[schluessel].setValue(float(wert))
        finally:
            self._sperre = False
        self.katalog_anzeige.setText(self._katalogtext(name, katalog))
        self._neu_zeichnen()

    @staticmethod
    def _katalogtext(name: str, katalog: Dict[str, Any]) -> str:
        """Alle Rohfelder der Region plus ihre drei Kuestenarchetypen, so
        wie sie in core/daten/regionen_welt.toml stehen."""
        zeilen = [f"{feld}: {katalog.get(feld)}" for feld in REGIONEN_FELDER
                  if feld in katalog]
        zeilen.append("")
        zeilen.append("Kuestenarchetypen:")
        for archetyp in rw.KUESTEN_ARCHETYPEN.get(name, ()):
            zeilen.append(
                "  {name} - hoehe_faktor {hoehe_faktor}, winkel_grad "
                "{winkel_grad}, kantig {kantig}, strand_anteil "
                "{strand_anteil}, max_anteil {max_anteil}, reichweite_km "
                "{reichweite_km}".format(**archetyp))
        return "\n".join(zeilen)

    def _zuruecksetzen(self):
        self.ueberschreibungen.pop(self._regionsname(), None)
        self._region_gewechselt(self.auswahl.currentIndex())

    def _regler_bewegt(self, *_a):
        if self._sperre:
            return
        # Erst sammeln, dann rechnen - siehe self._takt.
        self._takt.start()

    # ------------------------------------------------------------------
    def _neu_zeichnen(self):
        name = self._regionsname()
        katalog = self._katalog(name)
        eigene = {}
        for schluessel, _b, _mn, _mx, schritt in REGIONSREGLER:
            wert = float(self.regler[schluessel].getValue())
            # Toleranz von einer halben Schrittweite: ein Regler kann seinen
            # Katalogwert nicht genauer treffen als sein Raster, und eine
            # Rundung ist keine Nutzeraenderung.
            if abs(wert - float(katalog.get(schluessel, wert))) > 0.5 * schritt:
                eigene[schluessel] = wert
        if eigene:
            self.ueberschreibungen[name] = eigene
        else:
            self.ueberschreibungen.pop(name, None)

        erosion = {}
        for schluessel, _b, _attr in EROSIONSREGLER:
            wert = float(self.erosionsregler[schluessel].getValue())
            if schluessel == "erosion_scale":
                # Der Filter rechnet RELATIV (Einheitsquadrat), der Regler
                # steht in Metern - siehe ATEF_DEFAULTS.
                wert = wert / max(rw.REGIONSBREITE_M, 1.0)
            elif schluessel == "erosion_octaves":
                wert = int(round(wert))
            erosion[schluessel] = wert

        t0 = time.time()
        try:
            H, _r = rw.regionsfeld(
                name, VORSCHAU_BREITE, VORSCHAU_HOEHE, seed=20260804,
                ueberschreibung=eigene or None, erosion=erosion,
                kueste=bool(self.kueste_an.isChecked()))
        except Exception as fehler:                      # pragma: no cover
            self.logger.error("Regionsvorschau fehlgeschlagen: %s", fehler)
            self.status.setText(f"Fehler: {fehler}")
            return
        self._letztes_feld = np.asarray(H, dtype=np.float64)
        mpp = rw.REGIONSBREITE_M / VORSCHAU_BREITE
        land = self._letztes_feld > 0
        self.status.setText(
            "Region: {}\n{} x {} px\n{:.1f} m/px, {:.1f} km breit\n\n"
            "Land: {:.0f} %\nHoehe: {:.0f} bis {:.0f} m\n\n"
            "gerechnet in {:.2f} s".format(
                name, VORSCHAU_BREITE, VORSCHAU_HOEHE, mpp,
                rw.REGIONSBREITE_M / 1000.0, 100.0 * land.mean(),
                self._letztes_feld.min(), self._letztes_feld.max(),
                time.time() - t0))
        self._anzeigen()

    def _anzeigen(self):
        H = self._letztes_feld
        if H is None:
            return
        if self.stapel.currentIndex() == 1:
            if hasattr(self.anzeige_3d, "update_heightmap"):
                self.anzeige_3d.update_heightmap(H.astype(np.float32),
                                                 "terrain")
            return
        # AUF DIE FLAECHE SKALIEREN, Seitenverhaeltnis erhalten. Ohne das
        # steht die 256x160-Vorschau als Briefmarke in der Mitte.
        # FastTransformation ist hier richtig: die Vorschau soll ihre Pixel
        # ZEIGEN, nicht weichzeichnen - man stellt daran die Feinheit des
        # Gelaendes ein.
        pix = QPixmap.fromImage(self._als_bild(H))
        flaeche = self.bild.size()
        if flaeche.width() > 8 and flaeche.height() > 8:
            pix = pix.scaled(flaeche, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.FastTransformation)
        self.bild.setPixmap(pix)

    @staticmethod
    def _als_bild(H: np.ndarray) -> QImage:
        """
        Hoehenfeld als eingefaerbtes Bild.

        Bewusst hier und nicht ueber MapDisplay2D: dieser Reiter haengt
        nicht am Rechengraphen, und die 2D-Anzeige erwartet Pipeline-Daten.

        FARBBAENDER IN ABSOLUTEN METERN, nicht als Anteil des
        Regionsmaximums. Ein erster Entwurf normierte auf `H.max()` der
        jeweiligen Region - dadurch war das Clonagh (bis 211 m) zur
        Haelfte weiss "verschneit" und sah aus wie das Nevadin (1580 m).
        Eine Vorschau, die dem VERGLEICH dient, darf zwei Regionen mit
        siebenfach verschiedener Hoehe nicht gleich einfaerben.
        """
        land = H > 0
        rgb = np.zeros(H.shape + (3,), dtype=np.float64)
        if (~land).any():
            tiefe = np.clip(-np.minimum(H, 0.0)
                            / max(-float(H.min()), 1.0), 0.0, 1.0)
            rgb[~land] = np.stack([0.16 - 0.10 * tiefe[~land],
                                   0.42 - 0.20 * tiefe[~land],
                                   0.70 - 0.24 * tiefe[~land]], axis=-1)
        if land.any():
            # Tiefland bis 300 m, Bergland bis 1200 m, darueber Fels und
            # Schnee (das Nevadin reicht bis 1580 m).
            hoehe = np.maximum(H, 0.0)
            a = np.clip(hoehe / 300.0, 0.0, 1.0)
            b = np.clip((hoehe - 300.0) / 900.0, 0.0, 1.0)
            c = np.clip((hoehe - 1200.0) / 500.0, 0.0, 1.0)
            mischung = np.stack([0.32 + 0.34 * a + 0.16 * b + 0.32 * c,
                                 0.56 + 0.02 * a - 0.14 * b + 0.50 * c,
                                 0.30 + 0.02 * a - 0.06 * b + 0.72 * c],
                                axis=-1)
            rgb[land] = mischung[land]

        # Schattierung aus dem Gradienten - ohne sie liest sich die Form nicht.
        gy, gx = np.gradient(np.where(land, H, 0.0))
        schatten = np.clip(
            0.5 + 0.55 * (gx - gy) / (1.0 + np.abs(gx) + np.abs(gy)), 0.0, 1.0)
        rgb = np.clip(rgb * (0.55 + 0.75 * schatten[..., None]), 0.0, 1.0)

        daten = (rgb[::-1] * 255).astype(np.uint8).copy()
        hoehe_px, breite_px, _ = daten.shape
        return QImage(daten.data, breite_px, hoehe_px, 3 * breite_px,
                      QImage.Format.Format_RGB888).copy()
