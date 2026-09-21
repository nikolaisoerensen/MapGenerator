"""
Path: gui/tabs/kontinent_tab.py

Der Kontinentreiter - hier wird die Grundform der Landmasse eingestellt.

NUTZERENTWURF 2026-08-26 (docs/archiv/2026-08-25_AUFRAEUMPLAN.md 4.10):

    *"dann gehts in 'kontinent' sicht mit map size, form des kontinents und
    solcher sachen. zB laesst sich hier ein eher runder kontinent erstellen
    oder aber einer mit vielen armen (also die grundformen als slider, links
    rund, mitte laenglich, rechts mit vielen auslaeufern)."*

WAS DIESER REITER BESITZT - UND WAS NICHT. Er besitzt genau EINEN
Parameter: `kontinentform`. `map_size` und `map_distance_km` gehoeren dem
Terrain-Reiter; sie hier ZUSAETZLICH anzulegen waere ein Doppelschluessel,
und `tests/smoke_test_parameter_eindeutig.py` lehnt das zu Recht ab. Ein
Umzug waere ein VERSCHIEBEN, kein Kopieren - eigener Schritt.

Auch der Seed bleibt beim Terrain-Reiter: er gehoert zur Karte, nicht zu
dieser Ansicht. Die Vorschau liest ihn von dort, wenn sie ihn erreicht.

ZWEI ANSICHTEN MIT VERSCHIEDENEN KOSTEN, und das ist Absicht:

    2D   nur die Kontinentform plus die neun Regionen   0.34 - 0.49 s
    3D   ein VOLLES weltfeld() bei 192 px               rund 1.1 s

Die 2D-Ansicht ist die Arbeitsansicht - sie zeigt genau das, was der Regler
formt, und bleibt beim Ziehen fluessig. Die 3D-Ansicht beantwortet die
andere Frage ("was wird daraus?") und rechnet dafuer die ganze Welt; sie
laeuft deshalb nur, wenn man sie anfordert.
"""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QGroupBox, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QStackedWidget,
                             QVBoxLayout, QWidget)

import core.terrain_weltkarte as rw
from gui.widgets.widgets import ParameterSlider

# Vorschauaufloesung fuer die 2D-Ansicht. 256 px auf 21.3 km sind 83 m/px -
# grob, aber die Kontinentform ist eine Sache von Kilometern.
VORSCHAU_PX = 256
# Aufloesung fuer die 3D-Ansicht. Ein volles weltfeld() kostet dort rund
# 1.1 s; bei 256 waeren es schon 2.1 s.
VORSCHAU_3D_PX = 192

# Der Formregler. 0 rund, 0.5 laenglich, 1 viele Auslaeufer - siehe
# `_kontinent_gestalt()` in core/terrain_weltkarte.py.
#
# VORGABE 0.5 WAERE FALSCH. `form=None` erzeugt die Gestalt, gegen die alle
# Eichungen laufen, und die liegt NICHT auf der Reglerkurve. Der Regler
# beginnt deshalb ausgeschaltet: erst ein Haken schaltet ihn scharf.
FORM_MIN, FORM_MAX, FORM_SCHRITT = 0.0, 1.0, 0.05
FORM_VORGABE = 0.35


class _Vorschauflaeche(QWidget):
    """Traegt die Vorschau und meldet jede Groessenaenderung."""

    def __init__(self, beim_wachsen):
        super().__init__()
        self._beim_wachsen = beim_wachsen

    def resizeEvent(self, ereignis):
        super().resizeEvent(ereignis)
        self._beim_wachsen()


class KontinentTab(QWidget):
    """Kontinentform mit Live-Vorschau."""

    def __init__(self, data_lod_manager=None, parameter_manager=None,
                 navigation_manager=None, shader_manager=None,
                 generation_orchestrator=None, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("KontinentTab")
        self.parameter_manager = parameter_manager
        self.shader_manager = shader_manager

        self._maske: Optional[np.ndarray] = None
        self._gewichte: Optional[np.ndarray] = None
        self._feld_3d: Optional[np.ndarray] = None

        self._takt = QTimer(self)
        self._takt.setSingleShot(True)
        self._takt.setInterval(90)
        self._takt.timeout.connect(self._neu_zeichnen)

        self._aufbauen()
        if self.parameter_manager is not None:
            try:
                self.parameter_manager.register_tab("kontinent", self)
            except Exception as fehler:                  # pragma: no cover
                self.logger.warning("Anmeldung fehlgeschlagen: %s", fehler)
        self._neu_zeichnen()
        self.logger.info("KontinentTab initialized")

    # ------------------------------------------------------------------
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Was dieser Reiter zur Erzeugung beitraegt: der Formregler, und zwar
        NUR wenn er eingeschaltet ist.

        Ausgeschaltet liefert er nichts - dann bleibt es bei `form=None` in
        `kontinentform()`, also der Gestalt, gegen die alle Eichungen laufen
        (Regionsflaechen, Wasseranteile, smoke_test_regionen_welt). Ein
        Reiter, der beim blossen Oeffnen eine Form erzwingt, haette sie alle
        still verschoben.
        """
        if not self.form_an.isChecked():
            return {}
        return {"kontinentform": float(self.form.getValue())}

    # ------------------------------------------------------------------
    def _aufbauen(self):
        self.parameter_widget = self._parameter_bauen()
        self.viewport_widget = self._viewport_bauen()
        self.statistics_widget = self._statistik_bauen()

    def _parameter_bauen(self) -> QWidget:
        seite = QWidget()
        spalte = QVBoxLayout(seite)
        spalte.setContentsMargins(4, 4, 4, 4)

        kasten = QGroupBox("Grundform")
        innen = QVBoxLayout(kasten)
        self.form_an = QCheckBox("Form selbst bestimmen")
        self.form_an.setToolTip(
            "Aus: der Kontinent behaelt die eingemessene Gestalt, gegen die "
            "alle Regionsflaechen und Wasseranteile geeicht sind. Ein Haken "
            "uebernimmt den Regler darunter.")
        self.form_an.toggled.connect(self._regler_bewegt)
        innen.addWidget(self.form_an)

        self.form = ParameterSlider(
            label="rund  <->  laenglich  <->  Auslaeufer",
            min_val=FORM_MIN, max_val=FORM_MAX, default_val=FORM_VORGABE,
            step=FORM_SCHRITT,
            description="Links ein kompakter runder Kontinent, in der Mitte "
                        "ein laenglicher, rechts einer mit vielen "
                        "Halbinseln. Die Landflaeche bleibt bei jeder "
                        "Stellung gleich.")
        self.form.valueChanged.connect(self._regler_bewegt)
        innen.addWidget(self.form)
        spalte.addWidget(kasten)

        kasten2 = QGroupBox("Anzeige")
        innen2 = QVBoxLayout(kasten2)
        self.regionen_an = QCheckBox("Regionen einfaerben")
        self.regionen_an.setChecked(True)
        self.regionen_an.toggled.connect(self._regler_bewegt)
        innen2.addWidget(self.regionen_an)
        hinweis = QLabel(
            "Kartengroesse, Ausdehnung und Seed gehoeren dem Terrain-Reiter "
            "- sie stehen hier bewusst nicht ein zweites Mal.")
        hinweis.setWordWrap(True)
        innen2.addWidget(hinweis)
        spalte.addWidget(kasten2)
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
        self.knopf_3d.setToolTip(
            "Rechnet ein volles Gelaende mit dieser Kontinentform "
            "(rund 1.1 s) - langsamer als die 2D-Ansicht, dafuer sieht man, "
            "was aus der Form wird.")
        leiste.addStretch(1)
        spalte.addLayout(leiste)

        self.stapel = QStackedWidget()
        self.bild = QLabel()
        self.bild.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bild.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Expanding)
        self.stapel.addWidget(self.bild)
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
        spalte.addStretch(1)
        return seite

    # ------------------------------------------------------------------
    def _seed(self) -> int:
        """Der Seed der KARTE, nicht ein eigener."""
        if self.parameter_manager is not None:
            try:
                wert = self.parameter_manager.get_tab_parameters(
                    "terrain").get("map_seed")
                if wert is not None:
                    return int(wert)
            except Exception:                            # pragma: no cover
                pass
        return 20260804

    def _form(self):
        return (float(self.form.getValue())
                if self.form_an.isChecked() else None)

    def _regler_bewegt(self, *_a):
        self._takt.start()

    def _ansicht_wechseln(self, index: int):
        self.knopf_2d.setChecked(index == 0)
        self.knopf_3d.setChecked(index == 1)
        self.stapel.setCurrentIndex(index)
        if index == 1:
            self._feld_3d = None          # beim Wechsel frisch rechnen
        self._anzeigen()

    # ------------------------------------------------------------------
    def _neu_zeichnen(self):
        seed, form = self._seed(), self._form()
        t0 = time.time()
        try:
            self._maske, _sdf = rw.kontinentform(VORSCHAU_PX, seed, form=form)
            self._gewichte = None
            if self.regionen_an.isChecked():
                self._gewichte, _e = rw.voronoi_regionen(
                    self._maske, seed, punktzahl=200)
        except Exception as fehler:                      # pragma: no cover
            self.logger.error("Kontinentvorschau fehlgeschlagen: %s", fehler)
            self.status.setText(f"Fehler: {fehler}")
            return
        self._feld_3d = None
        land = float(self._maske.mean())
        self.status.setText(
            "Form: {}\nSeed: {}\n{} x {} px, {:.0f} m/px\n\n"
            "Landanteil: {:.1f} %\n\ngerechnet in {:.2f} s".format(
                "eingemessene Vorgabe" if form is None else f"{form:.2f}",
                seed, VORSCHAU_PX, VORSCHAU_PX,
                rw.WELT_KM * 1000.0 / VORSCHAU_PX, 100.0 * land,
                time.time() - t0))
        self._anzeigen()

    def _anzeigen(self):
        if self.stapel.currentIndex() == 1:
            self._anzeigen_3d()
            return
        if self._maske is None:
            return
        pix = QPixmap.fromImage(self._als_bild(self._maske, self._gewichte))
        flaeche = self.bild.size()
        if flaeche.width() > 8 and flaeche.height() > 8:
            pix = pix.scaled(flaeche, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.FastTransformation)
        self.bild.setPixmap(pix)

    def _anzeigen_3d(self):
        if not hasattr(self.anzeige_3d, "update_heightmap"):
            return
        if self._feld_3d is None:
            t0 = time.time()
            try:
                H, _f = rw.weltfeld(VORSCHAU_3D_PX, self._seed(),
                                    kontinentform_regler=self._form())
            except Exception as fehler:                  # pragma: no cover
                self.logger.error("3D-Vorschau fehlgeschlagen: %s", fehler)
                return
            self._feld_3d = np.asarray(H, dtype=np.float32)
            self.status.setText(
                self.status.text()
                + "\n\n3D: volles Gelaende bei {} px, {:.1f} s".format(
                    VORSCHAU_3D_PX, time.time() - t0))
        self.anzeige_3d.update_heightmap(self._feld_3d, "terrain")

    @staticmethod
    def _als_bild(maske: np.ndarray,
                  gewichte: Optional[np.ndarray]) -> QImage:
        """Landmaske, wahlweise nach fuehrender Region eingefaerbt."""
        rgb = np.zeros(maske.shape + (3,), dtype=np.float64)
        rgb[~maske] = (0.10, 0.26, 0.46)
        if gewichte is None:
            rgb[maske] = (0.74, 0.70, 0.54)
        else:
            from gui.config.gui_default import ColorSchemes
            fuehrend = np.argmax(gewichte, axis=0)
            farben = []
            for _z, _s, r in rw.alle_regionen():
                h = r.get("farbe", "#888888").lstrip("#")
                farben.append([int(h[i:i + 2], 16) / 255.0
                               for i in (0, 2, 4)])
            tafel = np.array(farben, dtype=np.float64)
            rgb[maske] = tafel[fuehrend][maske]
            # Regionsgrenzen als helle Linie - ohne sie verschwimmen
            # benachbarte Regionen mit aehnlicher Katalogfarbe.
            from scipy import ndimage
            kante = (ndimage.maximum_filter(fuehrend, 3)
                     != ndimage.minimum_filter(fuehrend, 3))
            rgb[kante & maske] = (0.95, 0.95, 0.92)

        daten = (np.clip(rgb, 0, 1)[::-1] * 255).astype(np.uint8).copy()
        hoehe, breite, _ = daten.shape
        return QImage(daten.data, breite, hoehe, 3 * breite,
                      QImage.Format.Format_RGB888).copy()
