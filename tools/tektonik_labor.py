"""
Path: tools/tektonik_labor.py

LABOR fuer die tektonischen Platten (core/tektonik.py) - live bedienbar.

Nutzer-Vorgabe 2026-08-16: Platten saeen, wachsen lassen, ineinander bewegen,
daraus Falten fuer Gebirge. Dieses Fenster zeigt jede Zwischenstufe einzeln,
damit am BILD entschieden werden kann, ob das Verfahren taugt - bevor
irgendetwas davon in `weltfeld()` wandert.

Bewusst getrennt vom Hauptprogramm: `core/terrain_weltkarte.py` bleibt
unangetastet. Die Regionseichung (smoke_test_regionen_welt.py) ist schon
ohne Tektonik verstimmt (docs/OFFENE_PUNKTE.md 3.9) - kaeme die Tektonik
jetzt in die Pipeline, waere hinterher nicht mehr trennbar, was was
verstellt hat.

DIE ANSICHTEN

    Platten          welche Platte wo liegt, plus Saatpunkte und
                     Bewegungspfeile. Zeigt, ob das Wachstum vernuenftige
                     Formen ergibt oder eine Platte alles schluckt.
    Konvergenz       rot = Platten laufen aufeinander zu (Gebirge),
                     blau = sie laufen auseinander (Graben), weiss = sie
                     schrammen aneinander vorbei.
    Hebung           das Ergebnis in Metern - das eigentliche Produkt.
    Hebung + Relief  die Hebung auf das heutige Weltgelaende addiert, damit
                     sichtbar wird, was sie dort tatsaechlich aendern wuerde.
    Nur Weltgelaende das heutige `weltfeld()` als Vergleichsbild.

Aufruf:
    .venv/Scripts/python.exe tools/tektonik_labor.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QGroupBox,
                             QHBoxLayout, QLabel, QMainWindow, QPushButton,
                             QRadioButton, QSlider, QVBoxLayout, QWidget)

from core.tektonik import tektonik
from core.terrain_weltkarte import WELT_KM, weltfeld

SEED_VORGABE = 20260804
GROESSEN = (256, 384, 512)


def _regler(name, minimum, maximum, wert, nachkomma=0, teiler=1.0):
    behaelter = QWidget()
    layout = QVBoxLayout(behaelter)
    layout.setContentsMargins(0, 2, 0, 2)
    beschriftung = QLabel()
    schieber = QSlider(Qt.Orientation.Horizontal)
    schieber.setMinimum(minimum)
    schieber.setMaximum(maximum)
    schieber.setValue(wert)

    def _text(v):
        beschriftung.setText(f"{name}: {v / teiler:.{nachkomma}f}"
                             if teiler != 1.0 else f"{name}: {v}")

    _text(wert)
    schieber.valueChanged.connect(_text)
    layout.addWidget(beschriftung)
    layout.addWidget(schieber)
    return behaelter, schieber


class TektonikLabor(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tektonik-Labor - Platten, Bewegung, Faltengebirge")
        self.resize(1400, 900)
        self._groesse = 384
        self._gelaende = None
        self._gelaende_schluessel = None
        self._ergebnis = None

        zentral = QWidget()
        aussen = QHBoxLayout(zentral)
        links = QVBoxLayout()
        links.setSpacing(8)

        # --- Ansicht ---
        ansicht_box = QGroupBox("Ansicht")
        ansicht_layout = QVBoxLayout(ansicht_box)
        self.ansicht_gruppe = QButtonGroup(self)
        ansichten = [("Platten + Bewegung", "platten"),
                     ("Konvergenz (rot = Gebirge)", "konvergenz"),
                     ("Hebung (m)", "hebung"),
                     ("Hebung + Weltgelaende", "kombiniert"),
                     ("nur Weltgelaende (Vergleich)", "gelaende")]
        for i, (text, schluessel) in enumerate(ansichten):
            knopf = QRadioButton(text)
            knopf.setProperty("art", schluessel)
            if schluessel == "platten":
                knopf.setChecked(True)
            self.ansicht_gruppe.addButton(knopf, i)
            ansicht_layout.addWidget(knopf)
        self.ansicht_gruppe.idToggled.connect(
            lambda _i, an: self.neu_zeichnen() if an else None)
        links.addWidget(ansicht_box)

        # --- Kartengroesse ---
        groesse_box = QGroupBox("Kartengroesse")
        groesse_layout = QHBoxLayout(groesse_box)
        self.groesse_gruppe = QButtonGroup(self)
        for i, g in enumerate(GROESSEN):
            knopf = QRadioButton(str(g))
            knopf.setProperty("px", g)
            if g == self._groesse:
                knopf.setChecked(True)
            self.groesse_gruppe.addButton(knopf, i)
            groesse_layout.addWidget(knopf)
        links.addWidget(groesse_box)

        # --- Regler ---
        regler_box = QGroupBox("Regler")
        regler_layout = QVBoxLayout(regler_box)
        self.regler = {}
        for name, mini, maxi, wert, nachkomma, teiler, schluessel in [
                ("Plattenzahl", 2, 6, 3, 0, 1.0, "platten"),
                ("Zellzahl", 40, 500, 200, 0, 1.0, "zellen"),
                ("Saat-Mindestabstand", 10, 45, 28, 2, 100.0, "abstand"),
                ("Falten-Reichweite (m)", 400, 6000, 2600, 0, 1.0, "reichweite"),
                ("Falten-Wellenlaenge (m)", 400, 9000, 3600, 0, 1.0, "wellenlaenge"),
                ("Hebung (m)", 100, 2500, 900, 0, 1.0, "hebung"),
                ("Graben (m)", 0, 800, 220, 0, 1.0, "graben"),
                ("Seed-Versatz", 0, 40, 0, 0, 1.0, "seed")]:
            behaelter, schieber = _regler(name, mini, maxi, wert, nachkomma, teiler)
            self.regler[schluessel] = schieber
            regler_layout.addWidget(behaelter)
        links.addWidget(regler_box)

        knopf = QPushButton("Neu rechnen")
        knopf.clicked.connect(self.neu_rechnen)
        links.addWidget(knopf)

        self.gelaende_an = QCheckBox("Weltgelaende mitrechnen (langsamer)")
        self.gelaende_an.setChecked(True)
        links.addWidget(self.gelaende_an)

        self.status = QLabel("...")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.RichText)
        self.status.setMinimumWidth(330)
        links.addWidget(self.status)
        links.addStretch(1)

        rahmen = QWidget()
        rahmen.setLayout(links)
        rahmen.setFixedWidth(370)
        aussen.addWidget(rahmen)

        self.figur = Figure(figsize=(8, 8))
        self.leinwand = FigureCanvasQTAgg(self.figur)
        self.achse = self.figur.add_subplot(111)
        aussen.addWidget(self.leinwand, 1)

        self.setCentralWidget(zentral)

    def showEvent(self, ereignis):
        super().showEvent(ereignis)
        if not getattr(self, "_erstaufbau", False):
            self._erstaufbau = True
            self.neu_rechnen()

    # ------------------------------------------------------------------ #

    def _wert(self, schluessel, teiler=1.0):
        return self.regler[schluessel].value() / teiler

    def _seed(self):
        return SEED_VORGABE + int(self._wert("seed"))

    def _gelaende_holen(self, size, seed):
        """Weltgelaende zwischenspeichern - es kostet mehr als die Tektonik."""
        schluessel = (size, seed)
        if self._gelaende_schluessel != schluessel:
            self.status.setText("Weltgelaende wird gerechnet ...")
            QApplication.processEvents()
            H, _felder = weltfeld(size, seed)
            self._gelaende = np.asarray(H, dtype=np.float64)
            self._gelaende_schluessel = schluessel
        return self._gelaende

    def neu_rechnen(self):
        knopf = self.groesse_gruppe.checkedButton()
        if knopf is not None:
            self._groesse = int(knopf.property("px"))
        size, seed = self._groesse, self._seed()

        self.status.setText("Tektonik wird gerechnet ...")
        QApplication.processEvents()

        t0 = time.time()
        self._ergebnis = tektonik(
            size, seed,
            plattenzahl=int(self._wert("platten")),
            zellzahl=int(self._wert("zellen")),
            mindestabstand=self._wert("abstand", 100.0),
            reichweite_m=self._wert("reichweite"),
            wellenlaenge_m=self._wert("wellenlaenge"),
            hebung_m=self._wert("hebung"),
            graben_m=self._wert("graben"))
        dauer = time.time() - t0

        if self.gelaende_an.isChecked():
            self._gelaende_holen(size, seed)
        self._bericht(dauer)
        self.neu_zeichnen()

    def _bericht(self, dauer):
        e = self._ergebnis
        anteile = " / ".join(f"{a:.0%}" for a in e["plattenanteile"])
        h = e["hebung"]
        zeilen = [f"<b>{e['plattenzahl']} Platten</b> &mdash; {self._groesse} px, "
                  f"Seed {self._seed()}",
                  f"Flaechenanteile: {anteile}",
                  f"Rechenzeit {dauer:.2f} s",
                  f"<b>Hebung:</b> max {h.max():.0f} m, min {h.min():.0f} m",
                  f"Grenzverlauf konvergent: {e['konvergenz_anteil']:.0%}"]
        if not e["konvergenz_gefunden"]:
            zeilen.append("<b style='color:#b00'>Kein ordentliches Gebirge "
                          "gefunden</b> - die Platten schrammen aneinander "
                          "vorbei. Seed-Versatz aendern.")
        if self._gelaende is not None:
            g = self._gelaende
            land = g > 0
            zeilen.append(f"Weltgelaende: Land {land.mean():.0%}, "
                          f"max {g.max():.0f} m")
        self.status.setText("<br>".join(zeilen))

    def neu_zeichnen(self):
        if self._ergebnis is None:
            return
        e = self._ergebnis
        art = self.ansicht_gruppe.checkedButton().property("art")
        # DIE GANZE FIGUR LEEREN, nicht nur die Achse: `figur.colorbar()` legt
        # jedes Mal eine EIGENE Achse an, die `achse.clear()` nicht beruehrt -
        # bei jedem Neuzeichnen kaeme eine weitere Farbleiste dazu und das
        # Bild wuerde Schritt fuer Schritt schmaler.
        self.figur.clear()
        self.achse = self.figur.add_subplot(111)

        if art == "platten":
            self.achse.imshow(e["platte"], cmap="tab10", origin="lower",
                              interpolation="nearest", vmin=0, vmax=9)
            # Grenzen dunkel nachziehen
            self.achse.contour(e["platte"].astype(float), levels=np.arange(
                0.5, e["plattenzahl"], 1.0), colors="black", linewidths=1.2)
            # Bewegungspfeile im Schwerpunkt jeder Platte
            for i in range(e["plattenzahl"]):
                ys, xs = np.nonzero(e["platte"] == i)
                if len(ys) == 0:
                    continue
                cy, cx = ys.mean(), xs.mean()
                vy, vx = e["geschwindigkeit"][i]
                laenge = 0.13 * self._groesse
                self.achse.arrow(cx, cy, vx * laenge, vy * laenge,
                                 width=self._groesse * 0.004,
                                 facecolor="white", edgecolor="black", zorder=5)
            self.achse.set_title("Platten, Wachstum und Bewegungsrichtung")

        elif art == "konvergenz":
            bild = self.achse.imshow(e["konvergenz"], cmap="RdBu_r",
                                     origin="lower", vmin=-1, vmax=1)
            self.figur.colorbar(bild, ax=self.achse, shrink=0.8,
                                label="rot = Faltung, blau = Graben")
            self.achse.set_title("Konvergenz an den Plattengrenzen")

        elif art == "hebung":
            bild = self.achse.imshow(e["hebung"], cmap="terrain", origin="lower")
            self.figur.colorbar(bild, ax=self.achse, shrink=0.8, label="Meter")
            self.achse.set_title("Hebungsfeld - das Faltengebirge")

        elif art == "kombiniert":
            if self._gelaende is None:
                self._gelaende_holen(self._groesse, self._seed())
            summe = self._gelaende + e["hebung"]
            bild = self.achse.imshow(summe, cmap="terrain", origin="lower")
            self.figur.colorbar(bild, ax=self.achse, shrink=0.8, label="Meter")
            self.achse.contour(summe, levels=[0.0], colors="black", linewidths=1.0)
            self.achse.set_title("Weltgelaende + Hebung (Kuestenlinie schwarz)")

        else:  # gelaende
            if self._gelaende is None:
                self._gelaende_holen(self._groesse, self._seed())
            bild = self.achse.imshow(self._gelaende, cmap="terrain", origin="lower")
            self.figur.colorbar(bild, ax=self.achse, shrink=0.8, label="Meter")
            self.achse.contour(self._gelaende, levels=[0.0], colors="black",
                               linewidths=1.0)
            self.achse.set_title("heutiges Weltgelaende, ohne Tektonik")

        self.achse.set_xticks([])
        self.achse.set_yticks([])
        self.figur.tight_layout()
        self.leinwand.draw()


def main():
    app = QApplication(sys.argv)
    fenster = TektonikLabor()
    fenster.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
