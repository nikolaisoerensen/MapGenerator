"""
Path: tools/vektor_kueste_labor.py

3D-LABOR fuer die Vektor-Kueste (core/vektor_kueste.py).

Nutzerfrage 2026-08-17: *"ist es moeglich den vergleich in 3D anzusehen?"*

Zeigt dieselbe Welt in drei Fassungen in der ECHTEN 3D-Anzeige der App
(`gui/widgets/map_display_3d.py`), umschaltbar ohne Neustart:

    A - Basis            das Rohgelaende, ohne Kuestenformung.
    B - Vektor gerastert die Vektoren auf die Pixelmitten abgetastet. Das
                         ist die Heightmap, die spaeter Hydrologie, Biome,
                         Siedlung und Export bekaemen.
    frei abgetastet      DIESELBE Hoehenfunktion, aber auf einem N-fach
                         feineren Gitter abgetastet - also das, was ein
                         nicht rastergebundenes Mesh sehen wuerde.

WAS DER VERGLEICH ZEIGT UND WAS NICHT

Er zeigt, wieviel Hoehendetail die Vektorbeschreibung TRAEGT, das im
Pixelraster verlorengeht - an der Klippe sind das gemessen bis 195 m
(tests/smoke_test_vektor_kueste.py, Gruppe `rasterfreiheit`).

Er zeigt NOCH NICHT die geloeste Kuestensilhouette. Dafuer muessten die
Vertices auf der Kuestenlinie selbst sitzen (Mesh-Schnitt entlang der
Polylinie); hier sitzen sie weiterhin auf einem - nur feineren - Gitter.
Die Treppe wird dadurch kleiner, aber sie verschwindet nicht. Das ist der
naechste Bauschritt, nicht ein Fehler dieses Labors.

Aufruf:
    .venv/Scripts/python.exe tools/vektor_kueste_labor.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QGroupBox,
                             QHBoxLayout, QLabel, QMainWindow, QPushButton,
                             QRadioButton, QSlider, QVBoxLayout, QWidget)

from core.vektor_kueste import (MESH_MINDEST_SKALA_M, VektorKueste,
                                als_raster, an_punkten)
from gui.widgets.map_display_3d import MapDisplay3D

WELT_KM = 21.3
SEED = 20260804
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


class VektorKuesteLabor(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vektor-Kueste in 3D - Basis / gerastert / frei")
        self.resize(1500, 900)
        self._groesse = 384
        self._basis = None
        self._felder = None
        self._vk = None
        self._schluessel = None

        zentral = QWidget()
        aussen = QHBoxLayout(zentral)
        links = QVBoxLayout()
        links.setSpacing(8)

        fassung_box = QGroupBox("Fassung")
        fassung_layout = QVBoxLayout(fassung_box)
        self.fassung_gruppe = QButtonGroup(self)
        for i, (text, schluessel) in enumerate([
                ("A - Basis (ohne Kuestenformung)", "basis"),
                ("B - Vektor gerastert (Pixelmitten)", "raster"),
                ("frei abgetastet (feineres Gitter)", "frei"),
                ("GESCHNITTEN - Kueste als Meshkante", "schnitt")]):
            knopf = QRadioButton(text)
            knopf.setProperty("art", schluessel)
            if schluessel == "raster":
                knopf.setChecked(True)
            self.fassung_gruppe.addButton(knopf, i)
            fassung_layout.addWidget(knopf)
        self.fassung_gruppe.idToggled.connect(
            lambda _i, an: self.anzeigen() if an else None)
        links.addWidget(fassung_box)

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

        regler_box = QGroupBox("Regler")
        regler_layout = QVBoxLayout(regler_box)
        self.regler = {}
        for name, mini, maxi, wert, nk, teiler, key in [
                ("Uebersampling (nur 'frei')", 1, 4, 2, 0, 1.0, "uebersampling"),
                ("Mindest-Anstiegsstrecke (m)", 1, 200, int(MESH_MINDEST_SKALA_M),
                 0, 1.0, "skala")]:
            behaelter, schieber = _regler(name, mini, maxi, wert, nk, teiler)
            self.regler[key] = schieber
            regler_layout.addWidget(behaelter)
        links.addWidget(regler_box)

        knopf = QPushButton("Neu rechnen")
        knopf.clicked.connect(self.neu_rechnen)
        links.addWidget(knopf)

        self.status = QLabel("...")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.RichText)
        self.status.setMinimumWidth(340)
        links.addWidget(self.status)
        links.addStretch(1)

        rahmen = QWidget()
        rahmen.setLayout(links)
        rahmen.setFixedWidth(380)
        aussen.addWidget(rahmen)

        self.anzeige = MapDisplay3D()
        self.anzeige.set_world_size_km(WELT_KM)
        aussen.addWidget(self.anzeige, 1)
        self.setCentralWidget(zentral)

    def showEvent(self, ereignis):
        """
        Erstaufbau ERST beim Anzeigen - `update_heightmap()` schreibt sofort
        in GL-Buffer, und den Kontext des Widgets gibt es erst jetzt. Im
        Konstruktor stuerzte genau das in der Mesh-Werkstatt mit
        `NullFunctionError: glBindVertexArray` ab.
        """
        super().showEvent(ereignis)
        if not getattr(self, "_erstaufbau", False):
            self._erstaufbau = True
            self.neu_rechnen()

    # ------------------------------------------------------------------ #

    def _wert(self, key):
        return self.regler[key].value()

    def _art(self):
        return self.fassung_gruppe.checkedButton().property("art")

    def neu_rechnen(self):
        knopf = self.groesse_gruppe.checkedButton()
        if knopf is not None:
            self._groesse = int(knopf.property("px"))
        schluessel = (self._groesse, SEED)
        if self._schluessel != schluessel:
            self.status.setText(f"Weltgelaende {self._groesse} px wird "
                                f"gerechnet ...")
            QApplication.processEvents()
            from core.terrain_weltkarte import weltfeld
            H, felder = weltfeld(self._groesse, SEED)
            self._basis = np.asarray(H, dtype=np.float64)
            self._felder = felder
            self._vk = VektorKueste(self._basis, felder["regionen"], SEED)
            self._schluessel = schluessel
        self.anzeigen()

    def anzeigen(self):
        if self._vk is None:
            return
        art = self._art()
        self.status.setText(f"<b>{art}</b> wird abgetastet ...")
        QApplication.processEvents()

        if art == "schnitt":
            self._schnitt_anzeigen()
            return

        # Ein etwaiger Schnitt-Bauer von vorher muss weg, sonst zeigt die
        # Anzeige weiter das geschnittene Netz, egal welche Heightmap kommt.
        try:
            self.anzeige.setze_mesh_bauer(None)
        except Exception:                                    # noqa: BLE001
            pass

        t0 = time.time()
        if art == "basis":
            feld = self._basis
            hinweis = "Rohgelaende, keine Kuestenformung."
        elif art == "raster":
            feld = als_raster(self._vk)
            hinweis = (f"Vektoren auf Pixelmitten, Mindestskala "
                       f"{2.0 * self._vk.mpp:.0f} m (= 2 Pixel).")
        else:
            faktor = int(self._wert("uebersampling"))
            n = self._groesse * faktor
            # DIESELBE Welt, nur feiner abgetastet: die Achsen laufen weiter
            # ueber 0..size-1 der Originalkarte, nur mit n Stuetzstellen.
            achse = np.linspace(0.0, self._groesse - 1.0, n)
            fx, fy = np.meshgrid(achse, achse)
            feld = an_punkten(self._vk, fx, fy,
                              mindest_skala_m=float(self._wert("skala")))
            hinweis = (f"{faktor}x feiner ({n} px), Mindestskala "
                       f"{self._wert('skala')} m - dieselbe Funktion, "
                       f"nur dichter abgetastet.")
        dauer = time.time() - t0

        feld = np.asarray(feld, dtype=np.float32)
        try:
            self.anzeige.update_heightmap(feld, tab_type="terrain")
        except Exception as fehler:                          # noqa: BLE001
            print(f"3D-Anzeige: {fehler}")
            self.status.setText(self.status.text() +
                                f"<br><b>3D-Anzeige:</b> {fehler}")
            return

        land = feld > 0
        mpp = WELT_KM * 1000.0 / feld.shape[0]
        gy, gx = np.gradient(feld.astype(np.float64), mpp)
        hang = np.degrees(np.arctan(np.hypot(gx, gy)))
        self.status.setText(
            f"<b>{art}</b> &mdash; {feld.shape[0]} px<br>"
            f"{hinweis}<br>"
            f"Abtastzeit {dauer:.2f} s<br>"
            f"Hoehe {feld.min():.0f} .. {feld.max():.0f} m<br>"
            f"Land {land.mean():.1%}<br>"
            f"<b>Hang auf Land:</b> Median {np.median(hang[land]):.1f} Grad, "
            f"p99 {np.percentile(hang[land], 99):.1f} Grad<br>"
            f"<i>In DIESER Fassung sitzen alle Vertices auf dem Gitter - die "
            f"Kuestensilhouette ist also rastergebunden. Fassung "
            f"'GESCHNITTEN' loest sie davon.</i>")


    def _schnitt_anzeigen(self):
        """
        Das GESCHNITTENE Netz: Kuestenlinie als echte Meshkante.

        Geht ueber `setze_mesh_bauer()` statt ueber eine Heightmap - der
        Schnitt ist Geometrie, keine Rasterkarte, und genau das ist der
        Punkt. Die Anzeige bekommt trotzdem eine Heightmap (fuer Farbskala
        und Schattierung), das Netz baut aber der Bauer.
        """
        from gui.widgets.kuesten_schnitt import baue_schnitt_mesh

        t0 = time.time()
        skala = float(self._wert("skala"))

        def _bauer(H, tsf, ths):
            return baue_schnitt_mesh(
                H, tsf, ths,
                hoehen_fn=lambda x, y: an_punkten(self._vk, x, y,
                                                  mindest_skala_m=skala))

        feld = als_raster(self._vk).astype(np.float32)

        # ERST RECHNEN UND BERICHTEN, DANN ANZEIGEN. Andersherum verschluckt
        # ein Anzeigefehler (headless: kein GL-Kontext) auch die Kennzahlen,
        # obwohl das Netz selbst tadellos gebaut wurde - dieselbe Lehre wie
        # in tools/mesh_werkstatt.py.
        tsf = getattr(self.anzeige, "terrain_scale_factor", None) or (
            10.0 / feld.shape[0])
        ths = getattr(self.anzeige, "terrain_height_scale", None) or (
            10.0 / (WELT_KM * 1000.0))
        ergebnis = _bauer(feld, tsf, ths)
        dauer = time.time() - t0

        if ergebnis is None:
            self.status.setText("<b>Schnitt:</b> kein Netz entstanden.")
            return
        _v, _i, s = ergebnis
        voll = s["voll_dreiecke"]
        bericht = (
            f"<b>GESCHNITTEN</b> &mdash; {feld.shape[0]} px<br>"
            f"Die Kuestenlinie ist eine echte Meshkante.<br>"
            f"Bauzeit {dauer:.2f} s<br>"
            f"{s['dreiecke']} Dreiecke ({s['dreiecke']/voll - 1:+.1%} "
            f"gegenueber dem vollen Gitter)<br>"
            f"<b>{s['konturvertices']} Konturvertices</b>, Median-Versatz "
            f"zur Pixelecke {s['versatz_median_px']:.4f} px<br>"
            f"<i>Zum Vergleich: Quadtree und Gittervertices liegen bei "
            f"0.0000 px, also exakt auf der Ecke.</i>")
        self.status.setText(bericht)

        try:
            self.anzeige.setze_mesh_bauer(None)
            self.anzeige.update_heightmap(feld, tab_type="terrain")
            self.anzeige.setze_mesh_bauer(_bauer)
        except Exception as fehler:                          # noqa: BLE001
            print(f"3D-Anzeige: {fehler}")
            self.status.setText(bericht + f"<br><b>3D-Anzeige:</b> {fehler}")


def main():
    app = QApplication(sys.argv)
    fenster = VektorKuesteLabor()
    fenster.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
