"""
Path: tools/mesh_werkstatt.py

WERKSTATT fuer die Terrain-Vernetzung - ein eigenes Fenster, live bedienbar.

Der Nutzer am 2026-08-16: *"koennen wir das ganze einmal in einer testsuite
nachbilden? wir haben die landschaft und koennen in 3d und slidern
verschiedene tesselation oder delaunay triangulation ueber vektoren
nachbilden"* - und davor, mehrfach: *"ich will immer auch mit eigener sicht
testen ob es mir gefaellt."*

Deshalb kein Skript, das Bilder schreibt, sondern ein Fenster mit derselben
3D-Anzeige, die auch die App benutzt (`gui/widgets/map_display_3d.py`) - was
hier zu sehen ist, ist genau das, was spaeter im Programm steht. Links die
Auswahl, rechts das Gelaende zum Drehen.

DIE VIER NETZARTEN

    Quadtree        was die App heute baut. Vertices auf Pixelecken,
                    deshalb Rastertreppe an der Kueste (6.16).
    Remesh          Kanten zusammenfallen lassen nach Quadric Error Metric,
                    Meer und Land getrennt. Vertices frei verschoben (6.33).
    Remesh global   dasselbe ohne die Meer/Land-Trennung - zeigt, was die
                    Trennung eigentlich bringt.
    Delatin         Punkte einfuegen statt Kanten zusammenfallen lassen
                    (6.30). Langsam und schwaecher, aber zum Vergleich da.

WAS DIE WERKSTATT MISST

Unter den Reglern steht nach jedem Bau eine Zeile mit Vertices, Dreiecken,
Bauzeit und - der eigentliche Punkt - dem **Median-Abstand der Vertices zur
naechsten Pixelecke**. Beim Quadtree ist der 0.000000 px; alles darueber
heisst, dass sich die Vertices wirklich vom Raster geloest haben.

Der Hoehenfehler wird auf Wunsch dazugerechnet ("Fehler messen"), getrennt
nach Land und Meer. Er ist absichtlich abschaltbar, weil er das Rastern aller
Dreiecke braucht und bei 512 px ein paar Sekunden kostet.

WICHTIG BEIM MESSEN: der Fehler wird ueber die ECHTEN Dreiecke gerastert,
nicht ueber eine neue Delaunay-Triangulierung der Vertices. Genau dieser
Fehler hat am 2026-08-16 das Remesh zu Unrecht als "5-fach schlechter"
dastehen lassen - eine Neutriangulierung wirft die langgestreckten Dreiecke
entlang der Grate weg und misst ein Netz, das es gar nicht gibt.

Aufruf:
    .venv/Scripts/python.exe tools/mesh_werkstatt.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QGroupBox,
                             QHBoxLayout, QLabel, QMainWindow, QPushButton,
                             QRadioButton, QSlider, QVBoxLayout, QWidget)

from core.terrain_generator import BaseTerrainGenerator
from gui.widgets.adaptive_terrain_mesh import (build_adaptive_mesh,
                                               ist_fuer_adaptives_mesh_geeignet)
from gui.widgets.map_display_3d import MapDisplay3D
from gui.widgets.terrain_contour_relax import baue_konturen_relax
from gui.widgets.terrain_remesh import baue_remesh, remesh_verfuegbar
from managers.data_lod_manager import DataLODManager

WELT_KM = 21.3
SEED = 20260804
GROESSEN = (256, 384, 512)

# LOD 7 erreicht bei jeder unserer GROESSEN die volle Aufloesung - siehe
# BaseTerrainGenerator._lod_level_to_size(): die Verdopplung bricht ab,
# sobald sie target_map_size erreicht/ueberschreitet, und klemmt danach
# fest auf target_map_size. Ein zu kleines LOD (z.B. 5 bei target=384,
# siehe smoke_test_regionen_welt.py) liefert sonst eine kleinere Karte als
# angefordert, ohne Fehler.
_LOD_VOLLE_AUFLOESUNG = 7


def echtes_gelaende(size, seed):
    """
    Heightmap wie sie die 3D-Anzeige der App bekommt: Weltfeld + Erosions-
    filter + Taeleingraben (Weltfluesse) - nicht das rohe `weltfeld()`.

    Vorher rief die Werkstatt `weltfeld()` direkt auf. Das ergibt eine
    ANDERE, glattere Karte: `_calc_redistribution()` haengt bei aktiver
    Weltkarte (`WELTKARTE_AKTIV`) hinter `weltfeld()` noch den
    ATEF-Erosionsfilter (Grate/Rinnen, staerker gewichtet nach Relief -
    genau das, was steile Kuestenklippen ausmacht) und danach das
    Taeleingraben des Flussnetzes an. Ohne diese zwei Schritte fehlen der
    Karte ihre schaerfsten Formen - beobachtet 2026-08-16: "im original
    programm sind noch steile klippen vorhanden [...] die haben wir hier
    nicht. die welt sieht etwas flacher aus."

    Exakt derselbe Weg wie `fertiges_gelaende()` in
    `tests/smoke_test_regionen_welt.py`, das denselben Unterschied schon
    einmal maß (2026-08-10): alleine das Taeleingraben nahm dem Land 4.9°
    mittleren Hang.
    """
    manager = DataLODManager()
    manager.set_map_seed(seed)
    manager.set_map_distance_km(WELT_KM)
    erzeuger = BaseTerrainGenerator(map_seed=seed, data_lod_manager=manager)
    erzeuger.set_active_parameters({
        "map_size": size, "map_seed": seed, "map_distance_km": WELT_KM,
        "amplitude": 100, "redistribute_power": 1.0})
    for knoten in ("terrain.noise", "terrain.redistribution"):
        manager.set_calculator_target_lod(knoten, _LOD_VOLLE_AUFLOESUNG)
    erzeuger._calc_noise("terrain.noise", _LOD_VOLLE_AUFLOESUNG)
    erzeuger._calc_redistribution("terrain.redistribution", _LOD_VOLLE_AUFLOESUNG)
    H = manager.get_calculator_output(
        "terrain.redistribution", "heightmap", _LOD_VOLLE_AUFLOESUNG)
    return np.asarray(H, dtype=np.float32)

# Nach so vielen Sekunden bricht Delatin ab und liefert, was es bis dahin hat.
# Ohne diese Grenze fror das Fenster beim ersten Druck auf "Delatin"
# minutenlang ohne Rueckmeldung ein - es sah aus wie ein Absturz.
DELATIN_ZEITGRENZE_S = 20.0


def _regler(name, minimum, maximum, wert, nachkomma=2, teiler=1.0):
    """Beschrifteter Schieber - dieselbe Bauform wie in flussnetz_werkstatt.py."""
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


def hoehenfehler(punkte_px, hoehen_m, dreiecke, H):
    """
    Abweichung des Netzes von der echten Heightmap, je Rasterpixel.

    Ueber die ECHTEN Dreiecke gerastert - siehe Modulkopf, warum das nicht
    ueber eine Neutriangulierung gehen darf. `matplotlib.tri` scheidet aus,
    weil dessen TrapezoidMapTriFinder das Quadtree-Netz wegen entarteter
    Dreiecke an den Nahtstellen als "invalid" ablehnt.
    """
    size = H.shape[0]
    A, B, C = punkte_px[dreiecke[:, 0]], punkte_px[dreiecke[:, 1]], punkte_px[dreiecke[:, 2]]
    zA = hoehen_m[dreiecke[:, 0]]
    zB = hoehen_m[dreiecke[:, 1]]
    zC = hoehen_m[dreiecke[:, 2]]
    flaeche = ((B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1])
               - (C[:, 0] - A[:, 0]) * (B[:, 1] - A[:, 1]))

    getroffen = np.full((size, size), np.nan)
    tief = lambda v: np.clip(np.floor(v), 0, size - 1).astype(int)
    hoch = lambda v: np.clip(np.ceil(v), 0, size - 1).astype(int)
    x0 = tief(np.minimum.reduce([A[:, 0], B[:, 0], C[:, 0]]))
    x1 = hoch(np.maximum.reduce([A[:, 0], B[:, 0], C[:, 0]]))
    y0 = tief(np.minimum.reduce([A[:, 1], B[:, 1], C[:, 1]]))
    y1 = hoch(np.maximum.reduce([A[:, 1], B[:, 1], C[:, 1]]))

    for t in np.flatnonzero(np.abs(flaeche) > 1e-12):
        gx, gy = np.meshgrid(np.arange(x0[t], x1[t] + 1), np.arange(y0[t], y1[t] + 1))
        if gx.size == 0:
            continue
        w0 = ((B[t, 0] - A[t, 0]) * (gy - A[t, 1])
              - (gx - A[t, 0]) * (B[t, 1] - A[t, 1])) / flaeche[t]
        w1 = ((gx - A[t, 0]) * (C[t, 1] - A[t, 1])
              - (C[t, 0] - A[t, 0]) * (gy - A[t, 1])) / flaeche[t]
        drin = (w0 >= -1e-9) & (w1 >= -1e-9) & (w0 + w1 <= 1 + 1e-9)
        if drin.any():
            wert = zA[t] + w1 * (zB[t] - zA[t]) + w0 * (zC[t] - zA[t])
            getroffen[gy[drin], gx[drin]] = wert[drin]

    gut = np.isfinite(getroffen).ravel()
    fehler = np.abs(getroffen.ravel()[gut] - H.ravel()[gut])
    return fehler, H.ravel()[gut] > 0


class Werkstatt(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mesh-Werkstatt - Terrain-Vernetzung im Vergleich")
        self.resize(1500, 900)
        self._H = None
        self._groesse = 384
        self._baut = False
        self._nicht_verfuegbar = set()

        zentral = QWidget()
        aussen = QHBoxLayout(zentral)
        links = QVBoxLayout()
        links.setSpacing(8)

        # --- Netzart ---
        art_box = QGroupBox("Netzart")
        art_layout = QVBoxLayout(art_box)
        self.art_gruppe = QButtonGroup(self)
        arten = [("Quadtree (wie die App heute)", "quadtree"),
                 ("Remesh - Meer/Land getrennt", "remesh"),
                 ("Remesh - ohne Trennung", "remesh_global"),
                 ("Delatin (Punkte einfuegen)", "delatin"),
                 ("Konturen-Relaxation (Iso-Contours)", "konturen")]
        for i, (text, schluessel) in enumerate(arten):
            knopf = QRadioButton(text)
            knopf.setProperty("art", schluessel)
            if schluessel == "remesh":
                knopf.setChecked(True)
            if schluessel in ("remesh", "remesh_global") and not remesh_verfuegbar():
                knopf.setEnabled(False)
                knopf.setText(text + "  (fast_simplification fehlt)")
                self._nicht_verfuegbar.add(schluessel)
            self.art_gruppe.addButton(knopf, i)
            art_layout.addWidget(knopf)
        self.art_gruppe.idToggled.connect(lambda _i, an: self.neu_bauen() if an else None)
        links.addWidget(art_box)

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
        if self.groesse_gruppe.checkedButton() is None:
            # Passt die Voreinstellung zu keiner angebotenen Groesse, waere
            # sonst gar keine gewaehlt und `gelaende_neu()` benutzte still
            # einen Wert, der auf keinem Knopf steht.
            erster = self.groesse_gruppe.buttons()[0]
            erster.setChecked(True)
            self._groesse = int(erster.property("px"))
        self.groesse_gruppe.idToggled.connect(
            lambda _i, an: self.gelaende_neu() if an else None)
        links.addWidget(groesse_box)

        # --- Regler ---
        regler_box = QGroupBox("Regler")
        regler_layout = QVBoxLayout(regler_box)
        self.regler = {}
        for name, mini, maxi, wert, nachkomma, teiler, schluessel in [
                ("Dreiecke, Anteil vom vollen Gitter", 2, 100, 25, 2, 100.0, "anteil"),
                ("Meer-Budget", 1, 60, 10, 2, 100.0, "meer"),
                ("Tiefenlinie (m)", 2, 200, 25, 0, 1.0, "tiefe"),
                ("Aggressivitaet", 10, 120, 70, 1, 10.0, "agg"),
                ("Quadtree-Toleranz (m)", 5, 400, 60, 1, 10.0, "toleranz"),
                ("Konturen (Anzahl)", 6, 120, 30, 0, 1.0, "konturen_n"),
                ("Kontur-Glaettung (Iterationen)", 0, 40, 10, 0, 1.0, "konturen_iter")]:
            # KEIN automatischer Bau beim Loslassen - der Nutzer will mehrere
            # Regler nacheinander verstellen und dann EINMAL auf "Neu bauen"
            # druecken (Rueckmeldung 2026-08-16: das automatische Rechnen bei
            # jedem Loslassen war Bedienung ohne Grund, der Knopf daneben
            # wirkungslos redundant).
            behaelter, schieber = _regler(name, mini, maxi, wert, nachkomma, teiler)
            self.regler[schluessel] = schieber
            regler_layout.addWidget(behaelter)
        links.addWidget(regler_box)

        # --- Messen ---
        mess_box = QGroupBox("Messen")
        mess_layout = QVBoxLayout(mess_box)
        self.fehler_an = QCheckBox("Hoehenfehler messen (dauert ein paar Sekunden)")
        mess_layout.addWidget(self.fehler_an)
        knopf = QPushButton("Neu bauen")
        knopf.clicked.connect(self.neu_bauen)
        mess_layout.addWidget(knopf)
        vergleich = QPushButton("Alle Netzarten vergleichen (in die Konsole)")
        vergleich.clicked.connect(self.alle_vergleichen)
        mess_layout.addWidget(vergleich)
        links.addWidget(mess_box)

        self.status = QLabel("...")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.RichText)
        self.status.setMinimumWidth(360)
        links.addWidget(self.status)
        links.addStretch(1)

        rahmen = QWidget()
        rahmen.setLayout(links)
        rahmen.setFixedWidth(400)
        aussen.addWidget(rahmen)

        self.anzeige = MapDisplay3D()
        self.anzeige.set_world_size_km(WELT_KM)
        aussen.addWidget(self.anzeige, 1)

        self.setCentralWidget(zentral)
        self.status.setText("Gelaende wird beim Oeffnen gerechnet ...")

    # ------------------------------------------------------------------ #

    def showEvent(self, ereignis):
        """
        Erster Aufbau ERST beim Anzeigen, nicht im Konstruktor.

        Vorher stand `self.gelaende_neu()` am Ende von `__init__` - und
        stuerzte mit `NullFunctionError: glBindVertexArray` ab, weil
        `update_heightmap()` das Mesh sofort in GL-Buffer schreibt, den
        OpenGL-Kontext des Widgets aber erst beim ersten Anzeigen gibt. Der
        Fehler war in der Werkstatt, nicht im Display.
        """
        super().showEvent(ereignis)
        if not getattr(self, "_erstaufbau_erledigt", False):
            self._erstaufbau_erledigt = True
            self.gelaende_neu()

    def _art(self):
        return self.art_gruppe.checkedButton().property("art")

    def _skalen(self):
        """
        Wie `_calculate_terrain_scaling()` im Display (dort Zeile 978/989).

        Hier nachgerechnet statt abgelesen, damit die Kennzahlen auch dann
        stimmen, wenn die 3D-Anzeige noch keinen GL-Kontext hat - sonst
        haenge die Messung an etwas, das mit dem Messen nichts zu tun hat.
        """
        size = max(self._H.shape)
        return 10.0 / size, 10.0 / (WELT_KM * 1000.0)

    def _wert(self, schluessel, teiler=1.0):
        return self.regler[schluessel].value() / teiler

    def gelaende_neu(self):
        knopf = self.groesse_gruppe.checkedButton()
        if knopf is not None:
            self._groesse = int(knopf.property("px"))
        self.status.setText(f"Gelaende {self._groesse} px wird gerechnet ...")
        QApplication.processEvents()
        self._H = echtes_gelaende(self._groesse, SEED)
        try:
            self.anzeige.update_heightmap(self._H, tab_type="terrain")
        except Exception as fehler:                     # noqa: BLE001
            print(f"3D-Anzeige konnte das Gelaende nicht uebernehmen: {fehler}")
        self.neu_bauen()

    def _bauer_fuer(self, art):
        """Liefert die Funktion (heightmap, tsf, ths) -> (v, i, stats)."""
        anteil = self._wert("anteil", 100.0)
        meer = self._wert("meer", 100.0)
        tiefe = -abs(self._wert("tiefe"))
        agg = self._wert("agg", 10.0)
        toleranz = self._wert("toleranz", 10.0)

        if art == "quadtree":
            # Dieselbe Vorpruefung wie map_display_3d._generate_terrain_mesh():
            # ohne sie wuerde die Werkstatt bei 384 px ein Quadtree zeigen, das
            # die App dort gar nicht baut (sie faellt auf das volle Gitter
            # zurueck). Der Vergleich waere dann gegen etwas Erfundenes.
            def _quadtree(H, tsf, ths):
                if not ist_fuer_adaptives_mesh_geeignet(H):
                    return None
                return build_adaptive_mesh(H, tsf, ths, fehler_toleranz_m=toleranz)
            return _quadtree
        if art in ("remesh", "remesh_global"):
            return lambda H, tsf, ths: baue_remesh(
                H, tsf, ths, ziel_anteil=anteil, meer_anteil=meer,
                tiefenlinie_m=tiefe, aggressivitaet=agg,
                zerlegen=(art == "remesh"))
        if art == "delatin":
            from gui.widgets.delatin_mesh import baue_delatin_mesh

            def _delatin(H, tsf, ths):
                voll = 2 * (H.shape[1] - 1) * (H.shape[0] - 1)
                # Delatin waehlt PUNKTE; ungefaehr halb so viele wie Dreiecke.
                return baue_delatin_mesh(
                    H, tsf, ths,
                    max_punkte=max(int(voll * anteil / 2), 100),
                    stichprobe=4,
                    # MIT ZEITGRENZE, sonst friert das Fenster minutenlang ein.
                    # Jede Runde rechnet eine komplette Delaunay-Triangulierung
                    # neu; gemessen 126 s fuer 65954 Punkte bei 512 px. Beim
                    # ersten Druck auf "Delatin" sah genau das wie ein Absturz
                    # aus - keine Rueckmeldung ueber eine Minute.
                    zeitgrenze_s=DELATIN_ZEITGRENZE_S,
                    fortschritt=self._delatin_fortschritt)
            return _delatin
        if art == "konturen":
            konturen_n = int(self._wert("konturen_n"))
            konturen_iter = int(self._wert("konturen_iter"))
            return lambda H, tsf, ths: baue_konturen_relax(
                H, tsf, ths, anzahl_konturen=konturen_n,
                glaettung_iterationen=konturen_iter)
        raise ValueError(art)

    def _delatin_fortschritt(self, runde, punkte, groesster_fehler_m):
        """Zeigt an, dass ueberhaupt etwas passiert - Delatin rechnet lange."""
        self.status.setText(
            f"<b>Delatin</b> rechnet ... Runde {runde}, {punkte} Punkte, "
            f"groesster Fehler noch {groesster_fehler_m:.0f} m"
            f"<br><i>Abbruch nach {DELATIN_ZEITGRENZE_S:.0f} s mit dem, "
            f"was bis dahin da ist.</i>")
        QApplication.processEvents()

    def neu_bauen(self):
        """
        Netz bauen, vermessen, anzeigen - in dieser Reihenfolge.

        Das Netz wird HIER gerechnet und nicht aus dem Display
        zurueckgelesen. So stehen die Kennzahlen auch dann da, wenn die
        3D-Anzeige klemmt - und beim Vermessen wird garantiert dasselbe
        Netz betrachtet, das gleich gezeichnet wird. Doppelt gerechnet wird
        nichts, beide Module speichern ihr Ergebnis ueber den
        Heightmap-Inhalt zwischen.
        """
        if self._H is None or self._baut:
            return
        self._baut = True
        # Bedienung sperren, solange gerechnet wird. Ohne das sieht ein
        # langsames Verfahren (Delatin) wie ein haengendes Programm aus, und
        # jeder weitere Klick stapelt eine zweite Rechnung obendrauf.
        self._bedienung_sperren(True)
        try:
            art = self._art()
            self.status.setText(f"<b>{art}</b> wird gebaut ...")
            QApplication.processEvents()

            bauer = self._bauer_fuer(art)
            tsf, ths = self._skalen()
            t0 = time.time()
            ergebnis = bauer(self._H, tsf, ths)
            dauer = time.time() - t0
            if ergebnis is None:
                self.status.setText(f"<b>{art}: kein Netz entstanden.</b><br>"
                                    "(Quadtree braucht eine Kantenlaenge 2^n+1 - "
                                    "bei 384 px faellt er auf das volle Gitter zurueck.)")
                return

            v, i, _stats = ergebnis

            # DAS FERTIGE NETZ WEITERREICHEN, nicht neu rechnen lassen.
            #
            # Vorher bekam die Anzeige den Bauer selbst und rief ihn ein
            # zweites Mal. Bei Quadtree und Remesh fiel das nicht auf (beide
            # speichern ihr Ergebnis zwischen), bei Delatin schon: es hat
            # keinen Zwischenspeicher, rechnete also alles doppelt - und der
            # Fortschrittstext des zweiten Laufs ueberschrieb dabei den schon
            # fertigen Messbericht in der Statuszeile.
            def _fertig(H, tsf_a, ths_a, _e=ergebnis, _t=tsf, _h=ths):
                if H.shape == self._H.shape and abs(tsf_a - _t) < 1e-12 \
                        and abs(ths_a - _h) < 1e-12:
                    return _e
                # Skalen abweichend - dann muss wirklich neu gerechnet werden
                return bauer(H, tsf_a, ths_a)

            self.status.setText(self._bericht(art, v, i, dauer))

            try:
                self.anzeige.setze_mesh_bauer(_fertig)
            except Exception as fehler:                 # noqa: BLE001
                self.status.setText(self.status.text() +
                                    f"<br><b>3D-Anzeige:</b> {fehler}")
                print(f"3D-Anzeige konnte das Netz nicht uebernehmen: {fehler}")
        finally:
            self._baut = False
            self._bedienung_sperren(False)

    def _bedienung_sperren(self, gesperrt):
        """Regler und Knöpfe waehrend der Rechnung stillegen."""
        for knopf in self.art_gruppe.buttons() + self.groesse_gruppe.buttons():
            # Deaktivierte Netzarten bleiben deaktiviert
            if knopf.property("art") in self._nicht_verfuegbar:
                continue
            knopf.setEnabled(not gesperrt)
        for schieber in self.regler.values():
            schieber.setEnabled(not gesperrt)
        QApplication.processEvents()

    def _kennzahlen(self, v, i):
        """Vertices in Pixelkoordinaten, Hoehen in Metern, Dreiecke."""
        size = self._H.shape[0]
        tsf, ths = self._skalen()
        V = np.asarray(v, dtype=np.float64).reshape(-1, 8)
        px = (V[:, 0] / (size * tsf) + 0.5) * (size - 1)
        py = (V[:, 2] / (size * tsf) + 0.5) * (size - 1)
        return (np.stack([px, py], axis=1), V[:, 1] / ths,
                np.asarray(i, dtype=np.int64).reshape(-1, 3))

    def _bericht(self, art, v, i, dauer):
        punkte_px, hoehen_m, dreiecke = self._kennzahlen(v, i)
        size = self._H.shape[0]
        voll_v = size * size
        voll_d = 2 * (size - 1) * (size - 1)
        abstand = np.hypot(punkte_px[:, 0] - np.round(punkte_px[:, 0]),
                           punkte_px[:, 1] - np.round(punkte_px[:, 1]))
        ueber_land = self._H[np.clip(np.round(punkte_px[:, 1]).astype(int), 0, size - 1),
                             np.clip(np.round(punkte_px[:, 0]).astype(int), 0, size - 1)] > 0

        zeilen = [f"<b>{art}</b> &mdash; {self._groesse} px",
                  f"{len(punkte_px)} Vertices ({len(punkte_px)/voll_v:.1%} vom vollen Gitter)",
                  f"{len(dreiecke)} Dreiecke ({len(dreiecke)/voll_d:.1%})",
                  f"Bauzeit {dauer:.2f} s",
                  f"<b>Abstand zur Pixelecke:</b> Median {np.median(abstand):.6f} px",
                  f"frei verschoben: {(abstand > 0.01).mean():.1%} der Vertices",
                  f"ueber Land: {ueber_land.mean():.1%} (Land ist {(self._H>0).mean():.1%} der Karte)"]

        if art == "konturen":
            # Sonst liest sich "0.0% frei verschoben" wie ein Fehlschlag -
            # ist hier aber Absicht: dieses Verfahren aendert die HOEHE, nicht
            # die XZ-Position der Vertices (siehe terrain_contour_relax.py).
            zeilen.append("<i>Vertex-Positionen bleiben absichtlich auf dem "
                          "Raster - hier wird stattdessen die Hoehe selbst "
                          "an geglaettete Konturen angepasst.</i>")

        if self.fehler_an.isChecked():
            self.status.setText("<br>".join(zeilen) + "<br><i>Fehler wird gemessen ...</i>")
            QApplication.processEvents()
            t0 = time.time()
            fehler, land = hoehenfehler(punkte_px, hoehen_m, dreiecke, self._H.astype(np.float64))
            zeilen.append("<b>Hoehenfehler</b> "
                          f"(in {time.time()-t0:.1f} s gemessen)")
            for etikett, maske in (("gesamt", np.ones_like(land)),
                                   ("Land", land), ("Meer", ~land)):
                f = fehler[maske]
                if f.size:
                    zeilen.append(f"&nbsp;&nbsp;{etikett}: RMS {np.sqrt((f**2).mean()):.2f} m, "
                                  f"p99 {np.percentile(f, 99):.2f} m, max {f.max():.1f} m")
        return "<br>".join(zeilen)

    def alle_vergleichen(self):
        """Alle Netzarten nacheinander, Ergebnis als Tabelle in die Konsole."""
        if self._H is None:
            return
        arten = ["quadtree", "remesh", "remesh_global"]
        if remesh_verfuegbar() is False:
            arten = ["quadtree"]
        arten.append("delatin")
        arten.append("konturen")
        print(f"\n{'Netzart':<16}{'Vertices':>9}{'Dreiecke':>10}{'Zeit':>8}"
              f"{'Versatz':>10}{'RMS Land':>10}{'p99 Land':>10}{'RMS Meer':>10}")
        print("-" * 83)
        for art in arten:
            self.status.setText(f"Vergleich: {art} ...")
            QApplication.processEvents()
            try:
                tsf, ths = self._skalen()
                t0 = time.time()
                ergebnis = self._bauer_fuer(art)(self._H, tsf, ths)
                dauer = time.time() - t0
                if ergebnis is None:
                    print(f"{art:<16}  kein Netz (bei dieser Kartengroesse nicht anwendbar)")
                    continue
                punkte_px, hoehen_m, dreiecke = self._kennzahlen(ergebnis[0], ergebnis[1])
                ab = np.hypot(punkte_px[:, 0] - np.round(punkte_px[:, 0]),
                              punkte_px[:, 1] - np.round(punkte_px[:, 1]))
                fehler, land = hoehenfehler(punkte_px, hoehen_m, dreiecke,
                                            self._H.astype(np.float64))
                fl, fm = fehler[land], fehler[~land]
                print(f"{art:<16}{len(punkte_px):>9}{len(dreiecke):>10}{dauer:>7.2f}s"
                      f"{np.median(ab):>10.4f}"
                      f"{np.sqrt((fl**2).mean()):>9.2f}m"
                      f"{np.percentile(fl, 99):>9.2f}m"
                      f"{np.sqrt((fm**2).mean()):>9.2f}m")
            except Exception as fehler_text:            # noqa: BLE001
                print(f"{art:<16}  FEHLER: {fehler_text}")
        print()
        self.neu_bauen()


def main():
    app = QApplication(sys.argv)
    fenster = Werkstatt()
    fenster.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
