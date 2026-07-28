"""
Path: core/erosion_generator.py

Hydraulische Erosion als FELDVERFAHREN (Euler) - Portierung des Modells aus
https://github.com/LanLou123/Webgl-Erosion, das seinerseits auf
Mei/Decaudin/Hu 2007 ("Fast Hydraulic Erosion Simulation and Visualization on
GPU") und Št'ava et al. ("Interactive Terrain Modeling Using Hydraulic
Erosion") aufbaut.

ABGRENZUNG ZUM BISHERIGEN VERFAHREN
-----------------------------------
Bis 2026-07-28 lief die Erosion als PARTIKELVERFAHREN (Lagrange) in
core/water_generator.py DropletErosionSystem: einzelne Tropfen laufen über das
Gelände, tragen Sediment mit sich und geben es wieder ab. Das Verfahren ist
nach zwei Optimierungsrunden messbar gut (0 Krater, Kanalnetz 757 px bei
512², höchste Geländenadel 17.0 m), erkauft das aber mit Mechanik, die es hier
gar nicht braucht:

  * Senkenfüllung zwischen den Durchgängen, damit Partikel nicht in
    selbstgegrabenen Gruben enden
  * eine Aufnahmefähigkeits-Grenze pro Zelle gegen Lockstep-Kollisionen
    (bis zu 541 Partikel auf einer Zelle in einem Schritt)
  * Wasser pro Tropfen invers zur Partikeldichte, damit mehr Partikel das
    Ergebnis verbessern statt verschlechtern

Im Feldverfahren entfällt das alles ersatzlos:

  * Ein abflussloses Becken ist schlicht ein See - das Pipe-Modell füllt ihn
    und lässt ihn überlaufen, ohne dass jemand "Senken" suchen müsste.
  * Eine Grube hat keine Fließgeschwindigkeit, also keine Transportkapazität,
    also entsteht dort keine Erosion. Krater sind strukturell ausgeschlossen.
  * Es gibt keine Partikel, die kollidieren könnten - jede Zelle wird pro
    Schritt genau einmal ausgewertet.

Der entscheidende Unterschied ist NICHT die Hydraulik: PipeFlowSimulator in
core/water_generator.py rechnet bereits formelgleich, inklusive der
Massenerhaltungs-Skalierung K. Der Unterschied ist, dass Sediment hier ein
EIGENES, MITSTRÖMENDES FELD ist statt einer Eigenschaft von Partikeln. Genau
das erzeugt die verästelten Entwässerungsnetze und die weichen Schwemmfächer.

DIE ZEHN PASSES PRO SCHRITT
---------------------------
Reihenfolge exakt wie im Vorbild (dort SimulatePerStep in src/main.ts):

  1. Regen              konstante Rate auf das Wasserfeld
  2. Fluss-Update       f = max(0, f + dt*g*A*dh/L), dann K-Skalierung
  3. Wasserhöhe + v     Divergenz -> Tiefe; Geschwindigkeit aus Flussdifferenz
  4. Erosion/Ablagerung C = Kc*slope*|v|*dt; C > s -> lösen, sonst ablagern
  5. Sediment-Advektion semi-lagrangesch, MacCormack mit Min/Max-Limiter
  6. Thermal            Böschungswinkel, zwei umschaltbare Verfahren
  7. Verdunstung        water *= (1 - Ke*dt)
  8. Glättung           bedingt, nur auf Ein-Pixel-Graten und -Rinnen

(Passes 6-8 des Vorbilds - maxslippage/thermalflux/thermalapply - sind hier zu
einem Schritt zusammengefasst, weil die Gather-Variante sie ohnehin nicht
trennt. Die Nummerierung oben ist deshalb 8-stufig statt 10-stufig.)

WAS BEWUSST VOM VORBILD ABWEICHT
--------------------------------
  * Härte-Kopplung (geology.hardness auf Löserate und Böschungswinkel) -
    Webgl-Erosion kennt kein Gestein. Der Einfluss ist über
    `hardness_influence` regelbar und steht per Default auf 0.0, also exakt
    auf Vorbild-Verhalten.
  * Abbruch über ein Konvergenzkriterium statt endloser Interaktivität.
  * Kein Pinsel, keine punktuellen Wasserquellen: der Regen fällt gleichmäßig
    auf die gesamte Karte (Nutzer-Vorgabe).

CPU-PFAD
--------
Dieser Kern ist die verifizierbare Referenz, nicht der Betriebsmodus. Ein Lauf
mit mehreren tausend Schritten ist oberhalb von etwa 256² auf CPU nicht mehr
sinnvoll - siehe MAX_CPU_RESOLUTION und die dortige Begründung. Der GPU-Pfad
(shaders/erosion/*.comp) rechnet dasselbe Modell Pass für Pass.
"""

import logging

import numpy as np
from scipy.ndimage import map_coordinates

# Naturkonstante aus dem Wasser-Modell mitbenutzt statt dupliziert - dieselbe
# Erdbeschleunigung treibt hier dieselbe Rohr-Gleichung (siehe
# PipeFlowSimulator._pipe_step_cpu, formelgleich).
from core.water_generator import GRAVITY, ThermalErosionSystem


class HydraulicFieldSimulator:
    """
    Funktionsweise: Treibt die acht Passes (siehe Modul-Docstring) so lange,
    bis die mittlere Höhenänderung pro Schritt unter die Konvergenzschwelle
    fällt oder die Schritt-Obergrenze erreicht ist.

    Aufgabe: Liefert die Differenzkarten erosion_map/sedimentation_map (Abtrag
    und Auftrag in Metern) plus die Diagnosefelder, aus denen der Erosion-Tab
    seine Ansichten speist. Die Differenzform ist bewusst gewählt: der
    restliche Programmablauf setzt die Heightmap über
    DataLODManager.get_calculator_combined_heightmap() als
    `Basis - Abtrag + Auftrag` zusammen, und dieser Vertrag bleibt damit
    unverändert.

    Zustand pro Zelle (alles float64, damit die Bilanz über tausende Schritte
    nicht wegdriftet):
        terrain   Geländehöhe (m)
        water     Wassertiefe (m)
        flux      4-Richtungs-Ausfluss (m³/s), Reihenfolge Links/Rechts/Oben/Unten
        sediment  gelöste Fracht (m Materialsäule, im Wasser gelöst gedacht)
        vx, vy    Fließgeschwindigkeit (m/s), jeden Schritt neu aus dem Fluss
                  abgeleitet - reine Diagnose- und Transportgröße

    Kartenrand: OFFEN für Wasser (Geisterzelle mit gleicher Geländehöhe, aber
    Wassertiefe 0 - Standard-"open boundary" für Flachwassermodelle),
    GESCHLOSSEN für Material (nichts rutscht von der Karte). Identisch zu
    PipeFlowSimulator bzw. ThermalErosionSystem, damit sich beide Modelle am
    Rand nicht widersprechen.
    """

    # ------------------------------------------------------------------
    # Physikalische Konstanten des Rohr-Modells
    # ------------------------------------------------------------------

    # Virtuelle Rohr-Querschnittsfläche (m²) - identischer Wert wie
    # PipeFlowSimulator.PIPE_CROSS_SECTION_AREA, damit beide Modelle dieselbe
    # Hydraulik zeigen. Bewusst dupliziert statt importiert: der Erosion-
    # Generator soll unabhängig vom Water-Generator lauffähig bleiben, wenn
    # dessen Altbestand irgendwann entfällt.
    PIPE_CROSS_SECTION_AREA = 0.6  # m²

    # Zeitschritt: NICHT konstant, sondern aus der Zellgröße abgeleitet.
    #
    # Die K-Skalierung macht die MASSENbilanz unconditionally stable, aber
    # nicht die Geschwindigkeit. Bei gesättigtem Fluss (auf steilem Gelände
    # der Normalfall, weil der Fluss als Impuls aufgebaut wird und nichts ihn
    # bremst ausser eben K) gilt
    #
    #     total_out = water * cell_area / dt   und damit
    #     |v| = 0.5 * total_out / (L * d) = 0.5 * L / dt
    #
    # also eine reine GITTER-Geschwindigkeit, unabhängig davon, wieviel Wasser
    # tatsächlich fliesst. Mit einem festen dt = 0.5 s und 104 m/px waren das
    # gemessen 107 m/s im Mittel und 207 m/s im Maximum - und weil die
    # Transportkapazität |v| als Faktor enthält, erodierte die Karte in 400
    # Schritten ein Drittel ihres Reliefs weg.
    #
    # dt wird deshalb so gewählt, dass diese Gitter-Geschwindigkeit genau
    # MAX_FLOW_VELOCITY_M_S trifft:
    #
    #     dt = 0.5 * meters_per_pixel / MAX_FLOW_VELOCITY_M_S
    #
    # Nebeneffekt, der genau richtig ist: dt skaliert damit automatisch mit
    # Map Distance. Eine 100-km-Karte bekommt grössere Zeitschritte als eine
    # 1-km-Karte, und dieselbe Anzahl Schritte bildet in beiden Fällen
    # dieselbe Menge Landschaftsentwicklung ab.
    MAX_FLOW_VELOCITY_M_S = 8.0

    # Mindest-Wassertiefe für die Geschwindigkeits-Ableitung - verhindert die
    # Division durch eine fast trockene Zelle. Gleicher Zweck und gleicher
    # Wert wie PipeFlowSimulator.MIN_DEPTH_FOR_VELOCITY.
    MIN_DEPTH_FOR_VELOCITY = 1e-4  # m

    # ------------------------------------------------------------------
    # Kalibrierungskonstanten (Startwerte - Stufe 5 des Plans)
    # ------------------------------------------------------------------

    # Umrechnung des Regen-Sliders (dimensionslos, 0..5) in Meter Wasser pro
    # Sekunde Simulationszeit. STARTWERT, noch nicht gegen die laufende App
    # kalibriert: mit dt=0.5 s bringt ein Slider-Wert von 1.0 über 1000
    # Schritte rund 0.5 m Wasser ein, wenn nichts abfliesst - genug, um auf
    # einer 4000-m-Relief-Karte durchgehende Rinnen zu speisen, ohne die Karte
    # zu fluten.
    RAIN_RATE_TO_DEPTH_M_PER_S = 1e-3

    # Obergrenze der Transportkapazität (m Materialsäule): was eine Zelle mit
    # voller Fliessgeschwindigkeit auf maximaler Steigung und ausreichend
    # Wasser höchstens gelöst mit sich führt. Zusammen mit `Kc` der Regler,
    # der bestimmt, wieviel Material insgesamt in Bewegung ist.
    #
    # Warum die Kapazität so und nicht wie im Vorbild formuliert ist: dort
    # steht C = Kc * slope * |v|, was dimensional eine Geschwindigkeit ist und
    # nur funktioniert, weil Gelände und Zellgröße dort in derselben
    # willkürlichen Einheit stehen. Übernimmt man das mit Metern und Sekunden
    # und multipliziert mit dt, wird die Kapazität proportional zur pro
    # Schritt zurückgelegten Strecke - also zur ZELLGRÖSSE. Gemessen bei
    # 104 m/px ergab das 26 m Kapazität pro Schritt und einen Reliefverlust
    # von einem Drittel in 400 Schritten.
    #
    # Hier ist die Kapazität stattdessen eine dimensionslose Konzentration
    # (Steigung x normierte Geschwindigkeit x Tiefenfaktor) mal dieser
    # Bezugssäule. Sie hängt damit nur noch vom Zustand des Wassers ab, nicht
    # von der Diskretisierung.
    CAPACITY_REFERENCE_M = 1.0

    # Bezugs-Durchfluss pro Meter Gewässerbreite (m²/s), bei dem eine Zelle
    # die HALBE Tragfähigkeit erreicht (der Faktor geht als q/(q+q_ref) ein).
    #
    # Warum der Durchfluss und nicht die Geschwindigkeit der kanalbildende
    # Faktor ist - das Vorbild nutzt |v|, und das funktioniert dort, hier
    # nicht: das Rohrmodell kennt keine Reibung, der Fluss wird als Impuls
    # aufgebaut und nur von der K-Skalierung begrenzt. Im eingeschwungenen
    # Zustand liegt deshalb JEDE wasserführende Zelle an dieser Grenze, und
    # dort gilt |v| = 0.5 * L / dt unabhängig von der Wassermenge. Gemessen
    # bei 104 m/px: mittlere Geschwindigkeit 11.9 m/s bei einer Streuung von
    # praktisch null - die Geschwindigkeit ist damit faktisch ein Ja/Nein und
    # taugt nicht als Regler.
    #
    # Der Durchfluss dagegen trägt genau das Signal, auf das es ankommt: eine
    # Talsohle sammelt das Wasser vieler Hangzellen, ein Grat fast keins. Das
    # ist zugleich die klassische Stream-Power-Form (Abtrag ~ Durchfluss x
    # Gefälle) und der Grund, warum sich überhaupt Kanäle statt einer
    # gleichmäßig abgetragenen Fläche bilden.
    #
    # Als SPEZIFISCHER Durchfluss (pro Meter Breite) formuliert, damit die
    # Grösse nicht von der Zellgröße abhängt.
    #
    # Der Faktor ist ein POTENZGESETZ min(1, (q/q_ref)^m), nicht die
    # sättigende Form q/(q+q_ref). Die sättigende Form war zuerst
    # implementiert und ist verworfen: sie presst die vorhandene Spanne
    # zusammen, statt sie zu nutzen. Gemessen im eingeschwungenen Zustand bei
    # 96 px, Median q nach Höhenlage:
    #
    #     oberes Drittel    q 0.34      mittleres 0.94      unteres 2.19
    #
    # also ein Verhältnis von 6.5 zwischen Tal und Gipfel. Die sättigende Form
    # machte daraus 0.69 gegen 0.94 - Faktor 1.36. Übrig blieb der Hangfaktor,
    # und der ist auf den Gipfeln am grössten (0.190 gegen 0.100): die Erosion
    # wanderte damit genau dorthin, wo sie laut Zielbild NICHT hin soll
    # (gemessener Erosions-Schwerpunkt 0.71 bei einem Geländemittel von 0.51).
    #
    # Das Potenzgesetz erhält die Spanne und ist zugleich die übliche
    # Stream-Power-Form (Abtrag ~ q^m * S^n).
    # Gemessen bei 96 px, 12 000 Schritten, gegen die fünf Qualitätsziele:
    #
    #   q_ref  m   Top5%   Netz  Krater  Ebenen%  Schwerpunkt  Bilanz
    #    1.0  2.0  0.227     51       1     42.9        0.700  -6.7e-3
    #    2.0  2.0  0.209     46      16     33.7        0.645  -7.8e-4
    #    2.0  3.0  0.265     27      30     27.8        0.601  +3.3e-3
    #
    # Default ist die erste Zeile: Krater praktisch null und Ebenenanteil im
    # Zielband. Top-5%-Anteil (Ziel > 0.30), Kanalnetz und Erosions-Schwerpunkt
    # (Ziel unter dem Geländemittel von 0.511) sind damit NOCH NICHT erreicht -
    # das ist die offene Arbeit aus Stufe 5 des Plans. Höhere Exponenten
    # verbessern Konzentration und Schwerpunkt, holen aber die Krater zurück;
    # dieser Zielkonflikt ist noch nicht aufgelöst.
    REFERENCE_SPECIFIC_DISCHARGE = 1.0  # m²/s
    DISCHARGE_EXPONENT = 2.0

    # Schwelle, unter der ueberhaupt kein Material bewegt wird (m²/s).
    #
    # Ohne sie nagt die Erosion FLAECHIG statt in Rinnen. Gemessen bei 128 px,
    # 2000 Schritten, Erosion nach Hoehenlage:
    #
    #     Band                q Median     Anteil an der Gesamterosion
    #     unteres Drittel        2.11                 0.0%
    #     mittleres              0.70                69.7%
    #     oberes                 0.17                30.3%
    #
    #     Zellen mit q < 0.5: 48% der Karte - und 64% der Gesamterosion.
    #
    # Die vielen schwach durchflossenen Hangzellen erodieren also mehr als die
    # wenigen Rinnen. Jede einzelne kaum, aber es sind eben sehr viele, und
    # ankommendes Sediment gibt es dort keins - also ist die Kapazitaet dort
    # immer untersaettigt. Genau das widerspricht dem Zielbild "Gipfel und
    # Kaemme groesstenteils unberuehrt, Erosion nimmt bergab zu".
    #
    # Die Schwelle ist keine Notloesung, sondern die uebliche Form: Abtrag
    # setzt erst oberhalb einer kritischen Schubspannung ein
    # (E ~ (tau - tau_c)), und geomorphologisch ist das die Grenze zwischen
    # HANG (von Massenbewegung geformt - hier die Boeschungserosion) und
    # GERINNE (von fliessendem Wasser geformt). Sie wird abgezogen statt hart
    # abgeschnitten, damit es keinen Sprung im Kartenbild gibt.
    #
    # NACHKALIBRIERT 2026-07-28 von 0.6 auf 0.4 (scratch_erosion_lab.py, Sweep
    # "schwelle", 512 px, uebrige Werte auf dem neuen Stand). Gemessen ueber
    # beta, den Exponenten der Hangneigung-ueber-Einzugsgebiet-Beziehung - je
    # negativer, desto klarer fluvial geformt:
    #
    #     Schwelle 0.6   beta -0.464   hypso 0.670
    #     Schwelle 0.4   beta -0.477   hypso 0.670   <- dichtestes Netz
    #     Schwelle 0.2   beta -0.419   hypso 0.702
    #     Schwelle 0.1   beta -0.370   hypso 0.549   Becken laufen wieder voll
    #
    # Zwei Vermutungen von mir haben sich dabei als falsch erwiesen, beide
    # widerlegt statt uebernommen:
    #   - "die Schwelle muss mit der Aufloesung skalieren": nein. Der
    #     Durchfluss-Median ist bei 128/256/512 px praktisch identisch
    #     (11.73 / 11.91 / 11.79 m^2/s).
    #   - "kleiner ist besser, dann entstehen mehr Seitenaeste": nein, unter
    #     0.2 nagen die Haenge wieder flaechig und die Becken fluten zu.
    EROSION_THRESHOLD_DISCHARGE = 0.4  # m²/s

    # Untergrenze für den Hangfaktor in der Kapazitätsformel. Direkt aus dem
    # Vorbild übernommen (dort max(0.1, |sin(theta)|)).
    #
    # Warum eine Untergrenze überhaupt sinnvoll ist: ohne sie hat eine ebene
    # Fläche exakt null Kapazität, das Wasser müsste dort also seine gesamte
    # Fracht sofort abladen. Genau am Übergang vom Hang zur Ebene entstünde
    # ein harter Wall, und die Ebene dahinter bliebe unberührt - das Gegenteil
    # des gewünschten Schwemmfächers. Mit der Untergrenze trägt langsames
    # Wasser noch etwas Fracht weiter in die Ebene hinein.
    MIN_SLOPE_FACTOR = 0.1

    # Bezugshärte und Exponent für die Erodierbarkeit. Nur wirksam, wenn
    # `hardness_influence` > 0 ist (Default 0.0 = Vorbild-Verhalten).
    HARDNESS_REFERENCE = 50.0
    HARDNESS_EXPONENT = 1.0
    HARDNESS_ERODIBILITY_MIN = 0.1
    HARDNESS_ERODIBILITY_MAX = 3.0

    # Böschungswinkel, wenn die Härte-Kopplung AUS ist (Grad). Mittelwert des
    # Bereichs, den ThermalErosionSystem härteabhängig aufspannt (15°-60°) -
    # so verhält sich der Erosion-Generator bei hardness_influence=0 wie ein
    # Gelände aus einheitlichem Material, nicht wie eines aus dem weichsten.
    NEUTRAL_REPOSE_ANGLE_DEG = 37.5

    # ------------------------------------------------------------------
    # Steuerung des Laufs
    # ------------------------------------------------------------------

    # Alle wie viele Schritte das Konvergenzkriterium geprüft wird. Die
    # Prüfung braucht eine Kopie der Höhenkarte, ist also nicht gratis; und
    # ein Kriterium, das jeden einzelnen Schritt misst, reagiert auf das
    # Rauschen einzelner Schritte statt auf den Trend.
    CONVERGENCE_CHECK_INTERVAL = 25

    # Alle wie viele Schritte ein Fortschritt gemeldet wird (Log + Ladebalken).
    #
    # Deutlich seltener als die Konvergenzpruefung: die laeuft alle 25 Schritte
    # und waere als Log-Zeile unlesbar. 500 ergibt bei einem typischen Lauf von
    # rund 12 000 Schritten etwa 24 Meldungen - genug, um zu sehen, dass es
    # vorangeht, wenig genug, um das Log nicht zu fluten.
    PROGRESS_REPORT_INTERVAL = 500

    # Oberhalb dieser Kantenlänge verweigert der CPU-Pfad den Dienst.
    #
    # Grund: ein Lauf besteht aus mehreren tausend Schritten mit je acht
    # vollständigen Feld-Operationen. Auf 512² sind das pro Schritt rund
    # 20 numpy-Durchläufe über 262 144 Zellen - hochgerechnet Stunden. Ein
    # stiller Hänger wäre genau die Art Falle, die in diesem Projekt zuletzt
    # systematisch entfernt wurde (siehe die Fallback-Aufräumarbeiten in
    # core/water_generator.py); deshalb ein klarer Fehler statt einer
    # Wartezeit ohne Ende. Der GPU-Pfad hat diese Grenze nicht.
    MAX_CPU_RESOLUTION = 256

    # Entfernung zu einem diagonalen Nachbarn in Vielfachen der Zellbreite.
    # Erklaerung und Messung siehe _pass_thermal().
    DIAGONAL_DISTANCE = float(np.sqrt(2.0))

    # Fixed-Point-Aufloesung fuer den einzigen atomaren Zaehler des GPU-Pfads
    # (ueber den Kartenrand exportierte Fracht, siehe
    # shaders/erosion/sedimentTransport.comp). GLSL 430 Core kennt
    # imageAtomicAdd nur fuer Integer-Bildformate; 1e6 gibt Mikrometer-
    # Aufloesung und laesst bei int32 rund 2 000 m Gesamtexport zu - derselbe
    # Wert und dieselbe Begruendung wie beim Droplet-Pfad.
    FIXED_POINT_SCALE = 1e6

    # Die GPU-Operation, die dieses Modell rechnen wuerde. Existiert noch
    # nicht - siehe has_gpu_path().
    GPU_OPERATION = ("erosion", "hydraulicField")

    def __init__(self, shader_manager=None):
        self.shader_manager = shader_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def has_gpu_path(self) -> bool:
        """
        Gibt es fuer dieses Modell einen NUTZBAREN GPU-Pfad?

        Bewusst nicht "ist ein ShaderManager da": in der laufenden App ist
        immer einer da, aber ob fuer eine bestimmte Operation auch ein
        Dispatch registriert ist, steht in shader_manager.DISPATCH_TABLE. Der
        Unterschied war teuer: die Aufloesungs-Begrenzung haing zuerst an
        `shader_manager is None` und griff deshalb im Programm nie - der Lauf
        endete stattdessen an der CPU-Grenze mit
        "verweigert 512x512", also einem harten Fehler mitten in der Pipeline.

        Sobald die Erosions-Shader registriert sind (Stufe 2), liefert diese
        Pruefung von selbst True - ohne dass hier etwas nachgezogen werden muss.
        """
        if self.shader_manager is None:
            return False
        try:
            from gui.OldManagers.shader_manager import DISPATCH_TABLE
        except ImportError:
            return False
        return self.GPU_OPERATION in DISPATCH_TABLE

    # ==================================================================
    # Öffentlicher Einstieg
    # ==================================================================

    def simulate(self, heightmap, hardness_map, parameters, meters_per_pixel,
                 progress_callback=None):
        """
        Funktionsweise: Führt den vollständigen Erosionslauf aus. Die
        übergebene heightmap wird NICHT mutiert.

        Parameter:
            heightmap        (H,W) Geländehöhe in m
            hardness_map     (H,W) Gesteinshärte 1..100 (aus geology.hardness)
            parameters       dict mit den Slider-Werten des Erosion-Tabs,
                             siehe _resolve_parameters() für Namen und Defaults
            meters_per_pixel Kantenlänge einer Zelle in m
            progress_callback optional callable(step, max_steps, terrain) -
                             wird alle PREVIEW_INTERVAL Schritte gerufen und
                             speist die Live-Vorschau im Tab. `terrain` ist
                             eine KOPIE, der Aufrufer darf sie behalten.

        Return: dict mit
            erosion_map, sedimentation_map      (H,W) float32, Meter
            thermal_erosion_map,
            thermal_deposition_map              (H,W) float32, Meter
            sediment_load_map                   (H,W) float32, verbleibende Fracht
            water_depth_map                     (H,W) float32, m
            flow_velocity_map                   (H,W) float32, m/s
            steps_taken                         int
            converged                           bool
            mass_balance                        float, siehe unten

        `mass_balance` ist die Summe aller Höhenänderungen plus der noch im
        Wasser gelösten Fracht, geteilt durch den Gesamtabtrag. Bei exakter
        Bilanz wäre der Wert 0; Abweichungen entstehen ausschließlich durch
        die Semi-Lagrange-Advektion (Pass 5), die Fracht über den Kartenrand
        hinaustragen kann. Der Wert ist damit ein direktes Mass für den
        einzigen bekannten Verlustpfad des Modells.
        """
        size = int(heightmap.shape[0])
        if not self.has_gpu_path() and size > self.MAX_CPU_RESOLUTION:
            raise ValueError(
                f"HydraulicFieldSimulator (CPU) verweigert {size}x{size}: oberhalb "
                f"{self.MAX_CPU_RESOLUTION}x{self.MAX_CPU_RESOLUTION} dauert ein Lauf "
                f"Stunden. Für diese Auflösung wird der GPU-Pfad gebraucht "
                f"(siehe MAX_CPU_RESOLUTION)."
            )

        cfg = self._resolve_parameters(parameters)
        state = self._initial_state(heightmap, hardness_map, cfg, meters_per_pixel)

        dt = 0.5 * float(meters_per_pixel) / self.MAX_FLOW_VELOCITY_M_S

        if self.has_gpu_path():
            gpu_result = self._simulate_gpu(state, cfg, dt, meters_per_pixel, progress_callback)
            if gpu_result is not None:
                return self._collect_results(
                    state, heightmap, gpu_result["steps_taken"], gpu_result["converged"])
        max_steps = cfg["max_steps"]
        interval = self.CONVERGENCE_CHECK_INTERVAL
        reference = state["terrain"].copy()
        converged = False
        steps_taken = 0

        target_rate = cfg["convergence_threshold"] * state["relief"]
        initial_rate = None
        last_rate = None
        progress = 0.0

        for step in range(1, max_steps + 1):
            self._step(state, cfg, dt)
            steps_taken = step

            if step % interval == 0:
                # Die Aenderungsrate wird RELATIV ZUM RELIEF gemessen, nicht in
                # absoluten Metern. Eine absolute Schwelle haengt sonst am
                # Hoehenmassstab der Karte: dieselbe Schwelle, die auf einer
                # 4000-m-Gebirgskarte nie erreicht wird, beendet einen Lauf auf
                # einer 200-m-Huegelkarte sofort. Derselbe Fehlertyp, der in
                # diesem Projekt bereits fuer lake_volume_threshold und den
                # Fluss-Schwellwert behoben wurde.
                rate = self.change_rate(state["terrain"], reference, interval, dt)
                last_rate = rate
                if initial_rate is None:
                    initial_rate = rate
                # MONOTON: der angezeigte Fortschritt darf nicht zurueckspringen.
                # Die Aenderungsrate schwankt von Pruefung zu Pruefung (gemessen
                # 47% -> 45% -> 51%), und ein Ladebalken, der rueckwaerts laeuft,
                # ist irrefuehrender als einer, der kurz stehenbleibt.
                progress = max(progress, self.convergence_progress(
                    initial_rate, rate, target_rate))

                if rate < target_rate:
                    converged = True
                    self._report(progress_callback, step, max_steps, state, 1.0, rate)  # fertig
                    break
                reference = state["terrain"].copy()

            if step % self.PROGRESS_REPORT_INTERVAL == 0:
                self._report(progress_callback, step, max_steps, state, progress, last_rate)

        return self._collect_results(state, heightmap, steps_taken, converged)

    def _report(self, progress_callback, step, max_steps, state, progress, rate):
        """
        Einen Zwischenstand melden - ins Log UND an den Aufrufer.

        Beides zusammen, weil beides denselben Zustand beschreibt und sonst
        auseinanderlaufen wuerde. Der Aufrufer (ErosionSystemGenerator) speist
        daraus den Ladebalken und die Live-Vorschau; das Log ist die Spur, die
        nach einem Abbruch noch da ist.
        """
        self.logger.info(
            "Erosion: Schritt %d (Obergrenze %d), Konvergenz %.0f%%, "
            "Aenderungsrate %.2e",
            step, max_steps, 100.0 * progress, rate if rate is not None else float("nan"))
        if progress_callback is not None:
            progress_callback(step, max_steps, state["terrain"].copy(), progress)

    def _simulate_gpu(self, state, cfg, dt, meters_per_pixel, progress_callback):
        """
        Den ganzen Lauf auf der GPU rechnen (shaders/erosion/*.comp, Dispatch
        in shader_manager._dispatch_hydraulic_field).

        Der Lauf wird in ABSCHNITTE von PROGRESS_REPORT_INTERVAL Schritten
        zerlegt; zwischen den Abschnitten prueft dieser Code die Konvergenz und
        meldet Fortschritt. Grund: GPUWorker.submit() bricht eine Operation nach
        30 s ab, und ein vollstaendiger Lauf reisst das in jedem Fall (gemessen
        bei 256 px: "GPU-Operation erosion/hydraulicField nach 30.0s nicht
        abgeschlossen", danach Rueckfall auf CPU). Ausserdem blockiert ein
        minutenlanger Aufruf die gemeinsame Worker-Queue fuer jeden anderen
        Generator.

        Der Zustand wandert dabei durch (`state` im Request): Hoehe, Wasser,
        Fracht, Fluss, Boeschungs-Akkumulator und Export-Karte. Ohne ihn wuerde
        jeder Abschnitt mit trockener Karte neu anfangen.

        Das Ergebnis wird in denselben `state` geschrieben, den auch der
        CPU-Pfad fuehrt - _collect_results() sieht danach keinen Unterschied
        mehr, und beide Pfade teilen sich Buchhaltung und Massenbilanz.

        Rueckgabe None bedeutet "GPU hat nicht geliefert" - der Aufrufer rechnet
        dann auf der CPU weiter. Das ist der einzige verbleibende Fallback in
        diesem Modul und faengt eine ERWARTETE Umgebungsbedingung ab (keine
        GPU, Treiberfehler), keinen Programmfehler.
        """
        constant_inputs = {
                    "heightmap": state["terrain"],
                    "erodibility": state["erodibility"],
                    "tan_repose": state["tan_repose"],
                    "dt": dt,
                    "meters_per_pixel": meters_per_pixel,
                    "pipe_area": self.PIPE_CROSS_SECTION_AREA,
                    "gravity": GRAVITY,
                    "rain_per_step": cfg["rain_rate"] * self.RAIN_RATE_TO_DEPTH_M_PER_S * dt,
                    "min_depth_for_velocity": self.MIN_DEPTH_FOR_VELOCITY,
                    "capacity_kc": cfg["capacity_kc"],
                    "dissolve_ks": cfg["dissolve_ks"],
                    "deposit_kd": cfg["deposit_kd"],
                    "reference_discharge": self.REFERENCE_SPECIFIC_DISCHARGE,
                    "threshold_discharge": self.EROSION_THRESHOLD_DISCHARGE,
                    "discharge_exponent": self.DISCHARGE_EXPONENT,
                    "capacity_reference": self.CAPACITY_REFERENCE_M,
                    "min_slope_factor": self.MIN_SLOPE_FACTOR,
                    "transfer_rate": ThermalErosionSystem.TRANSFER_RATE,
                    "thermal_strength": cfg["thermal_strength"],
                    "gather_cap": max(ThermalErosionSystem.CAP_MIN_M,
                                      ThermalErosionSystem.CAP_RELIEF_FRACTION * state["relief"]),
                    "diagonal_distance": self.DIAGONAL_DISTANCE,
                    "thermal_variant": 1 if cfg["thermal_variant"] == "flux" else 0,
                    "evaporation_factor": max(0.0, 1.0 - cfg["evaporation"] * dt),
                    "smoothing_threshold": cfg["smoothing"] * state["smoothing_scale"],
        }

        target_rate = cfg["convergence_threshold"] * state["relief"]
        chunk = self.PROGRESS_REPORT_INTERVAL
        reference = state["terrain"].copy()
        carried_state = None
        initial_rate = None
        progress = 0.0
        steps_taken = 0
        converged = False

        while steps_taken < cfg["max_steps"]:
            steps = min(chunk, cfg["max_steps"] - steps_taken)
            request = dict(constant_inputs)
            request["chunk_steps"] = steps
            request["state"] = carried_state
            try:
                result = self.shader_manager.request_shader_operation(
                    self.GPU_OPERATION[0], self.GPU_OPERATION[1], request, {})
            except Exception as error:
                self.logger.warning("GPU-Erosion fehlgeschlagen (%s) - CPU-Pfad", error)
                return None
            if not result.get("success"):
                self.logger.warning("GPU-Erosion nicht verfuegbar (%s) - CPU-Pfad",
                                    result.get("reason", "unbekannt"))
                return None

            carried_state = result["state"]
            steps_taken += steps
            self._apply_gpu_result(state, result)

            rate = self.change_rate(state["terrain"], reference, steps, dt)
            if initial_rate is None:
                initial_rate = rate
            progress = max(progress, self.convergence_progress(
                initial_rate, rate, target_rate))
            reference = state["terrain"].copy()

            if rate < target_rate:
                converged = True
                self._report(progress_callback, steps_taken, cfg["max_steps"],
                             state, 1.0, rate)
                break
            self._report(progress_callback, steps_taken, cfg["max_steps"],
                         state, progress, rate)

        return {"steps_taken": steps_taken, "converged": converged}

    @staticmethod
    def _apply_gpu_result(state, result):
        """Das Ergebnis eines GPU-Abschnitts in denselben `state` schreiben, den
        auch der CPU-Pfad fuehrt - danach sieht _collect_results() keinen
        Unterschied mehr, und beide Pfade teilen sich Buchhaltung und
        Massenbilanz."""
        state["terrain"] = result["height"].astype(np.float64)
        state["sediment"] = result["sediment"].astype(np.float64)
        state["water"] = result["water"].astype(np.float64)
        state["vx"] = result["velocity"][:, :, 0].astype(np.float64)
        state["vy"] = result["velocity"][:, :, 1].astype(np.float64)
        state["thermal_erosion"] = result["thermal_erosion"].astype(np.float64)
        state["thermal_deposition"] = result["thermal_deposition"].astype(np.float64)
        state["sediment_exported"] = float(result["sediment_exported"])

    # ==================================================================
    # Parameter und Anfangszustand
    # ==================================================================

    @staticmethod
    def convergence_progress(initial_rate, current_rate, target_rate):
        """
        Wie weit ist der Lauf auf dem Weg zur Konvergenz? 0.0 bis 1.0.

        Die Aenderungsrate faellt ueber den Lauf um GROESSENORDNUNGEN
        (gemessen 4.9e-5 -> 7.1e-6 ueber 8000 Schritte). Ein linearer
        Fortschritt daraus waere nutzlos: er staende nach wenigen hundert
        Schritten bei 90% und danach quasi still. Deshalb logarithmisch -
        "die Haelfte der noch fehlenden Groessenordnungen ist geschafft"
        entspricht 50%.

        Das ist die Zahl, die der Ladebalken zeigt und die im Log steht: sie
        beantwortet "wie weit ist das noch", nicht "wie viele Schritte sind
        gelaufen" (die Schrittzahl allein sagt nichts, weil die Gesamtzahl
        vorab unbekannt ist - es gibt ja ein Konvergenz- und kein Schrittziel).
        """
        import math

        if current_rate <= target_rate:
            return 1.0
        if initial_rate <= target_rate or current_rate >= initial_rate:
            return 0.0
        return max(0.0, min(1.0,
                            math.log(initial_rate / current_rate)
                            / math.log(initial_rate / target_rate)))

    @staticmethod
    def change_rate(current, reference, interval, dt):
        """
        Wie stark aendert sich das Gelaende pro SEKUNDE Simulationszeit?
        Mittlere Zellaenderung, auf die verstrichene Zeit umgerechnet.

        PRO SEKUNDE, nicht pro Schritt - das ist der Unterschied zwischen
        einem aufloesungsabhaengigen und einem aufloesungsunabhaengigen
        Kriterium. Der Zeitschritt haengt an der Zellgroesse
        (dt = 0.5 * meters_per_pixel / MAX_FLOW_VELOCITY_M_S), ein Schritt ist
        bei feiner Aufloesung also kuerzer und bewegt entsprechend weniger.
        Gemessen mit dem Kriterium pro SCHRITT und identischer Schwelle:
        128 px konvergierte nach 2650 Schritten, 256 px schon nach 700 - die
        groessere Karte lieferte damit die UNREIFERE Landschaft, obwohl sie
        laenger rechnete.

        Es ist derselbe Fehlertyp, der in diesem Projekt bereits fuer den Regen
        behoben wurde (Zuwachs pro Schritt -> Rate pro Sekunde, Plan 1
        Punkt 2.2), und fuer lake_volume_threshold und den Fluss-Schwellwert.

        Ein Versuch mit dem 99. PERZENTIL statt dem Mittelwert ist gemessen
        verworfen. Die Ueberlegung dahinter war plausibel - der Mittelwert wird
        von den vielen ruhigen Hangzellen nach unten gezogen, waehrend in den
        wenigen Rinnen noch heftig gegraben wird, und mit der Erosionsschwelle
        brach der Lauf deshalb schon nach 2650 statt 16 000 Schritten ab. Die
        Erwartung war, dass die laengeren Laeufe die offenen Krater schliessen.

        Sie tun es nicht. Gemessen bei 128 px, Schwelle 0.6:

            Kriterium      Schritte   Top-5%   Krater
            Mittelwert         2 650    0.371      119
            99. Perzentil     16 000    0.290      142
            99. Perzentil     20 000    0.282      170

        Das Ergebnis wird mit der Laufzeit nicht besser, sondern schlechter -
        die Krater sammeln sich an, statt sich zu fuellen. Der teurere Lauf
        kauft also nichts. Das eigentliche Kraterproblem liegt woanders und ist
        noch offen; bis dahin bleibt das billigere Kriterium.

        Diese Methode ist bewusst oeffentlich: der GPU-Pfad
        (shader_manager._dispatch_hydraulic_field) liest dieselbe Hoehenkarte
        zurueck und ruft sie auf, damit beide Pfade dasselbe Abbruchkriterium
        benutzen statt es zweimal zu implementieren.
        """
        return float(np.abs(current - reference).mean()) / (interval * dt)

    def _resolve_parameters(self, parameters):
        """
        Slider-Werte einlesen und auf ihre gültigen Bereiche klemmen.

        Bewusst KEIN stiller Default für fehlende Schlüssel im Sinne von "dann
        halt 0": jeder Wert hat hier seinen dokumentierten Default aus
        gui/config/value_default.py class EROSION. Fehlt ein Schlüssel, ist
        das ein Aufruf ohne vollständige Parameter (z.B. aus einem Test) - der
        Default ist dann die richtige Antwort, nicht ein Fehler.
        """
        def get(name, default, low, high):
            value = float(parameters.get(name, default))
            return float(np.clip(value, low, high))

        return {
            "capacity_kc": get("erosion_capacity", 1.0, 0.1, 5.0),
            "dissolve_ks": get("erosion_strength", 0.5, 0.0, 2.0),
            "deposit_kd": get("deposition_rate", 0.5, 0.0, 2.0),
            "rain_rate": get("rainfall", 1.0, 0.0, 5.0),
            "evaporation": get("evaporation_rate", 0.015, 0.0, 0.1),
            # Relativ zum Relief (siehe simulate()). Gemessen bei 64 px,
            # Relief 715 m, klingt die Aenderungsrate so ab:
            #
            #     nach   500 Schritten  4.9e-5
            #     nach  2000 Schritten  2.4e-5
            #     nach  5000 Schritten  1.2e-5
            #     nach  8000 Schritten  7.1e-6
            #
            # Ein Landschaftsmodell mit staendigem Regen und ohne Hebung
            # erreicht nie einen echten Stillstand - es traegt immer weiter ab,
            # nur langsamer. Die Schwelle ist deshalb kein "fertig", sondern
            # ein "es lohnt nicht mehr". 1e-5 trifft nach der Messung rund
            # 5000-6000 Schritte.
            "convergence_threshold": get("convergence_threshold", 1e-6, 1e-8, 1e-5),
            "max_steps": int(get("max_steps", 8000, 200, 20000)),
            "thermal_variant": str(parameters.get("thermal_variant", "gather")).lower(),
            "thermal_strength": get("thermal_strength", 0.3, 0.0, 2.0),
            "talus_scale": get("talus_angle_scale", 1.0, 0.5, 2.0),
            "hardness_influence": get("hardness_influence", 0.0, 0.0, 1.0),
            "smoothing": get("smoothing", 0.3, 0.0, 1.0),
            "preview_interval": max(1, int(parameters.get("preview_interval", 50))),
        }

    def _initial_state(self, heightmap, hardness_map, cfg, meters_per_pixel):
        """
        Anfangszustand: trockene Karte, kein Fluss, kein Sediment.

        Die aus der Härte abgeleiteten Felder (Erodierbarkeit, Böschungs-
        Tangens) werden EINMAL vorberechnet - sie ändern sich über den Lauf
        nicht, und ihre Berechnung pro Schritt zu wiederholen wäre bei
        mehreren tausend Schritten reine Verschwendung.
        """
        terrain = heightmap.astype(np.float64).copy()
        shape = terrain.shape
        hardness = np.clip(hardness_map.astype(np.float64), 1.0, 100.0)
        influence = cfg["hardness_influence"]

        # Erodierbarkeit: 1.0 bei ausgeschalteter Kopplung, sonst zwischen
        # 1.0 und dem härteabhängigen Wert interpoliert. Die Interpolation
        # statt einer Verzweigung erspart einen Sonderfall im Code UND macht
        # den Regler zu einem echten Dosierregler statt eines Schalters.
        hardness_factor = np.clip(
            (self.HARDNESS_REFERENCE / hardness) ** self.HARDNESS_EXPONENT,
            self.HARDNESS_ERODIBILITY_MIN, self.HARDNESS_ERODIBILITY_MAX)
        erodibility = 1.0 + influence * (hardness_factor - 1.0)

        # Böschungswinkel: derselbe Bereich, den ThermalErosionSystem
        # aufspannt (importiert statt dupliziert, damit beide Verfahren
        # garantiert dieselbe Härte->Winkel-Beziehung benutzen).
        span = ThermalErosionSystem.REPOSE_ANGLE_MAX_DEG - ThermalErosionSystem.REPOSE_ANGLE_MIN_DEG
        normalized = (hardness - ThermalErosionSystem.HARDNESS_REFERENCE_MIN) / (
            ThermalErosionSystem.HARDNESS_REFERENCE_MAX - ThermalErosionSystem.HARDNESS_REFERENCE_MIN)
        hardness_angle = ThermalErosionSystem.REPOSE_ANGLE_MIN_DEG + span * np.clip(normalized, 0.0, 1.0)
        repose_deg = np.clip(
            (self.NEUTRAL_REPOSE_ANGLE_DEG + influence * (hardness_angle - self.NEUTRAL_REPOSE_ANGLE_DEG))
            * cfg["talus_scale"], 1.0, 85.0)

        relief = float(terrain.max() - terrain.min())

        # Bezugsgroesse fuer die bedingte Glaettung: die mittlere
        # Nachbardifferenz des AUSGANGSGELAENDES, EINMAL berechnet.
        #
        # Zuerst wurde sie pro Schritt aus dem aktuellen Gelaende bestimmt.
        # Das ist eine globale Reduktion ueber die ganze Karte - auf der CPU
        # billig, auf der GPU ein eigener Reduktionspass pro Schritt, und
        # damit der einzige Grund, warum CPU und GPU nicht dasselbe rechnen
        # koennten. Der Wert aendert sich ueber den Lauf ohnehin nur langsam
        # und in die harmlose Richtung: waehrend das Gelaende flacher wird,
        # ueberschreiten immer weniger Zellen die feste Schwelle, die
        # Glaettung wird also zurueckhaltender statt aggressiver.
        h_pad = np.pad(terrain, 1, mode='edge')
        smoothing_scale = float(np.abs([
            terrain - h_pad[1:-1, 0:-2], terrain - h_pad[1:-1, 2:],
            terrain - h_pad[0:-2, 1:-1], terrain - h_pad[2:, 1:-1]]).mean())

        return {
            "terrain": terrain,
            "terrain_initial": terrain.copy(),
            "water": np.zeros(shape, dtype=np.float64),
            "flux": np.zeros(shape + (4,), dtype=np.float64),
            "sediment": np.zeros(shape, dtype=np.float64),
            "vx": np.zeros(shape, dtype=np.float64),
            "vy": np.zeros(shape, dtype=np.float64),
            "discharge": np.zeros(shape, dtype=np.float64),
            "outflow_fraction": None,
            "sediment_exported": 0.0,
            "thermal_erosion": np.zeros(shape, dtype=np.float64),
            "thermal_deposition": np.zeros(shape, dtype=np.float64),
            "erodibility": erodibility,
            "tan_repose": np.tan(np.radians(repose_deg)),
            "meters_per_pixel": float(meters_per_pixel),
            "cell_area": float(meters_per_pixel) ** 2,
            "relief": relief,
            "smoothing_scale": smoothing_scale,
        }

    # ==================================================================
    # Ein Zeitschritt = die acht Passes
    # ==================================================================

    def _step(self, state, cfg, dt):
        """Ein vollständiger Schritt. Mutiert `state` in place."""
        self._pass_rain(state, cfg, dt)
        self._pass_flux_and_depth(state, dt)
        self._pass_erode_deposit(state, cfg, dt)
        self._pass_advect_sediment(state, dt)
        self._pass_thermal(state, cfg)
        self._pass_evaporate(state, cfg, dt)
        self._pass_smooth(state, cfg)

    # --- Pass 1 -------------------------------------------------------

    def _pass_rain(self, state, cfg, dt):
        """
        Gleichmäßiger Regen auf die gesamte Karte.

        Das Vorbild speist sein Wasser über einen Pinsel und permanente
        Quellen ein (der Gleichregen ist dort im Code auf 0 gesetzt). Hier ist
        das umgekehrt: kein Pinsel, dafür ein Regen, der überall gleich fällt
        (Nutzer-Vorgabe). Wo daraus ein Wasserlauf wird, entscheidet allein
        das Gelände - genau wie in der Natur.
        """
        state["water"] += cfg["rain_rate"] * self.RAIN_RATE_TO_DEPTH_M_PER_S * dt

    # --- Pass 2 + 3 ---------------------------------------------------

    def _pass_flux_and_depth(self, state, dt):
        """
        Fluss-Update und Wasserhöhe in EINEM Aufruf, aber zwei logisch
        getrennten Teilschritten: das Fluss-Update braucht von den Nachbarn
        die AKTUELLE Tiefe, das Tiefe-Update deren GERADE berechneten Fluss.
        Vollständig materialisierte Zwischen-Arrays, keine In-Place-Mutation
        während des Sweeps.

        Formelgleich zu PipeFlowSimulator._pipe_step_cpu (dort mit Regen und
        Verdunstung im selben Aufruf; hier sind das eigene Passes, damit die
        Reihenfolge exakt der des Vorbilds entspricht).
        """
        terrain = state["terrain"]
        water = state["water"]
        flux = state["flux"]
        pipe_length = state["meters_per_pixel"]
        cell_area = state["cell_area"]

        # Offener Rand: Geisterzelle hat dieselbe Geländehöhe, aber Tiefe 0 -
        # Wasser am Rand entwässert damit mit genau seinem eigenen Pegel als
        # Gefälle. Kein Sonderfall im Code nötig, der Padding-Trick erledigt es.
        b_pad = np.pad(terrain, 1, mode='edge')
        d_pad = np.pad(water, 1, mode='constant', constant_values=0.0)
        surface_pad = b_pad + d_pad
        surface = surface_pad[1:-1, 1:-1]

        accel = dt * self.PIPE_CROSS_SECTION_AREA * GRAVITY / pipe_length
        f_l = np.maximum(0.0, flux[:, :, 0] + accel * (surface - surface_pad[1:-1, 0:-2]))
        f_r = np.maximum(0.0, flux[:, :, 1] + accel * (surface - surface_pad[1:-1, 2:]))
        f_t = np.maximum(0.0, flux[:, :, 2] + accel * (surface - surface_pad[0:-2, 1:-1]))
        f_b = np.maximum(0.0, flux[:, :, 3] + accel * (surface - surface_pad[2:, 1:-1]))

        # Massenerhaltungs-Skalierung K, jeden Schritt neu: eine Zelle kann
        # nie mehr Wasser abgeben als sie hat. Das ist es, was das System
        # unconditionally stable macht - deshalb braucht dt keine
        # CFL-Bedingung (siehe TIME_STEP_S).
        total_out = f_l + f_r + f_t + f_b
        k = np.minimum(1.0, (water * cell_area) / np.maximum(dt * total_out, 1e-12))
        f_l, f_r, f_t, f_b = f_l * k, f_r * k, f_t * k, f_b * k

        fr_pad = np.pad(f_r, 1, mode='constant', constant_values=0.0)
        fl_pad = np.pad(f_l, 1, mode='constant', constant_values=0.0)
        fb_pad = np.pad(f_b, 1, mode='constant', constant_values=0.0)
        ft_pad = np.pad(f_t, 1, mode='constant', constant_values=0.0)

        in_l = fr_pad[1:-1, 0:-2]
        in_r = fl_pad[1:-1, 2:]
        in_t = fb_pad[0:-2, 1:-1]
        in_b = ft_pad[2:, 1:-1]

        incoming = in_l + in_r + in_t + in_b
        outgoing = f_l + f_r + f_t + f_b
        new_water = np.maximum(0.0, water + dt / cell_area * (incoming - outgoing))

        # Geschwindigkeit aus der Flussdifferenz über die Zellgrenzen, bezogen
        # auf die MITTLERE Tiefe des Schritts - das ist die Größe, die Pass 4
        # als Transportkapazität und Pass 5 als Advektionsfeld braucht.
        d_avg = np.maximum(0.5 * (water + new_water), self.MIN_DEPTH_FOR_VELOCITY)
        net_x = 0.5 * (in_l - f_l + f_r - in_r)
        net_y = 0.5 * (in_t - f_t + f_b - in_b)
        state["vx"] = net_x / (pipe_length * d_avg)
        state["vy"] = net_y / (pipe_length * d_avg)

        # Spezifischer Durchfluss (m²/s, also pro Meter Gewässerbreite) - die
        # Grösse, die in Pass 4 die Transportkapazität treibt. Bewusst NICHT
        # aus der Geschwindigkeit abgeleitet (siehe
        # REFERENCE_SPECIFIC_DISCHARGE), sondern direkt aus dem Netto-Fluss.
        state["discharge"] = np.hypot(net_x, net_y) / pipe_length

        # Welcher ANTEIL des Wassers einer Zelle in diesem Schritt in welche
        # Richtung abfliesst. Genau dieser Anteil nimmt die gelöste Fracht mit
        # (siehe _pass_advect_sediment) - deshalb wird er hier, wo die Flüsse
        # entstehen, einmal berechnet statt später rekonstruiert.
        #
        # Die K-Skalierung oben garantiert bereits, dass die Summe der vier
        # Anteile nie über 1 liegt: eine Zelle kann nicht mehr Wasser abgeben
        # als sie hat. Der Sedimenttransport erbt diese Zusage damit gratis.
        water_volume = np.maximum(water * cell_area, 1e-12)
        state["outflow_fraction"] = np.stack(
            [dt * f_l / water_volume, dt * f_r / water_volume,
             dt * f_t / water_volume, dt * f_b / water_volume], axis=-1)

        # Vorbild-Verhalten: in fast trockenen Zellen wird die Geschwindigkeit
        # auf 0 gesetzt statt aus einem verschwindenden Nenner hochgerechnet.
        # Ohne das schleudert die Advektion in Pass 5 Sediment aus Zellen, in
        # denen praktisch kein Wasser steht.
        dry = new_water < self.MIN_DEPTH_FOR_VELOCITY
        state["vx"][dry] = 0.0
        state["vy"][dry] = 0.0

        state["water"] = new_water
        state["flux"] = np.stack([f_l, f_r, f_t, f_b], axis=-1)

    # --- Pass 4 -------------------------------------------------------

    def _pass_erode_deposit(self, state, cfg, dt):
        """
        Der eigentliche Erosionskern.

            C = Kc * Hangfaktor^n * min(1,(q/q_ref)^m) * Bezugssäule
            C > s  ->  loesen   (C - s) * Ks * Erodierbarkeit
            C <= s ->  ablagern (s - C) * Kd

        Die drei Faktoren vor der Bezugssäule sind alle dimensionslos und auf
        [0,1] beschränkt - die Kapazität ist damit eine Konzentration mal
        einer festen Materialsäule und hängt NICHT von der Zellgröße oder vom
        Zeitschritt ab (Begründung bei CAPACITY_REFERENCE_M).

        Warum genau diese Formel Krater ausschliesst - und warum dafür das
        WASSERSPIEGEL-Gefälle genommen wird und nicht das Geländegefälle:

        Ein erster Versuch mit dem Geländegefälle erzeugte gemessen 35-42
        echte Senken. Der Grund ist, dass in einer sich füllenden Grube der
        Zufluss (und damit der Durchfluss) hoch bleibt, während die
        Grubenwände weiterhin steil sind - beide Faktoren der Kapazität waren
        also gross, und die Grube grub sich weiter ein.

        Der Wasserspiegel `Gelände + Wassertiefe` behebt das von selbst: in
        einem gefüllten Becken ist er EBEN, das Gefälle geht gegen null und
        damit auch die Kapazität - dort wird abgelagert, nicht abgetragen. Auf
        einem Hang mit dünnem Wasserfilm folgt der Spiegel dagegen dem Gelände
        und das Gefälle ist praktisch das Geländegefälle.

        Das ist zugleich die physikalisch richtige Grösse: die Erosionsarbeit
        des Wassers folgt seinem ENERGIEgefälle, und das ist der
        Wasserspiegel, nicht die Sohle. Das Partikelverfahren brauchte für
        dieselbe Zusage eine nachgeschaltete Senkenfüllung.
        """
        terrain = state["terrain"]

        # Hangfaktor als Sinus des Neigungswinkels des WASSERSPIEGELS.
        # np.gradient liefert die Höhendifferenz PRO ZELLE; die Division durch
        # die Zellbreite macht daraus die dimensionslose Steigung - und genau
        # dadurch reagiert die Erosion auf Map Distance, statt sie zu
        # ignorieren.
        gy, gx = np.gradient(terrain + state["water"])
        grad = np.hypot(gx, gy) / state["meters_per_pixel"]
        slope_true = grad / np.sqrt(1.0 + grad * grad)

        # ZWEI Hangfaktoren, und das ist der Punkt, an dem die Untergrenze
        # hingehört:
        #
        # Für den ABTRAG gilt der wahre Hangfaktor OHNE Untergrenze. Nur so
        # hält die Zusage, dass ein gefülltes Becken sich nicht weiter
        # eingräbt: dort ist der Wasserspiegel eben, slope_true geht gegen 0,
        # und ohne Kapazität gibt es keine Erosion.
        #
        # Für das TRAGEN gilt die Untergrenze. Sonst müsste Wasser, das eine
        # Ebene erreicht, seine gesamte Fracht schlagartig am Hangfuß abladen -
        # es entstünde ein Wall am Übergang und dahinter eine unberührte
        # Fläche, statt eines Schwemmfächers.
        #
        # Ein erster Entwurf benutzte für beides den Faktor MIT Untergrenze.
        # Das hob die Wasserspiegel-Logik wieder auf - gemessen blieben 10-19
        # echte Senken auch nach 16 000 Schritten stehen, weil ein ebener
        # Seeboden über die Untergrenze weiterhin 10% der vollen Kapazität
        # bekam.
        slope_erode = slope_true
        slope_carry = np.maximum(slope_true, self.MIN_SLOPE_FACTOR)

        discharge = np.maximum(0.0, state["discharge"] - self.EROSION_THRESHOLD_DISCHARGE)
        discharge_factor = np.minimum(1.0, (discharge / self.REFERENCE_SPECIFIC_DISCHARGE)
                                      ** self.DISCHARGE_EXPONENT)
        common = cfg["capacity_kc"] * discharge_factor * self.CAPACITY_REFERENCE_M
        capacity_erode = slope_erode * common
        capacity_carry = slope_carry * common

        sediment = state["sediment"]

        # Die beiden Zweige schliessen sich gegenseitig aus, weil
        # capacity_erode <= capacity_carry gilt: liegt die Fracht unter der
        # Abtragsschwelle, wird gelöst; liegt sie über der Tragfähigkeit, wird
        # abgelagert; dazwischen passiert nichts. Das "dazwischen" ist kein
        # Sonderfall, sondern der eingeschwungene Zustand eines Wasserlaufs,
        # der genau so viel trägt, wie er tragen kann.
        dissolved = np.maximum(0.0, capacity_erode - sediment) * cfg["dissolve_ks"] * state["erodibility"]
        deposited = np.maximum(0.0, sediment - capacity_carry) * cfg["deposit_kd"]

        # Nie mehr ablagern als vorhanden ist.
        deposited = np.minimum(deposited, sediment)

        state["terrain"] = terrain - dissolved + deposited
        state["sediment"] = sediment + dissolved - deposited

    # --- Pass 5 -------------------------------------------------------

    def _pass_advect_sediment(self, state, dt):
        """
        Gelöste Fracht mit dem Wasser mitbewegen - der Pass, der das
        Feldverfahren vom Partikelverfahren unterscheidet und die verästelten
        Frachtspuren erzeugt.

        BEWUSSTE ABWEICHUNG VOM VORBILD - hier wird MIT DEN WASSERFLÜSSEN
        transportiert statt semi-lagrangesch.

        Das Vorbild advektiert das Sedimentfeld semi-lagrangesch mit
        MacCormack-Korrektur (sediadvect-frag.glsl + maccormack-frag.glsl):
        jede Zelle verfolgt das Geschwindigkeitsfeld rückwärts, tastet dort
        bilinear ab, wiederholt das vorwärts und korrigiert mit der halben
        Differenz. Das ist in einem Fragment-Shader die naheliegende Wahl,
        weil man dort nur GATHERN kann - eine Zelle kann lesen, aber nicht in
        Nachbarn schreiben.

        Diese Wahl wurde hier zuerst 1:1 übernommen und dann VERWORFEN. Grund:
        semi-lagrangesche Advektion ist grundsätzlich nicht erhaltend, und der
        Min/Max-Limiter kappt zusätzlich systematisch die Spitzen. Gemessen
        über 200 Schritte bei 96 px, Materialänderung je Pass:

            Regen, Fluss, Erosion, Thermal, Verdunstung      exakt 0
            Glättung                                          +113 m
            SEMI-LAGRANGE-ADVEKTION                        -27 695 m

        bei einem Gesamtabtrag von 39 893 m - also 69% des abgetragenen
        Materials verschwanden spurlos. Genau dieses Material soll aber die
        Ebenen und Schwemmfächer aufbauen; ohne es erodiert die Karte nur.

        Der Ersatz ist zugleich einfacher und physikalisch direkter: gelöstes
        Sediment geht dorthin, wohin das Wasser geht. Die Anteile, mit denen
        eine Zelle ihr Wasser in die vier Richtungen abgibt, liegen aus dem
        Fluss-Pass bereits exakt vor (state["outflow_fraction"]) - dieselben
        Anteile nehmen die Fracht mit. Das ist per Konstruktion erhaltend, und
        zwar aus demselben Grund, aus dem es der Wassertransport ist: was eine
        Zelle abgibt, bekommt genau ein Nachbar.

        Der einzige verbleibende Verlustpfad ist der Kartenrand - und der ist
        physikalisch gewollt: Flüsse tragen ihre Fracht aus dem Gebiet hinaus.
        Er wird in `sediment_exported` mitgeschrieben, damit die Bilanz
        vollständig aufgeht statt nur ungefähr.
        """
        fractions = state.get("outflow_fraction")
        if fractions is None:
            return

        sediment = state["sediment"]
        out_l = sediment * fractions[:, :, 0]
        out_r = sediment * fractions[:, :, 1]
        out_t = sediment * fractions[:, :, 2]
        out_b = sediment * fractions[:, :, 3]

        # Zufluss von den Nachbarn = deren jeweils ENTGEGENGESETZTER Ausfluss.
        # Am Rand liefert die Geisterzelle 0 - was dort hinausfliesst, verlässt
        # die Karte (siehe Docstring).
        l_pad = np.pad(out_l, 1, mode='constant', constant_values=0.0)
        r_pad = np.pad(out_r, 1, mode='constant', constant_values=0.0)
        t_pad = np.pad(out_t, 1, mode='constant', constant_values=0.0)
        b_pad = np.pad(out_b, 1, mode='constant', constant_values=0.0)

        incoming = (r_pad[1:-1, 0:-2] + l_pad[1:-1, 2:] +
                    b_pad[0:-2, 1:-1] + t_pad[2:, 1:-1])
        outgoing = out_l + out_r + out_t + out_b

        state["sediment"] = np.maximum(0.0, sediment - outgoing + incoming)
        state["sediment_exported"] += float(
            out_l[:, 0].sum() + out_r[:, -1].sum() +
            out_t[0, :].sum() + out_b[-1, :].sum())

    # --- Pass 6 -------------------------------------------------------

    def _pass_thermal(self, state, cfg):
        """
        Böschungswinkel-Erosion, zwei umschaltbare Verfahren:

        "gather" (Default) - das in diesem Projekt bereits vorhandene
            Verfahren aus ThermalErosionSystem: jede Zelle SAMMELT, was ihre
            Nachbarn abgeben. Weil jede transportierte Einheit dabei genau
            einer Quelle und einem Ziel zugeordnet ist, ist die Massenbilanz
            strukturell exakt - ohne Renormierung.

        "flux" - die Variante des Vorbilds (maxslippage + thermalflux +
            thermalapply): jede Zelle berechnet ihren Ausfluss und skaliert
            ihn nachträglich auf das Verfügbare herunter. Ergebnis ist ähnlich,
            die Bilanz aber nur näherungsweise.

        Beide teilen sich denselben härteabhängigen Böschungswinkel aus
        `tan_repose` (siehe _initial_state) - der Vergleich zeigt damit
        wirklich den Unterschied der VERFAHREN, nicht den unterschiedlicher
        Winkel.
        """
        if cfg["thermal_strength"] <= 0.0:
            return

        terrain = state["terrain"]
        mpp = state["meters_per_pixel"]

        h_pad = np.pad(terrain, 1, mode='edge')
        tan_pad = np.pad(state["tan_repose"], 1, mode='edge')
        tan_here = tan_pad[1:-1, 1:-1]

        # ACHT Nachbarn, jeder mit seiner TATSAECHLICHEN Entfernung.
        #
        # Mit nur vier Nachbarn kann Material ausschliesslich entlang der
        # Gitterachsen wandern. Ein Haufen relaxiert dann nicht zu einem Kegel,
        # sondern zu einer RAUTE - in der Diagonalen ist der wirksame
        # Boeschungswinkel um Faktor sqrt(2) steiler, weil dort dieselbe
        # Hoehendifferenz auf eine laengere Strecke faellt.
        #
        # Im alten Water-Pfad (ThermalErosionSystem) fiel das nie auf, weil er
        # nur rund 40 Iterationen je LOD lief. Hier laeuft derselbe Pass
        # mehrere tausend Mal, und das Ergebnis war ein Gelaende aus lauter
        # 45-Grad-Pyramiden - gerendert sofort sichtbar, in keiner Kennzahl.
        # Gegenprobe mit thermal_strength=0: Rauten vollstaendig weg.
        #
        # Die Diagonalen bekommen deshalb `mpp * sqrt(2)` als Bezugslaenge -
        # damit ist der Boeschungswinkel in alle acht Richtungen derselbe
        # WINKEL, und ein Haufen wird wieder rund.
        neighbours = (
            (h_pad[1:-1, 0:-2], tan_pad[1:-1, 0:-2], 1.0),
            (h_pad[1:-1, 2:], tan_pad[1:-1, 2:], 1.0),
            (h_pad[0:-2, 1:-1], tan_pad[0:-2, 1:-1], 1.0),
            (h_pad[2:, 1:-1], tan_pad[2:, 1:-1], 1.0),
            (h_pad[0:-2, 0:-2], tan_pad[0:-2, 0:-2], self.DIAGONAL_DISTANCE),
            (h_pad[0:-2, 2:], tan_pad[0:-2, 2:], self.DIAGONAL_DISTANCE),
            (h_pad[2:, 0:-2], tan_pad[2:, 0:-2], self.DIAGONAL_DISTANCE),
            (h_pad[2:, 2:], tan_pad[2:, 2:], self.DIAGONAL_DISTANCE),
        )

        # Geschlossener Rand für Material: Geisterzelle = eigene Höhe, also
        # kein Gefälle über den Kartenrand. Anders als beim Wasser soll hier
        # nichts von der Karte rutschen.
        diffs = []
        for h_neighbor, tan_neighbor, distance in neighbours:
            threshold = mpp * distance * 0.5 * (tan_here + tan_neighbor)
            # Auch der Transport wird mit der Entfernung gewichtet: ueber die
            # laengere Diagonale rutscht in derselben Zeit weniger.
            diffs.append(np.maximum(0.0, (terrain - h_neighbor) - threshold) / distance)

        raw = [d * ThermalErosionSystem.TRANSFER_RATE * cfg["thermal_strength"] for d in diffs]
        total_raw = sum(raw)

        if cfg["thermal_variant"] == "flux":
            # Vorbild: Deckel ist die verfügbare Höhe über dem tiefsten
            # Nachbarn - mehr kann eine Zelle in einem Schritt nicht abgeben,
            # ohne unter ihre eigene Umgebung zu fallen.
            cap = np.maximum(0.0, terrain - np.minimum.reduce(
                [n[0] for n in neighbours]))
        else:
            # Gather-Verfahren: relief-relativer Deckel pro Schritt, exakt der
            # Wert, der in ThermalErosionSystem gegen "Thermal zerfrisst die
            # Hügel" kalibriert wurde.
            cap = max(ThermalErosionSystem.CAP_MIN_M,
                      ThermalErosionSystem.CAP_RELIEF_FRACTION * state["relief"])

        k = np.minimum(1.0, cap / np.maximum(total_raw, 1e-12))
        out = [r * k for r in raw]

        # Zufluss = der jeweils ENTGEGENGESETZTE Ausfluss der Nachbarn.
        # Reihenfolge oben: L, R, O, U, OL, OR, UL, UR - die Gegenrichtung ist
        # also jeweils der Partner im Paar.
        out_pads = [np.pad(o, 1, mode='constant', constant_values=0.0) for o in out]
        incoming = (out_pads[1][1:-1, 0:-2] + out_pads[0][1:-1, 2:] +
                    out_pads[3][0:-2, 1:-1] + out_pads[2][2:, 1:-1] +
                    out_pads[7][0:-2, 0:-2] + out_pads[6][0:-2, 2:] +
                    out_pads[5][2:, 0:-2] + out_pads[4][2:, 2:])
        outgoing = sum(out)

        state["terrain"] = terrain + incoming - outgoing
        state["thermal_erosion"] += outgoing
        state["thermal_deposition"] += incoming

    # --- Pass 7 -------------------------------------------------------

    def _pass_evaporate(self, state, cfg, dt):
        """
        Verdunstung als zweite Senke neben dem Randabfluss. Ohne sie hätte
        eine Karte mit ausgeprägten Becken keinen Weg, Wasser wieder zu
        verlieren - der Pegel dort könnte nur steigen, und die Erosion käme in
        genau den Bereichen zum Erliegen, in denen die Ebenen entstehen sollen.
        """
        state["water"] *= max(0.0, 1.0 - cfg["evaporation"] * dt)

    # --- Pass 8 -------------------------------------------------------

    def _pass_smooth(self, state, cfg):
        """
        Bedingte Glättung - der Pass, der im Vorbild spürbar zur Optik
        beiträgt (average-frag.glsl).

        Er glättet NICHT flächig, sondern ausschliesslich dort, wo eine Zelle
        gegenüber ZWEI GEGENÜBERLIEGENDEN Nachbarn in dieselbe Richtung
        abweicht - also auf Ein-Pixel-Graten und Ein-Pixel-Rinnen. Ein echter
        Hang hat auf der einen Seite einen höheren und auf der anderen einen
        tieferen Nachbarn, die Vorzeichen sind dort entgegengesetzt, und er
        bleibt unangetastet.

        Genau diese Bedingung ist der Unterschied zwischen "Gitterartefakte
        entfernen" und "die Landschaft weichzeichnen".

        Die Schwelle ist relativ zur mittleren Nachbardifferenz des
        Ausgangsgeländes formuliert, nicht absolut - dadurch wirkt der Regler
        bei jedem Relief und jeder Kartenausdehnung gleich (siehe
        `smoothing_scale` in _initial_state(), dort auch die Begründung, warum
        sie EINMAL und nicht pro Schritt bestimmt wird). `smoothing = 0`
        schaltet den Pass vollständig ab.
        """
        if cfg["smoothing"] <= 0.0 or state["smoothing_scale"] <= 0.0:
            return

        terrain = state["terrain"]
        h_pad = np.pad(terrain, 1, mode='edge')
        d_l = terrain - h_pad[1:-1, 0:-2]
        d_r = terrain - h_pad[1:-1, 2:]
        d_t = terrain - h_pad[0:-2, 1:-1]
        d_b = terrain - h_pad[2:, 1:-1]

        threshold = cfg["smoothing"] * state["smoothing_scale"]

        spike = (((np.abs(d_l) > threshold) & (np.abs(d_r) > threshold) & (d_l * d_r > 0.0)) |
                 ((np.abs(d_t) > threshold) & (np.abs(d_b) > threshold) & (d_t * d_b > 0.0)))
        if not spike.any():
            return

        # Mittengewicht 8 gegen 4x1 + 4x0.707 wie im Vorbild - die betroffene
        # Zelle wird zu ihren Nachbarn hin gezogen, nicht durch sie ersetzt.
        neighbors = (h_pad[1:-1, 0:-2] + h_pad[1:-1, 2:] + h_pad[0:-2, 1:-1] + h_pad[2:, 1:-1])
        diagonals = (h_pad[0:-2, 0:-2] + h_pad[0:-2, 2:] +
                     h_pad[2:, 0:-2] + h_pad[2:, 2:])
        averaged = (terrain * 8.0 + neighbors + diagonals * 0.707) / (8.0 + 4.0 * 1.707)

        # UMVERTEILEN statt ersetzen. Das Vorbild schreibt den geglätteten Wert
        # einfach zurück - das erzeugt bzw. vernichtet Material, weil die
        # Nachbarn unverändert bleiben. Gemessen war das der einzige nicht
        # erhaltende Pass des ganzen Modells (+113 m über 200 Schritte bei
        # 96 px), und bei geringem Gesamtabtrag verdarb er die Bilanz komplett.
        #
        # Hier wird stattdessen der Höhenunterschied, den die Glättung
        # herausnimmt, gleichmäßig auf die acht Nachbarn verteilt. Optisch
        # dasselbe, bilanziell exakt - und damit bleibt der Kartenrand der
        # einzige Ort, an dem Material das System verlassen kann.
        delta = np.where(spike, averaged - terrain, 0.0)
        share = np.pad(-delta / 8.0, 1, mode='constant', constant_values=0.0)
        redistributed = (share[1:-1, 0:-2] + share[1:-1, 2:] +
                         share[0:-2, 1:-1] + share[2:, 1:-1] +
                         share[0:-2, 0:-2] + share[0:-2, 2:] +
                         share[2:, 0:-2] + share[2:, 2:])

        state["terrain"] = terrain + delta + redistributed

    # ==================================================================
    # Ergebnis
    # ==================================================================

    def _collect_results(self, state, heightmap, steps_taken, converged):
        """
        Netto-Höhenänderung in die beiden Differenzkarten aufteilen.

        Anders als beim Partikelverfahren wird hier NICHT brutto gebucht: das
        Feldmodell hat keine "Durchsatz"-Grösse, die sich sinnvoll
        akkumulieren liesse - eine Zelle löst und lagert im selben Schritt
        entweder das eine oder das andere, und was zählt, ist der Endzustand.
        Die Aufteilung nach Vorzeichen ist damit exakt und der Vertrag
        `Basis - Abtrag + Auftrag == Ergebnis` bitgenau erfüllt.

        (Beim Partikelverfahren war das anders und die Brutto-Buchung dort
        bewusst beibehalten - siehe den Abschnitt "BEIDE KARTEN SIND
        KUMULIERTER DURCHSATZ" in core/water_generator.py.)
        """
        net = state["terrain"] - heightmap.astype(np.float64)
        erosion_map = np.maximum(-net, 0.0)
        sedimentation_map = np.maximum(net, 0.0)

        # Vollstaendige Bilanz: was abgetragen wurde, liegt entweder anderswo
        # als Gelaende, ist noch geloest unterwegs, oder hat die Karte ueber
        # den Rand verlassen. Die drei Posten muessen sich zu null addieren.
        total_eroded = float(erosion_map.sum())
        residual = float(state["sediment"].sum())
        exported = float(state["sediment_exported"])
        balance = (float(net.sum()) + residual + exported) / max(total_eroded, 1e-9)

        return {
            "erosion_map": erosion_map.astype(np.float32),
            "sedimentation_map": sedimentation_map.astype(np.float32),
            "thermal_erosion_map": state["thermal_erosion"].astype(np.float32),
            "thermal_deposition_map": state["thermal_deposition"].astype(np.float32),
            "sediment_load_map": state["sediment"].astype(np.float32),
            "water_depth_map": state["water"].astype(np.float32),
            "flow_velocity_map": np.hypot(state["vx"], state["vy"]).astype(np.float32),
            "steps_taken": int(steps_taken),
            "converged": bool(converged),
            "mass_balance": balance,
            "sediment_exported": exported,
        }


class ErosionData:
    """
    Funktionsweise: Container fuer alle Erosion-Outputs mit LOD-Level.
    Aufgabe: Das Domain-Objekt, das DataLODManager.set_erosion_data_complete_lod()
        ablegt - dasselbe Muster wie WaterData/WeatherData.

    Die Feldliste steht in ErosionSystemGenerator.EROSION_DATA_KEYS; dieselbe
    Liste steuert die Ablage im DataLODManager.
    """

    def __init__(self, lod_level: int = 0):
        self.lod_level = lod_level
        # Gelaendeformend - diese vier gehen in
        # DataLODManager.get_calculator_combined_heightmap() ein.
        self.erosion_map = None             # (H,W) Netto-Abtrag in m
        self.sedimentation_map = None       # (H,W) Netto-Auftrag in m
        self.thermal_erosion_map = None     # (H,W) Boeschungs-Abtrag in m
        self.thermal_deposition_map = None  # (H,W) Boeschungs-Auftrag in m
        # Reine Anzeige- und Diagnosefelder.
        self.sediment_load_map = None       # (H,W) noch geloeste Fracht in m
        self.water_depth_map = None         # (H,W) eingeschwungener Wasserstand in m
        self.flow_velocity_map = None       # (H,W) Fliessgeschwindigkeit in m/s
        # Kennzahlen des Laufs (Skalare, fuer das Statistik-Widget).
        self.steps_taken = 0
        self.converged = False
        self.mass_balance = 0.0
        self.simulation_resolution = 0


class ErosionSystemGenerator:
    """
    Funktionsweise: Generator-Fassade um HydraulicFieldSimulator - genau EIN
    Calculator-Knoten (erosion.hydraulic), weil die acht Passes einen einzigen,
    eng gekoppelten Zustand teilen (Hoehe, Wasser, Fluss, Sediment). Sie ueber
    mehrere Knoten zu trennen hiesse, diesen Zustand durch den Storage zu
    schleusen.

    Aufgabe: Sitzt in der Pipeline ZWISCHEN Geology und Weather:

        terrain.redistribution -> geology.* -> erosion.hydraulic -> weather.* -> water.*

    Das ist der eigentliche Grund fuer den eigenen Generator. Bis 2026-07-28
    lag die Erosion im Water-Block und damit HINTER Weather - Temperatur, Wind
    und Niederschlag rechneten also auf dem UNerodierten Gelaende, obwohl die
    Erosion per Vorgabe gar nicht vom Regen abhaengt. Mit dem eigenen Knoten
    sehen alle nachgelagerten Generatoren die tatsaechlichen Taeler.

    Rechnet NUR in der letzten LOD-Runde (siehe _is_final_lod(), Muster von
    SettlementGenerator uebernommen); alle frueheren Runden schreiben
    Nullkarten, damit nachgelagerte Knoten nie auf einen fehlenden Key laufen.
    """

    # Dieselbe Liste steuert die Ablage im DataLODManager (siehe
    # set_erosion_data_complete_lod()).
    EROSION_DATA_KEYS = (
        "erosion_map", "sedimentation_map", "thermal_erosion_map", "thermal_deposition_map",
        "sediment_load_map", "water_depth_map", "flow_velocity_map",
    )

    # Skalare Kennzahlen des Laufs - keine Karten, deshalb getrennt gefuehrt.
    EROSION_SCALAR_KEYS = ("steps_taken", "converged", "mass_balance", "simulation_resolution")

    # Der eine Calculator-Knoten dieses Generators.
    CALCULATOR_ID = "erosion.hydraulic"

    # Fallback-Kartenausdehnung, falls der DataLODManager noch keinen Live-Wert
    # vom Terrain-Tab hat (Standalone/Tests) - identisch zu
    # TERRAIN.MAP_DISTANCE_KM["default"] und zu
    # HydrologySystemGenerator.FALLBACK_MAP_DISTANCE_KM.
    FALLBACK_MAP_DISTANCE_KM = 10.0

    def __init__(self, map_seed=42, shader_manager=None, data_lod_manager=None):
        self.map_seed = map_seed
        self.logger = logging.getLogger(self.__class__.__name__)
        self.shader_manager = shader_manager
        self.data_lod_manager = data_lod_manager
        self.simulator = HydraulicFieldSimulator(shader_manager=shader_manager)

        self._current_parameters = {}
        # Muss als Attribut existieren, damit CalculatorThread es per hasattr
        # findet (identisches Muster wie HydrologySystemGenerator).
        self.progress_callback = None
        # Kennzahlen des letzten Laufs - das Statistik-Widget des Tabs liest
        # sie, damit z.B. eine wegen der CPU-Grenze reduzierte Aufloesung
        # sichtbar ist statt still zu bleiben.
        self._last_run_info = {}

    def set_active_parameters(self, parameters):
        """Parameter der laufenden Anfrage (vom GenerationOrchestrator gesetzt)."""
        self._current_parameters = dict(parameters or {})

    def set_progress_callback(self, callback):
        """Fortschritts-Callback (phase, percentage, message) - identisches
        Muster wie GeologySystemGenerator/HydrologySystemGenerator."""
        self.progress_callback = callback

    def _update_progress(self, phase, percentage, message):
        if self.progress_callback:
            self.progress_callback(phase, percentage, message)

    # ==================================================================
    # Calculator-Knoten
    # ==================================================================

    def _calc_hydraulic(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node erosion.hydraulic - der gesamte Erosionslauf.

        Laeuft NUR in der letzten LOD-Runde. In allen frueheren Runden werden
        Nullkarten geschrieben; das ist kein Platzhalter, sondern die
        physikalisch richtige Aussage "auf dieser Stufe wurde noch nicht
        erodiert" - get_calculator_combined_heightmap() zieht sie ab bzw.
        addiert sie und erhaelt korrekt das unerodierte Gelaende.
        """
        # Eigenen Vorstand dieses LODs verwerfen, BEVOR die Heightmap gelesen
        # wird: get_calculator_combined_heightmap() zieht genau die hier
        # erzeugten Karten ab. Bei einer zweiten Generierung auf demselben LOD
        # wuerde sonst auf dem bereits erodierten Gelaende weitererodiert und
        # der Abtrag mit jedem Lauf anwachsen. Gleiche Begruendung wie
        # DataLODManager.clear_calculator_node_output().
        self.data_lod_manager.clear_calculator_node_output(calculator_id, lod_level)

        heightmap = self.data_lod_manager.get_calculator_combined_heightmap(lod_level)
        hardness_map = self.data_lod_manager.get_calculator_output(
            "geology.hardness", "hardness_map", lod_level)
        missing = [name for name, value in
                   (("heightmap", heightmap), ("hardness_map", hardness_map)) if value is None]
        if missing:
            raise ValueError(
                "Erosion: fehlende Dependencies fuer LOD {}: {}".format(
                    lod_level, ", ".join(missing)))

        target_size = int(heightmap.shape[0])

        if not self._is_final_lod(calculator_id, lod_level):
            self._store_zero_result(calculator_id, lod_level, target_size)
            return

        simulation_size = self._resolve_simulation_size()
        sim_height = self._resample(heightmap, simulation_size)
        sim_hardness = self._resample(hardness_map, simulation_size)
        meters_per_pixel = self._meters_per_pixel(simulation_size)

        max_steps = int(self._current_parameters.get("max_steps", 8000))
        self._update_progress(
            "Hydraulic Erosion", 5,
            "Feld-Simulation auf {0}x{0}...".format(simulation_size))

        def on_step(step, total, terrain, convergence):
            """
            Zwischenstand vom Simulator - wird zum Ladebalken und zur
            Live-Vorschau.

            Der Balken folgt der KONVERGENZ, nicht der Schrittzahl. Die
            Schrittzahl allein sagt nichts, weil die Gesamtzahl vorab unbekannt
            ist: das Ziel ist ein Konvergenzzustand, kein Schrittziel, und
            `max_steps` ist nur eine Obergrenze, die im Normalfall gar nicht
            erreicht wird.
            """
            self._last_run_info["preview_terrain"] = terrain
            self._last_run_info["convergence"] = convergence
            self._update_progress(
                "Hydraulic Erosion", 5 + int(85.0 * convergence),
                "Schritt {}, Konvergenz {:.0f}% (Obergrenze {})".format(
                    step, 100.0 * convergence, total))

        result = self.simulator.simulate(
            sim_height, sim_hardness, self._current_parameters, meters_per_pixel,
            progress_callback=on_step)

        self._update_progress("Hydraulic Erosion", 95, "Ergebnis skalieren...")

        outputs = {key: self._resample(result[key], target_size)
                   for key in self.EROSION_DATA_KEYS}
        outputs["steps_taken"] = result["steps_taken"]
        outputs["converged"] = result["converged"]
        outputs["mass_balance"] = result["mass_balance"]
        outputs["simulation_resolution"] = simulation_size
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, outputs)

        self._last_run_info = {
            "steps_taken": result["steps_taken"],
            "converged": result["converged"],
            "mass_balance": result["mass_balance"],
            "simulation_resolution": simulation_size,
        }
        self.logger.info(
            "Erosion LOD %d: %d Schritte auf %dx%d, konvergiert=%s, Bilanzabweichung %.2f%%",
            lod_level, result["steps_taken"], simulation_size, simulation_size,
            result["converged"], 100.0 * result["mass_balance"])

    def _store_zero_result(self, calculator_id, lod_level, target_size):
        """Nullkarten fuer die Zwischenrunden - siehe _calc_hydraulic()."""
        zeros = np.zeros((target_size, target_size), dtype=np.float32)
        outputs = {key: zeros.copy() for key in self.EROSION_DATA_KEYS}
        outputs.update({"steps_taken": 0, "converged": False,
                        "mass_balance": 0.0, "simulation_resolution": 0})
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, outputs)

    # ==================================================================
    # Hilfsmittel
    # ==================================================================

    def _resolve_simulation_size(self) -> int:
        """
        Auf welcher Aufloesung die Simulation tatsaechlich rechnet.

        Der eingestellte Wert (simulation_resolution) ist bewusst UNABHAENGIG
        von map_size: die Laufzeit haengt an der Zellzahl mal der Schrittzahl,
        und die Optik soll ueber alle Kartengroessen gleich bleiben. Das
        Ergebnis wird anschliessend auf die Kartengroesse skaliert.

        Ohne GPU greift die CPU-Grenze (siehe
        HydraulicFieldSimulator.MAX_CPU_RESOLUTION). Das wird geloggt und im
        Statistik-Widget angezeigt - eine stille Abweichung waere genau die Art
        Falle, die in diesem Projekt zuletzt systematisch entfernt wurde.
        """
        requested = int(self._current_parameters.get("simulation_resolution", 512))
        limit = HydraulicFieldSimulator.MAX_CPU_RESOLUTION
        if not self.simulator.has_gpu_path() and requested > limit:
            self.logger.warning(
                "Erosion: Simulationsaufloesung %d auf %d begrenzt - ohne "
                "registrierten GPU-Pfad ist mehr nicht in vertretbarer Zeit zu "
                "rechnen (siehe HydraulicFieldSimulator.has_gpu_path()).",
                requested, limit)
            return limit
        return requested

    @staticmethod
    def _resample(array, target_size: int):
        """
        Bilineare Skalierung auf target_size, Randbehandlung 'nearest' -
        dasselbe Verfahren wie HydrologySystemGenerator._interpolate_2d(),
        damit skalierte Karten sich zwischen beiden Generatoren nicht subtil
        unterscheiden.
        """
        array = np.asarray(array, dtype=np.float32)
        if array.shape[0] == target_size:
            return array
        source = array.shape[0]
        coords = np.linspace(0, source - 1, target_size)
        grid_y, grid_x = np.meshgrid(coords, coords, indexing="ij")
        return map_coordinates(array, [grid_y, grid_x], order=1,
                               mode="nearest").astype(np.float32)

    def _is_final_lod(self, calculator_id: str, lod_level: int) -> bool:
        """
        Ist lod_level die letzte Runde? Identisches Kriterium und identische
        Quelle wie HydrologySystemGenerator._is_final_lod() und
        SettlementGenerator._is_final_lod(): das beim Request gesetzte Ziel-LOD
        aus DataLODManager.get_calculator_target_lod(). Der Wert steht fest,
        sobald der Request gestellt wurde, und ist damit unabhaengig vom
        Fortschritt anderer Generatoren.
        """
        target = self.data_lod_manager.get_calculator_target_lod(calculator_id)
        if target is not None:
            return lod_level >= target

        from gui.OldManagers.data_lod_manager import calculate_max_lod_for_size
        full_heightmap = self.data_lod_manager.get_terrain_data("heightmap")
        if full_heightmap is not None:
            return lod_level >= calculate_max_lod_for_size(full_heightmap.shape[0])
        return lod_level >= self.data_lod_manager.get_max_lod_for_map_size()

    def _meters_per_pixel(self, target_size: int) -> float:
        """Reale Kantenlaenge einer Zelle in Metern - identische Ableitung wie
        HydrologySystemGenerator._meters_per_pixel()."""
        map_distance_km = self.data_lod_manager.get_map_distance_km()
        if not map_distance_km or map_distance_km <= 0:
            map_distance_km = self.FALLBACK_MAP_DISTANCE_KM
        return (map_distance_km * 1000.0) / target_size

    # ==================================================================
    # Domain-Objekt
    # ==================================================================

    def assemble_erosion_data(self, lod_level: int, parameters) -> ErosionData:
        """
        Funktionsweise: Baut das ErosionData-Objekt aus den Calculator-Outputs
        zusammen.
        Aufgabe: Wird vom GenerationOrchestrator aufgerufen, sobald
        erosion.hydraulic sein LOD abgeschlossen hat (siehe
        _maybe_assemble_generator()).
        """
        data = ErosionData(lod_level=lod_level)

        missing = []
        for key in self.EROSION_DATA_KEYS:
            value = self.data_lod_manager.get_calculator_output(
                self.CALCULATOR_ID, key, lod_level)
            if value is None:
                missing.append(key)
            setattr(data, key, value)
        if missing:
            raise ValueError(
                "assemble_erosion_data: fehlende Calculator-Outputs fuer LOD {}: {}".format(
                    lod_level, ", ".join(missing)))

        for key, default in (("steps_taken", 0), ("converged", False),
                             ("mass_balance", 0.0), ("simulation_resolution", 0)):
            value = self.data_lod_manager.get_calculator_output(
                self.CALCULATOR_ID, key, lod_level)
            setattr(data, key, default if value is None else value)
        return data
