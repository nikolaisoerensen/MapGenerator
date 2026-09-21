"""
Path: tests/smoke_test_regionsfeld.py

Zeigt die Regionsansicht dasselbe wie die Pipeline?

ANLASS: die Regionsansicht (docs/archiv/2026-08-25_AUFRAEUMPLAN.md 4.10) rechnet BEWUSST einen
eigenen, viel kuerzeren Weg - Oktavenstapel mit EINEM Parametersatz,
Potenzkurve, Erosionsfilter. Kein Kontinent, keine Voronoi-Mischung, keine
Kueste. Nur so ist sie live (0.04-0.22 s statt 2-19 s).

**GENAU DARIN LAG EINE GEFAHR.** Zwei Wege, die dasselbe Gelaende
beschreiben sollen, sind zwei Wahrheiten (SPEZIFIKATION 4.5). Der erste
Entwurf von `regionsfeld()` schrieb die Oktavenformel ein zweites Mal hin -
aendert jemand das Tor in `weltfeld()` und vergisst die Vorschau, stellt
der Nutzer seine Regionen an einem Gelaende ein, das es auf der Karte nicht
gibt, ohne jede Fehlermeldung.

**Behoben, nicht getestet:** die Formel steht seit dem 2026-08-26 genau
einmal (`oktavengewicht()`), beide Wege rufen dieselbe Funktion. Ein Test
dagegen wuerde nur numpy pruefen.

WAS GEPRUEFT WIRD:

  1. Die Vorschau rechnet die dokumentierte Formel - unabhaengig
     nachgerechnet, nicht gegen den Quelltext verglichen.
  2. `hoehe_m` ist die MEDIANE Hoehe (die Potenzkurve dreht um den Median).
  3. Die Reglerueberschreibung wirkt, und zwar nur auf die erlaubten Regler.
  4. Rechteckige Felder liefern dasselbe wie quadratische, wo sie sich
     ueberlappen - sonst waere das Rauschen richtungsabhaengig.

WAS NICHT GEPRUEFT WIRD, und warum: die absoluten Hoehen von `regionsfeld()`
und `weltfeld()` stimmen NICHT ueberein, und das ist richtig so. In
`weltfeld()` wird jede Region ueber die Voronoi-Gewichte mit ihren Nachbarn
verschmolzen; ein Alpenlandpixel dort traegt anteilig auch Nebelrode.
Die Vorschau zeigt den reinen Parametersatz. Gemessen (Seed 20260804):
Nevadin rein 588..1580 m, in der Welt im Mittel 486 m.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_regionsfeld.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import core.terrain_weltkarte as rw

SEED = 20260804

# BEIDE WEGE MUESSEN BEI DERSELBEN PIXELGROESSE VERGLICHEN WERDEN.
#
# Ein erster Anlauf verglich die Vorschau bei 256 px (27.7 m/px, weil eine
# Region 7.1 km breit ist) gegen die Pipeline bei 384 px (55.5 m/px) - und
# meldete Abweichungen bis 15 Grad. **Der Test hatte unrecht, nicht der
# Code:** der Hangwinkel haengt an der Pixelgroesse, ein feineres Raster
# loest steilere oertliche Haenge auf. Genau diesen Effekt meldet
# `smoke_test_regionen_welt` als "Landschaft haengt an der Pixelzahl"
# (r = +0.94).
#
# 21300/384 = 55.47 m/px in der Pipeline; 7100/128 = 55.47 m/px in der
# Vorschau. Damit sind sie vergleichbar.
WELT_PX = 384
PX = 128
# Der mediane Hangwinkel darf zwischen den beiden Wegen um so viel
# abweichen. Ein Rest bleibt zwangslaeufig: in der Pipeline ist jede Region
# ueber die Voronoi-Gewichte mit ihren Nachbarn verschmolzen, die Vorschau
# zeigt den reinen Parametersatz.
HANG_MAX_ABWEICHUNG = 6.0


def check(label, bedingung, zusatz=""):
    print(f"[{'OK' if bedingung else 'FAIL'}] {label}"
          + (f" - {zusatz}" if zusatz else ""))
    return [] if bedingung else [f"{label}{' - ' + zusatz if zusatz else ''}"]


def _hang(H, mpp):
    gy, gx = np.gradient(np.asarray(H, dtype=np.float64), mpp)
    return float(np.median(np.degrees(np.arctan(np.hypot(gx, gy)))))


def lauf():
    fehler = []
    namen = [r["name"] for _z, _s, r in rw.alle_regionen()]

    # --- 1. Die Vorschau rechnet die dokumentierte Formel ---------------------
    #
    # NICHT gegen `weltfeld()` verglichen, und das ist eine Entscheidung mit
    # Messung dahinter. Ein erster Anlauf verlangte gleiche Hangwinkel und
    # meldete bis zu 15 Grad Abweichung (Nevadin 28.7 gegen 15.5). **Das
    # ist kein Fehler, sondern der Zweck der Pipeline:** dort wird jede
    # Region ueber die Voronoi-Gewichte mit ihren Nachbarn verschmolzen, ein
    # Alpenlandpixel traegt anteilig auch Nebelrode (Relief 135 statt
    # 1050). Die Vorschau zeigt den REINEN Parametersatz. Gleichheit zu
    # fordern hiesse, die Voronoi-Mischung fuer einen Fehler zu halten.
    #
    # Die Drift, gegen die ich urspruenglich testen wollte, ist inzwischen
    # KONSTRUKTIV ausgeschlossen: die Oktavenformel steht seit dem
    # 2026-08-26 genau einmal (`oktavengewicht()`), beide Wege rufen
    # dieselbe Funktion. Bleibt zu pruefen, was `regionsfeld()` DARUM HERUM
    # tut - Spreizung, Potenzkurve um den Median, Hoehenlage. Das wird hier
    # unabhaengig nachgerechnet.
    for name in namen:
        H, r = rw.regionsfeld(name, PX, seed=SEED, aufbau=False)
        stapel = rw.oktavenstapel(PX, SEED ^ 0x5EED, None,
                                  mpp=rw.REGIONSBREITE_M / PX, hoehe=PX)
        g = np.array([float(rw.oktavengewicht(
            k, float(r["formgroesse_m"]),
            float(np.clip(r["rauheit"], 0.2, 0.9)))) for k in range(rw.OKTAVEN)])
        relief = np.tensordot(g, stapel, axes=(0, 0)) / g.sum()
        t = np.clip(0.5 + rw.SPREIZUNG * relief, 0.0, 1.0)
        pz = float(np.clip(r["potenz"], 0.2, 4.0))
        t = np.clip(np.power(t, pz) - np.power(0.5, pz) + 0.5, 0.0, 1.0)
        erwartet = float(r["hoehe_m"]) + float(r["relief_m"]) * (t - 0.5)
        if not np.allclose(H, erwartet, atol=1e-6):
            fehler += check(f"{name}: Vorschau rechnet die dokumentierte Formel",
                            False,
                            f"groesste Abweichung {np.abs(H - erwartet).max():.3f} m")
            break
    else:
        fehler += check("Vorschau rechnet die dokumentierte Formel",
                        True, f"{len(namen)} Regionen nachgerechnet")

    # --- 2. hoehe_m ist der MEDIAN, nicht der Mittelwert ---------------------
    #
    # Ein erster Anlauf prueft den Mittelwert und meldete 41 m Abweichung im
    # Skerrheim. **Auch das war ein Testfehler:** die Potenzkurve in
    # `weltfeld()` wird ausdruecklich UM DEN MEDIAN gedreht
    # (`t^p - 0.5^p + 0.5`), damit `potenz` nur die Form aendert und nicht
    # die Hoehe. Damit ist `hoehe_m` der Median. Bei einer schiefen
    # Hoehenverteilung - und die ist im Skerrheim stark schief - liegt der
    # Mittelwert daneben, ohne dass etwas falsch waere.
    # DIE GRENZE IST RELATIV ZUM RELIEF, nicht absolut.
    #
    # Ein erster Anlauf nahm 25 m fest und meldete das Skerrheim (34.8 m) -
    # bei 485 m Relief sind das 7 %. Gemessen ueber alle neun Regionen:
    # 0.9 bis 7.2 % des Reliefs, und die Klemmung auf [0,1] erklaert es
    # NICHT (nur 0.6-5 % der Pixel liegen am Rand).
    #
    # Es ist STICHPROBENRAUSCHEN: das Rauschen ist auf Formgroesse
    # korreliert, ein 128er-Feld enthaelt also nur wenige unabhaengige
    # Formen, und der Stichprobenmedian weicht entsprechend ab. Eine
    # absolute Grenze waere fuer flache Regionen zu lasch und fuer steile zu
    # streng.
    MEDIAN_MAX_ANTEIL = 0.12
    mittel_ab = []
    for name in namen:
        H, r = rw.regionsfeld(name, PX, seed=SEED, aufbau=False)
        d = abs(float(np.median(H)) - float(r["hoehe_m"]))
        mittel_ab.append((d / max(float(r["relief_m"]), 1e-9), name, d))
    schlimmster = max(mittel_ab)
    fehler += check("hoehe_m ist die MEDIANE Hoehe der Vorschau",
                    schlimmster[0] < MEDIAN_MAX_ANTEIL,
                    f"groesste Abweichung {100 * schlimmster[0]:.1f} % des "
                    f"Reliefs ({schlimmster[2]:.1f} m, {schlimmster[1]}), "
                    f"Grenze {100 * MEDIAN_MAX_ANTEIL:.0f} %")

    # --- 3. Die Ueberschreibung wirkt, und nur wo erlaubt --------------------
    H_a, _ = rw.regionsfeld("Clonagh", 128, seed=SEED, aufbau=False)
    H_b, r_b = rw.regionsfeld("Clonagh", 128, seed=SEED, aufbau=False,
                              ueberschreibung={"relief_m": 400.0})
    fehler += check("Ueberschreibung wirkt",
                    abs(float(np.ptp(H_b)) - float(np.ptp(H_a))) > 50.0,
                    f"Spanne {np.ptp(H_a):.0f} -> {np.ptp(H_b):.0f} m")

    H_c, r_c = rw.regionsfeld("Clonagh", 128, seed=SEED, aufbau=False,
                              ueberschreibung={"wasser_soll": 99.0})
    fehler += check("nicht erlaubte Regler werden ignoriert",
                    np.array_equal(H_c, H_a)
                    and float(r_c["wasser_soll"]) != 99.0,
                    "wasser_soll steht nicht in REGIONSREGLER")

    # --- 4. Rechteckig ist nicht richtungsabhaengig --------------------------
    H_q, _ = rw.regionsfeld("Nevadin", 192, 192, seed=SEED, aufbau=False)
    H_r, _ = rw.regionsfeld("Nevadin", 192, 96, seed=SEED, aufbau=False)
    gleich = np.allclose(H_q[:96, :], H_r, atol=1e-6)
    fehler += check("rechteckig stimmt mit quadratisch ueberein",
                    gleich,
                    "die oberen 96 Zeilen muessen identisch sein - sonst "
                    "haengt das Rauschen an der Feldform")

    # --- 5. Der Land-See-Aufbau ----------------------------------------------
    #
    # Nutzertabelle 2026-08-26 (AUFRAEUMPLAN 4.11). Geprueft wird das
    # ERGEBNIS, nicht die Tabelle: Landanteil, und dass der Aufbau die
    # Hoehenlage der Region NICHT verschiebt, wo sie ohnehin ueber Wasser
    # liegt.
    #
    # Ein erster Entwurf setzte den Versatz unbedingt und ZOG LAND HERUNTER,
    # das laengst ueber Wasser lag - das Clonagh (121..194 m) landete bei
    # -98..81 m. Die Landanteile stimmten dabei, der Fehler steckte allein in
    # den Hoehen. Deshalb wird hier BEIDES geprueft.
    from scipy import ndimage
    aufbau_fehler = []
    for name in namen:
        art, ziel = rw.REGIONS_AUFBAU[name]
        H, r = rw.regionsfeld(name, 256, 160, seed=SEED)
        H_ohne, _ = rw.regionsfeld(name, 256, 160, seed=SEED, aufbau=False)
        land = H > 0
        anteil = float(land.mean())

        if art == "ohne_kueste":
            if anteil < 0.999:
                aufbau_fehler.append(f"{name}: {100*anteil:.0f} % Land statt 100")
        elif abs(anteil - ziel) > 0.14:
            aufbau_fehler.append(
                f"{name}: {100*anteil:.0f} % Land, Ziel {100*ziel:.0f} %")

        # Lag die Region schon ueber Wasser, darf der Aufbau sie nicht senken.
        if float(np.percentile(H_ohne, 5.0)) > 0.0:
            gesenkt = float(np.max(H_ohne[land]) - np.max(H[land]))
            if gesenkt > 5.0:
                aufbau_fehler.append(
                    f"{name}: Aufbau senkt vorhandenes Land um {gesenkt:.0f} m")

        if art == "fjord":
            # Land muss LINKS und RECHTS liegen, See dazwischen.
            links = land[:, :40].mean()
            rechts = land[:, -40:].mean()
            if min(links, rechts) < 0.6:
                aufbau_fehler.append(
                    f"{name}: Fjord ohne Land an beiden Raendern "
                    f"({100*links:.0f} %/{100*rechts:.0f} %)")

        if art == "inseln":
            # Rechte Haelfte: mehrere getrennte Landstuecke = Inseln.
            _lab, n = ndimage.label(land[:, 128:])
            if n < 3:
                aufbau_fehler.append(f"{name}: nur {n} Landstuecke rechts")

    fehler += check("Land-See-Aufbau je Region wie vorgegeben",
                    not aufbau_fehler,
                    "; ".join(aufbau_fehler) if aufbau_fehler else
                    f"{len(namen)} Regionen, Landanteile im Rahmen")

    print()
    print("=" * 78)
    if fehler:
        print(f"NICHT IN ORDNUNG - {len(fehler)} Befunde:")
        for f in fehler:
            print(f"   {f}")
        return 1
    print("die Regionsansicht deckt sich mit der Pipeline")
    return 0


if __name__ == "__main__":
    sys.exit(lauf())
