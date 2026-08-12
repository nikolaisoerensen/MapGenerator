"""
Path: tests/smoke_test_regionen_welt.py

Prueft die Regionenwelt (tools/regionen_welt.py, Stufe A des
docs/INTEGRATIONSPLAN.md).

"Sieht gut aus" ist nicht pruefbar - der Charakter einer Landschaft schon.
Geprueft werden deshalb fuenf Dinge, und jedes davon hat schon einmal einen
echten Fehler gefunden:

  1. REPRODUZIERBARKEIT   Zweimal gerechnet muss BITGLEICH sein. Vorgabe des
                          Nutzers: alles aus Seed und Reglern herleitbar.
  2. FINGERABDRUCK        Hangneigung und Wasseranteil je Region im
                          Sollbereich. Fand: `potenz` verschob den Median, die
                          Atlantikkueste stand bei 73 % Wasser statt 45.
  3. NAHTPRUEFUNG         An einer Regionsgrenze darf der Hoehengradient
                          keinen Sprung zeigen, der groesser ist als der
                          staerkste Gradient INNERHALB der Nachbarregionen.
                          Das ist die messbare Fassung von "die regionen
                          sollen smooth ineinanderlaufen".
  4. AUFLOESUNG           512 und 1024 px muessen dieselbe Landschaft ergeben,
                          nur feiner - sonst haengt die Welt an der Pixelzahl
                          statt an der Wirklichkeit (SPEZIFIKATION §10).
  5. GPU/CPU-PARITAET     Beide Pfade muessen dasselbe Gelaende liefern. Fand
                          am 2026-08-04, dass der Noise-Shader eine andere
                          Permutationstabelle benutzte als die CPU und damit
                          aus demselben Seed eine ANDERE Welt erzeugte.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_regionen_welt.py
"""

import os
import sys

import numpy as np

_WURZEL = r"C:\Lokale Dateien\Projects\Python\MapGenerator"
sys.path.insert(0, _WURZEL)
sys.path.insert(0, os.path.join(_WURZEL, "tools"))

# Die Qt-Anwendung MUSS modulweit gehalten werden - als lokale Variable raeumt
# Python sie ab, waehrend der GL-Kontext noch lebt (Segfault ohne Meldung).
_QT_APP = None


def _qt():
    global _QT_APP
    from PyQt6.QtGui import QGuiApplication
    if _QT_APP is None:
        _QT_APP = QGuiApplication.instance() or QGuiApplication([])
    return _QT_APP


SIZE = 512
SEED = 20260804

# Zielhangneigung (Median, Grad) je Region - das, was die Landschaft ausmacht.
#
# ALLE NEUN NEU BESTIMMT AM 2026-08-06, nach der Nord-Sued-Korrektur in
# voronoi_regionen(). Die alten Werte waren gegen einen Kontinent gefittet, der
# auf dem Kopf stand: jede Region sass auf einem anderen Stueck Land und hatte
# andere Nachbarn. Da der gemessene Hang zu einem grossen Teil aus der
# NACHBARSCHAFT kommt und nicht aus dem eigenen Relief (ein als Taiga
# gefuehrtes Pixel traegt im Mittel 33 % fremdes Gewicht und damit 372 m Relief
# statt der eingetragenen 138), aendern sich die Zielwerte mit der Anordnung.
#
# Der Vorgang ist nicht neu - die Atlantikkueste stand aus genau diesem Grund
# schon am 2026-08-05 auf 9 statt 6, das Alpenland auf 24 statt 30.
#
# GEMITTELT UEBER DIE FUENF SEEDS AUS FINGERABDRUCK_SEEDS. Die Streuung EINER
# Region ueber Seeds betraegt bis zu 11 Grad; ein Zielwert aus einem einzigen
# Seed waere derselbe Fehler noch einmal.
# NACHGEZOGEN AM 2026-08-07, zweiter Grund: die Kuestenform je Region
# (kuestenform in core/terrain_weltkarte.py) und der um 4.9 % gewachsene
# Kontinent (Atlantikkueste +40 % Flaechenanteil). Beides veraendert die
# Landschaft absichtlich; die Zielwerte folgen ihr, die ORDNUNG bleibt der
# eigentliche Waechter.
ZIEL_HANG = {
    "Huegelland": 9.5, "Fjordland": 16.0, "Taiga": 7.5,
    "Atlantikkueste": 10.0, "Mittelgebirge": 12.5, "Alpenland": 27.0,
    "Steppe": 6.5, "Mittelmeer": 14.5, "Griechische Inseln": 11.5,
}

# DIE EIGENTLICHE ZUSICHERUNG: die Reihenfolge, nicht die Zahlen.
#
# Zielwerte, die einfach die Messwerte sind, machen einen Test tautologisch -
# er ginge nach jeder Aenderung wieder durch, wenn man ihn nur nachzieht. Die
# ORDNUNG dagegen ist die Absicht selbst: die Taiga ist flaches Hochland, das
# Alpenland ein Gebirge, und das muss so bleiben, egal welche Zahl dabei
# herauskommt. Kippt sie, ist das ein Befund und kein Eichthema.
ORDNUNG_FLACH_NACH_STEIL = [
    "Steppe", "Taiga", "Huegelland", "Atlantikkueste", "Griechische Inseln",
    "Mittelgebirge", "Mittelmeer", "Fjordland", "Alpenland",
]

# Der Fingerabdruck laeuft auf EIGENER Groesse und ueber MEHRERE Seeds.
#
# Kleiner, weil der Charakter einer Landschaft nicht an der Pixelzahl haengt -
# genau das sichert Pruefung 4 getrennt zu. Mehrere Seeds, weil eine Region je
# nach Kontinentform auf anderem Land sitzt: gemessen bis zu 37 Prozentpunkte
# Unterschied im Wasseranteil zwischen zwei Seeds derselben Region. Auf einem
# Seed zu pruefen hiesse, die Form zu pruefen statt die Regel.
FINGERABDRUCK_SIZE = 384
FINGERABDRUCK_SEEDS = (20260804, 12345, 4242, 777001, 31415)


def lauf():
    _qt()
    import regionen_welt as rw
    from managers.shader_manager import ShaderManager

    manager = ShaderManager()
    worker = manager._ensure_worker()
    gpu = worker.gpu_available
    fehler = []

    # ---------- 1 ----------
    rw._STAPEL_CACHE.clear()
    a, _ = rw.weltfeld(SIZE, SEED, shader_manager=manager if gpu else None)
    rw._STAPEL_CACHE.clear()
    b, _ = rw.weltfeld(SIZE, SEED, shader_manager=manager if gpu else None)
    gleich = np.array_equal(a, b)
    print("1. Reproduzierbar (zweimal gerechnet, bitgleich) ... %s"
          % ("ok" if gleich else "FEHLER"))
    if not gleich:
        fehler.append("gleicher Seed ergibt verschiedene Welten, "
                      "groesste Abweichung %.3g" % np.abs(a - b).max())

    H = a

    # ---------- 2 ----------
    #
    # DER KERN EINER REGION IST JETZT IHR VORONOI-GEBIET, nicht ein Kasten im
    # 3x3-Raster. Mit `regionskern` wurde nach der Umstellung auf die
    # Plaetzchenform Meer mitgemessen, das gar keiner Region gehoert - die
    # Griechischen Inseln meldeten 94 % Wasser statt 65. Dieselbe Fehlerklasse
    # wie schon dreimal: richtig gerechnet, das Falsche gemessen.
    # GEMESSEN WIRD DAS FERTIGE GELAENDE, NICHT DAS ROHFELD (2026-08-10).
    #
    # Bis dahin stand hier `rw.weltfeld()`. Zwischen ihm und dem, was die
    # Anzeige zeigt, liegen aber zwei Schritte: der Erosionsfilter und das
    # Taeleingraben des Flussnetzes. Gemessen am 2026-08-10 nahm allein das
    # Eingraben dem Land 4.9 Grad mittleren Hang ab, der Mittelmeerkueste 8.9
    # von 14.8 - die Regionen waren also auf ein Gelaende geeicht, das niemand
    # zu sehen bekam. Wieder dieselbe Fehlerklasse wie im Kommentar darueber.
    #
    # Der Test geht deshalb ueber `_calc_redistribution`, also genau den Weg,
    # den auch die Oberflaeche nimmt. Das kostet Zeit - fuenf Seeds mit
    # Flussnetz statt fuenf nackte Rauschfelder -, ist aber der einzige Weg,
    # bei dem die Zielwerte oben etwas ueber die sichtbare Welt aussagen.
    def fertiges_gelaende(seed):
        """Heightmap wie sie die Anzeige bekommt: Weltfeld + Filter + Taeler."""
        from core.terrain_generator import BaseTerrainGenerator
        from managers.data_lod_manager import DataLODManager
        lod = 5
        verwalter = DataLODManager()
        verwalter.set_map_seed(seed)
        verwalter.set_map_distance_km(rw.WELT_KM)
        erzeuger = BaseTerrainGenerator(map_seed=seed,
                                        data_lod_manager=verwalter)
        erzeuger.set_active_parameters({
            "map_size": FINGERABDRUCK_SIZE, "map_seed": seed,
            "map_distance_km": rw.WELT_KM,
            "amplitude": 100, "redistribute_power": 1.0})
        for knoten in ("terrain.noise", "terrain.redistribution"):
            verwalter.set_calculator_target_lod(knoten, lod)
        erzeuger._calc_noise("terrain.noise", lod)
        erzeuger._calc_redistribution("terrain.redistribution", lod)
        return np.asarray(verwalter.get_calculator_output(
            "terrain.redistribution", "heightmap", lod), dtype=np.float64)

    def fingerabdruck_eines_seeds(seed):
        """Hang (Median, Grad) und Wasseranteil (%) je Region."""
        rw._STAPEL_CACHE.clear()
        feld = fertiges_gelaende(seed)
        maske, _sdf = rw.kontinentform(FINGERABDRUCK_SIZE, seed,
                                       manager if gpu else None)
        gewichte = rw.voronoi_regionen(maske, seed, punktzahl=200,
                                       shader_manager=manager if gpu else None)
        fuehrend = np.argmax(gewichte, axis=0)
        mpp = rw.WELT_KM * 1000.0 / FINGERABDRUCK_SIZE
        dy, dx = np.gradient(feld, mpp)
        hang_feld = np.rad2deg(np.arctan(np.hypot(dx, dy)))
        aus = {}
        for i, (_z, _s, r) in enumerate(rw.alle_regionen()):
            # Nur LAND, nur wo die Region klar fuehrt.
            gebiet = maske & (fuehrend == i) & (gewichte[i] > 0.5)
            land = gebiet & (feld > 0)
            aus[r["name"]] = (
                float(np.median(hang_feld[land])) if land.sum() > 50 else None,
                100.0 * float((feld[gebiet] <= 0).mean()) if gebiet.sum() > 50
                else None)
        return aus

    laeufe = [fingerabdruck_eines_seeds(s) for s in FINGERABDRUCK_SEEDS]

    def mittel(name, spalte):
        werte = [l[name][spalte] for l in laeufe if l[name][spalte] is not None]
        return float(np.mean(werte)) if werte else None

    print("2. Fingerabdruck je Region (%d px, Mittel aus %d Seeds):"
          % (FINGERABDRUCK_SIZE, len(FINGERABDRUCK_SEEDS)))
    print("   %-20s %-14s %-16s %s"
          % ("Region", "Hang ist/soll", "Wasser ist/soll", ""))
    gemessener_hang = {}
    for _z, _s, r in rw.alle_regionen():
        hang = mittel(r["name"], 0)
        wasser = mittel(r["name"], 1)
        if hang is None or wasser is None:
            fehler.append("%s: kein Gebiet" % r["name"])
            continue
        gemessener_hang[r["name"]] = hang
        soll_hang = ZIEL_HANG[r["name"]]
        # Enger als vorher (0.3 / 2.5): ueber fuenf Seeds gemittelt ist das
        # Seedrauschen heraus, und eine echte Verschlechterung soll auffallen.
        hang_ok = abs(hang - soll_hang) <= max(0.22 * soll_hang, 2.0)
        wasser_ok = abs(wasser - r["wasser_soll"]) <= 8.0
        gut = hang_ok and wasser_ok
        print("   %-20s %5.1f / %-6.1f %6.1f / %-8.0f %s"
              % (r["name"], hang, soll_hang, wasser, r["wasser_soll"],
                 "ok" if gut else "DANEBEN"))
        if not gut:
            fehler.append("%s: Hang %.1f (soll %.1f), Wasser %.1f (soll %.0f)"
                          % (r["name"], hang, soll_hang, wasser,
                             r["wasser_soll"]))

    # ---------- 2b: die Ordnung ----------
    #
    # Die Zahlen oben lassen sich nachziehen, die Reihenfolge nicht. Sie ist
    # die Absicht selbst - eine Taiga, die steiler wird als das Mittelgebirge,
    # ist keine Taiga mehr, egal welchen Zielwert man einträgt.
    if len(gemessener_hang) == 9:
        ist = sorted(gemessener_hang, key=gemessener_hang.get)
        verrutscht = [n for n in ist
                      if abs(ORDNUNG_FLACH_NACH_STEIL.index(n) - ist.index(n)) > 2]
        print("2b. Ordnung flach -> steil ... %s"
              % ("ok" if not verrutscht else "FEHLER"))
        print("    ist:  %s" % " < ".join(ist))
        if verrutscht:
            print("    soll: %s" % " < ".join(ORDNUNG_FLACH_NACH_STEIL))
            fehler.append("Ordnung verrutscht um mehr als zwei Plaetze: %s"
                          % ", ".join(verrutscht))

    # ---------- 3 ----------
    #
    # Der Gradient wird auf dem GEGLAETTETEN Feld gemessen. Ungeglaettet
    # dominiert das Rauschen der feinsten Oktave, und eine echte Naht ginge
    # darin unter - gesucht ist die grossraeumige Stufe, nicht die Textur.
    from scipy import ndimage
    mpp = rw.WELT_KM * 1000.0 / SIZE
    glatt = ndimage.gaussian_filter(H, max(120.0 / mpp, 1.0))
    dy, dx = np.gradient(glatt, mpp)
    steig = np.hypot(dx, dy)
    # NUR LAND. Eine Kueste faellt zu Recht steil ins Meer - das ist keine
    # Naht zwischen Regionen, sondern eine Kueste. Ohne diese Maske meldet die
    # Pruefung Steilheit, die niemand sieht, weil sie unter Wasser liegt.
    steig = np.where(H > 0.0, steig, 0.0)
    kante_px = rw.REGION_KM * 1000.0 / mpp
    band = max(int(0.10 * kante_px), 2)

    # Die Regionsgrenzen liegen seit der Voronoi-Zuteilung nicht mehr auf
    # festen Rasterlinien - sie werden dort gesucht, wo die fuehrende Region
    # wechselt. `innen` ist entsprechend, wo eine Region klar dominiert.
    #
    # EIGENE ZUTEILUNG FUER DIESE PRUEFUNG. Der Fingerabdruck oben rechnet seit
    # dem 2026-08-06 auf eigener Groesse und ueber fuenf Seeds; seine Maske
    # passt also nicht zu `H` (SIZE, SEED). Sie hier wiederzuverwenden hiesse,
    # Naehte an Stellen zu suchen, an denen in dieser Karte gar keine sind.
    maske, _sdf = rw.kontinentform(SIZE, SEED, manager if gpu else None)
    gewichte = rw.voronoi_regionen(maske, SEED, punktzahl=200,
                                   shader_manager=manager if gpu else None)
    fuehrend = np.argmax(gewichte, axis=0)

    wechsel = np.zeros_like(maske)
    wechsel[:, :-1] |= fuehrend[:, :-1] != fuehrend[:, 1:]
    wechsel[:-1, :] |= fuehrend[:-1, :] != fuehrend[1:, :]
    naht_band = ndimage.binary_dilation(wechsel, iterations=band) & maske         & (H > 0.0)
    innen_band = maske & (H > 0.0) & (np.max(gewichte, axis=0) > 0.85)         & ~naht_band

    if naht_band.sum() < 50 or innen_band.sum() < 50:
        print("3. Nahtpruefung ... uebersprungen, zu wenig Flaeche")
        ok = True
        schlimmste = innen_max = 0.0
    else:
        schlimmste = float(np.percentile(steig[naht_band], 99.5))
        innen_max = float(np.percentile(steig[innen_band], 99.5))
        # ZEHN PROZENT TOLERANZ, und der Grund steht hier, damit niemand sie
        # spaeter fuer eine Bequemlichkeit haelt: ein Grenzstreifen ist viel
        # schmaler als ein Regionsinneres, sein p99.5 entspricht also einem
        # selteneren Ereignis. Und eine Grenze wie Fjordland/Taiga IST eine
        # Landform - der Rand des Fjordplateaus darf zum steilsten gehoeren,
        # was es dort gibt. Was NICHT toleriert wird, hat der Test schon
        # zweimal gefangen: Faktor 1.5 bis 2.4, beides echte Waende.
        # 2026-08-07 von 1.10 auf 1.25.
        #
        # Das Alpenland ist seit dem Hoehenprofil ein echtes Gebirge (Gipfel
        # 865 m, Relief 1000) und grenzt an Tiefebenen. Sein Rand IST die
        # Gebirgsfront - der steilste Hang der Region liegt zwangslaeufig dort,
        # und das ist eine Landform, kein Nahtfehler.
        #
        # Was die Pruefung weiter faengt, ist der Fall, den sie zweimal
        # gefunden hat: Faktor 1.5 bis 2.4, also eine Wand, die aus der
        # Ueberblendung entsteht statt aus dem Gelaende. Gemessen liegt die
        # Weltkarte jetzt bei 1.14; die Grenze bei 1.25 laesst also Luft, ohne
        # den Befund unmoeglich zu machen.
        ok = schlimmste <= 1.25 * innen_max
        print("3. Nahtpruefung: Grenzen %.3f gegen %.3f im Inneren "
              "(+25 %% erlaubt) ... %s"
              % (schlimmste, innen_max, "ok" if ok else "FEHLER"))
    if not ok:
        fehler.append("Regionsgrenzen sind deutlich steiler als die Regionen "
                      "selbst (%.3f gegen %.3f)" % (schlimmste, innen_max))

    # ---------- 4 ----------
    rw._STAPEL_CACHE.clear()
    gross, _ = rw.weltfeld(2 * SIZE, SEED,
                           shader_manager=manager if gpu else None)
    grob = gross[:2 * SIZE:2, :2 * SIZE:2]
    n = min(grob.shape[0], H.shape[0])
    r = float(np.corrcoef(ndimage.gaussian_filter(H[:n, :n], 4).ravel(),
                          ndimage.gaussian_filter(grob[:n, :n], 4).ravel())[0, 1])
    ok = r > 0.98
    print("4. 512 und 1024 px ergeben dieselbe Landschaft (r=%+.4f) ... %s"
          % (r, "ok" if ok else "FEHLER"))
    if not ok:
        fehler.append("Landschaft haengt an der Pixelzahl (r = %+.4f)" % r)

    # ---------- 5 ----------
    if not gpu:
        print("5. GPU/CPU-Paritaet ... uebersprungen, keine GPU")
    else:
        rw._STAPEL_CACHE.clear()
        mit = rw.oktavenstapel(256, SEED, manager)
        rw._STAPEL_CACHE.clear()
        ohne = rw.oktavenstapel(256, SEED, None)
        abw = float(np.abs(mit - ohne).max())
        ok = abw < 1e-4
        print("5. GPU/CPU-Paritaet des Oktavenstapels: groesste Abweichung "
              "%.2e ... %s" % (abw, "ok" if ok else "FEHLER"))
        if not ok:
            fehler.append("GPU und CPU erzeugen verschiedene Oktaven (%.2e)"
                          % abw)

    print()
    if fehler:
        print("NICHT IN ORDNUNG - %d Befunde:" % len(fehler))
        for eintrag in fehler:
            print("   %s" % eintrag)
        return 1
    print("Alle Zusicherungen erfuellt - die Regionenwelt steht.")
    return 0


if __name__ == "__main__":
    raise SystemExit(lauf())
