"""
Path: tests/smoke_test_kuestenprofiltreue.py

Sieht die ERZEUGTE Kueste so aus wie die GEMESSENEN Profile?

WARUM ES DIESEN TEST BRAUCHTE (Nutzerfrage 2026-08-24: "sag mal ob du
zugriff auf die ganzen kuestenprofile hast die es haben soll und ob es einen
test gibt ob die kuesten so aussehen wie die profile?")

Die Profile gibt es: `MESS_FORM_JE_ARCHETYP` (27 Archetypen, je 17
Stuetzstellen), `MESSWERTE_JE_REGION` (Klippenhoehe p10/p90) und
`MESS_REICHWEITE_M` in `core/vektor_kueste.py` - alle aus echten DEMs
gewonnen (tools/kuestenlaengsschnitt.py, Schnitte auf den Kuestennormalen
alle 100 m ueber die volle Vorbildkueste).

**Geprueft hat sie niemand.** `smoke_test_vektor_kueste.py` prueft sechs
Dinge - eine Funktion, Rasterfreiheit, Aufloesungsunabhaengigkeit,
Determinismus, Lage der 0-Linie, Randfaelle - und alle sechs sind
STRUKTURELL. Keines vergleicht die entstandene Form mit der Vorlage. Die
Tabellen haetten falsch abgetastet, vertauscht oder gar nicht angewandt
werden koennen, und jeder Test waere gruen geblieben. Genau das Muster aus
CLAUDE.md ("Gruene Tests koennen eine tote Funktion verdecken").

MIT DERSELBEN METHODE GEMESSEN

Der Vergleich benutzt `kuestenlinie()` und `schnitte_auf_normalen()` aus
`tools/kuestenlaengsschnitt.py` - dieselben Funktionen, mit denen die
Tabellen aus den echten Kuesten gewonnen wurden. Eine zweite Messmethode
waere kein Vergleich, sondern ein neuer Datensatz.

Die HOEHE wird nur INNERHALB der Profilreichweite gemessen, nicht ueber die
ganze Landseite (erste Fassung machte das falsch): ein Berg 800 m hinter der
Kueste hat mit dem Kuestenprofil nichts zu tun, und mit ihm sahen alle
Regionen zu hoch aus.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_kuestenprofiltreue.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tools")

from core.terrain_weltkarte import KUESTEN_ARCHETYPEN, alle_regionen, weltfeld
from core.vektor_kueste import (MESS_PROFIL_M_JE_ARCHETYP,
                                GEMESSENES_PROFIL_M, PROFIL_STELLEN_M,
                                PROFIL_VOLL_M, _zone_p1,
                                MESS_FORM_JE_ARCHETYP, MESS_REICHWEITE_M,
                                MESSWERTE_JE_REGION)

WELT_KM = 21.3
SIZE = 512
SEED = 20260804

# GRENZEN IN METERN, seit der Test die Meterprofile vergleicht.
#
# Bis 2026-08-24 stand hier eine RMS-Grenze auf der 0..1-Skala. Die hatte
# einen blinden Fleck: eine Kueste mit richtiger FORM und voellig falscher
# HOEHE war gruen. Die Weissmeer-Flachkueste stand bei 187 m, wo die
# Vorlage 15-30 m sagt, und der Test sah nur ihre Form.
#
# Gemessen nach dem Umbau (20 Gruppen, 512 px): Median 3.4 m.
FEHLER_MEDIAN_MAX_M = 8.0

# Was als FLACHE Kueste gilt, und wieviel sie danebenliegen darf.
#
# Flache Kuesten sind die eigentliche Zusicherung dieses Tests - sie waren
# der Anlass des Umbaus. Sie muessen eng treffen; gemessen liegen sie bei
# 0-2 m. Steile bekommen mehr Spielraum, weil eine 466-m-Fjordwand eine
# Landmasse braucht, die sie traegt - in einer 21-km-Welt mit schmalen
# Fjorden ist das die Ausnahme.
FLACH_GRENZE_M = 25.0
FLACH_FEHLER_MAX_M = 8.0

# BEKANNTE ABWEICHUNGEN, Stand 2026-08-24.
#
# Namentlich gefuehrt, damit sie nicht in Vergessenheit geraten UND damit
# eine NEUE Abweichung auffaellt, statt in einer Sammelzahl unterzugehen.
#
# Zur Zeit ist die Liste LEER.
#
#   San-Sebastian-Bucht stand hier bis zum 2026-08-24 (Ist 5 m, Soll 16 m).
#     Sie hat sich mit SAAT_KOHAERENZ_STATIONEN von selbst erledigt - der
#     Archetyp bekam vorher nur verstreute Einzelstationen und damit kein
#     Segment, das seinen Typ je zeigen konnte. Ist jetzt 16 / 16 m.
#     DAS WAR DIESELBE URSACHE wie bei der Fjordwand (23 statt 128 m); der
#     Verdacht "Landmasse der Samarcia" war falsch.
BEKANNTE_ABWEICHUNGEN = set()

# Archetypen, deren Name eine Klippe/Steilkueste ankuendigt.
KLIPPENWOERTER = ("Klippen", "Steil", "Kliff", "Wand", "Kreide")


def check(label, bedingung, zusatz=""):
    status = "OK" if bedingung else "FAIL"
    print(f"[{status}] {label}{(' - ' + zusatz) if zusatz else ''}")
    return bool(bedingung)


def _ist_klippe(name):
    return any(w in name for w in KLIPPENWOERTER)


def run_tabellen_sind_vollstaendig():
    """
    Billigste und wirksamste Zusicherung: jeder Archetyp, den eine Region
    benutzt, MUSS eine gemessene Form haben - sonst faellt er still auf die
    alte Einheitskurve zurueck und niemand sieht es.
    """
    ok = True
    fehlend_form = []
    benutzt = set()
    for _z, _s, r in alle_regionen():
        name = r["name"]
        for typ in KUESTEN_ARCHETYPEN.get(name, []):
            benutzt.add(typ["name"])
            if typ["name"] not in MESS_FORM_JE_ARCHETYP:
                fehlend_form.append(f"{name}/{typ['name']}")
        if name not in MESSWERTE_JE_REGION:
            ok &= check(f"Region '{name}' hat gemessene Klippenhoehe", False)
        if name not in MESS_REICHWEITE_M:
            ok &= check(f"Region '{name}' hat gemessene Reichweite", False)

    ok &= check("jeder benutzte Archetyp hat eine gemessene Form",
                not fehlend_form,
                ", ".join(fehlend_form) if fehlend_form
                else f"{len(benutzt)} Archetypen")

    laengen = {len(v) for v in MESS_FORM_JE_ARCHETYP.values()}
    ok &= check("alle Formen haben gleich viele Stuetzstellen",
                len(laengen) == 1, str(laengen))
    unmonoton = [n for n, v in MESS_FORM_JE_ARCHETYP.items()
                 if any(b < a - 1e-9 for a, b in zip(v, v[1:]))]
    ok &= check("alle Formen sind monoton steigend", not unmonoton,
                ", ".join(unmonoton))
    randfalsch = [n for n, v in MESS_FORM_JE_ARCHETYP.items()
                  if abs(v[0]) > 1e-9 or abs(v[-1] - 1.0) > 1e-9]
    ok &= check("alle Formen laufen von 0 nach 1", not randfalsch,
                ", ".join(randfalsch))

    # Verwaiste Eintraege sind Altlast - sie kosten nichts, verschleiern aber,
    # welche Archetypen wirklich im Einsatz sind.
    verwaist = sorted(set(MESS_FORM_JE_ARCHETYP) - benutzt)
    if verwaist:
        print(f"[--] {len(verwaist)} Formen ohne Region: {', '.join(verwaist)}")
    return ok


def _gruppen_messen():
    """Erzeugte Kueste nach (Region, Archetyp) gruppiert vermessen."""
    from kuestenlaengsschnitt import kuestenlinie, schnitte_auf_normalen

    H, felder = weltfeld(SIZE, SEED)
    H = np.asarray(H, dtype=np.float64)
    mpp = WELT_KM * 1000.0 / SIZE
    kopf = {"cellsize": mpp / 111320.0}

    archetyp_feld = felder.get("kuesten_archetyp")
    regionen = felder.get("regionen")
    if archetyp_feld is None or regionen is None:
        return None, None, "Archetyp- oder Regionenfeld fehlt"

    linien = kuestenlinie(H, kopf, mindestlaenge_m=2000.0, pegel=0.5)
    if not linien:
        return None, None, "keine Kuestenlinie gefunden"

    regionsnamen = {i: r["name"] for i, (_z, _s, r) in enumerate(alle_regionen())}
    gruppen = {}
    strecke = None
    for _laenge, linie in linien[:6]:
        stationen, _bogen, strecke, profile = schnitte_auf_normalen(
            H, kopf, linie, abstand_m=100.0)
        for k in range(len(stationen)):
            sx, sy = stationen[k]
            xi = int(np.clip(round(sx), 0, SIZE - 1))
            yi = int(np.clip(round(sy), 0, SIZE - 1))
            a = r = -1
            # Direkt auf der Wasserlinie ist das Archetypfeld oft -1 -
            # deshalb die Nachbarschaft absuchen, bevor die Station verfaellt.
            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                           (2, 0), (-2, 0), (0, 2), (0, -2)):
                xx = int(np.clip(xi + dx, 0, SIZE - 1))
                yy = int(np.clip(yi + dy, 0, SIZE - 1))
                if archetyp_feld[yy, xx] >= 0:
                    a = int(archetyp_feld[yy, xx])
                    r = int(regionen[yy, xx])
                    break
            if a < 0 or r < 0:
                continue
            name = regionsnamen.get(r)
            liste = KUESTEN_ARCHETYPEN.get(name, [])
            if not liste or a >= len(liste):
                continue
            gruppen.setdefault((name, liste[a]["name"]), []).append(profile[k])
    return gruppen, strecke, None


def _auswerten(gruppen, strecke):
    """
    Je Gruppe: Abweichung vom Vorlagenprofil IN METERN.

    UMGESTELLT 2026-08-24, weil sich die Messgroesse geaendert hat.

    Bis dahin verglich dieser Test NORMIERTE Formen: das erzeugte Profil
    wurde auf seine eigene Spanne gestreckt und gegen eine 0..1-Kurve
    gehalten. Das passte zum damaligen Modell (Form normiert, Hoehe aus
    dem Hinterland, Streckung auf die Regionsreichweite), hatte aber einen
    blinden Fleck: eine Kueste konnte die richtige FORM und die voellig
    falsche HOEHE haben und trotzdem gruen sein. Die Weissmeer-Flachkueste
    stand bei 187 m, wo die Vorlage 15-30 m sagt, mit RMS 0.320 - der Test
    sah nur die Form.

    Seit dem Umbau auf `MESS_PROFIL_M_JE_ARCHETYP` gibt es die Vorlage in
    METERN, und damit laesst sich direkt das pruefen, worauf es ankommt:
    steht die Kueste nach 150 m auf der richtigen Hoehe. Keine
    Normierung, kein Streckenmass, keine getrennte Hoehenpruefung mehr -
    eine Zahl in Metern.
    """
    ergebnisse = []
    land = strecke >= 0
    stellen = np.array(PROFIL_STELLEN_M)
    for (region, typ), profs in sorted(gruppen.items()):
        if len(profs) < 8:
            continue
        # GEGEN DIE ANGEWANDTE VORLAGE PRUEFEN, NICHT GEGEN DIE ROHE.
        #
        # Bis 2026-08-26 stand hier `MESS_PROFIL_M_JE_ARCHETYP` - die an
        # echten DEMs gemessene Tabelle. Seit dem Profilmassstab desselben
        # Tages ist das NICHT MEHR die Vorlage, die das Gelaende bekommt:
        # `GEMESSENES_PROFIL_M` skaliert die steilen Archetypen in beiden
        # Achsen auf den Weltmassstab herunter (siehe PROFILMASSSTAB in
        # core/vektor_kueste.py), und genau diese Tabelle traegt jedes
        # Segment als `profil_m`.
        #
        # Gegen die rohe Tabelle zu messen hiesse, dem Gelaende einen
        # Fehler vorzuwerfen, den es absichtlich macht. Der Test bestand
        # danach zwar weiter (Median 1.8 -> 3.9 m), aber er prueft dann
        # etwas anderes als das, was gebaut wurde - genau die Art stiller
        # Fehlmessung, die dieses Projekt schon mehrfach getroffen hat.
        soll = GEMESSENES_PROFIL_M.get(typ)
        if soll is None:
            continue
        P = np.array(profs)
        s_ = strecke[land]
        # Hoehe ueber der Uferhoehe, Median ueber alle Schnitte der Gruppe.
        ist = np.nanmedian(P[:, land] - P[:, land][:, :1], axis=0)

        # NUR BIS p1 INTEGRIEREN, UND ZWAR BIS SEINEM EIGENEN.
        #
        # Hier stand `xs = np.linspace(0.0, PROFIL_VOLL_M, 15)` mit
        # PROFIL_VOLL_M = 350 m. Das war einmal richtig, ist es seit dem
        # 2026-08-25 aber nicht mehr: `PROFIL_VOLL_M` ist in der Produktion
        # nur noch der RUECKFALLWERT fuer Segmente ohne gemessenes Profil.
        # Jedes echte Segment bekommt `voll_m = _zone_p1(name)`, und das
        # sind seit der Nutzervorgabe *"ich wuerde p1 noch naeher an die
        # kueste ziehen (30% naeher ran)"* nur noch 210 m (flach) bis
        # 245 m (steil). Zwischen p1 und p2 blendet die Kueste absichtlich
        # ins Rauschgelaende ueber.
        #
        # Der Test mass also 140 m Ueberblendzone mit, als waere dort noch
        # das Profil zustaendig - und bestrafte damit genau die Wirkung,
        # die der Nutzer angefordert hatte. GEMESSEN auf dem Festland
        # (384 px, Seed 20260804, Medianhoehe im Ring je Archetyp):
        #
        #     Archetyp                soll/ist 150 m    soll/ist 350 m
        #     Toskana-Straende             4 /  4          6 /  29
        #     Weissmeer-Flachkueste        5 /  5          7 /  31
        #     Luce-Bay-Straende            6 / 10         10 /  30
        #     Foerdenkueste                2 /  2          3 /  17
        #     Ostsee-Flachkueste          10 / 11         17 /  52
        #
        # Bei 150 m trifft JEDER flache Archetyp seine Vorlage. Bei 350 m
        # liegt JEDER um das Drei- bis Fuenffache darueber - weil das
        # Hinterland dort zu 41 % durchschlaegt ((350-210)/(550-210)).
        # Das ist kein Fehler der Kueste, das ist die Ueberblendung.
        xs = np.linspace(0.0, _zone_p1(typ) if typ in GEMESSENES_PROFIL_M
                         else PROFIL_VOLL_M, 15)
        ist_x = np.interp(xs, s_, ist)
        soll_x = np.interp(xs, stellen, np.array(soll))
        fehler_m = float(np.sqrt(np.mean((ist_x - soll_x) ** 2)))

        ergebnisse.append({
            "region": region, "typ": typ, "n": len(profs),
            "fehler_m": fehler_m,
            "ist150": float(np.interp(150.0, s_, ist)),
            "soll150": float(np.interp(150.0, stellen, np.array(soll))),
            "ist350": float(np.interp(350.0, s_, ist)),
            "soll350": float(np.interp(350.0, stellen, np.array(soll))),
        })
    return ergebnisse


def run_profiltreue():
    """Trifft die erzeugte Kueste ihr Vorlagenprofil - in Metern."""
    gruppen, strecke, fehler = _gruppen_messen()
    if fehler:
        return check("Kueste vermessbar", False, fehler)
    ergebnisse = _auswerten(gruppen, strecke)
    ok = check("genug Gruppen zum Vergleichen", len(ergebnisse) >= 10,
               f"{len(ergebnisse)} (Region, Archetyp)-Gruppen")
    if not ok:
        return False

    print(f"\n       {'Region':<19}{'Archetyp':<23}{'n':>4}"
          f"{'  h(150) Ist/Soll':>19}{'  h(350) Ist/Soll':>19}{'Fehler':>9}")
    for e in sorted(ergebnisse, key=lambda x: -x["fehler_m"]):
        print(f"       {e['region']:<19}{e['typ']:<23}{e['n']:>4}"
              f"{e['ist150']:>10.0f} /{e['soll150']:>6.0f} m"
              f"{e['ist350']:>10.0f} /{e['soll350']:>6.0f} m"
              f"{e['fehler_m']:>8.0f} m")
    print()

    fehler_werte = [e["fehler_m"] for e in ergebnisse]
    ok &= check("Profiltreue im Median",
                float(np.median(fehler_werte)) < FEHLER_MEDIAN_MAX_M,
                f"{float(np.median(fehler_werte)):.1f} m Abweichung "
                f"(Grenze {FEHLER_MEDIAN_MAX_M} m)")

    # FLACHE KUESTEN SIND DIE EIGENTLICHE ZUSICHERUNG.
    #
    # Sie waren der Anlass des ganzen Umbaus (Nutzervorgabe 2026-08-24:
    # *"flache kueste bleibt flach"*). Vor dem Umbau stand die
    # Weissmeer-Flachkueste bei 187 m, wo die Vorlage 15-30 m sagt - das
    # Rauschgelaende hatte sie zur Klippe hochgezogen. Eine flache Kueste
    # darf deshalb nur wenige Meter danebenliegen, waehrend bei den
    # steilen mehr Spielraum bleibt: eine 466-m-Fjordwand braucht eine
    # Landmasse, die sie traegt, und in einer 21-km-Welt mit schmalen
    # Fjorden gibt es die selten.
    flach = [e for e in ergebnisse if e["soll150"] <= FLACH_GRENZE_M]
    if flach:
        schlecht = [e for e in flach if e["fehler_m"] > FLACH_FEHLER_MAX_M
                    and (e["region"], e["typ"]) not in BEKANNTE_ABWEICHUNGEN]
        bekannt = [e for e in flach if e["fehler_m"] > FLACH_FEHLER_MAX_M
                   and (e["region"], e["typ"]) in BEKANNTE_ABWEICHUNGEN]
        gut = [e for e in flach if e["fehler_m"] <= FLACH_FEHLER_MAX_M]
        ok &= check("flache Kuesten bleiben flach", not schlecht,
                    ", ".join(f"{e['typ']} {e['fehler_m']:.0f} m"
                              for e in schlecht) if schlecht
                    else f"{len(gut)} von {len(flach)} eng getroffen, "
                         f"schlechtester {max(e['fehler_m'] for e in gut):.0f} m"
                         + (f"; {len(bekannt)} bekannt" if bekannt else ""))
        # Eine BEHOBENE Abweichung soll ebenfalls auffallen - sonst bleibt
        # sie ewig in der Liste stehen.
        behoben = BEKANNTE_ABWEICHUNGEN - {(e["region"], e["typ"])
                                           for e in flach
                                           if e["fehler_m"] > FLACH_FEHLER_MAX_M}
        if behoben:
            print(f"[HINWEIS] behoben, bitte aus BEKANNTE_ABWEICHUNGEN "
                  f"streichen: {sorted(behoben)}")
    return ok


# Wieviel duerfen die SPAETEREN Pipelinestufen das Kuestenband noch
# verschieben, bevor die Messung auf `weltfeld()` ihre Aussagekraft verliert.
#
# Gemessen am 2026-08-25 (384 px, Seed 20260804): Erosionsfilter p90 8.2 m,
# Fluesse/Taeler p90 11.9 m, zusammen p90 17.5 m; die Wasserlinie verschob
# sich um 0.04 % der Pixel. Die Grenzen liegen mit Reserve darueber.
SPAETERE_STUFEN_MAX_P90_M = 40.0
WASSERLINIE_MAX_ANTEIL = 0.005          # 0.5 % der Karte


def run_spaetere_stufen_aendern_die_kueste_kaum():
    """
    DIE ANNAHME, AUF DER DIESER GANZE TEST STEHT.

    Alle Gruppen oben messen auf `weltfeld()`. Das ist Stufe 1 von vier - im
    Terrain-Generator folgen danach noch der Erosionsfilter
    (`_weltkarte_erosionsfilter`), das Flussnetz mit den Taelern
    (`taeler_eingraben`) und die Redistribution. Was der Nutzer auf der Karte
    sieht, hat alle vier hinter sich.

    Nutzervorgabe 2026-08-25: *"profile pruefen (der Einfluss durch zb
    erosion, erosionfilter, Fluesse, anderer kuesten etc muss natuerlich
    erkannt werden, weil das die kueste auch betrifft)"*.

    Bis dahin war "die spaeteren Stufen lassen die Kueste in Ruhe" eine
    ANNAHME - nirgends geprueft. Diese Gruppe macht eine Messung daraus:
    schlagen die Grenzen an, misst der Rest des Tests ein Gelaende, das so
    nie angezeigt wird, und die Aussagen oben sind wertlos.
    """
    from scipy import ndimage

    from core.terrain_generator import BaseTerrainGenerator
    from core.terrain_weltfluesse import flussnetz, taeler_eingraben

    H1, felder = weltfeld(SIZE, SEED)
    H1 = np.asarray(H1, dtype=np.float64)
    mpp = WELT_KM * 1000.0 / SIZE

    # Erosionsfilter - ohne den Generator vollstaendig aufzubauen. Scheitert
    # der Aufruf, wird das GEMELDET statt uebersprungen: eine stillschweigend
    # ausgelassene Stufe waere genau der blinde Fleck, den diese Gruppe
    # schliessen soll.
    gen = BaseTerrainGenerator.__new__(BaseTerrainGenerator)
    gen.shader_manager = None
    gen.data_lod_manager = None
    gen._current_parameters = {}
    import logging
    gen.logger = logging.getLogger("kuestenprofiltreue")
    try:
        gefiltert = gen._weltkarte_erosionsfilter(H1.copy().astype(np.float32), felder)
        H2 = np.asarray(gefiltert["heightmap"], dtype=np.float64) if gefiltert else H1
        filter_lief = gefiltert is not None
    except Exception as fehler:                          # noqa: BLE001
        return check("Erosionsfilter laesst sich messen", False, str(fehler)[:90])

    netz = flussnetz(H2, SEED)
    H3 = (np.asarray(taeler_eingraben(H2.copy(), felder=felder, netz=netz,
                                      abstand_makro_m=1200.0), dtype=np.float64)
          if netz is not None else H2)

    ok = check("Erosionsfilter lief tatsaechlich", filter_lief,
               "sonst misst die Gruppe eine Stufe, die gar nicht stattfand")
    ok &= check("Flussnetz lief tatsaechlich", netz is not None)

    land = H1 > 0
    zur_kueste = np.where(land, ndimage.distance_transform_edt(land),
                          ndimage.distance_transform_edt(~land)) * mpp
    im_band = zur_kueste < 350.0        # die Zone, in der das Profil gilt

    for name, Ha, Hb in (("Erosionsfilter", H1, H2),
                         ("Fluesse und Taeler", H2, H3),
                         ("beide zusammen", H1, H3)):
        delta = np.abs(Hb - Ha)[im_band]
        p90 = float(np.percentile(delta, 90))
        verschoben = float(((Ha > 0) != (Hb > 0)).mean())
        ok &= check(f"{name}: Kuestenband bleibt weitgehend unberuehrt",
                    p90 < SPAETERE_STUFEN_MAX_P90_M,
                    f"p90 {p90:.1f} m (Grenze {SPAETERE_STUFEN_MAX_P90_M:.0f})")
        ok &= check(f"{name}: Wasserlinie bleibt liegen",
                    verschoben < WASSERLINIE_MAX_ANTEIL,
                    f"{100.0 * verschoben:.3f} % der Pixel wechseln Land/Wasser")
    return ok


def main():
    print("=" * 78)
    print("Sieht die erzeugte Kueste aus wie die gemessenen Profile?")
    print("=" * 78)
    ergebnisse = []
    for name, funktion in [("Tabellen vollstaendig", run_tabellen_sind_vollstaendig),
                           ("spaetere Stufen aendern die Kueste kaum",
                            run_spaetere_stufen_aendern_die_kueste_kaum),
                           ("Profiltreue in Metern", run_profiltreue)]:
        print(f"\n--- {name} ---")
        ergebnisse.append(funktion())
    print("\n" + "=" * 78)
    fehlend = ergebnisse.count(False)
    print(f"{len(ergebnisse) - fehlend}/{len(ergebnisse)} Gruppen gruen")
    return 0 if fehlend == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
