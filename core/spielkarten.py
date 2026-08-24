"""
Path: core/spielkarten.py

Zerlegt die Weltkarte in N zusammenhaengende "Spielkarten" (Nutzer-Vorgabe
2026-08-13, docs/OFFENE_PUNKTE.md 5.15): neun Vielecke mit jeweils etwa
gleich viel Landmasse, kompakt genug fuer eine quadratische Darstellung,
mit Schnittlinien zwischen den Staedten statt mitten hindurch.

WARUM EIN POWER-DIAGRAMM (gewichtetes Voronoi) UND NICHT EIN KOSTENFELD

Die Vorgabe enthaelt zwei Forderungen, die sich widersprechen koennen:

  (a) "optimiert um auf einem quadrat gezeigt werden zu koennen, also nicht
      unnoetig lang"
  (b) "dennoch logische zusammenhaenge enthalten ... die ganze halbinsel auf
      einer karte"

Ein kostenbasiertes Verfahren (guenstigste Wege durch ein Gelaendekostenfeld,
wie beim Wegenetz) erfuellt (b) sehr gut - die Grenzen folgen Meeresarmen und
Graten -, kann aber beliebig krumme, langgezogene Gebiete erzeugen und
verletzt damit (a). Ein Power-Diagramm hat dagegen die Eigenschaft, die (a)
GARANTIERT: seine Zellen sind **konvex**, begrenzt von geraden Kanten. Damit
ist jede Karte automatisch kompakt, ihr Umriss ein Vieleck mit wenigen Ecken,
und ihre Bounding-Box nicht unnoetig langgezogen.

(b) wird stattdessen ueber die LAGE DER SAATPUNKTE erreicht statt ueber die
Form der Zellen: liegen die Saatpunkte auf den Siedlungsschwerpunkten, dann
liegt jede Voronoi-Grenze per Definition mittig zwischen zwei benachbarten
Siedlungsclustern - genau die Forderung "die schnittlinien sollten irgendwo
zwischen den staedten verlaufen". Eine Halbinsel mit eigenen Siedlungen
bekommt dadurch ihren eigenen Saatpunkt und damit ihre eigene Karte.

DER KAPAZITAETSAUSGLEICH LAEUFT UEBER lambda, NICHT UEBER DIE SAATPUNKTE

"jeweils gleich viel Landmasse" liesse sich auch durch Verschieben der
Saatpunkte erreichen (Lloyd-Relaxation). Das waere hier aber falsch: es wuerde
die Saatpunkte von den Staedten wegziehen und damit die Forderung
"Schnittlinien zwischen den Staedten" wieder aufgeben. Stattdessen bekommt
jeder Saatpunkt ein additives Gewicht `lambda_i`, und zugeordnet wird nach

    argmin_i ( ||p - s_i||^2 - lambda_i )

Ein groesseres lambda vergroessert die Zelle, ohne den Saatpunkt zu bewegen.
Die Zellen bleiben dabei konvex (das ist die definierende Eigenschaft des
Power-Diagramms). lambda wird iterativ angepasst, bis alle Zellen etwa
dieselbe Landmasse tragen.

LANDMASSE ZAEHLT DAS KUESTENMEER MIT

Nutzer-Vorgabe: "der erste seegrad an kueste zaehlt auch als 50% landmasse, da
hier viel passiert". Das Gewichtsfeld ist deshalb 1.0 auf Land, 0.5 auf
Seezellen vom Grad 1 (die unmittelbar kuestennahe See, siehe
terrain_weltkarte.seegliederung()), 0.0 auf offener See. Zwei Karten mit
gleicher Gewichtssumme haben damit gleich viel *bespielbaren* Raum, nicht
gleich viel Flaeche.
"""

import numpy as np

# Zielspanne der Landmasse zwischen der kleinsten und der groessten Karte.
# Nutzer-Vorgabe: "die karten koennen bis zu 50% unterschiedlich gross sein um
# den zweck zu erfuellen kohaerent zu sein" - also max/min <= 1.5.
SPANNE_ZIEL = 1.5

# Gewicht der kuestennahen See (Seegrad 1) in der Landmassen-Rechnung.
KUESTENSEE_GEWICHT = 0.5

_MAX_RUNDEN = 200
_LERNRATE = 0.35


def gewichtsfeld(heightmap, seegrad=None):
    """
    Funktionsweise: (H,W) float32 - 1.0 auf Land, KUESTENSEE_GEWICHT auf
    Seezellen vom Grad 1, 0.0 sonst.
    Aufgabe: Das Mass, nach dem die Karten ausgeglichen werden ("Landmasse" im
    Sinne der Nutzer-Vorgabe, nicht reine Flaeche). Ohne `seegrad` zaehlt nur
    das Land - dann verhaelt sich alles wie ohne die Kuestensee-Regel.
    """
    hoehe = np.asarray(heightmap)
    w = (hoehe > 0.0).astype(np.float32)
    if seegrad is not None:
        grad = np.asarray(seegrad)
        if grad.shape == w.shape:
            w[(hoehe <= 0.0) & (grad == 1)] = KUESTENSEE_GEWICHT
    return w


def _kmeans_gewichtet(punkte, gewichte, k, seed, runden=60):
    """Schlichtes gewichtetes k-Means (k-Means++-Start). Eigenbau statt scipy/
    sklearn: scipy.cluster.vq kennt keine Punktgewichte, und sklearn ist keine
    Abhaengigkeit dieses Projekts."""
    rng = np.random.RandomState(seed)
    n = len(punkte)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if n <= k:
        return punkte.astype(np.float64).copy()

    # k-Means++: erster Punkt gewichtet gezogen, danach jeweils der Punkt mit
    # grossem Abstand zum naechsten bereits gewaehlten Zentrum.
    wahrsch = gewichte / gewichte.sum()
    zentren = [punkte[rng.choice(n, p=wahrsch)]]
    for _ in range(1, k):
        d2 = np.min([((punkte - z) ** 2).sum(axis=1) for z in zentren], axis=0)
        p = d2 * gewichte
        summe = p.sum()
        zentren.append(punkte[rng.choice(n, p=p / summe) if summe > 0
                              else rng.randint(n)])
    zentren = np.array(zentren, dtype=np.float64)

    for _ in range(runden):
        d2 = ((punkte[:, None, :] - zentren[None, :, :]) ** 2).sum(axis=2)
        zugehoerig = np.argmin(d2, axis=1)
        neue = zentren.copy()
        for i in range(k):
            treffer = zugehoerig == i
            if not treffer.any():
                continue
            g = gewichte[treffer][:, None]
            neue[i] = (punkte[treffer] * g).sum(axis=0) / g.sum()
        if np.allclose(neue, zentren, atol=1e-6):
            zentren = neue
            break
        zentren = neue
    return zentren


def saatpunkte(gewicht, anzahl, seed, siedlungen=None):
    """
    Funktionsweise: `anzahl` Saatpunkte als (anzahl, 2)-Array in (x, y).
    Aufgabe: Bestimmt, WO die Karten ihre Mittelpunkte haben - und damit
    mittelbar, wo die Grenzen verlaufen (mittig zwischen benachbarten
    Saatpunkten).

    Mit `siedlungen` werden die Saatpunkte auf Siedlungscluster gelegt
    (gewichtet nach `house_count`, ersatzweise gleich gewichtet) - dadurch
    verlaufen die Grenzen zwischen den Staedten, wie vom Nutzer gefordert.
    Ohne Siedlungen (oder bei zu wenigen) fallen wir auf ein k-Means ueber das
    Gewichtsfeld selbst zurueck; die Karten sind dann gleichmaessig ueber die
    Landmasse verteilt, nur eben ohne Bezug zu den Staedten. **Dieser
    Rueckfall meldet sich ueber den Rueckgabewert `aus_siedlungen`** - ein
    stiller Ersatzpfad waere von Erfolg nicht zu unterscheiden (CLAUDE.md).
    """
    orte = []
    massen = []
    for s in (siedlungen or []):
        x = getattr(s, "x", None)
        y = getattr(s, "y", None)
        if x is None or y is None:
            continue
        orte.append((float(x), float(y)))
        massen.append(float(getattr(s, "house_count", 0) or 1))

    if len(orte) >= anzahl:
        punkte = np.array(orte, dtype=np.float64)
        g = np.array(massen, dtype=np.float64)
        g = np.maximum(g, 1.0)
        return _kmeans_gewichtet(punkte, g, anzahl, seed), True

    # Rueckfall: k-Means ueber das Gewichtsfeld - aber NUR ueber die
    # Vollgewicht-Pixel, also echtes Land.
    #
    # WARUM NUR LAND (gemessen 2026-08-13): laeuft der k-Means ueber das
    # gesamte Gewichtsfeld einschliesslich der halb zaehlenden Kuestensee,
    # landet mindestens ein Saatpunkt draussen zwischen den Inseln. Seine
    # Karte erfuellt die Massenbilanz dann formal (Spanne 1.35), besteht aber
    # zu 43 % aus Kuestensee - gemessen 9.0 km2 Land gegen 18.8 km2 der
    # groessten Karte, also Faktor 2.1 beim ECHTEN Land, obwohl die Masse
    # ausgeglichen aussah. Die Kuestensee ist auf dieser Welt fast halb so
    # gross wie das Land (61.9 gegen 132.1 km2), sie kann eine Karte also
    # muehelos "auffuellen". Saatpunkte gehoeren dorthin, wo gespielt wird.
    ys, xs = np.nonzero(gewicht >= 1.0)
    if len(xs) == 0:
        ys, xs = np.nonzero(gewicht > 0)
    if len(xs) == 0:
        raise ValueError("spielkarten: kein Land im Gewichtsfeld")
    schritt = max(1, len(xs) // 20000)
    punkte = np.stack([xs[::schritt], ys[::schritt]], axis=1).astype(np.float64)
    g = gewicht[ys[::schritt], xs[::schritt]].astype(np.float64)
    return _kmeans_gewichtet(punkte, g, anzahl, seed), False


def _zuordnen(gitter_x, gitter_y, saat, lam):
    """Power-Diagramm-Zuordnung: argmin_i (||p-s_i||^2 - lambda_i).
    Ergebnis (H,W) int16 mit dem Index der zustaendigen Karte."""
    bestes = None
    beste_karte = None
    for i, (sx, sy) in enumerate(saat):
        d2 = (gitter_x - sx) ** 2 + (gitter_y - sy) ** 2 - lam[i]
        if bestes is None:
            bestes = d2
            beste_karte = np.zeros(d2.shape, dtype=np.int16)
        else:
            besser = d2 < bestes
            bestes = np.where(besser, d2, bestes)
            beste_karte = np.where(besser, np.int16(i), beste_karte)
    return beste_karte


def zerlegen(heightmap, seegrad=None, siedlungen=None, anzahl=9, seed=0,
             max_runden=_MAX_RUNDEN):
    """
    Funktionsweise: Zerlegt die Karte in `anzahl` konvexe Spielkarten mit
    moeglichst gleicher Landmasse (siehe Modul-Docstring).
    Aufgabe: Rueckgabe-dict mit
        "karte"        (H,W) int16 - Index der zustaendigen Spielkarte je Pixel
        "saat"         (anzahl,2)  - Saatpunkte in (x,y)
        "lambda"       (anzahl,)   - die additiven Gewichte
        "masse"        (anzahl,)   - Landmasse je Karte (Summe des Gewichtsfelds)
        "spanne"       float       - groesste/kleinste Masse
        "runden"       int         - benoetigte Iterationen
        "aus_siedlungen" bool      - ob die Saatpunkte aus Siedlungen kamen
    Die Zuordnung deckt die GANZE Karte ab (auch offene See) - fuer die
    Kartengrenzen zaehlt aber nur, was Gewicht traegt.
    """
    hoehe = np.asarray(heightmap)
    gew = gewichtsfeld(hoehe, seegrad)
    if gew.sum() <= 0:
        raise ValueError("spielkarten: Gewichtsfeld ist ueberall 0")

    saat, aus_siedlungen = saatpunkte(gew, anzahl, seed, siedlungen)
    hoehe_px, breite_px = hoehe.shape
    gy, gx = np.mgrid[0:hoehe_px, 0:breite_px]
    gx = gx.astype(np.float64)
    gy = gy.astype(np.float64)

    ziel = gew.sum() / anzahl
    lam = np.zeros(anzahl, dtype=np.float64)
    # Schrittweite in der Groessenordnung der quadrierten Abstaende, damit die
    # lambda-Aenderung ueberhaupt eine sichtbare Verschiebung bewirkt.
    skala = (hoehe_px * breite_px) / float(anzahl)
    nur_land = gew >= 1.0

    ziel_land = float(nur_land.sum()) / anzahl

    def messen(karte_):
        """Masse und reines Land je Karte."""
        m = np.array([gew[karte_ == i].sum() for i in range(anzahl)])
        l = np.array([(nur_land & (karte_ == i)).sum() for i in range(anzahl)],
                     dtype=np.float64)
        return m, l

    def guete(m, l):
        """Bewertet einen Zwischenstand am SCHLECHTEREN der beiden Spannen:
        Masse (das vorgegebene Ziel) und reines Land. Eine Loesung, die nur die
        Masse ausgleicht, kann beim echten Land um Faktor 2-3 danebenliegen
        (gemessen: Seed 777001 hatte Masse-Spanne 1.00 bei Land-Spanne 2.11),
        weil die halb zaehlende Kuestensee eine Karte auffuellen kann."""
        if m.min() <= 0 or l.min() <= 0:
            return float("inf")
        return max(m.max() / m.min(), l.max() / l.min())

    bestes_karte = None
    bestes_lam = lam.copy()
    bestes_g = float("inf")
    bestes_masse = None
    runden = 0

    for runde in range(1, max_runden + 1):
        runden = runde
        karte = _zuordnen(gx, gy, saat, lam)
        masse, land = messen(karte)
        g = guete(masse, land)

        if g < bestes_g:
            bestes_g = g
            bestes_karte = karte
            bestes_lam = lam.copy()
            bestes_masse = masse
        if g <= SPANNE_ZIEL:
            break

        # DAEMPFUNG UND ZENTRIERUNG (2026-08-13). Die erste Fassung nahm eine
        # feste Lernrate und behielt das LETZTE Ergebnis - bei Seed 12345
        # schaukelte sich das auf eine Spanne von 19.1 auf, also weit
        # schlechter als der Startzustand, und genau dieser Ausreisser wurde
        # dann zurueckgegeben. Drei Aenderungen: (1) die Lernrate faellt mit
        # der Rundenzahl, (2) lambda wird zentriert - nur die DIFFERENZEN
        # zwischen den lambdas bestimmen die Zellgrenzen, ein gemeinsamer
        # Offset laesst die Werte nur davondriften, (3) zurueckgegeben wird
        # das BESTE gesehene Ergebnis, nie ein schlechteres spaeteres.
        #
        # (4) Der Fehler mittelt MASSE UND LAND. Zielt die Anpassung nur auf
        # die Masse, verschwindet der Gradient, sobald die Masse ausgeglichen
        # ist - auch wenn das Land dann noch um Faktor 2 auseinanderliegt
        # (gemessen bei Seed 777001: Masse 1.00, Land 2.11, und die Iteration
        # hatte keinen Grund mehr, daran etwas zu aendern). Beide Anteile
        # zusammen geben der Iteration auch dann noch eine Richtung.
        rate = _LERNRATE / (1.0 + 0.05 * runde)
        fehler_masse = (ziel - masse) / max(ziel, 1e-9)
        fehler_land = (ziel_land - land) / max(ziel_land, 1e-9)
        fehler = 0.5 * (fehler_masse + fehler_land)
        lam = lam + rate * skala * np.clip(fehler, -1.0, 1.0)
        lam -= lam.mean()

    karte = bestes_karte if bestes_karte is not None else _zuordnen(gx, gy, saat, lam)
    masse = (bestes_masse if bestes_masse is not None
             else np.array([gew[karte == i].sum() for i in range(anzahl)]))
    spanne = float(masse.max() / masse.min()) if masse.min() > 0 else float("inf")
    return {
        "karte": karte,
        "saat": saat,
        "lambda": bestes_lam,
        "masse": masse,
        "spanne": spanne,
        "guete": bestes_g,
        "runden": runden,
        "aus_siedlungen": aus_siedlungen,
        "gewicht": gew,
    }


def geografische_reihenfolge(karte, gewicht=None, spalten=3):
    """
    Ordnet die Spielkarten so, wie sie auf der Karte LIEGEN - fuer ein
    Auswahlraster, dessen Felder der Geografie entsprechen (Nutzer-Vorgabe
    2026-08-13: "die nordwestlichste karte vom schwerpunkt sollte dann oben
    links liegen, etc. so das es logisch ist").

    Ohne das ist die Reihenfolge die des k-Means und damit willkuerlich: der
    Knopf oben links kann eine Karte im Suedosten waehlen.

    Verfahren: Schwerpunkt je Karte (nur ueber die gewichteten Pixel, also
    ueber Land - der Schwerpunkt inklusive Meer laege bei einer Kuestenkarte
    weit draussen), dann zeilenweise sortieren: zuerst nach Nord-Sued in
    `spalten` grosse Gruppen, innerhalb jeder Gruppe nach West-Ost.

    ACHTUNG BEI DER Y-ACHSE: die Karten werden mit `origin='lower'`
    gezeichnet, y=0 ist also UNTEN. "Nord" heisst damit GROSSES y - deshalb
    wird absteigend sortiert. Aufsteigend haette das Raster senkrecht
    gespiegelt, und zwar unauffaellig genug, um es fuer richtig zu halten.

    Rueckgabe: Liste von Kartenindizes in Rasterreihenfolge (erst Zeile 0
    von links nach rechts, dann Zeile 1, ...). Karten ohne Pixel entfallen
    und werden hinten angehaengt, damit die Liste vollstaendig bleibt.
    """
    karte = np.asarray(karte)
    anzahl = int(karte.max()) + 1 if karte.size else 0
    if anzahl <= 0:
        return []

    if gewicht is None:
        maske_basis = np.ones(karte.shape, dtype=bool)
    else:
        maske_basis = np.asarray(gewicht) > 0

    schwerpunkte = {}
    leer = []
    for i in range(anzahl):
        treffer = (karte == i) & maske_basis
        if not treffer.any():
            leer.append(i)
            continue
        ys, xs = np.nonzero(treffer)
        schwerpunkte[i] = (float(xs.mean()), float(ys.mean()))

    # Nord zuerst -> y ABSTEIGEND (origin='lower', siehe Docstring)
    nach_norden = sorted(schwerpunkte, key=lambda i: -schwerpunkte[i][1])
    reihenfolge = []
    for start in range(0, len(nach_norden), spalten):
        zeile = nach_norden[start:start + spalten]
        # West zuerst -> x aufsteigend
        reihenfolge.extend(sorted(zeile, key=lambda i: schwerpunkte[i][0]))
    return reihenfolge + leer


def kennzahlen(ergebnis, region_map=None, siedlungen=None):
    """
    Funktionsweise: Bewertet eine Zerlegung gegen die Vorgaben des Nutzers.
    Aufgabe: Rueckgabe-dict je Kriterium - damit ueberhaupt beurteilbar ist,
    ob die Zerlegung taugt, statt sich auf den Augenschein zu verlassen.

        "masse_spanne"     groesste/kleinste Landmasse (Ziel <= 1.5)
        "seitenverhaeltnis" je Karte laengere/kuerzere Seite der Bounding-Box
                            der GEWICHTETEN Flaeche (Ziel nahe 1 = quadratisch)
        "fuellgrad"        Anteil der Bounding-Box, der zur Karte gehoert
        "bruchstuecke"     Zahl der zusammenhaengenden Landstuecke je Karte,
                            die mind. 2 % der Kartenmasse ausmachen. Die
                            Schwelle ist noetig, weil eine Inselwelt (z.B. die
                            Griechischen Inseln dieser Karte) von Natur aus
                            aus Dutzenden Stuecken besteht - ohne sie misst
                            die Zahl die Welt, nicht die Qualitaet des
                            Schnitts (erste Fassung meldete bis zu 43).
        "kern_anteil"      Masse des GROESSTEN zusammenhaengenden Stuecks
                            geteilt durch die Gesamtmasse der Karte. Das ist
                            das eigentliche Kohaerenzmass: 1.0 = die Karte ist
                            ein Stueck, 0.5 = sie zerfaellt in zwei Haelften.
        "regionen_je_karte" wie viele Weltregionen je Karte vorkommen (weniger
                            = "regionen bleiben zusammen" besser erfuellt)
        "grenzabstand_min"  kleinster Abstand einer Siedlung zur Kartengrenze
                            in Pixeln (gross = Schnitte laufen nicht durch Orte)
    """
    from scipy import ndimage

    karte = ergebnis["karte"]
    gew = ergebnis["gewicht"]
    anzahl = len(ergebnis["saat"])
    aus = {"masse_spanne": ergebnis["spanne"]}

    # LAND-SPANNE GETRENNT VON DER MASSE-SPANNE (2026-08-13): die Masse zaehlt
    # die Kuestensee zur Haelfte mit (Nutzer-Vorgabe). Dadurch KANN die
    # Massenbilanz ausgeglichen aussehen, waehrend das echte Land ungleich
    # verteilt ist - genau das ist beim ersten Lauf passiert (Masse-Spanne
    # 1.35, Land-Spanne 2.09). Beide Zahlen gehoeren deshalb nebeneinander.
    nur_land = gew >= 1.0
    land_je_karte = np.array([(nur_land & (karte == i)).sum()
                              for i in range(anzahl)], dtype=np.float64)
    aus["land_je_karte"] = land_je_karte
    aus["land_spanne"] = (float(land_je_karte.max() / land_je_karte.min())
                          if land_je_karte.min() > 0 else float("inf"))

    seiten, fuell, stuecke, kern = [], [], [], []
    for i in range(anzahl):
        maske = (karte == i) & (gew > 0)
        if not maske.any():
            seiten.append(float("nan"))
            fuell.append(0.0)
            stuecke.append(0)
            kern.append(0.0)
            continue
        ys, xs = np.nonzero(maske)
        h = ys.max() - ys.min() + 1
        b = xs.max() - xs.min() + 1
        seiten.append(max(h, b) / max(1.0, min(h, b)))
        fuell.append(maske.sum() / float(h * b))

        marken, n = ndimage.label(maske)
        if n == 0:
            stuecke.append(0)
            kern.append(0.0)
            continue
        # Nach MASSE wiegen, nicht nach Pixelzahl - sonst zaehlt ein Streifen
        # Kuestensee so viel wie dieselbe Flaeche Land.
        massen = ndimage.sum(gew, marken, index=np.arange(1, n + 1))
        gesamt = massen.sum()
        kern.append(float(massen.max() / gesamt) if gesamt > 0 else 0.0)
        stuecke.append(int((massen >= 0.02 * gesamt).sum()))
    aus["seitenverhaeltnis"] = np.array(seiten)
    aus["fuellgrad"] = np.array(fuell)
    aus["bruchstuecke"] = np.array(stuecke)
    aus["kern_anteil"] = np.array(kern)

    if region_map is not None:
        rm = np.asarray(region_map)
        zahlen = []
        for i in range(anzahl):
            maske = (karte == i) & (gew > 0)
            zahlen.append(len(np.unique(rm[maske])) if maske.any() else 0)
        aus["regionen_je_karte"] = np.array(zahlen)

    if siedlungen:
        # Grenzpixel: dort wechselt die Kartenzugehoerigkeit.
        grenze = np.zeros(karte.shape, dtype=bool)
        grenze[:, :-1] |= karte[:, :-1] != karte[:, 1:]
        grenze[:-1, :] |= karte[:-1, :] != karte[1:, :]
        if grenze.any():
            abstand = ndimage.distance_transform_edt(~grenze)
            werte = []
            for s in siedlungen:
                x, y = getattr(s, "x", None), getattr(s, "y", None)
                if x is None or y is None:
                    continue
                xi = int(np.clip(round(float(x)), 0, karte.shape[1] - 1))
                yi = int(np.clip(round(float(y)), 0, karte.shape[0] - 1))
                werte.append(abstand[yi, xi])
            if werte:
                aus["grenzabstand_min"] = float(np.min(werte))
                aus["grenzabstand_median"] = float(np.median(werte))
    return aus
