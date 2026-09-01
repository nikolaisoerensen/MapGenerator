"""
Path: tools/archetyp_vorbilder.py

Je ARCHETYP eine eigene Vorbildkueste auf der Erde.

WARUM DAS NOETIG IST (Nutzerbefund 2026-08-24, am Bild "Nebelrode"):
die drei Mittelgebirgs-Archetypen hatten praktisch dieselbe Profilform und
unterschieden sich nur in der Hoehe. Normiert lagen sie bei 0.28/0.26/0.30
nach 50 m und 0.58/0.57/0.57 nach 100 m - eine Kueste in drei Hoehenstufen,
nicht drei Kuestentypen.

Zwei Ursachen:

  1. ALLE DREI kamen aus DERSELBEN Kachel. Fuer das Nebelrode gibt es
     genau ein Vorbild, Ruegen - und Ruegen ist eine Kreidekueste, dort ist
     alles steil. Eine "Ostsee-Flachkueste", die nach 150 m auf 73 m steigt,
     ist keine Flachkueste; das Vorbild dafuer existierte in den Daten nicht.
  2. Die Aufteilung sortierte die Schnitte nach h(150 m) und teilte in
     Drittel. Dann unterscheiden sich die Drittel zwangslaeufig in h(150 m)
     und in sonst nichts - ein Zirkelschluss.

Mit einer eigenen Vorbildkueste je Archetyp entfaellt beides: jeder misst
sein echtes Profil, und die Formunterschiede sind real statt konstruiert.

DIE AUSWAHL folgt den im Archetyp hinterlegten Eigenschaften
(`KUESTEN_ARCHETYPEN` in core/terrain_weltkarte.py): hoehe_faktor,
winkel_grad, strand_anteil. Ein Archetyp mit strand_anteil 0.65 und
30 Grad bekommt eine echte Flachkueste, einer mit 82 Grad eine Steilwand.

DIE KOORDINATEN SIND VORSCHLAEGE, NICHT GEPRUEFTE MESSPUNKTE. Sie stammen
aus geografischer Kenntnis, nicht aus einem Abruf. Beim ersten Lauf meldet
`archetyp_profile_messen.py` je Strecke, ob eine Kuestenlinie gefunden
wurde und wie viele Schnitte brauchbar waren - Ausschnitte mit zu wenigen
Schnitten oder ohne Kuestenlinie muessen nachgebessert werden.

WORAUF BEIM AUSSCHNITT ZU ACHTEN IST:
  * mindestens 2-3 km Kuestenlinie, sonst zu wenige Schnitte,
  * mindestens 900 m Land hinter der Kueste (Profiltiefe PROFIL_TIEFE_M),
  * moeglichst nur EINE zusammenhaengende Kueste im Bild - `kuestenlinie()`
    nimmt die laengste Kontur, und bei zwei getrennten Ufern gewinnt die
    falsche.

Einhaengen in tools/kuestenlaengsschnitt.STRECKEN oder direkt benutzen:
    from tools.archetyp_vorbilder import VORBILDER
"""

# name -> (Region, Titel, sued, nord, west, ost)
VORBILDER = {
    # ---------------------------------------------------- Clonagh (Irland)
    "Moher-Klippen": dict(
        region="Clonagh", titel="Cliffs of Moher, County Clare, Irland",
        sued=52.9600, nord=52.9820, west=-9.4450, ost=-9.4160, vorhanden=True),
    "West-Cork-Buchten": dict(
        region="Clonagh", titel="Sheep's Head, County Cork, Irland",
        sued=51.5300, nord=51.5700, west=-9.7500, ost=-9.6500),
    # ERSETZT INCH BEACH / DINGLE am 2026-08-25.
    #
    # Nutzervorgabe: *"such dir was anderes als Dingle, gibt es etwas
    # flaches mit Huegel im Hinterland in schottland mit ausreichend
    # Kueste? Oder nimm England. irgendwo bei lake district oder so. Wir
    # wollen ueberall genug Profile haben."*
    #
    # Dingle war an der Wasserlinie zwar flach, stieg dahinter aber steil
    # an - 51.9 m nach 900 m, 138 m nach 2 km. Das ist kein Strand mit
    # Huegeln dahinter, das ist ein Strand am Fuss eines Bergs. Ausserdem
    # lagen zwei Konturen im Bild (die zweite 3.7 km lang), und
    # `kuestenlinie()` nimmt die laengste - eine stille Fehlerquelle.
    #
    # Gemessen wurden vier Kandidaten (Median ueber alle Normalenschnitte,
    # relativ zur Uferhoehe):
    #
    #   Ausschnitt                       Kueste  Schnitte  h(150)  h(350)  h(900)  Konturen
    #   Sandhead / Luce Bay, Schottland   8.8 km      86    6.0 m  10.4 m  23.0 m     1
    #   Silecroft / Black Combe, Cumbria  7.9 km      79   22.3 m  26.6 m  19.2 m     1
    #   Drigg / Seascale, Cumbria         9.6 km      97   11.4 m  13.7 m  10.4 m     4
    #   Allonby / Solway, Cumbria         7.7 km      69    4.2 m   6.5 m   8.8 m     1
    #   Inch Beach, Dingle (bisher)       8.4 km      96    2.6 m   6.5 m  51.9 m     2
    #
    # Sandhead ist der einzige mit BEIDEM: die ersten 350 m bleiben unter
    # 11 m (flach), danach steigt es stetig auf 23 m (Huegel im
    # Hinterland). Silecroft ist an der Wasserlinie gar nicht flach und
    # FAELLT bei 900 m wieder - da liegt ein Tal im Bild. Drigg ist flach,
    # hat aber vier Konturen (Aestuar) und steigt hinten kaum. Allonby ist
    # sauber und flach, aber ohne jedes Hinterland.
    "Luce-Bay-Straende": dict(
        region="Clonagh", titel="Sandhead, Luce Bay, Galloway, Schottland",
        sued=54.7700, nord=54.8300, west=-4.9800, ost=-4.9200),

    # ------------------------------------------- Skerrheim (Norwegen/Schweden)
    "Fjordwand": dict(
        region="Skerrheim", titel="Geirangerfjord, Norwegen",
        sued=62.0700, nord=62.1300, west=7.0000, ost=7.1600, vorhanden=True),
    "Schaerenkueste": dict(
        region="Skerrheim", titel="Stockholmer Schaeren, Schweden",
        sued=59.3000, nord=59.4000, west=18.7000, ost=18.9500, vorhanden=True),
    "Fjordbucht": dict(
        region="Skerrheim", titel="Sognefjord bei Balestrand, Norwegen",
        sued=61.1900, nord=61.2300, west=6.5000, ost=6.6000),

    # ------------------------------------------- Morobora (Kola/Weissmeer/Labrador)
    "Kola-Steilkueste": dict(
        # ZWEIMAL NACHGEBESSERT 2026-08-24.
        #   Teriberka West (69.15-69.20 / 35.10-35.25): Steilheit 0.099,
        #     von den Labrador-Buchten (0.102) nicht zu unterscheiden.
        #   Rybachi (69.70-69.75 / 32.50-32.70): die Kachel lag zu 100 %
        #     an Land, Hoehen 37-192 m - das war das Landesinnere der
        #     Halbinsel, gar keine Kueste.
        # Vier Kandidaten wurden dann gemessen, bevor einer eingetragen
        # wurde: Rybachi-Nordkueste 0.050, Berlevaag 0.034, Slettnes
        # 0.043, Teriberka OSTKAP 0.146. Die arktische Kueste ist dort
        # ueberwiegend flach abgeschliffen; das Ostkap ist die steilste
        # Stelle, die sich finden liess.
        region="Morobora", titel="Teriberka, Ostkap, Halbinsel Kola, Russland",
        sued=69.1700, nord=69.2100, west=35.2000, ost=35.4000),
    "Weissmeer-Flachkueste": dict(
        region="Morobora", titel="Onegabucht bei Kem, Weisses Meer, Russland",
        sued=64.9500, nord=65.0000, west=34.5500, ost=34.7000),
    "Labrador-Buchten": dict(
        region="Morobora", titel="Kueste bei Nain, Labrador, Kanada",
        sued=56.5300, nord=56.5800, west=-61.7500, ost=-61.6000),

    # ------------------------------------------------ Estrande (Frankreich)
    "Bretagne-Klippen": dict(
        region="Estrande", titel="Cap Sizun / Pointe du Raz, Bretagne",
        sued=48.0300, nord=48.0700, west=-4.7500, ost=-4.6500),
    "Vendee-Straende": dict(
        region="Estrande", titel="La Tranche-sur-Mer, Vendee, Frankreich",
        sued=46.3500, nord=46.3900, west=-1.5500, ost=-1.4500),
    "Ile-de-Re-Watt": dict(
        region="Estrande", titel="Ile de Re, Nordkueste (Watt), Frankreich",
        sued=46.2000, nord=46.2400, west=-1.4000, ost=-1.3000),

    # ------------------------------------------------- Nevadin (Adria/Alpen)
    "Kotor-Steilfjord": dict(
        region="Nevadin", titel="Bucht von Kotor, Montenegro",
        sued=42.4600, nord=42.5000, west=18.6200, ost=18.7200),
    "Dalmatien-Klippen": dict(
        region="Nevadin", titel="Makarska Riviera unter dem Biokovo, Kroatien",
        sued=43.2700, nord=43.3100, west=17.0000, ost=17.1000),
    "Alpine-Flussmuendung": dict(
        # NACHGEBESSERT 2026-08-24: die Isonzo-Muendung bei Monfalcone
        # (45.72-45.76 / 13.53-13.63) mass ueber 900 m Land exakt 0 m
        # Anstieg - ein reines Schwemmdelta ohne jedes Relief. Fuer eine
        # ALPINE Flussmuendung braucht es ein Tal, das zwischen Bergen ans
        # Meer tritt; die Roja bei Ventimiglia tut genau das.
        region="Nevadin", titel="Roja-Muendung bei Ventimiglia, Ligurien",
        sued=43.7800, nord=43.8100, west=7.5800, ost=7.6500),

    # ------------------------------------------------ Nebelrode (Ostsee)
    "Ruegen-Kreidekueste": dict(
        region="Nebelrode", titel="Koenigsstuhl, Ruegen, Deutschland",
        sued=54.5300, nord=54.6000, west=13.6000, ost=13.7200, vorhanden=True),
    "Ostsee-Flachkueste": dict(
        region="Nebelrode", titel="Darss-Weststrand, Deutschland",
        sued=54.4200, nord=54.4600, west=12.4500, ost=12.5500),
    "Foerdenkueste": dict(
        region="Nebelrode", titel="Flensburger Foerde bei Gluecksburg",
        sued=54.7200, nord=54.7600, west=9.8500, ost=9.9500),

    # ----------------------------------------------------- Samarcia (Iberien)
    "Algarve-Klippen": dict(
        # NACHGEBESSERT 2026-08-24: Ponta da Piedade (37.07-37.10 /
        # -8.68..-8.60) kam auf Steilheit 0.243 und lag damit UNTER der
        # San-Sebastian-BUCHT (0.315) derselben Region - fuer den
        # steilsten Archetyp der Samarcia unbrauchbar. Cabo de Sao Vicente
        # hat die hohen Klippen (rund 75 m senkrecht).
        region="Samarcia", titel="Cabo de Sao Vicente, Algarve, Portugal",
        sued=37.0100, nord=37.0400, west=-8.9900, ost=-8.9400),
    "Costa-Brava-Buchten": dict(
        region="Samarcia", titel="Begur / Cap de Begur, Costa Brava, Spanien",
        sued=41.9400, nord=41.9800, west=3.2000, ost=3.3000),
    "San-Sebastian-Bucht": dict(
        # NACHGEBESSERT 2026-08-24. La Concha (43.31-43.34 / -2.00..-1.92)
        # mass Steilheit 0.315 und lag damit UEBER den Algarve-Klippen
        # (0.298) derselben Region - die Bucht ist von Monte Urgull und
        # Igueldo umschlossen, und die Schnitte treffen die Berge statt
        # den Strand. Als flachster Archetyp der Samarcia unbrauchbar.
        #
        # Vier Kandidaten gemessen: Zarautz 0.291 (auch Berge im Ruecken),
        # Ebro-Delta 0.002 (voellig eben, kein Profil), El Saler 0.024,
        # Donana/Matalascanas 0.106 - ein echter Duenenstrand mit
        # Restrelief. DER NAME des Archetyps bleibt vorerst
        # "San-Sebastian-Bucht", weil er in KUESTEN_ARCHETYPEN und
        # mehreren Tabellen als Schluessel steht; das VORBILD ist jetzt
        # Donana.
        region="Samarcia", titel="Donana / Matalascanas (Duenenstrand), Spanien",
        sued=36.9800, nord=37.0100, west=-6.6000, ost=-6.5200),

    # ---------------------------------------------------- Macchia (Italien)
    "Amalfi-Steilkueste": dict(
        region="Macchia", titel="Positano / Amalfikueste, Italien",
        sued=40.6000, nord=40.6400, west=14.5800, ost=14.6800),
    "Cinque-Terre-Buchten": dict(
        region="Macchia", titel="Vernazza / Monterosso, Cinque Terre, Italien",
        sued=44.1100, nord=44.1500, west=9.6500, ost=9.7500),
    "Toskana-Straende": dict(
        region="Macchia", titel="Marina di Alberese, Maremma, Italien",
        sued=42.6200, nord=42.6600, west=11.0300, ost=11.1300),

    # ------------------------------------------- Thalassia (Aegaeis)
    "Santorini-Kliff": dict(
        region="Thalassia", titel="Caldera-Steilwand, Santorini",
        sued=36.3400, nord=36.4800, west=25.3300, ost=25.4900, vorhanden=True),
    "Kreta-Buchten": dict(
        region="Thalassia", titel="Loutro / Suedkueste Kreta",
        sued=35.2000, nord=35.2400, west=24.0500, ost=24.1500),
    "Kykladen-Strand": dict(
        region="Thalassia", titel="Agios Prokopios, Naxos-Westkueste",
        sued=37.0500, nord=37.0900, west=25.3500, ost=25.4500),
}


def fehlende():
    """Die Archetypen, deren Kachel noch nicht im Zwischenspeicher liegt."""
    return {k: v for k, v in VORBILDER.items() if not v.get("vorhanden")}


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.terrain_weltkarte import alle_regionen, KUESTEN_ARCHETYPEN

    eigenschaften = {}
    for _z, _s, r in alle_regionen():
        for t in KUESTEN_ARCHETYPEN.get(r["name"], []):
            eigenschaften[t["name"]] = t

    print(f"{'Archetyp':<24}{'Hoehe':>6}{'Winkel':>7}{'Strand':>7}  "
          f"{'Vorbild':<44}{'da?':>4}")
    print("-" * 96)
    letzte = None
    for name, v in VORBILDER.items():
        if v["region"] != letzte:
            print(f"\n{v['region']}")
            letzte = v["region"]
        e = eigenschaften.get(name, {})
        print(f"  {name:<22}{e.get('hoehe_faktor', 0):>6.2f}"
              f"{e.get('winkel_grad', 0):>6}gr{e.get('strand_anteil', 0):>7.2f}  "
              f"{v['titel']:<44}{'ja' if v.get('vorhanden') else 'NEU':>4}")
    print(f"\n{len(VORBILDER)} Archetypen, davon {len(fehlende())} neu abzurufen.")
