# 13 — Klima, Wetter und Biome

Was gelten soll für Temperatur, Niederschlag, Jahresgang, Sonnenstand sowie für
Grund- und Superbiome — samt den Klimazielwerten der neun Regionen auf Meereshöhe
und der Prüfliste, welche Biome je Region herauskommen müssen. Nicht hier: die
Seegliederung (Seegrade, Tiefentabelle, Seewege, Seeeis) steht in `12_WASSER.md`,
die Regionskennwerte in `10_REGIONEN.md`. Die Meerestemperatur bleibt hier, weil
sie Teil des Temperaturmodells ist.

## 1. Leitlinie: Festlegung statt Regelkreis

Für Klima und Wetter gilt: keine Simulationskreise mehr, sondern in jedem Kreis
Festlegungen, damit es keinen Drift mehr gibt (Nutzervorgabe 2026-08-07).

| bisher | künftig |
|---|---|
| Atmosphäre über 25–50 Zeitschritte einschwingen lassen | Zielwerte je Region vorgeben, Feld daraus aufbauen |
| Schelftiefe als `-t * (1 - exp(-d/L))` | Tiefe je Seegrad aus einer Tabelle |
| Ergebnis hängt an Schrittzahl und Auflösung | Ergebnis hängt nur an Seed und Reglern |

Der Vorteil ist nicht bloß Geschwindigkeit: ein Regelkreis über 35 Schritte
liefert bei 50 etwas anderes — dieser Drift zwang bisher zum Nachkalibrieren bei
jeder Auflösungsänderung.

*Herkunft: docs/archiv/2026-09-01_KLIMA_UND_SEE.md (Stand 2026-08-07), Abschnitt „0. Die Leitlinie: Festlegung statt Regelkreis".*

## 2. Klimavorgaben je Region

Die Zielwerte sind Klimamittel realer Bezugsorte (rund 1990–2020) und gelten **auf
Meereshöhe**; die Höhenabnahme rechnet mit **0.6 K je 100 m** (`0.006` K/m), die
Bezugsorte sind dafür auf 0 m zurückgerechnet. Meereshöhe ist gewählt, weil sie
nicht mitwandert — eine Regionsmittelhöhe ist ein geeichter Wert und kann sich
ändern.

| Region | Bezugsort (Höhe, Jan/Jul gem.) | **Jan auf 0 m** | **Jul auf 0 m** | Niederschlag | Mittelhöhe | **Jul in der Region** |
|---|---|---:|---:|---:|---:|---:|
| Clonagh (Kelten) | Cork, Irland (10 m, 6.0/15.5) | **6.1** | **15.6** | 1200 mm | 147 m | **14.7** |
| Skerrheim (Wikinger) | Bergen, Norwegen (20 m, 2.0/15.0) | **2.1** | **15.1** | 2250 mm | 113 m | **14.4** |
| Morobora (Slawen) | Wologda, Russland (130 m, −11.5/17.5) | **−10.7** | **18.3** | 600 mm | 294 m | **16.5** |
| Estrande (Franken) | La Rochelle, Frankreich (15 m, 6.5/20.5) | **6.6** | **20.6** | 780 mm | — | — |
| Nevadin (Alemannen) | Chur, Schweiz (590 m, 0.0/18.5) | **3.5** | **22.0** | 850 mm | 800 m | **17.2** |
| Nebelrode (Sachsen) | Bamberg, Deutschland (240 m, 0.5/19.0) | **1.9** | **20.4** | 640 mm | 350 m | **18.3** |
| Samarcia (Andalusier) | Madrid, Spanien (660 m, 6.5/25.5) | **10.5** | **29.5** | 430 mm | 230 m | **28.1** |
| Macchia (Italiener) | Rom, Italien (20 m, 8.0/25.5) | **8.1** | **25.6** | 800 mm | 1 m | **25.6** |
| Thalassia (Byzantiner) | Iraklio, Kreta (40 m, 12.5/26.5) | **12.7** | **26.7** | 480 mm | Küste | **26.7** |

Die Spreizung ist gewollt: 24 K Unterschied im Januar zwischen Morobora und
Thalassia, Faktor 5 im Niederschlag zwischen Skerrheim und Samarcia.
**Festlegung 2026-08-07: die Samarcia bleibt bei 29.5 °C auf Meereshöhe**, also
28.1 °C auf 230 m, und ist damit die heißeste Region der Karte — Nutzer: „lassen,
kann gerne etwas heißer sein."

*Herkunft: docs/archiv/2026-09-16_BIOME_MATRIX.md (Stand 2026-08-07), Abschnitt „1. Klimavorgaben auf MEERESHOEHE"; docs/archiv/2026-09-01_KLIMA_UND_SEE.md (Stand 2026-08-07), Abschnitte „Die Bezugsorte" und „Die Bezugshoehe ist der Knackpunkt".*

## 3. Die Temperatur des Meeres

Alles unterhalb 0 m hat eine Mitteltemperatur nach Nördlichkeit innerhalb der
Karte und ist nicht von der Sonne beeinflusst:

    T_meer(y, monat) = T_sued(monat) + (T_nord(monat) - T_sued(monat)) * y_anteil
                       + stroemung(x, y) * AMPLITUDE_STROEMUNG

| | Süd | Nord |
|---|---|---|
| Juli | 25 °C | 15 °C |
| Januar | 14 °C | 4 °C |

`y_anteil` ist 0 am Südrand und 1 am Nordrand. Das Meer hat eine deutlich
kleinere Jahresspanne als das Land (Wärmeträgheit), rund 11 K statt 18–29 K.
`stroemung` ist eine eigene, sehr grobe Rauschkarte aus demselben Seed
(Wellenlänge ~8 km), Amplitude **±2 K**.

**Kein Sonneneinfluss, keine Schattenkarte, keine Höhenabnahme auf See** — eine
Festlegung, kein Näherungsverfahren; sie löscht den zuvor offenen
5.6-K-Küstensprung. An der Küste wird über wenige hundert Meter überblendet.
`weather.temperature` braucht damit keine Atmosphärensimulation mehr: Land aus
Region und Höhe, See aus Breite und Strömung; Wind und Feuchte bleiben zunächst.

*Herkunft: docs/archiv/2026-09-01_KLIMA_UND_SEE.md (Stand 2026-08-07), Abschnitte „Das Meer" und „Was daraus folgt".*

## 4. Das Temperaturmodell: Raum und Zeit getrennt

Sommer und Winter bleiben, die **jahreszeitlichen Sonnenstände nicht**. Statt
sechs Zwei-Monats-Durchläufen gilt eine Min-/Max-Temperatur mit einer
unregelmäßigen Kurve; das Muster über die Karte wird EINMAL gerechnet, der
Jahresgang ist eine SKALARE Funktion:

    T(x, y, t) = T_mittel(x, y) + spanne(x, y) / 2 * jahresgang(t)

    T_mittel(x,y)   Land:  T_mittel_region - 0.006 * hoehe_m
                    See:   T_sued + (T_nord - T_sued) * y_anteil + stroemung
    spanne(x,y)     die Jahresspanne der Region (See: deutlich kleiner)
    jahresgang(t)   -cos(2*pi*(t - phase)) + unruhe(t)
    unruhe(t)       summe ueber k=2..5 von  a_k * sin(2*pi*k*t + phi_k)

Die Phasen `phi_k` kommen aus dem Kartenseed, die Amplituden fallen mit 1/k ab
und summieren sich auf rund **15 %** der Jahresspanne: ein Jahr mit
Wärmeeinbrüchen und milden Wochen, aber unverändertem Mittel und unveränderten
Extremen — reproduzierbar, kein Zufall zur Laufzeit. Der Jahresgang ist auf jeden
Zeitpunkt auswertbar, nicht nur auf sechs; das fertige Spiel kann komplette Jahre
durchlaufen, ohne dass die Erzeugung teurer wird. Jahresmittel und halbe Spanne
auf Meereshöhe:

| Region | Jan | Jul | **Mittel** | **halbe Spanne** |
|---|---:|---:|---:|---:|
| Clonagh | 6.1 | 15.6 | 10.9 | 4.8 |
| Skerrheim | 2.1 | 15.1 | 8.6 | 6.5 |
| Morobora | -10.7 | 18.3 | 3.8 | **14.5** |
| Estrande | 6.6 | 20.6 | 13.6 | 7.0 |
| Nevadin | 3.5 | 22.0 | 12.8 | 9.3 |
| Nebelrode | 1.9 | 20.4 | 11.2 | 9.3 |
| Samarcia | 10.5 | 29.5 | 20.0 | 9.5 |
| Macchia | 8.1 | 25.6 | 16.9 | 8.8 |
| Thalassia | 12.7 | 26.7 | 19.7 | 7.0 |
| **See Süd** | 14.0 | 25.0 | 19.5 | 5.5 |
| **See Nord** | 4.0 | 15.0 | 9.5 | 5.5 |

Die Morobora hat mit 29 K die größte Jahresspanne der Karte — der Unterschied
zwischen Kontinental- und Seeklima fällt von selbst heraus. **Zwei Regionsregler**
kommen dazu, nach dem Muster von `kuestenform`: `temp_mittel_m0` und
`temp_spanne`; der Niederschlag folgt später genauso. Schneegrenze und Firn
bleiben möglich: im Januar liegt das Nevadin auf 800 m bei 3.5 − 4.8 =
**−1.3 °C**, die Gipfel darunter. Mit einem reinen Frühlingsmodell wäre selbst
der höchste Punkt schneefrei geblieben.

*Herkunft: docs/archiv/2026-09-01_KLIMA_UND_SEE.md (Stand 2026-08-07), Abschnitt „4. Das Temperaturmodell (entschieden 2026-08-07)"; docs/archiv/2026-09-16_BIOME_MATRIX.md (Stand 2026-08-07), Abschnitt „5. Entschieden am 2026-08-07".*

## 5. Was der Sonnenstand noch tut

Er formt das Muster innerhalb eines Tages und über das Gelände: Südhang wärmer
als Nordhang, Talschatten kühler. Das bleibt — es ist der Grund, warum die
Baumgrenze keine gerade Linie wird. Nur die jahreszeitliche Verschiebung fällt
weg: 5 feste Richtungen statt 7 Winkel × 6 Monate, Schattenwurf einmal statt
42-mal gerechnet.

*Herkunft: docs/archiv/2026-09-01_KLIMA_UND_SEE.md (Stand 2026-08-07), Abschnitt „Was der Sonnenstand dann noch tut".*

## 6. Die fünfzehn Grundbiome

Eingeteilt nach **Julitemperatur** und **Jahresniederschlag**. Aus dem alten
Bestand entfallen fünf Biome, die auf einer 21-km-Insel in Europa nichts zu
suchen haben (Ice Cap, Tropical Rainforest, Tropical Seasonal, Savanna,
Badlands); `Grassland`, `Temperate Forest` und `Mediterranean` zerfallen in zwei
bis drei Arten, sie waren zu grob.

| # | Biom | Juli | Niederschlag | Charakter |
|---|---|---|---|---|
| 1 | **Hochmoor** | < 17 °C | > 1000 mm | Torf, staunass, sauer |
| 2 | **Bruchwald** | 14–22 °C | > 800 mm | Erle und Weide auf nassem Grund, Flussauen |
| 3 | **Feuchtwiese / Marsch** | 13–22 °C | > 700 mm | Küstennahe Niederung, Salzwiesen, Deichland |
| 4 | **Grasland / Weide** | 13–22 °C | 500–900 mm | Der grüne Grundton, Vieh statt Pflug |
| 5 | **Heide** | 13–19 °C | 500–800 mm | Sandiger Magerboden, Zwergsträucher, Ginster |
| 6 | **Fjell / Bergheide** | < 12 °C | beliebig | Über der Baumgrenze, noch bewachsen: Zwergbirke, Flechten |
| 7 | **Nadelwald** | 12–18 °C | 450–900 mm | Fichte und Kiefer, der Taigatyp |
| 8 | **Mischwald** | 15–20 °C | 500–900 mm | Birke, Kiefer, Eiche — der Übergang |
| 9 | **Buchenwald** | 16–21 °C | 600–1000 mm | Der mitteleuropäische Hallenwald |
| 10 | **Eichen-Hainbuchenwald** | 18–23 °C | 500–800 mm | Wärmer und trockener als Buche |
| 11 | **Bergwald** | 10–17 °C | > 700 mm | Tanne und Fichte am Hang, oberhalb des Laubwalds |
| 12 | **Macchia** | > 22 °C | 350–700 mm | Hartlaubbusch, immergrün, dornig |
| 13 | **Steineichenwald** | > 21 °C | 550–900 mm | Der mediterrane Waldtyp |
| 14 | **Trockensteppe** | > 22 °C | 250–500 mm | Büscheliges Gras, weite offene Flächen |
| 15 | **Halbwüste** | > 24 °C | < 300 mm | Kahler Boden, Salzkrusten, Dornsträucher |

**Die Temperaturwerte sind Julitemperaturen.** Die Biom-Klassifikation muss
`temp_map_juli` gegenüberstellen, nicht `temp_map` — mit dem Jahresmittel liegt
sie 8–9 K zu kalt, das Macchia wurde zu 47 % Bruchwald statt Steineichenwald.

*Herkunft: docs/archiv/2026-09-16_BIOME_MATRIX.md (Stand 2026-08-07), Abschnitte „Was aus dem Bestand fliegt" und „Die sechzehn Grundbiome".*

## 7. Die neun Superbiome

Sie überschreiben das Grundbiom nach **einer** Regel.

| Superbiom | Regel | Bemerkung |
|---|---|---|
| **Alpin** | oberhalb der Baumgrenze | Die Grenze ist temperaturabhängig, nicht auf fester Höhe: dort, wo die Julitemperatur unter rund 7 °C fällt. Damit ist sie von selbst keine gerade Linie — sie folgt Hangausrichtung, Schattenwurf und Regionsklima. |
| **Firn / Schneegrenze** | Julitemperatur unter 0 °C | Dieselbe Bauform eine Stufe höher |
| **Fels / Blockhalde** | Hang > 40° | Überall, nicht nur im Gebirge |
| **Klippe** | Hang > 30° UND Küstenabstand < 100 m | Passt zu `kuestenform` |
| **Strand** | Höhe < 5 m UND Hang < 3° UND küstennah | Siehe Anmerkung unten |
| **Dünen / Küstensand** | sandiger Küstensaum | Festlegung 2026-08-07: Superbiom, kein Grundbiom — eine Lageregel, kein Klima |
| **Aue** | Flussabstand < Talbreite UND flach | Nutzt `river_generation` |
| **Fluss** | `river_generation` 1/2/3 | Bach, Fluss, Strom |
| **See / Meer** | Höhe <= 0 | — |

**Zur Strandtiefe:** die Breite folgt aus dem Gelände — eine flache Küste
(`kuestenform` 0.45) bekommt breite Strände, eine Klippenküste keine.

*Herkunft: docs/archiv/2026-09-16_BIOME_MATRIX.md (Stand 2026-08-07), Abschnitte „Die acht Superbiome" und „5. Entschieden am 2026-08-07".*

## 8. Welche Biome je Region zu erwarten sind

**Das ist keine Zuweisung, sondern eine Erwartung.** Die Biome entstehen aus
Temperatur und Niederschlag; die Tabelle ist die Prüfliste der Eichung.

| Region | Jul auf 0 m / Niederschlag | erwartete Grundbiome | typische Superbiome |
|---|---|---|---|
| **Clonagh** | 15.6 °C / 1200 mm | Hochmoor · Grasland · Feuchtwiese · Heide · Bruchwald | Klippe, Fels |
| **Skerrheim** | 15.1 °C / 2250 mm | Hochmoor · Nadelwald · Bergwald · Fjell | Alpin, Firn, Klippe, Fels |
| **Morobora** | 18.3 °C / 600 mm | Nadelwald · Mischwald · Hochmoor · Grasland | Strand, Aue |
| **Estrande** | 20.6 °C / 780 mm | Feuchtwiese · Eichen-Hainbuchenwald · Grasland · Bruchwald | Strand, Dünen, Aue |
| **Nevadin** | 22.0 °C / 850 mm | Bergwald · Fjell · Buchenwald (Täler) · Grasland (Almen) | **Alpin, Firn, Fels** |
| **Nebelrode** | 20.4 °C / 640 mm | Buchenwald · Mischwald · Grasland · Bergwald (Kämme) | Strand, Aue |
| **Samarcia** | 29.5 °C / 430 mm | Trockensteppe · Halbwüste · Macchia · Steineichenwald | Fels |
| **Macchia** | 25.6 °C / 800 mm | Steineichenwald · Macchia · Grasland · Bruchwald | Strand, Klippe |
| **Thalassia** | 26.7 °C / 480 mm | Macchia · Trockensteppe · Steineichenwald | Strand, Klippe, Fels |

Jede Region bekommt 4 bis 5 charakteristische Arten, und **keine zwei Regionen
haben dieselbe Kombination** — genau das ist die Vorgabe. **Ein Prüffall:** das
Nevadin ist mit 22.0 °C auf Meereshöhe zu warm für Bergwald; der entsteht erst
durch die Höhenabnahme auf 800 m und ist damit die Probe darauf, dass sie wirkt.

*Herkunft: docs/archiv/2026-09-16_BIOME_MATRIX.md (Stand 2026-08-07), Abschnitt „3. Welche Biome je Region zu erwarten sind".*

## 9. Vereinfachung des Wettersystems

Drei Festlegungen, zusammen Faktor 15 bis 20; alle `*_monthly`- und
`*_layers`-Ausgaben (rund 20 Stück) verschwinden oder schrumpfen auf ein Sechstel.

| Festlegung | vorher | nachher | Ersparnis |
|---|---|---|---|
| Ein Zeitraum statt Jahreszeiten (Jahresgang als Kurve, §4) | 6 Zwei-Monats-Perioden | 1 | **6x** auf der ganzen Wetterkette |
| Sonnenstand nur 5 Richtungen | 7 Winkel × 6 Monate = 42 Schattenrechnungen | 5 | **8.4x** auf den Schattenwurf |
| Nur eine Ebene statt drei | 3 Schichten (Ground/Mid/High) | 1 | rund **3x** auf die Atmosphärenrechnung |

**Die Temperatur wird nicht mehr simuliert, sondern festgelegt** (§2–§4).
`_semi_lagrangian_advect_many` machte 40 % der Temperaturrechnung aus und ließ
ein Feld über 25–50 Zeitschritte einschwingen, wobei die Schrittzahl an der
Auflösung hing — genau der Drift aus §1. Die Simulation bleibt nur für **Wind und
Feuchte**, und dort mit **fester Schrittzahl**.

*Herkunft: docs/archiv/2026-09-16_BIOME_MATRIX.md (Stand 2026-08-07), Abschnitt „4. Wie das Wettersystem einfacher wird".*

## 10. Umsetzungsstand

Umgesetzt: die 15 Grundbiome; die Regionsaffinität anstelle von neun Matrizen,
weich geglättet; Farben und Namen; Niederschlagsskala an die Ticklänge gebunden;
Alpin und Firn als Temperaturregel; Temperatur und Niederschlag direkt auf ihren
Zielwert normiert. Offen: **Fels, Dünen und Aue** als Superbiome; das
Zurückschneiden der Atmosphärensimulation auf Wind und Feuchte.

*Herkunft: docs/archiv/2026-09-16_BIOME_MATRIX.md (Stand 2026-08-07), Kopfvermerk; docs/archiv/2026-09-01_KLIMA_UND_SEE.md (Stand 2026-08-07), Kopfvermerk und Abschnitt „3. Reihenfolge".*

## Grenzfälle

An `12_WASSER.md` abgegeben, alle aus `docs/archiv/2026-09-01_KLIMA_UND_SEE.md`. Nur die Zeile
„Schelftiefe → Tiefe je Seegrad" in §1 bleibt hier, als Beispiel für die
Leitlinie.

| Abschnitt | Zeilen | Inhalt |
|---|---|---|
| Kopfvermerk, Aufzählung der umgesetzten Punkte | 6–10 | Seegliederung über den Voronoi-Zellgraphen, Tiefentabelle je Seegrad, Seewege ab Grad 1, Regionszuordnung der Seezellen mit Skerrheim-/Hügelland-Sondertabellen und Morobora-Seeeis, Küsten-Archetypen |
| „2. Die See als Voronoi-Gliederung" mit „Es ist fast schon da", „Der Seegrad", „Was das ersetzt", „Was es zusaetzlich bringt", „Vorbehalt" | 128–188 | Seegrad per Breitensuche, Tiefentabelle 0/−40/−90/−150/−200 m, `MEERESBODEN_M`, Glättung über Zellgrenzen, Ersatz des Küstenschelfs, Punktzahl der Seezellen |
| „3. Reihenfolge", Schritte 2 und 3 | 281–283 | Seegrad bauen und ansehen, Schelf durch die Gradtabelle ersetzen |

## Offene Fragen

1. **15 oder 16 Grundbiome (Widerspruch).** `docs/archiv/2026-09-16_BIOME_MATRIX.md:84` und :106
   führen 16 Grundbiome samt Dünen, :108 acht Superbiome; :217-219 legt Dünen
   als Superbiom fest, also 15/9. Hier ist der jüngere Stand verwendet (15/9).
2. **Samarcia 29.5 oder 27.0 °C (Widerspruch).** `docs/archiv/2026-09-16_BIOME_MATRIX.md:63-68`
   empfiehlt 27.0 °C, :214-215 entscheidet 29.5. Hier 29.5, gestützt durch
   dasselbe Nutzerzitat in `docs/archiv/2026-09-01_KLIMA_UND_SEE.md:84-86`.
3. **„Nur Frühling" gegen „Sommer und Winter bleiben" (Widerspruch).**
   `docs/archiv/2026-09-16_BIOME_MATRIX.md:168` führt „Nur Frühling" mit Faktor 6, :221-232
   verwirft es (sonst bliebe selbst der höchste Punkt schneefrei) zugunsten des
   Jahresgangs aus `docs/archiv/2026-09-01_KLIMA_UND_SEE.md:191-238`. Hier: Jahresgang.
4. **Sind die Klimazahlen gegengeprüft?** `docs/archiv/2026-09-01_KLIMA_UND_SEE.md:45-47` nennt sie
   aus dem Gedächtnis zusammengetragen, auf ein bis zwei Grad genau, und verlangt
   eine Gegenprüfung (:279, Schritt 1, Nutzer). Ob sie geschah, sagt keine Quelle.
5. **Hügelland als Vergleichsregion.** `docs/archiv/2026-09-01_KLIMA_UND_SEE.md:261` vergleicht die
   Morobora mit „dem Hügelland"; eine Region dieses Namens gibt es unter den
   neun nicht. Hier durch „die größte Jahresspanne der Karte" ersetzt.
6. **Affinitäten, Farben und Namen fehlen als Tabelle.**
   `docs/archiv/2026-09-16_BIOME_MATRIX.md:7-9` nennt sie als umgesetzt, ohne die Werte zu führen.
7. **Küstenüberblendung ohne Zahl.** `docs/archiv/2026-09-01_KLIMA_UND_SEE.md:123` sagt „über
   wenige hundert Meter", nennt aber keine Breite.
8. **Mittelhöhe der Estrande fehlt.** `docs/archiv/2026-09-16_BIOME_MATRIX.md:47-56` lässt sie als
   einzige Region aus, damit auch ihren Julitemperaturwert auf Regionshöhe.
9. **`AMPLITUDE_STROEMUNG` ist unbelegt.** `docs/archiv/2026-09-01_KLIMA_UND_SEE.md:97-98` benutzt
   die Konstante, die ±2 K stehen nur im Fließtext (:112-113), ein Codeort fehlt.
