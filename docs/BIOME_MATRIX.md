# Biome, Superbiome und die Vereinfachung des Wetters

Stand 2026-08-07, **umgesetzt und gemessen** (Vermerk 2026-08-12).

> **Der Kopf sagte bis zum 2026-08-12 "Noch nicht umgesetzt" — das war seit
> Wochen falsch.** Umgesetzt sind: die 15 Grundbiome (OFFENE_PUNKTE 2.1), die
> Regionsaffinität statt neun Matrizen (2.3, weich geglättet 2.4), Farben und
> Namen (2.5), die Bindung der Niederschlagsskala an die Ticklänge (2.6) und
> Alpin/Firn als Temperaturregel (2.2, 2.7). Offen sind aus diesem Dokument
> nur noch **Fels, Dünen und Aue** als Superbiome (2.2).
>
> **Diese Datei bleibt gültig als Referenz für die Zielwerte** — die
> Klimatabelle in §1 und die erwarteten Grundbiome je Region in §3 sind
> genau das, wogegen die Eichung misst (OFFENE_PUNKTE 9.1). Sie ist also
> kein Archiv, sondern die Sollvorgabe.
>
> **Eine Falle, die hier schon einmal zugeschnappt ist:** die Werte dieser
> Tabelle sind **Julitemperaturen**. Die Biom-Klassifikation las monatelang
> das Jahresmittel und lag damit systematisch 8–9 K zu kalt — das Macchia
> wurde zu 47 % Bruchwald statt Steineichenwald (OFFENE_PUNKTE 9.1). Wer hier
> Werte vergleicht, muss sicherstellen, dass er `temp_map_juli` gegenüberstellt
> und nicht `temp_map`.


## 1. Klimavorgaben auf MEERESHOEHE

Entscheidung des Nutzers 2026-08-07: die Werte gelten auf Meereshoehe, die
Hoehenabnahme rechnet mit **0.6 K je 100 m**. Die Bezugsorte sind deshalb auf
0 m zurueckgerechnet.

| Region | Bezugsort | Hoehe | Jan gemessen | Jul gemessen | **Jan auf 0 m** | **Jul auf 0 m** | Niederschlag |
|---|---|---:|---:|---:|---:|---:|---:|
| Clonagh | Cork | 10 m | 6.0 | 15.5 | **6.1** | **15.6** | 1200 mm |
| Skerrheim | Bergen | 20 m | 2.0 | 15.0 | **2.1** | **15.1** | 2250 mm |
| Morobora | Wologda | 130 m | −11.5 | 17.5 | **−10.7** | **18.3** | 600 mm |
| Estrande | La Rochelle | 15 m | 6.5 | 20.5 | **6.6** | **20.6** | 780 mm |
| Nevadin | Chur | 590 m | 0.0 | 18.5 | **3.5** | **22.0** | 850 mm |
| Nebelrode | Bamberg | 240 m | 0.5 | 19.0 | **1.9** | **20.4** | 640 mm |
| Samarcia | Madrid | 660 m | 6.5 | 25.5 | **10.5** | **29.5** | 430 mm |
| Macchia | Rom | 20 m | 8.0 | 25.5 | **8.1** | **25.6** | 800 mm |
| Thalassia | Iraklio | 40 m | 12.5 | 26.5 | **12.7** | **26.7** | 480 mm |

### Was daraus auf der TATSAECHLICHEN Regionshoehe wird

Das ist der Punkt, den man sehen muss, bevor man die Tabelle festschreibt:

| Region | Mittelhoehe | Jul auf 0 m | **Jul in der Region** | Bezugsort zum Vergleich |
|---|---:|---:|---:|---:|
| Clonagh | 147 m | 15.6 | **14.7** | Cork 15.5 |
| Skerrheim | 113 m | 15.1 | **14.4** | Bergen 15.0 |
| Morobora | 294 m | 18.3 | **16.5** | Wologda 17.5 |
| Nevadin | 800 m | 22.0 | **17.2** | Chur 18.5 |
| Nebelrode | 350 m | 20.4 | **18.3** | Bamberg 19.0 |
| Samarcia | 230 m | 29.5 | **28.1** | Madrid 25.5 |
| Macchia | 1 m | 25.6 | **25.6** | Rom 25.5 |
| Thalassia | Kueste | 26.7 | **26.7** | Iraklio 26.5 |

**Die Samarcia faellt auf.** Madrid liegt auf 660 m, unsere Samarcia auf 230 m —
zurueckgerechnet und wieder abgezogen bleiben 28.1 °C statt 25.5. Das ist
rechnerisch richtig (tiefer heisst waermer), aber sie waere damit die heisseste
Region der Karte, heisser als Kreta.

> **Zwei Wege, ich empfehle den zweiten:**
>
> 1. So lassen — die Samarcia ist eben eine Tiefebene, kein Hochplateau.
> 2. **Den Meereshoehenwert auf 27.0 °C senken** statt 29.5. Madrid ist als
>    Vorbild wegen seiner Hochlage gewaehlt worden; die Meseta-Hitze soll aus
>    der Trockenheit kommen, nicht aus einer Rueckrechnung.


## 2. Die Biome

### Was aus dem Bestand fliegt

Die heutige Tabelle hat 15 Grundbiome, davon fuenf, die auf einer 21-km-Insel
in Europa nichts zu suchen haben:

    Ice Cap · Tropical Rainforest · Tropical Seasonal · Savanna · Badlands

Und drei, die zu grob sind: `Grassland`, `Temperate Forest` und `Mediterranean`
muessen jeweils in zwei bis drei Arten zerfallen, sonst sieht das halbe
Festland gleich aus.

### Die sechzehn Grundbiome

Eingeteilt nach **Julitemperatur** und **Jahresniederschlag** — beides Groessen,
die das Wettersystem ohnehin liefert.

| # | Biome | Juli | Niederschlag | Charakter |
|---|---|---|---|---|
| 1 | **Hochmoor** | < 17 °C | > 1000 mm | Torf, staunass, sauer. Irland, Westnorwegen |
| 2 | **Bruchwald** | 14–22 °C | > 800 mm | Erle und Weide auf nassem Grund, Flussauen |
| 3 | **Feuchtwiese / Marsch** | 13–22 °C | > 700 mm | Kuestennahe Niederung, Salzwiesen, Deichland |
| 4 | **Grasland / Weide** | 13–22 °C | 500–900 mm | Der gruene Grundton, Vieh statt Pflug |
| 5 | **Heide** | 13–19 °C | 500–800 mm | Sandiger Magerboden, Zwergstraucher, Ginster |
| 6 | **Fjell / Bergheide** | < 12 °C | beliebig | Ueber der Baumgrenze, aber noch bewachsen: Zwergbirke, Flechten |
| 7 | **Nadelwald** | 12–18 °C | 450–900 mm | Fichte und Kiefer, der Taigatyp |
| 8 | **Mischwald** | 15–20 °C | 500–900 mm | Birke, Kiefer, Eiche — der Uebergang |
| 9 | **Buchenwald** | 16–21 °C | 600–1000 mm | Der mitteleuropaeische Hallenwald |
| 10 | **Eichen-Hainbuchenwald** | 18–23 °C | 500–800 mm | Waermer und trockener als Buche |
| 11 | **Bergwald** | 10–17 °C | > 700 mm | Tanne und Fichte am Hang, oberhalb des Laubwalds |
| 12 | **Macchia** | > 22 °C | 350–700 mm | Hartlaubbusch, immergruen, dorniger Bewuchs |
| 13 | **Steineichenwald** | > 21 °C | 550–900 mm | Der mediterrane Waldtyp |
| 14 | **Trockensteppe** | > 22 °C | 250–500 mm | Buescheliges Gras, weite offene Flaechen |
| 15 | **Halbwueste** | > 24 °C | < 300 mm | Kahler Boden, Salzkrusten, Dornstraeucher |
| 16 | **Duenen / Kuestensand** | beliebig | beliebig | Sandiger Kuestensaum (auch als Superbiom denkbar) |

### Die acht Superbiome

Sie ueberschreiben das Grundbiom nach EINER Regel — genau die Bauform, die der
Nutzer beschrieben hat.

| Superbiom | Regel | Bemerkung |
|---|---|---|
| **Alpin** | oberhalb der Baumgrenze | Die Grenze ist **temperaturabhaengig**, nicht auf fester Hoehe: sie liegt dort, wo die Julitemperatur unter rund 7 °C faellt. Damit ist sie von selbst keine gerade Linie — sie folgt Hangausrichtung, Schattenwurf und Regionsklima. |
| **Firn / Schneegrenze** | Julitemperatur unter 0 °C | Dieselbe Bauform eine Stufe hoeher |
| **Fels / Blockhalde** | Hang > 40° | Ueberall, nicht nur im Gebirge |
| **Klippe** | Hang > 30° UND Kuestenabstand < 100 m | Passt zur neuen `kuestenform` |
| **Strand** | Hoehe < 5 m UND Hang < 3° UND kuestennah | Siehe die Anmerkung unten |
| **Aue** | Flussabstand < Talbreite UND flach | Nutzt `river_generation` |
| **Fluss** | `river_generation` 1/2/3 | Bach, Fluss, Strom — steht schon |
| **See / Meer** | Hoehe <= 0 | steht schon |

**Zur Strandtiefe:** der Nutzer fragte, ob sie sich aus der Steigung ergeben
soll. *Vorschlag: ja, aber indirekt.* Wenn die Regel `Hoehe < 5 m UND Hang < 3°`
lautet, folgt die Breite des Strands von selbst aus dem Gelaende — eine flache
Kueste (Morobora, Nebelrode, `kuestenform` 0.45) bekommt breite Straende, eine
Klippenkueste (Clonagh, Skerrheim) gar keine. Das braucht keine eigene
Rechnung; die Kuestenform erledigt es bereits.


## 3. Welche Biome je Region zu erwarten sind

**Das ist keine Zuweisung, sondern eine ERWARTUNG.** Die Biome entstehen aus
Temperatur und Niederschlag; diese Tabelle sagt, was dabei herauskommen MUSS,
und ist damit die Pruefliste.

| Region | Klima | erwartete Grundbiome | typische Superbiome |
|---|---|---|---|
| **Clonagh** (Cork) | 15.6 °C / 1200 mm | Hochmoor · Grasland · Feuchtwiese · Heide · Bruchwald | Klippe, Fels |
| **Skerrheim** (Bergen) | 15.1 °C / 2250 mm | Hochmoor · Nadelwald · Bergwald · Fjell | Alpin, Firn, Klippe, Fels |
| **Morobora** (Wologda) | 18.3 °C / 600 mm | Nadelwald · Mischwald · Hochmoor · Grasland | Strand, Aue |
| **Estrande** (La Rochelle) | 20.6 °C / 780 mm | Feuchtwiese · Eichen-Hainbuchenwald · Grasland · Bruchwald | Strand, Duenen, Aue |
| **Nevadin** (Chur) | 22.0 °C / 850 mm | Bergwald · Fjell · Buchenwald (Taeler) · Grasland (Almen) | **Alpin, Firn, Fels** |
| **Nebelrode** (Bamberg) | 20.4 °C / 640 mm | Buchenwald · Mischwald · Grasland · Bergwald (Kaemme) | Strand, Aue |
| **Samarcia** (Madrid) | 29.5 °C / 430 mm | Trockensteppe · Halbwueste · Macchia · Steineichenwald | Fels |
| **Macchia** (Rom) | 25.6 °C / 800 mm | Steineichenwald · Macchia · Grasland · Bruchwald | Strand, Klippe |
| **Thalassia** (Iraklio) | 26.7 °C / 480 mm | Macchia · Trockensteppe · Steineichenwald | Strand, Klippe, Fels |

Jede Region bekommt damit 4 bis 5 charakteristische Arten, und **keine zwei
Regionen haben dieselbe Kombination**. Genau das war die Vorgabe.

**Ein Prueffall faellt dabei auf:** das Nevadin liegt mit 22.0 °C auf
Meereshoehe zu warm fuer Bergwald — der entsteht erst durch die Hoehenabnahme
auf 800 m und darueber. Das ist die Probe darauf, dass die Hoehenabnahme
wirklich wirkt.


## 4. Wie das Wettersystem einfacher wird

Alle Zahlen unten sind **gemessen** (2026-08-06/07, GPU-Pfad, siehe
OFFENE_PUNKTE 1.x, urspr. TODO B1), nicht geschaetzt.

### Die drei Vorschlaege des Nutzers

| Vorschlag | heute | danach | Ersparnis |
|---|---|---|---|
| **Nur Fruehling, keine Jahreszeiten** | 6 Zwei-Monats-Perioden | 1 | **6x** auf der ganzen Wetterkette |
| **Sonnenstand nur 5 Richtungen** | 7 Winkel × 6 Monate = 42 Schattenrechnungen | 5 | **8.4x** auf den Schattenwurf |
| **Nur eine Ebene statt drei** | 3 Schichten (Ground/Mid/High) | 1 | rund **3x** auf die Atmosphaerenrechnung |

Zusammen ist das Faktor 15 bis 20 — und der Speicher faellt entsprechend: alle
`*_monthly`- und `*_layers`-Ausgaben (rund 20 Stueck) verschwinden oder
schrumpfen auf ein Sechstel.

### Wo sonst noch hohe Kosten wenig leisten

**a) Die Atmosphaerensimulation selbst.** `_semi_lagrangian_advect_many` ist
bei 384 px **40 % der Temperaturrechnung** (21 von 52 s). Sie laesst ein Feld
ueber 25–50 Zeitschritte einschwingen — und die Schrittzahl haengt an der
Aufloesung, was genau der Drift ist, den du abschaffen willst.

> **Vorschlag:** die Temperatur wird gar nicht mehr simuliert, sondern
> FESTGELEGT (Region + Hoehe fuer Land, Breite + Stroemung fuer See, siehe
> docs/KLIMA_UND_SEE.md). Die Simulation bleibt nur fuer Wind und Feuchte —
> und dort mit FESTER Schrittzahl statt einer Tabelle nach Aufloesung.

**b) `settlement.plot_nodes`** kostet 9.6 s bei 512 px fuer ein Federmodell,
dessen Ergebnis heute leer ankommt. Die Physik soll ohnehin eingefroren werden
(urspr. TODO E10, heute OFFENE_PUNKTE Abschnitt 5) — damit faellt der Posten ganz weg.

**c) Der Schattenwurf auf der CPU** braucht 4.9 Millionen Aufrufe einer
Python-Funktion und ist 23x langsamer als die GPU-Fassung. Solange die GPU da
ist, kein Problem; faellt sie aus, wird das Programm unbenutzbar. Mit 5 statt
42 Rechnungen waere auch der CPU-Weg wieder tragbar.

**d) Der Erosionsfilter** LAEUFT (`EROSION_AKTIV = True`,
`gui/config/value_default.py:1088`), seit dem 27.08.2026. Hier stand bis zum
16.09.2026, er sei abgeschaltet und koste nichts - das war falsch. Er kostet
also, und die Regionsgewichtung hat er inzwischen: gemessen formt er in den
Bergen 64.5 m um und in den Niederungen 17.6, Faktor 3.7.

### Was ich NICHT vereinfachen wuerde

Den **Oktavenstapel** des Gelaendes. Er kostet 11.9 s bei 512 px, aber er ist
der Grund, warum Formgroesse und Rauheit sich raeumlich aendern koennen — ohne
ihn saehen alle neun Regionen gleich aus. Das ist Geld, das ankommt.


## 5. Entschieden am 2026-08-07

1. **Samarcia bleibt bei 29.5 Grad.** Nutzer: "lassen, kann gerne etwas heisser
   sein." Sie ist damit die heisseste Region der Karte, und das ist gewollt.

2. **Duenen werden SUPERBIOM**, nicht Grundbiom. Damit bleiben **15
   Grundbiome** (Nr. 1-15 oben) und **9 Superbiome**. Duenen sind eine
   Lageregel, kein Klima - dieselbe Bauform wie Strand und Klippe.

3. **Sommer und Winter bleiben, die SONNENSTAENDE nicht.** Nutzer: "Sommer und
   Winter sind noch vorhanden, aber die sonnenstaende sind die gleichen ...
   also kann einfach eine Min und eine Max temperatur sein und eine
   unregelmaessige kurve die ueber die jahreskurve gelegt wird."

   Das ist die weitreichendste der drei Entscheidungen; das Modell steht in
   `docs/KLIMA_UND_SEE.md`, Abschnitt 4.

   **Schneegrenze und Firn bleiben damit moeglich.** Im Januar liegt das
   Nevadin auf 800 m bei 3.5 - 4.8 = **-1.3 Grad**, die Gipfel deutlich
   darunter. Mit "nur Fruehling" waere selbst der hoechste Punkt schneefrei
   geblieben.
