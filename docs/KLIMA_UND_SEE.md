# Klimavorgaben und Seegliederung

Stand 2026-08-07, **abgestimmt und umgesetzt** (Vermerk 2026-08-12).

> **Der Kopf sagte bis zum 2026-08-12 "Noch nicht umgesetzt — zur Abstimmung".
> Das war überholt.** Umgesetzt sind: die Seegliederung über den Voronoi-
> Zellgraphen (OFFENE_PUNKTE 3.1), die Tiefentabelle je Seegrad statt der
> alten Schelfformel (3.2), Seewege ab Grad 1 (3.3), die Regionszuordnung der
> Seezellen samt Skerrheim-/Hügelland-Sondertabellen und Morobora-Seeeis (3.6).
> Dazu kamen später die Küsten-Archetypen (3.8), die es hier noch nicht gab.
>
> **Die Leitlinie in §0 gilt unverändert weiter und ist der wichtigste Teil
> dieses Dokuments** — "Festlegung statt Regelkreis". Sie hat seither jede
> Entscheidung getragen: Temperatur und Niederschlag werden direkt auf ihren
> Zielwert normiert (1.3, 1.10, 1.11), die Meerestiefe kommt aus einer
> Tabelle statt aus einer Formel mit freien Konstanten (3.2). Wer hier
> weiterbaut, sollte §0 gelesen haben, bevor er einen neuen Regelkreis
> einführt.


## 0. Die Leitlinie: Festlegung statt Regelkreis

Der Nutzer am 2026-08-07: *"wir moechten in den naechsten iterationen eher nach
vereinfachungen suchen. also keine simulationskreise mehr, sondern in jedem
kreis stecken festlegungen damit es keinen drift mehr gibt."*

Das ist die Richtschnur fuer beide Vorschlaege unten und aendert die Bauweise
grundsaetzlich:

| bisher | kuenftig |
|---|---|
| Atmosphaere ueber 25–50 Zeitschritte einschwingen lassen | Zielwerte je Region vorgeben, Feld daraus aufbauen |
| Schelftiefe als `-t * (1 - exp(-d/L))` | Tiefe je Seegrad aus einer Tabelle |
| Ergebnis haengt an Schrittzahl und Aufloesung | Ergebnis haengt nur an Seed und Reglern |

Der Vorteil ist nicht bloss Geschwindigkeit. Ein Regelkreis, der 35 Schritte
laeuft, liefert bei 50 Schritten etwas anderes — genau das ist der Drift, den
wir seit Wochen bei jeder Aufloesungsaenderung nachkalibrieren.


## 1. Klimavorgaben je Region

### Die Bezugsorte

Vom Nutzer vorgegeben. Die Zahlen sind Klimamittel (rund 1990–2020) und hier
aus dem Gedaechtnis zusammengetragen — **vor der Umsetzung an einer Quelle
gegenpruefen**, sie sind auf ein bis zwei Grad genau, nicht besser.

| Region | Bezugsort | Hoehe des Orts | Jan | Jul | Spanne | Niederschlag |
|---|---|---:|---:|---:|---:|---:|
| Clonagh (Kelten) | **Cork**, Irland | 10 m | 6.0 °C | 15.5 °C | 9.5 K | 1200 mm |
| Skerrheim (Wikinger) | **Bergen**, Norwegen | 20 m | 2.0 °C | 15.0 °C | 13.0 K | **2250 mm** |
| Morobora (Slawen) | **Wologda**, Russland | 130 m | **−11.5 °C** | 17.5 °C | **29.0 K** | 600 mm |
| Estrande (Franken) | **La Rochelle**, Frankreich | 15 m | 6.5 °C | 20.5 °C | 14.0 K | 780 mm |
| Nevadin (Alemannen) | **Chur**, Schweiz | 590 m | 0.0 °C | 18.5 °C | 18.5 K | 850 mm |
| Nebelrode (Sachsen) | **Bamberg**, Deutschland | 240 m | 0.5 °C | 19.0 °C | 18.5 K | 640 mm |
| Samarcia (Andalusier) | **Madrid**, Spanien | 660 m | 6.5 °C | **25.5 °C** | 19.0 K | **430 mm** |
| Macchia (Italiener) | **Rom**, Italien | 20 m | 8.0 °C | 25.5 °C | 17.5 K | 800 mm |
| Thalassia (Byzantiner) | **Iraklio**, Kreta | 40 m | **12.5 °C** | 26.5 °C | 14.0 K | 480 mm |

Das ergibt eine schoene Spreizung: 24 K Unterschied im Januar zwischen Morobora
und Kreta, und Faktor 5 im Niederschlag zwischen Bergen und Madrid.

### Die Bezugshoehe ist der Knackpunkt

Chur liegt auf 590 m, unser Nevadin hat 800 m Mittelhoehe. Wologda liegt auf
130 m, unsere Morobora auf 294 m. Die Tabellenwerte gelten also **nicht** ohne
Weiteres fuer unsere Regionen.

> **ENTSCHIEDEN 2026-08-07: die Werte gelten auf MEERESHOEHE**, und die
> Hoehenabnahme rechnet mit **0.6 K je 100 m**. Nutzer: "also werte fuer
> meereshoehe angeben ist richtig. also 0,6 C/100 m oder so hochrechnen."
>
> Die Bezugsorte sind dafuer zurueckgerechnet - die Tabelle steht in
> `docs/BIOME_MATRIX.md`, Abschnitt 1.
>
> Mein urspruenglicher Vorschlag war die MITTLERE HOEHE der Region als Bezug.
> Der Nutzer hat sich fuer Meereshoehe entschieden, und das hat einen Vorzug,
> den ich uebersehen hatte: eine Regionshoehe kann sich noch aendern (sie ist
> ein geeichter Wert, kein fester), und dann waere die Klimavorgabe stillschweigend
> mitgewandert. Meereshoehe ist der einzige Bezug, der nicht mitwandert.
>
> **Die Folge muss man sehen:** das Nevadin steht damit auf 22.0 Grad
> Meereshoehe im Juli und kommt auf seinen 800 m bei 17.2 Grad heraus. Die
> Samarcia kommt auf 28.1 Grad und ist die heisseste Region der Karte - der
> Nutzer dazu: "lassen, kann gerne etwas heisser sein."

### Das Meer

Nutzer: *"alles unterhalb von 0 m (meer) hat eine mitteltemperatur (je nach
noerdlichkeit innerhalb der karte) und ist nicht durch die sonne beeinflusst.
also sowas wie im sommer 25 °C im sueden und 15 °C im norden."*

> **Vorschlag:**
>
> ```
> T_meer(y, monat) = T_sued(monat) + (T_nord(monat) - T_sued(monat)) * y_anteil
>                    + stroemung(x, y) * AMPLITUDE_STROEMUNG
> ```
>
> mit `y_anteil` = 0 am Suedrand, 1 am Nordrand, und
>
> | | Sued | Nord |
> |---|---|---|
> | Juli | 25 °C | 15 °C |
> | Januar | 14 °C | 4 °C |
>
> Die Januarwerte sind mein Vorschlag: das Meer hat eine deutlich kleinere
> Jahresspanne als das Land (Waermetraegheit), rund 11 K statt 18–29 K.
>
> `stroemung` ist eine EIGENE Rauschkarte aus demselben Seed, sehr grob
> (Wellenlaenge ~8 km), Amplitude **±2 K**. Der Nutzer: *"geringer varianz ...
> dabei wird eine noisemap verwendet die meeresstroemungen etwas darstellt."*
>
> **KEIN Sonneneinfluss, keine Schattenkarte, keine Hoehenabnahme** auf See.
> Das ist eine Festlegung, kein Naeherungsverfahren — und sie loescht den
> 5.6-K-Kuestensprung, dessen Ursache seit Wochen offen ist.

### Was daraus folgt

`weather.temperature` braucht dann keine Atmosphaerensimulation mehr, um die
Temperatur zu bestimmen: Landtemperatur aus Region + Hoehe, Seetemperatur aus
Breite + Stroemung, an der Kueste ueber wenige hundert Meter ueberblendet.

Wind und Feuchte bleiben zunaechst, wie sie sind — sie sind der zweite Schritt.


## 2. Die See als Voronoi-Gliederung

Nutzer: *"wenn wir das inland als voronoi kacheln haben, dann koennen wir ja
auch das gleiche bei der see machen ... aber auch zB das voronois an der kueste
nicht staerker vertieft werden, aber dass die voronois mit dem grad 1 etwas
vertiefter sind, grad 2 noch vertiefter etc."*

### Es ist fast schon da

`voronoi_regionen()` berechnet `etikett` — die Zellzuordnung — bereits fuer die
**ganze Karte**, nicht nur fuer den Kontinent. Die Seezellen existieren also
schon; sie werden nur nicht benutzt.

### Der Seegrad

> **Vorschlag:** eine Breitensuche ueber den Zellnachbarschaftsgraphen.
>
> ```
> Grad 0   Zelle enthaelt Land            -> Kueste, keine Vertiefung
> Grad 1   Zelle grenzt an eine Grad-0-Zelle
> Grad 2   grenzt an Grad 1
> ...
> ```
>
> Und die Tiefe als **Tabelle**, nicht als Formel:
>
> | Seegrad | Zieltiefe |
> |---|---|
> | 0 | 0 m (unveraendert, die Kueste formt `kuestenform`) |
> | 1 | −40 m |
> | 2 | −90 m |
> | 3 | −150 m |
> | 4+ | −200 m (= `MEERESBODEN_M`) |
>
> Der Uebergang zwischen zwei Graden wird ueber die Zellgrenze geglaettet
> (dieselbe Gaussglaettung wie bei den Regionsgewichten), sonst stuenden dort
> Stufen.

### Was das ersetzt

Den **Kuestenschelf** `-t * (1 - exp(-d/L))`, der heute mit zwei Konstanten
(70 m, 350 m) arbeitet und dessen Ergebnis an der Aufloesung der
Abstandstransformation haengt. Die Tabelle ist eine Festlegung — genau die
Richtung aus Abschnitt 0.

### Was es zusaetzlich bringt

Der Seegrad ist eine **Karte, die spaeter etwas bedeutet**: wo Seemonster
spawnen, wo Seewege verlaufen duerfen (der Siedlungsentwurf verlangt "der
groesste Teil der Laenge in Wasser ab 10 m Tiefe" — das waere kuenftig
"ab Grad 1"), wo Fischgruende liegen. Sie kostet also nichts extra und wird
mehrfach gebraucht.

### Vorbehalt

Die Zellen sind rund 200 Stueck auf der ganzen Karte, also gut 1.5 km gross.
Ein Grad entspricht damit einem Sprung von rund 1.5 km — das ist grob. Wenn
das im Bild als Kachelung sichtbar wird, muss die Punktzahl fuer die See
hoeher liegen als fuer das Land (etwa 400 statt 200).

**Das sollte man messen, bevor man es festschreibt.**


## 4. Das Temperaturmodell (entschieden 2026-08-07)

Nutzer: *"Sommer und Winter sind noch vorhanden, aber die sonnenstaende sind die
gleichen. wir werden aber im fertigen spiel komplette jahre simulieren. also
kann einfach eine Min und eine Max temperatur sein und eine unregelmaessige
kurve die ueber die jahreskurve gelegt wird."*

Das ist die weitreichendste Vereinfachung des ganzen Umbaus, denn es trennt
zwei Dinge, die heute verschraenkt sind:

    RAUM   das Muster ueber die Karte   -> EINMAL gerechnet
    ZEIT   der Jahresgang               -> eine SKALARE Funktion

### Die Formel

    T(x, y, t) = T_mittel(x, y) + spanne(x, y) / 2 * jahresgang(t)

mit

    T_mittel(x,y)   Land:  T_mittel_region - 0.006 * hoehe_m
                    See:   T_sued + (T_nord - T_sued) * y_anteil + stroemung
    spanne(x,y)     die Jahresspanne der Region (See: deutlich kleiner)
    jahresgang(t)   -cos(2*pi*(t - phase)) + unruhe(t)

### Warum das so viel spart

Heute wird die **ganze Wetterkette sechsmal** gefahren, einmal je Zwei-Monats-
Periode, und jedes Mal mit eigenen Sonnenstaenden und eigener
Atmosphaerensimulation. Kuenftig:

* der Sonnenstand ist fest (5 Richtungen, keine Jahreszeit) -> der Schattenwurf
  wird **einmal** gerechnet statt 42-mal
* das raeumliche Muster wird **einmal** gerechnet statt sechsmal
* der Jahresgang ist eine Kosinusfunktion mit ein paar Oberwellen - er kostet
  nichts und laesst sich auf JEDEN Zeitpunkt auswerten, nicht nur auf sechs

Damit kann das fertige Spiel **komplette Jahre** durchlaufen, ohne dass die
Erzeugung teurer wird. Heute waere jeder zusaetzliche Zeitpunkt ein weiterer
voller Durchlauf.

### Die unregelmaessige Kurve

    unruhe(t) = summe ueber k=2..5 von  a_k * sin(2*pi*k*t + phi_k)

Die Phasen `phi_k` kommen aus dem Kartenseed, die Amplituden fallen mit 1/k ab
und summieren sich auf rund **15 %** der Jahresspanne. Ergebnis: ein Jahr mit
Waermeeinbruechen und milden Wochen, aber unveraendertem Mittel und
unveraenderten Extremen.

REPRODUZIERBAR, weil aus dem Seed abgeleitet - kein Zufall zur Laufzeit.

### Die neun Regionen als Zahlen

Jahresmittel und halbe Spanne, auf Meereshoehe, aus der Tabelle in
`docs/BIOME_MATRIX.md`:

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
| **See Sued** | 14.0 | 25.0 | 19.5 | 5.5 |
| **See Nord** | 4.0 | 15.0 | 9.5 | 5.5 |

Die Morobora hat mit 29 K die dreifache Jahresspanne des Huegellands - das ist der
Unterschied zwischen Kontinental- und Seeklima, und er faellt hier von selbst
heraus, ohne dass ihn jemand modellieren muss.

**Zwei neue Regionsregler** kommen dazu, nach dem Muster von `kuestenform`:
`temp_mittel_m0` und `temp_spanne`. Der Niederschlag folgt spaeter genauso.

### Was der Sonnenstand dann noch tut

Er formt das Muster INNERHALB eines Tages und ueber das Gelaende: Suedhang
waermer als Nordhang, Talschatten kuehler. Das bleibt - es ist der Grund,
warum die Baumgrenze keine gerade Linie wird. Nur die JAHRESZEITLICHE
Verschiebung des Sonnenstands faellt weg.


## 3. Reihenfolge

```
1  Klimatabelle gegenpruefen                    (Nutzer)
2  Seegrad bauen und ANSEHEN                    halber Tag
   -> Kachelung sichtbar? Dann Punktzahl hoch.
3  Schelf durch die Gradtabelle ersetzen        halber Tag
4  Seetemperatur als Festlegung                 1 Tag
5  Landtemperatur je Region als Festlegung      1 Tag
6  Atmosphaerensimulation auf Wind/Feuchte      offen, spaeter
   zurueckschneiden
```

Schritt 2 vor 3, weil die Kachelgroesse eine Sichtprobe braucht und nicht nur
eine Zahl.
