# Kulturen, Landmarken und Roadsites — Auswahlliste

Stand 2026-08-06. Zeitschnitt **932 n. Chr.**

Der Nutzer waehlt aus jeder Liste **5 Buchstaben** aus; nur diese werden
umgesetzt. Die uebrigen bleiben als Reserve stehen.


## Drei Korrekturen an der Kulturzuordnung

`core/terrain_weltkarte.py` REGIONEN enthaelt drei Eintraege, die zum Zeitschnitt
932 nicht passen:

| Region | bisher | Vorschlag | Grund |
|---|---|---|---|
| Thalassia | Phoenizier | **Byzantiner** | Die phoenizischen Stadtstaaten enden rund 1500 Jahre vorher. Die Aegaeis ist 932 byzantinisch; Kreta ist zu diesem Zeitpunkt sogar arabisches Emirat und wird erst 961 zurueckerobert. |
| Nevadin | "-" | **Alemannen** | Hatte gar keine Kultur, bekam damit auch keine Staedte. 932 stehen dort Alemannen, in den Hochtaelern Raetoromanen. |
| Nebelrode | Franken | **Sachsen** | War doppelt mit der Estrande belegt. 932 regiert Heinrich I.; das Gebiet ist saechsisch und traegt dessen Burgenordnung. |

Damit sind alle neun Regionen kulturell unterschieden.


## Die neun Kulturen

| Lage | Region | Kultur | Charakter 932 |
|---|---|---|---|
| NW | Clonagh | Kelten | christlich-monastisch, Ringwallgehoefte, keine Staedte im roemischen Sinn |
| N | Skerrheim | Wikinger | vorchristlich, Hof und Thing, alles haengt am Wasserweg |
| NO | Morobora | Slawen | heidnisch, Holzburgen (Gorod), Pelz- und Flusshandel |
| W | Estrande | Franken | Westfranken nach dem Zerfall, Wikingerzuege, Motten und Abteien |
| M | Nevadin | Alemannen | Passverkehr, Saumhandel, Bergkloester |
| O | Nebelrode | Sachsen | jung christianisiert, Burgwarde Heinrichs I., Landwehren |
| SW | Samarcia | Andalusier | Kalifat von Cordoba (seit 929), Bewaesserung, Grenzwehr |
| S | Macchia | Italiener | Koenigreich Italien, Incastellamento gegen Sarazenenzuege |
| SO | Thalassia | Byzantiner | Themenverwaltung, Inselkastra, arabische Seeraeuberei |


## AUSWAHL DES NUTZERS — 2026-08-06

Das ist der verbindliche Satz. Alles darunter ist Reserve.

| Kultur | Landmarks | Roadsites |
|---|---|---|
| **Kelten** | Steinkreis auf der Kuppe · Heilige Quelle mit Opfergaben · Ogham-Stein als Grenzmal · Ganggrab · Bienenkorbzellen am Kliff | Furtstein an der Flussquerung · Rastkreuz an der Wegscheide · Bardenlager · Pilgerherberge · Zollringwall |
| **Wikinger** | Runenstein · Langhaus des Jarls · Hoergr — Steinaltar auf der Hoehe · Thingplatz · Gestrandetes Langschiff | Faehrstelle ueber den Fjord · Sennhuette · Handelsplatz am Strand · Kohlenmeiler · Salzsiederei |
| **Slawen** | Gorod — Ringwallburg · Heiliger Hain · Wehrturm aus Blockholz · Verlassene Brandrodung · Baerenhoehle mit Opferstelle | Bohlenweg durchs Sumpfland · Pelzhaendlerlager · Blockhausherberge · Grenzverhau aus Staemmen · Faehre am Strom |
| **Franken** | Steinerne Abtei · Rest einer Koenigspfalz · Salzgarten am Aestuar · Kliffkapelle · Aquaeduktstueck der Roemer | Zollbruecke · Wechselstall fuer Pferde · Fischerweiler · Weinschenke · Muehlenwehr |
| **Alemannen** | Bergkloster auf dem Sattel · Trutzburg auf dem Felskopf · Gletscherzunge · Eisenerzgrube · Wildheu-Alm | Passhospiz · Wechselstall fuer Saumtiere · Klause mit Wegzoll · Holzriese · Kaesespeicher |
| **Sachsen** | Stumpf der gefaellten Irminsul · Missionskirche aus Bruchstein · Silbergrube · Alte Landwehr · Opferstein im Buchenwald | Warte auf dem Kamm · Gerichtslinde · Wuestung · Kruggasthof · Kalkofen |
| **Andalusier** | Hisn — Felsenburg · Alcazaba-Ruine · Noria — Schoepfrad am Fluss · Atalaya — Signalturm · Nekropole am Wadi | Funduq — Karawanserei · Aljibe — Zisterne am Weg · Canada — Herdenweg · Zoco — Marktflecken · Oelmuehle |
| **Italiener** | Roemische Bruecke · Bergdorfkastell · Basilika mit Campanile · Terrassierte Olivenhaenge · Schwefelquelle | Via-Rest mit Meilenstein · Fischtrockenplatz · Weinpresse · Rastplatz auf dem Bergsattel · Osteria an der Kreuzung |
| **Byzantiner** | Kastro — Inselfestung · Klippenkloster · Antiker Tempel als Steinbruch · Antikes Amphitheater · Schiffswrackriff | Skala — Anlegebucht · Zisternenhof · Schwammtaucherlager · Eselspfad mit Stuetzmauern · Xenodocheion |

45 Landmark-Arten, 45 Roadsite-Arten. Jede Art traegt Platzierungsvorlieben
(Gipfel, Kueste, Furt, Kreuzung ...) — die stehen in
`docs/SIEDLUNGEN_ENTWURF.md`, nicht hier.


## Landmarks

Landmarks stehen fuer sich; sie liegen nicht am Weg, sondern praegen die
Gegend. Sie duerfen abgelegen sein.

### Kelten — Clonagh
a) Steinkreis auf der Kuppe
b) Hochkreuz aus Stein
c) Heilige Quelle mit Opfergaben
d) Ringwall (rath) — Erdwallgehoeft
e) Pfahlbauinsel in der Bucht (crannog)
f) Ogham-Stein als Grenzmal
g) Rundturm des Klosters
h) Ganggrab, aelter als jede Erinnerung
i) Feenhuegel — gemieden, nie gepfluegt
j) Bienenkorbzellen am Kliff

### Wikinger — Skerrheim
a) Runenstein
b) Grabhuegelfeld
c) Schiffssetzung aus Steinen
d) Langhaus des Jarls
e) Bootshaus am Fjordufer (naust)
f) Hoergr — Steinaltar auf der Hoehe
g) Thingplatz
h) Wachtfeuerkuppe (viti)
i) Gestrandetes Langschiff
j) Wasserfallheiligtum

### Slawen — Morobora
a) Gorod — Ringwallburg aus Holz und Erde
b) Goetzenpfahl mit vier Gesichtern
c) Heiliger Hain
d) Kurgan — Grabhuegel
e) Bienenbaumwald der Waldimker
f) Teerschwelerei
g) Wehrturm aus Blockholz
h) Quellheiligtum mit Baendern
i) Verlassene Brandrodung
j) Baerenhoehle mit Opferstelle

### Franken — Estrande
a) Steinerne Abtei
b) Rest einer Koenigspfalz
c) Motte — Turmhuegelburg
d) Roemischer Leuchtturm, Ruine
e) Salzgarten am Aestuar
f) Brandruine eines Wikingerzuges
g) Kliffkapelle
h) Reliquienschrein
i) Aquaeduktstueck der Roemer
j) Marschendeich mit Warft

### Alemannen — Nevadin
a) Bergkloster auf dem Sattel
b) Trutzburg auf dem Felskopf
c) Gletscherzunge
d) Wasserfall ueber die Trogwand
e) Bergsturz-Blockfeld
f) Eisenerzgrube
g) Wildheu-Alm
h) Roemischer Passaltar
i) Klause — Einsiedelei in der Wand
j) Steinbogenbruecke ueber die Klamm

### Sachsen — Nebelrode
a) Burgward Heinrichs I. — Ringburg
b) Stumpf der gefaellten Irminsul
c) Felsenkamm mit ausgehauenen Nischen
d) Missionskirche aus Bruchstein
e) Koehlerwald
f) Silbergrube
g) Huegelgraeberfeld
h) Alte Landwehr aus Wall und Graben
i) Opferstein im Buchenwald
j) Wallburgruine der Alten

### Andalusier — Samarcia
a) Hisn — Felsenburg ueber der Ebene
b) Ribat — Grenzwehrkloster
c) Alcazaba-Ruine
d) Qanat-Schacht des unterirdischen Stollens
e) Noria — Schoepfrad am Fluss
f) Bewaesserte Huerta
g) Atalaya — Signalturm
h) Roemisches Theater als Steinbruch
i) Nekropole am Wadi
j) Salzpfanne

### Italiener — Macchia
a) Roemische Bruecke, noch benutzt
b) Bergdorfkastell (incastellamento)
c) Basilika mit Campanile
d) Kuestenturm gegen Sarazenen
e) Verlassene Villa Rustica
f) Marmorbruch
g) Terrassierte Olivenhaenge
h) Katakombe
i) Aquaedukt-Bogenreihe
j) Schwefelquelle

### Byzantiner — Thalassia
a) Kastro — Inselfestung auf dem Felsen
b) Kuppelkirche mit Fresken
c) Klippenkloster
d) Antiker Tempel als Steinbruch
e) Leuchtfeuerkette (phryktoria)
f) Versunkene Marmormole
g) Windmuehlenkuppe
h) Antikes Amphitheater
i) Salzgaerten der Bucht
j) Schiffswrackriff


## Roadsites

Roadsites gehoeren an den Weg. Sie entstehen NACH dem Wegenetz und werden an
Kreuzungen, Furten und langen Zwischenstuecken gesetzt.

### Kelten — Clonagh
a) Furtstein an der Flussquerung
b) Rastkreuz an der Wegscheide
c) Bardenlager
d) Viehtriebpferch
e) Grenzhecke mit Torstein
f) Schmiede am Bachlauf
g) Torfstich
h) Meilenstein aus Rohstein
i) Pilgerherberge
j) Zollringwall

### Wikinger — Skerrheim
a) Naust als Raststelle
b) Faehrstelle ueber den Fjord
c) Schiffszug — Umtragestelle
d) Sennhuette
e) Steinmann als Wegmarke (varde)
f) Handelsplatz am Strand
g) Bohlenweg durchs Moor
h) Kohlenmeiler
i) Salzsiederei
j) Passhuette unter dem Grat

### Slawen — Morobora
a) Bohlenweg durchs Sumpfland
b) Wolok — Umtragestelle zwischen zwei Fluessen
c) Pelzhaendlerlager
d) Blockhausherberge
e) Grenzverhau aus gefaellten Staemmen
f) Faehre am Strom
g) Koehlerplatz
h) Waage der Salzstrasse
i) Wegpfahl mit Zeichen
j) Winterlager mit Schlittenspur

### Franken — Estrande
a) Zollbruecke
b) Wechselstall fuer Pferde
c) Ladestelle mit Tretkran
d) Fischerweiler
e) Roemerstrassenrest mit Meilenstein
f) Faehrhaus am Aestuar
g) Marktkreuz
h) Weinschenke
i) Muehlenwehr
j) Wachturm gegen Nordmaenner

### Alemannen — Nevadin
a) Passhospiz
b) Wechselstall fuer Saumtiere
c) Klause mit Wegzoll in der Enge
d) Almhuette
e) Lawinenunterstand
f) Schmelzhuette
g) Wegkreuz auf der Passhoehe
h) Furt ueber den Gletscherbach
i) Holzriese
j) Kaesespeicher

### Sachsen — Nebelrode
a) Hohlwegbuendel — ausgefahrene Karrenspuren
b) Warte auf dem Kamm
c) Landwehrdurchlass
d) Gerichtslinde
e) Wuestung — verlassenes Dorf
f) Hammerschmiede am Bach
g) Kruggasthof
h) Kalkofen
i) Furt mit gelegtem Steinbett
j) Rastplatz der Salzstrasse

### Andalusier — Samarcia
a) Funduq — Karawanserei
b) Aljibe — Zisterne am Weg
c) Wachturm der Signalkette
d) Wadi-Furt
e) Canada — Herdenweg
f) Toepferofen
g) Zoco — Marktflecken
h) Wehrspeicher
i) Oelmuehle
j) Grenzstein der Mark

### Italiener — Macchia
a) Via-Rest mit Meilenstein
b) Pilgerhospiz
c) Fischtrockenplatz
d) Weinpresse
e) Zollturm am Talausgang
f) Faehre unter der zerstoerten Bruecke
g) Rastplatz auf dem Bergsattel
h) Ziegelei
i) Osteria an der Kreuzung
j) Saumpfad zum Bergwerk

### Byzantiner — Thalassia
a) Skala — Anlegebucht mit Bootshaeusern
b) Zisternenhof
c) Wachturm ueber der Meerenge
d) Schwammtaucherlager
e) Eselspfad mit Stuetzmauern
f) Xenodocheion — Klosterherberge
g) Fischsalzerei
h) Zollstation des Themas
i) Kapelle am Kap
j) Umschlagplatz fuer Oel und Wein
