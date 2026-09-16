# Sollbeschreibung MapGenerator — **GERÜST**

**Was das hier ist.** Die Beschreibung dessen, was der MapGenerator können
soll. Nicht, was er kann — das steht im Ist-Stand-Inventar
([Issue #15](https://github.com/nikolaisoerensen/MapGenerator/issues/15)).
Gegen dieses Dokument wird jede Umsetzung geprüft, und aus dem Abgleich
zwischen ihm und dem Ist-Stand entstehen die Arbeitstickets.

**Stand 2026-09-16: alle 45 Fragen des ersten Durchgangs sind beantwortet.**
Jede Antwort steht an der Stelle, an der vorher die Frage stand, und nennt das
Entscheidungsticket, in dem sie gefallen ist:

> **✅ ENTSCHIEDEN — [#NN](link):** die Antwort, mit Verweis auf das
> Arbeitsticket, das sie umsetzt.

Zwei Stellen sind ausdrücklich als **Offen geblieben** markiert: der Fragebogen
hat sie nie gestellt. Sie kommen in den nächsten Durchgang. Eine neue Frage
wird wieder als solche eingetragen:

> **❓ OFFEN — #NN:** die Frage im Wortlaut.

**Dieses Dokument wird nicht frei fortgeschrieben.** Es wächst über die Karte
[#13](https://github.com/nikolaisoerensen/MapGenerator/issues/13): jede
geschlossene Frage wird hier eingetragen, jede neue Frage kommt als
Fragezeichen dazu. Wer hier etwas ausformuliert, ohne dass das zugehörige
Ticket geschlossen ist, erfindet eine Entscheidung.

**Es ersetzt** `docs/SPEZIFIKATION.md` §1 und §2, sobald es fertig ist. Deren
Zielbild — zwanzig reale Landschaften der Erde, ausdrücklich *ohne Meer* —
beschreibt nicht das Programm, das gebaut wurde. Welche weiteren Dokumente
darin aufgehen, entscheidet
[#24](https://github.com/nikolaisoerensen/MapGenerator/issues/24).

---

## Problem Statement

Der Nutzer baut seit Monaten an einem Kartengenerator und kann bei keiner
Einzelentscheidung mehr sicher sagen, ob sie dem Ziel dient — **weil das Ziel
nirgends vollständig aufgeschrieben ist.**

Vier Folgen davon, alle belegt:

1. **Die Dokumentation beschreibt ein anderes Programm als der Code.** Das
   Inventar vom 2026-09-14 fand acht Widersprüche. Der schwerste: die
   Erosionskette **läuft** seit dem 2026-08-27, während vier Dokumente sie
   als abgeschaltet führen und ein fünftes ihre „Reaktivierung" als offenes
   Ziel. Damit steht der ungeklärte Faktor 385 zwischen GPU und CPU nicht vor
   einer Reaktivierung, sondern mitten im laufenden Betrieb.
2. **Es gibt drei konkurrierende Reihenfolgen** — in `OFFENE_PUNKTE.md`, in
   `AUFRAEUMPLAN.md` und als Ticketbündel — und keine sagt, welche gilt.
3. **Es gibt keine Abnahme.** Der Zielkatalog in `SPEZIFIKATION.md` §2 ist
   seit dem 2026-07-29 leer. Mehrere Messgrößen in §3 stehen auf „nicht
   gemessen". §6.1 vermerkt über sich selbst, dass ein Prüfstand fehlt, der
   alle Komponenten gleichzeitig bewertet. „Gut" heißt deshalb heute: der
   Nutzer sieht sich einen Bildschirm an.
4. **Niemand kann allein daran arbeiten.** Weder ein Agent noch der Nutzer
   nach einer Pause, weil jede Sitzung neu herleiten muss, wofür das Programm
   da ist.

## Solution

Ein Dokument, das beschreibt, was das Programm können soll — vollständig
genug, dass

* jede Umsetzung dagegen prüfbar ist,
* aus dem Abgleich mit dem Ist-Stand Arbeitstickets entstehen,
* ein Agent weitgehend allein daran arbeiten kann, auch nachts,
* und eine Funktion, die sich hier nicht wiederfindet, entweder eingetragen
  oder entfernt wird.

**Der Zweck des Programms, entschieden am 2026-09-14:** Der MapGenerator ist
der **Karteneditor für ein Spiel im Stil von Die Gilde 2, nur besser** — und
darüber hinaus die **Testwiese**, auf der Simulationsfunktionen ausprobiert
werden. Der Nutzer erzeugt eine Welt, und wenn sie ihm gefällt, startet er auf
Knopfdruck die Simulation. Im fertigen Spiel beginnt an derselben Stelle das
Spiel.

**„Nur besser" heißt:** ein Mittelalter, das halbwegs glaubwürdig ist — in dem
aber alles wahr ist, woran die Menschen damals glaubten. Eine dunkle,
gefährliche Welt mit hoher Schwierigkeit und Rückschlägen. Neun Kulturen und
eine erzeugte Welt, die Tiefe gibt. Ein Erzählton wie in RimWorld.
Gegenstände in Schalen wie in Diablo: Material, bis zu zwei Adjektive,
Qualität als Gaußkurve um das Können des Schmieds, Gravuren. Rezepte, die aus
verschiedenen Regionen heimgebracht werden müssen. Arenen. Ereignisse aus der
Wildnis, die Wirtschaft und Handel einer Region stören, Menschen entführen und
töten.

---

## User Stories

### A — Als Kartenbauer (der Nutzer am Editor)

1. Als Kartenbauer möchte ich eine vollständige Welt aus einem Seed erzeugen, damit ich nicht jede Landschaft von Hand bauen muss.
2. Als Kartenbauer möchte ich, dass kein Reglerstand die Welt zerstören kann, damit ich nicht wissen muss, wo Probleme aufschwingen.
3. Als Kartenbauer möchte ich Regler, die Landschaften beschreiben statt Rechenverfahren, damit ich einstellen kann, was ich sehen will, statt zu erraten, was ein Parameter tut.
4. Als Kartenbauer möchte ich erkennen, welcher Region eine Stelle der Karte angehört, damit ich die neun Regionen auseinanderhalten kann.
5. Als Kartenbauer möchte ich eine Region gezielt verändern können, ohne eine Programmdatei zu editieren, damit ich das Nevadin anpassen kann, ohne Python zu schreiben.
6. Als Kartenbauer möchte ich sehen, dass eine Region ihrem realen Vorbild ähnelt, damit die Welt glaubwürdig wirkt.
7. Als Kartenbauer möchte ich jede erzeugte Ebene sowohl in 2D als auch in 3D betrachten können, damit ich Fehler in der Form sehe, die eine Aufsicht verbirgt.
8. Als Kartenbauer möchte ich eine Live-Vorschau beim Reglerziehen, damit ich die Wirkung sofort sehe und nicht nach jedem Zug minutenlang warte.
9. Als Kartenbauer möchte ich beim Öffnen eines Reiters keine mehrsekündige Rechnung auslösen, damit das Programm zügig bedienbar bleibt.
10. Als Kartenbauer möchte ich die Jahreszeit umschalten und sehen, wie Temperatur, Wind, Feuchte, Schnee und Trockenheit sich ändern, damit ich die Welt über das Jahr beurteilen kann.
11. Als Kartenbauer möchte ich, dass Biome und Geologie sich mit der Jahreszeit **nicht** ändern, damit die Karte über das Jahr dieselbe Welt bleibt.
12. Als Kartenbauer möchte ich Siedlungen, Wege und Parzellen sehen, damit ich beurteilen kann, ob die Welt bewohnbar aussieht.
13. Als Kartenbauer möchte ich erkennen, welche Rohstoffe wo zu finden sind, damit ich einschätzen kann, ob die Welt eine Wirtschaft trägt.
14. Als Kartenbauer möchte ich eine fertige Welt exportieren, damit Spiel und Simulation sie lesen können.
15. Als Kartenbauer möchte ich eine erzeugte Welt wiederfinden, ohne sie neu berechnen zu lassen, damit Minuten Rechenzeit nicht mehrfach anfallen.
16. Als Kartenbauer möchte ich auf Knopfdruck die Simulation auf der fertigen Welt starten, damit ich sehe, ob die Welt funktioniert und nicht nur aussieht.

### B — Als Simulationsbauer (der Nutzer an der Testwiese)

17. Als Simulationsbauer möchte ich, dass jede Parzelle weiß, welche Rohstoffe dort zu finden sind, damit Berufsstätten daraus entstehen können.
18. Als Simulationsbauer möchte ich, dass auf einem Teil der freien Parzellen Berufsstätten entstehen, die zur Lage passen — Höfe im Acker, Holzfäller am Wald, Minen am Erz.
19. Als Simulationsbauer möchte ich Bürger als Gruppen nach Beruf, Altersgruppe und Kultur, damit ich keine Einzelpersonen simulieren muss.
20. Als Simulationsbauer möchte ich, dass Handel aus den Bedürfnissen der Stände und aus Mangel entsteht, damit die Wirtschaft nicht vorgegeben ist.
21. Als Simulationsbauer möchte ich die Simulation **ohne** den Editor laufen lassen können, damit ein Versuchslauf nicht jedes Mal den vollen Weltaufbau bezahlt.
22. Als Simulationsbauer möchte ich vorgebackene Testkarten als Eingabe verwenden, damit ich dieselbe Welt hundertmal durchrechnen kann.
23. Als Simulationsbauer möchte ich die Ergebnisse der Simulation im Editor anzeigen können, damit ich sehe, wo sich etwas staut.
24. Als Simulationsbauer möchte ich, dass die Simulation um Siedlungen herum dichter rechnet als in der Wildnis, damit der Aufwand dort anfällt, wo etwas geschieht.

### C — Als Spieler (mittelbar: was die Karte dem Spiel schuldet)

25. Als Spieler möchte ich eine nahtlose Welt über die vollen 21 km, damit ich nicht zwischen Karten wechseln muss.
26. Als Spieler möchte ich, dass sich die Welt in logischen Stücken aufdeckt — das Tal, in dem ich stehe, die Siedlungen, die ich erreichen kann — statt in einem Kreis um mich herum.
27. Als Spieler möchte ich, dass hinter einem Gebirge nichts aufgedeckt wird, solange der Weg dorthin beschwerlich ist, damit Entfernung sich nach Mühe anfühlt und nicht nach Luftlinie.
28. Als Spieler möchte ich, dass Regionen sich nicht nur anders **anfühlen**, sondern anderes **hergeben**, damit sich eine Reise lohnt und Rezepte Heimatorte haben.
29. Als Spieler möchte ich Orte finden, an denen der Aberglaube wahr wird — alte Wälder, Ruinen, Moore, abgelegene Stellen — damit die Welt gefährlich bleibt.
30. Als Spieler möchte ich, dass diese Orte nicht in jedem Durchgang gleich liegen, damit die Welt sich nicht auswendig lernen lässt.
31. Als Spieler möchte ich Jahreszeiten erleben, die Temperatur, Wind und Feuchte ändern, damit das Jahr eine Form hat.
32. Als Spieler möchte ich, dass Siedlungen und Kulturen sich unterscheiden, damit die neun Regionen mehr sind als Farbflächen.

### D — Als Entwickler oder Agent am Programm

33. Als Entwickler möchte ich in **einem** Dokument nachlesen, was das Programm können soll, damit ich nicht zwanzig Dateien gegeneinander abwägen muss.
34. Als Entwickler möchte ich, dass keine Dokumentation etwas behauptet, was der Code nicht tut, damit ich nicht auf einer falschen Annahme aufbaue.
35. Als Entwickler möchte ich eine Regel, die entscheidet, ob etwas in den Generator oder in die Simulation gehört, damit diese Frage nicht bei jedem Einbau neu verhandelt wird.
36. Als Entwickler möchte ich, dass jede Anzeige in derselben Änderung für 2D und 3D entsteht, damit nicht dreimal derselbe Fehler passiert.
37. Als Entwickler möchte ich, dass jeder stille Rückfall auf einen Ersatzpfad eine laute Logzeile schreibt, damit ein Ausfall von einem Erfolg unterscheidbar ist.
38. Als Entwickler möchte ich einen Ortstyp oder ein Eignungsfeld hinzufügen können, ohne eine zweite Geländeanalyse zu schreiben, damit keine zweite Wahrheit entsteht.
39. Als Entwickler möchte ich einen geschriebenen Hausstandard für Code, damit ich meine eigene Arbeit daran messen kann.
40. Als Agent möchte ich ein Ticket allein abschließen können, damit nachts gearbeitet werden kann.
41. Als Agent möchte ich wissen, welche Arbeit ich **nicht** allein abschließen darf, damit ich nichts ungeprüft nach `main` bringe.

### E — Als Prüfer

42. Als Prüfer möchte ich eine Kennzahl je Zielgröße, damit „gut" nicht Geschmackssache ist.
43. Als Prüfer möchte ich Tests, die mit den echten Kartengrößen laufen (128/256/512/1024), damit sie keine ausgedachte Situation prüfen.
44. Als Prüfer möchte ich, dass ein Test das Ergebnis prüft und nicht den Parameter, damit eine tote Funktion nicht grün durchläuft.
45. Als Prüfer möchte ich einen Testlauf, der schnell genug für rot-grün-umbauen ist, damit Messen nicht durch Raten ersetzt wird.
46. Als Prüfer möchte ich morgens in einem Blick sehen, was in der Nacht geschah, damit ich nicht durch Protokolle suchen muss.
47. Als Prüfer möchte ich wissen, welche Prüfungen nur am laufenden Programm möglich sind, damit ich sie nicht für vergessen halte.

---

## Implementation Decisions

### 1. Der Leitsatz: die Zeitpunkt-null-Regel

**Alles, was eine Eigenschaft der Welt zum Zeitpunkt null ist, gehört in den
Generator. Alles, was sich über die Zeit ändert, gehört in die Simulation.
Die gebackene Welt ist die Naht dazwischen.**

| gehört in den **Generator** | gehört in die **Simulation** |
|---|---|
| Gelände, Geologie, Gewässer, Biome | Bedürfnisse, Mangel, Preise |
| Siedlungen, Wege, Parzellen | Handel zwischen Siedlungen |
| Rohstoffe je Parzelle | Ereignisse der Wildnis |
| Berufsstätten als Anfangsbestand | Bevölkerungsentwicklung, Altern |
| Bürgergruppen als Anfangsbestand | alles Erzählerische |
| die jahreszeitlichen Klimafelder | |

Diese Regel beantwortet jede spätere Frage der Form „gehört das noch in den
Editor?" in einem Satz. Sie ist am 2026-09-14 vom Nutzer angenommen worden.

### 2. Drei Programme, eine Datei dazwischen

Der **Kartengenerator** schreibt die gebackene Welt. Eine **Simulation** liest
sie, rechnet Jahre und schreibt Zustände. Das **Spiel** liest beides. Der
Editor **startet** die Simulation auf Knopfdruck, **rechnet sie aber nicht**.

Begründung: Generator und Simulation haben unvereinbare Arbeitsschleifen. Ein
Generator wird abgenommen, indem man einmal rechnet und hinsieht; eine
Simulation, indem man sie fünfzigmal über zehn Jahre laufen lässt. In einem
Programm zahlt jedes Simulationsexperiment den vollen Weltaufbau mit — heute
Minuten je Lauf plus 19 Minuten Testlauf. Danach wird nicht mehr gemessen,
sondern geraten.

Die Naht ist eine **Datei**, beide Seiten sind einzeln dagegen prüfbar, und
die Hälfte steht bereits: der Godot-Export ist genau diese Datei.
`tools/biome_lab/` ist der Beleg, dass das Muster hier trägt — ein eigenes
Programm, um die Parzellenphysik außerhalb des Editors durchzuprobieren.

**Der Umzug ist entschieden, aber nicht fällig.** Solange nur Anfangszustände
entstehen, bleibt alles im Generator. Fällig wird die Trennung mit
Bedürfnissen und Handel.

### 3. Der Weltaufbau

* **Neun Regionen in einem 3×3-Raster**, mit je eigenem Parametersatz, eigener
  Kultur und einem realen Bezugsort. Die neun sind **fest**; höchstens Namen
  und Kleinigkeiten ändern sich.
* **Eine nahtlose Welt über die vollen 21 km**, kein Kartenwechsel je Region.
* **Festlegung statt Regelkreis** (Leitlinie seit 2026-08-07): Zielwerte
  vorgeben und das Feld daraus aufbauen, statt ein Gleichgewicht einschwingen
  zu lassen. Ein Regelkreis über 35 Schritte liefert bei 50 Schritten etwas
  anderes — genau das ist der Drift, der bei jeder Auflösungsänderung
  nachkalibriert werden musste.

### 4. Sichtbarkeit folgt dem Wegaufwand, nicht der Luftlinie

Die Welt gliedert sich in Stücke, die sich nach **Erreichbarkeit** bilden, nicht
nach Abstand: das Tal, in dem man steht, und die Siedlungen, die von dort
erreichbar sind. Dieselbe Gliederung trägt die Simulation, die ohnehin um
Siedlungen herum verdichtet ist.

Der Rechenkern dafür existiert bereits: ein Mehrquellen-Dijkstra über das
Bildpunktgitter mit hangneigungsgewichteten Schrittkosten, der je Bildpunkt
die billigst erreichbare Quelle und die Kosten dorthin liefert, mit
Kappungsgrenze und GPU-Fassung. Er wird heute für Stadtgrenzen und Parzellen
benutzt.

> **✅ ENTSCHIEDEN — [#27](https://github.com/nikolaisoerensen/MapGenerator/issues/27):** Die Gliederung **bleibt spielseitig**. Der Editor liefert die Wegekosten, aus denen sie folgt; wie fein das Spiel daraus seine Simulation staffelt — grob je Voronoi-Zelle, nicht je Bildpunkt —, ist nicht Sache des Karteneditors.
> **✅ ENTSCHIEDEN — [#27](https://github.com/nikolaisoerensen/MapGenerator/issues/27):** **Ja, nach Wegart gestaffelt.** Ein Trampelpfad, ein Karrenweg und eine gepflasterte Straße kosten je Kilometer Unterschiedliches. Der Haken ist die Rückkopplung: ein Weg macht sich selbst schneller und zieht damit weitere Wege an sich, weshalb kleine direkte Verbindungen seltener entstehen. Dazu kommt die Wegersparnis durch Mitbenutzung. Das ist ein kalibriertes System und muss im Betrieb gemessen werden, sonst läuft es weg.
> **✅ ENTSCHIEDEN — [#27](https://github.com/nikolaisoerensen/MapGenerator/issues/27):** **Beides.** Wasser verbindet längs — Wege folgen Ufern, und ein Teil der Händler fährt mit Booten, die Mehrzahl bleibt auf Straßen — und trennt quer. Eine Querung ohne Brücke ist teuer; **steht die Brücke, ist sie billig**. Die Brücke ist damit ein Bauwerk im selben Regelkreis, kein fester Geländewert.
> **✅ ENTSCHIEDEN — [#27](https://github.com/nikolaisoerensen/MapGenerator/issues/27):** **Nicht in Metern, sondern in Höhenkosten.** Nach unten sieht man weit, nach oben fast nichts — nur Bergumrisse im Nebel. Für die Simulations-Detaillierung zählen Wegkosten, grob je Voronoi-Zelle. Das gilt für Objekte wie für Stadteinzugsgebiete. **Spielseitig, Langzeitziel** — im Karteneditor derzeit ohne Bedeutung.

### 5. Die Zeit

Jahreszeiten sind **gesetzt**. Jahre sind erwünscht, Altern ist offen. Die Zeit
vergeht langsam — Größenordnung eine Stunde je Spieljahr.

**Im Editor sichtbar sind die jahreszeitlichen Klimagrößen** — Temperatur,
Wind, Feuchte — und was daraus folgt: Schnee, Trockenheit, Seeeis.
**Biome und Geologie ändern sich nicht.**

### 6. Rohstoffe

Die Kette ist **Biome → biotische Stoffe** (Holz, Wild, Fisch, Weide,
Ackerfrucht, Kräuter) und **geologische Schichten → mineralische Stoffe**
(Stein, Erz, Ton, Salz, Torf). Regionen binden Stoffe **mittelbar**, indem sie
bestimmen, welche Biome und Schichten dort vorkommen — nicht durch eine
Zuweisung je Region.

Vielfalt ist erwünscht, hat aber ihren Preis in der Stabilität der Simulation.

> **✅ ENTSCHIEDEN — [#26](https://github.com/nikolaisoerensen/MapGenerator/issues/26):** Das wird **gemeinsam erarbeitet**, in vier Runden: erst die Kategorien, dann die Stoffe je Kategorie — abgeleitet aus dem, was die neun Regionen, die Biome und die Geologie tatsächlich hergeben —, dann die Fertigungszweige je Stoff, zuletzt der Lückenabgleich von den herzustellenden Gegenständen rückwärts. Das bestehende Godot-Projekt liefert die Vorlage. Siehe [Rohstoffkategorien, Stoffe und Fertigungszweige](https://github.com/nikolaisoerensen/MapGenerator/issues/69).
> **✅ ENTSCHIEDEN — [#26](https://github.com/nikolaisoerensen/MapGenerator/issues/26):** **Die Tabelle zuerst**, eine Karte nur dort, wo die Tabelle nicht ausreicht. Eine Dichtekarte je Stoff ist teuer und für die meisten Stoffe überflüssig.
> **✅ ENTSCHIEDEN — [#26](https://github.com/nikolaisoerensen/MapGenerator/issues/26):** **Je Bildpunkt.** Eine Berufsstätte greift über ein **Einzugsgebiet** darauf zu, nicht über die Parzelle, auf der sie steht.
> **✅ ENTSCHIEDEN — [#26](https://github.com/nikolaisoerensen/MapGenerator/issues/26):** **Erst nachsehen, dann entscheiden** — siehe [Gibt die Geologie eine Erzwahrscheinlichkeit her](https://github.com/nikolaisoerensen/MapGenerator/issues/68). Gebraucht wird je Ort eine **Wahrscheinlichkeitsverteilung**, keine feste Zuweisung: niemand weiß vorher genau, was unter einer Mine liegt. Man leitet es aus dem Geförderten ab, abhängig von Können und Glück; Prospektion und Wirtshausgerüchte grenzen nur grob ein. Eine Mine ist teuer und kein Anfangsgebäude — man beginnt mit einer Lehmgrube, später vielleicht einem Steinbruch. Die Mechanik selbst ist **spielseitig**.

### 7. Eignungsfelder: viele Felder, eine Rechnung

Orte entstehen aus **Eignungsfeldern** — für Städte, Wegrandtavernen,
Landmarken, und künftig auch für die Orte, an denen der Aberglaube wahr wird.
Teuer ist nicht das Feld, teuer sind **mehrere getrennte Geländeanalysen**, die
auseinanderlaufen. Deshalb gilt: **gemeinsame Grundfaktoren, einmal gerechnet;
jeder Typ ist eine kurze Gewichtung darüber.**

Das Programm trifft dieses Muster heute schon bei den Stadttypen — fünf Typen
aus vier gemeinsamen Grundfaktoren, mit der Begründung im Code: *„Eine zweite,
eigene Geländeanalyse wäre eine zweite Wahrheit."*

**Mystische Orte:** der Generator liefert **Eignung** (alte Wälder, Höhlen,
Moore, Abgelegenheit), die Auswahl trifft das Spiel. So liegen die dunklen Orte
nicht in jedem Durchgang gleich.

> **✅ ENTSCHIEDEN — [#19](https://github.com/nikolaisoerensen/MapGenerator/issues/19):** **Offen geblieben** und nicht entschieden — die Frage nach dem Register wurde im Durchgang nicht gestellt. Sie kommt in den nächsten Fragebogen.
> **✅ ENTSCHIEDEN — [#19](https://github.com/nikolaisoerensen/MapGenerator/issues/19):** **Ja, die Kante wird gezogen.** Die Eignungsrechnung bekommt die Biomkarte, und zwar so, dass sich die Eignungswerte messbar ändern — eine Kante ohne Wirkung sieht wie Erfolg aus und ist keiner. Siehe [Biomkarte in die Siedlungs-Eignungsrechnung einhängen](https://github.com/nikolaisoerensen/MapGenerator/issues/34).

### 8. Stehende Regeln, die weiter gelten

* **Was in 2D sichtbar ist, wird in derselben Änderung auch in 3D gebaut.**
  Nicht „erst 2D, 3D später" — später kommt nicht.
* **Jeder stille Rückfall auf einen Ersatzpfad braucht eine laute Logzeile.**
  Diese Fehlerklasse ist hier nachweislich fünfmal durchgekommen; jedes Mal
  lieferte der Ersatzpfad ein plausibles Ergebnis.

### 9. Gelände und Geologie

> **✅ ENTSCHIEDEN — [#17](https://github.com/nikolaisoerensen/MapGenerator/issues/17):** **Offen geblieben** — die Frage nach Landschaftsbeschreibungen statt Rauschparametern wurde im Durchgang nicht entschieden. Sie kommt in den nächsten Fragebogen.
> **✅ ENTSCHIEDEN — [#17](https://github.com/nikolaisoerensen/MapGenerator/issues/17):** **In eine Datei** (TOML oder JSON), nicht in die Oberfläche. Die Oberfläche zeigt sie **nur lesend**. Die Werte selbst dürfen sich dabei nicht ändern. Siehe [Die neun Regionsparametersätze in eine Datei](https://github.com/nikolaisoerensen/MapGenerator/issues/29).
> **✅ ENTSCHIEDEN — [#17](https://github.com/nikolaisoerensen/MapGenerator/issues/17):** **Parität ist Pflicht.** Der Faktor 385 zwischen GPU und CPU wird gefunden und behoben. Zeigt die Messung aber, dass die CPU-Erosion unbrauchbar langsam ist — sechs Minuten gegen fünfundzwanzig Sekunden —, wird sie **laut stillgelegt** statt repariert. Erst messen, dann entscheiden. Siehe [CPU-Erosion messen und die Konsequenz ziehen](https://github.com/nikolaisoerensen/MapGenerator/issues/30).
> **✅ ENTSCHIEDEN — [#17](https://github.com/nikolaisoerensen/MapGenerator/issues/17):** **Versehen.** Beide werden eingebaut — und nach der stehenden Regel in derselben Änderung für 2D **und** 3D. Siehe [Zwei unerreichbare Gelände-Ansichten sichtbar machen](https://github.com/nikolaisoerensen/MapGenerator/issues/31).
> **✅ ENTSCHIEDEN — [#17](https://github.com/nikolaisoerensen/MapGenerator/issues/17):** **Nur was wiedererkennbares Gelände erzeugt**: Küstenarchetypen, Hinterland, Flussnetz und Erosion. Der Geologie-Querschnitt und das adaptive 3D-Netz sind Anzeige und Rechenersparnis, kein Sollbild.

### 10. Wetter und Wasser

> **✅ ENTSCHIEDEN — [#18](https://github.com/nikolaisoerensen/MapGenerator/issues/18):** **Offen geblieben** — ob die Atmosphäre dreischichtig bleibt, wurde im Durchgang nicht entschieden. Sie kommt in den nächsten Fragebogen.
> **✅ ENTSCHIEDEN — [#18](https://github.com/nikolaisoerensen/MapGenerator/issues/18):** **Erst nachmessen, dann entscheiden.** Bei 256, 512 und 1024 Bildpunkten, drei Seeds, je Region. Erst wenn die Zahl steht, ist klar, ob es ein Widerspruch oder eine überholte Messung war. Siehe [Seenfläche neu messen](https://github.com/nikolaisoerensen/MapGenerator/issues/32).
> **✅ ENTSCHIEDEN — [#18](https://github.com/nikolaisoerensen/MapGenerator/issues/18):** **Ja, mit einem Toleranzband von zehn Prozent.** Wichtig ist nur, dass es nicht ausufert; auf zwei Prozent genau muss es nicht sein. Die Bandgrenze steht versioniert im Repo, nicht im Testcode. Siehe [Wasserbilanz mit Toleranzband von zehn Prozent zusichern](https://github.com/nikolaisoerensen/MapGenerator/issues/70).
> **✅ ENTSCHIEDEN — [#18](https://github.com/nikolaisoerensen/MapGenerator/issues/18):** **Ja, echte Ziele** — und sie bekommen endlich eine Kennzahl: die **Sinuosität**, Lauflänge geteilt durch Luftlinie. Erwartet wird, dass Flachlandflüsse stärker mäandern als Gebirgsflüsse; das wird gemessen, nicht angenommen. Siehe [Sinuosität als Kennzahl für Mäander](https://github.com/nikolaisoerensen/MapGenerator/issues/33).

### 11. Biome und Siedlungen

> **✅ ENTSCHIEDEN — [#19](https://github.com/nikolaisoerensen/MapGenerator/issues/19):** **Eine feste Feldliste, und nur die Außenform.** Der Editor liefert die **Kontur der Stadtgrenze**, die **Anschlusspunkte der Wege**, sowie **Größe, Typ, Rang und Kultur**. Mehr nicht. Alles, was nicht in dieser Liste steht, ist Innenleben des Editors und darf sich ändern, ohne das Spiel zu brechen. Siehe [Die Feldliste der Siedlungsnaht festschreiben](https://github.com/nikolaisoerensen/MapGenerator/issues/35).
> **✅ ENTSCHIEDEN — [#19](https://github.com/nikolaisoerensen/MapGenerator/issues/19):** **Im Spiel.** Man klickt eine Stadt im Simulator an und sieht die Parzelle als Bild — gezeichnet aus Stadtgrenze, Weganschlüssen, Größe, Typ und **Kultur**. Zuerst nur Straßennetze und Häuserformen, später Symbole für das, was darin steckt (Schneider und dergleichen).
> **✅ ENTSCHIEDEN — [#19](https://github.com/nikolaisoerensen/MapGenerator/issues/19):** **Ja, als Werkzeug** — nicht als Programmteil des Editors.
> **✅ ENTSCHIEDEN — [#19](https://github.com/nikolaisoerensen/MapGenerator/issues/19):** **Arbeitsbenennungen.** Die Völker sind an echte Kulturen angelehnt und bekommen im Spiel vielleicht eigene Namen; derzeit reichen die Platzhalter oder die echten Namen.

### 12. Anzeige, Bedienung und Export

> **✅ ENTSCHIEDEN — [#20](https://github.com/nikolaisoerensen/MapGenerator/issues/20):** **Offen geblieben** — ob 3D Haupt- oder Kontrollansicht ist, wurde im Durchgang nicht entschieden. Es kommt in den nächsten Fragebogen. Unberührt gilt weiter die stehende Regel: was in 2D sichtbar ist, wird in derselben Änderung auch in 3D gebaut.
> **✅ ENTSCHIEDEN — [#20](https://github.com/nikolaisoerensen/MapGenerator/issues/20):** **`OFFENE_PUNKTE.md` hat recht, die README ist veraltet** — es sind zwei verschiedene Dinge mit demselben Namen. Das **Anzeige-LOD des Editors** wird nicht mehr verwendet und wird entfernt. Das **Simulations-LOD des Spiels** — entfernte Ereignisse gröber rechnen, Einheiten erst bei Annäherung erzeugen, und das ohne Ruckeln, also stufenweise nachladen — ist ein **Ziel des Spiels** und noch nicht geplant. Siehe [Das alte Anzeige-LOD entfernen und die README berichtigen](https://github.com/nikolaisoerensen/MapGenerator/issues/36).
> **✅ ENTSCHIEDEN — [#20](https://github.com/nikolaisoerensen/MapGenerator/issues/20):** **Scharfe Kanten aus den Vektordaten, das Raster bleibt grob.** Das gilt auch für das Mesh. Wird es an einer Stelle zu grob, kann dort später um einen Faktor vier nachgeschärft werden — aber erst, wenn es so weit ist.
> **✅ ENTSCHIEDEN — [#20](https://github.com/nikolaisoerensen/MapGenerator/issues/20):** **Ja, 2048 trägt.** Die Feinheit kommt aus Vektoren und Texturen, nicht aus mehr Bildpunkten.
> **✅ ENTSCHIEDEN — [#20](https://github.com/nikolaisoerensen/MapGenerator/issues/20):** **Ja.** Der Knotengraph wird behalten und als **Linienzüge** exportiert, mit Flussordnung und Breite je Abschnitt. Ein Fluss, der nur als Raster existiert, ist im Spiel eine Treppe. Siehe [Flussnetz als Linienzüge exportieren](https://github.com/nikolaisoerensen/MapGenerator/issues/37).

---

## Testing Decisions

### Was einen guten Test hier ausmacht

Die Lehren dieses Projekts, alle teuer erkauft:

* **Das Ergebnis prüfen, nicht den Parameter.** Zehn grüne Tests verdeckten
  wochenlang eine Funktion, die im Betrieb nie aufgerufen wurde.
* **Mit den echten Eingabegrößen bauen** — 128/256/512/1024. Eine ausgedachte
  Größe prüft eine ausgedachte Situation. Genau daran scheiterte die Prüfung
  des adaptiven Netzes: gebaut mit 129/257/513, während das Programm
  Zweierpotenzen benutzt.
* **Sind mehrere Einzelprüfungen grün und das Ergebnis trotzdem falsch:
  aufhören, Einzelglieder zu prüfen — zwei Enden der Kette gegeneinander
  messen.** Das fand den Verteilungsfehler in einem Schritt, nachdem sechs
  Einzelhypothesen ergebnislos geblieben waren.
* **Über die Naht prüfen, nicht daran vorbei.** Was ein Test nur erreicht,
  indem er ins Innere greift, hat vermutlich die falsche Form.

### Der Qualitätsmaßstab

**Erdähnlichkeit**, mit echten Orten als Vorbild — Cork, Bergen, Wologda,
La Rochelle, Chur, Bamberg, Madrid, Rom, Iraklio. Aber: Höhen sind gestaucht
und Regionen kleiner als ihre Vorbilder. **Das Ziel ist Wiedererkennbarkeit,
nicht Maßstabstreue.**

**Was bereits trägt:** das Klima trifft seine realen Vorbilder auf 1 K, über
jeden Seed, per Konstruktion statt per Handeichung. Das ist das Muster, dem
der Rest folgen soll.

**Ein Vorschlag, der auf seine Prüfung wartet:** Wiedererkennbarkeit ist nicht
nur optisch, sondern **materiell** — eine Region muss etwas anderes hergeben,
sonst lohnt keine Reise und kein Rezept hat einen Heimatort. Das ist messbar:
*Hat jede der neun Regionen mindestens einen Stoff, den keine andere in
nennenswerter Menge hergibt?* Nicht vom Generator erzwungen, sondern am
Ergebnis gemessen — zwei Enden gegeneinander.

> **✅ ENTSCHIEDEN — [#16](https://github.com/nikolaisoerensen/MapGenerator/issues/16):** **Offen geblieben** — wo der Stauchungsfaktor steht, wurde im Durchgang nicht geklärt. Er wird beim Verschmelzen der Dokumente gesucht; siehe [Spezifikation und Sollbeschreibung zu einer Datei verschmelzen](https://github.com/nikolaisoerensen/MapGenerator/issues/44).
> **✅ ENTSCHIEDEN — [#16](https://github.com/nikolaisoerensen/MapGenerator/issues/16):** **Beides.** Weite absolute Bänder als Leitplanke — damit nichts völlig entgleist —, **Verhältnisse als eigentliche Bewertung**: ob Gebirge zu Flachland, Tal zu Grat, Wasser zu Land im richtigen Verhältnis stehen. Ein Land ist nicht daran wiedererkennbar, dass ein Berg 2400 m hoch ist, sondern daran, dass er dreimal so hoch ist wie der Hügel davor.
> **✅ ENTSCHIEDEN — [#16](https://github.com/nikolaisoerensen/MapGenerator/issues/16):** **Erst messen, was wirklich beschreibt.** Die Liste wird nicht ausgedacht, sondern erarbeitet: welche Kennzahlen eine Landschaft tatsächlich unterscheidbar machen und welche davon aus dem Perlin-Rauschen ohnehin schon herausfallen. Siehe [Welche Kennzahlen eine Landschaft wirklich beschreiben](https://github.com/nikolaisoerensen/MapGenerator/issues/66).
> **✅ ENTSCHIEDEN — [#16](https://github.com/nikolaisoerensen/MapGenerator/issues/16):** **Echte Geländedaten, je ein Vorbild pro Region** — SRTM oder Copernicus, dazu wo möglich geologische Karten, Baumbestände und Biomkarten. Langfristig etwa zehn Orte je Region, damit die Kennzahl eine Streuung hat und nicht an einem Einzelfall hängt. Siehe [Reale Vorbildorte je Region beschaffen](https://github.com/nikolaisoerensen/MapGenerator/issues/67).
> **✅ ENTSCHIEDEN — [#16](https://github.com/nikolaisoerensen/MapGenerator/issues/16):** **Auf die neun Regionen umschreiben.** Die vier alten Geländetypen und die leere Zwanzig-Landschaften-Liste stammen aus der Zeit vor der Regionseinteilung. Siehe [Spezifikation Abschnitt 2 auf die neun Regionen umschreiben](https://github.com/nikolaisoerensen/MapGenerator/issues/43).

### Das Prüfumfeld

Die Ausgangslage: 70 Testdateien, 19 Minuten Gesamtlaufzeit, 60 grün. Die
Erzeugung ist seed-abhängig, teils auf der GPU, und die wertvollste Prüfung ist
bis heute der Blick des Nutzers auf den Bildschirm.

> **✅ ENTSCHIEDEN — [#22](https://github.com/nikolaisoerensen/MapGenerator/issues/22):** **Erst den Bestand sichten.** 70 Testdateien liegen da; welche veraltet sind, welche an der falschen Naht ansetzen und welche fehlen, wird einmal durchgegangen und bewertet, bevor neue dazukommen. Es braucht deutlich mehr Tests als heute — aber nicht mehr von denselben. Siehe [Den Testbestand sichten und bewerten](https://github.com/nikolaisoerensen/MapGenerator/issues/45).
> **✅ ENTSCHIEDEN — [#22](https://github.com/nikolaisoerensen/MapGenerator/issues/22):** **Zwei Ränge.** **Wächter** laufen unter zwei Minuten und gehören zu jeder Änderung; die **Eichung** darf nachts eine Stunde brauchen. Heute sind es 19 Minuten für alles — das ist für keinen der beiden Zwecke die richtige Zahl. Siehe [Testläufe in Wächter und Eichung trennen](https://github.com/nikolaisoerensen/MapGenerator/issues/46).
> **✅ ENTSCHIEDEN — [#22](https://github.com/nikolaisoerensen/MapGenerator/issues/22):** **Als Band, nicht als Punkt.** Eine Zusicherung lautet „der Wasseranteil liegt zwischen 4 und 9 Prozent“, nicht „ist 6,5“. Die Bandgrenzen stehen als **versionierte Daten** im Repo, nicht verstreut im Testcode — dann ist an der Dateihistorie ablesbar, wann ein Ziel verschoben wurde und warum. Siehe [Bandgrenzen als versionierte Daten](https://github.com/nikolaisoerensen/MapGenerator/issues/47).
> **✅ ENTSCHIEDEN — [#22](https://github.com/nikolaisoerensen/MapGenerator/issues/22):** **Fester Seed für die Wächter, drei wechselnde für die nächtliche Eichung.** Der feste Seed macht Fehlschläge reproduzierbar; die wechselnden fangen das ab, was nur bei genau diesem einen Seed zufällig gut aussieht. Siehe [Seedführung für Wächter und Eichung festlegen](https://github.com/nikolaisoerensen/MapGenerator/issues/48).
> **✅ ENTSCHIEDEN — [#22](https://github.com/nikolaisoerensen/MapGenerator/issues/22):** **Je ein Ticket mit Frist.** Kein Sammelposten — jeder der zehn roten Tests bekommt eine eigene Zeile, eine eigene Ursache und ein eigenes Datum. Bis dahin ist er ausdrücklich als bekannt markiert, danach ist er ein Fehler. Siehe [Die zehn roten Tests einzeln aufschlüsseln](https://github.com/nikolaisoerensen/MapGenerator/issues/49).
> **✅ ENTSCHIEDEN — [#22](https://github.com/nikolaisoerensen/MapGenerator/issues/22):** **Bildvergleich mit Toleranz.** Fester Seed, fester Ausschnitt, hinterlegtes Referenzbild, Abweichung in Prozent. Das ersetzt die Sichtprüfung nicht vollständig, grenzt sie aber auf das ein, was wirklich neu aussieht. Siehe [Bildvergleich mit Referenzbildern aufsetzen](https://github.com/nikolaisoerensen/MapGenerator/issues/50).

### Der nächtliche Betrieb

> **✅ ENTSCHIEDEN — [#23](https://github.com/nikolaisoerensen/MapGenerator/issues/23):** **Alle Abnahmekriterien des Tickets sind als Test formuliert und grün.** Ein Ticket, dessen Kriterien man nicht als Test schreiben kann, ist kein Nachtticket.
> **✅ ENTSCHIEDEN — [#23](https://github.com/nikolaisoerensen/MapGenerator/issues/23):** **Daran, dass die Tests grün sind — nicht an der Einschätzung des Agenten.** „Fertig“ ist ein Messwert, keine Meinung.
> **✅ ENTSCHIEDEN — [#23](https://github.com/nikolaisoerensen/MapGenerator/issues/23):** **Eine Sperrliste im Repo.** Dateien und Themen, an die nachts niemand geht. Siehe [Sperrliste für den Nachtbetrieb anlegen](https://github.com/nikolaisoerensen/MapGenerator/issues/57).
> **✅ ENTSCHIEDEN — [#23](https://github.com/nikolaisoerensen/MapGenerator/issues/23):** **Alles auf einen Nacht-Branch, morgens ein Merge.** Ein Branch ist eine abzweigende Arbeitskopie des Projekts: nachts wird nur dort geschrieben, `main` bleibt unberührt. Morgens wird gegengelesen — auch mit Code-Review — und einzelne Fehler werden korrigiert, bevor zusammengeführt wird. Siehe [Das Nacht-Branch-Verfahren aufsetzen](https://github.com/nikolaisoerensen/MapGenerator/issues/58).
> **✅ ENTSCHIEDEN — [#23](https://github.com/nikolaisoerensen/MapGenerator/issues/23):** **Zeitgrenze je Ticket.** Läuft sie ab, bleibt der Branch stehen und eine Notiz geht ins Ticket: was versucht wurde, woran es hing. Kein Agent arbeitet sich stundenlang an derselben Stelle fest. Siehe [Zeitgrenze und Steckenbleib-Notiz je Ticket](https://github.com/nikolaisoerensen/MapGenerator/issues/59).
> **✅ ENTSCHIEDEN — [#23](https://github.com/nikolaisoerensen/MapGenerator/issues/23):** **Ein Morgenbericht auf einer Seite:** was geschlossen wurde, was rot ist, welche Kennzahlen sich bewegt haben, welche stillen Rückfälle gemeldet wurden. Siehe [Der Morgenbericht auf einer Seite](https://github.com/nikolaisoerensen/MapGenerator/issues/60).

---

## Out of Scope

**Für dieses Dokument:**

* **Das Spiel selbst.** Hier steht, was der Karteneditor liefert — nicht, wie
  daraus ein Spiel wird.
* **Die Inhalte der Simulation.** Bedürfnisse, Preisbildung, Handelsregeln,
  Ereignisse, Erzählung. Nach der Zeitpunkt-null-Regel gehört das in ein
  eigenes Dokument, sobald die Simulation entsteht.
* **Die neun Regionen neu zuschneiden oder ihre Zahl ändern.** Vom Nutzer als
  fest erklärt.
* **Das Overlay-Bündel** ([#4](https://github.com/nikolaisoerensen/MapGenerator/issues/4) bis [#12](https://github.com/nikolaisoerensen/MapGenerator/issues/12)). Eigene Spezifikation in `docs/SPEC_OVERLAYS.md`, läuft unabhängig weiter.
* **Die Arbeitstickets aus dem Ist/Soll-Vergleich.** Sie entstehen, wenn
  dieses Dokument fertig ist.

**Noch nicht entschieden, aber bewusst nicht hier:** wann die Simulation
tatsächlich in ein eigenes Programm umzieht. Die Trennlinie ist festgelegt, der
Umzug ist es nicht.

---

## Further Notes

**Wie dieses Dokument wächst.** Über die Karte
[#13](https://github.com/nikolaisoerensen/MapGenerator/issues/13), Ticket für
Ticket. Die empfohlene Folge:

```
#16 Maßstab ──┬─→ #17 Gelände/Geologie   ─┐
              ├─→ #18 Wetter/Wasser       ├─→ #21 Sollbeschreibung ─┬─→ #22 TDD → #23 Nacht
              ├─→ #19 Biome/Siedlungen ─┬─┤                         ├─→ #24 Dokumente
              └─→ #20 Anzeige/Export    │ │                         └─→ #25 Architektur
                                        ├─┤
                        #26 Rohstoffe ──┘ │
                     #27 Sichtbarkeit ────┘
```

**Was bereits geschlossen ist:** [#14](https://github.com/nikolaisoerensen/MapGenerator/issues/14) (Zweck und Zielbild) und
[#15](https://github.com/nikolaisoerensen/MapGenerator/issues/15) (Ist-Stand-Inventar).

**Das Inventar in Zahlen** (2026-09-14, aus dem Code erhoben, nicht aus der
Dokumentation): 12 Reiter · 112 sichtbare Regler, davon 18 gesperrt ·
15 Erzeugungs- und 24 Anzeige-Häkchen · 60 auswählbare Anzeigemodi ·
38 Rechenknoten mit 73 eindeutigen Ausgabeschlüsseln · bis zu 21
Exportdateien.

**Acht Widersprüche zwischen Code und Dokumentation** sind dabei gefunden
worden. Sie sind der Anlass dieses Dokuments und stehen vollständig an #15.
Die drei schwersten: die Erosion läuft, obwohl vier Dokumente das Gegenteil
sagen; zwei Ansichten des Gelände-Reiters sind unerreichbar, obwohl das
Sitzungslog sie als fertig meldet; sechs Regler sind mit der Begründung
gesperrt, sie stünden in keinem Reiter — zwei davon stehen in einem.
