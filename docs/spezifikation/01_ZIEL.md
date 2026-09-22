# Ziel, Zweck und Grenzen des Programms

**Was das hier ist.** Wofür der MapGenerator da ist, welche Regel entscheidet,
was hineingehört, und wo seine Zuständigkeit endet. Wer eine Einzelentscheidung
zu treffen hat und nicht weiß, woran er sie messen soll, liest zuerst diese
Datei.

*Herkunft: `docs/archiv/2026-07-29_SPEZIFIKATION.md` (Stand 2026-07-29), §1
„Oberziel", und `docs/SOLLBESCHREIBUNG.md` (Stand 2026-09-16), Abschnitte
„Solution", „Implementation Decisions" 1–7 und „Out of Scope".*

---

## 1. Der Zweck

Der MapGenerator ist der **Karteneditor für ein Spiel im Stil von Die Gilde 2,
nur besser** — und darüber hinaus die **Testwiese**, auf der
Simulationsfunktionen ausprobiert werden. Der Nutzer erzeugt eine Welt, und
wenn sie ihm gefällt, startet er auf Knopfdruck die Simulation. Im fertigen
Spiel beginnt an derselben Stelle das Spiel.

**„Nur besser" heißt:** ein Mittelalter, das halbwegs glaubwürdig ist — in dem
aber alles wahr ist, woran die Menschen damals glaubten. Eine dunkle,
gefährliche Welt mit hoher Schwierigkeit und Rückschlägen. Neun Kulturen und
eine erzeugte Welt, die Tiefe gibt. Ein Erzählton wie in RimWorld.

*(Entschieden am 2026-09-14 über
[#14](https://github.com/nikolaisoerensen/MapGenerator/issues/14).)*

## 2. Das Oberziel der Erzeugung

**Reale Landschaften der Erde nachbilden**, an verschiedenen Orten. Ein Nutzer
stellt über wenige verständliche Regler ein, *welche* Landschaft er will, und
bekommt ein Ergebnis, das der echten Vorlage ähnlich sieht — in Form, Klima,
Gewässern und Bewuchs.

> **Aufgelöster Widerspruch.** Das Oberziel von 2026-07-29 lautete wörtlich
> „…an verschiedenen Orten, **ohne Meer**". Das gilt nicht mehr. Sechs der
> neun Regionen haben einen Wasseranteil größer null, Thalassia 65 %, und die
> Küstenarchetypen sind seit 2026-08-12 gebaut. Die Formulierung „ohne Meer"
> beschrieb nie das Programm, das entstanden ist; sie ist hier gestrichen.

Zwei Bedingungen, die das Oberziel mittragen:

**Kein Reglerstand darf die Welt zerstören.** Wenn eine Einstellung ein
unbrauchbares Ergebnis erzeugen kann, ist entweder ihr Bereich falsch oder sie
ist mit einer anderen Größe nicht verknüpft. Der Nutzer soll nicht wissen
müssen, wo Probleme aufschwingen.

**Die Regler sind Landschaftsbeschreibungen, keine Implementierungsgrößen.**
„Höhenunterschied" und „Geländecharakter" statt `redistribute_power` und
`capacity_kc`.

## 3. Der Leitsatz: die Zeitpunkt-null-Regel

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
Editor?" in einem Satz. Vom Nutzer am 2026-09-14 angenommen.

## 4. Drei Programme, eine Datei dazwischen

Der **Kartengenerator** schreibt die gebackene Welt. Eine **Simulation** liest
sie, rechnet Jahre und schreibt Zustände. Das **Spiel** liest beides. Der
Editor **startet** die Simulation auf Knopfdruck, **rechnet sie aber nicht**.

Begründung: Generator und Simulation haben unvereinbare Arbeitsschleifen. Ein
Generator wird abgenommen, indem man einmal rechnet und hinsieht; eine
Simulation, indem man sie fünfzigmal über zehn Jahre laufen lässt. In einem
Programm zahlt jedes Simulationsexperiment den vollen Weltaufbau mit.

Die Naht ist eine **Datei**, beide Seiten sind einzeln dagegen prüfbar, und
die Hälfte steht bereits: der Godot-Export ist genau diese Datei.
`tools/biome_lab/` ist der Beleg, dass das Muster hier trägt.

**Der Umzug ist entschieden, aber nicht fällig.** Solange nur Anfangszustände
entstehen, bleibt alles im Generator. Fällig wird die Trennung mit
Bedürfnissen und Handel.

## 5. Der Weltaufbau in drei Sätzen

* **Neun Regionen in einem 3×3-Raster**, mit je eigenem Parametersatz, eigener
  Kultur und einem realen Bezugsort. Die neun sind **fest**. Einzelheiten in
  [10_REGIONEN.md](10_REGIONEN.md).
* **Eine nahtlose Welt über die vollen 21 km**, kein Kartenwechsel je Region.
* **Festlegung statt Regelkreis** (Leitlinie seit 2026-08-07): Zielwerte
  vorgeben und das Feld daraus aufbauen, statt ein Gleichgewicht einschwingen
  zu lassen. Ein Regelkreis über 35 Schritte liefert bei 50 Schritten etwas
  anderes — genau das ist der Drift, der bei jeder Auflösungsänderung
  nachkalibriert werden musste.

## 6. Der Qualitätsmaßstab: Erdähnlichkeit

Gemessen wird gegen echte Orte — Cork, Bergen, Wologda, La Rochelle, Chur,
Bamberg, Madrid, Rom, Iraklio. Aber: Höhen sind gestaucht und Regionen kleiner
als ihre Vorbilder. **Das Ziel ist Wiedererkennbarkeit, nicht Maßstabstreue.**

Daraus folgen drei Festlegungen
([#16](https://github.com/nikolaisoerensen/MapGenerator/issues/16)):

* **Verhältnisse sind die eigentliche Bewertung**, absolute Bänder nur die
  Leitplanke. Ein Land ist nicht daran wiedererkennbar, dass ein Berg 2400 m
  hoch ist, sondern daran, dass er dreimal so hoch ist wie der Hügel davor.
* **Welche Kennzahlen eine Landschaft beschreiben, wird gemessen, nicht
  ausgedacht** ([#66](https://github.com/nikolaisoerensen/MapGenerator/issues/66)).
* **Die Vorbilder sind echte Geländedaten** — SRTM oder Copernicus, dazu wo
  möglich geologische Karten, Baumbestände und Biomkarten; langfristig etwa
  zehn Orte je Region, damit die Kennzahl eine Streuung hat
  ([#67](https://github.com/nikolaisoerensen/MapGenerator/issues/67)).

**Was bereits trägt:** das Klima trifft seine realen Vorbilder auf 1 K, über
jeden Seed, per Konstruktion statt per Handeichung. Das ist das Muster, dem
der Rest folgen soll.

**Ein Vorschlag, der auf seine Prüfung wartet:** Wiedererkennbarkeit ist nicht
nur optisch, sondern **materiell** — *hat jede der neun Regionen mindestens
einen Stoff, den keine andere in nennenswerter Menge hergibt?* Nicht vom
Generator erzwungen, sondern am Ergebnis gemessen.

> **Offen geblieben:** wo der Stauchungsfaktor zwischen Vorbild und Karte
> steht, ist bis heute nicht festgeschrieben.

## 7. Die Zeit

Jahreszeiten sind **gesetzt**. Jahre sind erwünscht, Altern ist offen. Die
Zeit vergeht langsam — Größenordnung eine Stunde je Spieljahr.

**Im Editor sichtbar sind die jahreszeitlichen Klimagrößen** — Temperatur,
Wind, Feuchte — und was daraus folgt: Schnee, Trockenheit, Seeeis.
**Biome und Geologie ändern sich nicht.**

## 8. Rohstoffe

Die Kette ist **Biome → biotische Stoffe** (Holz, Wild, Fisch, Weide,
Ackerfrucht, Kräuter) und **geologische Schichten → mineralische Stoffe**
(Stein, Erz, Ton, Salz, Torf). Regionen binden Stoffe **mittelbar**, indem sie
bestimmen, welche Biome und Schichten dort vorkommen — nicht durch eine
Zuweisung je Region.

Festlegungen
([#26](https://github.com/nikolaisoerensen/MapGenerator/issues/26)):

* **Die Tabelle zuerst**, eine Dichtekarte nur dort, wo die Tabelle nicht
  ausreicht.
* **Je Bildpunkt.** Eine Berufsstätte greift über ein **Einzugsgebiet** darauf
  zu, nicht über die Parzelle, auf der sie steht.
* **Erz ist eine Wahrscheinlichkeitsverteilung, keine feste Zuweisung** —
  niemand weiß vorher genau, was unter einer Mine liegt. Die Mechanik selbst
  ist spielseitig
  ([#68](https://github.com/nikolaisoerensen/MapGenerator/issues/68)).
* Die Stoffliste wird in vier Runden erarbeitet, nicht ausgedacht
  ([#69](https://github.com/nikolaisoerensen/MapGenerator/issues/69)).

## 9. Eignungsfelder: viele Felder, eine Rechnung

Orte entstehen aus **Eignungsfeldern** — für Städte, Wegrandtavernen,
Landmarken, und künftig auch für die Orte, an denen der Aberglaube wahr wird.
Teuer ist nicht das Feld, teuer sind **mehrere getrennte Geländeanalysen**,
die auseinanderlaufen. Deshalb gilt: **gemeinsame Grundfaktoren, einmal
gerechnet; jeder Typ ist eine kurze Gewichtung darüber.**

Das Programm trifft dieses Muster heute schon bei den Stadttypen — fünf Typen
aus vier gemeinsamen Grundfaktoren, mit der Begründung im Code: *„Eine zweite,
eigene Geländeanalyse wäre eine zweite Wahrheit."* Einzelheiten in
[14_SIEDLUNGEN.md](14_SIEDLUNGEN.md).

**Mystische Orte:** der Generator liefert **Eignung** (alte Wälder, Höhlen,
Moore, Abgelegenheit), die Auswahl trifft das Spiel. So liegen die dunklen
Orte nicht in jedem Durchgang gleich.

## 10. Sichtbarkeit folgt dem Wegaufwand, nicht der Luftlinie

Die Welt gliedert sich in Stücke, die sich nach **Erreichbarkeit** bilden,
nicht nach Abstand. Der Rechenkern dafür existiert bereits: ein
Mehrquellen-Dijkstra über das Bildpunktgitter mit hangneigungsgewichteten
Schrittkosten, heute für Stadtgrenzen und Parzellen benutzt.

**Die Gliederung selbst bleibt spielseitig**
([#27](https://github.com/nikolaisoerensen/MapGenerator/issues/27)). Der
Editor liefert die Wegekosten, aus denen sie folgt. Drei Festlegungen dazu:

* **Nach Wegart gestaffelt** — Trampelpfad, Karrenweg und gepflasterte Straße
  kosten je Kilometer Unterschiedliches. Der Haken ist die Rückkopplung: ein
  Weg macht sich selbst schneller und zieht weitere Wege an sich.
* **Wasser verbindet längs und trennt quer.** Eine Querung ohne Brücke ist
  teuer; **steht die Brücke, ist sie billig.** Die Brücke ist ein Bauwerk im
  selben Regelkreis, kein fester Geländewert.
* **Sichtweite nicht in Metern, sondern in Höhenkosten.** Nach unten sieht man
  weit, nach oben fast nichts. Spielseitig, Langzeitziel.

## 11. Was NICHT hierher gehört

* **Das Spiel selbst.** Hier steht, was der Karteneditor liefert — nicht, wie
  daraus ein Spiel wird.
* **Die Inhalte der Simulation.** Bedürfnisse, Preisbildung, Handelsregeln,
  Ereignisse, Erzählung. Nach der Zeitpunkt-null-Regel gehört das in ein
  eigenes Dokument, sobald die Simulation entsteht.
* **Die neun Regionen neu zuschneiden oder ihre Zahl ändern.** Vom Nutzer als
  fest erklärt.
* **Der Ist-Zustand des Programms.** Was gebaut ist, steht in
  `docs/HANDBUCH.md`; was noch zu tun ist, in `docs/OFFENE_PUNKTE.md`.

**Noch nicht entschieden, aber bewusst nicht hier:** wann die Simulation
tatsächlich in ein eigenes Programm umzieht. Die Trennlinie ist festgelegt,
der Umzug ist es nicht.
