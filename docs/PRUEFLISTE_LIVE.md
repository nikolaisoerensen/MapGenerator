# Prüfliste am laufenden Programm — Stand 2026-08-27

Alles hier ist **headless nicht prüfbar** und braucht deinen Blick. Was
automatisch geprüft werden konnte, ist geprüft (siehe `docs/TESTBERICHT.md`).

Start:

    .venv/Scripts/python.exe main.py

---

# TEIL A — der neue Ablauf Regionen → Kontinent → Flüsse

Das ist die Wochenvorgabe vom 2026-08-26. Die drei Reiter stehen jetzt **vor**
Terrain, in genau dieser Reihenfolge. Bitte in dieser Reihenfolge durchgehen —
jeder baut auf dem vorigen auf.

## A.0 Zuerst: sind die Reiter überhaupt richtig einsortiert?

Beim letzten Anlauf war *"oben der Reiter doppelt und jeder nächste Reiter
verschoben"*. Ursache war ein Reiter, der beim Bauen abstürzte, **nachdem** er
schon in die Leiste eingetragen war.

| # | Was ansehen | Was richtig ist |
|---|---|---|
| A.0.1 | Die Reiterleiste ganz oben | **Regionen, Kontinent, Flussnetzwerk, Terrain**, dann der bisherige Rest |
| A.0.2 | Jeden Reiter einmal anklicken | Jeder zeigt seinen **eigenen** Inhalt, keiner ist doppelt, keiner leer |
| A.0.3 | Die Konsole beim Start | Kein `AttributeError`, kein `viewport_widget` |

`tests/smoke_test_reiter_vertrag.py` prüft das jetzt bis zur fertigen Leiste
durchgezählt — wenn hier trotzdem etwas verrutscht ist, ist das ein Befund,
der in den Test gehört.

## A.1 Reiter „Regionen"

Dropdown mit den neun Regionen, fünf Geländeregler, vier Erosionsregler,
Vorschau 256 × 160 px.

| # | Was ansehen | Was richtig ist | Was schiefgehen kann |
|---|---|---|---|
| A.1.1 | Region im Dropdown wechseln | Bild **und** Regler springen auf die Katalogwerte der neuen Region | Bleiben die Regler stehen, ist das Nachführen kaputt |
| A.1.2 | *Mittlere Höhe* ziehen | Das Bild wandert als Ganzes nach oben/unten | |
| A.1.3 | *Relief* ziehen | Die Höhenspanne wächst, der Mittelwert bleibt | Wandert der Mittelwert mit, ist die medianerhaltende Kurve verstellt |
| A.1.4 | *Formgröße* ziehen | Die Landschaftsformen werden großzügiger bzw. kleinteiliger | |
| A.1.5 | *Rauheit* ziehen | Feine Struktur kommt dazu, die Großform bleibt | |
| A.1.6 | *Höhenverteilung* ziehen | Links Hochebenen mit Einschnitten, rechts Gipfel über einer Ebene | |
| A.1.7 | Häkchen **Küstentypen zeigen** | Es erscheint eine Küste im Bild | Kostet 0,2–1,0 s je Reglerzug — spürbare Zähigkeit ist erwartet, nicht falsch |
| A.1.8 | **Auf Katalogwerte zurücksetzen** | Alles steht wieder wie beim Öffnen | |
| A.1.9 | **Der eigentliche Punkt:** Regler verstellen, dann *Generieren* auf der großen Karte | Die große Karte zeigt die Änderung | Bis zum 2026-08-26 kam hier **nichts** an — die Einstellung lebte nur im Reiter. Jetzt bewacht von `smoke_test_regionsregler_wirken.py`, aber gesehen hat es noch niemand |

## A.2 Reiter „Kontinent"

| # | Was ansehen | Was richtig ist | Was schiefgehen kann |
|---|---|---|---|
| A.2.1 | Beim Öffnen | Häkchen *Form selbst bestimmen* ist **aus**, es gilt die bisherige Form | Ist es an, ändert sich jede bestehende Karte |
| A.2.2 | Häkchen an, Regler nach **links** | Ein kompakter, runder Kontinent | |
| A.2.3 | Regler nach **rechts** | Viele Arme und Halbinseln | |
| A.2.4 | Regler in die **Mitte** | Länglich gestreckt | |
| A.2.5 | Beim Ziehen auf den **Landanteil** in der Statistik schauen | Er bleibt weitgehend gleich | Springt er stark, greift die Flächenregelung nicht, und alle Regionsflächen verschieben sich mit |
| A.2.6 | Häkchen **Regionen einfärben** | Die neun Gebiete werden sichtbar | |
| A.2.7 | Form einstellen, dann *Generieren* | Die große Karte hat diese Form | Dasselbe Anschlussrisiko wie A.1.9 |

## A.3 Reiter „Flussnetzwerk"

Fünf Regler (Talabstand, Talbreite, Taltiefe, Talform, Flüsse folgen dem
Tiefland) und **Live-Vorschau (128 px)**.

| # | Was ansehen | Was richtig ist | Was schiefgehen kann |
|---|---|---|---|
| A.3.1 | Beim Öffnen | Die Vorschau ist **aus** | Sie kostet beim ersten Einschalten knapp 3 s — das ist so gewollt, nicht beim Start |
| A.3.2 | Häkchen an | Nach ~3 s ein Gelände mit eingeschnittenen Tälern | |
| A.3.3 | Danach einen Regler ziehen | Neues Bild nach **unter 1 s** (gemessen 0,67 s) | Dauert es jedes Mal 3 s, greift der Zwischenspeicher des Grundgeländes nicht |
| A.3.4 | *Talbreite* von links nach rechts | Breite, weiche Täler statt schmaler Kerben | |
| A.3.5 | *Talform* von links nach rechts | V-Kerbe wird zum U-Trog | Dieser Regler war bis zum 2026-08-26 **wirkungslos** — das regionale Feld hat ihn überschrieben |
| A.3.6 | **Auf 3D umschalten**, während die Vorschau an ist | Dasselbe Gelände in 3D | Die Vorschau geht bewusst über den gewöhnlichen Anzeigeweg; wäre sie ein Sonderweg, bliebe 3D lautlos leer |
| A.3.7 | Häkchen wieder aus | Es erscheint wieder, was die Pipeline zuletzt gerechnet hat | |

## A.4 Die neuen Karten in **2D und 3D**

Terrain-Reiter, Karte generieren, dann jede dieser vier Anzeigen einmal in 2D
**und** einmal in 3D ansehen. Genau hier ist am 2026-08-25 dreimal etwas
lautlos ausgefallen.

| # | Anzeige | Was richtig ist |
|---|---|---|
| A.4.1 | **Wassermenge** (`river_water`) | Eine blaue Wasserkarte; die Linien werden **breiter, je mehr Wasser** fließt (logarithmisch) |
| A.4.2 | **Flussordnung** (`river_order`) | Die drei Rechenstufen unterscheidbar |
| A.4.3 | **Hinterlandhöhe** (`hinterland_height`) | Ein weiches Feld, landeinwärts steigend, im Meer leer |
| A.4.4 | **Voronoi** (`voronoi_map`) | Die neun Regionsgebiete als Flächen — **das ist der Höhenfaktor, den du sehen wolltest** |
| A.4.5 | Jede davon in **3D** | Bild wie in 2D. **Bleibt 3D leer, ist das der bekannte Registerfehler** und ein Befund |

## A.5 Die drei Stufenschalter

Terrain-Reiter, drei Häkchen: **Flussnetzwerk**, **Erosionsfilter**,
**Küstentypen**. Jedes einzeln aus, neu generieren.

* **Erwartet:** Flussnetzwerk aus → keine Täler. Erosionsfilter aus → keine
  Rinnen. Küstentypen aus → glatte, strukturlose Küstenlinie.
* **Achten auf:** dass beim Ausschalten wirklich schneller gerechnet wird.
  Bleibt die Zeit gleich, greift der Schalter nicht.

## A.6 Nevadin — die Spitzen

Das war der Ausgangsbefund: hunderte Nadelspitzen. Gemessen ist der Zähler von
122 auf 24 gefallen.

* **Ansehen:** Nevadin in 3D, schräg von der Seite.
* **Erwartet:** Gipfel und Grate, keine Nadeln.
* **Wenn es dir immer noch zu spitz ist:** sag es — der Rest liegt in der
  Auflösung der Weltkarte, nicht mehr am Rauschen.

## A.7 Offene Frage an dich

In der **Regionsvorschau** kommt nicht jeder Küstentyp einer Region vor — im
Hügelland fehlt zum Beispiel *Moher*, weil die Vorschau nur ein Ausschnitt
ist. Soll die Vorschau **alle drei Archetypen der Region erzwingen** (dann
siehst du alle, aber es ist nicht mehr das, was die Karte macht), oder so
bleiben wie sie ist?

---

# TEIL B — aus der Sitzung vom 2026-08-24, weiterhin unbestätigt

## 1. Wege in 3D — der Hauptpunkt dieser Sitzung

Karte generieren, **Settlement-Reiter**, Häkchen bei *Roads*, dann auf
**3D-View** umschalten.

| # | Was ansehen | Was richtig ist | Was schiefgehen kann |
|---|---|---|---|
| 1.1 | Kamera flach über den Boden legen und an einem Weg entlangschauen | Der Weg liegt **auf** dem Gelände | Bisher schwebte er sichtbar darüber — man konnte darunterschauen. Wenn er jetzt stattdessen flackert oder streckenweise verschwindet, ist der Tiefenversatz zu klein |
| 1.2 | Nah an einen Wegrand heranzoomen | Der Rand **läuft weich aus** statt an einer harten Kante zu enden | Sieht der Rand ausgefranst oder gepunktet aus, ist die Alpha-Schwelle falsch |
| 1.3 | Einen Weg von schräg oben ansehen | Die Fahrbahn wirkt **leicht gewölbt**, mit hellerer Mitte und dunkleren Schultern | Sieht sie aus wie eine Wurst, ist `NORMALEN_WOELBUNG` zu hoch; sieht sie flach aus, zu niedrig |
| 1.4 | Einen Weg an einem **Querhang** verfolgen | Er bleibt quer eben und versinkt nicht im Hang | |
| 1.5 | Einen Weg über eine **Kuppe** verfolgen | Er schneidet leicht ein, wie eine echte Trasse | Gemessen bis 0,7 m. Wenn das Gelände sichtbar durch den Weg stößt, ist es zu viel |
| 1.6 | Kamera drehen und schwenken | Das Band bleibt ruhig | Springen/Flackern beim Bewegen wäre derselbe Fehler wie beim gescheiterten Klippenband |
| 1.7 | Roads **mehrfach an- und ausschalten**, Konsole beobachten | Kein Ruckeln, keine wachsende Verzögerung | Die GL-Puffer werden jetzt wiederverwendet; wenn es langsamer statt schneller wird, greift die Freigabe nicht |

**Wenn in der Konsole `FEHLER: wegband-Shaderprogramm fehlt` steht:** die
Shader ließen sich nicht übersetzen. Bitte die Meldung schicken — das ist neu
und war vorher unsichtbar.

## 2. Umschalten der Overlays — soll spürbar schneller sein

Settlement-Reiter, **2D**. *Settlements*, *Landmarks*, *Roadsites*, *Roads*
mehrfach an- und ausschalten.

* **Erwartet:** deutlich flotter als vorher (gemessen 150 → 42 ms je
  Neuzeichnung, und es wird nur noch einmal statt zweimal gezeichnet).
* **Achten auf:** dass die Karte nach dem Umschalten **wirklich neu
  gezeichnet** ist. `draw_idle()` zeichnet verzögert — falls ein Overlay erst
  nach einer weiteren Mausbewegung erscheint, ist das der Punkt, an dem es
  hakt.

## 3. Wetter-Monatsanimation

Weather-Reiter, Temperature/Wind wählen, Monatsanimation läuft. **Auf einen
anderen Reiter wechseln, eine Minute arbeiten, zurückwechseln.**

* **Erwartet:** im Hintergrund wird nicht mehr gerechnet, und beim
  Zurückkommen läuft die Animation bei **demselben Monat** weiter, an dem du
  sie verlassen hast.
* **Achten auf:** dass sie überhaupt weiterläuft. Bleibt sie stehen, greift
  der Sichtbarkeits-Test zu hart.

## 4. Erosionsreiter

**Korrektur (Ticket #63):** Diese Prüfung ist veraltet. `EROSION_AKTIV`
steht schon länger auf `True` (`gui/config/value_default.py:1088`) — die
zentrale Richtigstellung dazu steht in `docs/TESTBERICHT.md` Abschnitt 3.

Erosion-Reiter öffnen.

* **Erwartet:** **kein** Hinweisstreifen und **freie Regler**, alle 13
  bedienbar. Die Erosionskette läuft (`EROSION_AKTIV = True`,
  `gui/config/value_default.py:1088`). Den gelben Streifen zeigt das Programm
  nur bei `False` (`_create_stilllegungs_hinweis()` in
  `gui/tabs/erosion_tab.py` prüft `EROSION_AKTIV` zur Laufzeit).
* Hier stand bis zum 16.09.2026 die umgekehrte Erwartung. Wer danach prüfte,
  hätte das richtige Verhalten als Fehler gemeldet. Zeigt der Reiter
  trotzdem den Streifen mit gesperrten Reglern, ist das jetzt umgekehrt ein
  echter Fehler.
* **Achten auf:** dass die Regler auch tatsächlich etwas bewirken. Drei
  Erosionstests sind rot und ihre Ursache ist ungeklärt (siehe
  `docs/TESTBERICHT.md`, Abschnitt 3).

## 5. Export

Karte vollständig generieren, dann exportieren.

* **Erwartet:** alle PNG sind **2048 × 2048**, unabhängig von der eingestellten
  Kartengröße.
* Im Ordner liegen zusätzlich **`noise_damping_mask.png`** (schwarz an Wegen,
  Bauflächen und Ufern, weiß in der Wildnis) und **`vektor.json`**.
* `manifest.json` nennt `meter_pro_pixel`. **Bei 21,3 km sollten dort 10,4
  stehen.**
* In `vektor.json` unter `"fehlt"` steht ausdrücklich, dass **Flüsse nicht
  dabei sind** — das ist bekannt und begründet, kein Fehler.

## 6. Mesh-Werkstatt (aus der vorigen Sitzung, weiterhin unbestätigt)

    .venv/Scripts/python.exe tools/mesh_werkstatt.py

Vier Netzarten durchschalten, Regler bewegen, „Fehler messen" anhaken.
Delatin braucht ~20 s und zeigt währenddessen den Fortschritt.

## 7. Immer noch der größte offene Blocker: 3b.1

**Die gesamte Vektor-Küste ist visuell unbestätigt** — weder die neue
Küstenform noch der Meeresboden noch die Regionsübergänge. Das hängt an
nichts anderem als daran, dass du es dir einmal ansiehst.

## 8. Biome-Reiter: Settlements/Flussnetz in 3D (Ticket #5, 2026-09-16)

`BiomeTab.apply_overlays()` stieg bis heute in der ersten Zeile aus, sobald
die Ansicht nicht 2D war — die am 2026-08-25 eingebauten 3D-Zweige für
Settlements und Flussnetz wurden dadurch nie erreicht. Headless geprüft ist
jetzt nur, dass der Code bei `current_view == "3d"` überhaupt bis zu diesen
Aufrufen durchläuft (`tests/smoke_test_biome_overlays_3d.py`) — **nicht**,
ob am Bildschirm wirklich etwas erscheint.

| # | Was ansehen | Was richtig ist | Was schiefgehen kann |
|---|---|---|---|
| 8.1 | Karte generieren, Biome-Reiter, Häkchen **Settlements** setzen, dann auf **3D** umschalten | Siedlungen/Landmarken/Roadsites erscheinen als Textur auf dem Gelände | Bleibt 3D leer, ist das derselbe Registerfehler wie schon dreimal zuvor |
| 8.2 | Dasselbe mit Häkchen **Flussnetz** | Das Flussnetz erscheint auf dem Gelände | |
| 8.3 | Häkchen in 3D wieder abwählen | Die jeweilige Textur verschwindet sofort | Bleibt die alte Textur liegen, greift die Sichtbarkeits-Abschaltung nicht |
| 8.4 | Häkchen in **2D** setzen, danach erst auf 3D umschalten | Übernahme sofort, ohne erneutes Generieren | |

---

## Was diese Sitzung NICHT geprüft hat

* Alles OpenGL — headless gibt es keinen GL-Kontext.
* Ob die Wegoptik dir **gefällt**. Die Messungen sagen, dass es technisch
  richtig liegt; ob es schön ist, sagen sie nicht.
